# -*- coding: utf-8 -*-
'''
@File    : train.py
@Time    : 2023/11/13 16:08:27
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
Train the model with Imagenet2012 dataset.
'''
from katacv.utils.related_pkgs.utility import *
from katacv.utils.related_pkgs.jax_flax_optax_orbax import *  # jax, jnp, flax, nn, train_state, optax

from katacv.utils.logs import Logs, MeanMetric

logs = Logs(
  init_logs={
    'loss_train': MeanMetric(),
    'accuracy_top1_train': MeanMetric(),
    'accuracy_top5_train': MeanMetric(),
    'loss_val': MeanMetric(),
    'accuracy_top1_val': MeanMetric(),
    'accuracy_top5_val': MeanMetric(),
    'epoch': 0,
    'SPS': MeanMetric(),
    'SPS_avg': MeanMetric(),
    'learning_rate': 0,
  },
  folder2name={
    'train/metrics': ['loss_train', 'accuracy_top1_train', 'accuracy_top5_train'],
    'val/metrics': ['loss_val', 'accuracy_top1_val', 'accuracy_top5_val'],
    'charts': ['SPS', 'SPS_avg', 'epoch', 'learning_rate']
  }
)

from katacv.utils.parser import Parser, Path, cvt2Path, datetime, CVArgs, SummaryWriter
class ImagenetArgs(CVArgs):
  learning_rate_fn: Callable
  momentum: float
  steps_per_epoch: int
  warmup_epochs: int

def get_args_and_writer(
    model_name,
    no_writer=False, input_args=None,
  ) -> Tuple[ImagenetArgs, SummaryWriter] | ImagenetArgs:
  parser = Parser(model_name, "Imagenet2012")
  # Imagenet dataset
  parser.add_argument("--path-dataset-tfrecord", type=cvt2Path, default=Path("/media/yy/Data/dataset/imagenet/tfrecord"),
    help="the path of the tfrecord dataset directory")
  parser.add_argument("--image-size", type=int, default=224,
    help="the input image size of the model")
  parser.add_argument("--image-center-crop-padding-size", type=int, default=32,
    help="the padding size of the center crop of the origin image")
  parser.add_argument("--batch-size", type=int, default=128,
    help="the size of each batch")
  parser.add_argument("--shuffle-size", type=int, default=128*16,
    help="the shuffle size of the dataset")
  # Hyper-parameters
  parser.add_argument("--weight-decay", type=float, default=1e-4,
    help="the coef of the weight penalty")
  parser.add_argument("--momentum", type=float, default=0.9,
    help="the momentum of the SGD optimizer")
  parser.add_argument("--total-epochs", type=int, default=50,
    help="the total epochs of the training")
  parser.add_argument("--learning-rate", type=float, default=0.05,
    help="the learning rate of the optimizer")
  parser.add_argument("--warmup-epochs", type=int, default=3,
    help="the number of warming up epochs")
  args = parser.get_args(input_args)

  assert(args.total_epochs > args.warmup_epochs)
  args.run_name = f"{args.model_name}__load_{args.load_id}__warmup_lr_{args.learning_rate}__batch_{args.batch_size}__{datetime.datetime.now().strftime(r'%Y%m%d_%H%M%S')}".replace("/", "-")
  args.input_shape = (args.batch_size, args.image_size, args.image_size, 3)
  from katacv.utils.imagenet.build_dataset import DATASET_SIZE
  args.steps_per_epoch = DATASET_SIZE['train'] // args.batch_size
  if no_writer: return args

  writer = parser.get_writer(args)
  return args, writer

class TrainState(train_state.TrainState):
  batch_stats: dict

def get_learning_rate_fn(args: ImagenetArgs):
  warmup_fn = optax.linear_schedule(
    init_value=0.0,
    end_value=args.learning_rate,
    transition_steps=args.warmup_epochs * args.steps_per_epoch
  )
  cosine_epoch = args.total_epochs - args.warmup_epochs
  cosine_fn = optax.cosine_decay_schedule(
    init_value=args.learning_rate,
    decay_steps=cosine_epoch * args.steps_per_epoch
  )
  schedule_fn = optax.join_schedules(
    schedules=[warmup_fn, cosine_fn],
    boundaries=[args.warmup_epochs * args.steps_per_epoch]
  )
  return schedule_fn

def get_model_state(model: nn.Module, args: ImagenetArgs, verbose=False):
  args.learning_rate_fn = get_learning_rate_fn(args)
  key = jax.random.PRNGKey(args.seed)
  if verbose:
    print(model.tabulate(key, jnp.empty(args.input_shape), train=False))
  variables = model.init(key, jnp.empty(args.input_shape), train=False)
  return TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=optax.sgd(learning_rate=args.learning_rate_fn, momentum=args.momentum, nesterov=True),
    batch_stats=variables['batch_stats'],
  )

@partial(jax.jit, static_argnames=['train', 'weight_decay'])
def model_step(state: TrainState, x, y, train: bool = True, weight_decay: bool = 1e-4):

    def loss_fn(params):
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            x, train=train,
            mutable=['batch_stats'],
        )
        loss = -(y * jax.nn.log_softmax(logits)).sum(-1).mean()
        weight_l2 = 0.5 * sum(
            jnp.sum(x**2) for x in jax.tree_util.tree_leaves(params) if x.ndim > 1  # no bias
        )
        loss += weight_decay * weight_l2
        return loss, (logits, updates)
    
    if train:
        (loss, (logits, updates)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=updates['batch_stats'])
    else:
        loss, (logits, _) = loss_fn(state.params)

    y_cat = jnp.argmax(y, -1)
    top1 = (jnp.argmax(logits, -1) == y_cat).mean()
    top5_pred = jnp.argsort(logits, axis=-1)[:, -5:]
    top5 = jnp.any(top5_pred == y_cat[:, jnp.newaxis], axis=-1).mean()

    return state, loss, top1, top5

def train(
    model: nn.Module,
    model_name: str,
    verbose=True):

  ### Initialize arguments and tensorboard writer ###
  args, writer = get_args_and_writer(model_name)

  ### Initialize model state ###
  state = get_model_state(model, args, verbose=verbose)

  ### Load weights ###
  from katacv.utils import load_weights
  state = load_weights(state, args)

  ### Save config ###
  from katacv.utils import SaveWeightsManager
  save_weight = SaveWeightsManager(args)

  ### Initialize dataset ###
  from katacv.utils.imagenet.build_dataset import ImagenetBuilder
  ds_builder = ImagenetBuilder(args)
  train_ds, train_ds_size = ds_builder.get_dataset(sub_dataset='train')
  val_ds, val_ds_size = ds_builder.get_dataset(sub_dataset='val')

  ### Train and evaluate ###
  start_time, global_step = time.time(), 0
  if args.train:
    for epoch in range(state.step//args.steps_per_epoch+1, args.total_epochs+1):
      print(f"epoch: {epoch}/{args.total_epochs}")
      print("training...")
      logs.reset()
      for x, y in tqdm(train_ds, total=train_ds_size):
        global_step += 1
        state, *metrics = model_step(state, x.numpy(), y.numpy(), train=True)
        logs.update(
          ['loss_train', 'accuracy_top1_train', 'accuracy_top5_train'],
          metrics
        )
        if global_step % args.write_tensorboard_freq == 0:
          logs.update(
            ['SPS', 'SPS_avg', 'epoch', 'learning_rate'],
            [
              args.write_tensorboard_freq/logs.get_time_length(),
              global_step/(time.time()-start_time),
              epoch,
              args.learning_rate_fn(state.step),
            ]
          )
          logs.writer_tensorboard(writer, global_step)
          logs.reset()
      print("validating...")
      logs.reset()
      for x, y in tqdm(val_ds, total=val_ds_size):
        _, *metrics = model_step(state, x.numpy(), y.numpy(), train=False)
        logs.update(
          ['loss_val', 'accuracy_top1_val', 'accuracy_top5_val', 'epoch', 'learning_rate'],
          [*metrics, epoch, args.learning_rate_fn(state.step)]
        )
      logs.writer_tensorboard(writer, global_step)
      
      ### Save weights ###
      if epoch % args.save_weights_freq == 0:
        save_weight(state)
  writer.close()

if __name__ == '__main__':
  # from katacv.yolov3.darknet53 import PreTrain, DarkNet
  # model = PreTrain(darknet=DarkNet([1,2,8,8,4]))
  # train(model, "DarkNet53")
  from katacv.yolov4.csp_darknet53 import PreTrain, CSPDarkNet
  model = PreTrain(darknet=CSPDarkNet([1,2,8,8,4]))
  train(model, "CSPDarkNet53")
