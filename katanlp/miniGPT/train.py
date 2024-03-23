import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))
from katanlp.miniGPT.dataset import TextDatasetBuilder
from katanlp.miniGPT.miniGPT import GPT, TrainConfig, GPTConfig
from katanlp.miniGPT.ckpt_manager import CheckpointManager
import argparse
from tensorboardX.writer import SummaryWriter
from tqdm import tqdm
from pathlib import Path
import time
from katacv.utils.parser import str2bool
from katacv.utils.logs import Logs, MeanMetric
path_root = Path(__file__).parents[2]

# Train cmd: python katanlp/miniGPT/train.py --path-dataset /home/yy/Coding/datasets/china_offical_documents --total-epoch 20 --n-embd 768 --n-head 12 --n-block 12 --train-datasize 262114 --val-datasize 16384

def parse_args(input_args=None, with_writer=True) -> tuple[argparse.Namespace, SummaryWriter]:
  parser = argparse.ArgumentParser()
  parser.add_argument("--name", type=str, default="MiniGPT")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--learning-rate", type=float, default=3e-4)
  parser.add_argument("--total-epochs", type=int, default=10)
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--n-embd", type=int, default=512)  # 768
  parser.add_argument("--n-head", type=int, default=8)  # 12
  parser.add_argument("--n-block", type=int, default=6)  # 12
  parser.add_argument("--n-token", type=int, default=128)
  parser.add_argument("--wandb", type=str2bool, default=False, const=True, nargs='?')
  parser.add_argument("--train-datasize", type=int, default=512*128)
  parser.add_argument("--val-datasize", type=int, default=32*128)
  parser.add_argument("--path-dataset", type=str, default=path_root.joinpath("katanlp/demo_data"))
  args = parser.parse_args(input_args)
  args.lr = args.learning_rate
  args.run_name = f"{args.name}__{args.seed}__{time.strftime(r'%Y%m%d_%H%M%S')}"
  path_logs = path_root / "logs" / args.run_name
  path_logs.mkdir(parents=True, exist_ok=True)
  args.path_logs = path_logs
  if not with_writer:
    return args

  if args.wandb:
    import wandb
    wandb.init(
      project="mini-NLP",
      sync_tensorboard=True,
      config=vars(args),
      name=args.run_name,
    )
  writer = SummaryWriter(str(path_logs / "tfboard"))
  return args, writer

logs = Logs(
  init_logs={
    'loss_train': MeanMetric(),
    'acc_train': MeanMetric(),
    'loss_val': MeanMetric(),
    'acc_val': MeanMetric(),
    'epoch': 0,
    'SPS': MeanMetric(),
    'learning_rate': 0,
  },
  folder2name={
    'train': ['loss_train', 'acc_train'],
    'val': ['loss_val', 'acc_val'],
    'charts': ['SPS', 'epoch', 'learning_rate'],
  }
)

def train():
  args, writer = parse_args()
  ### Dataset ###
  ds_builder = TextDatasetBuilder(path_dataset=args.path_dataset, val_ratio=0.2, seed=args.seed, n_divide=100)
  train_ds = ds_builder.get_dataset('train', batch_size=args.batch_size, n_token=args.n_token, datasize=args.train_datasize)
  val_ds = ds_builder.get_dataset('val', batch_size=args.batch_size, n_token=args.n_token, datasize=args.val_datasize)
  args.n_vocab = ds_builder.n_vocab
  ### Model ###
  train_cfg = TrainConfig(steps_per_epoch=len(train_ds), **vars(args))
  gpt_cfg = GPTConfig(**vars(args))
  gpt = GPT(cfg=gpt_cfg)
  gpt.create_fns()
  state = gpt.get_state(train_cfg)
  ### Checkpoint ###
  ckpt_manager = CheckpointManager(str(args.path_logs / f"ckpt"))

  ### Train and Validate ###
  for ep in range(args.total_epochs):
    print(f"Epoch: {ep+1}/{args.total_epochs}")
    print("Training...")
    logs.reset()
    bar = tqdm(train_ds)
    for x, y in bar:
      x, y = x.numpy(), y.numpy()
      state, (loss, acc) = gpt.model_step(state, x, y, train=True)
      logs.update(['loss_train', 'acc_train'], [loss, acc])
      bar.set_description(f"loss={loss:.4f}, acc={acc:.4f}")
      if state.step % 100 == 0:
        logs.update(
          ['SPS', 'epoch', 'learning_rate'],
          [100 / logs.get_time_length(), ep+1, train_cfg.lr_fn(state.step)]
        )
        logs.writer_tensorboard(writer, state.step)
        logs.reset()
    print("Validating...")
    bar = tqdm(val_ds)
    logs.reset()
    for x, y in bar:
      x, y = x.numpy(), y.numpy()
      _, (loss, acc) = gpt.model_step(state, x, y, train=False)
      logs.update(['loss_val', 'acc_val'], [loss, acc])
      bar.set_description(f"loss={loss:.4f}, acc={acc:.4f}")
    logs.update(
      ['epoch', 'learning_rate'],
      [ep+1, train_cfg.lr_fn(state.step)]
    )
    logs.writer_tensorboard(writer, state.step)

    ckpt_manager.save(ep+1, state, vars(args))
    # last_weights = args.path_logs / f"ckpt"
    # if last_weights.exists(): last_weights.unlink()
    # gpt.save_model(state, str(args.path_logs / f"weights_{ep+1}.ckpt"))
  ckpt_manager.close()

if __name__ == '__main__':
  train()
