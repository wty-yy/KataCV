# -*- coding: utf-8 -*-
'''
@File    : ocr_ctc.py
@Time    : 2023/10/14 20:03:24
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
'''

from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.utils.related_pkgs.utility import *

from katacv.ocr.cnn_model import TrainState
from ctc_loss.ctc_loss import ctc_loss
def model_step(
    state: TrainState,
    x: jax.Array,
    y: jax.Array,
    train: bool,
):
    def loss_fn(params):
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            x, train=train,
            mutable=['batch_stats']
        )
        weight_l2 = 0.5 * sum(
            jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(params) if x.ndim > 1
        )
        regular = args.weight_decay * weight_l2
        loss = ctc_loss(logits, y).mean() + regular
        return loss, updates
    
    if train:
        (loss, updates), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=updates['batch_stats'])
    else:
        loss, _ = loss_fn(state.params)
    return state, (loss,)

if __name__ == '__main__':
    ### Initialize arguments and tensorboard writer ###
    from katacv.ocr.parser import get_args_and_writer
    args, writer = get_args_and_writer()

    ### Initialize log manager ###
    from katacv.ocr.logs import logs

    ### Initialize model state ###
    from katacv.ocr.cnn_model import get_ocr_cnn_state
    state = get_ocr_cnn_state(args)

    ### Load weights ###
    from katacv.utils import load_weights
    state = load_weights(state, args)

    ### Save config ###
    from katacv.utils import SaveWeightsManager
    save_weight = SaveWeightsManager(args)

    ### Initialize dataset ###
    from katacv.utils.ocr.build_dataset import DatasetBuilder
    ds_builder = DatasetBuilder(args)
    train_ds, train_ds_size = ds_builder.get_dataset('train')
    val_ds, val_ds_size = ds_builder.get_dataset('val')

    ### Train and evaluate ###
    start_time, global_step = time.time(), 0
    if args.train:
        for epoch in range(state.step//train_ds_size+1, args.total_epochs+1):
            print(f"epoch: {epoch}/{args.total_epochs}")
            print("training...")
            logs.reset()
            for x, y in tqdm(train_ds, total=train_ds_size):
                x, y = x.numpy(), y.numpy()
                global_step += 1
                state, metrics = model_step(state, x, y, train=True)
                logs.update(
                    ['loss_train'],
                    metrics
                )
                if global_step % args.write_tensorboard_freq == 0:
                    logs.update(
                        ['SPS', 'SPS_avg', 'epoch', 'learning_rate'],
                        [
                            args.write_tensorboard_freq/logs.get_time_length(),
                            global_step/(time.time()-start_time),
                            epoch,
                            args.learning_rate_fn(state.step)
                        ]
                    )
                    logs.start_time = time.time()
                    logs.writer_tensorboard(writer, global_step)
            print("validating...")
            logs.reset()
            for x, y in tqdm(val_ds, total=val_ds_size):
                x, y = x.numpy(), y.numpy()
                global_step += 1
                _, metrics = model_step(state, x, y, train=False)
                logs.update(
                    ['loss_val'],
                    metrics
                )
                if global_step % args.write_tensorboard_freq == 0:
                    logs.update(
                        ['SPS', 'SPS_avg', 'epoch', 'learning_rate'],
                        [
                            args.write_tensorboard_freq/logs.get_time_length(),
                            global_step/(time.time()-start_time),
                            epoch,
                            args.learning_rate_fn(state.step)
                        ]
                    )
                    logs.start_time = time.time()
                    logs.writer_tensorboard(writer, global_step)
            
            ### Save weights ###
            if epoch % args.save_weights_freq == 0:
                save_weight(state)
    writer.close()
