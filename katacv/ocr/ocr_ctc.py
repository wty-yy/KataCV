# -*- coding: utf-8 -*-
'''
@File    : ocr_ctc.py
@Time    : 2023/10/14 20:03:24
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
In `parser.py`:
`model_name`: `OCR-CNN`, `OCR-CRNN-LSTM`, `OCR-CRNN-BiLSTM`
``
'''
import os, sys
sys.path.append(os.getcwd())

from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.utils.related_pkgs.utility import *

from katacv.ocr.cnn_model import TrainState
from ctc_loss.ctc_loss import ctc_loss
@partial(jax.jit, static_argnames=['train', 'blank_id'])
def model_step(
    state: TrainState,
    x: jax.Array,
    y: jax.Array,
    train: bool,
    blank_id: int = 0
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
        return loss, (updates, logits)
    
    if train:
        (loss, (updates, logits)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=updates['batch_stats'])
    else:
        loss, (_, logits) = loss_fn(state.params)

    pred_probs = jax.nn.softmax(logits)
    pred_idxs = jnp.argmax(pred_probs, -1)  # (B,T)
    mask = (
        (pred_idxs != jnp.pad(pred_idxs[:,:-1], ((0,0),(1,0)))) &
        (pred_idxs != blank_id)
    )
    acc_params = (pred_idxs, mask)

    return state, acc_params, loss

import numpy as np
def calc_accuracy(y, acc_params, max_len):
    pred_idxs, mask = acc_params
    y_pred = np.zeros((pred_idxs.shape[0], max_len), dtype=np.int32)
    for i in range(pred_idxs.shape[0]):
        seq = pred_idxs[i][mask[i]]
        N = min(max_len, seq.size)
        y_pred[i,:N] = seq[:N]
    acc = np.mean((y_pred == y).all(-1))
    return acc

if __name__ == '__main__':
    ### Initialize arguments and tensorboard writer ###
    from katacv.ocr.parser import get_args_and_writer
    args, writer = get_args_and_writer()

    ### Initialize log manager ###
    from katacv.ocr.logs import logs

    ### Initialize model state ###
    from katacv.ocr.cnn_model import get_ocr_cnn_state
    from katacv.ocr.crnn_model_lstm import get_ocr_crnn_lstm_state
    from katacv.ocr.crnn_model_bilstm import get_ocr_crnn_bilstm_state
    if 'OCR-CNN' in args.model_name:
        state = get_ocr_cnn_state(args)
    elif 'OCR-CRNN-LSTM' in args.model_name:
        state = get_ocr_crnn_lstm_state(args)
    elif 'OCR-CRNN-BiLSTM' in args.model_name:
        state = get_ocr_crnn_bilstm_state(args)

    ### Load weights ###
    from katacv.utils.model_weights import load_weights
    state = load_weights(state, args)

    ### Save config ###
    from katacv.utils.model_weights import SaveWeightsManager
    save_weight = SaveWeightsManager(args)

    ### Initialize dataset ###
    from katacv.utils.ocr.build_dataset import DatasetBuilder
    ds_builder = DatasetBuilder(args)
    train_ds, train_ds_size = ds_builder.get_dataset('train', use_lower=args.use_lower)
    val_ds, val_ds_size = ds_builder.get_dataset('val', use_lower=args.use_lower)

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
                state, acc_params, loss = model_step(state, x, y, train=True)
                acc = calc_accuracy(y, jax.device_get(acc_params), args.max_label_length)
                logs.update(
                    ['loss_train', 'acc_train'],
                    [loss, acc]
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
                x, y = x.numpy(), y.numpy()
                _, acc_params, loss = model_step(state, x, y, train=False)
                acc = calc_accuracy(y, jax.device_get(acc_params), args.max_label_length)
                logs.update(
                    ['loss_val', 'acc_val', 'epoch', 'learning_rate'],
                    [loss, acc, epoch, args.learning_rate_fn(state.step)]
                )
            logs.writer_tensorboard(writer, global_step)
            
            ### Save weights ###
            if epoch % args.save_weights_freq == 0:
                save_weight(state)
    writer.close()
