# -*- coding: utf-8 -*-
'''
@File    : ocr_ctc.py
@Time    : 2023/10/14 20:03:24
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
'''
import os, sys
sys.path.append(os.getcwd())

from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.utils.related_pkgs.utility import *

from katacv.ocr.crnn_model_lstm import TrainState

@partial(jax.jit, static_argnums=2)
def predict(state: TrainState, x, blank_id=0) -> Tuple[jax.Array, jax.Array]:
    logits = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        x, train=False
    )
    pred_probs = jax.nn.softmax(logits)
    pred_idxs = jnp.argmax(pred_probs, -1)  # (B,T)
    mask = (
        (pred_idxs != jnp.pad(pred_idxs[:,:-1], ((0,0),(1,0)))) &
        (pred_idxs != blank_id)
    )
    conf = jnp.prod(pred_probs.max(-1), -1)  # (B,) confidence
    return pred_idxs, mask, conf

import numpy as np
def apply_mask(pred_idxs, mask, max_len=23) -> jax.Array:
    y_pred = np.zeros((pred_idxs.shape[0], max_len), dtype=np.int32)
    for i in range(pred_idxs.shape[0]):
        seq = pred_idxs[i][mask[i]]
        N = min(max_len, seq.size)
        y_pred[i,:N] = seq[:N]
    return y_pred

def predict_result(
        state: TrainState,
        x: jax.Array,
        max_len: int,
        idx2ch: dict
    ) -> Sequence[str]:
    pred_idx, mask, conf = predict(state, x)
    y_pred = apply_mask(pred_idx, mask, max_len)
    pred_seq = []
    for i in range(y_pred.shape[0]):
        seq = []
        for j in range(y_pred.shape[1]):
            if y_pred[i,j] == 0: break
            seq.append(chr(idx2ch[y_pred[i,j]]))
        pred_seq.append("".join(seq))
    return pred_seq, conf

if __name__ == '__main__':
    from katacv.ocr.parser import get_args_and_writer
    from katacv.ocr.cnn_model import get_ocr_cnn_state
    args = get_args_and_writer(no_writer=True)
    state = get_ocr_cnn_state(args)
    from katacv.utils import load_weights
    state = load_weights(state, args)
    
    from katacv.utils.ocr.build_dataset import DatasetBuilder
    args.batch_size = 2
    ds_builder = DatasetBuilder(args)
    ds, ds_size = ds_builder.get_dataset('8examples')
    for x, y in ds:
        x = x.numpy(); y = y.numpy()
        pred_idxs, mask = predict(state, x)
        y_pred = apply_mask(jax.device_get(pred_idxs), jax.device_get(mask))
        print(y_pred, y)
        print("acc:", np.mean((y_pred == y).all(-1)))
