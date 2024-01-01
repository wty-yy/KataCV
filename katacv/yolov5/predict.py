from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.utils.yolo.predictor import BasePredictor
from katacv.yolov5.loss import cell2pixel
from katacv.yolov5.parser import YOLOv5Args

class Predictor(BasePredictor):
  
  def __init__(self, args: YOLOv5Args, state: train_state.TrainState, iout=None):
    super().__init__(state, iout)
    self.args = args

  @partial(jax.jit, static_argnums=0)
  def predict(self, state: train_state.TrainState, x: jnp.ndarray):
    logits, updates = state.apply_fn(
      {'params': state.params, 'batch_stats': state.batch_stats},
      x, train=False, mutable=['batch_stats']
      # x, train=True, mutable=['batch_stats']  # Update: Must use train BN, if no freeze backbone BN statistic
    )
    y, batch_size = [], x.shape[0]
    for i in range(3):
      xy = (jax.nn.sigmoid(logits[i][...,:2]) - 0.5) * 2.0 + 0.5
      xy = cell2pixel(xy, scale=2**(i+3))
      wh = (jax.nn.sigmoid(logits[i][...,2:4]) * 2) ** 2 * self.args.anchors[i].reshape(1,3,1,1,2)
      conf = jax.nn.sigmoid(logits[i][...,4:5])
      cls = jax.nn.sigmoid(logits[i][...,5:])
      conf = conf * jnp.max(cls, axis=-1, keepdims=True)
      cls = jnp.argmax(cls, axis=-1, keepdims=True)
      y.append(jnp.concatenate([xy,wh,conf,cls], -1).reshape(batch_size,-1,6))
    y = jnp.concatenate(y, 1)  # shape=(batch_size,all_pbox_num,6)
    return y
  