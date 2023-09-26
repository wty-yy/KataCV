from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.yolov3.yolov3 import TrainState
from katacv.utils.detection import slice_by_idxs, cvt_coord_cell2image

@partial(jax.jit, static_argnames=['B'])
def predict(state: TrainState, x, anchors, B=3):
    logits = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        x, train=False
    )
    @partial(jax.jit, static_argnames=['S'])
    def convert_coord2image(logits, anchors, S):
        N = logits.shape[0]
        ret = []
        for k in range(B):
            now = logits[:,:,:,k,:]  # (N,B,B,5+C)
            xy = cvt_coord_cell2image(jax.nn.sigmoid(now[...,1:3]))  # (N,B,B,2)
            w = jnp.exp(now[...,3:4]) * anchors[k][0]  # (N,B,B,1)
            h = jnp.exp(now[...,4:5]) * anchors[k][1]  # (N,B,B,1)
            cls = jnp.argmax(jax.nn.softmax(now[...,5:]), -1, keepdims=True)  # (N,B,B,1)
            c = jax.nn.sigmoid(now[...,0:1]) * jnp.max(jax.nn.softmax(now[...,5:]), -1, keepdims=True)  # (N,B,B,1)
            # c = jax.nn.sigmoid(now[...,0:1])  # (N,B,B,1)
            # print(c.shape, xy.shape, w.shape, h.shape, cls.shape)
            ret.append(jnp.concatenate([c,xy,w,h,cls], -1).reshape(N, S*S, 6))
        return jnp.array(ret).reshape(N, S*S, B, 6)
        return ret

    def get_best_box_in_cell(y):  # (N,S*S,B,6)
        idxs = jnp.argmax(y[...,0], -1)
        best_boxes = slice_by_idxs(y.reshape(*y.shape[:-2], -1), idxs * 6, 6)
        return best_boxes  # (N,S*S,6)

    best_boxes = []
    for i in range(len(logits)):
        pred = convert_coord2image(logits[i], anchors[i*B:(i+1)*B], logits[i].shape[1])
        best_boxes.append(get_best_box_in_cell(pred))
    best_boxes = jnp.concatenate(best_boxes, 1)
    return best_boxes