import jax, jax.numpy as jnp
import optax

# concatenate is slow, real slow, faster twice!

@jax.jit
def ctc_loss(logits, labels, blank_id=0, log_eps=-1e5):
    logprobs = jax.nn.log_softmax(logits)
    B, T, C = logits.shape
    B, N = labels.shape
    lens = jnp.max(jnp.where(labels!=0, jnp.arange(N)+1, 0), axis=-1)  # (B,)
    one_hot = jax.nn.one_hot(labels, C)  # (B,N,C)
    logprobs_char = jnp.einsum('btc,bnc->tbn', logprobs, one_hot)  # (T,B,N)
    logprobs_blank = jnp.transpose(logprobs[..., blank_id:blank_id+1], (1,0,2))  # (T,B,1)
    pre_log_g = jnp.ones((B,N)) * log_eps
    pre_log_h = jnp.ones((B,N+1)) * log_eps
    pre_log_h = pre_log_h.at[:,0].set(0.0)
    repeat = jnp.pad(labels[:,:-1] == labels[:,1:], ((0,0),(0,1))).astype(jnp.float32)

    def update_h(h, delta):
        return jnp.logaddexp(
            h,
            jnp.pad(delta, ((0,0),(1,0)), constant_values=log_eps)
        )
    
    def loop_func(pre, x):
        pre_log_g, pre_log_h = pre
        logprob_char, logprob_blank = x
        tmp = update_h(pre_log_h, pre_log_g + repeat * log_eps)
        log_g = jnp.logaddexp(pre_log_g, tmp[:,:-1]) + logprob_char
        log_h = update_h(
            tmp,
            pre_log_g + (1.0 - repeat) * log_eps
        ) + logprob_blank
        ret = (log_g, log_h)
        return ret, ret
    
    init = (pre_log_g, pre_log_h)
    xs = (logprobs_char, logprobs_blank)
    _, (log_g, log_h) = jax.lax.scan(loop_func, init, xs)  # (T,B,N)
    ans = update_h(log_h[-1], log_g[-1])
    ans_mask = jax.nn.one_hot(lens, N+1)  # (B,N+1)
    per_loss = -jnp.einsum('bn,bn->b', ans, ans_mask)
    return per_loss

if __name__ == '__main__':
    import numpy as np
    logits = np.ones((4, 16, 12))
    labels = np.array([
        [1,2,2,4,5,0,0,0],
        [6,2,1,1,7,7,5,0],
        [1,2,2,0,0,0,0,0],
        [3,2,1,2,0,0,0,0],
    ])
    logits_padding = np.zeros(logits.shape[:2])
    labels_padding = np.array([
        [0,0,0,0,0,1,1,1],
        [0,0,0,0,0,0,0,1],
        [0,0,0,1,1,1,1,1],
        [0,0,0,0,1,1,1,1],
    ])
    print(jax.jit(optax.ctc_loss)(logits, logits_padding, labels, labels_padding))
    print(ctc_loss(logits, labels))
    from ctc_loss_zhihu_v1 import ctcloss as ctcloss_v1
    from ctc_loss_zhihu_v2 import ctcloss as ctcloss_v2
    input_len = np.array([16,16,16,16])
    label_len = np.array([5,7,3,4])
    print(ctcloss_v1(logits, labels, input_len, label_len))
    print(ctcloss_v2(logits, labels, input_len, label_len))
    import torch
    print(torch.ctc_loss(torch.tensor(np.transpose(logits, (1,0,2))).log_softmax(-1), torch.tensor(labels), torch.tensor(input_len), torch.tensor(label_len)))

    np.random.seed(42)
    # B, T, C, N = 64, 24, 27, 10
    B, T, C, N = 128, 50, 80, 40
    logits = np.random.randn(B, T, C)
    labels = np.random.randint(1, C, (B, N))
    logits_padding = np.zeros(logits.shape[:2])
    labels_padding = np.zeros(labels.shape)
    optax_ctc_loss = jax.jit(optax.ctc_loss)
    input_len = np.array([T for _ in range(B)])
    label_len = np.array([N for _ in range(B)])

    # JAX compile
    optax_ctc_loss(logits, logits_padding, labels, labels_padding)
    ctc_loss(logits, labels)
    ctcloss_v1(logits, labels, input_len, label_len)
    ctcloss_v2(logits, labels, input_len, label_len)

    import time
    from tqdm import tqdm
    def test(func, args, name="", times=1000):
        start_time = time.time()
        for _ in tqdm(range(times)):
            func(*args)
        print(name, "use:", time.time() - start_time, "s")
    test(optax_ctc_loss, (logits, logits_padding, labels, labels_padding), 'optax_ctc_loss')
    test(ctc_loss, (logits, labels), 'ctc_loss')
    test(ctcloss_v1, (logits, labels, input_len, label_len), 'ctcloss_v1')
    test(ctcloss_v2, (logits, labels, input_len, label_len), 'ctcloss_v2')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = torch.tensor(np.transpose(logits, (1,0,2)), requires_grad=True).to(device).log_softmax(2)
    labels = torch.tensor(labels).to(device)
    input_len = torch.tensor(input_len).to(device)
    label_len = torch.tensor(label_len).to(device)
    test(torch.ctc_loss, (inputs, labels, input_len, label_len), 'torch.ctc_loss')
