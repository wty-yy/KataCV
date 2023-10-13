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

    def pad_one_before(a, constant_values=0):
        return jnp.pad(a, ((0,0),(1,0)), constant_values=constant_values)

    repeat = pad_one_before(labels[:,:-1] == labels[:,1:])
    repeat_mask = repeat * log_eps  # (B,N)

    def loop_func(pre, x):
        pre_log_g, pre_log_h = pre
        logprob_char, logprob_blank = x
        log_g = jnp.logaddexp(pre_log_g + logprob_char, pre_log_h[:,:-1] + logprob_char)
        log_g = jnp.logaddexp(
            log_g,
            pad_one_before(pre_log_g[:,:-1], log_eps) + repeat_mask + logprob_char
        )
        log_h = jnp.logaddexp(
            pre_log_h + logprob_blank,
            pad_one_before(pre_log_g, log_eps) + logprob_blank
        )
        ret = (log_g, log_h)
        return ret, ret
    
    init = (pre_log_g, pre_log_h)
    xs = (logprobs_char, logprobs_blank)
    _, (log_g, log_h) = jax.lax.scan(loop_func, init, xs)  # (T,B,N)
    ans = jnp.logaddexp(
        log_h[-1],
        pad_one_before(log_g[-1], log_eps)
    )
    ans_mask = jax.nn.one_hot(lens, N+1)  # (B,N+1)
    per_loss = -jnp.einsum('bn,bn->b', ans, ans_mask)
    return per_loss

if __name__ == '__main__':
    logits = jnp.ones((4, 16, 12))
    labels = jnp.array([
        [1,2,3,4,5,0,0,0],
        [6,2,1,5,7,8,5,0],
        [1,2,1,0,0,0,0,0],
        [3,2,1,2,0,0,0,0],
    ])
    logits_padding = jnp.zeros(logits.shape[:2])
    labels_padding = jnp.array([
        [0,0,0,0,0,1,1,1],
        [0,0,0,0,0,0,0,1],
        [0,0,0,1,1,1,1,1],
        [0,0,0,0,1,1,1,1],
    ])
    print(optax.ctc_loss(logits, logits_padding, labels, labels_padding))
    print(ctc_loss(logits, labels))
    from ctc_loss_zhihu_v1 import ctcloss_v1
    from ctc_loss_zhihu_v2 import ctcloss_v2
    input_len = jnp.array([16,16,16,16])
    label_len = jnp.array([5,7,3,4])
    print(ctcloss_v1(logits, labels, input_len, label_len))
    print(ctcloss_v2(logits, labels, input_len, label_len))

    import numpy as np
    np.random.seed(42)
    B, T, C, N = 64, 24, 27, 10
    logits = np.random.randn(B, T, C)
    labels = np.random.randint(1, C, (B, N))
    logits_padding = np.zeros(logits.shape[:2])
    labels_padding = np.zeros(labels.shape)
    optax_ctc_loss = jax.jit(optax.ctc_loss)
    input_len = jnp.array([T for _ in range(B)])
    label_len = jnp.array([N for _ in range(B)])

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
    test(optax_ctc_loss, (logits, logits_padding, labels, labels_padding))
    test(ctc_loss, (logits, labels))
    test(ctcloss_v1, (logits, labels, input_len, label_len))
    test(ctcloss_v2, (logits, labels, input_len, label_len))

