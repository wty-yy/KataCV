#jax_ctcloss/ctclossv1.py
import jax.numpy as np
import jax

ninf =-1e5#-np.inf

def insert_blank(labels, blank=0):
    new_labels=[blank]
    for l in labels:
        new_labels += [l, blank]
    new_labels=np.array(new_labels)
    return new_labels
def compute_loss(log_alpha,t,i):
    return np.logaddexp(log_alpha[t-1,i-1],log_alpha[t-1,i-2])

@jax.jit
def ctcloss(logits, labels,input_len,label_len):
    log_y=jax.nn.log_softmax(logits)    
    labels=np.array([insert_blank(i) for i in labels ])

    B,T, K = log_y.shape
    B,L=labels.shape

    one_hot=jax.nn.one_hot(labels,K)
    logprobs=np.einsum("blk,btk->blt",one_hot,log_y)
    logprobs=logprobs.transpose(2,0,1)

    pre_log_alpha=np.ones((B,L))*ninf
    pre_log_alpha=pre_log_alpha.at[:,0].set(0.0)
    
    mask=np.array(labels[:,:-2]==labels[:,2:],np.int32)
    mask=1-mask
    mask=np.pad(mask,((0,0),(2,0)))
    mask=np.where(mask>0,0,ninf)
    
    def loop_for_t(pre_log_alpha,t):
        a = pre_log_alpha 
        b = pre_log_alpha[:,:-1] 
        b=np.pad(b,((0,0),(1,0)),mode="constant",constant_values=ninf)
        c=pre_log_alpha[:,:-2]  
        c=np.pad(c,((0,0),(2,0)),mode="constant",constant_values=ninf)
        
        d= np.logaddexp(a,b)   
        e= np.logaddexp(d,c+mask)
        next_log_alpha=e+t
        return next_log_alpha,next_log_alpha

    _,next_log_alpha_t=jax.lax.scan(loop_for_t,pre_log_alpha,logprobs)  
    next_log_alpha_t=next_log_alpha_t.transpose((1,0,2)) #(B,T,L)
    label_len=label_len*2+1
    
    loss=jax.vmap(compute_loss,in_axes=0,out_axes=0)(next_log_alpha_t,input_len,label_len)       
    return -loss  