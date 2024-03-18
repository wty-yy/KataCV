from katanlp.miniGPT.miniGPT import GPT, GPTConfig, TrainConfig
from katanlp.miniGPT.train import parse_args
from katanlp.miniGPT.dataset import TextDatasetBuilder
from katanlp.miniGPT.ckpt_manager import CheckpointManager
import numpy as np
import jax

# Under log path
# weights_path = "MiniGPT__0__20240316_222541/weights_10.ckpt"
weights_path = "/home/yy/Coding/GitHub/KataCV/logs/MiniGPT__0__20240317_191903/ckpt"
load_step = 10

class Predictor:
  def __init__(self):
    # self.args = args = parse_args("", with_writer=False)
    ckpt_mngr = CheckpointManager(weights_path)
    load_info = ckpt_mngr.restore(load_step)
    params, args = load_info['params'], load_info['config']
    self.n_token = args['n_token']
    self.rng = jax.random.PRNGKey(args['seed'])
    self.ds_builder = TextDatasetBuilder()
    train_cfg = TrainConfig(steps_per_epoch=1e9, **args)
    self.gpt = GPT(cfg=GPTConfig(**args))
    self.gpt.create_fns()
    state = self.gpt.get_state(train_cfg, train=False)
    self.state = state.replace(params=params, tx=None, opt_state=None)
    self.encode = self.ds_builder.encode
    self.decode = self.ds_builder.decode
    self.predict(np.zeros(1, np.int32))
  
  def predict(self, x):  # previous text -> predict char
    if len(x) > self.n_token:
      x = x[-self.n_token:]
    n = len(x)
    x = np.pad(x, [0, self.n_token - len(x)]).reshape(1, -1)
    rng, self.rng = jax.random.split(self.rng)
    mask_len = np.array([n], np.int32)
    pred = jax.device_get(self.gpt.predict(self.state, x, rng, mask_len)[:1])
    return self.decode(pred)[0]
  
  def loop(self, max_times=float('inf'), generate_length=256):
    while max_times > 0:
      max_times -= 1
      s = input(">> ")
      if len(s) == 0: continue
      x = self.encode(s)
      print(s, end='', flush=True)
      for _ in range(generate_length):
        ch = self.predict(x)
        print(ch, end='', flush=True)
        x = np.concatenate([x, self.encode([ch])])
      print()

if __name__ == '__main__':
  predictor = Predictor()
  predictor.loop()