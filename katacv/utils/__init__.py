from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.utils.related_pkgs.utility import *

from katacv.utils.parser import CVArgs
def load_weights(state: train_state.TrainState, args: CVArgs):
    if args.load_id == 0: return state
    path_load = args.path_cp.joinpath(f"{args.model_name}-{args.load_id:04}")
    assert(path_load.exists())
    with open(path_load, 'rb') as file:
        state = flax.serialization.from_bytes(state, file.read())
    print(f"Successfully load weights from '{str(path_load)}'")
    return state

class SaveWeightsManager:
    path_save: Path

    def __init__(self, args: CVArgs):
        self.path_cp, self.model_name = args.path_cp, args.model_name
        self.save_id = args.load_id + 1
        self.update_path_save()
        if self.path_save.exists():
            print(f"The weights file '{str(self.path_save)}' already exists, still want to continue? [enter]", end=""); input()
    
    def update_path_save(self):
        self.path_save = self.path_cp.joinpath(f"{self.model_name}-{self.save_id:04}")
    
    def __call__(self, state: train_state.TrainState):
        self.update_path_save()
        with open(self.path_save, 'wb') as file:
            file.write(flax.serialization.to_bytes(state))
        print(f"Save weights at '{str(self.path_save)}'")
        self.save_id += 1
