import numpy as np
from typing import Any, List, Optional, Tuple

# ensure that object passed in is correct array format aka n dim array format
def ensure_array(x: Any):
    if isinstance(x, np.ndarray):
        return x
    return np.array(x, dtype=np.float32)

#TODO: unbroadcast function

# Context class which saves any data needed for backpropagation

class Context:
    # initialize saved tensors as empty tuple
    def __init__(self):
        self.saved_tensor: Tuple[np.ndarray, ...] = ()

    # create the saved tensor passing in the tuple of all the np.ndarrays
    def saved_for_backprop(self, *tensors: np.ndarray):
        self.saved_tensor = tuple(tensors)

