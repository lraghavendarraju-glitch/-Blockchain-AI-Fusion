import pytest
import numpy as np
import torch
from blockai.generator import make_dataset

@pytest.fixture(scope="module")
def small_dataset():
    torch.set_num_threads(2)
    return make_dataset(84,seed=42,min_steps=5,max_steps=18)
