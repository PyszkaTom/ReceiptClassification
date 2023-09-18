import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn
import sklearn
import numpy as np

model_2 = nn.Sequential(
    nn.Linear(in_features=1536, out_features=1536),
    nn.ReLU(),
    nn.Linear(in_features=1536, out_features=1536),
    nn.ReLU(),
    nn.Linear(in_features=1536, out_features=14),
).to(device)

