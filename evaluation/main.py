from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd

from data_loader import data_loader
from gain import gain
from utils import rmse_loss


def GAIN (data): 
  # data is in DataFrame
  gain_parameters = {'batch_size': 128,
                     'hint_rate': 0.9,
                     'alpha': 100,
                     'iterations': 10000}
  df = data.copy()
  miss_data_x = df.values
  data_m = 1.0 - pd.isna(miss_data_x)
  imputed_data_x = gain(miss_data_x, gain_parameters)
  df = pd.DataFrame(imputed_data_x)
  return df
