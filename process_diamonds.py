# reshape diamonds

import os
import numpy as np
import pandas as pd

current_dir = os.getcwd()
dataset_path = os.path.join(os.getcwd(), os.pardir, 'data', 'diamond_prices.csv')
diamonds = read_csv(dataset_path)

# cut, color, clarity
diamonds_dummy =pd.get_dummies(diamonds)

