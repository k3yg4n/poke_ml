# A convolutional neural network which can predict if a Pokemon is a fire-type 
# or a water type based on its image.

# Change to only predict one type

import logging
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub

from keras.preprocessing import image
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from keras import layers

from utils import *
import glob
from pathlib import Path

# Filter warnings and select logging level
warnings.filterwarnings('ignore')
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Create dataframe from .csv and perform data cleaning
poke_df = pd.read_csv("pokemon.csv")

poke_df = poke_df.rename(columns={"#": "Num", "Type 1": "Type1", "Type 2": "Type2"})
poke_df = poke_df.drop_duplicates(subset=['Num'])
poke_df = poke_df.fillna(value={"Type2": ""})
poke_df['Types'] = poke_df['Type1'] + "," + poke_df['Type2']
poke_df['Types'] = poke_df['Types'].apply(
  lambda types_entry: [l for l in str(types_entry).split(',') if l not in [""]]
  ) # Remove comma if types entry only has one type


# A list of bools to determine if a Pokemon entry has a second type
conditions = [poke_df['Type2'] == '', poke_df['Type2'] != '']

# A list of types
# First element has one type
# Second element contains both types (if applicable) separated by '|'
values = [poke_df['Type1'], poke_df['Type1'] + '|' + poke_df['Type2']]

def get_images(gen_folder_path: Path):
  pre_gen5_poke_df = poke_df[poke_df['Num'] < 650] # Filter Pokemon from gen5 and earlier
  pre_gen5_poke_df = pre_gen5_poke_df.set_index('Num')
  
  png_files = glob.glob(str(gen_folder_path) + "/**/*.png", recursive=True) # Generate list of png file path names
  
  # Remove invalid pokemon entries
  png_filtered = [file for file in png_files if Path(file).stem.isdigit()]
  png_filtered = [file for file in png_filtered if int(Path(file).stem) in list(pre_gen5_poke_df['Num'])]
  
  # Create new dataframe from paths to Pokemon images
  gen_poke_df = pd.DataFrame(png_filtered)
  # Add types column to new df using pre_gen5_poke_df
  gen_poke_df['Types'] = [pre_gen5_poke_df.loc[int(Path(file).stem), 'Types'] for file in png_filtered]
  # 102 doesn't exist in pre_gen5_pokedf eggsecute but exists in png_files?!
