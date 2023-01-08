# A convolutional neural network which can predict if a Pokemon's types
# Using Pokemon sprites from multiple generations and games, in addition to back 
# sprites, shiny variants, and female Pokemon as a built-in way of augmenting the 
# size of our training data. Through this, our training dataset consists of >15,000 images.

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

# File path constants
GEN1_PATH = Path("./sprites/sprites/pokemon/versions/generation-i")
GEN2_PATH = Path("./sprites/sprites/pokemon/versions/generation-ii")
GEN3_PATH = Path("./sprites/sprites/pokemon/versions/generation-iii")
GEN4_PATH = Path("./sprites/sprites/pokemon/versions/generation-iv")
GEN5_PATH = Path("./sprites/sprites/pokemon/versions/generation-v")

# Pandas output options
pd.set_option('display.max_colwidth',1000)

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

# Generates training and testing datasets for Pokemon images (X) and types (y)
def get_test_and_train_datasets(gen_folder_path: Path):
  pre_gen5_poke_df = poke_df[poke_df['Num'] < 650] # Filter Pokemon from gen5 and earlier
  
  png_files = glob.glob(str(gen_folder_path) + "/**/*.png", recursive=True) # Generate list of png file path names
  
  # Remove invalid pokemon entries
  png_filtered = [file for file in png_files if Path(file).stem.isdigit()]
  png_filtered = [file for file in png_filtered if int(Path(file).stem) in list(pre_gen5_poke_df['Num'])]
  
  # Change the index of the data frame to none
  pre_gen5_poke_df = pre_gen5_poke_df.set_index('Num')

  # Create new dataframe from paths to Pokemon images
  gen_poke_df = pd.DataFrame(png_filtered)

  # Add types column to new df using pre_gen5_poke_df
  gen_poke_df['Types'] = [pre_gen5_poke_df.loc[int(Path(file).stem), 'Types'] for file in png_filtered]

  gen_poke_df = gen_poke_df.rename(columns={0: "sprites"})

  gen_string = str(gen_folder_path).rsplit('\\', 1)[1]
  file_name = f"{gen_string}-poke-df.csv"
  df_output_filepath = Path(f"./output/{file_name}")
  gen_poke_df.to_csv(df_output_filepath)

  # Split the sprites and types dataset into training data and test(validation) data
  X_train, X_val, y_train, y_val = train_test_split(gen_poke_df['sprites'], gen_poke_df['Types'], test_size=0.2, random_state=44)

  # Add an index to newly created datasets
  X_train.index = list(range(len(X_train)))
  y_train.index = list(range(len(y_train)))
  X_val.index = list(range(len(X_val)))
  y_val.index = list(range(len(y_val)))

  return X_train, X_val, y_train, y_val

# Create training and validation datasets for each generation of Pokemon images
X_train_gen1, X_val_gen1, y_train_gen1, y_val_gen1 = get_test_and_train_datasets(GEN1_PATH)
X_train_gen2, X_val_gen2, y_train_gen2, y_val_gen2 = get_test_and_train_datasets(GEN2_PATH)
X_train_gen3, X_val_gen3, y_train_gen3, y_val_gen3 = get_test_and_train_datasets(GEN3_PATH)
X_train_gen4, X_val_gen4, y_train_gen4, y_val_gen4 = get_test_and_train_datasets(GEN4_PATH)
X_train_gen5, X_val_gen5, y_train_gen5, y_val_gen5 = get_test_and_train_datasets(GEN5_PATH)

# Combine all generations into one training and one validation dataset
X_train = pd.concat([X_train_gen1, X_train_gen2, X_train_gen3, X_train_gen4, X_train_gen5], ignore_index=True)
X_val = pd.concat([X_val_gen1, X_val_gen2, X_val_gen3, X_val_gen4, X_val_gen5], ignore_index=True)
y_train = pd.concat([y_train_gen1, y_train_gen2, y_train_gen3, y_train_gen4, y_train_gen5], ignore_index = True)
y_val = pd.concat([y_val_gen1, y_val_gen2, y_val_gen3, y_val_gen4, y_val_gen5], ignore_index = True)

# Shuffle the training and validation data and re-index it
p = np.random.permutation(len(y_train))
X_train = X_train[p]
y_train= y_train[p]
X_train.index = list(range(len(X_train)))
y_train.index = list(range(len(y_train)))

q = np.random.permutation(len(y_val))
X_val = X_val[q]
y_val = y_val[q]
X_val.index = list(range(len(X_val)))
y_val.index = list(range(len(y_val)))




