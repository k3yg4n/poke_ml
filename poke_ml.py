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
import keras.utils as image

from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.calibration import calibration_curve
from keras import layers

from utils import *
import glob
from pathlib import Path
from datetime import datetime

# File path constants
GEN1_PATH = Path("./sprites/sprites/pokemon/versions/generation-i")
GEN2_PATH = Path("./sprites/sprites/pokemon/versions/generation-ii")
GEN3_PATH = Path("./sprites/sprites/pokemon/versions/generation-iii")
GEN4_PATH = Path("./sprites/sprites/pokemon/versions/generation-iv")
GEN5_PATH = Path("./sprites/sprites/pokemon/versions/generation-v")

# Other constants
IMG_SIZE = 224 # Width and height of normalized images to match input format of model
CHANNELS = 3 # Keep RGB channels to match input format of model
BATCH_SIZE = 256 # Large enough to store an F1 score (measures accuracy)
AUTOTUNE = tf.data.experimental.AUTOTUNE # Adapt preprocessing and prefetching dynamically
SHUFFLE_BUFFER_SIZE = 1024 # Shuffle training data by a chunk of 1024 observations
LR = 1e-5 # The learning rate. Kept small because it is used for transfer learning
EPOCHS = 60 # Repetitions for the model's training

def get_normalized_images_and_labels(filepath, label):
  """Returns a tuple of normalized images array and labels array
  Arguments:
    filepath: string representing path to the image
    label: one dimensional array of size NUM_LABELS
  """
  # Read image from filepath
  image_string = tf.io.read_file(filepath)
  # Decode the image into a dense vector
  image_decoded = tf.image.decode_jpeg(image_string, channels=CHANNELS)
  # Resize the image to predefined dimensions
  image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])
  # Normalize the image from [0, 255] to [0.0, 1.0]
  image_normalized = image_resized / 255.0
  return image_normalized, label

def create_dataset(filenames, labels, is_training=True):
  """Load and parse a dataset
  Arguments:
    filenames: list of image paths
    labels: numpy array of shape (BATCH_SIZE, NUM_LABELS)
    is_training: boolean to indicate training dataset
  """

  # Create a dataset of file paths and labels
  path_and_label_dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
  # Parse and preprocess observations (values/rows)
  path_and_label_dataset = path_and_label_dataset.map(get_normalized_images_and_labels, num_parallel_calls=AUTOTUNE)

  if is_training:
    # Training set is a smaller dataset, so only load it once and keep it in memory
    path_and_label_dataset = path_and_label_dataset.cache()
    # Shuffle the data each buffer size
    path_and_label_dataset = path_and_label_dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)

  # Batch (process/group) the data into groups of BATCH_SIZE for multiple steps
  path_and_label_dataset = path_and_label_dataset.batch(BATCH_SIZE)
  # Fetch batches in the background while the model is training itself
  path_and_label_dataset = path_and_label_dataset.prefetch(buffer_size=AUTOTUNE)

  return path_and_label_dataset

# The loss function for our model
def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost) # average on all labels
    
    return macro_cost

# The metric for our model
def macro_f1(y, y_hat, thresh=0.5):
  """Compute the macro F1-score on a batch of observations (average F1 across labels)
  
  Args:
      y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
      y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
      thresh: probability value above which we predict positive
      
  Returns:
      macro_f1 (scalar Tensor): value of macro F1 for the batch
  """
  y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
  tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
  fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
  fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
  f1 = 2*tp / (2*tp + fn + fp + 1e-16)
  macro_f1 = tf.reduce_mean(f1)
  return macro_f1

def get_pokemon_dataframe():
  poke_df = pd.read_csv("pokemon.csv")
  poke_df = poke_df.rename(columns={"#": "Num", "Type 1": "Type1", "Type 2": "Type2"})
  poke_df = poke_df.drop_duplicates(subset=['Num'])
  poke_df = poke_df.fillna(value={"Type2": ""})
  poke_df['Types'] = poke_df['Type1'] + "," + poke_df['Type2']
  poke_df['Types'] = poke_df['Types'].apply(
    lambda types_entry: [l for l in str(types_entry).split(',') if l not in [""]]
    ) # Remove comma if types entry only has one type
  return poke_df

# Pandas output options
pd.set_option('display.max_colwidth',1000)

# Filter warnings and select logging level
warnings.filterwarnings('ignore')
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Create dataframe from .csv and perform data cleaning
poke_df = get_pokemon_dataframe()

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

# Binarize our types using one-hot encoding 
print("Labels:")
mlb = MultiLabelBinarizer()
mlb.fit(y_train)

NUM_LABELS = len(mlb.classes_)
for(i, label) in enumerate(mlb.classes_):
  print(f"{i} - {label}")

# Encode targets (labels) to multi-label-binarizer format
y_train_bin = mlb.transform(y_train)
y_val_bin = mlb.transform(y_val)

# Print 3 examples of Pokemon image and binary labels
for i in range(3):
    print(X_train[i], y_train_bin[i])

# Create datasets such that images and labels are formatted 
# in a manner that TensorFlow can process.
train_ds = create_dataset(X_train, y_train_bin)
val_ds = create_dataset(X_val, y_val_bin)

# Create and train our model.
#   Transfer learning is used to improve performance and decrease
#   training time. It consists of using a pre-trained model on 
#   a much larger dataset (not necessarily related to our own)
#   to identify classes in a new context.

# Import a pre-trained computer vision model for feature extraction
feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"

# Dense hidden layer (in between input and output layer where neurons take in weighted inputs
# generated using the feature extractor and produce an output through and activiation function)
feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=(IMG_SIZE,IMG_SIZE,CHANNELS))

# Prevent the parameters of the pre-trained model from being adjusted during training
feature_extractor_layer.trainable = False

# Create sequential model (a plain stack of layers which each layer has one input tensor)
# and one output tensor.

# The relu function is a piecewise linear function thay outputs the input if it is positive,
# otherwise outputs 0. It transforms the summed weighted input from the node into the activation
# of the node.

# The sigmoid function is known as the squashing function as its domain is the set of all
# real numbers and its range is (0,1). Thus, the output of the function is always between 0
# and 1, regardless if the number is very positive or very negative.

model = tf.keras.Sequential([
  feature_extractor_layer, # The first layer is the feature extractor layer
  layers.Dense(1024, activation='relu', name='hidden_layer'), # The second layer has 1024 neurons with the relu(recitifier) activiation function
  layers.Dense(NUM_LABELS, activation='sigmoid', name='output') # The third layer has NUM_LABELS units with the sigmoid activation function 
])

model.summary()

# Optimizer is the function that adapts the neural network after an epoch's results are analyzed
# Loss is the function that determines how well a specific prediction has performed. Used alongside optimizer in training.
# Metrics is a function used to judge the performance of a model (but its results are not used in training)

model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
  loss=macro_soft_f1,
  metrics=[macro_f1])

history = model.fit(train_ds,
                    epochs=EPOCHS,
                    validation_data=create_dataset(X_val, y_val_bin))

# After being trained, this model is exported to the models directory
t = datetime.now().strftime("%Y%m%d")
MODEL_EXPORT_PATH = "./models/my_model"
model.save(MODEL_EXPORT_PATH)
