import logging
import warnings

import numpy as np
import pandas as pd
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

# File path constants for pokemon sprites
GEN1_PATH = Path("./sprites/sprites/pokemon/versions/generation-i")
GEN2_PATH = Path("./sprites/sprites/pokemon/versions/generation-ii")
GEN3_PATH = Path("./sprites/sprites/pokemon/versions/generation-iii")
GEN4_PATH = Path("./sprites/sprites/pokemon/versions/generation-iv")
GEN5_PATH = Path("./sprites/sprites/pokemon/versions/generation-v")

# Other constants
IMG_SIZE = 224 # Width and height of normalized images. Used to match input format of the model
CHANNELS = 3 # Keep R,G, and B channels to match input format of model. Colour is important in distungishing type.
BATCH_SIZE = 256 # Number of samples processed before the model is updated, doing this saves memory. Large enough to store an F1 score, which measures accuracy
EPOCHS = 60 # Number of passes through entire data set, with dataset divided into batches
AUTOTUNE = tf.data.experimental.AUTOTUNE # Adapt preprocessing and prefetching dynamically/automatically -> helping with tasks such as reducing CPU idle time
SHUFFLE_BUFFER_SIZE = 1024 # Used to shuffle training data by 1024 observations
LR = 1e-5 # The learning rate, which determines how much to shift the model's weights depending on results. Kept small because it is used for transfer learning, 
          # because we are importing a pre-trained computer vision model for feature extraction based on a much larger dataset (not necessarily related to our own)
          # to use as a starting point for our model, applying it to a more specific task. 


def get_normalized_images_and_labels(filepath, label):
  """Returns a tuple of normalized images array and labels array
  Arguments:
    filepath: string representing the path to the image
    label: one dimensional array of size NUM_LABELS
  """
  # Read image from filepath
  image_string = tf.io.read_file(filepath)
  # Decode the image into a dense vector (machine readable format)
  image_decoded = tf.image.decode_jpeg(image_string, channels=CHANNELS)
  # Resize the image to predefined dimensions for our model
  image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])
  # Normalize the image from [0, 255] to [0.0, 1.0]
  image_normalized = image_resized / 255.0
  return image_normalized, label

def create_dataset(filenames, labels, is_training=True):
  """Load and parse a dataset
  Arguments:
    filenames: list of image paths
    labels: numpy array of shape (BATCH_SIZE, NUM_LABELS)
    is_training: boolean to indicate training dataset taking place
  """

  # Create a dataset of file paths and labels
  path_and_label_dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
  # Parse and preprocess observations (values/rows)
  path_and_label_dataset = path_and_label_dataset.map(get_normalized_images_and_labels, num_parallel_calls=AUTOTUNE)

  if is_training:
    # Training set is a smaller dataset, so only load it once and keep it in memory
    path_and_label_dataset = path_and_label_dataset.cache()
    # Maintain a buffer of SHUFFLE_BUFFER_SIZE elements and randomly selects the next element from it, then replaces it with the next input element. Effectively shuffles the data each buffer_size.
    # Shuffling is required because it can cause undesirable effects in the model if a specific generation is viewed all at once rather than introducing some degree of randomness.
    # For example, a generation with disproportionately many water pokemon running all at once can cause skews in the model.
    path_and_label_dataset = path_and_label_dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)

  # Batch (process/group) the data into groups of BATCH_SIZE for multiple steps
  path_and_label_dataset = path_and_label_dataset.batch(BATCH_SIZE)
  # Fetch batches in the background while the model is training itself
  path_and_label_dataset = path_and_label_dataset.prefetch(buffer_size=AUTOTUNE)

  return path_and_label_dataset

# The metric (evaluation function) for our model. Used to evaluate the perfomance of our classification model -- mostly treated like a blackbox
# The F1 score is the harmonic mean (average calculated by dividing # of observations by the reciprocal of each number in the series) of Precision and Recall (see https://en.wikipedia.org/wiki/Confusion_matrix).
# Precision is the number of true positive results (all predictions correct), divided by the number of all positive results (some predictions correct), including those not identified completely correctly.
# Recall is the number of true positive results divided by the number of all samples that should have been identified as positive. 
# This function computes as many F1-scores as the total number of labels (18 in our case, for the number of types), and then averages them to get a Macro F1-score. 
# It is reasonable to take the average over all labels because they each have the same importance in the multi-label classification task (each is a distinct type).
# SOURCED FROM: https://github.com/ashrefm/multi-label-soft-f1
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

# The loss function for our model, which determines how well a specific prediction has performed and is used alongside optimizer in training. -- mostly treated like a blackbox 
# Measures the model error on training batches and updates weights accordingly.
# Must be differentiable to backpropogate error in the neural network and update weights accordingly. (in each forward pass through a network, backpropogation performs a backwards pass while adjusting parameters).
# However, F1-score is not differentiable and so it cannot be used as a loss function because it needs binary predictions to be measured (0 and 1). We make the F1-score differentiable by computing the number of 
# true positives, true negatives, false positives, false negatives as a continous sum rather than discrete integer values by snapping inputs above a threshold to 1 and all others to 0. 
# This is accomplished by using probabilities without applying any threshold. 
# EX1:  
# If the target is 1 for a movie being Action and the model prediction for Action is 0.8, it will count as:
# 0.8 x 1 = 0.8 TP (because the target is 1 and the model predicted 1 with 0.8 chance)
# 0.2 x 1 = 0.2 FN (because the target is 1 and the model predicted 0 with 0.2 chance)
# 0.8 x 0 = 0 FP (because the target is 1 not 0, condition negative is not valid)
# 0.2 x 0 = 0 TN (because the target is 1 not 0, condition negative is not valid)

# EX 2: If the target is 0 for a movie being Action and the model prediction for Action is 0.8, it will count as:
# 0.8 x 0 = 0 TP (because the target is 0 not 1, condition positive is not valid)
# 0.2 x 0 = 0 FN (because the target is 0 not 1, condition positive is not valid)
# 0.8 x 1 = 0.8 FP (because the target is 0 and the model predicted 1 with 0.8 chance)
# 0.2 x 1 = 0.2 TN (because the target is 0 and the model predicted 0 with 0.2 chance)

# This version of F1-score is called a soft-F1-score and it can be used as a loss function. 
# SOURCED FROM: https://github.com/ashrefm/multi-label-soft-f1
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

def get_pokemon_dataframe():
  """Returns a pandas dataframe from the pokemon.csv file"""
  poke_df = pd.read_csv("pokemon.csv")
  poke_df = poke_df.rename(columns={"#": "Num", "Type 1": "Type1", "Type 2": "Type2"})
  poke_df = poke_df.drop_duplicates(subset=['Num']) # We do not want multiple entries for the same pokemon
  poke_df = poke_df.fillna(value={"Type2": ""}) # Change N/A fields in the type 2 field to be an empty string
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

# Generates training and testing datasets for Pokemon images (X) and types (y)
def get_test_and_train_datasets(gen_folder_path: Path):
  pre_gen5_poke_df = poke_df[poke_df['Num'] < 650] # Filter Pokemon from gen5 and earlier
  
  png_files = glob.glob(str(gen_folder_path) + "/**/*.png", recursive=True) # Generate list of png file path names
  
  # Remove invalid pokemon entries
  png_filtered = [file for file in png_files if Path(file).stem.isdigit()] # The png must be named after its corresponding pokedex entry number
  png_filtered = [file for file in png_filtered if int(Path(file).stem) in list(pre_gen5_poke_df['Num'])] # Filter pngs to gen5 and earlier
  
  # Change the index of the data frame to pokedex entry number
  pre_gen5_poke_df = pre_gen5_poke_df.set_index('Num')

  # Create new dataframe containing paths to filtered Pokemon images
  gen_poke_df = pd.DataFrame(png_filtered)

  # Add 'Types' column to new df using pre_gen5_poke_df data in its own 'Types' column
  gen_poke_df['Types'] = [pre_gen5_poke_df.loc[int(Path(file).stem), 'Types'] for file in png_filtered]

  # Rename the paths column to "sprites"
  gen_poke_df = gen_poke_df.rename(columns={0: "sprites"})

  # Output the dataframe to a csv file to verify correct processing
  gen_string = str(gen_folder_path).rsplit('\\', 1)[1]
  file_name = f"{gen_string}-poke-df.csv"
  df_output_filepath = Path(f"./output/{file_name}")
  gen_poke_df.to_csv(df_output_filepath)

  # Split the sprites and types dataset into training data and test(validation) data
  # Test/validation set will use 20% of the data. Training set will use the 80% remaining.
  # If training set is too small. the model will not have enough data to learn. 
  # If the validation set is too small, then evaluation metrics like accuracy, precision, recall, and F1 score will have large variance and will not lead to the proper tuning of the model.
  # Random state controls how data is randomly selected to go to the training or test set. Providing an integer ensures everytime we pass 44, we will get the same split everytime.
  # X refers to the pokemon png while y refers to the corresponding type.
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

# Shuffle the training data and re-index it
p = np.random.permutation(len(y_train))
X_train = X_train[p]
y_train= y_train[p]
X_train.index = list(range(len(X_train)))
y_train.index = list(range(len(y_train)))

# Shuffle the validation data and re-index it
q = np.random.permutation(len(y_val))
X_val = X_val[q]
y_val = y_val[q]
X_val.index = list(range(len(X_val)))
y_val.index = list(range(len(y_val)))

# Binarize our types using one-hot encoding 
# EX: [0 0 0 0 0 1 0 0 1] 
# Where each idx corresponds to a type. 1 represents true and 0 represents false
# In this example, the pokemon is of type 5 and 8. 
mlb = MultiLabelBinarizer()
mlb.fit(y_train)

# Print out a numbered list of the labels, ie: a list of pokemon types.
NUM_LABELS = len(mlb.classes_)
print("Labels:")
for(i, label) in enumerate(mlb.classes_):
  print(f"{i} - {label}")

# Encode targets (labels) to multi-label-binarizer format 
y_train_bin = mlb.transform(y_train)
y_val_bin = mlb.transform(y_val)

# Print 3 examples of Pokemon png path and their associated binary targets (type array)
for i in range(3):
    print(X_train[i], y_train_bin[i])

# Create datasets such that images and labels are formatted in a manner that TensorFlow can process.
train_ds = create_dataset(X_train, y_train_bin)
val_ds = create_dataset(X_val, y_val_bin)

# Create and train our model.
#   Transfer learning is used to improve performance and decrease
#   training time. It consists of using a pre-trained model on 
#   a much larger dataset (not necessarily related to our own)
#   to identify classes in a new context.

# Import a pre-trained computer vision model for feature extraction
feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"

# Create Dense hidden layer from the pre-trained feature extraction model
# (A layer in between input and output layers in which neurons take in weighted inputs
# generated using the feature extractor and produce an output through an activiation function)
feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=(IMG_SIZE,IMG_SIZE,CHANNELS))

# Prevent the parameters of the pre-trained model from being adjusted during training
feature_extractor_layer.trainable = False

# Create sequential model (a plain stack of layers which each layer has one input tensor and one output tensor (mathematical unit of data -> a set of primitive values shaped into an array of any number of dimensions)).
# EX: Tensor of shape [ 2, 1, 4 ]: [ [ [ 1., 2., 3., 4. ] ], [ [ 5., 6., 7., 8. ] ] ]
#                [#entries, #rows, #columns]
#   An activation function is a function used to get the output of a neural network's node.
#
#   The relu (rectified linear unit) function is a piecewise linear function that outputs the input if it is positive,
#   and otherwise outputs 0. It transforms a summed weighted input into the activation of a node. It is a popular
#   activation function because a model that uses it is easier to train and often achieves better performance.
#   A network like ours that uses the rectifier function for hidden layers are referred to as rectified networks.
#
#   The sigmoid function is known as the squashing function as its domain is the set of all
#   real numbers and its range is (0,1). This is because the output of the function is always between 0
#   and 1, regardless if the number is very positive or very negative. It snaps inputs much smaller than 0 to 0 and 
#   snaps inputs much larger than 1 to 1.0. It is a popular activation function for neural networks to predict the probability as an output.
#   In our case, we are predicting the probability that a pokemon is of a certain type, which will be between 0 and 1.

model = tf.keras.Sequential([
  feature_extractor_layer, # The first layer is the feature extractor layer
  layers.Dense(1024, activation='relu', name='hidden_layer'), # The second layer has 1024 neurons with the relu activiation function
  layers.Dense(NUM_LABELS, activation='sigmoid', name='output') # The third layer has NUM_LABELS (# of types) units with the sigmoid activation function 
])

model.summary()

# Compile the model.
#   Optimizer is the function that adapts the neural network after an epoch's results are analyzed
#   Loss is the function that determines how well a specific prediction has performed. Used alongside optimizer in training.
#   Metrics is a function used to judge the performance of a model (but its results are not used in training)

model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
  loss=macro_soft_f1,
  metrics=[macro_f1])

history = model.fit(train_ds,
                    epochs=EPOCHS,
                    validation_data=create_dataset(X_val, y_val_bin))

# After being trained, this model is exported to the models directory where it can be imported into test_model_results.py
t = datetime.now().strftime("%Y%m%d")
MODEL_EXPORT_PATH = "./models/my_model"
model.save(MODEL_EXPORT_PATH)
