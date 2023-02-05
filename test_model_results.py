# Run to test the model once it is generated and trained in poke_ml.py

from pathlib import Path
from tensorflow import keras
from poke_ml import MODEL_EXPORT_PATH, get_pokemon_dataframe, IMG_SIZE, CHANNELS
import keras.utils as image
import numpy as np
import pandas as pd

# Load the model
model = keras.models.load_model(MODEL_EXPORT_PATH)

# Get the pokemon data frame
poke_df = get_pokemon_dataframe()

# Multi-Label-Binarization classes, or pokemon Types
MLB_CLASSES = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire',
       'Flying', 'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison',
       'Psychic', 'Rock', 'Steel', 'Water']

# Path to generation 6 png's as we are trying to predict outside of our training data
GEN6_PATH = Path("./sprites/sprites/pokemon/versions/generation-vi/x-y")

poke_to_test = [
  "Delphox",
  "Clauncher",
  "Noivern",
  "Quilladin",
  "Gogoat",
  "Hawlucha",
  "Goomy",
  "Sylveon",
  "Chespin",
  "Froakie",
  "Fletchling",
  "Pikachu",
  "Charizard"
]

# Test the model.
# It was trained on the first 5 generations of Pokemon, so we will
# see how it performs when predicting Pokemon in Generation 6.
def show_prediction(name, model):

  # Get the Pokemon's number and type, and png path
  pokeId = poke_df.loc[poke_df['Name']==name]['Num'].iloc[0]
  types = poke_df.loc[poke_df['Name']==name]['Types'].iloc[0]
  img_path = GEN6_PATH / f"{pokeId}.png"

  # Read and prepare image
  img = image.load_img(img_path, target_size=(IMG_SIZE,IMG_SIZE,CHANNELS))
  img = image.img_to_array(img)
  img = img/255 # normalize image
  img = np.expand_dims(img, axis=0)

  # Generate prediction
  prediction = (model.predict(img) > 0.5).astype('int')
  prediction = pd.Series(prediction[0])
  prediction.index = MLB_CLASSES
  prediction = prediction[prediction==1].index.values

  # Output the Pokemon's name, actual types, and the model's prediction
  print(f'\n\n{name}\n'
        f'Type\n{types}'
        f'\n\nPrediction\n{list(prediction)}\n')

# View the results of applying the mdoel on each pokemon in the poke_to_test array
for pokemon in poke_to_test:
  show_prediction(pokemon, model)