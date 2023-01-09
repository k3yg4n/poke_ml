from pathlib import Path
from tensorflow import keras
from poke_ml import MODEL_EXPORT_PATH, get_pokemon_dataframe, IMG_SIZE, CHANNELS
import keras.utils as image
import numpy as np
import pandas as pd

model = keras.models.load_model(MODEL_EXPORT_PATH)

poke_df = get_pokemon_dataframe()

MLB_CLASSES = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire',
       'Flying', 'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison',
       'Psychic', 'Rock', 'Steel', 'Water']

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
  "Fletchling"
]

# Test the model.
# It was trained on the first 5 generations of Pokemon, so we will
# see how it performs when predicting Pokemon in Generation 6.
# One nuance is that Pokemon shifted from 2D sprites to 3D models
# from gen6 onwards.
def show_prediction(name, model):

  # Get each Pokemon's unique number and type from the Gen6 games
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

  # Predict each Pokemon's type
  print(f'\n\n{name}\n'
        f'Type\n{types}'
        f'\n\nPrediction\n{list(prediction)}\n')

for pokemon in poke_to_test:
  show_prediction(pokemon, model)