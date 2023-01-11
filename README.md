# Machine Learning Typing Pokedex <img src="https://user-images.githubusercontent.com/91648600/211432292-98c9c826-e7ef-4578-8fa7-94c5b48cea1a.png" width="30">
A trained neural network that can be used to determine a Pokemon's typing given its image.

 ## Background
 
 Pokemon are creatures from the hit video game franchise that are captured and trained to battle against other Pokemon. Each Pokemon has one or more elemental types which largely determines its moves in combat, and if it will have an advantage in a particular fight. For instance, Pikachu, the adorable yellow mascot of the franchise is an electric type.
 <img src="https://user-images.githubusercontent.com/91648600/211431531-3794c806-ca96-4b5d-b50b-7ac1a82f6487.png" width="100" align="left">

 This means it an advantage against water types and a disadvantage against ground types. The goal of the game is to Catch Em' All, by capturing them and recording their data - including their type - in a logging device known as a Pokedex. Thus, if a Pokemon is not captured, you cannot determine its Typing. The aim of this Neural Network is to discern a Pokemon's type without having to capture it, using its image.

## Where does Machine Learning come in?
This is a multi-label classification scenario as each Pokemon can have one or two elemental types, so we will have cases where we need to assign more than one label to a target input. In this case, we will classify each image using 18 different outputs, corresponding to the 18 types from Pokemon Generations 1 through 5.

## Noteworthy Observations
- The number of individual Pokemon is quite small, so we use Pokemon sprites from multiple generations and games, in addition to back sprites, shiny variants, and female Pokemon to increase our training dataset to consist of >15,000 images.
- We use the sigmoid function to train the model rather than the softmax function. The softmax function is suited for cases where a single class is assigned to each input whereas the sigmoid function calculates the probabilities of each class independently. For instance, a Pokemon being a fire type does not lower the probability of it being any other type.

## Files
1. poke_ml.py - Parses csv file, generates the neural network, and trains it.
2. test_models_results.py - Tests the model given an array of Pokemon names.

## Next Steps
- There is a large variance in the number of Pokemon of each type. For instance, there are almost 120 Water Pokemon compared to the sub 40 Ice Pokemon when viewing generations 1 through 6. As a result, the neural network will likely perform more accurately when predicting Water Pokemon as opposed to Ice. This could be addressed by taking subsets of images to train using an equal type distribution.
- Experiment by training using different loss functions to see if there is an increase in accuracy.
- Perform the neural network training using some images from Generation 6 onwards. This is because 3D Pokemon models were introduced in this generation, which can increase the accuracy of the model when predicting images past the sixth generation.

## Acknowledgements
- Pokemon Sprites/Images are sourced from the PokeAPI repository: https://github.com/PokeAPI/sprites.
- Kaggle dataset containing each Pokemon's type and number in the national Pokedex: https://www.kaggle.com/datasets/abcsds/pokemon.
- Article detailing Multi-Label Image Classification in TensorFlow 2.0 and providing code examples: https://towardsdatascience.com/multi-label-image-classification-in-tensorflow-2-0-7d4cf8a4bc72.

