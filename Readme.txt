Included is all the code used to for our genre classification project. All
python code is to be used with a Python 3.4 interpreter.

- query.py - Contains the code used for querying songs of different genres.
  Data is stored in a "pickled" data file.
- save-clean.py - Reads the queried files, cleans it, and stores it in a new
  pickled data file.
- save-compress.py - Save the compressed versions of the songs
- split.py - Reads the cleaned data files, and then stores a random split
  using an equal number of each genre for the training and test data.
- bag-words-and-repetitiveness.py - Contains the majority of this project's
  code. It includes the function that builds the model, which accepts different
  types of vectorizers and functions that will convert lyrics to their
  corresponding n-grams.
- compress.py - Contains the code that reads the compressed songs and measures
  the effectiveness of a model using it.

Libraries that were used:
- lz77kit - for compressing text https://github.com/olle/lz77-kit
- NTLK - for data cleaning
- re - for data cleaning
- numpy - for storing the feature vectors
- pickle - for storing data in files for later use
- Scikit's sklearn - for building the models and generating features for the
  bag of wordsmodel
- spotipy - for making Spotify API requests
- pylyrics - for pulling lyrics from lyrics.wikia.com
- scipy - for performing the kurtosis measurement
