import pickle
import re
import numpy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

ps = PorterStemmer()

f = open('data_country_new','rb')
country = pickle.load(f)
f.close()
print('%d country songs' % len(country))

f = open('data_hiphop_new', 'rb')
hiphop = pickle.load(f)
f.close()
print('%d hip hop songs' % len(hiphop))

f = open('data_metal_new', 'rb')
metal = pickle.load(f)
f.close()
print('%d metal songs' % len(metal))

def clean_up(lines):
    # convert to lower case
    lines = lines.lower()

    # replace sequences of digits with NUM
    lines = re.sub("[0-9]+", "NUM", lines)
    lines = re.sub("[^A-z]", " ", lines)

    # split into individual words
    words = lines.split()

    # convert the stop words to a set
    stops = set(stopwords.words("english"))

    # Remove stop words
    meaningful_words = [w for w in words if not w in stops]

    # stem words
    meaningful_words = [ps.stem(w) for w in meaningful_words]

    # Join the words back into one string separated by space and return
    return( " ".join( meaningful_words ))

combined = [(clean_up(s[2]),'country') for s in country] + \
        [(clean_up(s[2]), 'hiphop') for s in hiphop] + \
        [(clean_up(s[2]),'metal') for s in metal]

f=open('cleaned_data', 'wb')
pickle.dump(combined,f)
f.close()
