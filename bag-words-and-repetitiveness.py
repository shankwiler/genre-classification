import pickle
import re
import numpy as np
import scipy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from lz77kit.src.main.python.lz77 import Compressor

ps = PorterStemmer()
c = Compressor()

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

def word_distribution(words):
    words_set = set(words)
    dist = {w:0 for w in words_set}
    for w in words:
        dist[w] += 1
    return dist

def simple_repet(words):
    # repetitiveness test 1.
    distr = word_distribution(words)
    if len(distr) == 0:
        return 0
    return len([i for i in distr.values() if i > 1]) / len(distr)

def simple_repet_vectorizer(data):
    return [[simple_repet(s.split())] for s in data]

def repet_average(words):
    # repetitiveness test 3.
    distr = word_distribution(words)
    return np.average(list(distr.values()))

def repet2(data):
    return [[repet_average(s.split())] for s in data]

def repet_max(words):
    # repetitiveness test 4.
    distr = word_distribution(words)
    return max(distr.values()) / len(words)

def repet3(data):
    return [[repet_max(s.split())] for s in data]

def repet_non_ones(words):
    # repetitiveness test 5.
    distr = word_distribution(words)
    return sum([i for i in distr.values() if i > 1]) / len(words)

def repet4(data):
    return [[repet_non_ones(s.split())] for s in data]

def repet_kurt(words):
    # repetitiveness test 6.
    distr = word_distribution(words)
    return scipy.stats.kurtosis(list(distr.values()))

def repet5(data):
    return [[repet_kurt(s.split())] for s in data ]

def bigrams(words):
    split = words.split()
    bis = []
    for i in range(1, len(split)):
        bis.append(split[i - 1] +'_'+ split[i])
    return ' '.join(bis)

def trigrams(words):
    split = words.split()
    tris = []
    for i in range(2, len(split)):
        tris.append(split[i-2] +'_'+ split[i-1] +'_'+ split[i])
    return ' '.join(tris)

def fourgrams(words):
    split = words.split()
    fours = []
    for i in range(3, len(split)):
        fours.append(split[i-3] + '_' + split[i-2] +'_'+ split[i-1] +'_'+ split[i])
    return ' '.join(fours)

def fivegrams(words):
    split = words.split()
    tris = []
    for i in range(4, len(split)):
        tris.append(split[i-4] +'_'+ split[i-3] +'_'+ split[i-2] +'_'+ split[i-1] +'_'+ split[i])
    return ' '.join(tris)

def precision_recall_f_measure(TP, FP, FN):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_measure = 2 * ((precision * recall)/(precision + recall))
    return precision, recall, f_measure

with open('split_new', 'rb') as f:
    data = pickle.load(f)

combined = data['country_train'] + data['hiphop_train'] + data['metal_train']

def train_test(vec1, vec2, preprocess=lambda x:x):
    # function that builds and tests the models
    features = vec1([preprocess(i) for i in combined])
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(features, (['country'] * len(data['country_train'])) + (['hiphop'] * len(data['hiphop_train'])) + \
        (['metal'] * len(data['metal_train'])))


    print('\nPredicting...')
    country_pred = forest.predict(vec2([preprocess(i) for i in data['country_test']]))
    hiphop_pred = forest.predict(vec2([preprocess(i) for i in data['hiphop_test']]))
    metal_pred = forest.predict(vec2([preprocess(i) for i in data['metal_test']]))
    print('Calculating Accuracy...')
    country_true_pos = len([i for i in country_pred if i == 'country'])
    country_false_pos = len([i for i in hiphop_pred if i == 'country']) + len([i for i in metal_pred if i == 'country'])
    country_false_neg = len([i for i in country_pred if i != 'country'])
    country_precision_recall_f = precision_recall_f_measure(country_true_pos, country_false_pos, country_false_neg)
    args = (country_true_pos, country_false_pos, country_false_neg) + country_precision_recall_f
    print('Country: TP: %d, FP: %d, FN: %d, Precision: %f, Recall: %f, F-Measure:%f' % args)

    hiphop_true_pos = len([i for i in hiphop_pred if i == 'hiphop'])
    hiphop_false_pos = len([i for i in country_pred if i == 'hiphop']) + len([i for i in metal_pred if i == 'hiphop'])
    hiphop_false_neg = len([i for i in hiphop_pred if i != 'hiphop'])
    hiphop_precision_recall_f = precision_recall_f_measure(hiphop_true_pos, hiphop_false_pos, hiphop_false_neg)
    args = (hiphop_true_pos, hiphop_false_pos, hiphop_false_neg) + hiphop_precision_recall_f
    print('Hip hop: TP: %d, FP: %d, FN: %d, Precision: %f, Recall: %f, F-Measure:%f' % args)

    metal_true_pos = len([i for i in metal_pred if i == 'metal'])
    metal_false_pos = len([i for i in hiphop_pred if i == 'metal']) + len([i for i in country_pred if i == 'metal'])
    metal_false_neg = len([i for i in metal_pred if i != 'metal'])
    metal_precision_recall_f = precision_recall_f_measure(metal_true_pos, metal_false_pos, metal_false_neg)
    args = (metal_true_pos, metal_false_pos, metal_false_neg) + metal_precision_recall_f
    print('Metal: TP: %d, FP: %d, FN: %d, Precision: %f, Recall: %f, F-Measure:%f' % args)

    print('(' + str(round(country_precision_recall_f[2],3)) + ',' + str(round(hiphop_precision_recall_f[2],3)) + ',' + str(round(metal_precision_recall_f[2],3)) + ')')
    return country_pred, hiphop_pred, metal_pred

def no_preprocess(x):
    return x

for vectorizer in [simple_repet_vectorizer, repet2, repet3, repet4, repet5]:
    for preprocess in [no_preprocess, bigrams, trigrams, fourgrams, fivegrams]:
        print(vectorizer.__name__, preprocess.__name__)
        train_test(vectorizer, vectorizer, preprocess)
        print('------------')

vectorizer = CountVectorizer(analyzer = "word",   \
    tokenizer=None,    \
    preprocessor=None, \
    stop_words=None,   \
    max_features=500,
    lowercase=False)
train_test(vectorizer.fit_transform, vectorizer.transform)
