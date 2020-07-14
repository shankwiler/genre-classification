import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from lz77kit.src.main.python.lz77 import Compressor

def compress_ratio(lines, comp_lines):
  pairs = list(zip(lines, comp_lines))
  ratios = np.zeros((len(pairs),1))
  count = 0
  for pair in pairs:
    orig = pair[0]
    comp = pair[1]
    ratios[count][0] = (len(comp) * 1.0 / len(orig))
    count += 1
  return ratios

def precision_recall_f_measure(TP, FP, FN):
  # returns the precision, recall, and F-measure given the true positives,
  # false positives, and false negatives
  precision = TP / (TP + FP)
  recall = TP / (TP + FN)
  f_measure = 2 * ((precision * recall)/(precision + recall))
  return precision, recall, f_measure


print('Loading data...')

cleaned_file = open('split_new', 'rb')
cleaned = pickle.load(cleaned_file)
cleaned_file.close()

comp_file = open('compressed_data', 'rb')
compressed = pickle.load(comp_file)
comp_file.close()


print('\nCalculating compression ratios')

features = compress_ratio(cleaned['country_train'], compressed['country_train'])
print('country done')

features = np.vstack((features, compress_ratio(cleaned['hiphop_train'],
        compressed['hiphop_train'])))
print('hip hop done')

features = np.vstack((features, compress_ratio(cleaned['metal_train'],
        compressed['metal_train'])))
print('metal done')


print('\nBuilding forest...')

forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(features, (['country'] * len(cleaned['country_train'])) + \
    (['hiphop'] * len(cleaned['hiphop_train'])) + \
        (['metal'] * len(cleaned['metal_train'])))

test_country = compress_ratio(cleaned['country_test'], compressed['country_test'])
test_hiphop = compress_ratio(cleaned['hiphop_test'], compressed['hiphop_test'])
test_metal = compress_ratio(cleaned['metal_test'], compressed['metal_test'])


print('\nPredicting...')

found = forest.predict(test_country)
country_true_pos = found.tolist().count('country')
hiphop_false_pos = found.tolist().count('hiphop')
metal_false_pos = found.tolist().count('metal')
country_false_neg = len(found) - country_true_pos
total = len(found)

found = forest.predict(test_hiphop)
hiphop_true_pos = found.tolist().count('hiphop')
country_false_pos = found.tolist().count('country')
metal_false_pos += found.tolist().count('metal')
hiphop_false_neg = len(found) - hiphop_true_pos
total += len(found)

found = forest.predict(test_metal)
metal_true_pos = found.tolist().count('metal')
country_false_pos += found.tolist().count('country')
hiphop_false_pos += found.tolist().count('hiphop')
metal_false_neg = len(found) - metal_true_pos
total += len(found)

country_precision_recall_f = precision_recall_f_measure(country_true_pos, country_false_pos, country_false_neg)
args = (country_true_pos, country_false_pos, country_false_neg) + country_precision_recall_f
print('Country: TP: %d, FP: %d, FN: %d, Precision: %f, Recall: %f, F-Measure: %f' % args)

hiphop_precision_recall_f = precision_recall_f_measure(hiphop_true_pos, hiphop_false_pos, hiphop_false_neg)
args = (hiphop_true_pos, hiphop_false_pos, hiphop_false_neg) + hiphop_precision_recall_f
print('Hip hop: TP: %d, FP: %d, FN: %d, Precision: %f, Recall: %f, F-Measure: %f' % args)

metal_precision_recall_f = precision_recall_f_measure(metal_true_pos, metal_false_pos, metal_false_neg)
args = (metal_true_pos, metal_false_pos, metal_false_neg) + metal_precision_recall_f
print('Metal: TP: %d, FP: %d, FN: %d, Precision: %f, Recall: %f, F-Measure: %f' % args)

wrong = country_false_neg + hiphop_false_neg + metal_false_neg
print('\n%d wrong of %d' % (wrong, total))
print('error: %f' % (wrong * 1.0 / total))
