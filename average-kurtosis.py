import scipy, numpy,pickle
with open('split_new', 'rb') as f:
    d=pickle.load(f)
country = d['country_test'] + d['country_train']
hiphop = d['hiphop_test'] + d['hiphop_train']
metal = d['metal_test'] + d['metal_train']

def word_distribution(words):
    words_set = set(words)
    dist = {w:0 for w in words_set}
    for w in words:
        dist[w] += 1
    return dist

import scipy.stats
def repet_kurt(words):
    distr = word_distribution(words)
    return scipy.stats.kurtosis(list(distr.values()))

def repet5(data):
    return [[repet_kurt(s.split())] for s in data ]

print(numpy.average(repet5(country)), numpy.average(repet5(hiphop)), numpy.average(repet5(metal)))
