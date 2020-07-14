import spotipy
from PyLyrics import *
import pickle
import os

from spotipy.oauth2 import SpotifyClientCredentials

#client_credentials_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(auth=os.environ['SPOTIFY_API_KEY'])

used = set()

def query_tracks(genre, offset):
    # get the tracks from spotify
    res = sp.search(q='genre:"' + genre + '"', limit=50, offset=offset,
            type='track', market=None)
    ret = []
    for r in res['tracks']['items']:
        if r['id'] in used:
            continue
        used.add(r['id'])
        ret.append(
            [r['name'],
            [a['name'] for a in r['artists']],
            r['id']
            ])
    return ret

def getLyrics(artist, title):
    # get the lyrics from lyrics.wikia.com
    try:
        return PyLyrics.getLyrics(artist, title)
    except ValueError:
        return None

def compileData(genres, offset):
    data = []
    for g in genres:
        data += [x for x in
          [[t[1][0], t[0], getLyrics(t[1][0], t[0]), g]
          for t in query_tracks(g, offset)] if x[2] is not None]
    return data

# adjustable range to search in, using different "offsets" in the list of
# results.
for i in range(0,15100,50):
    try:
      f = open('metal', 'rb')
      data = pickle.load(f)
      f.close()
    except:
      data = []

    genres = ['metal']
    data += compileData(genres, i)
    #print (data)
    print ('%d: %d' % (i, len(data)))

    f = open('metal', 'wb')
    pickle.dump(data,f)
    f.close()
