import spotipy
sp = spotipy.Spotify()

offset = 0

def query_tracks(genre, offset):
    res = sp.search(q='genre:"' + genre + '"', limit=50, offset=offset,
            type='track', market=None)
    return [[r['name'], 
            [a['name'] for a in r['artists']], 
            genre] 
            for r in res['tracks']['items']]

print(query_tracks('hip hop', offset))


