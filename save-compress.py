import pickle
from lz77kit.src.main.python.lz77 import Compressor

c = Compressor()
count = [0]

def compress(line):
  count[0] += 1
  if (count[0] % 50 == 0):
    print('.', end='', flush=True)
  if (count[0] % 1000 == 0):
    print(count[0])
  return c.compress(line)

print('Loading data...')

cleaned_file = open('split_new', 'rb')
data = pickle.load(cleaned_file)
cleaned_file.close()

print('Compressing data', end='')

data['country_train'] = [compress(s) for s in data['country_train']]
data['hiphop_train'] = [compress(s) for s in data['hiphop_train']]
data['metal_train'] = [compress(s) for s in data['metal_train']]

data['country_test'] = [compress(s) for s in data['country_test']]
data['hiphop_test'] = [compress(s) for s in data['hiphop_test']]
data['metal_test'] = [compress(s) for s in data['metal_test']]

f=open('compressed_data', 'wb')
pickle.dump(data,f)
f.close()
