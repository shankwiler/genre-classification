import pickle
from sklearn.model_selection import train_test_split as tts
cleaned_file = open('cleaned_data', 'rb')
cleaned = pickle.load(cleaned_file)
cleaned_file.close()

# get rid of short songs and keep only the lyrics
country = [i[0] for i in cleaned if i[1] == 'country' and len(i[0].split()) >= 5]
hiphop = [i[0] for i in cleaned if i[1] == 'hiphop' and len(i[0].split()) >= 5]
metal = [i[0] for i in cleaned if i[1] == 'metal' and len(i[0].split()) >= 5]

# keep an even amount among all genres
min_len = min(len(country), len(hiphop), len(metal))

country = country[:min_len]
hiphop = hiphop[:min_len]
metal = metal[:min_len]

country_train, country_test, hiphop_train, hiphop_test, \
        metal_train, metal_test = tts(country, hiphop, metal, \
        test_size=0.25)

# save the split data in a pickle file
data = {
    'country_train': country_train,
    'country_test': country_test,
    'hiphop_train': hiphop_train,
    'hiphop_test': hiphop_test,
    'metal_train': metal_train,
    'metal_test': metal_test
}
f=open('split_new', 'wb')
pickle.dump(data, f)
f.close()
