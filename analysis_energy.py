import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

from wordcloud import WordCloud

# Constants
ENERGY = ['coal', 'solar', 'wind', 'gas', 'petro']
TWITTER_USER_REGEX = r'@([a-zA-Z0-9_]+)'

# Paths
path = os.getcwd()
path_to_ml_model = os.path.join(path, 'models', 'sentiment_classifier.pkl')
path_to_vectorizer = os.path.join(path, 'models', 'vectorizer.pkl')

# Import data
df_dict = {}
for energy in ENERGY:
    path_current = os.path.join(path, 'data', energy+'.pkl')
    df_dict[energy] = pd.read_pickle(path_current)

# # Import models
with open(path_to_ml_model, 'rb') as f:
    clf = pickle.load(f)
with open(path_to_vectorizer, 'rb') as f:
    vectorizer = pickle.load(f)

def clean_text(df, column):
    # Regex for pattern matching
    unicode_regex_1 = r'(\\u[0-9A-Fa-f]+)'
    unicode_regex_2 = r'[^\x00-\x7f]'
    url_regex = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
    # Unicode
    df[column] = df[column].str.replace(unicode_regex_1, r' ', regex=True)
    df[column] = df[column].str.replace(unicode_regex_2, r' ', regex=True)
    # Urls
    df[column] = df[column].str.replace(url_regex, 'url', regex=True)
    
    return df

for key, df in df_dict.items():
    # reset index
    df = df.reset_index(drop=True)
    # date formating
    df['date'] = pd.to_datetime(df['date'])
    # add mentions list column
    df['mentions'] = df['tweet'].str.findall(TWITTER_USER_REGEX).apply(','.join).str.split(',')
    
    # clean tweet column
    df['tweet_clean'] = df['tweet']
    df = clean_text(df, 'tweet_clean')
    
    # Vectorize clean tweets
    X_test = df['tweet_clean'].values
    X_test_vectors = vectorizer.transform(X_test)
    clf_prediction = clf.predict(X_test_vectors)
    df['sentiment'] = pd.Series(clf_prediction)
    df['energy'] = key
    path_pkl = os.path.join(path, 'data', key+'_clean.pkl')
    df.to_pickle(path_pkl)

# Create a master df
df_master = pd.concat([df_dict['coal'], df_dict['solar'], df_dict['wind'], df_dict['gas'], df_dict['petro']], ignore_index=True).reset_index(drop=True)
path_master = os.path.join(path, 'data', 'master.pkl')
df_master.to_pickle(path_master)

# Plot Sentiment vs. Energy type
# df_group_energy = df_master.groupby('energy')['sentiment'].value_counts()
# df_group_energy = pd.DataFrame(df_group_energy).transpose()
# df_group_energy = df_group_energy.rename(columns={'sentiment': 'count'})
# df_group_energy = df_group_energy.reset_index()

# plt.figure(figsize=(20, 10))
# sns.set_context('talk')
# sns.barplot(x=df_group_energy['sentiment'], y=df_group_energy['count'], hue=df_group_energy['energy'])
# plt.title('Sentiment By Energy Type')
# plt.xticks(ticks=[0, 1, 2], labels=['Negative', 'Neutral', 'Positive'])