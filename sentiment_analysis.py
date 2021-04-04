import numpy as np
import pandas as pd
import os
import re
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer

from textblob import TextBlob

# from matplotlib.colors import ListedColormap
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Import data
path = os.getcwd()
path_train = os.path.join(path, 'data', 'semeval-2017-train.csv')
path_test = os.path.join(path, 'data', 'tweets_df.pkl')
path_spacy = os.path.join(path, 'data', 'clean_spacy_df.pkl')
train = pd.read_csv(path_train, sep='\t')
train = train.sample(frac=1, random_state=42).reset_index(drop=True)
train_spacy = pd.read_pickle(path_spacy)
train_spacy = train_spacy.sample(frac=1, random_state=42).reset_index(drop=True)
test = pd.read_pickle(path_test)
test = test['tweet']

# Constants
num_neg = len(train[train['label'] == -1])
num_samples = len(train)

# Pre-processing
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

df_bow = train.copy()
df_bow = clean_text(df_bow, 'text')

# Split data
def balance_data(df, cuttoff_length=num_neg):
    num_pos = len(df[df['label'] == 1])
    num_neg = len(df[df['label'] == -1])
    num_neu = len(df[df['label'] == 0])
    
    df_pos = df[df['label'] == 1].iloc[:cuttoff_length]
    df_neg = df[df['label'] == -1].iloc[:cuttoff_length]
    df_neu = df[df['label'] == 0].iloc[:cuttoff_length]
    
    df = pd.concat([df_pos, df_neg, df_neu], ignore_index=True)
    return df

def split_data(df):
    stratify = df['label']
    train, val = train_test_split(df, test_size=0.1, random_state=42, stratify=stratify)
    return train, val

df_bow_train = balance_data(df_bow)

# SVC prediction using Textblob
df_bow_train_blob = df_bow_train.copy()
df_bow_train_blob['polarity'] = [TextBlob(x).sentiment.polarity for x in df_bow_train_blob['text']]
df_bow_train_blob['subjectivity'] = [TextBlob(x).sentiment.subjectivity for x in df_bow_train_blob['text']]
df_bow_train_blob = df_bow_train_blob.drop(columns=['text'])
df_train, df_val = split_data(df_bow_train_blob)

X_train = df_train[['polarity', 'subjectivity']].values
X_val = df_val[['polarity', 'subjectivity']].values
y_train = df_train['label'].values
y_val = df_val['label'].values

svc_txt = SVC()
svc_txt.fit(X_train, y_train)
svc_txt_prediction = svc_txt.predict(X_val)
svc_txt_accuracy = accuracy_score(y_val, svc_txt_prediction)
print("Textblob SVC Training accuracy Score  : ",svc_txt.score(X_train,y_train))
print("Textblob SVC Validation accuracy Score: ",svc_txt_accuracy)
print(classification_report(y_val, svc_txt_prediction))

# BOW SVC
df_train, df_val = split_data(df_bow_train)
X_train = df_train['text'].values
X_val = df_val['text'].values
y_train = df_train['label'].values
y_val = df_val['label'].values

vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_val_vectors = vectorizer.transform(X_val)

clf = SVC()
clf.fit(X_train_vectors, y_train)
clf_prediction = clf.predict(X_val_vectors)
clf_accuracy = accuracy_score(y_val, clf_prediction)
print("BOW SVC Training accuracy Score  : ",clf.score(X_train_vectors,y_train))
print("BOW SVC Validation accuracy Score: ",clf_accuracy)
print(classification_report(y_val, clf_prediction))

path_pickle_model = os.path.join(path, 'models', 'sentiment_classifier.pkl')
with open(path_pickle_model, 'wb') as f:
    pickle.dump(clf, f)

path_pickle_vectorizer = os.path.join(path, 'models', 'vectorizer.pkl')
with open(path_pickle_vectorizer, 'wb') as f:
    pickle.dump(vectorizer, f)

# Confusion Matrix
labels = ['Neg', 'Neu', 'Pos']

cm = confusion_matrix(y_val, clf_prediction)
df_cm = pd.DataFrame(cm, columns=labels)
df_cm = df_cm.rename(index={0: 'Neg', 1: 'Neu', 2: 'Pos'})

plt.figure(figsize=(10, 8))
sns.heatmap(df_cm, annot=True, fmt='d')
path_cm = os.path.join(path, 'images', 'confustion_matrix.png')
plt.title('Confustion Matrix - SVC Box of Words')
plt.savefig(path_cm, bbox_inches='tight')

























# # Ploting for classification results
# cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
# cmap_bold = ['darkorange', 'c', 'darkblue']
# h=0.02
# def plot_results(X_train, X_valid, model, hue_train, hue_valid):
#     x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
#     y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#     Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     plt.figure(figsize=(10, 6))
#     plt.contourf(xx, yy, Z, cmap=cmap_light)
#     sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=hue_train, palette=cmap_bold, alpha=1.0, edgecolor="black")
#     plt.xlim(xx.min(), xx.max())
#     plt.ylim(yy.min(), yy.max())
    
#     x_min, x_max = X_valid[:, 0].min(), X_valid[:, 0].max()
#     y_min, y_max = X_valid[:, 1].min(), X_valid[:, 1].max()
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#     Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     plt.figure(figsize=(10, 6))
#     plt.contourf(xx, yy, Z, cmap=cmap_light)
#     sns.scatterplot(x=X_valid[:, 0], y=X_valid[:, 1], hue=hue_valid, palette=cmap_bold, alpha=1.0, edgecolor="black")
#     plt.xlim(xx.min(), xx.max())
#     plt.ylim(yy.min(), yy.max())
#     return


'''
# Spacy Analysis
df_bow_spacy = train_spacy.copy()
df_bow_train_spacy = balance_data(df_bow_spacy, cuttoff_length=num_neg)
df_train_spacy, df_val_spacy = split_data(df_bow_train_spacy)

X_train_spacy = df_train_spacy['text'].values
X_val_spacy = df_val_spacy['text'].values
y_train_spacy = df_train_spacy['label'].values
y_val_spacy = df_val_spacy['label'].values

X_train_vectors_spacy = vectorizer.fit_transform(X_train_spacy)
X_val_vectors_spacy = vectorizer.transform(X_val_spacy)

# SVC prediction using BOW + Spacy
svc_spacy = SVC()
svc_spacy.fit(X_train_vectors_spacy, y_train_spacy)
svc_spacy_prediction = svc_spacy.predict(X_val_vectors_spacy)
svc_spacy_accuracy = accuracy_score(y_val_spacy, svc_spacy_prediction)
print("Spacy BOW SVC Training accuracy Score  : ",svc_spacy.score(X_train_vectors_spacy,y_train_spacy))
print("Spacy BOW SVC Validation accuracy Score: ",svc_spacy_accuracy)
print(classification_report(y_val_spacy, svc_spacy_prediction))
'''






'''
# BOW vectorizer sklearn
stop = list(stopwords.words('english'))
vectorizer = CountVectorizer(stop_words=stop)
X_train_vectors = vectorizer.fit_transform(X_train)
X_val_vectors = vectorizer.transform(X_val)

# SVC model using Textblob sentiment
clf = SVC()
clf.fit(X_train_vectors, y_train)
clf_prediction = clf.predict(X_val_vectors)
clf_accuracy = accuracy_score(y_val, clf_prediction)
print("Textblob SVC Training accuracy Score  : ",clf.score(X_train_vectors,y_train))
print("Textblob SVC Validation accuracy Score: ",clf_accuracy)
print(classification_report(y_val, clf_prediction))
'''


'''
df_train, df_val = train_test_split(df_bow, test_size=0.1, random_state=42, stratify=stratify)

X_train = df_train['text'].values
X_val = df_val['text'].values
y_train = df_train['label'].values
y_val = df_val['label'].values

# BOW vectorizer sklearn
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_val_vectors = vectorizer.transform(X_val)

# SVC model using Textblob sentiment
clf = SVC()
clf.fit(X_train_vectors, y_train)
clf_prediction = clf.predict(X_val_vectors)
clf_accuracy = accuracy_score(y_val, clf_prediction)
print("Textblob SVC Training accuracy Score  : ",clf.score(X_train_vectors,y_train))
print("Textblob SVC Validation accuracy Score: ",clf_accuracy)
print(classification_report(y_val, clf_prediction))
'''


# print(len(df_train), len(df_val))

# num_pos = len(df_train[df_train['label'] == 1])
# num_neg = len(df_train[df_train['label'] == -1])
# num_neu = len(df_train[df_train['label'] == 0])
# print(num_pos, num_neg, num_neu)
# num_pos = len(df_val[df_val['label'] == 1])
# num_neg = len(df_val[df_val['label'] == -1])
# num_neu = len(df_val[df_val['label'] == 0])
# print(num_pos, num_neg, num_neu)








'''
# TextBlob Sentiment Analysis
def plot_results(X_train, X_valid, model, hue_train, hue_valid):
    x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
    y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=hue_train, palette=cmap_bold, alpha=1.0, edgecolor="black")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
    x_min, x_max = X_valid[:, 0].min(), X_valid[:, 0].max()
    y_min, y_max = X_valid[:, 1].min(), X_valid[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    sns.scatterplot(x=X_valid[:, 0], y=X_valid[:, 1], hue=hue_valid, palette=cmap_bold, alpha=1.0, edgecolor="black")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    return


test_tweets = train.copy()
test_tweets['polarity'] = [TextBlob(x).sentiment.polarity for x in test_tweets['text']]
test_tweets['subjectivity'] = [TextBlob(x).sentiment.subjectivity for x in test_tweets['text']]
test_tweets = test_tweets.drop(columns=['text'])
stratify = test_tweets['label']
df_train, df_val = train_test_split(test_tweets, test_size=0.1, random_state=42, stratify=stratify)

# Using Textblob sentiment(polarity, subjectivity)
X_train = df_train[['polarity', 'subjectivity']].values
X_valid = df_val[['polarity', 'subjectivity']].values
y_train = df_train['label'].values
y_valid = df_val['label'].values

# SVC model using Textblob sentiment
svc = SVC()
svc.fit(X_train, y_train)
svc_prediction = svc.predict(X_valid)
svc_accuracy = accuracy_score(y_valid, svc_prediction)
print("Textblob SVC Training accuracy Score  : ",svc.score(X_train,y_train))
print("Textblob SVC Validation accuracy Score: ",svc_accuracy )
print(classification_report(y_valid, svc_prediction))

# Plotting Results
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ['darkorange', 'c', 'darkblue']
h=0.02
plot_results(X_train, X_valid, svc, y_train, y_valid)
plt.show()
'''









'''


# Sentiment Analysis Using Vader
analyzer = SentimentIntensityAnalyzer()
df_vader = train.copy()

df_vader['vs'] = [analyzer.polarity_scores(x) for x in df_vader['text']]
df_vader = df_vader.join(pd.json_normalize(df_vader.vs))
df_vader.drop(columns=['vs', 'text'], inplace=True)
stratify = df_vader['label']
df_train, df_val = train_test_split(df_vader, test_size=0.1, random_state=42, stratify=stratify)

# Using Vader sentiment(neg, neu, pos, compound)
X_train = df_train.drop('label', axis=1).values
X_valid = df_val.drop('label', axis=1).values
y_train = df_train['label'].values
y_valid = df_val['label'].values

# SVC model using Vader sentiment
# svc = SVC()
# svc.fit(X_train, y_train)
# svc_prediction = svc.predict(X_valid)
# svc_accuracy = accuracy_score(y_valid, svc_prediction)
# print("Vader SVC Training accuracy Score  : ",svc.score(X_train,y_train))
# print("Vader SVC Validation accuracy Score: ",svc_accuracy )
# print(classification_report(y_valid, svc_prediction))

# KNN model using Vader sentiment
n_neighbors = n_neighbors = list(range(12, 20, 4))

h=0.02
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ['darkorange', 'c', 'darkblue']

for n_neighbors in n_neighbors:
    print(f'Number of Neighbors: {n_neighbors}')
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train,y_train)
    knn_prediction = knn.predict(X_valid)
    knn_accuracy = accuracy_score(y_valid, knn_prediction)
    print("Vader KNN Training accuracy Score  : ",knn.score(X_train,y_train))
    print("Vader SVC Validation accuracy Score: ",knn_accuracy )
    print(classification_report(y_valid, knn_prediction))

'''























# Expanding the tweet using count vectorizer
# stop = list(stopwords.words('english'))
# vectorizer = CountVectorizer(decode_error = 'replace',stop_words = stop)

# X_train = vectorizer.fit_transform(df_train['text'].values)
# X_valid = vectorizer.fit_transform(df_val['text'].values)

# y_train = df_train['label'].values
# y_valid = df_val['label'].values





# # Spacy processing for tweet analysis
# spacy.prefer_gpu()
# nlp = spacy.load("en_core_web_sm")

# # Modify token matching for hashtags
# re_token_match = spacy.tokenizer._get_regex_pattern(nlp.Defaults.token_match)
# re_token_match = f"({re_token_match}|#\\w+)"
# nlp.tokenizer.token_match = re.compile(re_token_match).match

# def spacy_process(s, nlp, features):
#     s = s.lower()
#     doc = nlp(s)
#     lemmas = []
#     for token in doc:
#         lemmas.append(token.lemma_)
        
#     features |= set(lemmas)
    
#     freq = {'#': 0, '@': 0, 'URL': 0}
#     for word in lemmas:
#         freq[str(word)] = 0
#     for token in doc:
#         if '#' in str(token): freq['#'] += 1 # count num of hashtags
#         if '@' in str(token): freq['@'] += 1 # count num of mentions
#         if 'http://' in str(token): freq['URL'] += 1 # count num of URLs
#         freq[str(token.lemma_)] += 1
        
#     return features, freq

# preprocess_df = train
# features = set({'#', '@', 'URL'})

# bow_array = []
# for i in range(len(preprocess_df)):
#     features, freq = spacy_process(preprocess_df.iloc[i]['text'], nlp, features)
#     bow_array.append(freq)
    
# bow = pd.DataFrame('0', columns=features, index=range(len(preprocess_df)))
# bow['id'] = preprocess_df.index
# bow.set_index('id', drop=True, inplace=True)

# for i in range(len(preprocess_df)):
#     freq = bow_array[i]
#     for f in freq:
#         bow.loc[i+1, f] = freq[f]
        
# preprocess_df = preprocess_df.join(bow, lsuffix='_data') 

# y = preprocess_df['label']
# df_train, df_val = train_test_split(preprocess_df, test_size=0.1, random_state=42, stratify=y)

# print(df_train.shape, df_val.shape)

# def cal_sum(df, describe_which):
#     ''' 
#     Check the balance between original, test, and val using this function.
#     Prints results
#     '''
#     pos_sum = np.sum(df['label']==1)
#     neu_sum = np.sum(df['label']==0)
#     neg_sum = np.sum(df['label']==-1)
#     tot_sum = pos_sum + neu_sum + neg_sum
    
#     print(describe_which, ' Pos: ', pos_sum / tot_sum)
#     print(describe_which, ' Neg: ', neu_sum / tot_sum)
#     print(describe_which, ' Neu: ', neg_sum / tot_sum)
#     return

# cal_sum(preprocess_df, 'Original')
# cal_sum(df_train, 'Train')
# cal_sum(df_val, 'Validation')

# X_train = df_train.drop(columns=['label', 'text']).to_numpy()
# X_valid = df_val.drop(columns=['label', 'text']).to_numpy()

# y_train = df_train['label'].to_numpy()
# y_valid = df_val['label'].to_numpy()

















'''
# TextBlob Sentiment Analysis
test_tweets = train

test_tweets['polarity'] = [TextBlob(x).sentiment.polarity for x in test_tweets['text']]
test_tweets['subjectivity'] = [TextBlob(x).sentiment.subjectivity for x in test_tweets['text']]
test_tweets['predict'] = [1 if (x > 0.1 and y > 0.6) else (-1 if (x < -0.05 and y > 0.6) else 0) for x, y in zip(test_tweets['polarity'], test_tweets['subjectivity'])]
test_tweets['correct'] = [1 if x==y else 0 for x, y in zip(test_tweets['label'], test_tweets['predict'])]

num_correct = test_tweets['correct'].value_counts()[1]

print(f'Predict accuracy: {num_correct / len(test_tweets) * 100}')
'''



'''
train['text'] = process_tweets(train['text'])
# test = process_tweets(test)

# Sentiment Analysis Using Vader
analyzer = SentimentIntensityAnalyzer()
vs_results = pd.DataFrame()

vs_results['vs'] = [analyzer.polarity_scores(x) for x in train['text']]

vs_results = vs_results.join(pd.json_normalize(vs_results.vs))

vs_results.drop(columns=['vs'], inplace=True)
vs_results['label'] = train['label']

len_results = len(vs_results)
neg_count = len(vs_results[vs_results['neg'] >= 0.1])
neu_count = len(vs_results[vs_results['neu'] > 0.8])
pos_count = len(vs_results[vs_results['pos'] >= 0.1])

neg_label = len(vs_results[vs_results['label'] == -1])
neu_label = len(vs_results[vs_results['label'] == 0])
pos_label = len(vs_results[vs_results['label'] == 1])

print('neg count', neg_label)
print('neu count', neu_label)
print('pos count', pos_label)

print(vs_results.head(30))
print(vs_results.sample(n=30))
vs_results['predict'] = [1 if (x > 0.1 and y < 0.1) else (-1 if (y >= 0.1 and x <= 0.1) else 0) for x, y in zip(vs_results['pos'], vs_results['neg'])]
vs_results['predict_compound'] = [1 if x > 0.3 else (-1 if x < -0.3 else 0) for x in vs_results['compound']]

vs_results['correct'] = [1 if x==y else 0 for x, y in zip(vs_results['label'], vs_results['predict'])]
vs_results['correct_compound'] = [1 if x==y else 0 for x, y in zip(vs_results['label'], vs_results['predict_compound'])]

num_correct = vs_results['correct'].value_counts()[1]
num_correct_compound = vs_results['correct_compound'].value_counts()[1]

print(f'Predict accuracy: {num_correct / len_results * 100}')
print(f'Predict accuracy compound: {num_correct_compound / len_results * 100}')



sns.countplot(data=vs_results, x='label', hue='correct')
plt.show()
sns.countplot(data=vs_results, x='label', hue='correct_compound')

plt.show()





sample_size = min(len(train_pos), len(train_neg))

raw = np.concatenate((train_pos['text'].values[:sample_size], 
                 train_neg['text'].values[:sample_size]), axis=0)
labels = [1]*sample_size + [0]*sample_size

'''






















# # Constants for Regex
# url_regex = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
# twitter_handle_regex = r'@([a-zA-Z0-9_]+)'
# hashtag_regex = r'#([^\s]+)'
# emoticon_regex = r':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:'
# contraction_patterns = [ (r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'), (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'), (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'), (r'dont', 'do not'), (r'wont', 'will not') ]
# multiexclamation_regex = r'(\!)\1+'
# multiquestion_regex = r'(\?)\1+'
# multistop_regex = r'(\.)\1+'







# # Clean tweets for urls, @, etc.
# def replace_contractions(text):
#     patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
#     for (pattern, repl) in patterns:
#         (text, count) = re.subn(pattern, repl, text)
#     return text

# def process_tweets(tweets):
#     tweets = tweets.str.replace(url_regex, 'url', regex=True)
#     tweets = tweets.str.replace(twitter_handle_regex, 'user', regex=True)
#     tweets = tweets.str.replace(hashtag_regex, '', regex=True)
#     tweets = pd.Series([replace_contractions(x) for x in tweets])
#     tweets = tweets.str.lower()
#     tweets = tweets.str.replace(emoticon_regex, '', regex=True)
#     tweets = tweets.str.replace('  ', ' ')
#     tweets = tweets.str.replace(multiexclamation_regex, ' multiExclamation ', regex=True)
#     tweets = tweets.str.replace(multiquestion_regex, ' multiQuestion ', regex=True)
#     tweets = tweets.str.replace(multistop_regex, ' multiStop ', regex=True)
#     # tweets = tweets.str.replace(extra_regex, '', regex=True)
    
#     return tweets
