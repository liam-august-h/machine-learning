# -*- coding: utf-8 -*-
"""all_in_one.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1B2O9ESxP1lRlH-HtJBPrxevvXHYJCyvF
"""

## Install the PyDrive wrapper & import libraries.
## This only needs to be done once per notebook.
#!pip install -U -q PyDrive
#from pydrive.auth import GoogleAuth
#from pydrive.drive import GoogleDrive
#from google.colab import auth
#from oauth2client.client import GoogleCredentials
#
## Authenticate and create the PyDrive client.
## This only needs to be done once per notebook.
#auth.authenticate_user()
#gauth = GoogleAuth()
#gauth.credentials = GoogleCredentials.get_application_default()
#drive = GoogleDrive(gauth)
#
#train_file = drive.CreateFile({'id':'1Y_Z9-SfrlGsvkE2o0LbXj8HUlQA-BSJT'})
#train_file.GetContentFile('train.csv')
#print(train_file)
#
#import io
#
#train_file.content.seek(0)
#train_file_io = io.BytesIO(train_file.content.read())
#
## To determine which version you're using:
#!pip show tensorflow
#
## For the current version:
#!pip install --upgrade tensorflow
#
#import tensorflow as tf
#
## See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#
## https://keras.io/
#!pip install -q keras
#import keras

"""# For Mercari Price Suggestion Challenge"""

# %matplotlib inline

import math
#import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import pickle
import pprint
import random
import re
import scipy as sp
from scipy import stats, integrate
from scipy.sparse import coo_matrix, hstack
#import seaborn as sns
from sklearn import linear_model
from sklearn.decomposition import MiniBatchSparsePCA
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVR
import string


#nltk.download(['stopwords', 'punkt'])

#sns.set(color_codes=True)

random.seed(37)

#orig_train_data = pd.read_csv(train_file_io, sep='\t')
orig_train_data = pd.read_csv('./data/train.tsv', sep='\t')

train_data = orig_train_data
train_data_count = train_data.shape[0]
sub_train_data_count = 1024 #int(train_data_count * 0.002)
print('sub_train_data_count ' + str(sub_train_data_count))
sub_train_data = train_data.sample(sub_train_data_count, random_state=37)

# switch between train_data and sub_train_data
dataset = sub_train_data
#dataset = train_data

"""## Data Exploration

## Data Processing

### Drop *train_id*
"""

def drop_train_id(data):
    return data.drop(columns=['train_id'])

dataset = drop_train_id(dataset)

"""### Drop None *item_description*"""

def drop_missing_item_desc(data):
    return data.drop(data[data['item_description'].isna()].index)

dataset = drop_missing_item_desc(dataset)

"""### category_name

normally, category_name is in a form like 'a/b/c', but some products category_name contain more than three '/' character. Like Electronics/Computers & Tablets/iPad/Tablet/eB.... There will be more than three segments if to split category_name with '/'. It's reasonable to keep first two first two segment and merge others. For example, Electronics/Computers & Tablets/iPad/Tablet/eB... is transfered to Electronics, Computers & Tablets, and iPad/Tablet/eB...
"""

def filter_long_category(data):
    long_category_product = data[data['category_name'].str.count('/') > 3]
    return long_category_product['category_name'].unique()

#print(filter_long_category(dataset))

def split_category(data):
    splited = data['category_name'].str.split('/', expand=True)
    splited[2] = np.where(splited[3].isnull(),
                          splited[2], splited[2]+'/'+splited[3]+'/'+splited[4])
    data['c1'], data['c2'], data['c3'] = splited[0], splited[1], splited[2]
    return data

dataset = split_category(dataset)
print('c1 count is {}'.format(dataset['c1'].unique().shape[0]))
print('c2 count is {}'.format(dataset['c2'].unique().shape[0]))
print('c3 count is {}'.format(dataset['c3'].unique().shape[0]))

def drop_category_name(data):
    return data.drop(columns=['category_name'])

dataset = drop_category_name(dataset)

#dataset.dtypes

"""### Missing Data"""

pd.isna(dataset).sum()

"""### Processing Text

#### fill np.NaN and None with an empty string ('')
"""

dataset = dataset.fillna('')

"""#### merge fileds together"""

def merge_remove_text_fields(dataset, fields):
  dataset['word'] = ''
  for f in fields:
    dataset['word'] += dataset[f]

  return dataset.drop(fields, axis=1)

dataset = merge_remove_text_fields(dataset,
                  fields=['name', 'c1', 'c2', 'c3', 'brand_name', 'item_description'])

"""#### basic text processing"""

STEMMER = SnowballStemmer("english")
STOPS = stopwords.words('english')
PUNCTUATIONS_GROUP = re.compile('\W+')
NUMERIC = re.compile('\d+')

def basic_text_processing(row, stops=STOPS):
  document = row['word']

  #print(document)
  # tokenize
  tokens = word_tokenize(document)

  # lower case
  texts = [w.lower() for w in tokens ]

  # remove punctuation but keep a word if the punctuation is in the word
  texts = [w for w in texts if w not in string.punctuation]

  # remove punctuation groups
  texts = [w for w in texts if PUNCTUATIONS_GROUP.match(w) is None]

  # remove pure numeric words
  texts = [w for w in texts if NUMERIC.match(w) is None]

  # remove stopwords
  texts = [w for w in texts if w not in stops]

  # stemming ignore_stopwords=True
  texts = [STEMMER.stem(w) for w in texts]

  # remove duplicated
  texts = set(texts)

  if len(texts) == 0:
    texts = None
  else:
    texts = ' '.join(texts)

  #print('\t' + texts)
  row['word'] = texts
  return row

def normalize_text(dataset):
  return dataset.apply(basic_text_processing, axis=1)

#dataset = normalize_text(dataset)
#
#dataset.head()
#
#dataset = dataset.dropna(axis=0)
##dataset[dataset['word'].isna()]
#
#doc_len = dataset['word'].str.len()
#n_docs = len(dataset)
#print('the max length of documents is ' + str(doc_len.max()))
#max_doc_len = doc_len.max()
#print('there are {} documents'.format(len(dataset)))

"""### Baseline estimator with Sklearn

#### Root Mean Squared Logarithmic Error
"""

def rmsle(y_true, y_pred, **kwargs):
  """
  in case y_pred has negative values and y_pred + 1 can't make it greater than 0
  """
  y_pred[y_pred + 1 < 0] = -0.99 # make the error bigger
  return np.sqrt(np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean())

"""estimator and score"""

def regr_pred_and_score(regr, X_train, X_test, y_train, y_test):
  regr = regr.fit(X_train, y_train)
  pred = regr.predict(X_test)
  return rmsle(y_test, pred)

def regr_pred_and_score_cv(regr, X_train, X_test, y_train, y_test):
  skf = StratifiedKFold(n_splits=3)
  scoring_rmsle = make_scorer(rmsle, greater_is_better=False)
  scoring = {'explained_variance': 'explained_variance',
             'rmsle': scoring_rmsle}
  scores = cross_validate(regr, X_train, y_train, scoring=scoring,
          cv=3, return_train_score=True)
  #for k, v in scores.items():
  #    print('    --> {} is {}'.format(k, v))
  ret = '{:.2f} (+/- {:.2f})'.format(scores['test_rmsle'].mean(), scores['test_rmsle'].std())
  #return scores['test_rmsle']
  return ret


"""to combine text features with category features"""

def construct_feature_target(dataset_doc, dataset):
  """
  dataset_doc is a compressed sparse row matrix
  """
  item_condition = np.array(dataset['item_condition_id']).reshape([-1,1]).astype(int)
  item_condition = coo_matrix(item_condition)
  #print('item_condition {}'.format(item_condition.dtype))
  #print('item_condition_id {}'.format(item_condition.shape))

  shipping = np.array(dataset['shipping']).reshape([-1,1]).astype(int)
  shipping = coo_matrix(shipping)
  #print('shipping {}'.format(shipping.shape))
  #print('shipping {}'.format(shipping.dtype))

  features = hstack([dataset_doc, item_condition, shipping])

  #target = np.array(dataset['price']).reshape([-1,1])
  target = np.array(dataset['price'])
  return features, target

"""#### bags of words

Since *price* is related to some unique features, we should ignore a word if it keeps showing in most products. In other words, a low threshold of *max_df* is necessary
"""

def bags_of_words(dataset):
  count_vect = CountVectorizer(decode_error='ignore', max_df=0.25)
  train_count = count_vect.fit_transform(dataset['word'])
  print('bags_of_words shape is {}'.format(train_count.shape))

  X, y = construct_feature_target(train_count, dataset)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=37)
  return X_train, X_test, y_train, y_test

#X_train, X_test, y_train, y_test = bags_of_words(dataset)

#print(type(X_train))

"""#### LinearRegression"""

#regr = linear_model.LinearRegression()
#print('baseline: bags of words + linear regression score is {}'.format(
#    regr_pred_and_score(regr, X_train, X_test, y_train, y_test)))

"""so, with *bags of words* plus *LinearRegression*, we have a base line value 1.84

### Optimize Estimator

*bags of words* use occurence count may have a problem when longer documents may have higher average occurence count values which may confuse the estimator. we are going to use a better text model to optimize it

#### Tf and IDf
"""

def tfidf(dataset):
  tfidf_vect = TfidfVectorizer(max_df=0.3)
  train_count = tfidf_vect.fit_transform(dataset['word'])

  X, y = construct_feature_target(train_count, dataset)
  print('tfidf shape is {}'.format(X.shape))
  return train_test_split(X, y, test_size=0.4, random_state=37)

#X_train, X_test, y_train, y_test = tfidf(dataset)

"""#### LinearRegression"""

#regr = linear_model.LinearRegression(n_jobs=-1)
#print('tfidf + linear regression score is {}'.format(
#    regr_pred_and_score(regr, X_train, X_test, y_train, y_test)))

"""with a tfidf text model, *linearRegression* score is 1.76 now"""

def hashing(dataset):
  tfidf_vect = HashingVectorizer(binary=True)
  train_count = tfidf_vect.fit_transform(dataset['word'])

  X, y = construct_feature_target(train_count, dataset)
  print('hashing shape is {}'.format(X.shape))
  return train_test_split(X, y, test_size=0.4, random_state=37)

X_train, X_test, y_train, y_test = hashing(dataset)

regr = linear_model.LinearRegression(n_jobs=-1)
print('hashing + linear regression score is {}'.format(
    regr_pred_and_score_cv(regr, X_train, X_test, y_train, y_test)))

def hashingK(dataset):
  tfidf_vect = HashingVectorizer()
  train_count = tfidf_vect.fit_transform(dataset['word'])

  X, y = construct_feature_target(train_count, dataset)
  estimator = SelectKBest(f_regression).fit(X.toarray(), y)
  X = estimator.transform(X.toarray())
  print('hashingK shape is {}'.format(X.shape))
  return train_test_split(X, y, test_size=0.4, random_state=37)

X_train, X_test, y_train, y_test = hashingK(dataset)

regr = linear_model.LinearRegression(n_jobs=-1)
print('hashingK + linear regression score is {}'.format(
    regr_pred_and_score_cv(regr, X_train, X_test, y_train, y_test)))

def hashingP(dataset):
  tfidf_vect = HashingVectorizer()
  train_count = tfidf_vect.fit_transform(dataset['word'])

  X, y = construct_feature_target(train_count, dataset)
  estimator = SelectPercentile(f_regression).fit(X.toarray(), y)
  X = estimator.transform(X.toarray())
  print('hashingP shape is {}'.format(X.shape))
  return train_test_split(X, y, test_size=0.4, random_state=37)

X_train, X_test, y_train, y_test = hashingP(dataset)

regr = linear_model.LinearRegression(n_jobs=-1)
print('hashingP + linear regression score is {}'.format(
    regr_pred_and_score_cv(regr, X_train, X_test, y_train, y_test)))

#def tfidfK(dataset):
#  tfidf_vect = TfidfVectorizer(max_df=0.3)
#  train_count = tfidf_vect.fit_transform(dataset['word'])
#
#  X, y = construct_feature_target(train_count, dataset)
#  estimator = SelectKBest(f_regression).fit(X.toarray(), y)
#  X = estimator.transform(X.toarray())
#
#  print('tfidfK shape is {}'.format(X.shape))
#  return train_test_split(X, y, test_size=0.4, random_state=37)
#
#X_train, X_test, y_train, y_test = tfidfK(dataset)
#
#regr = linear_model.LinearRegression(n_jobs=-1)
#print('tfidfK + linear regression score is {}'.format(
#    regr_pred_and_score(regr, X_train, X_test, y_train, y_test)))
#
#
#def tfidfPCA(dataset):
#  tfidf_vect = TfidfVectorizer(max_df=0.3)
#  train_count = tfidf_vect.fit_transform(dataset['word'])
#
#  X, y = construct_feature_target(train_count, dataset)
#  estimator = MiniBatchSparsePCA(n_components=32).fit(X.toarray(), y)
#  X = estimator.transform(X.toarray())
#  print('tfidfPCA shape is {}'.format(X.shape))
#  return train_test_split(X, y, test_size=0.4, random_state=37)
#
#X_train, X_test, y_train, y_test = tfidfPCA(dataset)
#
#regr = linear_model.LinearRegression(n_jobs=-1)
#print('tfidfPCA + linear regression score is {}'.format(
#    regr_pred_and_score(regr, X_train, X_test, y_train, y_test)))

"""
#### Ridge

use penalty to avoid *variance*, since most of the features are formed by text and they are a big sparse martrix with high dimensions.
"""

regr = linear_model.Ridge(alpha=.1)
print('Ridge score is {}'.format(
    regr_pred_and_score_cv(regr, X_train, X_test, y_train, y_test)))

"""with *Ridge* algorithm, the score is 1.36 now"""

regr = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=3)
print('RidgeCV score is {}, alpha is {}'.format(
    regr_pred_and_score(regr, X_train, X_test, y_train, y_test), regr.alpha_))

"""#### ElasticNet

use *L1*  regularization to handle a sparse model but maintain regularization properties with *L2*
"""

regr = linear_model.ElasticNet(alpha=.1, l1_ratio=.7)
print('ElasticNet score is {}'.format(
    regr_pred_and_score_cv(regr, X_train, X_test, y_train, y_test)))

regr = linear_model.ElasticNetCV(l1_ratio=[.01, .05, .1, .5], alphas=[.01, .1, 1.0])
print('ElasticNetCV score is {}, alpha is {}, l1_ratio is {}'.format(
    regr_pred_and_score(regr, X_train, X_test, y_train, y_test), regr.alpha_, regr.l1_ratio_))

"""#### SVM"""

regr = SVR(kernel='rbf', gamma=0.1, C=0.1)
print('SVR rbf score is {}'.format(
    regr_pred_and_score_cv(regr, X_train, X_test, y_train, y_test)))

svr = GridSearchCV(SVR(gamma=0.1), cv=5,
                    param_grid={
                        'kernel': ['rbf'],
                        'C': [1, .1, .01, .001, .0001],
                        'gamma': ['auto', 50, 20, 10, 1, 0.1]
                    },
                   scoring=make_scorer(rmsle, greater_is_better=False)
                   )
print('SVR score is {}'.format(
    regr_pred_and_score(svr, X_train, X_test, y_train, y_test)))
print('best_params_ is {}'.format(svr.best_params_))

regr = RandomForestRegressor(n_estimators=X_train.shape[0])
print('RandomForestRegressor score is {}'.format(
    regr_pred_and_score_cv(regr, X_train, X_test, y_train, y_test)))

regr = GridSearchCV(RandomForestRegressor(), cv=3,
                    param_grid={
                        'n_estimators': range(1000, 20000, 2000)
                    },
                   scoring=make_scorer(rmsle, greater_is_better=False)
                   )
print('RandomForestRegressor GridSearchCV score is {}'.format(
    regr_pred_and_score(regr, X_train, X_test, y_train, y_test)))
print('best_params_ is {}'.format(regr.best_params_))

svr = SVR(kernel='rbf', gamma=50, C=1)
regr = AdaBoostRegressor(base_estimator=svr, n_estimators=1000,
        learning_rate=.6)
print('AdaBoostRegressor score is {}'.format(
    regr_pred_and_score_cv(regr, X_train, X_test, y_train, y_test)))

regr = GridSearchCV(AdaBoostRegressor(base_estimator=svr), cv=3,
                    param_grid={
                        'n_estimators': range(1000, 20000, 2000),
                        'learning_rate': [.1, .5, .8, 1]
                    },
                   scoring=make_scorer(rmsle, greater_is_better=False)
                   )
print('AdaBoostRegressor GridSearchCV score is {}'.format(
    regr_pred_and_score(regr, X_train, X_test, y_train, y_test)))
print('best_params_ is {}'.format(regr.best_params_))

#MAX_DOCUMENT_LENGTH = max_doc_len * 0.75
#MIN_FREQUENCY = n_docs * 0.001
#vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
#  MAX_DOCUMENT_LENGTH, MIN_FREQUENCY)
#vocab_processor.fit(train_data_doc)
#n_words = len(vocab_processor.vocabulary_)
#print('Total words: %d' % n_words)
#
#def bag_of_words_model(features, labels, mode):
#  """A bag-of-words model. Note it disregards the word order in the text."""
#  bow_column = tf.feature_column.categorical_column_with_identity(
#      'words', num_buckets=n_words)
#  bow_embedding_column = tf.feature_column.embedding_column(
#      bow_column, dimension=MAX_DOCUMENT_LENGTH * 0.75)
#  bow = tf.feature_column.input_layer(
#      features, feature_columns=[bow_embedding_column])
#
#  return estimator_spec_for_softmax_classification(
#      logits=logits, labels=labels, mode=mode)
#
#words = np.array(list(vocab_processor.transform(train_data_doc)))
#print('words shape {}'.format(words.shape))
#item_condition = np.array(train_data['item_condition_id']).reshape([-1,1])
#print('item_condition_id {}'.format(item_condition.shape))
#shipping = np.array(train_data['shipping']).reshape([-1,1])
#print('shipping {}'.format(shipping.shape))
#
#X = np.concatenate((words, item_condition, shipping), axis=1)
#y = np.array(train_data['price']).reshape([-1,1])
#
#print('y {}'.format(y.shape))
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=37)

'''
tfidf + linear regression score is 2.4325014173516677
tfidf + Ridge score is 1.8588441998874234
tfidf + RidgeCV score is 1.0817120351734226, alpha is 1.0
tfidf + ElasticNet score is 0.8031445818535682
tfidf + ElasticNetCV score is 0.8041760006169305, alpha is 0.1, l1_ratio is 0.1
tfidf + SVR rbf score is 0.746172186159556
tfidf + SVR score is 0.7236980002132651
best_params_ is {'kernel': 'rbf', 'gamma': 1, 'C': 1}
'''
