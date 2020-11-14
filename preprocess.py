import numpy as np
import pandas as pd
import random
import io
import json
from utils import get_sentiment, process
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from config import *

def load_data():

  df = pd.read_csv('data/Reviews.csv')
  df = df[['Text', 'Score']]
  df['Review'] = df['Text']
  df['rating'] = df['Score']
  df.drop(['Text', 'Score'], axis=1, inplace=True)

  df = df[df['rating'] != 3]
  df.drop_duplicates(subset=['Review', 'rating'], keep='first', inplace=True)

  df['sentiment'] = df['rating'].apply(get_sentiment)

  df.drop('rating', axis=1, inplace=True)


  corpus = []
  for index, row in df.iterrows():
    list_item = []
    list_item.append(row["Review"])
    if(row['sentiment'] == 0):
      list_item.append(0)
    else:
      list_item.append(1)
    corpus.append(list_item)

  random.shuffle(corpus)

  sentences = []
  labels = []
  for x in range(len(corpus)):
    sentence = process(corpus[x][0])
    sentence = corpus[x][0]
    sentences.append(sentence)
    labels.append(corpus[x][1])

  train_portion = 0.6
  val_portion = 0.2 
  train_index = int(len(corpus) * train_portion)
  val_index = train_index + int(len(corpus) * val_portion)

  x_train = sentences[:train_index]
  x_val = sentences[train_index:val_index]
  x_test = sentences[val_index:]

  y_train = labels[:train_index]
  y_val = labels[train_index:val_index]
  y_test = labels[val_index:]


  tokenizer = Tokenizer(num_words=top_words, oov_token=oov_tok)
  tokenizer.fit_on_texts(x_train)
  # save vocab
  tokenizer_json = tokenizer.to_json()
  with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))
  
  word_index = tokenizer.word_index
  vocab_size = len(word_index) + 1

  x_train = tokenizer.texts_to_sequences(x_train)
  x_val = tokenizer.texts_to_sequences(x_val)
  x_test = tokenizer.texts_to_sequences(x_test)

  x_train = pad_sequences(x_train, maxlen=max_length, padding=padd_type, truncating=trunc_type)
  x_val = pad_sequences(x_val, maxlen=max_length, padding=padd_type, truncating=trunc_type)
  x_test = pad_sequences(x_test, maxlen=max_length, padding=padd_type, truncating=trunc_type)

  x_train = np.array(x_train)
  x_val = np.array(x_val)
  x_test = np.array(x_test)
  y_train = np.array(y_train)
  y_val = np.array(y_val)
  y_test = np.array(y_test)

  return x_train, y_train, x_val, y_val, x_test, y_test, word_index
