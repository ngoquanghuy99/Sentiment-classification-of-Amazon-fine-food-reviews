import numpy as np
import io
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dropout, Conv1D, Activation, Dense, MaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

def create_model(vocab_size, max_length, embedding_dim, word_index):
  
  embeddings_index = {}
  with io.open('models/glove.6B.100d.txt', 'r', encoding='utf-8') as f:
    for line in f:
      values = line.strip().split()
      curr_word = values[0]
      coefs = np.asarray(values[1:], dtype='float64')
      embeddings_index[curr_word] = coefs

  embeddings_matrix = np.zeros((vocab_size, embedding_dim))
  for word, i in word_index.items():
    if i > vocab_size:
      continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
      embeddings_matrix[i] = embedding_vector

  model = Sequential()
  embedding_layer = Embedding(vocab_size, embedding_dim, input_length = max_length,
                                weights = [embeddings_matrix], trainable=False)
  model.add(embedding_layer)
  model.add(Dropout(0.2))

  model.add(Conv1D(64, 5))
  model.add(Activation('relu'))
  model.add(MaxPooling1D(pool_size = 4))

  model.add(Bidirectional(LSTM(100)))
  model.add(Dropout(0.3))

  model.add(Dense(1))
  model.add(Activation('sigmoid'))

  model.summary()
  return model




