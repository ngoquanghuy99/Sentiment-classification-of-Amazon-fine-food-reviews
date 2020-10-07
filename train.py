from preprocess import load_data
from config import *
from model import create_model
from keras.optimizers import Adam

x_train, y_train, x_val, y_val, x_test, y_test, word_index = load_data()

vocab_size = len(word_index) + 1

model = create_model(vocab_size, max_length, embedding_dim, word_index)
optimizer = Adam(lr=1e-3, beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))
model.save('models/2ndmodel.h5')
