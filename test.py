import argparse
import json
import io
import tensorflow as tf
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from utils import process
from config import max_length, padd_type, trunc_type

model = load_model('models/1stmodel.h5')

def parse_argument():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--review', help='Review of the product')
    return parser.parse_args()


def load_tokenizer():
    with open('tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    return tokenizer

def predict():
    args = parse_argument()
    review = args.review
    
    tokenizer = load_tokenizer()
    
    processed_review = process(review)
    encoded_review = tokenizer.texts_to_sequences([processed_review])[0]
    encoded_review = pad_sequences([encoded_review], maxlen=max_length, padding=padd_type, truncating=trunc_type)
    pred = model.predict(encoded_review)

    if pred[0][0] > 0.6:
        print('Positive with {}%'.format(pred[0][0]*100))
    else:
        print('Negative with {}%'.format(100-pred[0][0]*100))

if __name__ == "__main__":
    predict()
    
