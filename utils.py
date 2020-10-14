import re
import nltk
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def get_sentiment(rating):
  return 0 if rating < 3 else 1

def get_wordnet_pos(pos_tag):
  if pos_tag.startswith('J'):
    return 'a'
  elif pos_tag.startswith('V'):
    return 'v'
  elif pos_tag.startswith('R'):
    return 'r' # adverb
  else:
    return 'n'


def process(text):
  # lowercase
  text = text.lower()
  # remove all html tags 
  text = re.sub(r'<.*?>', '', text)
  # remove hyperlinks
  text = re.sub(r'http\S+', '', text)
  # remove special characters and punctuation
  text = re.sub('[^a-z ]', '', text)
  # split and remove stopwords
  words = [word for word in text.split() if word == 'not' or not word in stop_words]
  words_pos = pos_tag(words)
  lemmatized_words = []
  for word, tag in words_pos:
    pos = get_wordnet_pos(tag)
    lemmatized_word = lemmatizer.lemmatize(word, pos=pos)
    lemmatized_words.append(lemmatized_word)
  text = ' '.join(lemmatized_words)
  return text
