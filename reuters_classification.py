import re

import scipy
from keras.engine.saving import load_model
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import reuters
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import hstack, csr_matrix

from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Bidirectional, \
    GlobalMaxPool1D, GRU
from keras.layers.embeddings import Embedding

from classification import callback_list

categories = reuters.categories()
cat_index = {}
cat_index_inv = {}
for cat in categories:
    cat_index[str(cat)] = categories.index(cat)
    cat_index_inv[str(categories.index(cat))] = cat

def transform_categories_to_vector(categories):
    vector = np.zeros(90)
    for categ in categories:
        vector[cat_index[categ]] = 1
    return vector


def transform_vector_to_categories(vector):
    categories = []
    index = 0
    vector = reversed([i[0] for i in sorted(enumerate(vector), key=lambda x:x[1])])
    for element in vector:
        categories.append(cat_index_inv[str(element)])
        index += 1
    return categories

lemmatizer = WordNetLemmatizer()
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
stop_words = set(stopwords.words("english"))

#text preprocessing
def preprocessing(r, stop_words = None):
    r = r.lower().replace("<br />", " ")
    r = re.sub(strip_special_chars, "", r.lower())
    if stop_words is not None:
        words = word_tokenize(r)
        filtered_sentence = []
        for w in words:
            w = lemmatizer.lemmatize(w)
            if w not in stop_words:
                filtered_sentence.append(w)
        return " ".join(filtered_sentence)
    else:
        return r


#list of documents in training
doc_x = []

#list of categories in train
doc_y = []

#list of documents in testing
doc_x_test = []

#list of categories in testing
doc_y_test = []


for doc_id in reuters.fileids():
    if doc_id.startswith("train"):
        doc_x.append(preprocessing(reuters.raw(doc_id), stop_words))
        doc_y.append(transform_categories_to_vector(reuters.categories(doc_id)))
    if doc_id.startswith("test"):
        doc_x_test.append(preprocessing(reuters.raw(doc_id), stop_words))
        doc_y_test.append(transform_categories_to_vector(reuters.categories(doc_id)))


doc_x = np.array(doc_x)
doc_x_test = np.array(doc_x_test)
doc_y_test = np.array(doc_y_test)
all_text = np.concatenate((doc_x, doc_x_test),axis=0)
#vectorize the docuemnts in word vectorizer
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 4),
    max_features=7000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(doc_x)
test_word_features = word_vectorizer.transform(doc_x_test)


#vectorize the docuemnts in char vectorizer

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 5),
    max_features=15000)

#train the vectorizer in all text data to make it know all words in the corpus of reuters
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(doc_x)
test_char_features = char_vectorizer.transform(doc_x_test)

train_features = hstack([train_char_features, train_word_features]).tocsr()
test_features = hstack([test_char_features, test_word_features]).tocsr()

train_features = csr_matrix(train_features).toarray()
test_features = csr_matrix(test_features).toarray()


l = np.array(doc_y)
vocabulary_size = 50000
model = Sequential()
model.add(Embedding(vocabulary_size, 100, input_length=22000))
model.add(Dropout(0.2))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(Bidirectional(GRU(100, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.2))
model.add(Dense(200, activation="relu"))
model.add(Dense(90, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print test_features.shape
model.fit(train_features, l, batch_size=28, epochs=2, validation_data=(test_features, doc_y_test), callbacks=callback_list)



def predict(test,model):
    test_yy = word_vectorizer.transform(test)
    test_yy_char = char_vectorizer.transform(test)
    test_features_s = hstack([test_yy, test_yy_char]).tocsr()
    test_features_arr = csr_matrix(test_features_s).toarray()
    yy_test = model.predict(test_features_arr, batch_size=1, verbose=1)
    predictions = []
    for element in yy_test:
        print transform_vector_to_categories(element)
        predictions.append(transform_vector_to_categories(element))
        #print element



