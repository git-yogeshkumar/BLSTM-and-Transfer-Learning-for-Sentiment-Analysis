import tflearn
import numpy as np
import argparse
import pickle
import string
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
import preprocessor as p
from collections import Counter
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix 
from tensorflow.contrib import learn
from tflearn.data_utils import to_categorical, pad_sequences
from scipy import stats
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
import tensorflow as tf
import os
os.environ['KERAS_BACKEND']='theano'
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model,Sequential
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers, optimizers
#These Functions are for loading the data
def load_data(filename):
    print("Loading data from file: " + filename)
    data = pickle.load(open(filename, 'rb'))
    data
    x_text = []
    labels = [] 
    for i in range(len(data)):
        x_text.append(p.tokenize((data[i]['text'])))
        labels.append(data[i]['label'])
    return x_text,labels

def get_filename(dataset):
    global HASH_REMOVE
    if(dataset=="twitter"):
        HASH_REMOVE = True
        EPOCHS = 10
        BATCH_SIZE = 128
        MAX_FEATURES = 2
        filename = "/home/yogesh/Desktop/Minor/data/twitter_data.pkl"
    return filename
data_1 = "twitter"
data_2 = "wiki"
model_type ="blstm_attention"
vector_type = "sswe"
embed_size = 50
oversampling_rate = 3
max_document_length=None
EMBED_SIZE = 50
EPOCHS = 5
BATCH_SIZE = 128
MAX_FEATURES = 2
NUM_CLASSES = 2
DROPOUT = 0.25
LEARN_RATE = 0.01
#HASH_REMOVE=True

x_text, labels = load_data(get_filename(data_1))
dict1 = {'racism':1,'sexism':1,'none':0} #Transfer learning only two classes
labels = [dict1[b] for b in labels]        
racism = [i for i in range(len(labels)) if labels[i]==2]
sexism = [i for i in range(len(labels)) if labels[i]==1]
x_text = x_text + [x_text[x] for x in racism]*(oversampling_rate-1)+ [x_text[x] for x in sexism]*(oversampling_rate-1)
labels = labels + [2 for i in range(len(racism))]*(oversampling_rate-1) + [1 for i in range(len(sexism))]*(oversampling_rate-1)
print("Counter after oversampling")
from collections import Counter
print(Counter(labels))

#splitting the data into training set and testing set
X_train, X_test, Y_train, Y_test = train_test_split( x_text, labels, random_state=121, test_size=0.10)

post_length = np.array([len(x.split(" ")) for x in x_text])
if(data_1 != "twitter"):
    max_document_length = int(np.percentile(post_length, 95))
else:
    max_document_length = max(post_length)
print("Document length : " + str(max_document_length))

vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, MAX_FEATURES)
vocab_processor = vocab_processor.fit(x_text)
trainX = np.array(list(vocab_processor.transform(X_train)))
testX = np.array(list(vocab_processor.transform(X_test)))

vocab_size = len(vocab_processor.vocabulary_)
print("Vocabulary Size: {:d}".format(vocab_size))

vocab = vocab_processor.vocabulary_._mapping
trainY = np.asarray(Y_train)
testY = np.asarray(Y_test)
       
trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
trainY = to_categorical(trainY, nb_classes=NUM_CLASSES)
testY = to_categorical(testY, nb_classes=NUM_CLASSES)

print("Running Model: " + model_type + " with word vector initiliazed with " + vector_type + " word vectors.")

#Attention Layer Class for blstm attention model
class AttLayer(Layer):

    def __init__(self, **kwargs):
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='kernel', 
                                      shape=(input_shape[-1],),
                                      initializer='random_normal',
                                      trainable=True)
        super(AttLayer, self).build(input_shape)

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))
        
        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
        
        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

model = Sequential()
model.add(Embedding(vocab_size, embed_size, input_length=trainX.shape[1]))
model.add(Dropout(0.25))
model.add(Bidirectional(LSTM(embed_size, return_sequences=True)))
model.add(AttLayer())
model.add(Dropout(0.50))
model.add(Dense(NUM_CLASSES, activation='softmax'))
adam = optimizers.Adam(lr=LEARN_RATE, beta_1=0.9, beta_2=0.999)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json,custom_objects={'AttLayer': AttLayer})
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def get_pred(xyz):
    tweet = xyz
    tokens=p.tokenize(tweet)
    l=[]
    l.append(tokens)
    arr=np.array(list(vocab_processor.transform(l)))
    tmp = loaded_model.predict_proba(arr)
    acc=max(tmp[0])*100
    print(arr)
    res=np.argmax(loaded_model.predict(arr),1)
    print(res)
    if res[0]==1 :
        val = "Sexist or Racist Post"
    else :
        val = "Neutral Post"
    return {'result': val, 'accuracy': acc}