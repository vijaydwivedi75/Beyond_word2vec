# ------------------------------------------------- #
# Siamese MLP for word-paraphrase similarity        #
# May 26, 2015, 1455 IST                            #
# Vijay Prakash Dwivedi                             #
#-------------------------------------------------- #

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os
import gc
import time

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, Lambda
from keras.datasets import imdb
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop, SGD
from keras import backend as K
import tensorflow as tf

# custom module to read activations of layers
from read_activations import get_activations

# ignore TensorFlow messages and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# os.system('clear')

# fix random seed for reproducibility
np.random.seed(7)

# importing custom module for data preprocessing
import preprocess_data

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
    # return K.abs(x-y)

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()

def create_base_network(input_dim):
    '''Base network to be shared.
    '''
    seq = Sequential()
    seq.add(Dense(512, input_shape=(input_dim,), activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(256, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(300, activation='tanh'))
    return seq

start_time = time.time()

x_data, y_data, visualise_pairs = preprocess_data.dataset()
input_dim = 1000
epochs = 25

tr_pairs = x_data[:263000]      # 263000
tr_y = y_data[:263000]          
te_pairs = x_data[263000:]      # 113000
te_y = y_data[263000:]
# print(tr_pairs[:, 1])

base_network = create_base_network(input_dim)
print(base_network.summary())

input_a = Input(shape=(input_dim,))
input_b = Input(shape=(input_dim,))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# train
sgd = SGD(lr=0.5)
model.compile(loss=contrastive_loss, optimizer=sgd)
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=128,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

# compute final accuracy on training and test sets
pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(pred, tr_y)
pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(pred, te_y)

print(model.summary())

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

data_for_test = visualise_pairs
functor = K.function([input_a, input_b]+ [K.learning_phase()], [processed_a, processed_b])
layer_out = functor([data_for_test[:, 0], data_for_test[:, 1], 0.])
# print(len(layer_out[0]))
# print(len(layer_out[1]))
# print(len(visualise_pairs))

paraphrase_embedding = tf.stack(layer_out[1])
batch_word_embedding = tf.stack(layer_out[0][144450:144500])

norm_paraphrase_embedding = tf.nn.l2_normalize(paraphrase_embedding, dim=1)
norm_batch_word_embedding = tf.nn.l2_normalize(batch_word_embedding, dim=1)

cosine_similarity = tf.matmul(norm_batch_word_embedding, tf.transpose(norm_paraphrase_embedding, [1, 0]))
# closest_words = tf.argmax(cosine_similarity, 1)
tf.InteractiveSession()
sim = cosine_similarity.eval()

words = preprocess_data.getWords()
paraphrase = preprocess_data.get_pp()

for i in range(50):
    valid_word = words[i+144450]
    # print(valid_word)
    top_k = 6       # number of nearest neighbours
    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
    
    log_str = "Nearest to '%s':" % valid_word
    
    for k in range(top_k):
        close_word = paraphrase[nearest[k]]
        log_str = "%s %s," % (log_str, close_word)

    print(log_str+"\n")

total_time_taken = time.time() - start_time
print("Total time taken: ", total_time_taken)

gc.collect()

##############################################################
#                      O  U  T  P  U  T                      #
##############################################################
# Using TensorFlow backend.
# Indexing word vectors.
# Found 400000 word vectors.
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# dense_1 (Dense)              (None, 512)               512512    
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 512)               0         
# _________________________________________________________________
# dense_2 (Dense)              (None, 256)               131328    
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 256)               0         
# _________________________________________________________________
# dense_3 (Dense)              (None, 300)               77100     
# =================================================================
# Total params: 720,940
# Trainable params: 720,940
# Non-trainable params: 0
# _________________________________________________________________
# None
# Train on 263000 samples, validate on 113000 samples
# Epoch 1/25
# 263000/263000 [==============================] - 11s - loss: 0.2327 - val_loss: 0.2260
# Epoch 2/25
# 263000/263000 [==============================] - 10s - loss: 0.2088 - val_loss: 0.2256
# Epoch 3/25
# 263000/263000 [==============================] - 10s - loss: 0.2004 - val_loss: 0.2184
# Epoch 4/25
# 263000/263000 [==============================] - 10s - loss: 0.1962 - val_loss: 0.2185
# Epoch 5/25
# 263000/263000 [==============================] - 9s - loss: 0.1938 - val_loss: 0.2170
# Epoch 6/25
# 263000/263000 [==============================] - 10s - loss: 0.1918 - val_loss: 0.2206
# Epoch 7/25
# 263000/263000 [==============================] - 9s - loss: 0.1902 - val_loss: 0.2187
# Epoch 8/25
# 263000/263000 [==============================] - 10s - loss: 0.1885 - val_loss: 0.2181
# Epoch 9/25
# 263000/263000 [==============================] - 10s - loss: 0.1869 - val_loss: 0.2160
# Epoch 10/25
# 263000/263000 [==============================] - 10s - loss: 0.1856 - val_loss: 0.2181
# Epoch 11/25
# 263000/263000 [==============================] - 9s - loss: 0.1846 - val_loss: 0.2190
# Epoch 12/25
# 263000/263000 [==============================] - 10s - loss: 0.1835 - val_loss: 0.2197
# Epoch 13/25
# 263000/263000 [==============================] - 10s - loss: 0.1825 - val_loss: 0.2201
# Epoch 14/25
# 263000/263000 [==============================] - 10s - loss: 0.1814 - val_loss: 0.2206
# Epoch 15/25
# 263000/263000 [==============================] - 10s - loss: 0.1806 - val_loss: 0.2180
# Epoch 16/25
# 263000/263000 [==============================] - 9s - loss: 0.1795 - val_loss: 0.2216
# Epoch 17/25
# 263000/263000 [==============================] - 10s - loss: 0.1788 - val_loss: 0.2220
# Epoch 18/25
# 263000/263000 [==============================] - 10s - loss: 0.1780 - val_loss: 0.2225
# Epoch 19/25
# 263000/263000 [==============================] - 10s - loss: 0.1769 - val_loss: 0.2227
# Epoch 20/25
# 263000/263000 [==============================] - 9s - loss: 0.1764 - val_loss: 0.2219
# Epoch 21/25
# 263000/263000 [==============================] - 10s - loss: 0.1758 - val_loss: 0.2232
# Epoch 22/25
# 263000/263000 [==============================] - 9s - loss: 0.1751 - val_loss: 0.2269
# Epoch 23/25
# 263000/263000 [==============================] - 10s - loss: 0.1743 - val_loss: 0.2233
# Epoch 24/25
# 263000/263000 [==============================] - 10s - loss: 0.1736 - val_loss: 0.2323
# Epoch 25/25
# 263000/263000 [==============================] - 10s - loss: 0.1729 - val_loss: 0.2282
# ____________________________________________________________________________________________________
# Layer (type)                     Output Shape          Param #     Connected to                     
# ====================================================================================================
# input_1 (InputLayer)             (None, 1000)          0                                            
# ____________________________________________________________________________________________________
# input_2 (InputLayer)             (None, 1000)          0                                            
# ____________________________________________________________________________________________________
# sequential_1 (Sequential)        (None, 300)           720940      input_1[0][0]                    
#                                                                    input_2[0][0]                    
# ____________________________________________________________________________________________________
# lambda_1 (Lambda)                (None, 1)             0           sequential_1[1][0]               
#                                                                    sequential_1[2][0]               
# ====================================================================================================
# Total params: 720,940
# Trainable params: 720,940
# Non-trainable params: 0
# ____________________________________________________________________________________________________
# None
# * Accuracy on training set: 91.06%
# * Accuracy on test set: 74.93%