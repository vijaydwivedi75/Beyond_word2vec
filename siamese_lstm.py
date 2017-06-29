# ------------------------------------------------- #
# Siamese LSTM for word-paraphrase similarity        #
# Jun 24, 2015, 1655 IST                             #
# Vijay Prakash Dwivedi                             #
#-------------------------------------------------- #

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os
import gc

from keras.preprocessing import sequence
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, Lambda
from keras.datasets import imdb
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop, Adamax
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
import preprocess_data_lstm
vocab_size = preprocess_data_lstm.vocab_size()
emb_dim = preprocess_data_lstm.embed_dim()

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
    # return labels[predictions.ravel() < 0.5].mean()
    return np.mean(np.equal(predictions.ravel() < 0.5, labels))

def create_base_network(timesteps, input_dim):
    '''Base network LSTM to be shared.
    '''
    seq = Sequential()
    # seq.add(LSTM(32, input_shape=(int(input_dim/emb_dim), emb_dim)))
    seq.add(LSTM(768, return_sequences=True, input_shape=(timesteps, input_dim)))
    seq.add(Dropout(0.6))
    seq.add(LSTM(512, return_sequences=True))
    seq.add(Dropout(0.6))
    seq.add(LSTM(300))
    # seq.add(Dropout(0.6))
    return seq

x_data, y_data, visualise_pairs = preprocess_data_lstm.dataset()
input_dim = 1000
epochs = 20

timesteps = int(input_dim/emb_dim)
input_dim = emb_dim

tr_pairs = x_data[:26300]		# 263000
tr_y = y_data[:26300]			
te_pairs = x_data[26300:]		# 113000
te_y = y_data[26300:]
print(tr_pairs[1, 1].shape)

base_network = create_base_network(timesteps, input_dim)
print(base_network.summary())

input_a = Input(shape=(timesteps, input_dim))
input_b = Input(shape=(timesteps, input_dim))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)
# model = load_model('checkpoints/model_siamese_lstm_0.764513274336.h5')

# train
rms = RMSprop()
opt = Adamax()
model.compile(loss=contrastive_loss, optimizer=opt)
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=384,
          epochs=epochs)
          # validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

# compute final accuracy on training and test sets
pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(pred, tr_y)
pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(pred, te_y)

print(model.summary())

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

model.save('checkpoints/model_siamese_lstm_'+str(te_acc)+'.h5')

data_for_test = visualise_pairs
functor = K.function([input_a, input_b]+ [K.learning_phase()], [processed_a, processed_b])
layer_out = functor([data_for_test[:, 0], data_for_test[:, 1], 0.])
# print(len(layer_out[0]))
# print(len(layer_out[1]))
# print(len(visualise_pairs))

paraphrase_embedding = tf.stack(layer_out[1])
batch_word_embedding = tf.stack(layer_out[0][14450:14500])

norm_paraphrase_embedding = tf.nn.l2_normalize(paraphrase_embedding, dim=1)
norm_batch_word_embedding = tf.nn.l2_normalize(batch_word_embedding, dim=1)

cosine_similarity = tf.matmul(norm_batch_word_embedding, tf.transpose(norm_paraphrase_embedding, [1, 0]))
# closest_words = tf.argmax(cosine_similarity, 1)
tf.InteractiveSession()
sim = cosine_similarity.eval()

words = preprocess_data_lstm.getWords()
paraphrase = preprocess_data_lstm.get_pp()

for i in range(50):
    valid_word = words[i+14450]
    # print(valid_word)
    top_k = 8       # number of nearest neighbours
    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
    
    log_str = "Nearest to '%s':" % valid_word
    
    for k in range(top_k):
        close_word = paraphrase[nearest[k]]
        log_str = "%s %s," % (log_str, close_word)

    print(log_str+"\n")

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
# dense_1 (Dense)              (None, 128)               128128    
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 128)               0         
# _________________________________________________________________
# dense_2 (Dense)              (None, 128)               16512     
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 128)               0         
# _________________________________________________________________
# dense_3 (Dense)              (None, 128)               16512     
# =================================================================
# Total params: 161,152
# Trainable params: 161,152
# Non-trainable params: 0
# _________________________________________________________________
# None
# Train on 2630 samples, validate on 1130 samples
# Epoch 1/20
# 2630/2630 [==============================] - 154s - loss: 0.4331 - val_loss: 0.2220
# Epoch 2/20
# 2630/2630 [==============================] - 0s - loss: 0.1643 - val_loss: 0.2054
# Epoch 3/20
# 2630/2630 [==============================] - 0s - loss: 0.1192 - val_loss: 0.1957
# Epoch 4/20
# 2630/2630 [==============================] - 0s - loss: 0.0880 - val_loss: 0.2025
# Epoch 5/20
# 2630/2630 [==============================] - 0s - loss: 0.0652 - val_loss: 0.1998
# Epoch 6/20
# 2630/2630 [==============================] - 0s - loss: 0.0498 - val_loss: 0.2042
# Epoch 7/20
# 2630/2630 [==============================] - 0s - loss: 0.0399 - val_loss: 0.1967
# Epoch 8/20
# 2630/2630 [==============================] - 0s - loss: 0.0336 - val_loss: 0.2018
# Epoch 9/20
# 2630/2630 [==============================] - 0s - loss: 0.0266 - val_loss: 0.2134
# Epoch 10/20
# 2630/2630 [==============================] - 0s - loss: 0.0217 - val_loss: 0.2146
# Epoch 11/20
# 2630/2630 [==============================] - 0s - loss: 0.0203 - val_loss: 0.2163
# Epoch 12/20
# 2630/2630 [==============================] - 0s - loss: 0.0163 - val_loss: 0.2268
# Epoch 13/20
# 2630/2630 [==============================] - 0s - loss: 0.0189 - val_loss: 0.2203
# Epoch 14/20
# 2630/2630 [==============================] - 0s - loss: 0.0146 - val_loss: 0.2252
# Epoch 15/20
# 2630/2630 [==============================] - 0s - loss: 0.0132 - val_loss: 0.2327
# Epoch 16/20
# 2630/2630 [==============================] - 0s - loss: 0.0129 - val_loss: 0.2316
# Epoch 17/20
# 2630/2630 [==============================] - 0s - loss: 0.0129 - val_loss: 0.2320
# Epoch 18/20
# 2630/2630 [==============================] - 0s - loss: 0.0127 - val_loss: 0.2359
# Epoch 19/20
# 2630/2630 [==============================] - 0s - loss: 0.0117 - val_loss: 0.2400
# Epoch 20/20
# 2630/2630 [==============================] - 0s - loss: 0.0126 - val_loss: 0.2407
# ____________________________________________________________________________________________________
# Layer (type)                     Output Shape          Param #     Connected to                     
# ====================================================================================================
# input_1 (InputLayer)             (None, 1000)          0                                            
# ____________________________________________________________________________________________________
# input_2 (InputLayer)             (None, 1000)          0                                            
# ____________________________________________________________________________________________________
# sequential_1 (Sequential)        (None, 128)           161152      input_1[0][0]                    
#                                                                    input_2[0][0]                    
# ____________________________________________________________________________________________________
# lambda_1 (Lambda)                (None, 1)             0           sequential_1[1][0]               
#                                                                    sequential_1[2][0]               
# ====================================================================================================
# Total params: 161,152
# Trainable params: 161,152
# Non-trainable params: 0
# ____________________________________________________________________________________________________
# None
# * Accuracy on training set: 99.47%
# * Accuracy on test set: 68.16%