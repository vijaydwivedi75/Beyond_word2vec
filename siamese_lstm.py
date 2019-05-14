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
    '''Base LSTM network to be shared.
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

# sts_pairs = preprocess_data_lstm.getSemEval()
# print(sts_pairs.shape)
x_data, y_data, visualise_pairs = preprocess_data_lstm.dataset()
input_dim = 1000
epochs = 50

timesteps = int(input_dim/emb_dim)
input_dim = emb_dim

train_test_split = 400000                   # 263000
tr_pairs = x_data[:train_test_split]		# 263000
tr_y = y_data[:train_test_split]			
te_pairs = x_data[train_test_split:]		# 113000
te_y = y_data[train_test_split:]
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

# model = Model([input_a, input_b], distance)
model = load_model('checkpoints/model_siamese_lstm_0.766548672566.h5')

# # train
# rms = RMSprop()
# opt = Adamax()
# model.compile(loss=contrastive_loss, optimizer=opt)
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=384,
          epochs=epochs)
          # validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

print(model.summary())

# compute final accuracy on training and test sets
pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(pred, tr_y)
pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(pred, te_y)



print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

model.save('checkpoints/model_siamese_lstm_'+str(te_acc)+'.h5')

# sts_result = []
# pred = model.predict([sts_pairs[:, 0], sts_pairs[:, 1]])
# for item in pred:
#     for res in item:
#         sts_result.append(res*5.0)

# for res in sts_result:
#     print(res)

data_for_test = visualise_pairs

# Storing embeddings in a file
file_word = open('embeddings_word', 'a')
file_paraphrase = open('embeddings_paraphrase', 'a')
words = preprocess_data_lstm.getWords()
paraphrase = preprocess_data_lstm.get_pp()

ptr = 0
batch_sz = 1000
nm_batch = 300
for j in range(nm_batch):
    L, R = data_for_test[ptr:ptr+batch_sz, 0], data_for_test[ptr:ptr+batch_sz, 1]

    functor = K.function([input_a, input_b]+ [K.learning_phase()], [processed_a, processed_b])
    layer_out = functor([L, R, 0.])

    word_embd = layer_out[0]
    paraphrase_embd = layer_out[1]

    k = 0
    for i in range(ptr, ptr+batch_sz):
        file_word.write(words[i]+" -- "+' '.join(map(str, word_embd[k]))+'\n')
        file_paraphrase.write(paraphrase[i]+" -- "+' '.join(map(str, paraphrase_embd[k]))+'\n')
        k += 1

    ptr += batch_sz

file_word.close()
file_paraphrase.close()

############################################################################
#                            T E S T     P A R T                           #  
############################################################################
# paraphrase_embedding = tf.stack(layer_out[1])
# batch_word_embedding = tf.stack(layer_out[0][14450:14500])

# norm_paraphrase_embedding = tf.nn.l2_normalize(paraphrase_embedding, dim=1)
# norm_batch_word_embedding = tf.nn.l2_normalize(batch_word_embedding, dim=1)

# cosine_similarity = tf.matmul(norm_batch_word_embedding, tf.transpose(norm_paraphrase_embedding, [1, 0]))
# # closest_words = tf.argmax(cosine_similarity, 1)
# tf.InteractiveSession()
# sim = cosine_similarity.eval()

# words = preprocess_data_lstm.getWords()
# paraphrase = preprocess_data_lstm.get_pp()

# for i in range(50):
#     valid_word = words[i+14450]
#     # print(valid_word)
#     top_k = 8       # number of nearest neighbours
#     nearest = (-sim[i, :]).argsort()[1:top_k + 1]
    
#     log_str = "Nearest to '%s':" % valid_word
    
#     for k in range(top_k):
#         close_word = paraphrase[nearest[k]]
#         log_str = "%s %s," % (log_str, close_word)

#     print(log_str+"\n")

gc.collect()
