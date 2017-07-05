import os
import numpy as np
import tensorflow as tf
from random import randint

# from siamese_lstm import
# from preprocess_data_lstm import 

# -------------------------------------------------------------------- #
# Preparing Beyond_Word2Vec embeddings                                 #
# -------------------------------------------------------------------- #
print('Indexing Beyond_Word2Vec embeddings...')

word_embd = {}
para_embd = {}
f_word = open(os.path.join('./', 'embeddings_word'))
f_para = open(os.path.join('./', 'embeddings_paraphrase'))

for line in f_word:
	word_n_embd = line.split('--')
	word = word_n_embd[0][:-1]
	embds = word_n_embd[1][1:-1].split(' ')
	coefs = np.asarray(embds, dtype='float32')
	word_embd[word] = coefs
f_word.close()

for line in f_para:
	para_n_embd = line.split('--')
	para = para_n_embd[0][:-1]
	embds = para_n_embd[1][1:-1].split(' ')
	coefs = np.asarray(embds, dtype='float32')
	para_embd[para] = coefs
f_para.close()

print('Found %s Beyond_Word2Vec word vectors.' % len(word_embd))
print('Found %s Beyond_Word2Vec paraphrase vectors.' % len(para_embd))
# -------------------------------------------------------------------- #

words = []
paraphrase = []
word_embeddin = []
paraphrase_embeddin = []
id_word = {}
id_para = {}
word_id = {}
para_id = {}

cnt = 0
for key, value in word_embd.items():
	words.append(key)
	word_embeddin.append(value)
	id_word[cnt] = key
	word_id[key] = cnt
	cnt += 1

cnt = 0
for key, value in para_embd.items():
	paraphrase.append(key)
	paraphrase_embeddin.append(value)
	id_para[cnt] = key
	para_id[key] = cnt
	cnt += 1

# EVAL 1: NEARBY  WORDS
def eval_nearby(start, end):	
	batch_word_embedding = tf.stack(word_embeddin[start:end])
	paraphrase_embedding = tf.stack(paraphrase_embeddin)

	norm_paraphrase_embedding = tf.nn.l2_normalize(paraphrase_embedding, dim=1)
	norm_batch_word_embedding = tf.nn.l2_normalize(batch_word_embedding, dim=1)

	cosine_similarity = tf.matmul(norm_batch_word_embedding, tf.transpose(norm_paraphrase_embedding, [1, 0]))
	# closest_words = tf.argmax(cosine_similarity, 1)
	tf.InteractiveSession()
	sim = cosine_similarity.eval()

	for i in range(end-start):
	    valid_word = words[i+start]
	    top_k = 5       # number of nearest neighbours
	    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
	    
	    log_str = "Nearest to '%s':" % valid_word
	    
	    for k in range(top_k):
	        close_word = paraphrase[nearest[k]]
	        log_str = "%s %s," % (log_str, close_word)

	    print(log_str+"\n")

# EVAL 2: WORD  ANALOGIES
def eval_analogies(a_word, b_word, c_word, num_analogies):
	word_embedding = tf.stack(word_embeddin)
	norm_word_embedding = tf.nn.l2_normalize(word_embedding, dim=1)

	analogy_a = tf.stack(a_word)
	analogy_b = tf.stack(b_word)
	analogy_c = tf.stack(c_word)

	a_emb = tf.gather(norm_word_embedding, analogy_a)
	b_emb = tf.gather(norm_word_embedding, analogy_b)
	c_emb = tf.gather(norm_word_embedding, analogy_c)

	target = c_emb + (b_emb - a_emb)

	dist = tf.matmul(target, norm_word_embedding, transpose_b=True)
	tf.InteractiveSession()
	_, pred_idx = tf.nn.top_k(dist.eval(), 4)

	for i in range(num_analogies):
		print("* " + str(id_word[a_word[i]]) + " : " + str(id_word[b_word[i]]) + "  =>  "
			+ str(id_word[c_word[i]]) + " :::: " + str(id_word[pred_idx[i][0]]) + str(id_word[pred_idx[i][1]])
			+ str(id_word[pred_idx[i][2]]) + str(id_word[pred_idx[i][3]]))

def main():
	# 1. EVALUATION FOR NEARBY WORDS
	# eval_nearby(1540, 1565)

	# 2. EVALUATION FOR WORD ANALOGIES
	num_analogies = 15
	analogy_a = []
	analogy_b = []
	analogy_c = []
	for i in range(num_analogies):
		analogy_a.append(randint(0, len(word_embd)))
		analogy_b.append(randint(0, len(word_embd)))
		analogy_c.append(randint(0, len(word_embd)))

	eval_analogies(analogy_a, analogy_b, analogy_c, num_analogies)

main()