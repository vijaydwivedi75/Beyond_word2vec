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
# f_all_words = open(os.path.join('./', 'embeddings_all_words'))

# exception_counter = 0

# for line in f_all_words:
# 	word_n_embd = line.split('--')
# 	word = word_n_embd[0][:-1]
# 	embds = word_n_embd[1][1:-1].split(' ')
# 	try:
# 		coefs = np.asarray(embds, dtype='float32')
# 	except ValueError:
# 		exception_counter += 1
# 		continue
# 	word_embd[word] = coefs
# print("ValueError: ", exception_counter)
# f_all_words.close()

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
print('Found %s Beyond_Word2Vec phrase vectors.' % len(para_embd))
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
def eval_nearby_words(start, end):	
	batch_word_embedding = np.stack(word_embeddin[start:end])
	word_embedding = np.stack(word_embeddin)

	norm_word_embedding = tf.nn.l2_normalize(word_embedding, dim=1)
	norm_batch_word_embedding = tf.nn.l2_normalize(batch_word_embedding, dim=1)

	cosine_similarity = tf.matmul(norm_batch_word_embedding, tf.transpose(norm_word_embedding, [1, 0]))
	# closest_words = tf.argmax(cosine_similarity, 1)
	tf.InteractiveSession()
	sim = cosine_similarity.eval()

	for i in range(end-start):
	    valid_word = words[i+start]
	    top_k = 5       # number of nearest neighbours
	    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
	    
	    log_str = "Nearest to '%s':" % valid_word
	    
	    for k in range(top_k):
	        close_word = words[nearest[k]]
	        log_str = "%s %s," % (log_str, close_word)

	    print(log_str+"\n")

# EVAL 2: WORD  ANALOGIES
# tf.InteractiveSession()
norm_word_embedding = np.stack(word_embeddin)
# nor_word_embedding = tf.nn.l2_normalize(word_embedding, dim=1)
# norm_word_embedding = nor_word_embedding.eval()
# norm_word_embedding = word_embedding
def eval_analogies(a_word, b_word, c_word, num_analogies):
	# norm_word_embedding = tf.stack(word_embeddin)
	# norm_word_embedding = tf.nn.l2_normalize(word_embedding, dim=1)
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
	pred_ids = pred_idx.eval()

	f_MSR_out = open('MSR_output', 'a')
	# print(pred_ids)
	for i in range(num_analogies):
		print("* IF '" + str(id_word[a_word[i]]) + "' is to '" + str(id_word[b_word[i]]) + "'  THEN  '"
			+ str(id_word[c_word[i]]) + "' :: " + str(id_word[pred_ids[i][0]]) + ", " + str(id_word[pred_ids[i][1]])
			+ ", " + str(id_word[pred_ids[i][2]]) + ", " + str(id_word[pred_ids[i][3]]))
		f_MSR_out.write(str(id_word[pred_ids[i][0]]) + " " + str(id_word[pred_ids[i][1]])
			+ " " + str(id_word[pred_ids[i][2]]) + " " + str(id_word[pred_ids[i][3]])+'\n')

	f_MSR_out.close()

# EVAL 3: NEARBY PHRASES
def eval_nearby_phrases(start, end):
	batch_word_embedding = np.stack(word_embeddin[start:end])
	paraphrase_embedding = np.stack(paraphrase_embeddin)

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

# EVAL 4: NEARBY PHRASES FOR PHRASE
def eval_nearby_phrases_for_phrase(start, end):
	batch_paraphrase_embedding = np.stack(paraphrase_embeddin[start:end])
	paraphrase_embedding = np.stack(paraphrase_embeddin)

	norm_paraphrase_embedding = tf.nn.l2_normalize(paraphrase_embedding, dim=1)
	norm_batch_paraphrase_embedding = tf.nn.l2_normalize(batch_paraphrase_embedding, dim=1)

	cosine_similarity = tf.matmul(norm_batch_paraphrase_embedding, tf.transpose(norm_paraphrase_embedding, [1, 0]))
	# closest_words = tf.argmax(cosine_similarity, 1)
	tf.InteractiveSession()
	sim = cosine_similarity.eval()

	for i in range(end-start):
	    valid_phrase = paraphrase[i+start]
	    top_k = 5       # number of nearest neighbours
	    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
	    
	    log_str = "Nearest to '%s':" % valid_phrase
	    
	    for k in range(top_k):
	        close_phrase = paraphrase[nearest[k]]
	        log_str = "%s %s," % (log_str, close_phrase)

	    print(log_str+"\n")

def main():
	# 1. EVALUATION FOR NEARBY WORDS
	# eval_nearby_words(14540, 14640)


	# 2. EVALUATION FOR WORD ANALOGIES
	# analogy_a = []
	# analogy_b = []
	# analogy_c = []

	# f_MSR = open('./test_set/word_relationship.questions', 'r')
	# for line in f_MSR:
	# 	words = line.split(' ')
	# 	analogy_a.append(word_id[words[0]])
	# 	analogy_b.append(word_id[words[1]])
	# 	analogy_c.append(word_id[words[2][:-1]])
	# f_MSR.close()

	# num_analogies = 500
	# ptr = 6500
	# num_loop = int(len(analogy_a)/num_analogies)
	# for i in range(num_loop-1):
	# 	eval_analogies(analogy_a[ptr:ptr+num_analogies], 
	# 		analogy_b[ptr:ptr+num_analogies], analogy_c[ptr:ptr+num_analogies], num_analogies)
	# 	ptr += num_analogies



	# 3. EVALUATION FOR NEARBY PHRASES
	# eval_nearby_phrases(14540, 14640)



	# 4. EVALUATION FOR NEARBY PHRASES FOR PHRASE
	eval_nearby_phrases_for_phrase(97500, 98000)

main()