# ---------------------------------------------------------	#
# Following commented code is to prepare 'data_pair_neg 	#
#---------------------------------------------------------- #

# lines = [line.strip('\n') for line in open('data_pair_pos')]

# left_list = []
# right_list = []

# for line in lines:
# 	left_right = line.split(' -- ')
# 	left_list.append(left_right[0][1:])
# 	right_list.append(left_right[1][:-1])
	
# right_list.reverse()
# zipped_data_pair_neg = zip(left_list, right_list)

# data_pair_neg = list(zipped_data_pair_neg)

# print(data_pair_neg[1512])

# with open('data_pair_neg', 'w+') as f:
# 	for a, b in data_pair_neg:
# 		f.write(' '+a+' -- '+b+' \n')

# --------------------------------------------------------- #
import collections
import numpy as np
import os

lines = [line.strip('\n') for line in open('data_pair_pos')]

left_list = []
right_list = []

for line in lines:
	left_right = line.split(' -- ')
	left_list.append(left_right[0][1:])
	right_list.append(left_right[1][:-1])

wordlist = []

for item in left_list:
	tempList = item.split()

	for word in tempList:
		wordlist.append(word)

for item in right_list:
	tempList = item.split()

	for word in tempList:
		wordlist.append(word)

uniqueWords = collections.Counter(wordlist)

# print("Total unique words: ", len(uniqueWords))
# --------------------------------------------------- #
# Preparing GloVe embeddings                          #
# --------------------------------------------------- #
print('Indexing word vectors.')

EMBEDDING_DIM = 200
embeddings_index = {}
f = open(os.path.join('glove.6B/', 'glove.6B.200d.txt'))
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

# print(embeddings_index['the'])
print('Found %s word vectors.' % len(embeddings_index))
# --------------------------------------------------- #

vocabulary_size = 32137

def vocab_size():
	return vocabulary_size

def embed_dim():
	return EMBEDDING_DIM

words = wordlist

def build_dictionary(words, vocabulary_size):
	count = [['UNK', -1]]
	count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
	dictionary = dict()

	for word, _ in count:
		dictionary[word] = len(dictionary)	# here ranking (index) is done... Eg. dictionary['the'] = 1

	data = list()
	unk_count = 0

	for word in words:
		if word in dictionary:
			index = dictionary[word]
		else:
			index = 0	# dictionary['UNK']
			unk_count += 1
		data.append(index)

	count[0][1] = unk_count
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

	return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dictionary(words, vocabulary_size)
# print(reverse_dictionary)

def word_to_dict(word):
	try:
		return dictionary[word]
	except KeyError:
		return 0

def word_to_vec(word_map):
	if word_map == 0:
		return np.zeros(EMBEDDING_DIM)
	try:
		return embeddings_index[reverse_dictionary[word_map]]
	except KeyError:
		# return np.zeros(200)
		return np.random.uniform(-1, 1, size=EMBEDDING_DIM)

pos = []
neg = []

temp = []
right_neg_list = []
temp = list(right_list)
temp.reverse()

right_neg_list = temp

train_pairs = []
visualise_pairs = []

num_data = 18800		
# data split 
# 1462803 x 2 = 2925606 total samples
# Test: 25% = 731400
# Train: 75% = 2194206
for i in range(num_data):
	train_item = []
	train_item_inside = []
	train_item_inside_vectored = []
	for word in left_list[i].split(' '):
		train_item_inside.append(word_to_dict(word))
	while len(train_item_inside) < 5:
		train_item_inside.append(0)
	if(len(train_item_inside) >= 5):
		train_item_inside = list(train_item_inside[:5])
	for word_map in train_item_inside:
		train_item_inside_vectored.append(list(word_to_vec(word_map)))
	train_item.append(train_item_inside_vectored)

	train_item_inside = []
	train_item_inside_vectored = []
	for word in right_list[i].split(' '):
		train_item_inside.append(word_to_dict(word))
	while len(train_item_inside) < 5:
		train_item_inside.append(0)
	if(len(train_item_inside) >= 5):
		train_item_inside = list(train_item_inside[:5])
	for word_map in train_item_inside:
		train_item_inside_vectored.append(list(word_to_vec(word_map)))
	train_item.append(train_item_inside_vectored)
	train_pairs.append(train_item)

	train_item = []
	train_item_inside = []
	train_item_inside_vectored = []
	for word in left_list[i].split(' '):
		train_item_inside.append(word_to_dict(word))
	while len(train_item_inside) < 5:
		train_item_inside.append(0)
	if(len(train_item_inside) >= 5):
		train_item_inside = list(train_item_inside[:5])
	for word_map in train_item_inside:
		train_item_inside_vectored.append(list(word_to_vec(word_map)))
	train_item.append(train_item_inside_vectored)

	train_item_inside = []
	train_item_inside_vectored = []
	for word in right_neg_list[i].split(' '):
		train_item_inside.append(word_to_dict(word))
	while len(train_item_inside) < 5:
		train_item_inside.append(0)
	if(len(train_item_inside) >= 5):
		train_item_inside = list(train_item_inside[:5])
	for word_map in train_item_inside:
		train_item_inside_vectored.append(list(word_to_vec(word_map)))
	train_item.append(train_item_inside_vectored)
	train_pairs.append(train_item)

	# for preparing data used for visualising embeddings
	data_item = []
	data_item_inside = []
	data_item_inside_vectored = []
	for word in left_list[i].split(' '):
		data_item_inside.append(word_to_dict(word))
	while len(data_item_inside) < 5:
		data_item_inside.append(0)
	if(len(data_item_inside) >= 5):
		data_item_inside = list(data_item_inside[:5])
	for word_map in data_item_inside:
		data_item_inside_vectored.append(list(word_to_vec(word_map)))
	data_item.append(data_item_inside_vectored)

	data_item_inside = []
	data_item_inside_vectored = []
	for word in right_list[i].split(' '):
		data_item_inside.append(word_to_dict(word))
	while len(data_item_inside) < 5:
		data_item_inside.append(0)
	if(len(data_item_inside) >= 5):
		data_item_inside = list(data_item_inside[:5])
	for word_map in data_item_inside:
		data_item_inside_vectored.append(list(word_to_vec(word_map)))
	data_item.append(data_item_inside_vectored)
	visualise_pairs.append(data_item)

train_y = []
for i in range(num_data):
	train_y.append(0)
	train_y.append(1)

# Utility code to check max and count of words in a phrase
# max = 0
# cnt = 0
# for i in range(num_data):
# 	if(len(train_pairs[i][1]) > 5):
# 		cnt+=1
# 	# print(train_pairs[i])
# 	if(len(train_pairs[i][1]) > max):
# 		max = len(item)
# # print(max)
# # print(cnt)

# print(len(train_pairs[5][1]))

def dataset():
	return np.array(train_pairs), np.array(train_y), np.array(visualise_pairs)
# train_pairs, train_y, visualise_pairs = dataset()

def getWords():
	return left_list

def get_pp():
	return right_list