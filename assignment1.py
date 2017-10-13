import sys
import nltk
import re
import math
import numpy
from nltk.corpus import gutenberg
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import shelve
flag = 'r'
count = 100 #number of documents to read, change this to -1 to read all
inp = sys.argv # arguments(running with -n argument will refresh database)

#*******************INDEXING******************

def get_tokens_with_indexes(tokens):
	wordIndexesInFile = {}
	for index, token in enumerate(tokens):
		if token in wordIndexesInFile.keys():
			wordIndexesInFile[token].append(index)
		else:
			wordIndexesInFile[token] = [index]
	return wordIndexesInFile


def get_magnitude(vec):
	return pow(sum(map(lambda x: x**2, vec)),.5)

def magnitudes(documents, vectors):
	mags = {}
	for document in documents:
		mags[document] = get_magnitude(vectors[document])
	return mags

def term_frequency(tf, mags, term, document):
	return tf[document][term]/mags[document] if term in tf[document].keys() else 0

def idf_func(Nd, Nt):
	if Nt != 0:
		return math.log(Nd/Nt)
	else:
	 	return 0

def save(m, f):
	with shelve.open(f, 'c') as f:
		for x in m.keys():
			f[x] = m[x]
	f.sync()

def make_index(count):
	print('indexing...')
	index = {}
	tf = {}
	df = {}
	mags = {}
	vectors = {}
	idf = {}
	stopw = stopwords.words('english')
	stemmer = SnowballStemmer("english")
	pattern = re.compile('[\W_]+')
	documents = reuters.fileids()[:count]
	for document_id in documents:
		tf[document_id] = {}
		tokens = reuters.raw(document_id).lower();
		tokens = pattern.sub(' ', tokens)
		tokens = tokens.split()
		tokens = [x for x in tokens if x not in stopw]
		tokens = [stemmer.stem(t) for t in tokens]
		# tokens = nltk.word_tokenize(reuters.raw(document_id).lower())
		tokens_indexed = get_tokens_with_indexes(tokens)
		for term in tokens_indexed.keys():
			if term not in index.keys():
				index[term] = {}
			l = index[term]
			l[document_id] = (tokens_indexed[term])
			index[term] = l
			length = tf[document_id]
			length[term] = len(l[document_id])
			tf[document_id] = (length)
			# if term in df.keys():
			# 	df[term] += 1
			# else:
			# 	df[term] = 1 
		vectors[document_id] = [len(tokens_indexed[word]) for word in tokens_indexed.keys()]
	for x in index.keys():
		df[x] = len(index[x])
	mags = magnitudes(documents, vectors)
	for document in documents:
		for term in index.keys():
			tf[document][term] = tf[document][term]/mags[document] if term in tf[document].keys() else 0
			if term in df.keys():
				idf[term] = idf_func(len(documents), df[term]) 
			else:
				idf[term] = 0
	save(index, 'index')
	save(tf, 'tf')
	save(df, 'df')
	save(vectors, 'vectors')
	save(mags, 'mags')
	save(idf, 'idf')
	print('finished')
	return tf, df, mags, vectors, idf, index
		
#*******************SEARCHING******************

def search(query, index, tf, idf):
	return search_phrase(query, index, tf, idf)

def get_ranked_results(resultDocs, query, index, tf, idf):
	vectors = make_vectors(resultDocs, index, tf, idf)
	qVector = get_query_of_vector(query, idf, index)
	results = [[numpy.dot(vectors[result], qVector), result] for result in resultDocs]
	results.sort(key=lambda x: x[0])
	results = [x[1] for x in results]
	return results

def make_vectors(documents, index, tf, idf):
	vecs = {}
	for doc in documents:
		docVec = []
		for term in index.keys():
			docVec.append(tf[doc][term] * idf[term])
		vecs[doc] = docVec
	return vecs

def get_query_of_vector(query, idf, index):
	queryls = query.split()
	qVector = []
	for word in queryls:
		qVector.append(count_term_in_text(word, query))
	query_idf = [idf[word] for word in index.keys()]
	magnitude = get_magnitude(qVector)
	freq = termfreq(index.keys(), query)
	tf = [x/magnitude for x in freq]
	final = [tf[i]*query_idf[i] for i in range(len(index.keys()))]
	return final

def termfreq(terms, query):
	temp = []
	for term in terms:
		temp.append(count_term_in_text(term, query))
	return temp

def count_term_in_text(term, query):
	count = 0
	for word in query.split():
		if word == term:
			count += 1
	return count

def search_phrase(query, index, tf, idf):
	stopw = stopwords.words('english')
	stemmer = SnowballStemmer("english")
	pattern = re.compile('[\W_]+')
	query = pattern.sub(' ',query)
	query = query.split()
	query = [x for x in query if x not in stopw]
	query = [stemmer.stem(t) for t in query]
	query = " ".join(query)
	if len(query) < 1:
		return 'Error: Not informative query!'
	listOfLists, result = [],[]

	for word in query.split():
		listOfLists.append(one_word_query(word, index, tf, idf))
	setted = set(listOfLists[0]).intersection(*listOfLists)
	for filename in setted:
		temp = []
		for word in query.split():
			temp.append(index[word][filename])
		for i in range(len(temp)):
			for ind in range(len(temp[i])):
				temp[i][ind] -= i
		if set(temp[0]).intersection(*temp):
			result.append(filename)
	return get_ranked_results(result, query, index, tf, idf)

def one_word_query(word, index, tf, idf):
	if word in index.keys():
		return get_ranked_results([filename for filename in index[word].keys()], word, index, tf, idf)
	else:
		return []

def read_from_shelve(m, f):
	with shelve.open(f, 'c') as f:
		for x in f.keys():
			m[x] = f[x]

if __name__ == '__main__':
	tf = {}
	df = {}
	mags = {}
	vectors = {}
	idf = {}
	index = {}
	read_from_shelve(tf, 'tf')
	read_from_shelve(df, 'df')
	read_from_shelve(mags, 'mags')
	read_from_shelve(vectors, 'vectors')
	read_from_shelve(idf, 'idf')
	read_from_shelve(index, 'index')
	new = False
	if len(inp) > 1:
		new = inp[1] == '-n'
	if new or not tf or not df or not mags or not vectors or not idf or not index:
		tf, df, mags, vectors, idf, index = make_index(count)
	print('There are', count, 'files in database')
	query = input('Enter your query: ')
	print(search(query, index, tf, idf))