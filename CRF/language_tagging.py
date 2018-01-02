import pycrfsuite
import argparse
import enchant
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import numpy as np 
import random

import nltk.data
from nltk.util import ngrams

classifier = nltk.data.load("classifiers/hindi_english_Maxent.pickle")


def get_freq_dict(filename):
	ll = []
	with open(filename) as f:
		lines = f.readlines()
		for line in lines:
			ll.append(line.strip().lower())	
	return ll
	
def get_idfdict(filename):
    freq_dict = defaultdict(int)
    with codecs.open(filename, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:  
        words = line.split()
        for word in words:
            freq_dict[word] += 1
    return freq_dict
    
def preprocess(filename, size=None):
    docs = []
    with open(filename, 'r') as f:
        sentences = f.read().split('\n\n')
    if size:
	print "Size passed"
	sentences = random.sample(sentences, size)
    for sentence in sentences:
        lines = sentence.split('\n')
        text = []
        for line in lines:
	    #print "Debug", line
            tokens = line.split()
            if len(tokens)!=2:
		continue
	    # Word Langid
	    #text.append((tokens[1], tokens[0]))
            text.append((tokens[0], tokens[1]))
        docs.append(text)
    return docs

def freq(word, lang_dict):
    
    if word in lang_dict:
		return True
    return False
    #For the time being just pass False
    '''
    if lang == 'en':
        if word in freq_en.keys():
            return True
        else:
            return False
    else:
        if word in freq_hi.keys():
            return True
        else:
            return False
    '''

def word2features(doc,i, hindi_dict, en_dict):
    #print "Extracting features", i
    word = doc[i][0]
    #postag = doc[i][1]
    #d = enchant.Dict("en_US")
    # Common features for all words
    ngs = [word] + list(ngrams(list(word), 2)) + list(ngrams(list(word), 3))
    feats = dict([(ng, True) for ng in ngs])
    features = [
        'bias',
        'word.lower=' + word.lower(),
        #'word[-3:]=' + word[-3:],
        #'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'word.startwithdigit=%s' %word[0].isdigit(),
        'word.length=' + str(len(word)),
	'word.alphabet=%s' % classifier.classify(feats),
        'word.ismostfreqEN=%s' % freq(word,en_dict),
        'word.ismostfreqHI=%s' % freq(word,hi_dict),
    ]

    # Features for words that are not
    # at the beginning of a document
    if i > 0:
        word1 = doc[i-1][0]
        #postag1 = doc[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.isdigit=%s' % word1.isdigit(),
        ])
    else:
        # Indicate that it is the 'beginning of a document'
        features.append('BOS')

    # Features for words that are not
    # at the end of a document
    if i < len(doc)-1:
        word1 = doc[i+1][0]
        #postag1 = doc[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.isdigit=%s' % word1.isdigit(),
        ])
    else:
        # Indicate that it is the 'end of a document'
        features.append('EOS')

    return features


def extract_features(doc, hi_dict, en_dict):
    print "Extracting fetaures"
    return [word2features(doc, i, hi_dict, en_dict) for i in range(len(doc))]

# A function fo generating the list of labels for each document
def get_labels(doc):
    return [label for (token, label) in doc]

def train(X_train, Y_train, hparams):
	trainer = pycrfsuite.Trainer(verbose=False)

	# Submit training data to the trainer
	for xseq, yseq in zip(X_train, Y_train):
    		trainer.append(xseq, yseq)

	# Set the parameters of the model
	trainer.set_params({
    	# coefficient for L1 penalty
    	'c1': 0.1,

    	# coefficient for L2 penalty
    	'c2': 0.01,  

    	# maximum number of iterations
    	'max_iterations': 200,

    	# whether to include transitions that
    	# are possible, but not observed
    	'feature.possible_transitions': True
	})
	# Provide a file name as a parameter to the train function, such that
	# the model will be saved to the file when training is finished
	trainer.train(hparams.model)
	

def parseargument():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vocab_file', type=str, default='vocab.txt',
                        help='file for the vocabulary')
    parser.add_argument('--data_file', type=str, default='data.txt',
                        help='file for data')
    parser.add_argument('--en_file', type=str, default='data.txt',
                        help='file for data')
    parser.add_argument('--hi_file', type=str, default='data.txt',
                        help='file for data')

    parser.add_argument('--train_file', type=str, default='train.txt',
                        help='file with training data/ synthetic data')
    parser.add_argument('--test_file', type=str, default='test.txt',
                        help='file for test data')
    parser.add_argument('--output_file', type=str, default='output.txt',
                        help='file for data')
    parser.add_argument('--size', type=int, default=None,
                        help='training size')

    parser.add_argument('--padding', type=bool, default=False,
                        help='padding with original file')
    parser.add_argument('--padding_file', type=str, default='original.txt',
                        help='training file for padding with the original test')


    parser.add_argument('--model', type=str, default='train.model',
                        help='name of the model to be stored')

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    hparams = parseargument()
    #data = preprocess(hparams.data_file)
    hi_dict = get_freq_dict(hparams.en_file) 
    en_dict = get_freq_dict(hparams.hi_file) 
    #print "Preprocess done", len(data[0])
    train_data = preprocess(hparams.train_file, hparams.size)
    #train_data = preprocess(hparams.train_file)
    if hparams.padding :
	train_data.extend(preprocess(hparams.padding_file, 200))
    
    X_train = [extract_features(doc, hi_dict, en_dict) for doc in train_data]
    Y_train = [get_labels(doc) for doc in train_data]
    
    #hparams.size = 500
    test_data = preprocess(hparams.test_file)
    X_test = [extract_features(doc,hi_dict, en_dict) for doc in test_data]
    Y_test = [get_labels(doc) for doc in test_data]
    ''' 
    X_train_new = X_train[:250]
    Y_train_new = Y_train[:250]

    X_test = X_train[250:]
    Y_test  = Y_train[250:]

    X_train = X_train_new
    Y_train = Y_train_new
    '''
    #print "Fetaure extraction done", len(X_test), len(Y_test), len(X_train), len(Y_train)
    '''
    kf = KFold(n_splits=2)
    i = 0
    for train_index, test_index in kf.split(X):
	#print train_index, test_index, X
    	X_train = [] 
	X_test= []
	Y_train= []
	Y_test = []
	for i in train_index:
		X_train.append(X[i])
		Y_train.append(Y[i])
	for i in test_index:
                X_test.append(X[i])
                Y_test.append(Y[i])
     '''
    #X[train_index], X[test_index], Y[train_index], Y[test_index]
    #train_test_split(X, Y, test_size=0.2)

    train(X_train, Y_train, hparams)
    tagger = pycrfsuite.Tagger()
    tagger.open(hparams.model) 
    y_pred = [tagger.tag(xseq) for xseq in X_test]

    labels = {"EN": 1, "HI": 0, "OTHER": 2, "NE": 3}
    # Let's take a look at a random sample in the testing set
    
    i = 12
    '''
    for i in range(len(X_test)):
     		for x, y in zip(y_pred[i], [x[1].split("=")[1] for x in X_test[i]]):
    			print("%s (%s)" % (y, x))
    '''
    # Convert the sequences of tags into a 1-dimensional array
    predictions = np.array([labels[tag] for row in y_pred for tag in row])
    truths = np.array([labels[tag] for row in Y_test for tag in row])
    	
    # Print out the classification report
    with open(hparams.output_file, 'a') as f:	
     	f.write(classification_report(
    	truths, predictions,
    	target_names=["HI", "EN", "OTHER", "NE"]))
