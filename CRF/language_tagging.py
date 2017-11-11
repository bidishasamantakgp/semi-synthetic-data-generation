import pycrfsuite
import argparse
import enchant
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import numpy as np 

def get_idfdict(filename):
    freq_dict = defaultdict(int)
    with codecs.open(filename, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:  
        words = line.split()
        for word in words:
            freq_dict[word] += 1
    return freq_dict
    
def preprocess(filename):
    docs = []
    with open(filename, 'r') as f:
        sentences = f.read().split('\n\n')

    for sentence in sentences:
        lines = sentence.split('\n')
        text = []
        for line in lines:
	    #print "Debug", line
            tokens = line.split()
            if len(tokens)!=2:
		continue
	    # Word Langid
	    text.append((tokens[1], tokens[0]))
            #text.append((tokens[0], tokens[1]))
        docs.append(text)
    return docs


def freq(word, lang):
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

def word2features(doc,i):
    #print "Extracting features", i
    word = doc[i][0]
    #postag = doc[i][1]
    d = enchant.Dict("en_US")
    # Common features for all words
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'word.startwithdigit=%s' %word[0].isdigit(),
        'word.length=' + str(len(word)),
        'word.alphabet=%s' % d.check(word),
        'word.ismostfreqEN=%s' % freq(word,'en'),
        'word.ismostfreqHI=%s' % freq(word,'hi'),
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


def extract_features(doc):
    #print "Extracting fetaures"
    return [word2features(doc, i) for i in range(len(doc))]

# A function fo generating the list of labels for each document
def get_labels(doc):
    return [label for (token, label) in doc]

def train(X_train, Y_train):
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
	trainer.train('crf.model')
	

def parseargument():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vocab_file', type=str, default='vocab.txt',
                        help='file for the vocabulary')
    parser.add_argument('--data_file', type=str, default='data.txt',
                        help='file for data')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    hparams = parseargument()
    data = preprocess(hparams.data_file)
    #print "Preprocess done", len(data[0])
    X = [extract_features(doc) for doc in data]
    Y = [get_labels(doc) for doc in data]
    #print "Fetaure extraction done"
    kf = KFold(n_splits=5)
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
	#X[train_index], X[test_index], Y[train_index], Y[test_index]
	#train_test_split(X, Y, test_size=0.2)

    	train(X_train, Y_train)
    	tagger = pycrfsuite.Tagger()
    	tagger.open('crf.model') 
    	y_pred = [tagger.tag(xseq) for xseq in X_test]

    	# Let's take a look at a random sample in the testing set
    	'''
	i = 12
    	for i in range(len(X_test)):
     		for x, y in zip(y_pred[i], [x[1].split("=")[1] for x in X_test[i]]):
    			print("%s (%s)" % (y, x))
    	'''
	# Create a mapping of labels to indices
    	labels = {"EN": 1, "HI": 0, "OTHER": 2, "NE": 3}

    	# Convert the sequences of tags into a 1-dimensional array
    	predictions = np.array([labels[tag] for row in y_pred for tag in row])
    	truths = np.array([labels[tag] for row in Y_test for tag in row])
	i = i + 1
	print "i=", i
    	# Print out the classification report
    	print(classification_report(
    	truths, predictions,
    	target_names=["HI", "EN", "OTHER", "NE"]))
