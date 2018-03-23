import sys
import operator
import numpy as np
from scipy.optimize import linprog
import math
from collections import defaultdict
def get_idfdict(filename):
        freq_dict = defaultdict(int)
        idf_dict = defaultdict(float)
        with open(filename) as f:
        #with codecs.open(filename, encoding="utf-8") as f:
                lines = f.readlines()
        count = 0.0
        for line in lines:
                count += 1
                words = line.replace(' \'', '\'').split()
                for word in set(words):
                        freq_dict[word] += 1
        for word in freq_dict.keys():
                idf_dict[word] = math.log(count / freq_dict[word])
		#idf_dict[word] = count / freq_dict[word]
        return idf_dict

def get_freqdict(filename):
        freq_dict = defaultdict(int)
        with open(filename) as f:
        #with codecs.open(filename, encoding="utf-8") as f:
                lines = f.readlines()
        for line in lines:
                words = line.replace(' \'', '\'').split()
                for word in words:
                        freq_dict[word] += 1
        return freq_dict

#freq_list = get_freqdict(sys.argv[1])
idf_dict = get_idfdict(sys.argv[1])
sorted_freq = sorted(idf_dict.items(), key=operator.itemgetter(1), reverse=True)
#print sorted_freq[:10]
#print sorted_freq[-10:]
top_el = sorted_freq[0]
bottom_el = sorted_freq[-1]
vara = 0.0
varb = 0.0

A_ub = np.zeros((20,2))
b_ub = np.zeros(20)
c = np.zeros(2)
top = sorted_freq[:10]
bottom = sorted_freq[-10:]

for i in range(10):
	A_ub[i][0] = top[i][1]
	A_ub[i][1] = -1
	b_ub[i] = 2

	A_ub[i+10][0] = -bottom[i][1]
        A_ub[i+10][1] = 1
	b_ub[i] = 2

res = linprog(c, A_ub, b_ub, bounds=[(0,None),(0,None)])
print res.x
x = res.x[0] * (top_el[1]) - res.x[1]
print 1 / (1 + math.exp(-x))

#x = res.x[0] * (1.0/bottom_el[1]) - res.x[1]
x = res.x[0] * 0.001 - res.x[1]

print 1 / (1 + math.exp(-x))

x = (top_el[1])
print "IDF top", x
print 1 / (1 + math.exp(-x))

#x = 0.001
x = (bottom_el[1])
print "IDF low", x
print 1 / (1 + math.exp(-x))


print 0.5 + (res.x[0] * (1.0/top[0][1])- res.x[1])/4
print 0.5 + (res.x[0] * (1.0/bottom_el[1])- res.x[1])/4

#res.x

