import urllib2
import urllib
import sys
import json
import ast
import re
import time
import argparse
from transliterate_util import *
from collections import defaultdict
import numpy as np

def parseargument():
 	parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 	parser.add_argument('--synthetic_file', type=str, default='attention.csv',
                        help='synthetic_file which stores the synthetic sentences')
 	parser.add_argument('--align_file', type=str, default='forward.align',
                        help='file which stores the alignment')
 	parser.add_argument('--pos_file', type=str, default='POS.csv',
                        help='stores the POS tagged file')
 	parser.add_argument('--out_file', type=str, default='out',
                        help='file to be written')
 	parser.add_argument('--source_lang', type=str, default='en',
                        help='source language hi, en')
 	parser.add_argument('--target_lang', type=str, default='hi',
                        help='target language to be converted en, hi')
	args = parser.parse_args()
	return args

# the format is <linenumber>, <hindi>, <english>, <codemixed>, <scroes>* , <ens>, <ene>, <his>, <hie>
def getSynthetic(filename):
	indexlist = []
	englishpos = defaultdict(list)
	hindipos = defaultdict(list)
	codemixed = defaultdict(list)

	with codecs.open(filename, encoding='utf-8') as f:
		for line in f:
			tokens = line.split("\t")
			#print "tokens", tokens
			index = int(tokens[0])
			indexlist.append(index)
			hindipos[index].append([int(tokens[-2]), int(tokens[-1])])
			englishpos[index].append([int(tokens[-4]), int(tokens[-3])])
			codemixed[index].append(tokens[3].strip().split())
			#hindi[index].append(tokens[1].split())
			
	
	return (indexlist, englishpos, hindipos, codemixed)

def getAlignDict(filename, linenumberlist):
	f = codecs.open(filename, encoding='utf-8')
	lines = f.readlines()
	align_dict = defaultdict(defaultdict)

	for i in linenumberlist:
		line = lines[i]
		tokens = line.split()
		temp_dict = defaultdict(int)
		for token in tokens:
			[s,t] = token.split("-")
			temp_dict[int(t)] = int(s)
		align_dict[i] = temp_dict
	return align_dict
	 
def getPosDict(filename, linenumberlist):
	f = codecs.open(filename, encoding='utf-8')
	lines = f.read().split("\n\n")
	pos_dict = defaultdict(defaultdict)
	for i in linenumberlist:
                line = lines[i]
                tokens = line.split("\n")
                temp_dict = defaultdict()
                for j in range(len(tokens)):
                        [word,pos] = tokens[j].split("\t")
                        temp_dict[j] = pos
                pos_dict[i] = temp_dict
	return pos_dict

def alignPOS(indexlist, codemixed, pos_dict, align_dict, englishpos, hindipos, outfile):

	for i in indexlist:
	   pos_sent = pos_dict[i]
	   align_sent = align_dict[i]
	   englishposlist = englishpos[i]
	   hindiposlist = hindipos[i]
	   codemixedlist = codemixed[i]

	   for ([e_st, e_end],[hi_st, hi_end], codemixedsent) in zip(englishposlist, hindiposlist, codemixedlist): 
		#[e_st, e_end] = englishpos[i]
		#[hi_st, hi_end] = hindipos[i]

		#codemixedsent = codemixed[i]
		#l = len(codemixedsent)
		l = e_st + 1 + e_end + hi_end - hi_st + 1
		k = hi_st
		
		#print codemixedsent, e_st, hi_st, hi_end
		values = np.array(codemixedsent[e_st+1: e_st +1 +  (hi_end - hi_st +1)])
		searchval = ','
		ii = np.where(values == searchval)[0]
		hindi_segment = ' '.join(word for word in codemixedsent[e_st+1: e_st +1 +  (hi_end - hi_st +1)])
	
		#hindi_segment = ' '.join(word for word in hindisent[hi_st: hi_end + 1])
		#print "hindi_segment", hindi_segment
		hparams.data = hindi_segment
		try:	
			transliterated = transliterate(hparams)[0].split()
			for ci in ii:
				transliterated.insert(ci, ',')
		except:
			#print "Error"
			transliterated = hindi_segment.split()
		
		#print "transliterated", transliterated
		#hi_seg_len = hi_end - hi_st
			
		for j in range(len(codemixedsent)):
			print "Debug j", j, l, e_end, hi_st, hi_end
			if j <= e_st:
				with codecs.open(outfile, 'a', encoding='utf-8') as fw:
					fw.write(codemixedsent[j] + '\t' + 'EN' + '\t' + pos_sent[j]+'\n')
			elif j >= (l - e_end):
				with codecs.open(outfile, 'a', encoding='utf-8') as fw:
					idx = len(pos_sent) - e_end + (j - l + e_end)
					fw.write(codemixedsent[j] + '\t' + 'EN' + '\t' + pos_sent[idx]+'\n')
			else:
				with codecs.open(outfile, 'a', encoding='utf-8') as fw:
                                        #fw.write(codemixed[j] + '\t' + 'HI' + '\t' + pos_sent[align_sent[k]])
                                        try:
					 fw.write(transliterated[k - hi_st] + '\t' + 'HI' + '\t' + pos_sent[align_sent[k]]+'\n')
					except:
					 continue
				k+=1
			'''
			if j <= e_st or j >= (l - e_end):
				with codecs.open(outfile, 'a', encoding='utf-8') as fw:
					if j >= (l - e_end):
						idx = len(pos_sent) - e_end + (j - l + e_end) 	
					else:
						idx = j
					print "Debug idx", idx
					fw.write(codemixedsent[j] + '\t' + 'EN' + '\t' + pos_sent[idx]+'\n')
					#fw.write(codemixedsent[j] + '\t' + 'EN' + '\t' + pos_sent[j]+'\n') 
			else:
				print "Debug index", align_sent, transliterated, k, k - hi_st
				with codecs.open(outfile, 'a', encoding='utf-8') as fw:
					#fw.write(codemixed[j] + '\t' + 'HI' + '\t' + pos_sent[align_sent[k]])
					fw.write(transliterated[k - hi_st] + '\t' + 'HI' + '\t' + pos_sent[align_sent[k]]+'\n')
				#print codemixed[j], pos_sent[align_sent[k]]
				k += 1
			'''
 		with codecs.open(outfile, 'a', encoding='utf-8') as fw:
			fw.write("\n")

if __name__=="__main__":

	hparams = parseargument()
	indexlist, englishpos, hindipos, codemixed  = getSynthetic(hparams.synthetic_file)
	indexlist = list(set(indexlist))
	align_dict = getAlignDict(hparams.align_file, indexlist)
	pos_dict = getPosDict(hparams.pos_file, indexlist)
        
	#print "Debug calculations"
	#print "indexlist", indexlist
	#print "englishpos", englishpos[1]
	#print "hindipos", hindipos
	#print "codemix", codemixed		
	#print "aligndict", align_dict
	#print "pos_dict", pos_dict


	alignPOS(indexlist, codemixed, pos_dict, align_dict, englishpos, hindipos, hparams.out_file)
	#Assume that we have tagged english POS tagged

