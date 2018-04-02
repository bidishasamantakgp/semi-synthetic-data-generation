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
	englishpos = defaultdict()
	hindipos = defaultdict()
	codemixed = defaultdict()

	with codecs.open(filename, encoding='utf-8') as f:
		for line in f:
			tokens = line.split("\t")
			print "tokens", tokens
			index = int(tokens[0])
			indexlist.append(index)
			hindipos[index] = [int(tokens[-2]), int(tokens[-1])]
			englishpos[index] = [int(tokens[-4]), int(tokens[-3])]
			codemixed[index] = tokens[3].split()
	
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
	f = open(filename)
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

		[e_st, e_end] = englishpos[i]
		[hi_st, hi_end] = hindipos[i]

		codemixedsent = codemixed[i]
		l = len(codemixedsent)
		k = hi_st
		
		#print codemixedsent
		hindi_segment = ' '.join(word for word in codemixedsent[e_st+1: e_st + 1 + (hi_end - hi_st + 1)])
		#print "hindi_segment", hindi_segment
		hparams.data = hindi_segment
		transliterated = transliterate(hparams)[0].split()
		#print "transliterated", transliterated
		#hi_seg_len = hi_end - hi_st

		for j in range(len(codemixedsent)):
			if j <= e_st or j >= (l - e_end):
				with codecs.open(outfile, 'a', encoding='utf-8') as fw:
					fw.write(codemixedsent[j] + '\t' + 'EN' + '\t' + pos_sent[j]+'\n') 
			else:
				with codecs.open(outfile, 'a', encoding='utf-8') as fw:
					#fw.write(codemixed[j] + '\t' + 'HI' + '\t' + pos_sent[align_sent[k]])
					fw.write(transliterated[k - hi_st] + '\t' + 'HI' + '\t' + pos_sent[align_sent[k]]+'\n')
				#print codemixed[j], pos_sent[align_sent[k]]
				k += 1
 		with codecs.open(outfile, 'a', encoding='utf-8') as fw:
			fw.write("\n")

if __name__=="__main__":

	hparams = parseargument()
	indexlist, englishpos, hindipos, codemixed  = getSynthetic(hparams.synthetic_file)
	align_dict = getAlignDict(hparams.align_file, indexlist)
	pos_dict = getPosDict(hparams.pos_file, indexlist)
        
	#print "Debug calculations"
	#print "indexlist", indexlist
	#print "englishpos", englishpos
	#print "hindipos", hindipos
	#print "codemix", codemixed		
	#print "aligndict", align_dict
	#print "pos_dict", pos_dict

	alignPOS(indexlist, codemixed, pos_dict, align_dict, englishpos, hindipos, hparams.out_file)
	#Assume that we have tagged english POS tagged

