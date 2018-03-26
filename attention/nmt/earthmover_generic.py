"""For training NMT models."""
from __future__ import print_function

import collections
from collections import defaultdict
import operator
import math
import os
import random
import time
import numpy as np
import csv
import copy
import ast
import urllib2
import urllib
import re
import argparse
from emd import emd

import tensorflow as tf
import json
import codecs

def loaddict(endictfile,hidictfile):
	with codecs.open(endictfile,'r', 'utf-8') as f:
		#print(f.read())
		endict = json.loads(f.read())
	with codecs.open(hidictfile, encoding='utf-8') as f:
	#with open(hidictfile, 'r') as f:
		#print(f.read())
		hidict = json.loads(f.read())
	return (endict,hidict)

def calculate_emd(hidict, endict, ensentence, hindisentence):
	x = []
	y = []
	#print('Inside EMD', ensentence)
	for word in ensentence:
		word = word.lower()
		if word not in ('!','.',':', ';', ','):
			#print('ENWORD', word)
			try:
				x.append(endict[word])
			except:
				#print("except", word)	
				continue
				#print('Error', word)
	for word in hindisentence:
		if word not in ('!','.',':', ';', ','):
			#print('HIWORD', word)
			try:
				y.append(hidict[word])
			except:
				#print("except", word)
				continue
				#print('Error', word)
	#print('ENG',np.array(x).shape, 'Hndi', np.array(y).shape)
	distance = 99
	if len(y) > 0 and len(x)> 0:
		#print("correct", ensentence)
		distance = emd(np.array(x),np.array(y))
		#distance = emd(np.array(y),np.array(x))
	return distance

def parsearguments():
	parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--train_prefix', type=str, default='train.txt',
                        help='data file to store the corpus which needs to be transformed to code mixed')
        parser.add_argument('--segment_file', type=str, default='segment.txt',
                        help='file to store the segmentation information genearted by a grammar')
	parser.add_argument('--dict_file', type=str, default='dict.txt',
                        help='file to store the embedding')
	parser.add_argument('--out_file', type=str, default='dict.txt',
                        help='outputfile to store the embedding')
	parser.add_argument('--src', type=str, default='en',
                        help='src language')
	parser.add_argument('--tgt', type=str, default='hi',
                        help='target file')


	hparams = parser.parse_args()
	return hparams


def getembeddings(hparams, en_dict, hi_dict, segment_sr, engsentence, hindisentence,scope=None, target_session="", single_cell_fn=None):
  '''
  with open(hparams.train_prefix+'.en','r') as f:
  	engsentence = f.read().replace(',', '').split()
  with codecs.open(hparams.train_prefix+'.hi', encoding='utf-8') as f:
  #with open(hparams.train_prefix+'.hi') as f:
	hindisentence = f.read().replace(',', '').split()
  #print(engsentence, hindisentence)
  with open(hparams.segment_file, 'r') as f:
	tokens = f.read().split('\t')
	segment_sr = ast.literal_eval(tokens[0])
	#segment_caseless = ast.literal_eval(tokens[1])
	segment_PCFG = ast.literal_eval(tokens[1])
  '''
  #segmentslist = [segment_sr,segment_caseless,segment_PCFG]
  segmentslist = [segment_sr]
  #length = len(hindisentence)
  enlen = len(engsentence)
  name = -1
  hindilen = len(hindisentence)
  newenglishsentence = copy.copy(engsentence)
  newhindisentence = copy.copy(hindisentence)
  #print("english sentence", engsentence, "hindi sentence", hindisentence, "new eng", newengsentence, "new hindi", newhindisentence)
  segmentname = ['SR']
  #, 'PCFG']
  for segments in segmentslist:
	name += 1
	for segment in segments:
		  segmentlist = segment.strip().split()
		  #print(segmentlist)
		  #out_seglist = [word for word in ensentence if word not in segmentlist]
		  if len(segmentlist) == enlen:
                        continue
		  random_seed1 = engsentence.index(segmentlist[0])
		  reverselist = copy.copy(engsentence)
                  reverselist.reverse()
                  random_seed2 = enlen - reverselist.index(segmentlist[-1]) - 1

		  segment_dict = defaultdict(defaultdict)
		  for l in range(1,max(len(segmentlist),hindilen-1) + 1):
                     for k in range(hindilen - l):
                        j = min(k + l -1, hindilen-1)
	  
		  #for k in range(hindilen):
		  #   for j in range(k+1, hindilen):
			mapping = ''
                  	hindisegment = ''
                  	indexlist = []
                  	newenglishsentence = copy.copy(engsentence)
                  	newhindisentence = copy.copy(hindisentence)
			count = 0
			overallenglist = []
            		emd_in = calculate_emd(hidict, endict, segmentlist, hindisentence[k:j+1])
			emd_out_left = 0
			emd_out_right = 0
			engoutseg = engsentence[0:max(0,random_seed1)]+ engsentence[random_seed2+1:]
			hioutseg = hindisentence[0:k]+ hindisentence[j+1:]
			emd_out = calculate_emd(hidict, endict, engoutseg, hioutseg)
			#if len(engsentence[0:max(0,random_seed1)]) > 0:
			#	emd_out_left = calculate_emd(hidict, endict, engsentence[0:max(0,random_seed1)], hindisentence[0:k])
			#if len(engsentence[random_seed2+1:])>0:
			#	emd_out_right = calculate_emd(hidict, endict, engsentence[random_seed2+1:], hindisentence[j+1:])
			newenglishsentence[random_seed1:random_seed2+1] = [x.upper() for x in newenglishsentence[random_seed1: random_seed2+1]]
                  	newsentence = [' '.join(engsentence[0:max(0,random_seed1)]), ' '.join(hindisentence[k:j+1]), ' '.join(engsentence[random_seed2+1:])]
			#segment_dict[' '.join(newsentence)] = (emd_in+emd_out_left+emd_out_right, emd_in, emd_out_left, emd_out_right)
			segment_dict[' '.join(newsentence)] = (emd_in+emd_out, emd_in, emd_out, random_seed1 - 1, enlen - random_seed2 - 1, k, j)

		  newenglishsentence[random_seed1:random_seed2+1] = [x.upper() for x in newenglishsentence[random_seed1: random_seed2+1]]
		  # ascending order sorting
		  sorted_candidates = sorted(segment_dict.items(), key=operator.itemgetter(1,0))
		  #getsortedlist(segment_dict)
                  with codecs.open(hparams.output_file,'a','utf-8') as csvfile:
                        for (candidate, (score, emdin, emdout, rs1, rs2, k, j)) in sorted_candidates[:10]:
                                csvfile.write(' '.join(hindisentence)+'\t'+' '.join(newenglishsentence)+'\t'+candidate+ '\t'+str(score)+'\t'+str(emdin)+'\t'+str(emdout)+'\t'+ str(rs1)+"\t"+str(rs2)+"\t"+str(k)+'\t'+str(j)+'\n')
			csvfile.write("\n")
                


if __name__=="__main__":
	hparams = parsearguments()
	(endict, hidict) = loaddict(hparams.dict_file+'_'+hparams.src+'.txt', hparams.dict_file+'_'+hparams.tgt+'.txt')	
	#print(hidict.keys())
  	with codecs.open(hparams.segment_file, 'r', "utf-8") as f:
        	lines = f.readlines()
	for line in lines:
		tokens = line.split('\t')
        	segment_sr = ast.literal_eval(tokens[0])
		engsentence = tokens[1].strip().split()
		hindisentence = tokens[2].strip().split()
		getembeddings(hparams, endict, hidict, segment_sr, engsentence, hindisentence)
