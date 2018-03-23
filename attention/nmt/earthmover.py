"""For training NMT models."""
from __future__ import print_function

import collections
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
	with open(endictfile,'r') as f:
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
				print('Error', word)
	for word in hindisentence:
		if word not in ('!','.',':', ';', ','):
			#print('HIWORD', word)
			try:
				y.append(hidict[word])
			except:
				print('Error', word)
	#print('ENG',np.array(x).shape, 'Hndi', np.array(y).shape)
	distance = 99
	if len(y) > 0 and len(x)> 0:
		distance = emd(np.array(x),np.array(y))
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
	hparams = parser.parse_args()
	return hparams


def getembeddings(hparams, en_dict, hi_dict, scope=None, target_session="", single_cell_fn=None):
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
  #segmentslist = [segment_sr,segment_caseless,segment_PCFG]
  segmentslist = [segment_sr,segment_PCFG]
  #length = len(hindisentence)
  enlen = len(engsentence)
  name = -1
  hindilen = len(hindisentence)
  newenglishsentence = copy.copy(engsentence)
  newhindisentence = copy.copy(hindisentence)
  #print("english sentence", engsentence, "hindi sentence", hindisentence, "new eng", newengsentence, "new hindi", newhindisentence)
  segmentname = ['SR', 'PCFG']
  for segments in segmentslist:
	name += 1
	for segment in segments:
		  segmentlist = segment.strip().replace(',', '').split()
		  print(segmentlist)
		  #out_seglist = [word for word in ensentence if word not in segmentlist]
		  if len(segmentlist) == enlen:
                        continue
		  random_seed1 = engsentence.index(segmentlist[0])
		  try:
		   random_seed2 = engsentence.index(segmentlist[-1])
		  except:
		   random_seed2 = enlen - 1 			
	
		  for l in range(2,hindilen/2 + 1):
                    for k in range(hindilen - l+1):
                        j = k + l -1
	  
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
			if len(engsentence[0:max(0,random_seed1)]) > 0:
				emd_out_left = calculate_emd(hidict, endict, engsentence[0:max(0,random_seed1)], hindisentence[0:k])
			if len(engsentence[random_seed2+1:])>0:
				emd_out_right = calculate_emd(hidict, endict, engsentence[random_seed2+1:], hindisentence[j+1:])
			newenglishsentence[random_seed1:random_seed2+1] = [x.upper() for x in newenglishsentence[random_seed1: random_seed2+1]]
                  	newsentence = [' '.join(engsentence[0:max(0,random_seed1)]), ' '.join(hindisentence[k:j+1]), ' '.join(engsentence[random_seed2+1:])]
                
                  	with codecs.open("/tmp/output_emd_all.csv",'a','utf-8') as csvfile:
			#with open("/tmp/output_emd.csv", 'a') as csvfile:
                        	fieldnames = ['English', 'Hindi', 'Segment English','Greedy Hindi', 'Scorein', 'Scoreoutleft','ScoreoutRight']
                        	#writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        	#writer.writeheader()
				csvfile.write(' '.join(hindisentence)+','+' '.join(newenglishsentence)+','+' '.join(newsentence)+','+str(emd_in)+ ','+str(emd_out_left)+ ','+str(emd_out_right)+','+str(emd_in+emd_out_left+emd_out_right)+'\n')
                        	#csvfile.write( ' '.join(engsentence)+','+ ' '.join(hindisentence)+ ','+ ' '.join(newenglishsentence)+','+' '.join(newsentence)+','+str(emd_in)+ ','+str(emd_out_left)+ ','+str(emd_out_right)+','+str(emd_in+emd_out_left+emd_out_right)+'\n')
				#print(' '.join(engsentence),',', ' '.join(hindisentence), ',', ' '.join(newenglishsentence),',',' '.join(newsentence),',',emd_in, ',',emd_out_left, ',',emd_out_right)
				#writer.writerow({fieldnames[0]: ' '.join(engsentence),  fieldnames[1]: ' '.join(hindisentence), fieldnames[2]: ' '.join(newenglishsentence), fieldnames[3]: ' '.join(newsentence), fieldnames[4]: emd_in, fieldnames[5]: emd_out_left, fieldnames[6]: emd_out_right})



if __name__=="__main__":
	hparams = parsearguments()
	(endict, hidict) = loaddict(hparams.dict_file+'_en.txt', hparams.dict_file+'_hi.txt')	
	#print(hidict.keys())
	getembeddings(hparams, endict, hidict)
	
