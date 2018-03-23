
from translatetransliterate_util import * 

import urllib2
import urllib
import sys
import json
import ast
import re
import time
import argparse
import enchant

def parseargument():
 	parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--source_lang', type=str, default='en',
                        help='source language hi, en')
        parser.add_argument('--target_lang', type=str, default='hi',
                        help='target language to be converted en, hi')
	parser.add_argument('--file_name', type=str, default='hi',
                        help='file to be converted en, hi')
        parser.add_argument('--out_file', type=str, default='out',
                        help='file to be written')
	args = parser.parse_args()
	return args

def getAlignDict(filename, linenumberlist):
	f = open(filename)
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
                tokens = line.split()
                temp_dict = defaultdict()
                
		for token in tokens:
                        [word,pos] = token.split("\t")
                        temp_dict[word] = pos
                pos_dict[i] = temp_dict
	return pos_dict

def alignPOS(datafile, codemixed):
	flines = open(filename).readlines()
	linenumberlist = []

	for line in flines:
		tokens = line.split("|||")
		[linenumber, src] = tokens[0].split("\t")
		tgt = tokens[1].strip()
		src = src.strip()

		linenumberlist.append(linenumber)
	
	align_dict = getAlignDict(filename, linenumberlist)
 	pos_dict = getPosDict(filename, linenumberlist)	
		
	with open() as f:
		candidates = f.read().split("=====")
		for candidate in candidates:
			tokens = 		

if __name__=="__main__":
	args = parseargument()
	d = enchant.Dict("en_US")
	
	fw = open(args.out_file, 'a')
	f = open(args.file_name)
	
	for line in f:
		line = line.strip()
		if len(line) == 0:
			fw.write("\n")
			continue
		[word, tag] = line.split()
		transword = word
		if not d.check(word):
			args.data=word
			temp = transliterate(args) 
			if len(temp) > 0:
				transword = temp[0]
		fw.write(transword+" "+tag+"\n")
	#'''
