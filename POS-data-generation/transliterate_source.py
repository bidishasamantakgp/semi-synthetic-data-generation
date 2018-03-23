
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
