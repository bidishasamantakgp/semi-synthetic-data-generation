#Author: Bidisha Samanta
#This file will be used to transliterate and translate text using google API
#Transliteration API
#http://www.google.com/transliterate?langpair=hi|en&text=""
#Translation API
#https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=hn&dt=t&q=""

import urllib2
import urllib
import sys
import json
import ast
import re
import time
import argparse 



proxy = urllib2.ProxyHandler({"https": "https://10.3.100.207:8080", "http": "http://10.3.100.207:8080"})

opener = urllib2.build_opener(proxy)
urllib2.install_opener(opener)

#query = f.readline()

#user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
user_agent = 'Mozilla/5.0 (X11; U; Linux i686) Gecko/20071127 Firefox/2.0.0.11'
headers={'User-Agent':user_agent,}

def main():
	parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    	parser.add_argument('--source_lang', type=str, default='en',
                        help='source language hi, en')
    	parser.add_argument('--target_lang', type=str, default='hi',
                        help='target language to be converted en, hi')
    	parser.add_argument('--data_file', type=str, default='newspaper.txt',
                        help='directory to store tensorboard logs')
        parser.add_argument('--data', type=str, default='test',
                        help='line to be translated')
	parser.add_argument('--output_translated', type=str, default='translated.txt',
                        help='store the translated text in a file')
	parser.add_argument('--output_transliterated', type=str, default='transliterated.txt',
                        help='store the transliterated text in a file')
        args = parser.parse_args()
	process_new(args)
	
def translate(args):
	
	proxy = urllib2.ProxyHandler({"https": "https://172.16.2.30:8080", "http": "http://172.16.2.30:8080"})

	opener = urllib2.build_opener(proxy)
	urllib2.install_opener(opener)
	user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
	headers={'User-Agent':user_agent,}
	
	query = args.data
        url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl="+args.source_lang+"&tl="+args.target_lang+"&dt=t&q="+urllib.quote(query)
        print url
        #request=urllib2.Request(url,None,headers)
        #response = urllib2.urlopen(request)
	request = ''
        response = ''
        i = 0
	while i<5 :
		try:
                	request=urllib2.Request(url,None,headers)
                	response = urllib2.urlopen(request)
			if i < 5:
				i = 5
        	except:
                	print 'sleeping'
                	time.sleep(30)
			i = i + 1
			continue
                #request=urllib2.Request(url,None,headers)
                #response = urllib2.urlopen(request)

        data = response.read()
        new_data = re.sub('null','""',data)
        formatted = ast.literal_eval(new_data)
	retlist = []
	for el in formatted[0]:
		retlist.append(el[0])
	return retlist

def transliterate(args):
	proxy = urllib2.ProxyHandler({"https": "https://172.16.2.30:8080", "http": "http://172.16.2.30:8080"})
        opener = urllib2.build_opener(proxy)
        urllib2.install_opener(opener)
        user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
        headers={'User-Agent':user_agent,}

	translatedList = args.data.replace(',','').split()
	#for x in translatedList: print x
	transliterated = ''
	i = 0
	count = 0
	while i < len(translatedList):
		translated = translatedList[i : 30 + i]
		i = i + 30
		#print len(translated[0])
		translated = ' '.join([word for word in translated if len(word)>0])
        	#print translated
        	url = "http://www.google.com/transliterate?langpair="+args.target_lang+"|"+args.source_lang+"&text="+urllib.quote(translated)
        	print url
		#request=urllib2.Request(url,None,headers)
		#response = urllib2.urlopen(request)
        	request = ''
        	response = ''
        	try:
                	request=urllib2.Request(url,None,headers)
                	response = urllib2.urlopen(request)
        	except:
                	print 'sleeping'
                	time.sleep(30)
			#i = i - 30
			#count = count + 1
			if count < 6:
				i = i-30
				count = count + 1
				#continue
			else:
				count = 0
			continue
                	#request=urllib2.Request(url,None,headers)
                	#response = urllib2.urlopen(request)
		count = 0
		data = response.read()
		if len(data) == 0:
			return
		try:
        		#print url, translated, data
        		formatted = ast.literal_eval(data)
			if len(formatted) > 0:
        			transliterated += formatted[0]['hws'][0]
				
		except:
			continue
	return transliterated

def process_new(args):
	f = open(args.data_file, 'r')
	for line in f:
		args.data = line
		transliterated = transliterate(args)
		with open(args.output_transliterated, 'a') as fw:
                                for t in transliterated:
                                        fw.write(t+'\n')
		#print transliterate(args)
	
	
def process(args):
  f = open(args.data_file, 'r')
  for line in f:
	query = line
	#print
	url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl="+args.source_lang+"&tl="+args.target_lang+"&dt=t&q="+urllib.quote(query)
	print url
	#headers={'User-Agent':user_agent,}
	request = '' 
	response = ''
	try:
		request=urllib2.Request(url,None,headers)
		response = urllib2.urlopen(request)
	except:
		print 'sleeping'
		time.sleep(60)
		request=urllib2.Request(url,None,headers)
		response = urllib2.urlopen(request)
		
	#response = urllib2.urlopen(request)
	data = response.read()
	new_data = re.sub('null','""',data)
	formatted = ast.literal_eval(new_data)
	#formatted = list(data)
	for el in formatted[0]:
		translated = el[0]
		#print translated
		url = "http://www.google.com/transliterate?langpair="+args.target_lang+"|"+args.source_lang+"&text="+urllib.quote(translated)

		try:
			request=urllib2.Request(url,None,headers)
		except:
			print 'sleeping'
                	time.sleep(60)
                	request=urllib2.Request(url,None,headers)
		response = ''
		try:
			response = urllib2.urlopen(request)
		#except:
		#	print 'Error', url
		#	continue	
			data = response.read()
			print url, translated, data
			formatted = ast.literal_eval(data)
			transliterated = formatted[0]['hws']
	
			with open(args.output_translated,'a') as fw:
				fw.write(translated+'\n')
			with open(args.output_transliterated, 'a') as fw:
				for t in transliterated:
					fw.write(t+'\n')
		except:
                        print 'Error', url
                        continue

		#print translated, transliterated

if __name__=='__main__':
	main()
