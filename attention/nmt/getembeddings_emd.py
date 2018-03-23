"""For training NMT models."""
from __future__ import print_function

import json
import operator
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
from collections import defaultdict
import codecs
import tensorflow as tf
from emd import emd

from . import attention_model
from . import gnmt_model
from . import inference
from . import model as nmt_model
from . import model_helper
from .utils import iterator_utils
from .utils import misc_utils as utils
from .utils import nmt_utils
from .utils import vocab_utils

class TrainModel(
    collections.namedtuple("TrainModel", ("graph", "model", "iterator",
                                          "skip_count_placeholder"))):
  pass


def create_model(
	model_creator, hparams, scope=None, single_cell_fn=None,
    model_device_fn=None):
  
  """Create train graph, model, and iterator."""
  src_file = "%s.%s" % (hparams.train_prefix, hparams.src)
  tgt_file = "%s.%s" % (hparams.train_prefix, hparams.tgt)
  src_vocab_file = hparams.src_vocab_file
  tgt_vocab_file = hparams.tgt_vocab_file

  graph = tf.Graph()

  with graph.as_default():
    src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
        src_vocab_file, tgt_vocab_file, hparams.share_vocab)

    src_dataset = tf.contrib.data.TextLineDataset(src_file)
    tgt_dataset = tf.contrib.data.TextLineDataset(tgt_file)
    skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)
    

    iterator = iterator_utils.get_iterator(
        src_dataset,
        tgt_dataset,
        src_vocab_table,
        tgt_vocab_table,
        batch_size=hparams.batch_size,
        sos=hparams.sos,
        eos=hparams.eos,
        source_reverse=hparams.source_reverse,
        random_seed=hparams.random_seed,
        num_buckets=hparams.num_buckets,
        src_max_len=hparams.src_max_len,
        tgt_max_len=hparams.tgt_max_len,
        skip_count=skip_count_placeholder)

    # Note: One can set model_device_fn to
    # `tf.train.replica_device_setter(ps_tasks)` for distributed training.
    with tf.device(model_device_fn):
      model = model_creator(
          hparams,
          iterator=iterator,
          mode=tf.contrib.learn.ModeKeys.TRAIN,
          source_vocab_table=src_vocab_table,
          target_vocab_table=tgt_vocab_table,
          scope=scope,
          single_cell_fn=single_cell_fn)
  
  return TrainModel(
      graph=graph,
      model=model,
      iterator=iterator,
      skip_count_placeholder=skip_count_placeholder)
  #return (graph, model, iterator, skip_count_placeholder)
def load_embeddings(endictfile,hidictfile):
        with codecs.open(endictfile,'r', 'utf-8') as f:
        #with open(endictfile,'r') as f:
                #print(f.read())
                endict = json.loads(f.read())
        with codecs.open(hidictfile, encoding='utf-8') as f:
        #with open(hidictfile) as f: 
        #with open(hidictfile, 'r') as f:
                #print(f.read())
                hidict = json.loads(f.read())
        return (endict,hidict)


def get_idfdict(filename):
        freq_dict = defaultdict(int)
        #with open(filename) as f:
	with codecs.open(filename, encoding="utf-8") as f:
                lines = f.readlines()
        for line in lines:
                words = line.replace(' \'', '\'').split()
                for word in words:
                        freq_dict[word] += 1
        return freq_dict
def calculate_emd(hidict, endict, ensentence, hindisentence, probdict, idf_dict, idf_dict_hi, enidlist, hiidlist):
        x = []
        y = []
	xin = []
	yin = []
        #print('Inside EMD', ensentence)
        #for word in ensentence:
	en_idf = 0.0
	for en in enidlist:
		word = ensentence[en]
                word = word.lower()
                if word == '\'re':
                        word ='are'
                if word not in ('!','.',':', ';', ','):
                        #print('ENWORD', word)
                        try:
                                x.append(endict[word])
				xin.append(en)
				en_idf += 1.0/(idf_dict[ensentence[en]]+1)
                        except:
                                #print("except", word)
                                continue
                                #print('Error', word)
        #for word in hindisentence:
	hi_idf = 0.0
	for hi in hiidlist:
		word = hindisentence[hi]
                if word not in ('!','.',':', ';', ','):
                        #print('HIWORD', word)
                        try:
                                y.append(hidict[word])
				yin.append(hi)
				hi_idf += 1.0/(idf_dict_hi[hindisentence[hi]]+1)
                        except:
                                #print("except", word)
                                continue
                                #print('Error', word)
        #print('DEBUG', hi_idf, en_idf)
	distance=np.zeros((len(xin), len(yin)))
        for en in range(len(xin)):
                for hi in range(len(yin)):
                        #print "Debug",probdict[ensentence[en]]
                        #print("idf_dict_hi[hindisentence[yin[hi]]", idf_dict_hi[hindisentence[yin[hi]]], idf_dict[ensentence[xin[en]]])
			try:
				#distance[en][hi] = (1 - probdict[yin[hi],0][xin[en]]) 
				#* abs(1.0/(1.0+idf_dict[ensentence[xin[en]]]) - 1.0/(1.0 + idf_dict_hi[hindisentence[yin[hi]]]))
				distance[en][hi] = (1 - probdict[yin[hi],0][xin[en]])*1.0*abs(1.0 - idf_dict_hi[hindisentence[yin[hi]]]/idf_dict[ensentence[xin[en]]])
                                #print("DEBUG D", distance[en][hi])
				#distance[en][hi] = 1 - probdict[yin[hi],0][xin[en]]*(1.0/idf_dict[ensentence[xin[en]]]+1)
                        except:
                                distance[en][hi] = 1
        #print("Debug dist", distance)
	#print('ENG',np.array(x).shape, 'Hndi', np.array(y).shape)
        distVal= 99
        if len(y) > 0 and len(x)> 0:
                #print("correct", ensentence)
                #distVal = emd(np.array(x),np.array(y), D=distance) 
		#* hi_idf/en_idf 
		distVal = emd(np.array(x),np.array(y), D=distance) 
		#* abs(en_idf - hi_idf)/en_idf
		#* (en_idf/(hi_idf+1))
                #distance = emd(np.array(y),np.array(x))
        return distVal

def score(hindisentence, hindisegments, enlen):
	#engsentecne -> list of tokens in engsentence
	#engsegments -> list of lists for contiguous segments
	#hindisentence -> list of tokens in hindi sentence
	#hindisegments -> index for segments
        if len(hindisegments) == 0:
		return 0	
	discontinuous = [] 
	contiguous = []

	lencont = 0
	lendis = 0

        hilen = len(hindisentence)
        #print(hilen, hindisegments[-1])
	#for i in range(hilen):
	for i in range(hindisegments[0], hindisegments[-1]+1):
	        #print(i)
		if i in hindisegments:
			lencont += 1
			if lendis != 0:
				discontinuous.append(lendis)
				lendis = 0
		else:
			lendis += 1
			if lencont != 0:
				contiguous.append(lencont)
				lencont = 0 
	if lendis != 0:
        	discontinuous.append(lendis)
	if lencont != 0:
		contiguous.append(lencont)
         
        #print("hindi segments", hindisegments)
	#print("len segments", hilen, contiguous, discontinuous)
	
	contiguous = [(x+0.0)/enlen for x in contiguous]
	#discontinuous = [(x+0.0)/hilen for x in discontinuous]
	denom = np.average(discontinuous) * len(discontinuous) / (hindisegments[-1] - hindisegments[0]+1) 
	if len(discontinuous) == 0:
		denom = 1
	#sc = (np.sum(contiguous) * 1.0) / denom
	sc = 1.0/denom
	#print(contiguous, np.average(discontinuous), sc)
	return sc


def translate(query, src, tgt):

        proxy = urllib2.ProxyHandler({"https": "https://172.16.2.30:8080", "http": "http://172.16.2.30:8080"})

        opener = urllib2.build_opener(proxy)
        urllib2.install_opener(opener)
        user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
        headers={'User-Agent':user_agent,}

        query = query.replace(',','')
        url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl="+src+"&tl="+tgt+"&dt=t&q="+urllib.quote(query)
        print(url)
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
                        #print 'sleeping'
                        time.sleep(30)
                        i = i + 1
                        continue
                #request=urllib2.Request(url,None,headers)
                #response = urllib2.urlopen(request)

        data = response.read()
        new_data = re.sub('null','""',data)
	new_data = re.sub(',.*', '', new_data)
	new_data = re.sub('\[\[\[', '', new_data)
        #formatted = ast.literal_eval(new_data)
	print(data, new_data)
        retlist = []
        #for el in formatted[0]:
        #        retlist.append(el[0])
        #print("retlist",retlist)
	return new_data
	
def getembeddings(hparams,segments, engsentence, hindisentence, idf_dict, idf_dict_hi, endict, hidict,scope=None, target_session="", single_cell_fn=None):
#def getembeddings(hparams, scope=None, target_session="", single_cell_fn=None):
  """Train a translation model."""
  log_device_placement = hparams.log_device_placement
  out_dir = hparams.out_dir
  num_train_steps = hparams.num_train_steps
  steps_per_stats = hparams.steps_per_stats
  steps_per_external_eval = hparams.steps_per_external_eval
  steps_per_eval = 10 * steps_per_stats
  if not steps_per_external_eval:
    steps_per_external_eval = 5 * steps_per_eval

  if not hparams.attention:
    model_creator = nmt_model.Model
  elif hparams.attention_architecture == "standard":
    model_creator = attention_model.AttentionModel
  elif hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
    model_creator = gnmt_model.GNMTModel
  else:
    raise ValueError("Unknown model architecture")

  #graph, model, iterator, skip_count_placeholder = 
  train_model = create_model(model_creator, hparams, scope, single_cell_fn)
  model_dir = hparams.out_dir
  config_proto = utils.get_config_proto(log_device_placement=log_device_placement)

  sess = tf.Session(
      target=target_session, config=config_proto, graph=train_model.graph)
  with train_model.graph.as_default():
    loaded_model, global_step = model_helper.create_or_load_model(
       train_model.model, model_dir, sess, "train")
  sess.run(
      train_model.iterator.initializer,
      feed_dict={train_model.skip_count_placeholder: 0})
  #print("iterator soucre", iterator.source.eval(session=sess), iterator.source.shape) 
  step_result = loaded_model.getembeddings(sess)
  encoder_outputs, decoder_outputs, encoder_inputs, decoder_inputs, history = step_result
  #print("encoder input shape",encoder_inputs.shape)
  #print(encoder_inputs)
  #print("decoder input shape",decoder_inputs.shape)
  #print(decoder_inputs)
  #print("decoder_outputs_shape",decoder_outputs.rnn_output.shape)
  #print("history_shape",history.shape)
  enlen = len(engsentence)

  hindilen = len(hindisentence)
  newenglishsentence = copy.copy(engsentence)
  newhindisentence = copy.copy(hindisentence)

  name = -1
  segmentname = ['SR']
  for segment in segments:
				 
		  segmentlist = segment.replace('-LSB-','[').replace('-RSB-',']').strip().split()
		  if len(segmentlist) == enlen:
			continue
		  #print("DEBUG",segmentlist, engsentence)
		  random_seed1 = -1
		  try:
			random_seed1 = engsentence.index(segmentlist[0])
		  except:
			for w in segmentlist:
		  		for el in range(len(engsentence)):
					if(w in engsentence[el]):
						random_seed1 = el
						break
				if(random_seed1!=-1):
					break 
		  random_seed2 = -1
		  try:
		   random_seed2 = engsentence[random_seed1+1:].index(segmentlist[-1])+random_seed1 + 1
		  except:
			temp = copy.copy(segmentlist)
			temp.reverse()
			for w in temp:
		   		for el in reversed(range(random_seed1, len(engsentence))):
                        		#print("Debug w", w)
					if(w in engsentence[el]):
					
		   				random_seed2 = el
						break
				if(random_seed2!=-1):
                                        break

			 			
		  #print("DEBUG",segmentlist, engsentence, random_seed1, random_seed2)
		  mapping = ''
  		  hindisegment = ''
		  indexlist = []
		  newenglishsentence = copy.copy(engsentence)
		  newhindisentence = copy.copy(hindisentence)
		  segment_dict = defaultdict() 
                  #for l in range(1,max(len(segmentlist),hindilen-1) + 1):
                  for l in range(1,max(len(segmentlist),hindilen-1) + 1):   
		     for k in range(hindilen - l):
                        j = min(k + l , hindilen-1) 
			mapping = ''
                  	#hindisegment = ''
                  	indexlist = []
                  	newenglishsentence = copy.copy(engsentence)
                  	newhindisentence = copy.copy(hindisentence)
			count = 0
			probmul = 1
			overallenglist = []
			#enlen * 0.3
			sumscore = 0.0
			score = 1.0
                        #print(k,j)
			emd_in = calculate_emd(hidict, endict, engsentence, hindisentence, history, idf_dict, idf_dict_hi,range(random_seed1, random_seed2+1), range(k,j+1))
                        engoutseg = engsentence[0:max(0,random_seed1)]+ engsentence[random_seed2+1:]
                        hioutseg = hindisentence[0:k]+ hindisentence[j+1:]
                        emd_out = calculate_emd(hidict, endict, engsentence, hindisentence, history, idf_dict, idf_dict_hi, range(0,max(0,random_seed1))+range(random_seed2+1,enlen), range(0,k)+range(j+1, hindilen))
                        #score = emd_in * min(abs((j+1.0-k)-(random_seed2+1 - random_seed1))/(random_seed2+1 - random_seed1),1.0)+ emd_out * min(abs((hindilen + 1.0-(j+1 -k)) - (enlen+1-(random_seed2 +1 - random_seed1)))/(enlen+1-(random_seed2 +1 - random_seed1)),1.0)
			score = emd_in + emd_out
			#score = emd_in *1.0/(j+1.0-k)
			#/(random_seed2+1 - random_seed1))
			#score +=emd_out * 1.0 /(hindilen + 1.0-(j+1 -k))
			#/(enlen+1-(random_seed2 +1 - random_seed1)))
                        newsentence = [' '.join(engsentence[0:max(0,random_seed1)]), ' '.join(hindisentence[k:j+1]), ' '.join(engsentence[random_seed2+1:])]
                        segment_dict[' '.join(newsentence)] = (score, emd_in, emd_out)
		  newenglishsentence[random_seed1:random_seed2+1] = [x.upper() for x in newenglishsentence[random_seed1: random_seed2+1]]
          	  sorted_candidates = sorted(segment_dict.items(), key=operator.itemgetter(1,0))
          	  #with codecs.open("/tmp/output_attention_emd_all.csv",'a','utf-8') as csvfile:
		  #with codecs.open("/tmp/output_attention_emd_test_new1.csv",'a','utf-8') as csvfile:
		  with codecs.open("/tmp/code_mixed_eval/output_attention_emd.csv",'a','utf-8') as csvfile:
		  #with open("/tmp/output_attention_emd_all.csv",'a') as csvfile:

                        for (candidate, (score, score_in, score_out)) in sorted_candidates[:10]:
                                csvfile.write(' '.join(hindisentence)+'\t'+' '.join(newenglishsentence)+'\t'+candidate+'\t'+str(score)+'\t'+str(score_in)+'\t'+str(score_out)+'\n')
                        csvfile.write("\n")

def getsegment(segment_file):
    i = 0
    map_dict = defaultdict()
    listenglish = []
    listhindi = []
    listsegment = []
    english_sample = []
    englishlist = []
    hindilist = []
    with codecs.open(segment_file, 'r', 'utf-8') as f:
    #with open(segment_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
                tokens = line.split('\t')
                if len(tokens)<3:
			continue
		segment_sr = ast.literal_eval(tokens[0].strip())
                englishwords = tokens[1].split()
                for seg in segment_sr:
                        if len(seg.split()) <= len(englishwords)/2 and len(seg.split())>= len(englishwords)/4 :
                                listsegment.append(seg)
                listsegment = listsegment[:min(10, len(listsegment))]

		#listsegment.append(segment_sr)
                englishlist.append(tokens[1].replace(' \'','\'').replace('-LSB-',']').replace('-RSB-',']').strip())
                hindilist.append(tokens[2].strip())
    return (listsegment, englishlist, hindilist)

def parsearguments():
        parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--segment_file', type=str, default='segment.txt',
                        help='file to store the segmentation information genearted by a grammar')
        #parser.add_argument('--stop_words', type=str, default='stopwords-hi.txt',
        #                help='file containing the stop words')
        parser.add_argument('--english_sentence', type=str, default='english_senetence.txt',
                        help='file containing the stop words')
        hparams = parser.parse_args()
        return hparams

#def main_embeddings(hparams, scope=None, target_session="", single_cell_fn=None):
if __name__=="__main__":
      #out_dir = "/tmp/nmt_model_en_hi_devnagari_wordembed_all/" 
      #out_dir = "/tmp/nmt_model_en_hi_devnagari_wordembed_wordpiece_old/"
      out_dir = "/tmp/nmt_model_en_es_all/"
      hparams = utils.load_hparams(out_dir)
      hparams.num_embeddings_partitions = 0
      hparams.get_embeddings = True

      #parsearguments()
      dict_file = "/home/rs/bidisha/TF/code/nmt_alignments_final/data/wordembed"
      #dict_file = "/home/rs/bidisha/basava/wordembed"
      #dict_file = "/home/rs/bidisha/"
      (endict, hidict) = load_embeddings(dict_file+'_en.txt', dict_file+'_es.txt') 
      

      idf_dict = get_idfdict(hparams.train_prefix+".en")
      idf_dict_hi = get_idfdict(hparams.train_prefix+".es")
      hparams.train_prefix = hparams.train_prefix.replace("train", "test")
      
      #segment_file = "/home/rs/bidisha/giza++/data/converted_segment.txt" 
      #segment_file = "/home/rs/bidisha/TF/data/translation/evaluation/SR_pair.txt"
      #segment_file = "/home/rs/bidisha/TF/data/translation/evaluation/converted_SR_pair.txt"
      segment_file = "/home/rs/bidisha/TF/code/corenlp/CoreNLP/en_es/evaluation.txt"
      #segment_file = "/home/rs/bidisha/giza++/data/segment_test.txt"
      (listsegment, englist, hilist) = getsegment(segment_file)
      count = 0
      i = 0
      for (segments, engsentence, hindisentence) in zip(listsegment, englist, hilist):
                #print("inside getemb", count)
                #engsentence = engsentence.split()
		#hindisentence = hindisentence.split()
		#if engsentence
		'''
                with codecs.open('/tmp/nmt_data_en_hi_devnagari_wordpiece/test.en', 'w', 'utf-8') as f:
		#with codecs.open('/tmp/nmt_data_en_hi_devnagari_all/test.en', 'w', 'utf-8') as f:
		#with open('/tmp/nmt_data_en_hi_devnagari_all/test.en', 'w') as f:
                	f.write(engsentence+'\n')
		with codecs.open('/tmp/nmt_data_en_hi_devnagari_wordpiece/test.hi', 'w', 'utf-8') as f:
		#with codecs.open('/tmp/nmt_data_en_hi_devnagari_all/test.hi', 'w', 'utf-8') as f:
        	#with open('/tmp/nmt_data_en_hi_devnagari_all/test.hi', 'w') as f:
                	f.write(hindisentence+'\n')
        	#with open('/tmp/nmt_data_en_hi_devnagari_all/segment_file', 'w') as f:
                #	f.write(+'\n')		
                '''
		with open(hparams.train_prefix+".en", 'w') as f:
                        f.write(engsentence+'\n')
                with open(hparams.train_prefix+".es", 'w') as f:
                        f.write(hindisentence+'\n')

		getembeddings(hparams, segments, engsentence.split(), hindisentence.split(), idf_dict, idf_dict_hi, endict, hidict)
		#break

