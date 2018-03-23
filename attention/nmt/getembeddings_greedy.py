"""For training NMT models."""
from __future__ import print_function

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
def get_idfdict(filename):
        freq_dict = defaultdict(int)
        with open(filename) as f:
	#with codecs.open(filename, encoding="utf-8") as f:
                lines = f.readlines()
        for line in lines:
                words = line.replace(' \'', '\'').split()
                for word in words:
                        freq_dict[word] += 1
        return freq_dict
def accuracy(hindiidlist):
	
	previndex = hindiidlist[0]
	nc = 0
	for i in hindiidlist:
		if previndex != i-1:
			nc +=1
		previndex = i
	c = nc + 1	
	return ((c*1.0)/(c+nc))
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
        print(hilen, hindisegments[-1])
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
	print(contiguous, np.average(discontinuous), sc)
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
	
def getembeddings(hparams,segments, engsentence, hindisentence, idf_dict, scope=None, target_session="", single_cell_fn=None):
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
  print("encoder input shape",encoder_inputs.shape)
  #print(encoder_inputs)
  print("decoder input shape",decoder_inputs.shape)
  #print(decoder_inputs)
  print("decoder_outputs_shape",decoder_outputs.rnn_output.shape)
  print("history_shape",history.shape)
  enlen = len(engsentence)

  hindilen = len(hindisentence)
  newenglishsentence = copy.copy(engsentence)
  newhindisentence = copy.copy(hindisentence)

  name = -1
  segmentname = ['SR']
  segment_dict = defaultdict()
  for segment in segments:
				 
		  segmentlist = segment.replace('-LSB-','[').replace('-RSB-',']').strip().split()
		  if len(segmentlist) == enlen:
			continue
		  print("DEBUG",segmentlist, engsentence)
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
		  #segment_dict = defaultdict() 
		  score = 1.0
		  sumscore = 0.0
		  overallenglist = []
		  for i in range(hindilen):
                        	hindiword = hindisentence[i]
                        	engindexlist = history[i,0].argsort()[-int(enlen * 0.3):]
				#engindex = history[i,0].argsort()[-1]
				overallenglist.extend(engindexlist)
				#overallenglist.append(engindex)
				
				#sumscore += history[i,0][engindex] 
                        	#if engindex in range(random_seed1, random_seed2+1):
				inter = set(engindexlist).intersection(range(random_seed1, random_seed2 + 1))
				if len(inter) > 0:
                                	indexlist.append(i)
					for enindex in inter:
						sumscore += history[i,0][enindex] * (1.0/(idf_dict[engsentence[enindex]]+1))
		  if(len(indexlist)==0):
			continue		  
		  print("DEBUG Overall", overallenglist, random_seed1, random_seed2, segment, engsentence)
		  score *= sumscore
		  #accuracy = 2.0*TP /(2.0*TP + FP + FN)
                  score *= accuracy(indexlist) * len(set(overallenglist).intersection(range(random_seed1, random_seed2+1)))/(random_seed2 - random_seed1 + 1)
		  newsentence = [' '.join(engsentence[0:max(0,random_seed1)]), ' '.join([hindisentence[x] for x in indexlist]), ' '.join(engsentence[random_seed2+1:])]
                  #segment_dict[' '.join(newsentence)] = score	
                  newenglishsentence[random_seed1:random_seed2+1] = [x.upper() for x in newenglishsentence[random_seed1: random_seed2+1]]
          	  segment_dict[' '.join(newsentence)] = (score, ' '.join(newenglishsentence))
  sorted_candidates = sorted(segment_dict.items(), key=operator.itemgetter(1,0), reverse=True)
  with open("/tmp/output_greedy_test_all.csv",'a') as csvfile:
	for (candidate, (score, newenglishsentence)) in sorted_candidates:
                       csvfile.write(' '.join(hindisentence)+','+ newenglishsentence +','+candidate+','+ ','+str(score)+','+'\n')
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
    #with codecs.open(segment_file, 'r', 'utf-8') as f:
    with open(segment_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
                tokens = line.split('\t')
                segment_sr = ast.literal_eval(tokens[0].strip())
                listsegment.append(segment_sr)
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

if __name__=="__main__":
      out_dir = "/tmp/nmt_model_en_hi_devnagari_wordembed_all/" 
      hparams = utils.load_hparams(out_dir)
      #parsearguments()
      idf_dict = get_idfdict(hparams.english_sentence)
      segment_file = "/home/rs/bidisha/giza++/data/segment.txt" 
      (listsegment, englist, hilist) = getsegment(segment_file)
      count = 0
      i = 0
      for (segments, engsentence, hindisentence) in zip(listsegment, englist, hilist):
		with open('/tmp/nmt_data_en_hi_devnagari_all/test.en', 'w') as f:
                	f.write(engsentence+'\n')
        	with open('/tmp/nmt_data_en_hi_devnagari_all/test.hi', 'w') as f:
                	f.write(hindisentence+'\n')
                getembeddings(hparams, segments, engsentence.split(), hindisentence.split(), idf_dict)
		#break

