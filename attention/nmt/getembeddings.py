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
	

def getembeddings(hparams, scope=None, target_session="", single_cell_fn=None):
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
  print(encoder_inputs.shape)
  print(encoder_inputs)
  print(decoder_inputs.shape)
  print(decoder_inputs)
  print(decoder_outputs.rnn_output.shape)
  print(history.shape)
  print(history)
  print(history[0,0])
  #transposed_history = history
  #transposed_history = np.transpose(history, (2, 1, 0))
  with open(hparams.train_prefix+'.en','r') as f:
  	engsentence = f.read().split()
  with open(hparams.train_prefix+'.hi','r') as f:
  	hindisentence = f.read().split()

  with open(hparams.segment_file, 'r') as f:
	tokens = f.read().split('\t')
	segment_sr = ast.literal_eval(tokens[0])
	#segment_caseless = ast.literal_eval(tokens[1])
	segment_PCFG = ast.literal_eval(tokens[1])
  #segmentslist = [segment_sr,segment_caseless,segment_PCFG]
  segmentslist = [segment_sr,segment_PCFG]
  #length = len(hindisentence)
  enlen = len(engsentence)

  hindilen = len(hindisentence)
  newenglishsentence = copy.copy(engsentence)
  newhindisentence = copy.copy(hindisentence)
  #print("english sentence", engsentence, "hindi sentence", hindisentence, "new eng", newengsentence, "new hindi", newhindisentence)

  csi = '\x1B['
  red = csi + '31;1m'
  yellow = csi + '33;1m'
  end = csi + '0m'
  mapping = ''
  hindisegment = '' 
  name = -1
  segmentname = ['SR', 'PCFG']
  for segments in segmentslist:
	name += 1
	for segment in segments:
	
		  segmentlist = segment.strip().split()
		  random_seed1 = engsentence.index(segmentlist[0])
		  try:
		   random_seed2 = engsentence.index(segmentlist[-1])
		  except:
		   random_seed2 = enlen -1 			
  #for random_seed1 in range(enlen):
  #	for random_seed2 in range(random_seed1+2, enlen):
		  mapping = ''
  		  hindisegment = ''
		  indexlist = []
		  newenglishsentence = copy.copy(engsentence)
		  newhindisentence = copy.copy(hindisentence)
		  '''
		  for k in range(hindilen):
		     for j in range(k+1, hindilen):
			mapping = ''
                  	hindisegment = ''
                  	indexlist = []
                  	newenglishsentence = copy.copy(engsentence)
                  	newhindisentence = copy.copy(hindisentence)
			count = 0
			probmul = 1
			overallenglist = []
			for i in range(k,j+1):
                        	hindiword = hindisentence[i]
                        	engindexlist = history[i,0].argsort()[-3:]
				overallenglist.extend(engindexlist)
                        	if len(set(engindexlist).intersection(range(random_seed1, random_seed2+1))) > 0:
				
                                	indexlist.append(i)
                                	for engindex in engindexlist:
                                        	mapping += hindiword + '->'+ engsentence[engindex]+ ' '
					count += 1
                                hindisegment += ' ' + hindiword
					#count += 1

			val = (count * 1.0) / (j - k + 1) + len(set(overallenglist).intersection(range(random_seed1, random_seed2+1)))*1.0/enlen 
                  	newenglishsentence[random_seed1:random_seed2+1] = [x.upper() for x in newenglishsentence[random_seed1: random_seed2+1]]
                  	newsentence = [' '.join(engsentence[0:max(0,random_seed1)]), hindisegment, ' '.join(engsentence[random_seed2+1:])]
                
                  	with open("/tmp/output_brute_top3_raj_contiguous2.csv", 'a') as csvfile:
                        	fieldnames = ['English', 'Hindi', 'Segment English','Greedy Hindi', 'Mapping', 'Score']
                        	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        	#writer.writeheader()
                        	writer.writerow({fieldnames[0]: ' '.join(engsentence),  fieldnames[1]: ' '.join(hindisentence), fieldnames[2]: ' '.join(newenglishsentence), fieldnames[3]: ' '.join(newsentence), fieldnames[4]: mapping, fieldnames[5]: val})
		  '''
                  for i in range(hindilen):
			hindiword = hindisentence[i]
			#engindex = np.argmax(history[i,0])
        		engindexlist = history[i,0].argsort()[-3:]
			#if engindex in range(random_seed1, random_seed2+1):
			#list(set(engindexlist).intersection(range(random_seed1, random_seed2)))
			#if engindexlist[2] in range(random_seed1, random_seed2+1) and engindexlist[1] in range(random_seed1, random_seed2+1):
			if len(set(engindexlist).intersection(range(random_seed1, random_seed2))) > 1:
				indexlist.append(i)
				for engindex in engindexlist:
					mapping += hindiword + '->'+ engsentence[engindex]+ ' '
				#newenglishsentence[engindex] = newenglishsentence[engindex].upper()
				hindisegment += ' ' + hindiword
		  
 		  val = score(hindisentence, indexlist, enlen)
		  newenglishsentence[random_seed1:random_seed2+1] = [x.upper() for x in newenglishsentence[random_seed1: random_seed2+1]]
		  newsentence = [' '.join(engsentence[0:max(0,random_seed1)]), hindisegment, ' '.join(engsentence[random_seed2+1:])]
  		  googletranslate_segment = translate(segment, 'en', 'hi')
		  #print(googletranslate_segment)
		  googlenewsentence = [' '.join(engsentence[0:max(0,random_seed1)]), googletranslate_segment, ' '.join(engsentence[random_seed2+1:])]
		  #print("newenglishsentence", newsentence)
                
  		  with open("/tmp/output_contituent_SR_PCFG.csv", 'ab') as csvfile:
   			fieldnames = ['English', 'Hindi', 'Segment English','Greedy Hindi', 'Mapping', 'Score', 'Google','segment']
   			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
   			#writer.writeheader()
   			writer.writerow({fieldnames[0]: ' '.join(engsentence),  fieldnames[1]: ' '.join(hindisentence), fieldnames[2]: ' '.join(newenglishsentence), fieldnames[3]: ' '.join(newsentence), fieldnames[4]: mapping, fieldnames[5]: val, fieldnames[6]: ' '.join(googlenewsentence), fieldnames[7]: segmentname[name]}) 
	
