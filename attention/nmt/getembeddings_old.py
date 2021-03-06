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

def score(hindisentence, hindisegments):
	#engsentecne -> list of tokens in engsentence
	#engsegments -> list of lists for contiguous segments
	#hindisentence -> list of tokens in hindi sentence
	#hindisegments -> index for segments
	
	discontinuous = [] 
	contiguous = []

	lencont = 0
        hilen = len(hindisentence)
	for i in range(hilen):
		if i in hindisegment:
			lencont += 1
			if lendis != 0:
				discontinuous.append(lendis)
				lendis = 0
		else:
			lendis += 1
			if lencont != 0:
				contigous.append(lencont)
				lencont = 0 

	contiguous = [(len(x)+0.0)/hilen for x in contiguous]
	discontinuous = [(len(x)+0.0)/hilen for x in discontinuous]

	return (np.prod(contiguous) / np.prod(discontiguous))

	

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
  random_seed1 = 0
  random_seed2 = 0
  with open(hparams.train_prefix+'.en','r') as f:
  	engsentence = f.read().split()
  with open(hparams.train_prefix+'.hi','r') as f:
  	hindisentence = f.read().split()

  #length = len(hindisentence)
  length = len(engsentence)
  while(abs(random_seed1 - random_seed2) <2):
    random_seed1 = random.randint(0, length-1)
    random_seed2 = random.randint(0, length-1)
  if random_seed1 > random_seed2:
	temp = random_seed1
	random_seed1 = random_seed2
	random_seed2 = temp

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
  print("randomseed", random_seed1, random_seed2)
  for i in range(hindilen):
	hindiword = hindisentence[i]
	engindex = np.argmax(history[i,0])

        	
	if engindex in range(random_seed1, random_seed2+1):
		mapping += hindiword + '->'+ engsentence[engindex]
		#newenglishsentence[engindex] = newenglishsentence[engindex].upper()
		hindisegment += ' ' + hindiword
  
  #print("segment", hindisegment)
  newenglishsentence[random_seed1:random_seed2+1] = [x.upper() for x in newenglishsentence[random_seed1: random_seed2+1]]  
  print("newenglishsentence", newenglishsentence)
  newsentence = [' '.join(engsentence[0:max(0,random_seed1)]), hindisegment, ' '.join(engsentence[random_seed2+1:])]
  print("newenglishsentence", newsentence)		
  with open("/tmp/output_dev.csv", 'a') as csvfile:
   fieldnames = ['English', 'Hindi', 'Segment English','Greedy Hindi', 'Mapping']
   writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
   #writer.writeheader()
   writer.writerow({fieldnames[0]: ' '.join(engsentence),  fieldnames[1]: ' '.join(hindisentence), fieldnames[2]: ' '.join(newenglishsentence), fieldnames[3]: ' '.join(newsentence), fieldnames[4]: mapping})
  '''
  while(1):
     #try:
  	#step_result = loaded_model.getembeddings(sess)
  	#encoder_outputs, decoder_outputs, encoder_inputs, decoder_inputs = step_result
  	print("input embeddings and output embedings")
  	#print(encoder_outputs.shape)
	print(encoder_inputs.shape)
        print(decoder_inputs.shape)
  	print(decoder_outputs.rnn_output.shape)
     #   break
     #except:
     #   print("except")
	break
  #, decoder_outputs.shape)
  '''
