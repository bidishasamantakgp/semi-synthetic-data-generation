import sys
import subprocess
import numpy as np

with open(sys.argv[1],'r') as fen:
	enlines = fen.readlines()

#enlinesnew = np.random.choice(enlines, 5000)
#enlines = enlinesnew

with open(sys.argv[3], 'a') as fw:
	for lines in enlines:
		fw.write(lines.strip()+"\n")

for i in range(len(enlines)):
        #'''
	tokens = enlines[i].split("|||")
	engsentence = tokens[0]
	hindisentence = tokens[1]
	#'''

	#engsentence = enlines[i].strip()
	with open(sys.argv[2], 'a') as f:
		f.write("=====\n\n")
        command = "java -cp \"./*\"  edu.stanford.nlp.parser.shiftreduce.demo.ShiftReduceDemo -model edu/stanford/nlp/models/srparser/englishSR.ser.gz -filename "+"\""+engsentence+"\" >>" + sys.argv[2]
	#command = "java -cp \"./*\"  edu.stanford.nlp.parser.shiftreduce.demo.ShiftReduceDemo -model edu/stanford/nlp/models/srparser/englishSR.ser.gz -filename "+"\""+enlines[i]+"\" >> sr_mod.out"
	#command = "nohup python -m nmt.nmt --attention=scaled_luong --src=en --tgt=hi --vocab_prefix=/tmp/nmt_data_en_hi_devnagari/vocab --train_prefix=/tmp/nmt_data_en_hi_devnagari/train --dev_prefix=/tmp/nmt_data_en_hi_devnagari/test --test_prefix=/tmp/nmt_data_en_hi_devnagari/test --out_dir=/tmp/nmt_model_en_hi_devnagari/bidirectional/ --num_train_steps=7000 --steps_per_stats=100 --num_layers=4 --num_units=128 --dropout=0.2 --metrics=bleu --encoder_type=bi >> /tmp/nmt_model_en_hi_devnagari/bidirectional/log.out"
        #command = "python -m nmt.nmt --attention=scaled_luong --src=en --tgt=hi --vocab_prefix=/tmp/nmt_data_en_hi/vocab1 --train_prefix=/tmp/nmt_data_en_hi/test1 --out_dir=/tmp/nmt_model_en_hi_parallel/bidirectional --num_train_steps=10000 --steps_per_stats=100 --num_layers=4 --num_units=128 --dropout=0.2 --metrics=bleu --encoder_type=bi --get_embeddings=True >> /tmp/log_attention_bah.out"
	process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
	process.wait()
	break
	'''
	command = "java -cp \"*\" edu.stanford.nlp.parser.lexparser.LexicalizedParser edu/stanford/nlp/models/lexparser/englishPCFG.caseless.ser.gz file.txt >> caseless.out"
	process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        process.wait()
        '''
	#command = "java -mx200m -cp \"*\" edu.stanford.nlp.parser.lexparser.LexicalizedParser -retainTmpSubcategories -originalDependencies -outputFormat \"penn\" -outputFormatOptions \"basicDependencies\" edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz file.txt >> PCFG_mod.out"
	#process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        #process.wait()	
	
	
