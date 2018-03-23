import sys
import subprocess
import numpy as np

with open(sys.argv[1],'r') as fen:
	enlines = fen.readlines()


enlinesnew = np.random.choice(enlines, 2000)
enlines = enlinesnew
#'''
#sample file to store the samples
with open(sys.argv[4], 'a') as fw:
	for lines in enlines:
		fw.write(lines.strip()+"\n")
#'''

for i in range(len(enlines)):
#for i in range(2):
	tokens = enlines[i].split("|||")
	engsentence = tokens[0]
	hindisentence = tokens[1]

	#engsentence = enlines[i].strip()
        command = "java -cp \"./*\"  edu.stanford.nlp.parser.shiftreduce.demo.ShiftReduceDemo -model edu/stanford/nlp/models/srparser/englishSR.ser.gz -filename "+"\""+engsentence+"\" >" + sys.argv[2]
	process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
	process.wait()

	with open(sys.argv[2], 'a') as f:
                f.write("=====\n\n")
	
	command = "python parsefiles.py "+sys.argv[2] + " "+ sys.argv[3]  
	process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        process.wait()
	#break
	
