from nltk import Tree
import sys


def corpus2trees(text):
	""" Parse the corpus and return a list of Trees """
	rawparses = text.split("\n\n")
	trees = []
 	listtrees = []
	for rp in rawparses:
		if not rp.strip():
			continue
		if rp.strip() == "=====":
			listtrees.append(trees)
			trees = []
			continue
 
		t = Tree.fromstring(rp)
		trees.append(t)
 
	return listtrees

def getsegments(listtree):
	segments = []
	for tree in listtree:
		for s in tree.subtrees():
			if s.label()=='VP' or s.label()=='NP':
				segments.append(' '.join([x for x in s.leaves() if x != ","]))	
	
	return segments

if __name__=="__main__":
	f = open(sys.argv[1])
	if not isinstance(f, basestring):
		content = f.read()
	trees =  corpus2trees(content)
	
	for t in trees:
		with open(sys.argv[2], 'a') as f:
			f.write("["+','.join(["\""+x+"\"" for x in getsegments(t)])+"]\n")
			#print getsegments(t)
