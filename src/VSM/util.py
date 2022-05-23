import sys
import numpy as np
import nltk

#http://www.scipy.org/
try:
	from numpy import dot
	from numpy.linalg import norm
except:
	print("Error: Requires numpy from http://www.scipy.org/. Have you installed scipy?")
	sys.exit() 

def removeDuplicates(list):
	""" remove duplicates from a list """
	return set((item for item in list))


def cosine(vector1, vector2):
	""" related documents j and q are in the concept space by comparing the vectors :
		cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
	return float(dot(vector1,vector2) / (norm(vector1) * norm(vector2)+0.001))

def euclidean(vector1, vector2):
	return norm(vector1 - vector2)

# keep only none and verb in a list of words
def filterNnV( words ):
	words_pos = nltk.pos_tag(words)
	NnV = [w[0] for w in words_pos if 'NN' in w[1] or 'VB' in w[1]]
	# print(NnV)
	return NnV

def is_chinese(s):
    for _char in s:
        if  '\u4e00' <= _char <= '\u9fa5':
            return True
    return False
