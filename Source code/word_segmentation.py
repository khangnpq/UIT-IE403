# This Python file uses the following encoding: utf-8
from pyvi import ViTokenizer
import pandas as pd

def WordSegmentation(*args):
	''' segmenting words in sentence in order to get the full meaning'''
	
	output = []
	for df in args:
		sentence_list = []
		
		for sentence in df.Sentence:
			segmented_sentence = ViTokenizer.tokenize(sentence)
			sentence_list.append(segmented_sentence)
		new_df = df.copy()
		new_df.Sentence = sentence_list
		output.append(new_df)
	if len(output) == 1:
		return output[0]
	else:
		return output