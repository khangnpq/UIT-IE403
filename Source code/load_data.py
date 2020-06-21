# -*- coding: utf-8 -*-
import pandas as pd
import os

#data_dir = '../data/raw data"
#train_dir = os.path.join(data_dir, "train_nor_811" + "." + "xlsx")
#test_dir = os.path.join(data_dir, "test_nor_811" + "." + "xlsx")
#valid_dir = os.path.join(data_dir, "valid_nor_811" + "." + "xlsx")

def LoadData(data_path):
	train_path = os.path.join(data_path, "train_nor_811" + "." + "xlsx")
	valid_path = os.path.join(data_path, "valid_nor_811" + "." + "xlsx")
	test_path = os.path.join(data_path, "test_nor_811" + "." + "xlsx")
	
	train_df = pd.read_excel(train_path, encoding="ISO-8859-1",usecols=[1,2])
	valid_df = pd.read_excel(valid_path, encoding="ISO-8859-1",usecols=[1,2])
	test_df = pd.read_excel(test_path, encoding="ISO-8859-1",usecols=[1,2])
	
	return train_df, valid_df, test_df
	
def SentenceEmotionAppend(*args):

	output = []
	for tuples in args:
		for df in tuples:
			x = df.Sentence
			y = df.Emotion
			output.append(x)
			output.append(y)
			
	return output
	