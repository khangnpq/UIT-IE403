# This Python file uses the following encoding: utf-8
import pandas as pd
def WordStatistics(*args):
	''' segmenting words in sentence in order to get the full meaning'''
	if len(args) > 1:
		df = pd.concat(args)
	count_list = []
	for df in args:
		sentence_list = []
		for emotion in df.Emotion.unique(): #For each emotion
			sentence_list = list(df[df.Emotion == emotion].Sentence) #make a list of sentence of that emo
			word_dict = {} #empty dict contain word as key and number of times it appears in the text as value.
			
			for sentence in sentence_list:
				for word in sentence.split():
					if word in word_dict:
						word_dict[word] += 1 #if the word is already in the dict, increase counter
					else:
						word_dict[word] = 1 #otherwise add it to the dict
			count_list.append(word_dict)
		count_df = pd.DataFrame(count_list) #make a dataframe from list
		count_df = count_df.transpose() #switch row with column
		count_df.columns = list(df.Emotion.unique()) #add emotion as column name
		count_df = count_df.fillna(0) #replace NaN with 0 - that value dont exist in this emotion
		count_df = count_df.round(decimals=0).astype(object) #remove the '.0' trailing e.g: 5.0, 4.0...
	if len(count_df) == 1:
		return count_df[0]
	else:
		return count_df