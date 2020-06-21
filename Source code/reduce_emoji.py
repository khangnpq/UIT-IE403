# This Python file uses the following encoding: utf-8
import emoji
import regex
import re
import pandas as pd

def LoadRuleDict(file_path):
	'''Load the dictionary file contain rules to change emoticon (such as :D :p ) to readable form'''

	df = pd.read_excel(file_path, encoding="ISO-8859-1")
	the_dict = df.to_dict()
	rule_dict = {}
	for k, v in the_dict.items():
		for key, elem in v.items():
			if type(elem) == str:
				rule_dict[elem] = k
	return rule_dict

def RemoveRepeatedEmoji(data):
	'''Remove emoji that appear more than two times in a sentence'''
	emoji_list = []
	rm_dict = {}
	for word in data:
		if any(char in emoji.UNICODE_EMOJI for char in word):
			if word not in emoji_list: #distinct list of emoji
				emoji_list.append(word)
				rm_dict[word] = 0 #for later removal of duplicates emoji
			else:
				rm_dict[word] += 1
	for key, val in rm_dict.items():
		for i in range (0,val): # remove each emoji in the text if it appears more than 1 time
			data.remove(key)
				
	return data

def ReduceEmoji(*args, keep_repeated_emoji = False):
	'''Remove repeated special characters, change emoticon and/or emoji in sentence'''
	
	
	#args = list(args)
		
	output = []
		
	for df in args:
		sentence_list = []
		file_path = '../data/emoticon-emoji.xlsx'
		rule_dict = LoadRuleDict(file_path)
		to_remove = "_.,-!?:()[]{}aeiouchmlprwyz<3"
		remove_list = '(")[]#/\~;*@+=<>'
		
		for text in df.Sentence:
		
			
			word_list = []
			change = False #Change the emoticon in rule_dict to it right form
			
			text =  "".join(text[i] for i in range(len(text)) if i == 0 or not (text[i-1] == text[i] and text[i] in to_remove)) #remove duplicate word, non-character word
			text = re.sub(' +', ' ',text) # remove multiple space character
			
			for word in text.split(" "):
				for key in rule_dict:
					if key in word:
						word = rule_dict[key]
						change = True
				word_list.append(word)
			if change:
				text = " ".join(word_list)	

			data = regex.findall(r'\X', text) #list of every character in the text string
			flags = regex.findall(u'[\U0001F1E6-\U0001F1FF]', text)	#Get all flag emoji
			if not keep_repeated_emoji:
				data = RemoveRepeatedEmoji(data)
			new_text = emoji.demojize("".join(data)) #text string that has removed duplicate emoji
			final_text = "".join(new_text[i] if new_text[i] not in remove_list else '' for i in range(len(new_text)))
			sentence_list.append(final_text)
		new_df = df.copy()
		new_df.Sentence = sentence_list	#replace Sentence column
		output.append(new_df)
	if len(output) == 1:
		return output[0]
	else:
		return output