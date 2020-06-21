# This Python file uses the following encoding: utf-8
import pandas as pd
import re

def LoadRuleDict(file_path):
	'''Load the dictionary file contain rules to correct spelling error (such as dk, hog ->được, không)'''
	
	df = pd.read_excel(file_path, encoding="ISO-8859-1")
	the_dict = df.to_dict()
	rule_dict = {}
	for k, v in the_dict.items():
		for key, elem in v.items():
			if type(elem) == str:
				rule_dict[elem] = k
				
	return rule_dict
	
def ChangeWord(sentence, the_list, rule_dict):

	word_list = []
	change = False
	new_sentence = sentence
	remove_list = """([:.`~,'"()])""" #List of some special characters to remove
	for word in the_list:
		if word in rule_dict:
			word = rule_dict[word]
			change = True
		word_list.append(word)
		if change:
			new_sentence = " ".join(word_list)
			
	new_sentence = re.sub('([:.,!?()])', r' \1 ', new_sentence) #Add space between punctuation such as ([:.,!?()])
	new_sentence = re.sub('\s{2,}', ' ', new_sentence) #Remove extra space
	new_sentence = ' '.join([w for w in new_sentence.split() if len(w)>1 and w not in remove_list]) #Remove single character on the string
	
	return new_sentence
	
def SpellingCorrection(*args):
	'''Correct spelling error and convert acronyms in various forms  back to their original words'''
	
	output = []
	for df in args:
		sentence_list = []
		file_path = '../data/Loi_ngu_phap.xlsx'
		rule_dict = LoadRuleDict(file_path)
		
		for sentence in df.Sentence:
		
			word_list = []
			unigram_sentence = ChangeWord(sentence, sentence.split(" "), rule_dict)
			one_word_list = unigram_sentence.split(" ")
			two_word_list_1 = [' '.join(one_word_list[x]) for x in range(0, len(one_word_list),2)] #list of every 2 word starting from 1st word
			bigram_sentence_1 = ChangeWord(unigram_sentence, two_word_list_1, rule_dict)
			two_word_list_2 = [bigram_sentence_1.split()[0]]+[' '.join(bigram_sentence_1.split()[x]) for x in range(1, len(bigram_sentence_1.split()),2)] #list of every 2 word starting from 2nd word
			bigram_sentence_2 = ChangeWord(bigram_sentence_1, two_word_list_2, rule_dict)
			one_word_list = bigram_sentence_2.split(" ")
			try:
				three_word_list_1 = [' '.join(one_word_list[x]) for x in range(0, len(one_word_list),3)]
				trigram_sentence_1 = ChangeWord(bigram_sentence_2,three_word_list_1, rule_dict)
				three_word_list_2 = [trigram_sentence_1.split()[0]]+[' '.join(trigram_sentence_1.split()[x]) for x in range(1, len(trigram_sentence_1.split()),3)] #list of every 2 word starting from 2nd word
				trigram_sentence_2 = ChangeWord(trigram_sentence_1, three_word_list_2, rule_dict)
				three_word_list_3 = [trigram_sentence_2.split()[0], trigram_sentence_2.split()[1]]+[' '.join(trigram_sentence_2.split()[x]) for x in range(2, len(trigram_sentence_2.split()),3)] #list of every 2 word starting from 2nd word
				trigram_sentence_3 = ChangeWord(trigram_sentence_2, three_word_list_3, rule_dict)
			except:
				pass #Don't care if sentence have less than 3 words
		
			new_sentence = re.sub(' +', ' ',trigram_sentence_3) # remove multiple space character
			sentence_list.append(new_sentence)
		new_df = df.copy()
		new_df.Sentence = sentence_list
		output.append(new_df)
	
	if len(output) == 1:
		return output[0]
	else:
		return output