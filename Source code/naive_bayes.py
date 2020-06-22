def CreateWordDict(df):
	'''Count the number of time a word appears in all the sentence'''

	word_dict = {'word':{'Fear':0,'Enjoyment':0,'Other':0,'Disgust':0,'Anger':0,'Sadness':0,'Surprise':0}}

	for sentence, emotion in zip(df['Sentence'],df['Emotion']):
		word_list = sentence.split(' ')
		for word in word_list:
			if word in word_dict.keys():
				word_dict[word][emotion] += 1
			else:
				word_dict[word] = dict(word_dict['word'])
				word_dict[word][emotion] +=1
	del word_dict['word']
	return word_dict
	
def CountFeature(word_dict):
	'''Count number of word in each emotion and total vocabulary in all class'''

	count_feature = {}
	total_feature = 0
	for key, val in word_dict.items():
		for emo, count in val.items():
			if emo in count_feature.keys():
				count_feature[emo] += count
			else:
				count_feature[emo] = count
		total_feature +=1
	return count_feature, total_feature
	
def ProbabilityOfWord(word_dict):
	'''Calculate the P(word|emotion) of each word in the word_dict'''
	
	count_feature, total_feature = CountFeature(word_dict)
	prob_dict = {}
	for word, emo_count in word_dict.items():
		word_prob = {} #{'word':{'Fear':0.02, 'Enjoyment':0.03},...}
		for emo, count in emo_count.items():
			prob = (count + 1) / (count_feature[emo] + total_feature) 
			word_prob[emo] = prob
		prob_dict[word] = word_prob
	return prob_dict

def ProbabilityOfEmotion(df):
	'''Calculate the P(emotion) for each emotion in the corpus'''
	
	emo_prob = {}
	for emo in df.Emotion.unique():
		count = len(df[df.Emotion==emo])
		prob = count/len(df)
		emo_prob[emo] = prob
	return emo_prob
	
def ProbabilityOfSentence(train_df, test_df):
	'''Calculate the P(emotion|sentence) = P(sentence|emotion)*P(emotion) for each sentence in the corpus'''
	
	sentence_prob = {}
	word_dict = CreateWordDict(train_df)
	emo_prob_dict = ProbabilityOfEmotion(train_df)
	prob_dict = ProbabilityOfWord(word_dict)
	count_feature, total_feature = CountFeature(word_dict)
	sentence_predict = []
	for sentence in test_df.Sentence:
		word_list = sentence.split(' ')
		for emo, emo_prob in emo_prob_dict.items():
			probability = emo_prob
			for word in word_list:
				if word not in prob_dict.keys():
					word_prob = 1 / (count_feature[emo] + total_feature)
				else:
					word_prob = prob_dict[word][emo]
				probability *= word_prob
			sentence_prob[emo]=probability
		for k,v in sentence_prob.items():
			if v == max(sentence_prob.values()):  
				sentence_predict.append(k)

	answer_df = test_df.copy()
	answer_df['Predict Label'] = sentence_predict
	return answer_df
