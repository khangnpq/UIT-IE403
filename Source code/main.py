import pandas as pd
import load_data
import reduce_emoji 
import spelling_correction
import word_segmentation
import word_statistics
import logistic_regression
import convolutional_nn
from keras.models import load_model
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def EncodeLabel(df):
	
	mapping_df = pd.read_excel('../data/le_name_mapping.xlsx')
	mapping_dict ={}
	for key in mapping_df['Encoded Label']:
		mapping_dict[mapping_df.loc[key][0]] = key
	new_df = df.copy()
	emo_list=[]
	for emo in new_df.Emotion:
		emo_list.append(mapping_dict[emo])
	new_df.Emotion = emo_list
	return new_df
def main():
	data_path = '../data/raw data'
	
	train_df, valid_df, test_df = load_data.LoadData(data_path)
	#raw_data_word_statistic = word_statistics.WordStatistics(train_df, valid_df, test_df)
	#raw_data_word_statistic.to_excel('../output/statistics/raw_data_word_statistic.xlsx') #Make excel file
	
	train_reduced, valid_reduced, test_reduced = reduce_emoji.ReduceEmoji(train_df, valid_df, test_df)
	#reduced_emoji_data_word_statistic = word_statistics.WordStatistics(train_reduced, valid_reduced, test_reduced)
	#reduced_emoji_data_word_statistic.to_excel('../output/statistics/reduced_emoji_data_word_statistic.xlsx') #Make excel file
	
	train_spelling, valid_spelling, test_spelling = spelling_correction.SpellingCorrection(train_reduced, valid_reduced, test_reduced)
	#spelling_correction_data_word_statistic = word_statistics.WordStatistics(train_spelling, valid_spelling, test_spelling)
	#spelling_correction_data_word_statistic.to_excel('../output/statistics/spelling_correction_data_word_statistic.xlsx') #Make excel file
	
	train_segmented, valid_segmented, test_segmented = word_segmentation.WordSegmentation(train_spelling, valid_spelling, test_spelling)
	#segmented_data_word_statistic = word_statistics.WordStatistics(train_segmented, valid_segmented, test_segmented)
	#segmented_data_word_statistic.to_excel('../output/statistics/segmented_data_word_statistic.xlsx') #Make excel file
	print('Which algorithm do you want to use?')
	prompt = int(input(f'[1]: Logistic Regression\n[2]: Convolutional Neural Network\n[3]: Test on built model\n[1/2/3] '))
	if prompt == 1:
		logistic_regression.UseLogisticRegression(train_spelling, valid_spelling, test_spelling)
	elif prompt == 2:
		convolutional_nn.UseConvolutionalNN(train_segmented, valid_segmented, test_segmented)
	else:
		logistic_regression_model = pickle.load(open('../output/best_logistic_regression_model.h1', 'rb'))
		cnn_model = load_model('../output/cnn_model.h1')
		test_what = int(input(f'Do you want to test on sentence [1] or excel file contains sentence [2] ? [1/2] '))
		if test_what == 1:
			x_sentence = input(f'Type in the sentence that you need to test: ')
			test_df = pd.DataFrame(list(zip([x_sentence], ['Other'])), columns =['Sentence', 'Emotion'])
		else:
			test_df = pd.read_excel('../data/test_set.xlsx')
		test_reduced = reduce_emoji.ReduceEmoji(test_df)
		test_spelling = spelling_correction.SpellingCorrection(test_reduced)
		test_segmented = word_segmentation.WordSegmentation(test_spelling)
		sgemented_encode = EncodeLabel(test_segmented)
		
		#LogisticRegression
		x_test_LR = test_spelling.Sentence
		y_test_LR = test_spelling.Emotion
		y_pred_LR = logistic_regression.TestCase(x_test_LR, y_test_LR) #Calling function
		
		#CNN
		#tokenizer = cnn_model.tokenizer
		
		y_pred_CNN = convolutional_nn.TestCase(sgemented_encode) # Calling function
		
		if test_what == 2:
			test_set_output_LR = test_spelling.copy()
			test_set_output_LR['Predict Emotion'] = y_pred_LR
			test_set_output_LR.to_excel('../output/test_set_output_LR.xlsx')
			
			test_set_output_CNN = sgemented_encode.copy()
			test_set_output_CNN['Predict Emotion'] = y_pred_CNN
			test_set_output_CNN.to_excel('../output/test_set_output_CNN.xlsx')
		else:
			print('Multinominal Logistic Regression model: {}'.format(y_pred_LR[0]))
			print('Convolutional Neural Network model: {} '.format(y_pred_CNN[0]))
if __name__ == '__main__':
	main()
			