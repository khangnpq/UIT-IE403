# -*- coding: utf-8 -*-
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
import os, os.path
import load_data
import operator
import pickle
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix

plt.style.use('fivethirtyeight')

#data_dir = r"D:\1.UIT\Nam 4\Hoc ky 2\Khai thac du lieu truyen thong xa hoi - IE403.K21\3. Implementing\src\data\raw data"
#train_dir = os.path.join(data_dir, "train_nor_811" + "." + "xlsx")
#test_dir = os.path.join(data_dir, "test_nor_811" + "." + "xlsx")
#valid_dir = os.path.join(data_dir, "valid_nor_811" + "." + "xlsx")
n_features = np.arange(1200,6100,100)
cvec = CountVectorizer()
tvec = TfidfVectorizer()
lr = LogisticRegression(multi_class='multinomial', solver='newton-cg',max_iter= 250)
le = preprocessing.LabelEncoder()

def get_stop_words(stop_file_path):
	"""load stop words """
	
	with open(stop_file_path, 'r', encoding="utf-8") as f:
		stopwords = f.readlines()
		stop_set = list(m.strip() for m in stopwords)
		return stop_set

def EncodeLabel(df):
	le.fit(df['Emotion'])
	le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
	print(le_name_mapping)

class Logistic_Regression:
	def __init__(self, x_train, y_train, x_valid, y_valid, x_test, y_test):
	
		self.x_train = x_train
		self.x_valid = x_valid
		self.x_test = x_test
		self.y_train = y_train
		self.y_valid = y_valid
		self.y_test = y_test
		self.accuracy_valid = []
		self.weighted_f1_score_valid = 0
		self.accuracy_test = []
		self.weighted_f1_score_test = 0
		self.model = []
		self.accuracy = []
	def accuracy_summary(self, pipeline, n_features):
		
		accuracy_dict = {}
		t0 = time()
		
		sentiment_fit = pipeline.fit(self.x_train, self.y_train) #Train model
		y_pred_valid = sentiment_fit.predict(self.x_valid) #Predict on valid set	
		accuracy_valid = accuracy_score(self.y_valid, y_pred_valid) #Accuracy on valid set
		weighted_f1_score_valid = f1_score(self.y_valid, y_pred_valid, average="weighted") #Weighted f1-score on valid set 
		y_pred_test = sentiment_fit.predict(self.x_test) #Predict on test set
		accuracy_test = accuracy_score(self.y_test, y_pred_test) #Accuracy on test set
		weighted_f1_score_test = f1_score(self.y_test, y_pred_test, average="weighted") #Weighted f1-score on test set
		accuracy_dict['valid'] = accuracy_valid
		accuracy_dict['test'] = accuracy_test
		accuracy_dict['n_features'] = n_features
		if weighted_f1_score_valid > self.weighted_f1_score_valid:
			self.model.clear()
			self.accuracy.clear()
			self.model.append(sentiment_fit)
			self.accuracy.append(accuracy_dict)
			self.weighted_f1_score_valid = weighted_f1_score_valid
			self.weighted_f1_score_test = weighted_f1_score_test
		elif weighted_f1_score_valid == self.weighted_f1_score_valid:
			if weighted_f1_score_test > self.weighted_f1_score_test:
				self.model.clear()
				self.accuracy.clear()			
				self.model.append(sentiment_fit)
				self.accuracy.append(accuracy_dict)
				self.weighted_f1_score_test = weighted_f1_score_test
			elif weighted_f1_score_test == self.weighted_f1_score_test:
				self.model.append(sentiment_fit)
				self.accuracy.append(accuracy_dict)
		train_valid_time = time() - t0
		
		return train_valid_time
		
	def nfeature_accuracy_checker(self, vectorizer=cvec, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=lr, min_df=1):
		result = []
		self.model.clear()
		self.accuracy.clear()
		self.weighted_f1_score_valid = 0
		self.weighted_f1_score_test = 0
		if isinstance(n_features, (int, float, complex)) and not isinstance(n_features, bool):
			vectorizer.set_params(min_df=min_df, stop_words=stop_words, max_features=n_features, ngram_range=ngram_range)
			checker_pipeline = Pipeline([
				('vectorizer', vectorizer),
				('classifier', classifier)
			])
			tt_time = self.accuracy_summary(checker_pipeline, n_features)
			
			print("Validation result for {} features".format(n_features))
			print("Accuracy score on valid set: {0:.2f}%".format(self.accuracy[0]['valid']*100))
			print("Weighted F1-score on valid set: {0:.2f}%".format(self.weighted_f1_score_valid*100))
			print("Accuracy score on test set:: {0:.2f}%".format(self.accuracy[0]['test']*100))
			print("Weighted F1-score on test set: {0:.2f}%".format(self.weighted_f1_score_test*100))
			result.append((n_features,self.weighted_f1_score_test,tt_time))
		else:
			for n in n_features:
				vectorizer.set_params(min_df=min_df, stop_words=stop_words, max_features=n, ngram_range=ngram_range)
				checker_pipeline = Pipeline([
					('vectorizer', vectorizer),
					('classifier', classifier)
				])
				tt_time = self.accuracy_summary(checker_pipeline, n)
				result.append((n,self.weighted_f1_score_test,tt_time))
			print("Highest weighted f1-score on valid set: {0:.2f}%".format(self.weighted_f1_score_valid*100))
			print("Highest weighted f1-score on test set with that model: {0:.2f}%".format(self.weighted_f1_score_test*100))
			for accuracy_dict in self.accuracy:
				print("Validation result for {} features".format(accuracy_dict['n_features']))
				print("Accuracy score on valid set: {0:.2f}%".format(accuracy_dict['valid']*100))
				print("Accuracy score on test set: {0:.2f}%".format(accuracy_dict['test']*100))
				
		return result
		
	def UseCountVectorizer(self, n_features = np.arange(1200,6100,100), ngram_range = [1,2,3], use_stop_words = True, min_df=1):
		if not use_stop_words:
			stop_words=get_stop_words('..\data\stopwords.txt')
			print("WITHOUT STOPWORD")
		else:
			stop_words=None
			print("WITH STOPWORD")
		if type(ngram_range) == list:
			print("\nRESULT FOR UNIGRAM CountVectorizer")
			feature_result_ug = self.nfeature_accuracy_checker(vectorizer=cvec, n_features=n_features, stop_words=stop_words, min_df=min_df)
			print("\nRESULT FOR BIGRAM CountVectorizer")
			feature_result_bg = self.nfeature_accuracy_checker(vectorizer=cvec,n_features=n_features, stop_words=stop_words, ngram_range=(1, 2), min_df=min_df)
			print("\nRESULT FOR TRIGRAM CountVectorizer")
			feature_result_tg = self.nfeature_accuracy_checker(vectorizer=cvec,n_features=n_features, stop_words=stop_words, ngram_range=(1, 3), min_df=min_df)
			nfeatures_plot_tg = pd.DataFrame(feature_result_tg,columns=['nfeatures','Weighted_F1_Score','train_test_time'])
			nfeatures_plot_bg = pd.DataFrame(feature_result_bg,columns=['nfeatures','Weighted_F1_Score','train_test_time'])
			nfeatures_plot_ug = pd.DataFrame(feature_result_ug,columns=['nfeatures','Weighted_F1_Score','train_test_time'])

			plt.figure(figsize=(8,6))
			plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.Weighted_F1_Score,label='trigram count vectorizer',linestyle=':', color='royalblue')
			plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.Weighted_F1_Score,label='bigram count vectorizer',linestyle=':',color='orangered')
			plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.Weighted_F1_Score, label='unigram count vectorizer',linestyle=':',color='gold')
			plt.title("N-gram(1~3) CountVectorizer test result : Weighted F1 Score")
			plt.xlabel("Number of features")
			plt.ylabel("Weighted F1-score")
			plt.legend()
			plt.show(block=True)
		else:
			print("\nRESULT FOR {} GRAM CountVectorizer".format(ngram_range))
			feature_result = self.nfeature_accuracy_checker(vectorizer=cvec,n_features=n_features, stop_words=stop_words, ngram_range=(1, ngram_range), min_df=min_df)
			nfeatures_plot = pd.DataFrame(feature_result,columns=['nfeatures','Weighted_F1_Score','train_test_time'])
			plt.figure(figsize=(8,6))
			plt.plot(nfeatures_plot.nfeatures, nfeatures_plot.Weighted_F1_Score,label='{} gram count vectorizer'.format(ngram_range),linestyle=':', color='royalblue')
			plt.title("N-gram {} CountVectorizer test result : Weighted F1 Score".format(ngram_range))
			plt.xlabel("Number of features")
			plt.ylabel("Weighted F1-score")
			plt.legend()
			plt.show(block=True)
		
	def UseTFIDFVectorizer(self,n_features=np.arange(100,5000,100), ngram_range = [1,2,3], use_stop_words = True, min_df=1):
		
		if not use_stop_words:
			stop_words=get_stop_words('..\data\stopwords.txt')
			print("WITHOUT STOPWORD")
		else:
			stop_words=None
			print("WITH STOPWORD")
		if type(ngram_range) == list:
			print("RESULT FOR UNIGRAM TfidfVectorizer")
			feature_result_ugt = self.nfeature_accuracy_checker(vectorizer=tvec, n_features=n_features, stop_words=stop_words, min_df=min_df)
			print("\nRESULT FOR BIGRAM TfidfVectorizer")
			feature_result_bgt = self.nfeature_accuracy_checker(vectorizer=tvec, n_features=n_features, stop_words=stop_words, ngram_range=(1, 2), min_df=min_df)
			print("\nRESULT FOR TRIGRAM TfidfVectorizer")
			feature_result_tgt = self.nfeature_accuracy_checker(vectorizer=tvec,n_features=n_features, stop_words=stop_words, ngram_range=(1, 3), min_df=min_df)

			nfeatures_plot_tgt = pd.DataFrame(feature_result_tgt,columns=['nfeatures','Weighted_F1_Score','train_test_time'])
			nfeatures_plot_bgt = pd.DataFrame(feature_result_bgt,columns=['nfeatures','Weighted_F1_Score','train_test_time'])
			nfeatures_plot_ugt = pd.DataFrame(feature_result_ugt,columns=['nfeatures','Weighted_F1_Score','train_test_time'])

			plt.figure(figsize=(8,6))
			plt.plot(nfeatures_plot_tgt.nfeatures, nfeatures_plot_tgt.Weighted_F1_Score,label='trigram tfidf vectorizer',color='royalblue')
			plt.plot(nfeatures_plot_bgt.nfeatures, nfeatures_plot_bgt.Weighted_F1_Score,label='bigram tfidf vectorizer',color='orangered')
			plt.plot(nfeatures_plot_ugt.nfeatures, nfeatures_plot_ugt.Weighted_F1_Score, label='unigram tfidf vectorizer',color='gold')
			plt.title("N-gram(1~3) TFIDF Vectorizer test result : Weighted F1 Score")
			plt.xlabel("Number of features")
			plt.ylabel("Weighted F1-score")
			plt.legend()
			plt.show(block=True)
		else:
			print("\nRESULT FOR {} GRAM TfidfVectorizer".format(ngram_range))
			feature_result = self.nfeature_accuracy_checker(vectorizer=tvec,n_features=n_features, stop_words=stop_words, ngram_range=(1, ngram_range), min_df=min_df)
			nfeatures_plot = pd.DataFrame(feature_result,columns=['nfeatures','Weighted_F1_Score','train_test_time'])
			plt.figure(figsize=(8,6))
			plt.plot(nfeatures_plot.nfeatures, nfeatures_plot.Weighted_F1_Score,label='{} gram count vectorizer'.format(ngram_range), color='royalblue')
			plt.title("N-gram {} TfidfVectorizer test result : Weighted F1 Score".format(ngram_range))
			plt.xlabel("Number of features")
			plt.ylabel("Weighted F1-score")
			plt.legend()
			plt.show(block=True)
	def ShowConfusionMatrix(self):
		
		plt.rcParams['figure.figsize'] = [12, 6]
		# Plot non-normalized confusion matrix
		titles_options = [("Confusion matrix, without normalization", None),
						  ("Normalized confusion matrix", 'true')]
		for title, normalize in titles_options:
			disp = plot_confusion_matrix(self.model[0], self.x_test, self.y_test,
										 cmap=plt.cm.Blues,
										 normalize=normalize)
			disp.ax_.set_title(title)
		plt.show()
def UseLogisticRegression(*args):
	'''abcxyz'''
	
	x_train, y_train, x_valid, y_valid, x_test, y_test = load_data.SentenceEmotionAppend(args)
	model = Logistic_Regression(x_train, y_train, x_valid, y_valid, x_test, y_test)
	while True:
		print('Which Vectorizer do you want to use?')
		prompt = int(input(f'1: CountVectorizer; 2:TFIDFVectorizer \n'))
		if prompt == 1:
			use_stop_words = input(f'Do you want to keep stopword? [y/n] ')
			if use_stop_words == 'y':
				use_stop_words = True
			else:
				use_stop_words = False
			ngram = input(f'Do you want to see all ngram_range from 1 to 3? [y/n] ')
			if ngram.lower() == 'n':
				ngram = int(input(f'Please input number of ngram_range you want to see? [1,2,3]'))
				model.UseCountVectorizer(ngram_range = ngram, use_stop_words = use_stop_words)
				show_cfm = input(f'Do you want to see confusion matrix of best n_feature? [y/n] ')
				if show_cfm == 'y':
					model.ShowConfusionMatrix()
				same_setup = input(f'Do you want to check TFIDFVectorizer with same set-up? [y/n] ')
				if same_setup == 'y':
					model.UseTFIDFVectorizer(ngram_range = ngram, use_stop_words = use_stop_words)
					show_cfm = input(f'Do you want to see confusion matrix of best n_feature? [y/n] ')
					if show_cfm == 'y':
						model.ShowConfusionMatrix()
				cont = input(f'Do you want to keep configure parameters [Y] or move on to other algorithm [N]?\n')
				if cont.lower() == 'n':
					break
			else:
				model.UseCountVectorizer(use_stop_words = use_stop_words)
				same_setup = input(f'Do you want to check TFIDFVectorizer with same set-up? [y/n] ')
				if same_setup == 'y':
					model.UseTFIDFVectorizer(use_stop_words = use_stop_words)
				cont = input(f'Do you want to keep configure parameters [Y] or move on to other algorithm [N]?\n')
				if cont.lower() == 'n':
					break
		else:
			use_stop_words = input(f'Do you want to keep stopword? [y/n] ')
			if use_stop_words == 'y':
				use_stop_words = True
			else:
				use_stop_words = False
			ngram = input(f'Do you want to see all ngram_range from 1 to 3? [y/n] ')
			if ngram.lower() == 'n':
				ngram = int(input(f'Please input number of ngram_range you want to see? [1,2,3]'))
				model.UseTFIDFVectorizer(ngram_range = ngram, use_stop_words = use_stop_words)
				pickle.dump(model.model, open('../output/logistic_regression_model.h1', 'wb'))
				show_cfm = input(f'Do you want to see confusion matrix of best n_feature? [y/n] ')
				if show_cfm == 'y':
					model.ShowConfusionMatrix()
				same_setup = input(f'Do you want to check CountVectorizer with same set-up? [y/n] ')
				if same_setup == 'y':
					model.UseCountVectorizer(ngram_range = ngram, use_stop_words = use_stop_words)
					show_cfm = input(f'Do you want to see confusion matrix of best n_feature? [y/n] ')
					if show_cfm == 'y':
						model.ShowConfusionMatrix()
				cont = input(f'Do you want to keep configure parameters [Y] or move on to other algorithm [N]?\n')
				if cont.lower() == 'n':
					break
			else:
				model.UseTFIDFVectorizer(use_stop_words = use_stop_words)
				same_setup = input(f'Do you want to check CountVectorizer with same set-up? [y/n] ')
				if same_setup == 'y':
					model.UseCountVectorizer( use_stop_words = use_stop_words)
				cont = input(f'Do you want to keep configure parameters [Y] or move on to other algorithm [N]?\n')
				if cont.lower() == 'n':
					break

def TestCase(x_test, y_test):
	
	logistic_regression_model = pickle.load(open('../output/logistic_regression_model.h1', 'rb'))
	y_pred = logistic_regression_model[0].predict(x_test)
	#test_accuracy = accuracy_score(y_test, y_pred)
	#weighted_f1_score = f1_score(y_test, y_pred, average="weighted")
	#print("Logistic Regression: ")
	#print("Test {}: {:.2f}".format('accuracy', test_accuracy*100))
	#print("f1-score: {:.2f}%".format(weighted_f1_score*100))
	
	return y_pred