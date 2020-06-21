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
n_features = np.arange(100,5000,100)
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
		self.accuracy_valid = 0
		self.accuracy_test = 0
		self.weighted_f1_score = []
		self.model = []
		
	def accuracy_summary(self, pipeline, n_features):
		
		t0 = time()
		sentiment_fit = pipeline.fit(self.x_train, self.y_train)
		y_pred = sentiment_fit.predict(self.x_valid)
		accuracy = accuracy_score(self.y_valid, y_pred)
		y_pred_test = sentiment_fit.predict(self.x_test)
		accuracy_test = accuracy_score(self.y_test, y_pred_test)
		weighted_f1_score = f1_score(self.y_test, y_pred_test, average="weighted")
		if accuracy > self.accuracy_valid:
			self.accuracy_valid = accuracy
			self.model.clear()
			self.weighted_f1_score.clear()
			self.model.append(sentiment_fit)
			self.accuracy_test = accuracy_test
			self.weighted_f1_score.append((n_features, weighted_f1_score))
		elif accuracy == self.accuracy_valid:
			if accuracy_test > self.accuracy_test:
				self.model.clear()
				self.weighted_f1_score.clear()
				self.model.append(sentiment_fit)
				self.accuracy_test = accuracy_test
				self.weighted_f1_score.append((n_features, weighted_f1_score))
			elif accuracy_test == self.accuracy_test:
				self.model.append(sentiment_fit)
				self.weighted_f1_score.append((n_features, weighted_f1_score))
		train_valid_time = time() - t0
		
		return accuracy, train_valid_time
		
	def nfeature_accuracy_checker(self, vectorizer=cvec, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=lr):
		result = []
		self.accuracy_valid = 0
		self.accuracy_test = 0
		self.model.clear()
		self.weighted_f1_score.clear()
		if isinstance(n_features, (int, float, complex)) and not isinstance(n_features, bool):
			vectorizer.set_params(min_df = 2, stop_words=stop_words, max_features=n_features, ngram_range=ngram_range)
			checker_pipeline = Pipeline([
				('vectorizer', vectorizer),
				('classifier', classifier)
			])
			nfeature_accuracy,tt_time = self.accuracy_summary(checker_pipeline, n_features)
			
			print("Validation result for {} features".format(n_features))
			print("accuracy_valid score: {0:.2f}%".format(self.accuracy_valid*100))
			print("accuracy_test score: {0:.2f}%".format(self.accuracy_test*100))
			print("f1_score: {0:.2f}%".format(self.weighted_f1_score[0][1]*100))
			result.append((n_features,nfeature_accuracy,tt_time))
		else:
			for n in n_features:
				vectorizer.set_params(min_df = 2, stop_words=stop_words, max_features=n, ngram_range=ngram_range)
				checker_pipeline = Pipeline([
					('vectorizer', vectorizer),
					('classifier', classifier)
				])
				nfeature_accuracy,tt_time = self.accuracy_summary(checker_pipeline, n)
				result.append((n,nfeature_accuracy,tt_time))
			print("Max accuracy score on valid set: {0:.2f}%".format(self.accuracy_valid*100))
			print("Max accuracy score on test set: {0:.2f}%".format(self.accuracy_test*100))
			for n_feat, weighted_f1_score in self.weighted_f1_score:
				print("Validation result for {} features".format(n_feat))
				print("f1_score: {0:.2f}%".format(weighted_f1_score*100))
				
		return result
		
	def UseCountVectorizer(self, n_features = np.arange(100,5000,100), ngram_range = [1,2,3], use_stop_words = True):
		if not use_stop_words:
			stop_words=get_stop_words('..\data\stopwords.txt')
			print("WITHOUT STOPWORD")
		else:
			stop_words=None
			print("WITH STOPWORD")
		if type(ngram_range) == list:
			print("\nRESULT FOR UNIGRAM CountVectorizer")
			feature_result_ug = self.nfeature_accuracy_checker(vectorizer=cvec, n_features=n_features, stop_words=stop_words)
			print("\nRESULT FOR BIGRAM CountVectorizer")
			feature_result_bg = self.nfeature_accuracy_checker(vectorizer=cvec,n_features=n_features, stop_words=stop_words, ngram_range=(1, 2))
			print("\nRESULT FOR TRIGRAM CountVectorizer")
			feature_result_tg = self.nfeature_accuracy_checker(vectorizer=cvec,n_features=n_features, stop_words=stop_words, ngram_range=(1, 3))
			nfeatures_plot_tg = pd.DataFrame(feature_result_tg,columns=['nfeatures','validation_accuracy','train_test_time'])
			nfeatures_plot_bg = pd.DataFrame(feature_result_bg,columns=['nfeatures','validation_accuracy','train_test_time'])
			nfeatures_plot_ug = pd.DataFrame(feature_result_ug,columns=['nfeatures','validation_accuracy','train_test_time'])

			plt.figure(figsize=(8,6))
			plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy,label='trigram count vectorizer',linestyle=':', color='royalblue')
			plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy,label='bigram count vectorizer',linestyle=':',color='orangered')
			plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='unigram count vectorizer',linestyle=':',color='gold')
			plt.title("N-gram(1~3) CountVectorizer test result : Accuracy")
			plt.xlabel("Number of features")
			plt.ylabel("Validation set accuracy")
			plt.legend()
			plt.show(block=True)
		else:
			print("\nRESULT FOR {} GRAM CountVectorizer".format(ngram_range))
			feature_result = self.nfeature_accuracy_checker(vectorizer=cvec,n_features=n_features, stop_words=stop_words, ngram_range=(1, ngram_range))
			nfeatures_plot = pd.DataFrame(feature_result,columns=['nfeatures','validation_accuracy','train_test_time'])
			plt.figure(figsize=(8,6))
			plt.plot(nfeatures_plot.nfeatures, nfeatures_plot.validation_accuracy,label='{} gram count vectorizer'.format(ngram_range),linestyle=':', color='royalblue')
			plt.title("N-gram {} CountVectorizer test result : Accuracy".format(ngram_range))
			plt.xlabel("Number of features")
			plt.ylabel("Validation set accuracy")
			plt.legend()
			plt.show(block=True)
		
	def UseTFIDFVectorizer(self,n_features=np.arange(100,5000,100), ngram_range = [1,2,3], use_stop_words = True):
		
		if not use_stop_words:
			stop_words=get_stop_words('..\data\stopwords.txt')
			print("WITHOUT STOPWORD")
		else:
			stop_words=None
			print("WITH STOPWORD")
		if type(ngram_range) == list:
			print("RESULT FOR UNIGRAM TfidfVectorizer")
			feature_result_ugt = self.nfeature_accuracy_checker(vectorizer=tvec, n_features=n_features, stop_words=stop_words)
			print("\nRESULT FOR BIGRAM TfidfVectorizer")
			feature_result_bgt = self.nfeature_accuracy_checker(vectorizer=tvec, n_features=n_features, stop_words=stop_words, ngram_range=(1, 2))
			print("\nRESULT FOR TRIGRAM TfidfVectorizer")
			feature_result_tgt = self.nfeature_accuracy_checker(vectorizer=tvec,n_features=n_features, stop_words=stop_words, ngram_range=(1, 3))

			nfeatures_plot_tgt = pd.DataFrame(feature_result_tgt,columns=['nfeatures','validation_accuracy','train_test_time'])
			nfeatures_plot_bgt = pd.DataFrame(feature_result_bgt,columns=['nfeatures','validation_accuracy','train_test_time'])
			nfeatures_plot_ugt = pd.DataFrame(feature_result_ugt,columns=['nfeatures','validation_accuracy','train_test_time'])

			plt.figure(figsize=(8,6))
			plt.plot(nfeatures_plot_tgt.nfeatures, nfeatures_plot_tgt.validation_accuracy,label='trigram tfidf vectorizer',color='royalblue')
			plt.plot(nfeatures_plot_bgt.nfeatures, nfeatures_plot_bgt.validation_accuracy,label='bigram tfidf vectorizer',color='orangered')
			plt.plot(nfeatures_plot_ugt.nfeatures, nfeatures_plot_ugt.validation_accuracy, label='unigram tfidf vectorizer',color='gold')
			plt.title("N-gram(1~3) TFIDF Vectorizer test result : Accuracy")
			plt.xlabel("Number of features")
			plt.ylabel("Validation set accuracy")
			plt.legend()
			plt.show(block=True)
		else:
			print("\nRESULT FOR {} GRAM TfidfVectorizer".format(ngram_range))
			feature_result = self.nfeature_accuracy_checker(vectorizer=tvec,n_features=n_features, stop_words=stop_words, ngram_range=(1, ngram_range))
			nfeatures_plot = pd.DataFrame(feature_result,columns=['nfeatures','validation_accuracy','train_test_time'])
			plt.figure(figsize=(8,6))
			plt.plot(nfeatures_plot.nfeatures, nfeatures_plot.validation_accuracy,label='{} gram count vectorizer'.format(ngram_range), color='royalblue')
			plt.title("N-gram {} TfidfVectorizer test result : Accuracy".format(ngram_range))
			plt.xlabel("Number of features")
			plt.ylabel("Validation set accuracy")
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