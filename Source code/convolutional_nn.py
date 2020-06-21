from sklearn import preprocessing
from sklearn.metrics import f1_score
from keras.models import Sequential, load_model
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
from gensim.models import Word2Vec
import pandas as pd
import matplotlib.pyplot as plt
import load_data
import numpy as np

plt.style.use('ggplot')
le = preprocessing.LabelEncoder()
tokenizer = Tokenizer(num_words=3500)

def EncodeLabel(*args):
	
	output = []
	mapping_df = pd.read_excel('../data/le_name_mapping.xlsx')
	mapping_dict ={}
	for value in mapping_df['Encoded Label']:
		mapping_dict[mapping_df.loc[value][0]] = value
	for tuples in args:
		for df in tuples:
			new_df = df.copy()
			emo_list=[]
			for emo in new_df.Emotion:
				emo_list.append(mapping_dict[emo])
			new_df.Emotion = emo_list
			output.append(new_df)
	if len(output) == 1:
		return output[0]
	else:
		return output

def create_embedding_matrix(filepath, word_index, embedding_dim):
    
    embeddings_index = {}
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    w2v_model = Word2Vec.load(filepath)
    
    for w in w2v_model.wv.vocab.keys():
        embeddings_index[w] = w2v_model.wv[w]
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def plot_history(history):
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
class Convolutional_NN:
	def __init__(self, x_train, y_train, x_valid, y_valid, x_test, y_test):
	
		self.x_train = x_train
		self.x_valid = x_valid
		self.x_test = x_test
		self.y_train = y_train
		self.y_valid = y_valid
		self.y_test = y_test
		self.embedding_dim = 100
		self.accuracy_valid = 0
		self.accuracy_test = 0
		self.weighted_f1_score = []
		self.model = []
	def GetMaxLen(self):
	
		length = []
		for x in self.x_train:
			length.append(len(x.split()))
		for x in self.x_valid:
			length.append(len(x.split()))
		self.maxlen = max(length)

	def Tokenization(self):
	
		tokenizer.fit_on_texts(self.x_train)
		x_train = tokenizer.texts_to_sequences(self.x_train)
		self.x_train_pad = pad_sequences(x_train, padding='post', maxlen=self.maxlen)
		x_valid = tokenizer.texts_to_sequences(self.x_valid)
		self.x_valid_pad = pad_sequences(x_valid, padding='post', maxlen=self.maxlen)
		x_test = tokenizer.texts_to_sequences(self.x_test)
		self.x_test_pad = pad_sequences(x_test, padding='post', maxlen=self.maxlen)
		self.vocab_size = len(tokenizer.word_index) + 1
		#self.embedding_matrix = create_embedding_matrix('../data/vi.bin', tokenizer.word_index, self.embedding_dim)
		return ''
		
	def TrainModel(self):
	
		self.GetMaxLen()
		self.Tokenization()
		model = Sequential()
		model.add(layers.Embedding(input_dim=self.vocab_size, 
								   output_dim=self.embedding_dim,
								   trainable=True,
								   input_length=self.maxlen))
		model.add(layers.Conv1D(128, 4, activation='relu'))
		model.add(layers.GlobalMaxPool1D())
		model.add(layers.Dense(64, activation='relu'))
		model.add(layers.Dense(7, activation='softmax'))
		model.compile(optimizer='adam',
					  loss='sparse_categorical_crossentropy',
					  metrics=['accuracy'])
		callbacks = [ReduceLROnPlateau(),
			EarlyStopping(patience=2),
			ModelCheckpoint(filepath='../output/cnn_model.h1', save_best_only=True)]
		model.summary()
		self.history = model.fit(self.x_train_pad, self.y_train,
                    epochs=10,
                    verbose=2,
                    validation_data=(self.x_valid_pad, self.y_valid),
                    batch_size=10,
                   callbacks=callbacks)
		plot_history(self.history)
		plt.show()
	
def UseConvolutionalNN(*args):
	'''abcxyz'''

	train_encoded, valid_ecoded, test_encoded = EncodeLabel(args)
	x_train, y_train, x_valid, y_valid, x_test, y_test = load_data.SentenceEmotionAppend((train_encoded, valid_ecoded, test_encoded))
	model = Convolutional_NN(x_train, y_train, x_valid, y_valid, x_test, y_test)
	model.TrainModel()
	#plot_history(model.history)
	cnn_model = load_model('../output/cnn_model.h1')
	train_loss, train_accuracy = cnn_model.evaluate(model.x_train_pad, model.y_train, verbose=False)
	val_loss, val_accuracy = cnn_model.evaluate(model.x_valid_pad, model.y_valid, verbose=False)
	test_loss, test_accuracy = cnn_model.evaluate(model.x_test_pad, model.y_test, verbose=False)
	weighted_f1_score = f1_score(model.y_test, cnn_model.predict_classes(model.x_test_pad), average="weighted")
	print("Training Accuracy: {:.2f}%".format(train_accuracy*100))
	print("Valid {}: {}".format('loss', val_loss))
	print("Valid {}: {:.2f}".format('accuracy', val_accuracy*100))
	print("Test {}: {}".format('loss', test_loss))
	print("Test {}: {:.2f}".format('accuracy', test_accuracy*100))
	print("f1-score: {:.2f}%".format(weighted_f1_score*100))

def TestCase(df):
	
	x_test_CNN = df.Sentence
	x_test_token = tokenizer.texts_to_sequences(x_test_CNN)
	x_test_pad = pad_sequences(x_test_token, padding='post', maxlen=134)
	y_test_CNN = df.Emotion
	cnn_model = load_model('../output/cnn_model.h1')
	#test_loss, test_accuracy = cnn_model.evaluate(x_test, y_test, verbose=False)
	y_pred = cnn_model.predict_classes(x_test_pad)
	#weighted_f1_score = f1_score(y_test, y_pred, average="weighted")
	#print("CNN: ")
	#print("Test {}: {}".format('loss', test_loss))
	#print("Test {}: {:.2f}".format('accuracy', test_accuracy*100))
	#print("f1-score: {:.2f}%".format(weighted_f1_score*100))
	
	mapping = pd.read_excel('../data/le_name_mapping.xlsx')
	mapping_dict ={}
	for key in mapping['Encoded Label']:
		mapping_dict[key] = mapping.loc[key][0]
	label_list = []
	for y in y_pred:
		label_list.append(mapping_dict[y])
	return label_list
	
	