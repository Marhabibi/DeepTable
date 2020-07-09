from inputvars import *
epochs, md1,emb,lrs,inp = inputvarsTrain(sys.argv[1:],sys.argv[0])
import numpy as np
import gensim
from gensim.models.keyedvectors import KeyedVectors
import re
import random 
import pickle
from keras_self_attention import SeqSelfAttention
import sys
import os
os.environ['KERAS_BACKEND']='tensorflow'
import keras
from keras import regularizers
from keras.layers import Embedding,Dense, Input, Flatten,Conv1D,Conv2D, MaxPooling1D, Embedding, Concatenate, Dropout,AveragePooling1D,LSTM, GRU, Bidirectional, TimeDistributed,Convolution2D,MaxPooling2D,AveragePooling2D,Permute,Activation,Reshape, BatchNormalization,Permute,Activation,Reshape, BatchNormalization,RepeatVector
from keras.layers.core import Permute
from keras.models import Model,Sequential,load_model 
from keras.callbacks import ModelCheckpoint
from keras import backend as K
np.random.seed(813306)	
from tensorflow import set_random_seed
set_random_seed(2)

def read_input(inp):
	with open(inp,'rb') as f:
		[data,Y,dictionary]=pickle.load(f)
	
	train_size = int(Y.shape[0]*.75)	
	X_train = data[0:train_size,:,:,:]
	y_train = Y[0:train_size]
	
	return X_train,y_train,dictionary

def read_embeddings(dictionary):
	
	modelemb = gensim.models.KeyedVectors.load_word2vec_format('wikipedia-pubmed-and-PMC-w2v.bin', binary=True)
	w2v = dict(zip(modelemb.index2word, modelemb.syn0))
	embedding_matrix = np.zeros((len(dictionary) + 1, 200))
	
	for j, i in dictionary.items():
		if w2v.get(j) is not None:
			embedding_matrix[i] = w2v[j]
	return embedding_matrix
	
def cell_encoder(input_shape,dic_length,embedding_matrix,embedding_flag):

	r_in = Input(shape=input_shape)
	if embedding_flag == 1:
		c_emb = Embedding(dic_length,200,weights=[embedding_matrix],input_length=input_shape[-1],trainable=True)(r_in)
	else:
		c_emb = Embedding(dic_length,200,input_length=input_shape[-1],trainable=True)(r_in)	
	c_lstm = Bidirectional(LSTM(50))(c_emb)
	c_dense = Dense(100,activation='relu')(c_lstm)
	c_dense = Dropout(0.1)(c_dense)
	c_dense = Dense(100,activation='relu')(c_dense)
	c_dense = Dropout(0.1)(c_dense)
	
	return Model(r_in, c_dense)
	
def column_encoder(input_shape, config):

	input_layer = Input(shape = input_shape)
	if config == 1:
		conv_layer1 = MaxPooling2D((1,input_shape[1]))(input_layer)	
	else:
		conv_layer1 = MaxPooling2D((input_shape[0],1))(input_layer)
	dense_layer1 = Dense(100,activation='relu')(conv_layer1)
	dropout_layer1 = Dropout(0.1)(dense_layer1)
	dense_layer2 = Dense(100,activation='relu')(dropout_layer1)
	dropout_layer2 = Dropout(0.1)(dense_layer2)
	flat_layer1 = Flatten()(dropout_layer2)

	return Model(input_layer,flat_layer1)
	
def deep_table_model(input_shape,dictionary,embedding_matrix,embedding_flag):
	
	c_in = Input(shape=input_shape, dtype='float64')
	reshape_in = Reshape((input_shape[-2]*input_shape[-3],input_shape[-1]),input_shape = input_shape)(c_in)
	
	# column representation
	embedding_layer_col = TimeDistributed(cell_encoder((input_shape[-1],),len(dictionary)+1,embedding_matrix,embedding_flag))(reshape_in)
	col_in = Reshape((input_shape[-3],input_shape[-2],100),input_shape = (input_shape[-2]*input_shape[-3],100,))(embedding_layer_col)
	col_wise_layer = column_encoder((input_shape[-3],input_shape[-2],100,),1)(col_in)
	
	# row representation
	embedding_layer_row = TimeDistributed(cell_encoder((input_shape[-1],),len(dictionary)+1,embedding_matrix,embedding_flag))(reshape_in)
	row_in = Reshape((input_shape[-3],input_shape[-2],100),input_shape = (input_shape[-2]*input_shape[-3],100,))(embedding_layer_row)
	row_wise_layer = column_encoder((input_shape[-3],input_shape[-2],100,),0)(row_in)

	# concatenate rows and columns representations
	flats = keras.layers.Concatenate(axis=-1)([col_wise_layer,row_wise_layer])
	
	# softmax classifier
	final_dense = Dense(3, activation='softmax', kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01))(flats)
		
	
	return Model(c_in,final_dense)

if __name__ == "__main__":
	
	# variable initialization
	MAX_COL=9
	MAX_COL_LENGTH=9	
	MAX_CELL_LENGTH=4	
	embedding_flag = int(emb)
	learning_r = float(lrs)
	epoch_s = int(epochs)
	filepath=md1+"/model"+"-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}-{loss:.4f}-{acc:.4f}.hdf5"
	
	# read train samples
	X_train,y_train,dictionary = read_input(inp)
	
	# read embedding vector
	embedding_matrix = read_embeddings(dictionary)
	
	# model initialization and training
	final_model = deep_table_model((MAX_COL,MAX_COL_LENGTH,MAX_CELL_LENGTH,),dictionary,embedding_matrix,embedding_flag)
	final_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=learning_r), metrics=['accuracy'])
	callbacks_list = [ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min',period = 1)]
	final_model.fit(X_train, y_train, epochs=epoch_s, verbose=1, validation_split =0.25,shuffle=True,callbacks=callbacks_list)
