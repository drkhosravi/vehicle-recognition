

#Author: Hossein Khosravi 1398

import utils, hogutils, vars
import sys, os, signal
import math
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"	#force tf to use only CPU
from tensorflow.python.framework import ops
import keras
from keras import models, layers, regularizers
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import time
from train import train_model, test_model, visualize_model
from torchsummary import summary
import torch.utils.data as Data

#global variables and handlers (to stop training on user input CTRL+C)
network = keras.models.Sequential() 
stop_training = False

def handler(signum, frame):
	print('Signal handler called with signal', signum)
	print('Training will finish after this epoch')
	global stop_training
	stop_training = True
	#raise OSError("Couldn't open device!")

signal.signal(signal.SIGINT, handler) # only in python version >= 3.2

class ManageTrainEvents(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.loss = []
		self.val_loss = []
		self.acc = []
		self.val_acc = []

	def on_batch_end(self, batch, logs={}):
		#self.losses.append(logs.get('loss'))
		global stop_training
		global network
		if(stop_training):
			network.stop_training = True
			
	def on_epoch_end(self, epoch, logs=None):
		self.loss.append(logs.get('loss'))
		self.val_loss.append(logs.get('val_loss'))
		self.acc.append(logs.get('acc'))
		self.val_acc.append(logs.get('val_acc'))
		utils.plot_graphs(self.loss, self.val_loss, self.acc, self.val_acc)

def svmInit(C=5, gamma=1):#C=12.5, gamma=0.50625):
	print("c = ", c, "gamma = ", gamma )
	model = cv2.ml.SVM_create()
	model.setGamma(gamma)
	model.setC(C)
	model.setKernel(cv2.ml.SVM_LINEAR)
	model.setType(cv2.ml.SVM_C_SVC)
	
	return model

def svmTrain(model, samples, responses):
	model.train(samples, cv2.ml.ROW_SAMPLE, responses)
	return model

def svmEvaluate(model, digits, samples, labels):
	predictions = model.predict(samples)[1].ravel()
	accuracy = (labels == predictions).mean()
	print('Percentage Accuracy: %.2f %%' % (accuracy*100))

	confusion = np.zeros((5, 5), np.int32)
	for i, j in zip(labels, predictions):
		confusion[int(i), int(j)] += 1
	print('confusion matrix:')
	print(confusion)

	# vis = []
	# for img, flag in zip(digits, predictions == labels):
	# 	img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	# 	if not flag:
	# 		img[...,:2] = 0
		
	# 	vis.append(img)
	# return mosaic(25, vis)

def main(args):
	global network
	train_set = utils.get_dataset(vars.train_dir) #Load dataset
	val_set = utils.get_dataset(vars.val_dir) 

	# Get a list of image paths and their labels
	train_img_list, train_labels = utils.get_image_paths_and_labels(train_set)
	assert len(train_img_list)>0, 'The training set should not be empty'

	val_img_list, val_labels = utils.get_image_paths_and_labels(val_set)

	#utils.augment_images(train_img_list, 4) #it only must be called one time to generate several images from single image (don't forget to set validation_set_split_ratio = 0)
	
	if(os.path.exists('train_descs.npy')):
		train_descs	= np.load('train_descs.npy')
	else:
		train_descs = hogutils.get_hog_desc(train_img_list, False)
		np.save('train_descs.npy', train_descs)

	if(os.path.exists('val_descs.npy')):
		val_descs	= np.load('val_descs.npy')
	else:
		val_descs = hogutils.get_hog_desc(val_img_list, False)
		np.save('val_descs.npy', val_descs)

	train_labels = np.array(train_labels, dtype=np.int64)
	val_labels = np.array(val_labels, dtype=np.int64)
	# Shuffle data
	rand = np.random.RandomState(10)
	shuffle = rand.permutation(len(train_labels))	
	train_descs, train_labels = train_descs[shuffle], train_labels[shuffle]

	##############################################################################
	if(vars.model_name == 'mlp-torch'):
		model = torch.nn.Sequential(
				torch.nn.Linear(2025, 128),
				torch.nn.ReLU(),
				torch.nn.Linear(128, 64),
				torch.nn.ReLU(),
				torch.nn.Linear(64, 5),
			)

		train_dataset = Data.TensorDataset(torch.from_numpy(train_descs), torch.from_numpy(train_labels))
		val_dataset = Data.TensorDataset(torch.from_numpy(val_descs), torch.from_numpy(val_labels))
		datasets = {'train': train_dataset, 'val': val_dataset}

		vars.dataloaders = {x: Data.DataLoader(datasets[x], batch_size=vars.batch_size, shuffle=True, num_workers=0)
				for x in ['train', 'val']}

		vars.dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
		#vars.class_names = datasets['train'].classes

		optimizer = optim.SGD(model.parameters(), lr = vars.learning_rate, momentum=0.9)
		#optimizer = optim.Adam(model.parameters(), lr=0.05)
		# Decay LR by a factor of 0.6 every 6 epochs
		exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = vars.scheduler_step_size, gamma = vars.scheduler_gamma)
		model = model.to(vars.device)
		model = train_model(model, vars.criterion, optimizer, exp_lr_scheduler, vars.num_epochs)

		log_file = open(".\\Time-{}-{}.log".format(vars.model_name, vars.batch_size),"w")	
		for dev in ['cuda', 'cpu']:
			vars.device = torch.device(dev)
			model = model.to(vars.device)
			#run model on one batch to allocate required memory on device (and have more exact results)
			inputs, classes = next(iter(vars.dataloaders['train']))
			inputs = inputs.to(vars.device)
			outputs = model(inputs)

			s = test_model(model, vars.criterion, 'val', 100)
			log_file.write(s)
			#log_file.write('\n' + '-'*80)
		
		#log_file.write(summary(model, input_size=(3, vars.input_size, vars.input_size), batch_size=-1, device=vars.device.type))
		log_file.close() 
	elif (vars.model_name == 'svm'):
		print('Training SVM model ...')
		model = svmInit()
		svmTrain(model, train_descs, train_labels)
		model.save('svm_model.xml')
		print('Evaluating model ... ')
		svmEvaluate(model, None, train_descs, train_labels)
		t0 = time.time()
		svmEvaluate(model, None, val_descs, val_labels)
		time_elapsed = time.time()-t0
		print('Test completed over {} samples in {:.2f}s'.format(len(train_labels), time_elapsed))
		print('Test time per sample {:.3f}ms'.format(time_elapsed * 1000 / len(train_labels)))
	elif (vars.model_name == 'knn'):
		print('Training KNN model ...')
		model = cv2.ml.KNearest_create()
		model.setDefaultK(5)
		model.setIsClassifier(True)
		model.train(train_descs, cv2.ml.ROW_SAMPLE, train_labels)
		model.save('knn.xml')
		print('Evaluating model ... ')
		svmEvaluate(model, None, train_descs, train_labels)
		t0 = time.time()
		svmEvaluate(model, None, val_descs, val_labels)
		time_elapsed = time.time()-t0
		print('Test completed over {} samples in {:.2f}s'.format(len(train_labels), time_elapsed))
		print('Test time per sample {:.3f}ms'.format(time_elapsed * 1000 / len(train_labels)))		
	elif(vars.model_name == 'bayes'):
		print('Training Bayes model ...')
		model = cv2.ml.NormalBayesClassifier_create()
		model.train(train_descs, cv2.ml.ROW_SAMPLE, train_labels)
		model.save('bayes.xml')
		print('Evaluating model ... ')
		svmEvaluate(model, None, train_descs, train_labels)
		t0 = time.time()
		svmEvaluate(model, None, val_descs, val_labels)
		time_elapsed = time.time()-t0
		print('Test completed over {} samples in {:.2f}s'.format(len(train_labels), time_elapsed))
		print('Test time per sample {:.3f}ms'.format(time_elapsed * 1000 / len(train_labels)))

	elif(vars.model_name == 'mlp-keras'):
		train_labels = to_categorical(train_labels)
		if (len(val_labels) > 0):
			val_labels = to_categorical(val_labels)


		network.add(layers.Dense(128, activation='relu', input_shape=(2025,)))
		network.add(layers.Dense(64, activation='relu'))
		network.add(layers.Dense(5, activation='softmax'))
		
		opt = keras.optimizers.SGD(lr=0.05, momentum=0.5, decay=1e-3, nesterov=False)]
			#keras.optimizers.RMSprop(lr=0.001, decay=1e-6)]#
			#keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
			#keras.optimizers.Nadam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-10, schedule_decay=0.004)

		network.summary()

		network.reset_states()
		network.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])		
		#saves the model weights after each epoch if the validation loss decreased
		now = datetime.now() # current date and time
		checkpointer = ModelCheckpoint(filepath='best_model_' + now.strftime("%Y%m%d") + '.hdf5', verbose=1, save_best_only=True)
		
		manageTrainEvents = ManageTrainEvents()
		history = network.fit(train_descs, train_labels, validation_data=(val_descs, val_labels), 
				epochs=vars.num_epochs, batch_size=vars.batch_size, callbacks=[checkpointer, manageTrainEvents])

		network.save('Rec_' + now.strftime("%Y%m%d-%H%M") + '.hdf5')
		#Plot loss and accuracy
		acc = history.history['acc']
		val_acc = history.history['val_acc']
		loss = history.history['loss']
		val_loss = history.history['val_loss']
		utils.plot_graphs(loss, val_loss, acc, val_acc, True)
		#Evaluate on test dataset
		print("\nComputing test accuracy")
		test_loss, test_acc = network.evaluate(val_descs, val_labels)
		print('test_acc:', test_acc)
     

if __name__ == '__main__':
	main(None)
