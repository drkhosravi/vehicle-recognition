import os
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import glob
import shutil
import random
from tqdm import tqdm
import utils

def get_hog() : 
	winSize = (256, 256)
	blockSize = (32,32)
	blockStride = (16,16)
	cellSize = (32,32)
	nbins = 9
	derivAperture = 1
	winSigma = -1.
	histogramNormType = 0
	L2HysThreshold = 0.2
	gammaCorrection = 1
	nlevels = 64
	signedGradient = True

	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)

	return hog	
# W: Resize width (before crop)
# crop_size: square crop size
def get_hog_desc(image_paths, do_prewhiten=False):
	nrof_samples = len(image_paths)
	#images = np.zeros((nrof_samples, crop_size, crop_size, 3), dtype = np.float32)
	hog_descriptors = np.zeros((nrof_samples, 2025), dtype = np.float32)
	hog = get_hog()
	for i in range(nrof_samples):
		img = cv2.imread(image_paths[i], cv2.IMREAD_COLOR)		
		if img.ndim == 2:
			img = utils.to_rgb(img)

		if i % 10 == 0:
			#cv2.imshow('img', img)
			#cv2.waitKey(1)
			print('image # %d'%(i))
		#if do_prewhiten:
			#img = utils.prewhiten(img)

		hog_descriptors[i,:] = np.squeeze(hog.compute(img))

	hog_descriptors = np.squeeze(hog_descriptors)
	print(str(nrof_samples) + ' images loaded successfully')
	cv2.destroyAllWindows()
	return hog_descriptors