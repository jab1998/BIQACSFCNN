#########################################################################################################################################
import scipy.io as si
import numpy as np
import cv2
import os
import func as f
import ggd
from sklearn.feature_extraction import image
from sklearn.model_selection import train_test_split
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import Model
from keras.models import model_from_json
from tqdm import tqdm
from keras import optimizers
import hickle as hkl 
from skimage.util import view_as_windows
from scipy.stats import mode
from scipy import signal
from scipy.fftpack import fft, fftshift
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math
import pickle
import scipy.io as si
import os
import func as f
import ggd
from tqdm import tqdm
from sklearn.externals import joblib
import pickle
import csv
import pandas as pd

#########################################################################################################################################

SM=[]
totalImgs = 100
lastPoint= 72


root="/home/sunil/Downloads/BTP/BTP/"

imgPath=root+"Data/AllImages_release.mat"
labelPath = root+"Data/AllMOS_release.mat"

svr = joblib.load('SVR.joblib')

#Loading Saved Model
json_file = open('/home/sunil/Downloads/BTP/BTP/BIQASelf/Models/Resnet_Live_Chl.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("/home/sunil/Downloads/BTP/BTP/BIQASelf/Models/Resnet_Live_Chl.h5")
print("Stage1: Loaded DL model from disk")


for img in range(lastPoint,lastPoint+totalImgs):
	
	Labels=si.loadmat(labelPath)['AllMOS_release'][0][img:img+1]
	Image_Names=si.loadmat(imgPath)['AllImages_release'][img:img+1]
	
	# Reading Image
	Images= np.array([cv2.resize(cv2.imread(os.path.join(root+'/Images/'+Image_Names[j][0][0])),(500,500)) for j in range(1)])
	print ("Stage:2 Read the Image")

	window_shape = (224, 224, 3)
	step=38
	i=0
	for a in tqdm(range(len(Labels))):
		if i==0:
			B = view_as_windows(Images[a], window_shape,step=step)
			patch=B.reshape(B.shape[0]*B.shape[1],224,224,3)
			patchlabel=np.array(patch.shape[0]*[Labels[a]])
			i=1
		else:
			B = view_as_windows(Images[a], window_shape,step=step)
			patchi=B.reshape(B.shape[0]*B.shape[1],224,224,3)
			patchlabeli=np.array(patchi.shape[0]*[Labels[a]])
			patch=np.append(patch,patchi,axis=0)
			patchlabel=np.append(patchlabel,patchlabeli,axis=0)	
	
	print ("Stage3: Completed (Patch Extraction)")

	#Normalizing
	patch = patch.astype('float32')
	patch /= np.max(patch)

	
	probs = loaded_model.predict(patch)

	
	mod=mode(np.argmax(probs, axis=1))[0][0]
	outputfromCNN=np.array([[0]*10])
	outputfromCNN[0][mod]=1

	##############################################################################################################
	# NSS Features Extraction

	window = signal.gaussian(49, std=7)
	window=window / sum(window)
	NSSfeatures=[]
	count=0

	for I in tqdm(Images):
		count=count+1 
		I = I.astype(np.float32)
		u = cv2.filter2D(I,-1,window)
		u = u.astype(np.float32)
		
		diff=pow(I-u,2)
		diff = diff.astype(np.float32)
		sigmasquare=cv2.filter2D(diff,-1,window)
		sigma=pow(sigmasquare,0.5)

		Icap=(I-u)/(sigma+1)
		Icap = Icap.astype(np.float32)
		gamparam,sigma = ggd.estimateggd(Icap)
		feat=[gamparam,sigma]



		shifts = [ (0,1), (1,0) , (1,1) , (-1,1)];
		for shift in shifts:
			shifted_Icap= np.roll(Icap,shift,axis=(0,1))
			pair=Icap*shifted_Icap
			alpha,leftstd,rightstd=ggd.estimateaggd(pair)
			const=(np.sqrt(math.gamma(1/alpha))/np.sqrt(math.gamma(3/alpha)))
			meanparam=(rightstd-leftstd)*(math.gamma(2/alpha)/math.gamma(1/alpha))*const;
			feat=feat+[alpha,leftstd,rightstd,meanparam]

		feat=np.array(feat)
		NSSfeatures.append(feat)



	
	NSSfeatures=np.array(NSSfeatures)
	###########################################################################
	
	print ("Stage4: Extracted NSS Features")
	combinedFeatures=np.concatenate((NSSfeatures,outputfromCNN),axis=1)


	#Predicting scores of Image

	k=svr.predict(combinedFeatures)
	print ([Image_Names[0][0][0],k[0],Labels[0]])

	SM.append([Image_Names[0][0][0],k[0],Labels[0]])
	print (SM)
pd.DataFrame(SM,columns=['File','Pred','Actual']).to_csv('Scores.csv',mode='a',header=False)

print("Stage 5 : Saved to File")
