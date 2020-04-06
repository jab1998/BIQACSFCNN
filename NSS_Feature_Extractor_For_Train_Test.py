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
from keras.models import Model # basic class for specifying and training a neural network
from tqdm import tqdm
from keras import optimizers
import hickle as hkl 
from skimage.util import view_as_windows
from scipy import signal
import math


Labels=si.loadmat('./Data/AllMOS_release.mat')['AllMOS_release'][0]

Image_Names=si.loadmat('./Data/AllImages_release.mat')['AllImages_release']
#print np.array([cv2.imread(os.path.join('./Images/'+Image_Names[i][0][0])).shape for i in range(1169)])

Images= np.array([cv2.resize(cv2.imread(os.path.join('./Images/'+Image_Names[i][0][0])),(500,500)) for i in range(1169)])

print ("Stage:1 Read all the Images")
print 
#########################################################################################################################################

print (len(Labels),Images.shape)

'''
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

print (patch.shape)
print (patchlabel.shape)	
print ("Stage2: Completed (Patch Extraction)")
print
'''
 
#########################################################################################################################################

X_train, X_test, Y_traintarget, Y_testtarget = train_test_split(Images, Labels, test_size=0.2, random_state=42)
print (X_train.shape,Y_traintarget.shape)
print (X_test.shape, Y_testtarget.shape)
print ("stage3: Test Train Split Completed")
'''
X_train= hkl.load( 'X_train.hkl' )
X_test= hkl.load( 'X_test.hkl' )

Y_train= hkl.load( 'Y_train.hkl' )
Y_test= hkl.load( 'Y_test.hkl' )

Y_traintarget= hkl.load( 'Y_traintarget.hkl' )
Y_testtarget= hkl.load( 'Y_testtarget.hkl' )
'''
window = signal.gaussian(49, std=7)
window=window/sum(window)
NSStrainfeatures=[]
NSStestfeatures=[]
count=0

for I in tqdm(X_train):
	#print count
	count=count+1 
        #I = cv2.imread('img5.bmp',cv2.IMREAD_GRAYSCALE)
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
        NSStrainfeatures.append(feat)


for I in tqdm(X_test):
        #I = cv2.imread('img5.bmp',cv2.IMREAD_GRAYSCALE)
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
        NSStestfeatures.append(feat)

#print NSStestfeatures.shape
NSStrainfeatures=np.array(NSStrainfeatures)
NSStestfeatures=np.array(NSStestfeatures)
hkl.dump( NSStrainfeatures, 'NSStrainfeatures1.hkl' )
hkl.dump( NSStestfeatures , 'NSStestfeatures1.hkl' )

hkl.dump( Y_traintarget, 'Y_traintarget1.hkl' )
hkl.dump( Y_testtarget, 'Y_testtarget1.hkl' )


hkl.dump( X_train, 'X_TrainImages.hkl' )
hkl.dump( X_test, 'X_TestImages.hkl' )

'''
j=0
for Y in tqdm(Y_traintarget):
	Y_traini= np.array([f.probabilisticvecs(Y,10)])
	if j==0:
		Y_train=Y_traini
		j=1
	else:
		Y_train=np.append(Y_train,Y_traini,axis=0)
j=0
for Y in tqdm(Y_testtarget):
	Y_testi= np.array([f.probabilisticvecs(Y,10)])
	if j==0:
		Y_test=Y_testi
		j=1
	else:
		Y_test=np.append(Y_test,Y_testi,axis=0)

print (X_train.shape,Y_train.shape)
print (X_test.shape, Y_test.shape)
print ("Stage4 : Completed (probabilistic vecs extraction)")
print
'''
#########################################################################################################################################
'''
num_train, height, width, depth = X_train.shape
num_test = X_test.shape[0]
num_classes = 10

n=X_train.shape[0]
X_train1 = X_train[:, :n/2].astype('float32')
X_train2 = X_train[:, n/2:].astype('float32') 
X_train = np.hstack((X_train1, X_train2))

X_test = X_test.astype('float32')
X_train /= np.max(X_train)
X_test /= np.max(X_test)

Y_train=Y_train
Y_test=Y_test

hkl.dump( X_train, 'X_train.hkl' )
hkl.dump( X_test, 'X_test.hkl' )
hkl.dump( Y_train, 'Y_train.hkl' )
hkl.dump( Y_test, 'Y_test.hkl' )

hkl.dump( Y_traintarget, 'Y_traintarget.hkl' )
hkl.dump( Y_testtarget, 'Y_testtarget.hkl' )
'''
