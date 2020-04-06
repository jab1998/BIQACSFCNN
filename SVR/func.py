from scipy.integrate import quad
import numpy as np
def integrand(t, a):
	return pow(t,a-1)*pow(np.e,-t)

def probabilisticvecs(yi,m):
	pvrs=[0]*m
	pvrs[int(yi)/10]=1
	return pvrs

def probabilisticvecs1(yi,m):
	B=64
	den=0
	sum1=0
	pvrs=[]
	for i in range(m):
		den=den+np.exp(-B*pow((float(yi)/100)-(((i*10)+5.0)/100),2))
	for i in range(m):
		p1=np.exp(-B*pow((float(yi)/100)-(((i*10)+5.0)/100),2))
		sum1=sum1+p1/den
		pvrs.append(p1/den)
	return pvrs




	
