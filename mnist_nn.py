# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
import argparse

def deskewing(img,par = 'cubic'): ## De-Skew the Images 
	print 'De-skewing Images..........'
	import cv2
	deskew_img = np.zeros((img.shape[0], img.shape[1]))
	SZ = int(np.sqrt(img.shape[1]))
	for i in np.arange(img.shape[0]):
		gray = img[i,:].reshape(SZ,SZ)
		gray = gray.astype(np.uint8)
		m = cv2.moments(gray)
		if abs(m['mu02']) < 1e-2:
			  # no deskewing needed. 
			  deskew_img[i] =  gray.flatten()
			  continue

		# Calculate skew based on central momemts. 
		skew = m['mu11']/m['mu02']

		# Calculate affine transform to correct skewness. 
		M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])

		# Apply affine transform
		if par == 'cubic':
			gray = cv2.warpAffine(gray, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
		else:
			gray = cv2.warpAffine(gray, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

		deskew_img[i] =  gray.flatten()

	return deskew_img

def load_training(train, pca = False, deskew = True):
    print 'Loading Training Dataset .........' 
    mndata = MNIST(train)
    X, y = mndata.load_training()
    X = np.array(X)  
    y = np.array(y)

    ## OpenCV is needed when deskew is set to True
    if deskew:
	X = deskewing(X)

    ## Normalization of Data(Images)
    X = X/255.0
    
    ## PCA Runs to reduce Diomensionality of Data Retaining MAx. Variance
    if pca == True:
        print "PCA Running SVD........"
        print
        sigma = 1/float(X.shape[0]) * np.dot(X.T, X)
        U, s, V = np.linalg.svd(sigma)
        X = np.dot(U[:,0:324].T,X.T) ## 324 Componenets retained 99.58% variance of data
        X = X.T

    ## Adding Bias "1" terms
    X = np.insert(X, 0, 1, axis=1)
    if pca:
        return X, y, U[:,0:324]
    else:
        return X, y, 0
    
def load_testing(test, U, pca = False, deskew = True):
    print 'Loading Testing Dataset .........' 
    mndata = MNIST(test)
    X, y = mndata.load_testing()
    X = np.array(X)  
    y = np.array(y)

    ## OpenCV is needed when deskew is set to True
    if deskew:
	X = deskewing(X)

    ## Normalization of Data(Images)
    X = X/255.0
    if pca == True:
        X = np.dot(U.T,X.T)
        X = X.T

    ## Adding Bias "1" terms
    X = np.insert(X, 0, 1, axis=1)
    return X,y

def NN_Model(neuron,initialize=False):
	## Good Weight Initialization Cited from Paper
    if initialize:
        r1 = np.sqrt(6.0/(neuron[0] + neuron[-1]))
    else:
        r1 = 1.0
    theta1 = 2.0*np.random.random((neuron[1],neuron[0]))*r1 - 1*r1
    theta2 = 2.0*np.random.random((neuron[-1],neuron[1]+1))*r1 - 1*r1
    return {'Theta1':theta1, 'Theta2':theta2}
    
def sigmoid(X): ## Sigmoid activation Function
    return 1.0 / (1.0 + np.exp(-X))

## Different Activaion Function
def h(theta,X,func='sig'):
    a = theta.dot(X.T)
    if(func== 'tanh'):
        return np.tanh(a)
    if func == 'none':
        return a
    if func == 'softplus':
        return np.log(1 + np.exp(a))
    if func == 'relu':
        return np.maximum(0.01*a, a)
    
    if func == 'softmax':
        a1 = np.exp(a)
        a1 = a1 / np.sum(a1, axis = 0, keepdims = True)
        return a1
    
    return sigmoid(a)

## Diagram to show the Weight and Input matrix Multiplication

#==============================================================================
#                                        --- Total examples -----
#  [theta0 theta1 t2 t3 t4 ........ ]  x0 x10
#              Total features          x1 x11 ... .. . .. .. ..   
#                    .                 x2 x12
#                    .                 x3 .
#                    .                 x4
#                    .                 x5
#                    .                 ..
#                    .                 .
#                    .                 .
#                    .                 . 
#                    .                 . .
#                    .                 xn x1n .........
#==============================================================================
#==============================================================================

## Cost Function Implementing L2 Norm, Not
	## penalizing the Bias terms in weight values
def cost(a4, y_new, theta, lambdaa):
    reg = (lambdaa/2.0)*(np.sum(theta['Theta1'][1:,:]**2)
						+ np.sum(theta['Theta2'][1:,:]**2))
    reg = reg/float(y_new.shape[0])
    first = (-1.0) * ( y_new*a4 + (1-y_new)*np.log(1 - a4))
    return (np.mean(first) + reg)

## Derivative of Correspnding Activation Function
def derivative(a,func='sig'):
    if func == 'tanh':
        return (1 - a*a)
    if func == 'none':
        return 1
    if func == 'softplus':
        return 1.0/(1 + np.exp(-a))
    if func == 'relu': 			## Noisy ReLU , Noise is added to it.
        a[a >= 0.00] = 1.00
        a[a < 0.00] = 0.01
        return a
    
    return a*(1-a)


## Core of ANN, BackProp..
def back_propagate(X1, y1, theta1, theta2, X, y, alpha, lambdaa, nclass,  max_iter
                   , act, seed=10, batch_size=32):
    parameters = {}
    gamma = 0.9 ## Momentum Factor
    rate = 0
    dtheta1 , dtheta2 = 0.0, 0.0
    y_new = output_encoding(y, nclass) ## Convert the value of labels to dimension of classes
    theta1_up, theta2_up = np.zeros((theta1.shape[0],theta1.shape[1])), np.zeros((theta2.shape[0],theta2.shape[1]))
    cost_new = []
    X, y, y_new = random_shuffle(X, y, y_new, seed)
    err = 100.0
    for j in np.arange(0,max_iter):
	#X, y, y_new = random_shuffle(X, y, y_new, seed)  ## Shuffling the Training Data
        k = 0
	print
        rate = rate + 1
        print 'Overall Min. Error rate : ' + str(err)
        print
        if rate == 4:
            print 'alpha decreased .......' 
            print
            if j > max_iter/2:
	        alpha = 0.82*alpha
		if gamma <= 0.98:
			gamma+= 0.01
	    	else:
			alpha = 0.70*alpha
            rate = 0

		## Softmax in Final Layer 
        for batchX , batchY in get_batch(X,y_new,batch_size):
            m, n = batchX.shape
            a2 = h(theta1,batchX,act)
            a2 = np.insert(a2, 0, 1, axis=0)
            a3 = h(theta2,a2.T,func='softmax')
            eps = alpha/float(m)

			## Error in Hidden and Output Layer
            delta3 = (a3 - batchY)*derivative(a3,'none')
            delta2 = ((theta2.T).dot(delta3))*derivative(a2,act)

			## Gradient of Theta Matrices
            ktheta1 = np.dot(delta2[1:,:],batchX)
            ktheta2 = np.dot(delta3,a2.T)

			## Momemtum Part to Accelerate the Learning Rate
            dtheta1 = eps*(ktheta1 + lambdaa*theta1) + gamma*dtheta1
            dtheta2 = eps*(ktheta2 + lambdaa*theta2) + gamma*dtheta2
            theta1 = theta1 - dtheta1
            theta2 = theta2 - dtheta2

			## Cost Per Iteration
            cost_new.append(cost(a3,batchY, {'Theta1':theta1, 'Theta2':theta2}, lambdaa))
            
			## Summary of Back Prop
            if (k % 99  == 0):
                pred_y = validate(theta1, theta2, X1, act)
                y1 = y1.flatten()
                error = 100.0 - np.mean(pred_y == y1)*100.0

				## Error Updation if LEss Error is Discovered
                if(error < err):
		    rate = 0
                    err = error
                    theta1_up = theta1
                    theta2_up = theta2
                
				## Info of Learning of NN
                if k == 0:
                    print "Epoch " + str(j+1) + " in " + str(k+1) + " iterations"+ " Error rate :  " + str(error) + "%" + " loss: " + str(cost(a3,batchY, {'Theta1':theta1, 'Theta2':theta2}, lambdaa)) 
                else:
                    print "Epoch " + str(j+1) + " in " + str(k+1) + " iterations"+ " Error rate :  " + str(error) + "%" + " loss: " + str(cost(a3,batchY, {'Theta1':theta1, 'Theta2':theta2}, lambdaa))
 		    
            k = k + 1
            
    parameters = {'Theta1':theta1_up, 'Theta2':theta2_up, 'Loss':cost_new}
    return parameters

## Extracting the Batch per Epoch in Training  
def get_batch(X, y, batch_size):
    for i in np.arange(0, X.shape[0], batch_size):
        yield(X[i:i+batch_size,:],y[:,i:i+batch_size])
        
## Find the Result of Model
def validate(theta1, theta2, X, act = 'sig'):
    aa1 = h(theta1,X,act)
    aa1 = np.insert(aa1, 0, 1, axis=0)
    aa2 = h(theta2,aa1.T,'softmax')
    accu_matrix = np.argmax(aa2,axis=0) 
    return accu_matrix

## Plot the Cost vs Iteration Curve
def show_plot(cost):
    plt.plot(np.arange(0,len(cost)) , cost)
    plt.xlabel('Iterations.......')
    plt.ylabel('Loss.............')
    plt.show() 
    
## Convert the labels to classes dimension
## same as one_hot_encoding()
def output_encoding(y, nclass):
    y_new = np.zeros((nclass,y.shape[0]))
    for  c in np.arange(0,nclass):
        pos = np.where(y==c)
        y_new[c][pos] = 1 
    return y_new
   
def random_shuffle(X, y, y_new, seed=10):
    np.random.seed(seed)
    sample = np.random.choice(X.shape[0],X.shape[0])
    y = y[sample]
    X = X[sample,:]
    y_new = y_new[:,sample]
    return  X, y, y_new


#####////////////////////********* Main Code Start Here **************///////////////////////////// 

## Input Layer -> 785 U
## 1 Hidden Layers -> 300 HU 
## 1 Output Layer -> 10 Neurons


# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-train_path", "--train", required=True,
	help="path to input directory of MNIST Train dataset")
ap.add_argument("-test_path", "--test", required = True,
	help="path to input directory of MNIST Test dataset")
args = vars(ap.parse_args())

## Getting the Same Result in Shuffle in each Run.
seed = 10
np.random.seed(seed)

## Creating Path Variable 
train = args["train"]
test = args["test"]

## Loading MNIST Dataset
print 'Fetching Data ..........'
X, y, U = load_training(train, pca=True)
X_test, y_test = load_testing(test, U, pca=True)

## Parameters for Model
max_iter = 50
alpha = 0.1
lambdaa = 0.0001
nclass = np.unique(y).shape[0]
act = 'sig'

## May Used for Calculate the No Of Neuron as hyper-parameters to Good value
nof_neuron = X.shape[0]/(2*(X.shape[1]+10))

print "Intializing the Network ........."
theta = NN_Model([X.shape[1],300,10])

## BAckProp
print "BAckPROP ................."
print
params = back_propagate(X_test, y_test, theta['Theta1'], theta['Theta2'], X, y, alpha, lambdaa,
                        nclass, max_iter, act, seed, batch_size=10)

## Calculating the predicted labels
pred_y = validate(params['Theta1'], params['Theta2'], X_test, act)
y_test = y_test.flatten()
accuracy = np.mean(pred_y == y_test)*100

## Accuracy of Our Model
print 'Accuracy :' + str(accuracy) + ' %'

## Plotting the Curve
show_plot(params['Loss'])
