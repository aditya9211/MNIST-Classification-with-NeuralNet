# MNIST-Classification-with-NeuralNet
MNIST Handwritten Digits Classification using 3 Layer Neural Net  98.7% Accuracy

## Classifying the MNIST Digits using 3 Layer Neural Networks

`Deskewing the Images yields much good accuracy.`

**Accuracy was 98.7% after deskewing the images
before it was 98.4% simple 3 Layer Neural Nets.

## Here the Dependencies Required for Running the Code:
1. Python 2.7xx
2. Numpy, scipy, matplotlib Library Installed 
2. OpenCV 3.xx, MNIST for reading data  **pip install mnist**

Code are segmented as follows:

1. Execute :
    **mnist_nn.py**

`Our Model has 3 Layers`
`Containing`
```
 1 Input Layer -> 100*100 U
 
 1 Hidden Layer -> 300 HU
 
 1 Output Layer -> 10 U
```
**We have used the Backprop Algorithm for Training using the SGD Optimizer with Momentum .**

`Need the Dataset that are for training in separate folder and one with test in other folder.Because it is a supervised Learning`


Run as :

**python train.py  --train  '/home/......'  --test  '/home/.......'**
