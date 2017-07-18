# MNIST-Classification-with-NeuralNet
MNIST Handwritten Digits Classification using 3 Layer Neural Net  98.7% Accuracy

## Classifying the MNIST Digits using 3 Layer Neural Networks

`Deskewing the Images yields much good accuracy.`

**Accuracy was 98.7% after deskewing the images
before it was 98.4% simple 3 Layer Neural Nets.**

## Here the Dependencies Required for Running the Code:
1. Python 2.7xx
2. Numpy, scipy, matplotlib Library Installed 
2. OpenCV 3.xx, "MNIST" for reading data. Eg.  **pip install mnist**


`Our Model has 3 Layers`
`Containing`
```
 1 Input Layer -> 28*28 U
 
 1 Hidden Layer -> 300 HU
 
 1 Output Layer -> 10 U
```
**We have used the Backprop Algorithm for Training using the SGD Optimizer with Momentum .
  Applied PCA Dimensionality Reduction Technique to reduce the dimension to make dataset smaller, using 324 components to         retain 99.78% variance of input data images**

`Need the Dataset that are for training in separate folder and one with test in other folder`




*Run as :*

`python mnist_nn.py  --train  '/home/......'  --test  '/home/.......'**`
