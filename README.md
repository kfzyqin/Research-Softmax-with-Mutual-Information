<h1 align="center">
    <img width="250" height="auto" src="resources/tai-chi.png" />
    <br>
    Rethinking Softmax with Cross-Entropy
    <br>
</h1>

<h2 align="center">
Neural Network Classifier as Mutual Information Estimator
</h2>

<p align="center">
  <a href="#MI Estimator">MI Estimator</a> •
  <a href="#PC Softmax">PC Softmax</a> •
  <a href="#InfoCAM">InfoCAM</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

<p align="center">
    <img width="500" height="auto" src="resources/info-CAM-Illustration.png" alt="InfoCAM" />
</p>

## MI Estimator
In the paper, we prove that classification neural networks that 
optimise their weights to minimise the softmax cross-entropy are 
equivalent to the ones that maximise mutual information between 
inputs and labels with the balanced datasets. This repository 
includes the implementation for evaluating the effectiveness of 
classification mutual information estimator via synthetic 
datasets. We also show the balanced dataset assumption can be 
relaxed by modifying the traditional softmax to the 
Probability-Correct (PC) softmax. This repository also contains 
implementation for evaluating mutual information with PC-softmax
on the synthetic dataset. 

## PC Softmax
We modify the traditional softmax to the 
Probability-Correct (PC) softmax. This repository contains 
implementation for demonstrating PC-softmax can improve a large
margin than the traditional softmax for the average of the per-class 
classification accuracy. We experiment on two datasets: MNIST and
CUB-200-2011. In terms of the accuracy value, we achieve a new 
state-of-art of the micro classification accuracy 
(ours: 89.73; previous: 89.6) on CUB-200-2011. 

## InfoCAM 
We propose infoCAM: Informative Class Activation Map, which 
highlights regions of the input image that are the most relevant to a 
given label based on differences in information. The activation 
map helps localise the target object in an image. We in this 
repository show the effectiveness of the informative-theoretic 
approach than the traditional CAM. 

<p align="center">
    <img width="500" height="auto" src="resources/all-birds.png" alt="InfoCAM" />
</p>
   