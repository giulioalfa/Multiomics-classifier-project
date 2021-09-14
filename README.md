# Multiomics classifier project - USER MANUAL
## Introduction
Here we present the code and the related user manual for the Multiomics Classifier Project for Bioinformatics exam. The following instruction are useful in order to reproduce the results of the Machine Learning and Deep Learning experiments presented in the slides. The "Data" folder contains the pre-processed and structured data, ready to be fed to the models. Each folder in the project contains the useful python files to run the experimented approach. The models we present are:
* Machine Learning algorithms: Random Forest, SVM, KNN, Lasso, Rigde.
* Shallow neural network: basic multi-layer perceptron
* MORONET
* MORONET + Similarity network fusion
* Graph Convolutional Network + SNF
* 3-branch MLP + VCDN
* 3-branch MLP + 4th branch GCN with SNF + VCDN

## Brief workpath description

## Approaches instructions
### Machine learning algorithms
### Deep Learning Algorithms

* MORONET + Similarity 

It’s a four-branches architecture integrated with the VCDN . Every branch consists of a Graph Convolutional Network (GCN). Every GCN is made of three graph   convolution, which is the multiplication between weighted data and adjacency matrix, after every graph convolution there is a leaky relu and a dropout. Every GCN has its own Classifier, a fully connected layer which takes as input the output of GCN, to predict class labels. The VCDN, which is made of a fully connected layer, a leaky relu and another fully connected layer for class output, integrates the four branches to guess the class label. The first three branches inspect one single omic and the  cosine similarity respect to that omic. The fourth computes early aggregation, computes SNF with the similarity matrixes computed omic per omic with Euclidean metric. The loss chosen is a Cross Entropy Loss.
The correnspondig code folder is called moronet_branchSNF. How to use it : open the moronet_branchSNF_main.py and substitute at line 5 the dataset name you want to use. The syntax is the following: data_folder = dataset_path +'LuadLusc100', substitute "LuadLusc100" with the chosen dataset. 


* Graph Convolutional Network + SNF

It's a single stream architecture. It consists of a single GCN which computes early aggretion of the three omics and SNF with the similarity matrixes computed omic per omic with Euclidean metric. The simple Classifier uses GCN outputs as inputs for class label prediction. 
