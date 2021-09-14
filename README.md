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

The project focus was to develop our own integration method for multi-omics classification. We had to look at the classification methods in the literature and to compare our integration method with at least two existing ones. The results had to be reported based on the data retrieved from the GDC Portal and on a previosuly provided synthetic dataset of 5000 samples. Due to the poor xomplexity of both the datasets, we decided to re-use also the ones provided by the MORONET project, from which we started to study.  Our first ste was to test all the datasets with basic machine learning algorithms in order to have an initial benchmark, computed through prediction accuracy, F-1 score and AUC score. In the following analysis, we will ignore the results obtained from the synthetic dataset since every algorithm has overfitted due to the high separability of the sampeles.  The ML algorithms gave quite good results, especially for the LuadLusc (GDC) dataset, where they reach the 96% of F1 score. For ROSMAP and BRCA, the results are slightly lower, but sill very good, reaching the 70-80% of F1 score. After this step, we tested the MORONET model and we obtained the results claimed in the paper. From this point, our main aim was to improve the two main characteristics of the network in some way:
* Similarity network
* Architecture and aggregation approaches  
  
Our first experiment was focused on how to improve the similarity links between the patients: MORONET performs a simple cosine similarity among patients, we wanted to explore the various and more complex solutions: we tried to substitute the cosin similarity with an Eucledian distance and then with the SNF (Similarity Newtork Fusion) algorithm. These approaches did not lead to notable improvements.  Noticing the great influence of the early aggregation method on the ML algorithms (results from aggregated data happen to be always higher than the single omic experiments), we tried this approach in a Deep Learning scenario, so we fed a Graph Convolutional Network (GCN) with the early aggregated data, but this approach has worsened the performance.

## Approaches instructions
### Machine learning algorithms
### Deep Learning Algorithms

* MORONET + Similarity 

It’s a four-branches architecture integrated with the VCDN . Every branch consists of a Graph Convolutional Network (GCN). Every GCN is made of three graph   convolution, which is the multiplication between weighted data and adjacency matrix, after every graph convolution there is a leaky relu and a dropout. Every GCN has its own Classifier, a fully connected layer which takes as input the output of GCN, to predict class labels. The VCDN, which is made of a fully connected layer, a leaky relu and another fully connected layer for class output, integrates the four branches to guess the class label. The first three branches inspect one single omic and the  cosine similarity respect to that omic. The fourth computes early aggregation, computes SNF with the similarity matrixes computed omic per omic with Euclidean metric. The loss chosen is a Cross Entropy Loss.
The correnspondig code folder is called moronet_branchSNF. How to use it : open the moronet_branchSNF_main.py and substitute at line 5 the dataset name you want to use. The syntax is the following: data_folder = dataset_path +'LuadLusc100', substitute "LuadLusc100" with the chosen dataset. 


* Graph Convolutional Network + SNF

It's a single stream architecture. It consists of a single GCN which computes early aggretion of the three omics and SNF with the similarity matrixes computed omic per omic with Euclidean metric. The simple Classifier uses GCN outputs as inputs for class label prediction. The loss chosen is a Cross Entropy Loss. 
The correnspondig code folder is called snf_gcn. How to use it : open the snf_gcn_main.py and substitute at line 5 the dataset name you want to use. The syntax is the following: data_folder = dataset_path +'LuadLusc100', substitute "LuadLusc100" with the chosen dataset.
