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

## Instructions
Here we present a brief architecture description (for DL models) together with the instructions to run them.
### Libraries
The Python libraries used are 
* pandas
* numpy
* sklearn
* pytorch
* snfpy  
Snfpy is a library that performs Similarity Network Fusion from multiple omics data. To install it use the following command on the prompt:
```
pip install snfpy
```


### Machine learning algorithms
For Machine Learning algorithms, they are all wrapped in an unique `.py` file, to try the algorithms the following file must be opened:
```
ML/ML_algorithms.py
```
Once run, it will make appear a menu where the algorithm can be chosen and it will proceed to perform that model on all the datasets, showing the results. 
```
Select an alghorithm with corresponding number:  
1 - SVM  
2 - Random Forest  
3 - KNN  
4 - Ridge  
5 - Lasso  
Number: 
```


### Deep Learning Algorithms
In order to reproduce the experimented results, every indicated `main` file must be opened and run with the preferred environment (PyCharm, VS Code, etc.). The default database which the prediction will perform is BRCA, but it can be modified by replacing the `defalut` variable in the line ```parse.add_argument('--dataset', type=str, default='BRCA')``` with the preferred one among the availables dataset: `5000samples`, `LuadLusc100`, `BRCA`, `ROSMAP`.

* Multi-layer perceptron

It's a shallow basic neural network with three fully connected layes, leaky relu and dropout. Multi-omics data aggregation is performed and then data are feed to ne net for label predction. The loss chosen is a Cross Entropy Loss.
The correnspondig code folder is called mlp.  
File to run (and modify): `mlp/mlp_main.py`


* MORONET + Similarity 

It’s a four-branches architecture integrated with the VCDN. Every branch consists of a Graph Convolutional Network (GCN). Every GCN is made of three graph   convolution, which is the multiplication between weighted data and adjacency matrix, after every graph convolution there is a leaky relu and a dropout. Every GCN has its own Classifier, a fully connected layer which takes as input the output of GCN, to predict class labels. The VCDN, which is made of a fully connected layer, a leaky relu and another fully connected layer for class output, integrates the four branches to guess the class label. The first three branches inspect one single omic and the  cosine similarity respect to that omic. The fourth computes early aggregation, computes SNF with the similarity matrixes computed omic per omic with Euclidean metric. The loss chosen is a Cross Entropy Loss.
The correnspondig code folder is called moronet_branchSNF.  
File to run (and modify): `moronet_branchSNF/moronet_branchSNF_main.py`


* Graph Convolutional Network + SNF

It's a single stream architecture. It consists of a single GCN which computes early aggregation of the three omics and SNF with the similarity matrixes computed omic per omic with Euclidean metric. The simple Classifier uses GCN outputs as inputs for class label prediction. The loss chosen is a Cross Entropy Loss. 
The correnspondig code folder is called snf_gcn.  
File to run (and modify): `snf_gcn/snf_gcn_main.py`

* 3-branch MLP + VCDN

Every branch is a multi-layer perceptron as described in the first point. Every neural net inspects one single omic giving its label probability distribution. The labels distrbutions of the three braches are then integrated by the VCDN to the final class label prediction. The loss chosen is a Cross Entropy Loss. The correnspondig code folder is called multi_mlp_vcdn.  
File to run (and modify): `multi_mlp_vcdn/multi_mlp_vcdn_main.py`

* 3-branch MLP + 4th branch GCN with SNF + VCDN

It's composed of four branches: the first three are basic multi-layer perceptrons, the fourth is a GCN. Every perceptron inspects a single omic and gives its label distribution. In the fourth branch early aggregation and SNF are computed and fed to the GCN twhose output goes through the simple Classifier for label predction. These four label distributions are integrated by the VCDN for the final label prediction. The correnspondig code folder is called 3mlp_gcn_withSNF_vcdn.  
File to run (and modify): `3mlp_gcn_withSNF_vcdn/mlp_gcn_main.py`  

## Brief workpath description

The project focus was to develop our own integration method for multi-omics classification. We had to look at the classification methods in the literature and to compare our integration method with at least two existing ones. The results had to be reported based on the data retrieved from the GDC Portal and on a previosuly provided synthetic dataset of 5000 samples. Due to the poor xomplexity of both the datasets, we decided to re-use also the ones provided by the MORONET project, from which we started to study.  Our first ste was to test all the datasets with basic machine learning algorithms in order to have an initial benchmark, computed through prediction accuracy, F-1 score and AUC score. In the following analysis, we will ignore the results obtained from the synthetic dataset since every algorithm has overfitted due to the high separability of the sampeles.  The ML algorithms gave quite good results, especially for the LuadLusc (GDC) dataset, where they reach the 96% of F1 score. For ROSMAP and BRCA, the results are slightly lower, but sill very good, reaching the 70-80% of F1 score. Then with a shallow neural network we obtain the same results as ML algorithms or we improved them (in the case of BRCA and ROSMAP). After this step, we tested the MORONET model and we obtained the results claimed in the paper. From this point, our main aim was to improve the two main characteristics of the network in some way:
* Similarity network
* Architecture and aggregation approaches  
  
Our first experiment was focused on how to improve the similarity links between the patients: MORONET performs a simple cosine similarity among patients, we wanted to explore the various and more complex solutions: we tried to substitute the cosine similarity with an Eucledian distance and then with the SNF (Similarity Newtork Fusion) algorithm. These approaches did not lead to notable improvements.  Noticing the great influence of the early aggregation method on the ML algorithms (results from aggregated data happen to be always higher than the single omic experiments), we tried this approach in a Deep Learning scenario, so we fed a Graph Convolutional Network (GCN) with the early aggregated data, but this approach has worsened the performance. Then we focused on architecture, replacing the GCNs with shallow MLPs, in order to understand the contribution of the Graph similarities and indeed the results has worsened of circa 2%. Finally, we tried to incorporate both the inter-omic patient similarity by replacing the GCNs with MLPs, but adding a 4th GCN branch with SNF logic. This last experiment gave us the most satisfactory results, which are greater than the MORONET ones in some cases.
