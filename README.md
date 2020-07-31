# Introduction

This page presents an end-to-end network intrusion detector based on HAST-IDS model. Unlike HAST-IDS which uses one-hot vectors as input, our model uses pretrained embeddings, taking advantage of the large amount of unlabeled data. Experiments on ISCX 2012 dataset show competitive results. Additionally, we argue that our model is superior to the state-of-art model when dealing with few-shot or zero-shot attacks.(work in progress)
- input
- output
- example of this task

# Results
- results of intrustion detection
- results of pretraining(latest loss/loss curve)

# Details

### Dataset(ISCX 2012)

- dataset introduction
- dataset preprocessing

### data analysis

- feature selection

### Pretraining

- model
- loss function

### Intrusion Detection Model

- model
- loss function

# References

# network_traffice_detection

In the case of Intrusion Detection, NSL-KDD is consider as the standard datasets and have been used by many reseacher for their reseach work. 
Therefore we also perform the network traffic classification using various range of deep learning model on NSL-KDD.
The peroformance of model is promising. All the models are uploaded in this repository with proper name.

The NSL-KDD datasets have 41 features but all of the features are not equally important for the prediction. Therefore we have also done some feature engineering jobs which are uploaded under Feature_Engineering folder. We tried different apporach for feature learning. First we performed constant, quasi constant features elimination and than perform feature selection based on corelation coefficient. We have also tried feature selection based on sklearn feature importance method. 
Instead of selecting the features based on one machine learning feature importnace, we have calculated the feature importance of various model and than compute the average feature importance. We also tried permutation and recursive feature importance technique as well. 

Though NSL-KDD is consider as standard datasets, it is also very true that this a old and label data sets with limited number of features. Therefore we have selected another datasets called ISCX 2012 which contains very high volume of raw data. Our plan is to propose various way to present raw data and apply transfer learning and few-shot techniuque with combination of cnn and lstm model. 

In this project we are using ISCX 2012 datasets for classification of network traffic.

Plan:
1. Download the datasets from https://www.unb.ca/cic/datasets/ids.html
2. Convert the single pcap file to multiple pcap files based on flow
3. Seperate the network traffic data accoirding to corresponding tag (Normal, DoS, DDos, BFSSH, INI)
4. Convert each flow of pcap file to text file
5. Make the datasets ready for Few-shot based model
6. Build the model
7. Test the model 
8. Compare the performance with the latest related work
