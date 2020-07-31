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

The ISCX data set was created in 2012 by capturing traffic in an emulated network environment over one week. The authors used a dynamic approach to generate an
intrusion detection data set with normal as well as malicious network behavior. The 7-day simulation dataset consists of 3 days of attack-free traffic and 4 days of mixed benign and malign traffic. This datasets contins four types of attack, they are HTTP Denial of Service (DoS), Infiltrating the network from inside, Distributed Denial of Service (DDoS), and Brute Force SSH. The simulation was created to simulate and mimic user behaviour activity. Profile-based user behaviour was created by executing a user-profile that synthetically generates at random synchronized times. The dataset came with labelled traffic that could assist the researcher for testing, comparison, and evaluation purposes. The datasets is available at https://www.unb.ca/cic/datasets/ids.html to download. 


- dataset preprocessing

  - The original datasets contain the 7 days packet capture file in pcap format and also provide the label data in xml format with information of each flow and their tag name. The tag contains whether it is normal or attack information.
  - Therefore to generate the datasets with associated labeling, we need to first convert the single pacp file of each day to multiple pcap file based on flows. 
  - The conversion was done using a script presented in ……….
  - Each flow pcap file name contains five fields of information. They are source IP, destination IP, source port, destination port and protocol. 
  - Based on this information we match each flow data with provided information and split all the match records into normal and attack groups for each day. 
  - Read all the flow based pcap files and store raw bytes as text using … script
  - Read all the text files and did preprocessing using …. Script and store as a csv file.


### data analysis

- feature selection

### Pretraining

- model
- loss function

### Intrusion Detection Model

- model
- loss function

# References
1. Wang, W., Sheng, Y., Wang, J., Zeng, X., Ye, X., Huang, Y., & Zhu, M. (2017). HAST-IDS: Learning hierarchical spatial-temporal features using deep neural networks to improve intrusion detection. IEEE Access, 6, 1792-1806.

2. Ring, M., Wunderlich, S., Scheuring, D., Landes, D., & Hotho, A. (2019). A survey of network-based intrusion detection data sets. Computers & Security, 86, 147-167.

3. Kamarudin, M. H., Maple, C., Watson, T., & Safa, N. S. (2017). A new unified intrusion anomaly detection in identifying unseen web attacks. Security and Communication Networks, 2017.

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
