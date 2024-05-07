# NAIVE BASED

Naive Bayes classifiers are a ==collection of classification algorithms== based on Bayes’ Theorem. It is not a single algorithm but a family of algorithms where all of them share a common principle, i.e. ==every pair of features being classified== is independent of each other.

One of the most simple and effective classification algorithms, the Naïve Bayes classifier aids in the rapid development of machine learning models with rapid prediction capabilities.

Naïve Bayes algorithm is used for classification problems. It is highly used in ==text classification==. In text classification tasks, data contains high dimension (as each word represent one feature in the data). It is used in ==spam filtering, sentiment detection, rating classification== etc. The advantage of ==using naïve Bayes is its speed==. It is fast and making prediction is easy with high dimension of data.

![[Pasted image 20240328081312.png]]

## Assumption of Naive Bayes

The fundamental Naive Bayes assumption is that each feature makes an:
- ***Feature independence:*** The features of the data are conditionally  independent of each other, given the class label.
- ***Continuous features are normally distributed:**** If a feature is continuous, then it is assumed to be normally distributed within each class.
- ****Discrete features have multinomial distributions:**** If a feature is discrete, then it is assumed to have a multinomial distribution within each class.
- ****Features are equally important:**** All features are assumed to contribute equally to the prediction of the class label.
- ****No missing data:**** The data should not contain any missing values.

==The assumptions made by Naive Bayes are not generally correct in real-world situations. In-fact, the independence assumption is never correct but often works well in practice.==

## Advantages of Naive Bayes Classifier

- **Easy to implement** and computationally efficient.
- Effective in cases with a large number of features.
- Performs well even with **limited training data**.
- It performs well in the presence of categorical features.
- For numerical features data is assumed to come from normal distributions

## Disadvantages of Naive Bayes Classifier

- Assumes that features are independent, which may not always hold in real-world data.
- Can be influenced by irrelevant attributes.
- May assign zero probability to unseen events, leading to poor generalization.

## Applications of Naive Bayes Classifier

- ****Spam Email Filtering****: Classifies emails as spam or non-spam based on features.
- ****Text Classification****: Used in sentiment analysis, document categorization, and topic classification.
- ****Medical Diagnosis:**** Helps in predicting the likelihood of a disease based on symptoms.
- ****Credit Scoring:**** Evaluates creditworthiness of individuals for loan approval.
- ****Weather Prediction****: Classifies weather conditions based on various factors.

# BAYES THEOREM
helps determine the probability of an event (A) occurring, considering the fact that another event (B) has already happened.
![[Pasted image 20240328082634.png]]
# WORKING
![[Pasted image 20240401210351.png]]

# TYPES

### 1. Gaussian Naive Bayes: Championing Continuous Data

- **Data Type:** Continuous - Numbers that can take on any value within a range (e.g., temperature, height, stock prices).
- **Assumption:** Features are assumed to follow a normal distribution (bell-shaped curve). This means the data points tend to cluster around a central value with a gradual decrease in frequency as you move away from the center.
- **How it Works:** Gaussian Naive Bayes calculates the probability of each feature value belonging to a specific class under the assumption that it follows a normal distribution. By combining these individual probabilities using Bayes' theorem, it predicts the class with the highest overall probability for a new data point.

**Example:** Imagine classifying emails as spam or not spam based on features like word frequency (e.g., number of times "urgent" appears) and email length. Gaussian Naive Bayes would assume the word frequencies and email length follow normal distributions for both spam and non-spam emails. It would then calculate the probability of each feature value (e.g., specific word frequency) given the class (spam or not spam) based on these distributions. Finally, it would combine these probabilities to predict the class (spam or not spam) for a new email.
![[Pasted image 20240328083754.png]]
### 2. Multinomial Naive Bayes: Mastering the Discrete with Counts

- **Data Type:** Discrete (counts) - Data represented as whole numbers indicating frequency or occurrence (e.g., word counts in a document, number of website visits per user).
- **Assumption:** Features follow a multinomial distribution. This is a generalization of the binomial distribution (two possible outcomes) for scenarios with multiple categories. In simpler terms, it represents the probability of observing a specific set of counts for each feature within a class.
- **How it Works:** Similar to Gaussian Naive Bayes, Multinomial Naive Bayes calculates the probability of each feature count belonging to a specific class, considering the multinomial distribution. It then combines these probabilities using Bayes' theorem to predict the class with the highest overall probability for a new data point.

**Example:** Classifying documents as sports-related or not based on word frequencies (e.g., frequency of words like "athlete" and "baseball"). Multinomial Naive Bayes would assume the word frequencies follow a multinomial distribution for both sports and non-sports documents. It would calculate the probability of each word count (e.g., specific frequency of "athlete") given the class (sports or not sports) based on these distributions. Finally, it would combine these probabilities to predict the class (sports or not sports) for a new document.

### 3. Bernoulli Naive Bayes: The Binary Maestro

- **Data Type:** Binary - Features can only have two possible values (0 or 1). This is often used for text classification where features represent word presence or absence in a document (1 for present, 0 for absent).
- **Assumption:** Features follow a Bernoulli distribution. This is a special case of the binomial distribution where there are only two outcomes (success or failure). In Naive Bayes context, it represents the probability of a feature being present (1) or absent (0) given a particular class.
- **How it Works:** Bernoulli Naive Bayes calculates the probability of each feature being present (1) or absent (0) belonging to a specific class, considering the Bernoulli distribution. It then combines these probabilities using Bayes' theorem to predict the class with the highest overall probability for a new data point.

**Example:** Classifying emails as spam or not spam based on the presence or absence of specific words (e.g., "urgent" present = 1, absent = 0). Bernoulli Naive Bayes would assume the presence/absence of each word follows a Bernoulli distribution for both spam and non-spam emails. It would calculate the probability of each word being present (1) or absent (0) given the class (spam or not spam) based on these distributions. Finally, it would combine these probabilities to predict the class (spam or not spam) for a new email.


# SVM (SUPPORT VECTOR MACHINE)

SVMs, or Support Vector Machines, are a powerful type of supervised learning algorithm used in machine learning for various tasks, primarily ==**classification**==. They can also be used for regression and outlier detection in some cases.
 The main idea behind SVMs is to find a ==hyperplane that maximally separates the different classes==  in the training data. 
 This is done by finding the hyperplane t==hat has the largest margin==, which is defined as the distance between the hyperplane and the closest data points from each class. 
 Once the hyperplane is determined, new data can be classified by determining on which side of the hyperplane it falls. SVMs are particularly useful when the data has many features, and/or when there is a clear margin of separation in the data.
The dimension of the hyperplane depends upon the ==number of features==. If the number of input features is two, then the hyperplane is just a line. ==If the number of input features is three, then the hyperplane becomes a 2-D plane==. It becomes difficult to imagine when the number of features exceeds three.
## MATHS BEHIND SVM / HOW SVM WORKS
[[Drawing 2024-03-28 09.42.26.excalidraw]]
![[Pasted image 20240401221430.png]]
![[Pasted image 20240401221455.png]]

### Support Vector Machine Terminology

1. **Hyperplane:** Hyperplane is the decision boundary that is used to separate the data points of different classes in a feature space. In the case of linear classifications, it will be a linear equation i.e. wx+b = 0.
2. **Support Vectors:** Support vectors are the closest data points to the hyperplane, which makes a critical role in deciding the hyperplane and margin. 
3. **Margin**: Margin is the distance between the support vector and hyperplane. The main objective of the support vector machine algorithm is to maximize the margin.  The wider margin indicates better classification performance.
4. **Kernel**: Kernel is the mathematical function, which is used in SVM to map the original input data points into high-dimensional feature spaces, so, that the hyperplane can be easily found out even if the data points are not linearly separable in the original input space. Some of the common kernel functions are **linear, polynomial, radial basis function(RBF), and sigmoid.**
5. **Hard Margin:** The maximum-margin hyperplane or the hard margin hyperplane is a hyperplane that properly separates the data points of different categories without any misclassifications.
6. **Soft Margin:** When the data is not perfectly separable or contains outliers, SVM permits a soft margin technique. Each data point has a slack variable introduced by the soft-margin SVM formulation, which softens the strict margin requirement and permits certain misclassifications or violations. It discovers a compromise between increasing the margin and reducing violations.
7. **C:** Margin maximisation and misclassification fines are balanced by the regularisation parameter C in SVM. The penalty for going over the margin or misclassifying data items is decided by it. A stricter penalty is imposed with a greater value of C, which results in a smaller margin and perhaps fewer misclassifications.
8. **Hinge Loss:** In machine learning, hinge loss acts as a function that measures the penalty an SVM incurs when a data point is misclassified, or classified with a low confidence level. Imagine a decision boundary separating the data into classes. Hinge loss assigns a higher penalty the farther a data point falls from the correct side of the decision boundary, with the greatest penalty for complete misclassifications.
9. **Dual Problem:** A dual Problem of the optimisation problem that requires locating the Lagrange multipliers related to the support vectors can be used to solve SVM. The dual formulation enables the use of kernel tricks and more effective computing.

https://www.javatpoint.com/machine-learning-support-vector-machine-algorithm
## SVM Kernals
Kernel is the mathematical function, which is used in SVM to map the original input data points into ==high-dimensional feature spaces==, so, that the hyperplane can be easily found out even if the data points are not linearly separable in the original input space. Some of the common kernel functions are linear, polynomial, radial basis function(RBF), and sigmoid.
![[Pasted image 20240328120033.png]]

# K MEANS CLUSTERING
- K-means clustering is a popular unsupervised machine learning algorithm used for grouping ==unlabeled data into K distinct clusters== based on similarity.
- Each data point belongs to the cluster with the nearest mean.
- The objective is to partition the data into distinct groups such that the total sum of the squared distance between the data points and the centroid of their clusters is minimized.

### ADVANTAGES
- **Simplicity and Ease of Use:** K-means is a well-understood algorithm with a straightforward concept. It partitions data points into groups (clusters) based on their similarity. This makes it easy to learn and implement, even for beginners in machine learning.
    
- **Speed and Efficiency:** The algorithm is computationally efficient, allowing it to handle large datasets quickly. This is because it uses a relatively simple iterative process to assign data points to clusters and refine those clusters.
    
- **Scalability:** K-means scales well with increasing data size. Even with massive datasets, it can efficiently process information and identify patterns.
    
- **Versatility:** The algorithm can be adapted to various applications by using different distance metrics to measure similarity between data points. This flexibility makes it applicable across a wide range of domains.
    
- **Interpretable Results:** K-means provides clear cluster assignments, making it easy to understand the structure of the data and the characteristics of each group. This allows for better communication and interpretation of the results.
    
- **Baseline for Comparison:** K-means serves as a baseline for evaluating more complex clustering algorithms. Its efficiency and interpretability make it a benchmark to compare the performance of other methods.
### DISADVANTAGES

- **Pre-defined number of clusters (k):** You need to specify the number of clusters (k) upfront, which can be tricky if you don't have a good idea of how many clusters exist in your data. Picking the wrong number can lead to inaccurate results.
- **Sensitivity to outliers:** ==Outliers can significantly skew the centroids== (which represent the center of each cluster), leading to poorly defined clusters.
- **Limited cluster shapes:** K-means works best with data that forms spherical clusters. It can struggle with data that has irregular shapes or clusters of varying densities.
- **Initial centroid placement:** The initial placement of the centroids can impact the final clustering. Different starting positions can lead to different (and potentially suboptimal) results.
- **Curse of dimensionality:** K-means becomes less effective in high-dimensional data (data with many features). The distances between data points become harder to interpret in high dimensions.
### ALGO
1. **Initialization**: Choose K initial cluster centroids randomly from the data points or by some heuristic method.
    
2. **Assignment Step**: Assign each data point to the nearest centroid, forming K clusters. The nearest centroid is typically determined using a distance metric, commonly the Euclidean distance.
    
3. **Update Step**: Recalculate the centroids of the K clusters based on the mean of all data points assigned to each cluster.
    
4. **Repeat**: Repeat steps 2 and 3 until centroids no longer change

![[Pasted image 20240329152937.png]]
## DIAGRAM
https://www.javatpoint.com/k-means-clustering-algorithm-in-machine-[[]]learning
## NUMERICAL WITH 2D DATA
![[Pasted image 20240329160137.png]]
> **JISKA DISTANCE KAM HAI VO US CLUSTER MEIN AAYEGA AFTER THIS CALCULATE NEW CENTROID VALUE BY TAKING OLD AND NEW'S AVERAGE**

## NUMERICAL WITH 1D DATA
![[Pasted image 20240329160916.png]]
![[Pasted image 20240329160935.png]]

# C MEANS CLUSTERING
  
- C-means clustering, often referred to as fuzzy c-means clustering, is a method of clustering that allows one piece of data to belong to two or more clusters.
- This approach is different from k-means clustering, where each data point belongs exclusively to one cluster. 
- In c-means clustering, each data point belongs to a cluster to a degree specified by a membership grade. 
- This grade is a number between 0 and 1 that represents the strength of the association between that data point and a particular cluster.
## ALGO
The c-means algorithm works as follows:

1. Initialize the membership coefficients randomly, subject to the constraint that the sum of membership coefficients for each data point is 1.
2. Compute the cluster centroids using the current membership coefficients.
3. Find the distance of each datapoint from the centroid
4. Update the membership coefficients based on the distances between data points and cluster centroids.
5. Repeat steps 2 and 3 until the change in membership coefficients is below a specified threshold.
![[Pasted image 20240329165854.png]]
![[Pasted image 20240329170030.png]]
![[Pasted image 20240329170801.png]]![[Pasted image 20240329171240.png]]
![[Pasted image 20240329171429.png]]
**Advantages:**

- **Flexibility:** Handles overlapping data effectively.
- **Informative Membership Degrees:** Membership degrees provide additional insight into data point relationships with clusters.

**Disadvantages:**

- **Initialization Sensitivity:** Like K-means, sensitive to initial centroid placement. Running it multiple times can help.
- **Parameter Tuning:** The fuzziness parameter (m) needs to be tuned for optimal results.

## **Applications**

**Image Segmentation:**

- C-means excels at segmenting images into meaningful regions. It can be used to group pixels with similar characteristics (color, intensity, texture) into distinct objects or regions of interest. This is valuable for applications like:
    - Medical image analysis (segmenting tumors, organs, or other structures)
    - Object detection in computer vision (identifying and isolating objects like faces, cars, or animals)
    - Content-based image retrieval (finding images with similar visual content based on segmented regions)

**Customer Segmentation:**

- In marketing, C-means can be used to group customers into segments based on their purchase history, demographics, or other factors. The fuzzy membership degrees reveal the likelihood of a customer belonging to each segment, providing a more nuanced view compared to hard clustering. This allows for:
    - Targeted marketing campaigns tailored to specific customer segments
    - Personalized product recommendations based on customer profiles
    - Understanding customer behavior and preferences

**Medical Diagnosis:**

- C-means can be applied to medical datasets to group patients based on a combination of symptoms, lab results, or other clinical data. By analyzing membership degrees, medical professionals can:
    - Identify potential disease clusters or patient subgroups
    - Develop more precise diagnoses by considering the degree of overlap between symptoms
    - Improve patient risk stratification and treatment planning

**Pattern Recognition:**

- C-means is useful for identifying patterns and grouping similar data points in complex datasets. This can be applied in various domains like:
    - Financial analysis (identifying patterns in stock market trends or customer spending habits)
    - Text mining (grouping documents based on topic similarity)
    - Social network analysis (identifying communities or clusters of users with similar interests)
    - Scientific data analysis (finding patterns in gene expression data or other scientific measurements)

# HIERARCHICAL CLUSTERING
Hierarchical clustering is an unsupervised machine learning algorithm that groups similar data points into a hierarchy of clusters. It works by either merging smaller clusters into larger ones (agglomerative clustering) or splitting larger clusters into smaller ones (divisive clustering).
![[Pasted image 20240329193114.png]]
### ****Hierarchical clustering has several advantages over other clustering methods****

- The ability to handle non-convex clusters and clusters of different sizes and densities.
- The ability to handle missing data and noisy data.
- The ability to reveal the hierarchical structure of the data, which can be useful for understanding the relationships among the clusters.

### ****Drawbacks of Hierarchical Clustering****

- The need for a criterion to stop the clustering process and determine the final number of clusters.
- The computational cost and memory requirements of the method can be high, especially for large datasets.
- The results can be sensitive to the initial conditions, linkage criterion, and distance metric used.  
    In summary, Hierarchical clustering is a method of data mining that groups similar data points into clusters by creating a hierarchical structure of the clusters. 
- This method can handle different types of data and reveal the relationships among the clusters. However, it can have high computational cost and results can be sensitive to some conditions.


| **Agglomerative Clustering**                                                          | **Divisive Clustering**<br>                                                 |
| ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Starts with each data point as its own cluster                                        | Starts with all data points in one cluster<br>                              |
| Iteratively merges the two closest clusters until all data points are in one cluster. | Iteratively splits the cluster with the greatest diameter into two clusters |
| 33                                                                                    | Continues splitting until each data point is its own cluster                |
| 00                                                                                    | Less common than agglomerative clustering                                   |

## AGGLOMERATIVE
![[Pasted image 20240329193213.png]]

![[Pasted image 20240329193353.png]]
# USE CASES OF CLUSTERING
**Customer Segmentation:** Clustering can group customers into segments based on purchase history, demographics, or website behavior. This allows businesses to tailor marketing campaigns, product recommendations, and loyalty programs to specific customer groups, boosting engagement and sales.

**Market Research:** By clustering consumers based on similar characteristics and preferences, businesses can gain valuable insights into market trends and identify new customer segments with high potential.

**Image Segmentation:** In image processing, clustering algorithms can group pixels with similar features (color, intensity) to segment images into objects or regions. This has applications in medical imaging (tumor detection), self-driving cars (obstacle identification), and satellite imagery analysis (land cover classification).

**Anomaly Detection:** Clustering can be used to identify data points that fall outside of expected clusters. These anomalies might indicate fraudulent transactions, network intrusions, or faulty equipment, allowing for timely intervention.

**Document Clustering:** Text documents can be clustered based on word usage, topics, or themes. This is useful for organizing large document collections, information retrieval systems, and plagiarism detection.

**Social Network Analysis:** By clustering users in social networks based on their connections and interactions, we can identify communities with shared interests, track the spread of information, and recommend new connections.

**Exploratory Data Analysis (EDA):** Clustering is a great tool for initial data exploration. It can help discover hidden patterns, identify outliers, and guide further analysis.

# GRID SEARCH VS RANDOM SEARCH
- **Hyperparameters:** These are settings defined **before** training that control how the model learns from data. They are not directly learned by the model itself. Examples include:
    
    - **Learning rate:** Controls how much the model updates its internal parameters during each training step.
    - **Number of hidden layers (neural networks):** Defines the model's complexity.
    - **Kernel size (Support Vector Machines):** Affects how the model identifies patterns in the data.

![[Pasted image 20240402004326.png]]
![[Pasted image 20240402004404.png]]

![[Pasted image 20240402003700.png]]
# REINFORCEMENT LEARNING
Reinforcement Learning (RL) is a subfield of machine learning that focuses on how an intelligent agent should take actions in an environment to maximize some notion of cumulative reward. It differs from supervised learning, where the agent is given labeled examples to learn from, and unsupervised learning, where the agent has to discover patterns in unlabeled data.Here are the key points about reinforcement learning in machine learning:

## What is Reinforcement Learning?

- RL is about learning the optimal behavior in an environment to obtain maximum reward through trial-and-error interactions with the environment.
- The agent learns by observing the consequences of its actions and adjusting its behavior accordingly, without relying on labeled examples or a model of the environment

# WHY REINFORCEMENT LEARNING
![[Pasted image 20240402005639.png]]
# ELEMENTS OF REINFORCEMENT LEARNING
![[Pasted image 20240402010159.png]]
https://www.geeksforgeeks.org/what-is-reinforcement-learning/