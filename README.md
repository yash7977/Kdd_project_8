# Project Name/Intro:
### - Project Name: Kdd_project_8
### - Project Introduction:
*Social media and social networking sites are online platforms where people can connect with their real-life family, friends, and colleagues, and build new relations with others who share similar interests. The most popular English social media sites in 2019 are Twitter, Facebook, and Reddit. Our aim is to handle the social media related dataset by using classification, regression, performing Sentiment analysis and using various machine learning algorithm on the same. We have selected the social media dataset because it can have huge variety of subdomains, multiple attributes, large number of data points and much more.*

### - Team Introduction:
*Aniruddha Sudhindra Shirahatti, Chandrakanth Rajesh, Digvijay Gole, Krishna Vishwanatham, Yash Bonde.* 
*We are having common interest in pursuing specialization in the field of Data Science. We as a group are planning to work on social media dataset to implement the data mining concepts.*

# Data and Source description:-
-Social media analytics is the practice of gathering data from social media websites and analyzing that data to make business decisions.

-The most common use of social media analytics is to mine customer sentiment to support marketing and customer service activities.

-The first step in a social media dataset analysis is to determine for which  goals the data that is gathered and analyzed will benefit. Typical objectives include classification of the choices made by social media profile, performing sentiment analysis on the profile, getting feedback on topics,products and services, and improving public opinion for a particular topic.More advanced types of social media analysis involve sentiment analytics.

-This practice involves sophisticated natural-language-processing machine learning algorithms parsing the text in a person's social media post about a company to understand the meaning behind that person's statement.

-These algorithms can create a quantified score of the public's feelings toward a company based on social media interactions and give reports to management on how well the company interacts with customers.

-There are a number of types of social media analytics tools for analyzing unstructured data found in tweets and Facebook posts. 

-In addition to text analysis, many enterprise-level social media analytics tools will harvest and store the data. 

-As more social media analytics rely on machine learning, popular open platforms like R, Python and TensorFlow serve as social media analytics tools.


# Core Technical Concepts

### - Application of the CRISP-DM Process:
**1. Business/Research Understanding Phase -** *In this phase, we have explored various resources on the internet to learn about the challenges and applications of the social media data for data mining. We did our research on how the social media data can be used to uncover knowledge in various domains like branding and marketing, crime and law enforcement, crisis monitoring and management, as well as public and personalized health management, etc.*

**2. Data Understanding Phase -** *In this phase, we will make observations by looking at the data and checking of the relationship among the variables, potential independent and dependent variables, continuous variables, flag variables, categorical varaibles, mean, median and standard deviation in the data fetaures.*

**3. Data Preparation Phase -** *In this phase, we will deal with obsolete/redundant fields, missing values, outliers and make the data useable to feed as an input to the data mining models. Thus minimizing GIGO (Garbage In - Garbage Out)*

**4. Modeling Phase -** *we will fit different data mining models on the preprocessed dataset, check the results and try to fine tune the models to achieve better performance.*

**5. Evaluation Phase -** *We will try to fit the dataset on various data mining models and evaluate their performance by implementing concepts like Cross-validation, Confusion matrix, etc. If the model performance is good and the results are as per the business understanding requirement, we will deploy, else we will repeat the step 1 to 5 repeatedly till we achieve the desired results.*

**6. Deployment Phase -** *In this phase, we will deploy the model so that the intended users can incorporate our product for their business requirements.*

### - Domain Knowledge (document sources):
[DOMAIN-SPECIFIC USE CASES FOR KNOWLEDGE-ENABLED SOCIAL MEDIA ANALYSIS](http://www.knoesis.org/node/2895)

[Social Media Domain Analysis (SoMeDoA)](https://pdfs.semanticscholar.org/10cc/18164991ce56ef151cb70d80a8ccff016b49.pdf)

### - Data Understanding and EDA:(Yash)
Data Understanding: Social media data has a broad term that encompasses many diverse types of data. Any type of information can be subjected to data analytics techniques to get insight that can be used to improve things. 
Social media data offered by most social media sites gives insight into what people are responding to and engaging with on your social channels. We can use this data to measure the growth and effectiveness of your social channels, usually to improve brand awareness, profits, return on investment (ROI), and also to analyze and predict sentiments. Analytics also can help you understand what works for your competitors and their audiences.
Each social platform has its' own analytics or insights tool:
•	Twitter uses Twitter Analytics
•	Facebook offers in-depth analytics on the Insights tab of Facebook pages
•	Instagram uses the Facebook Insights platform
•	LinkedIn offers basic, free data on your company page and full analytics software with a premium account
•	YouTube uses the YouTube analytics dashboard
Tracking and understanding analytics for your social media campaigns is one of the key factors to success.

EDA:



### - Data Preparation: (CK)
#Steps To Prepare The Data.
1. Get the dataset and import the libraries.
2. Handle missing data.
3. Encode categorical data.
4. Splitting the dataset into the Training set and Test set.
5. Feature Scaling, if all the columns are not scaled correctly.

So, we will be all the steps on the dataset one by one and prepare the final dataset on which we can apply regression and different algorithms.
#1: Get The Dataset.
We need to extract data from anyone API’s. But the data could be in any form. So we need to convert it to the CSV format. CSV stands for Comma Separated Values.
These are some of the libraries that we are going to use for this project:
Numpy
Matplotlib
Pandas
Sklearn 

#2. Handle Missing Values
When we convert the dataset to the CSV format and get the info about the data it will have missing values which is usually represented by NA. There are many ways to handle missing values. Whenever we come across minute missing values we are going to drop the rows using the .dropna function. Whenever we come across large missing values we are going to perform KNN Imputation and different interpolation methods to handle the missing values.

#3.Encode categorical data.
As we come across categorical data, we need to encode them into numerical format to proceed further to make the analysis. For this we need to encode the data. We are going to use the one hot encoding to encode the categorical variables when we encounter more than. 2 variables. Whenever we come across one or two variables we are going to use the label encoding. We import LabelEncoder and OneHotEncoder from sklearn.preprosseing

#4. Split the Dataset into Training and Test Set.
We have to feed our Data Model Training and test datasets. Generally, we split the data with a ratio of 70% for the Training Data and 30% to test data. Training Data is used to build the m model where as Test set is used to evaluate the model.  From sklearn.model_selection we are going to import train_test_split.

#5. Feature Scaling
In a general scenario, Machine Learning is based on Euclidean distance. In our dataset we will be encountering different coulmns with different range of values. That is why this is called feature scaling. We use StandardScaler from sklearn.preprocessing to perform the scaling 

### - Machine Learning: (CK)
These are the 6 different algorithms that we are going to make use of:
* Logistic Regression (LR)
* Linear Discriminant Analysis (LDA)
* K-Nearest Neighbors (KNN).
* Classification and Regression Trees (CART).
* Gaussian Naive Bayes (NB).
* Support Vector Machines (SVM).

Select Best Model
We now have 6 models and accuracy estimations for each. We need to compare the models to each other and select the most accurate.

6. Make Predictions
The KNN algorithm is very simple and was an accurate model based on our tests. Now we want to get an idea of the accuracy of the model on our validation set.
This will give us an independent final check on the accuracy of the best model. It is valuable to keep a validation set just in case you made a slip during training, such as overfitting to the training set or a data leak. Both will result in an overly optimistic result.


### - Evaluation: Krishna **Social media data like facebook, Twitter and other blogs is increasing day by day. After understanding the sentiments of the social media data. We need to mine knowledge and capture the ideas from the dataset. In this process of capturing the insights from the data, we need to evaluate the results at each datamining phase which is a challenging task.
The common problem which everyone can face is that for unbalanced social media data streams with, for example, 90%
of the instances in one class, the simplest classifiers will have high accuracies of at least 90%.

We can use different evaluation techniques like:
Accuracy
Confusion matrix
F1 score
Precision-Recall or PR curve
ROC (Receiver Operating Characteristics) curve
AOC and ROC curves**

### - Conclusion: krishna


# Contributors:
1. Aniruddha Sudhindra Shirahatti
2. Chandrakanth Rajesh
3. Digvijay Gole
4. Krishna Vishwanatham
5. Yash Bonde

# TODO
Our next steps include:
1. Aggregating data from multiple sources
2. Performing Exploratory Data Analysis
3. Performing Data Preprocessing on the dataset

# Contact
Aniruddha Sudhindra Shirahatti - ashiraha@uncc.edu

Digvijay Gole - dgole@uncc.edu

Krishna Vishwanatham - kvishwa1@uncc.edu
