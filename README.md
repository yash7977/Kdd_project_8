# Project Name/Intro: Digvijay
### - Project Name: Kdd_project_8
### - Project Introduction: Social Media Data Set
### - Team Introduction:

# Data and Source description: Digvijay


# Core Technical Concepts/Inspiration

### - Application of the CRISP-DM Process:

### - Domain Knowledge (document sources):

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


### - Evaluation: Krishna

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
