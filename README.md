# Predict Customer Churn using Spark

## Motivation

Customer churn, defined as the number of individuals discontinuing a service over a specific time period, is one of the key performance 
indicators of a company. Identifying customers who are more likely to churn and taking actions to prevent churn are critical to a 
company's business. The reason is that it can cost a company more to attract a new customer than it does to retain an existing customer 
according to studies. 

In this project, I predicted the customer churn of a fictitious company Sparkify that provides music streaming service as Spotify or 
Pandora does. Customers can use the service through a free tier plan with roll advertisements playing between songs or through a 
subscription plan with a month flat rate. Customers can upgrade, downgrade or cancel the service at any time. Additionally, I wrote a scalable script (`Scale_up.py`) to clean and extract features from a large amount of data for analysis to track user engagement and identify potential churn users.

## Data

I first worked on a mini subset (see result at `Sparkify_mini.ipynb`) locally, and then deployed a Spark cluster on AWS to analyze the dataset (12GB).

## Libraries

Libraries I used in this project include:

- Python
- Pyspark
- Numpy
- Pandas
- Matplotlib
- Seaborn

## Result

The mini dataset has 225 unique users with 278,154 valid events and 18 columns. The data was collected from Oct 1, 2018 to Dec 3, 2018. Given that I only have 2 months of data and there are new registered users and churn users during that, **I decide to use the most recent 10 days of data to predict churn**. For nonchurn users, this means 10 days of data before the last event; for churn users, this means 10 days of data before cancellation. I excluded users with less than 10 days of data including new users who registered in late Nov and churn users who churned within 10 days of registration. Finally, I included 202 unique users (40 churn users) and their most recent 10 days of data.

After exploratory data analysis, I found that the following features are more promising in predicting churn:
- user's gender
- days being a user
- number of thumbs down
- number of downgrades
- number of ads

I extracted these features and performed necessary feature engineering. 

To create a baseline model, I labeled all predictions as "0" because the number of churn users is small. If I predict all of them as nonchurn, I will have relatively better accuracy and F1 score. **The baseline model has an accuracy of 0.82 and an F1 score of 0.75**. F1 score is a better metric in this project because the dataset is imbalanced. 

Then I compared three different models based on training time, accuracy and F1 score. I decide to choose **random forest** because:
- it requires the least time to train
- it performs well as logistic regression or gradient-boosted tree does
- it generalizes better than gradient-boosted tree

After hyperparameter tuning, **the final model resulted in an F1 score of 0.89 with 19% improvement from baseline**. The plot of feature importance indicates that the following 3 features have the most predictive power in predicting customer churn:
- days being a user
- number of thumbs down (in the past 10 days) 
- number of advertisements (in the past 10 days)

My first suggestion for Sparkify is to **launch a 7-day free trial plan** to improve user experience. During the trial, the user can enjoy the service without ads. At the end of the trial, the user has the option of continuing the service as a paid subscriber or downgrading to the free plan. Sparkify can perform an A/B test to determine if this action will reduce customer churn.

Another suggestion is to **improve the personalization service**. This action will recommend songs that the user may like thereby reducing the number of thumbs down. This can be done by building a collaborative filtering recommender.

I created a [blog post](https://medium.com/@fivecentsly/predict-customer-churn-using-spark-71ac4f3b6b14) explaining the details of the analyses if you're interested.

## Acknowledgements

Thank Udacity for inspiring me with this project and providing necessary dataset. 
