# Predict Customer Churn using Spark

## Motivation

Customer churn, defined as the number of individuals discontinuing a service over a specific time period, is one of the key performance 
indicators of a company. Identifying customers who are more likely to churn and taking actions to prevent churn are critical to a 
company's business. The reason is that it can cost a company more to attract a new customer than it does to retain an existing customer 
according to studies. 

In this project, I predicted the customer churn of a fictitious company Sparkify that provides music streaming service as Spotify or 
Pandora does. Customers can use the service through a free tier plan with roll advertisements playing between songs or through a 
subscription plan with a month flat rate. Customers can upgrade, downgrade or cancel the service at any time. Additionally, I built a 
pipeline (`Scale_up.py`) to import, clean and analyze data to track user engagement and identify potential churn users.

## Data

I first worked on a mini subset (see result at `Sparkify_mini.ipynb`) locally, and then deployed a Spark cluster on the cloud using IBM
Cloud to analyze a larger amount of data (see result at `Sparkify_midi.ipynb`).

## Libraries

Libraries I used in this project include:

- Python
- Pyspark
- Numpy
- Pandas
- Matplotlib
- Seaborn

## Result

My analyses focused on the activities during the most recent 10 days. The result from the mini dataset indicates that **days being a user, number of thumbs down (in the past 10 days) and number of advertisements (in the past 10 days)** are the top 3 important features in predicting customer churn. 

My first suggestion for Sparkify is to **launch a 7-day free trial plan** to improve user experience. During the trial, the user can enjoy the service without ads. At the end of the trial, the user has the option of continuing the service as a paid subscriber or downgrading to the free plan. Sparkify can perform an A/B test to determine if this action will reduce customer churn.

Another suggestion is to **improve the personalization service**. This action will recommend songs that the user may like thereby reducing the number of thumbs down. This can be done by building a collaborative filtering recommender.

I created a [blog post](https://medium.com/@fivecentsly/predict-customer-churn-using-spark-71ac4f3b6b14) explaining the details of the analyses if you're interested.

## Acknowledgements

Thank Udacity for inspiring me with this project and providing necessary dataset. 
