# import libraries
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import udf, col, lit, avg, count, sum, max, min, desc
from pyspark.sql.types import IntegerType, StringType, FloatType

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel

import re
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


# create a Spark session
spark = SparkSession.builder \
    .appName('Sparkify Project') \
    .getOrCreate()

# load data
def load_data(data_path):
    """
    Load json file
    """
    df = spark.read.json(data_path)
    
    return df


# clean data
def clean_data(df):
    """
    Clean df
    
    INPUT:
        df - original df
        
    OUTPUT:
        df - clean df with valid userids, 
             replaced lengths with 0, 
             replaced artists and songs with None
    """
    df = df.filter(df.userId != '')
    df = df.fillna({'length':0, 'artist':'None', 'song':'None'})   
    
    return df


# extract features
def extract_feature(df, save_path):
    """
    Extract features and create new columns
    
    INPUT:
        df - clean df
        
    OUTPUT:
        df - df with extracted features
    """
    # define churn
    flag_churn_event = udf(lambda x: 1 if x == 'Cancellation Confirmation' else 0, IntegerType())
    df = df.withColumn('churnEvent', flag_churn_event('page'))

    windowval = Window.partitionBy('userId')
    df = df.withColumn('churn', max('churnEvent').over(windowval))
    
    # create a column dataRange
    df = df.withColumn('dataRange', 
                       ((max('ts').over(windowval)-min('ts').over(windowval))/1000/3600/24).cast('int'))
    
    # create a column countDown 
    df = df.withColumn('lastEvent', max('ts').over(windowval))
    df = df.withColumn('countDown', ((df.lastEvent-df.ts)/1000/3600/24).cast('int'))
    
    # filter data
    df = df.filter((df.dataRange >= 10) & (df.countDown <= 10))
    
    # create a column thumbDown
    thrumb_down = udf(lambda x: 1 if x == 'Thumbs Down' else 0, IntegerType())
    df = df.withColumn('thumbDown', thrumb_down('page'))
    
    # create a column downgrade
    downgrade = udf(lambda x: 1 if x == 'Downgrade' else 0, IntegerType())
    df = df.withColumn('downgrade', downgrade('page'))
    
    # create a column ads
    roll_advert = udf(lambda x: 1 if x == 'Roll Advert' else 0, IntegerType())
    df = df.withColumn('ads', roll_advert('page'))
    
    # save df
    df.write.csv(save_path)
    
    return df


# create data with engineered features
def engineer_feature(df, save_path):
    """
    Create a new df with engineered features for model training
    
    INPUT:
        df - df containing extracted features
        
    OUTPUT:
        data - new df with engineered features
    """
    # convert gender to numerical values F: 0, M: 1
    df_gender = df \
        .select('userId','gender').dropDuplicates() \
        .replace({'F':'0', 'M':'1'}, subset='gender') \
        .select('userId', col('gender').cast('int'))
    
    # extract days being a user
    df_servicedays = df \
        .select('userId','churn','registration','ts') \
        .withColumn('serviceDays', (df.ts - df.registration)) \
        .groupby('userId','churn').agg({'serviceDays':'max'}) \
        .select('userId','churn',(col('max(serviceDays)')/1000/3600/24).cast('int').alias('serviceDays'))
    
    # extract number of thumbs down
    df_thumbdown = df \
        .select('userId','thumbDown') \
        .groupby('userId').agg({'thumbDown':'sum'}) \
        .withColumnRenamed('sum(thumbDown)','numThumbdown')
    
    # extract number of downgrades
    df_downgrade = df \
        .select('userId','downgrade') \
        .groupby('userId').agg({'downgrade':'sum'}) \
        .withColumnRenamed('sum(downgrade)','numDowngrade')
    
    # extract number of ads
    df_ads = df \
        .select('userId','ads') \
        .groupby('userId').agg({'ads':'sum'}) \
        .withColumnRenamed('sum(ads)','numAds')
    
    # combine all features, drop userId and rename churn with label
    data = df_gender.join(df_servicedays, 'userID') \
        .join(df_thumbdown, 'userID') \
        .join(df_downgrade, 'userID') \
        .join(df_ads, 'userID') \
        .drop('userID') \
        .withColumnRenamed('churn','label')
    
    # save data
    data.write.csv(save_path)
    
    return data


# preprocess data for model training
def preprocess(data):
    """
    Preprocess data for model training: transform skewed features,
                                        combine features,
                                        scale features, 
                                        and split data
    
    INPUT:
        data - data containing label and features
    
    OUTPUT:
        train, test sets
    """
    # transform skewed features
    data_pd = data.toPandas()
    skewed = ['numAds', 'numDowngrade', 'numThumbdown']

    for col in skewed:
        data_pd[col] = data_pd[col].astype('float64').replace(0.0, 0.01) 
        data_pd[col] = np.log(data_pd[col])
    
    # convert pandas df to spark df
    data = spark.createDataFrame(data_pd)
    
    # combine features to a vector
    cols = data.drop('label').columns
    assembler = VectorAssembler(inputCols=cols, outputCol='numFeatures')
    data = assembler.transform(data)

    # standardize the vectors
    scaler = StandardScaler(inputCol='numFeatures', outputCol='features', withStd=True)
    scalerModel = scaler.fit(data)
    data = scalerModel.transform(data)
    
    train, test = data.randomSplit([0.8, 0.2], seed=42)
    
    return train, test


# display model evaluation result
def model_eval(result, evaluator, test_set):
    """
    Display accuracy, precision, recall and F1 score of test set
    
    INPUT:
        result - result containing predictions (df)
        evaluator - evaluator
        test_set - name of dataset used in prediction (str)
    """    
    accuracy = evaluator.evaluate(result, {evaluator.metricName: 'accuracy'})
    precision = evaluator.evaluate(result, {evaluator.metricName: 'weightedPrecision'})
    recall = evaluator.evaluate(result, {evaluator.metricName: 'weightedRecall'})
    f1 = evaluator.evaluate(result, {evaluator.metricName: 'f1'})
    
    print(f'\nAccuracy of {test_set} set: {accuracy}')
    print(f'Precision of {test_set} set: {precision}')
    print(f'Recall of {test_set} set: {recall}')
    print(f'F1 score of {test_set} set: {f1}')
    

# build a model
def build_model(model, model_name):
    """
    Build a model with 3-fold cross validation and display the evaluation result
    
    INPUT:
        model - model with parameters
        model_name - name of the classifier (str)
    
    OUTPUT:
        trained_model - trained model
    """
    # initialize model
    model = model

    # build paramGrid
    paramGrid = ParamGridBuilder().build()

    crossval_model = CrossValidator(estimator=model,
                                    estimatorParamMaps=paramGrid,
                                    evaluator=MulticlassClassificationEvaluator(),
                                    numFolds=3)

    # train the training set and calculate training time
    start = time.time()
    trained_model = crossval_model.fit(train)
    end = time.time()
    train_time = end - start

    # predict the traing and validation set
    evaluator = MulticlassClassificationEvaluator()
    result_train = trained_model.transform(train)
    result_test = trained_model.transform(test)
    
    print(f'{model_name}')
    print(f'\nTraining time: {train_time} seconds')
    model_eval(result_train, evaluator, 'train')
    model_eval(result_test, evaluator, 'test')
    
    return trained_model


# save model
def save_model(model, save_path):
    """
    Save a model
    """
    try:
        model.save(save_path)
    except:
        model.write().overwrite().save(save_path)

        
# load a pretrained model
def load_model(model_path):
    """
    Load a pretrained model
    """
    model = CrossValidatorModel.load(model_path)
    
    return model