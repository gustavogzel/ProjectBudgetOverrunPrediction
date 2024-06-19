# Predicting Project Budget Overruns in Government Procurement
## Problem :
In government procurement processes, accurately predicting the likelihood of project budget overruns based on various factors such as project type, region, project size, and historical data can help in financial planning and risk management. This project develop a program that uses artificial intelligence, specifically Apache Mahout , to predict the likelihood of budget overruns in new projects. 

## Solution Approach:
It use Java 17  leveraging Apache Mahout's  capabilities for machine learning.

Data Collection: Gather historical project data, including features such as project type, region, project size, and past project costs.
Data Preprocessing: Clean and preprocess the data for training.
Model Training: Train a Mahout model using MapReduce to predict project budget overruns.
Prediction: Use the trained model to predict the likelihood of budget overruns for new projects.
Evaluation: Evaluate the model's performance and accuracy.


## Solution
The  code  preprocess the data, train a Mahout model using MapReduce, and predict project budget overruns.

## Explanation
Data Collection: It create a sample list of Project objects with features relevant to predicting project budget overruns.
Data Preprocessing: It encode categorical variables (projectType and region) as numeric values and save the list of Project objects into a CSV file.
Model Training: It use an OnlineLogisticRegression model from Apache Mahout to train on the project data.
Prediction: It predict the likelihood of budget overruns for a new project based on its features.
JUnit Test: It create a JUnit test to validate that the process runs correctly and produces reasonable predictions.


