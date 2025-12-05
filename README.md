**Credit Default Prediction Using Machine Learning**

**Objective**

Predict the likelihood of credit card default using demographic and financial variables, and compare the performance of multiple supervised machine learning models.

**Project Overview**

This project analyses a dataset of 7,000+ credit card clients to understand behavioural patterns and build predictive models for default risk. The analysis includes data preprocessing, exploratory data analysis (EDA), feature engineering, and model evaluation using AUC.
The dataset originates from Yeh & Lien (2009), containing 23 predictors including credit limit, payment history, bill amounts, and past payments.

**Methods Used**
1. Exploratory Data Analysis (EDA)
Examined distribution differences between defaulters vs non-defaulters
Identified key drivers such as repayment status and credit limit
Detected strong correlations among monthly bill-statement features (X12–X17), motivating dimensionality reduction

2. Data Preprocessing
Log-transformation of skewed variables (e.g., bill amounts, payments)
Creation of consolidated bill-statement feature to mitigate multicollinearity
Cleaning & preparing variables for modelling

3. Feature Engineering
Log-transformed continuous variables to stabilize variance
Combined correlated features into a single engineered variable for improved model interpretability

4. Machine Learning Models
Implemented and compared four supervised ML algorithms:
Logistic Regression
K-Nearest Neighbours (KNN)
Naïve Bayes
Decision Tree

**Key Results**
- Naïve Bayes achieved the best model performance with AUC = 0.76.
- Logistic Regression and Decision Trees showed competitive performance after preprocessing.
- Feature engineering and data cleaning improved overall model stability.
- Repayment history variables were among the strongest predictors of default.

**Tools & Libraries**

R
tidyverse (data cleaning, EDA)
ggplot2 (visualisation)
caret / base R ML functions (modelling & evaluation)


