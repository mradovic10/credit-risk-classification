# credit-risk-classification
Module 20 Challenge

For this challenge, I was tasked with training and evaluating two models based on loan risk and identifying the creditworthiness of borrowers. A dataset called 'lending_data', found in the Resources folder, was used for analysis. It compiles historical lending activity from a peer-to-peer lending services company.

All the analysis for the challenge was performed in the 'credit_risk_classification' Jupyter Notebook using numpy, pandas, pathlib, imblearn, and scikit-learn.

## Overview of the Analysis

The purpose of this analysis was to identify the creditworthiness of current and future borrowers by looking at the 7 metrics used in the dataset: loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, and total debt. In order to do this, the data was first split into a labels set (y) containing the status of each loan and a features set (X) with the metrics. The data was then further split into training and testing sets.

Next, the training data of the original dataset was used to create a Logistic Regression Model (Model 1). With this model and the testing metrics data, the machine made predictions on the creditworthiness of each borrower.

This step of creating a Logistic Regression Model was repeated using resampled training data with the help of RandomOverSampler from imblearn (Model 2). Both models were then evaluated on their effectiveness.

## Results

* Machine Learning Model 1:
    * Balanced accuracy score: 0.95
    * Precision scores: 1.00 for healthy loans, 0.85 for high-risk loans
    * Recall scores: 0.99 for healthy loans, 0.91 for high-risk loans

* Machine Learning Model 2:
    * Balanced accuracy score: 0.99
    * Precision scores: 1.00 for healthy loans, 0.84 for high-risk loans
    * Recall scores: 0.99 for both healthy and high-risk loans

## Summary

The overall accuracy of both models is great, especially for Model 2. Both models are near-perfect at predicting healthy-loans. The real difference between the two is the recall score of high-risk loans. Model 2's score is 0.08 greater than Model 1's, making it a near-perfect 0.99. This means that out of all actual high-risk loans, Model 2 was able to correctly predict 99% of them. The weakness of both models is their precision at identifying high-risk loans that are actually healthy loans.

Even with this weakness, I would nonetheless recommend Model 2. Because we are predicting the creditworthiness of borrowers, it does not hurt the lender too much to occasionally label a borrower incorrectly as high-risk even though they may have a healthy stature. The machine virtually never identifies a high-risk loan as a healthy loan. It is better to play it safe, and that is exactly what Model 2 does. It is able to successfully pick out 99% of all actual healthy and high-risk loans, and that is why it is a worthwhile machine learning model.