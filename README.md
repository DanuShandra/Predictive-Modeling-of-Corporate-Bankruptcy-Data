# Predictive-Modeling-of-Corporate-Bankruptcy-Data
The current unstable economic circumstances triggered by the SARS-Cov-2 (COVID-19) pandemic has triggered an increased focus on insolvency and bankruptcy as many businesses struggle to handle the drastic changes in the operating models of their respective businesses, such as lockdowns, lower foot traffic, or curbside pick-ups.
Generally accepted accounting principles (“GAAP”) provide a number of liquidity and solvency metrics that assist in assessing the financial strength of a business. Although these metrics provided a snapshot picture of the financial position of a company and can indicate financial struggles, it is generally accepted other financial metrics - not immediately related to liquidity of solvency ratios - can predict the financial success of a company.
Our aim is to build a predictive model based on historical bankruptcy data to predict if a company will go bankrupt. From these models we will perform analysis to identify relevant financial variable to predict that are important for bankruptcy prediction

**Data Source**

For the purpose of our analysis, our primary resource was the website www.kaggle.com. This website is a subsidiary of Google LLC, and represents an online community for data scientists and machine learning practitioners. The website allows users to find and publish datasets to explore and build machine learning models.

We used this website for our dataset which represents data collected from the Taiwan Economic Journal for the years 1999 to 2009. Company bankruptcy was defined based on the business regulations of the Taiwan Stock Exchange. The link to the dataset is as follows: -

https://www.kaggle.com/fedesoriano/company-bankruptcy-prediction

**Data Preprocessing**

The data quality overall was strong and in general provided all relevant financial metrics. The dataset had 95 financial metrics with a portion of the metrics reflecting combinations of individual metrics to calculate GAAP liquidity and insolvency ratios. As such, there was no requirement for any feature engineering.

OneHotEncoder and MinMaxScaler were applied to the respective categorical and discrete variables before splitting the dataset into a 70-30% split.

As the dataset was imbalanced with only 3.2% of the data corresponding to a bankrupt company, and the remaining not being bankrupt, we decided to apply Synthetic Minority Oversampling Technique (SMOTE) to address this concern and create synthetic ‘bankruptcy” data to give a 50-50 split of outcomes

**Model design and analysis**

A decision stump was developed to test how well the descriptors can split the data, however no single descriptor was able to split it. The end result was an accuracy of 96.8% and balanced accuracy of 50%. These values are achieved when all predictions are ‘not bankrupt’. 

We applied the following two models: -

RandomForestClassifier
GradienBoostingClassifier

Hyperparameter tuning was applied to ‘max_depth’, ‘min_samples_leaf’, and ‘min_samples_split’. The collection of hyperparameters which gave the best result according to GridSearchCV were chosen as the parameters to use on our fitted models. It was found that the max_depth went to the maximum where the min_leaf_sample and min_samples_split went to mid to low values for both models. The hyperparameters for the models not listed above were given the defaults from sklearn.

We also tested if PCA dimensionality reduction would benefit machine learning, however, the balanced accuracy was not strong at <65% and we opted to utilise the dataset without dimensionality reduction.
When the RF model and GBC model were fit to their best parameters the balanced accuracies increased to 80.0% and 72.1%, respectively. In order to achieve balanced accuracies this high, the RF and GBC accuracies decreased to 92.3% and 96%, respectively.

**Conclusion**

In this work we developed two machine learning models, a Random Forest and a Gradient Boosted Classifier to predict if a company would go bankrupt based on a series of financial descriptors. It was found that although the accuracies of the ML models decreased to classifying all companies as going ‘not bankrupt’, the balanced accuracies increased from 50% up to 80% for the random forest model and 72.1% for the gradient boosted classifier. The predictive power of these models may allow for early detection that a company may be facing bankruptcy.

