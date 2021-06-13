
## Problem Statement:
Predict next-day rain by training classification models on the target variable RainTomorrow based on dataset containing past 10 years of weather observations from many locations across Australia.
## Dataset Description:
The file ‘weatherAUS.csv’ contains 142,193 data records each containing 23 columns. We have a large input feature set of 22 columns consisting of numeric as well as categorical features and one target variable ‘RainTomorrow’ as described below-
**Numerical variables:**
'MinTemp', 'MaxTemp', 'Rainfall','WindGustSpeed','WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am','Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm'
**Categorical variables:**
'Date', 'Location','WindGustDir','WindDir9am','WindDir3pm','RainToday', 'RainTomorrow'.

Below is the correlation mapping of the features with the target variable ‘RainTomorrow’.
![alt text](https://github.com/KoushikGrandhi/Machine-Learning-Projects/blob/master/Rain_Prediction_Australia/resources/Picture1.png "Logo Title Text 1")

## Data Pre-processing: 
From the correlation plotting, it can be seen that some features like 'Humidity3pm', ‘Humidity9am', 'Cloud9am', 'Cloud3pm' have stronger say in predicting the next day rain while features like 'Pressure9am', 'Pressure3pm', ‘Sunshine’ has negative correlation with the target variable ‘RainTomorrow’. Also we have features which contains Null values. So as a part of pre-processing we used the below pipeline –

**i) Data Cleaning:**
Many of the above dataset had Null values in some of the features like 'Cloud9am', 'Cloud3pm' which were refilled by the mean value of the respective features. Some Null values were present in the target variable ‘RainTomorrow’ and important feature ‘RainToday’ which had to be dropped to avoid misfitting of data.

**ii) Feature Selection:** 
Features like ‘'Pressure9am', 'Pressure3pm', ‘Sunshine’, ‘Evaporation’, 'Temp9am', 'Temp3pm' etc have negative (or negligible) correlation with the target variable, hence dropped to avoid overfitting. Features having stronger say in predicting rain like ‘Rainfall’, 'Cloud9am', 'Cloud3pm', 'Humidity3pm', ‘Humidity9am' were selected for the training purpose.

The target variable ‘RainTomorrow’ and ‘RainToday’ has strong relation for not raining and class imbalance ratio of Yes and No is around 0.28 before cleaning ,as it can be seen from the below graphsFig2(a)(b), that shows that when ‘RainToday’ has a value ‘No’ then majority of examples in our target variable is also value ‘No’. So, for the ‘NaN’ values in target variable we used the ‘RainToday’ value, and then used the average of value in the next entry and previous day entry. Similar approach was done for the ‘RainToday’ feature.

![alt text](https://github.com/KoushikGrandhi/Machine-Learning-Projects/blob/master/Rain_Prediction_Australia/resources/Picture2.png "Logo Title Text 1")
![alt text](https://github.com/KoushikGrandhi/Machine-Learning-Projects/blob/master/Rain_Prediction_Australia/resources/Picture3.png "Logo Title Text 1")

**iii) Standard Scalar:**
Features like 'Rainfall','WindGustSpeed', 'Humidity9am’ etc have different scale (in respective units), hence were normalized using Scikit’s off the box StandardScalar transformer.

## Model Selection and Performance Evaluation:

After pre-processing the data, we are ready to train different ML models and evaluate the performance on test set. Below are some points which needs to be highlighted -
**i) Imbalanced data:** 
The dataset, after pre-processing, had 74132 records out of which only 17504 records have ‘RainTomorrow’ value ‘Yes’. That means, if we map ‘RainTomorrow=yes’ as our  ‘positive’ class then we only have 23% positive examples in our dataset which is significantly less. Also, we used 70:30 as Train: Test split ratio to evaluate the models.
**ii) Choosing best metric** for such an imbalanced dataset can be tricky. Clearly, accuracy is not a good metric to go for which such class imbalance. Precision-Recall and AUCROC can be good choice for evaluation.

iii) While tuning the models, we have emphasized on Precision-0 and Recall-0 where ‘0’ denotes the negative class, in our case it is equivalent to predicting ‘No Rain’ for tomorrow. The logic behind is, if we predict a day as ‘non-rainy’ and if it actually rains, it might have more adverse affect than predicting a day ‘Rainy’ and getting it wrong. So, the goal was to minimize the number of false negatives.

I Trained the resulted dataset using: Logistic Regression, Gaussian Naïve Bayes, Bagging Classifier, k-Nearest Neighbours, Linear SVM and XGBoost, below is the 
**performance table**: 

![alt text](https://github.com/KoushikGrandhi/Machine-Learning-Projects/blob/master/Rain_Prediction_Australia/resources/Picture10.png "Logo Title Text 1")

