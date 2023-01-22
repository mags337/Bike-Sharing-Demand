# Bike-Sharing-Demand
In this project the goal is to develop a machine learning model that accurately predicts hourly bike rental demand based on seasonal and environmental factors, for the Capital Bikeshare program in Washington, D.C.. The dataset is from a Kaggle competition.
https://www.kaggle.com/competitions/bike-sharing-demand/overview

# Results of EDA
![](/Project_3/eda_plots.png)
![](/Project_3/Bike_rental_distribution_holiday_working_day.png)

DAILY: During the week there are two peaks of bike demand, one in the morning (8 am) and one in the afternoon (6-7 pm). On the weekend the bikes are mainly used during the day, with a peak in noon/early afternoon (12am - 4pm). On holidays there is only one peak of bike demand during the day, similar to the weekend distribution. 
MONTHLY: During the year the bike demand in the individual months is slightly different between registered users and casual users. The first use the bikes all year around, mainly from mai till december, the latter use the bikes far less (lower toal number), but mainly from april till october.
YEARLY: There is an inrease in demand visible from 2011 to 2012

# Feature Engineering
- extract hour, month, year from datetime column
- add "rushhour" collumn
- add "daynight" collumn
- drop highly correlated features
- use log on taget variable (count) to account for skewness of data
- remove data outliers 
- scale numerical features (temp, humidity, windspeed)
- categorical features

# Pipeline
- define transformers & features
- create column transformer (preprocessor)
- test (1) Linear Regression, (2) Polynomial + Linear Regression, (3) Random Forest Regression
- build pipelines

# Check models
- Crossvalidation score (cv=5)
- RMSLE (root mean square log error)/ R2:
    (1) 0.65/ 0.76
    (2) 0.65/ 0.93
    (3) 0.65/ 0.99
    
# Predictions
### Randomly chosen week of prediction with random forest model
![](/Project_3/RandomForest_Model_true_pred_oneweek.png)
### Comparison of prediction results of linear model and random forest model
![](/Project_3/Linear_RF_Model_true_predtest.png)
RandomForest model performed better than linear regression model, and lead to a kaggle (RSMLE) score of 0.442
