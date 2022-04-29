# AirFlight-Price-Prediction
Team Member:
Shubham Shah
Jeel Patel
Manjusha Dondeti
Rama Sri Saladi

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Agenda
Motivation
Data & Task
Visual Encoding (UI Example)
Data Modeling Plan and Prediction
Conclusion

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Project Goal
Our goal is to analyze the prices of the fare for different source, destination, coupons, number of passengers, month, distance, ticket career, airline, etc.

We will also train the Machine Learning model based on the available dataset and will predict the price for the future month. 

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Dataset Introduction
Air travel price data is for the year of 2020. 

There are 12257491 number of records or transactions and 42 columns.

There are multiple attributes which concludes the same will be removed according to the correlation matrix.


# Dataset snapshot

![image](https://user-images.githubusercontent.com/74948720/165872909-acae7ef9-4b71-48ef-a911-c0a778bd1cac.png)

![image](https://user-images.githubusercontent.com/74948720/165872918-cdd02af4-5758-4f0b-b1a6-c39605168c1f.png)

# Correlation Matrix

![image](https://user-images.githubusercontent.com/74948720/165872986-88099da1-2094-4ddc-bd6a-90bba20a976c.png)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Removing unnecessary columns

Based on correlation matrix, we will drop the not required columns.

ITIN_ID and MKT_ID are highly correlated so we can remove one of them.

ORIGIN_AIRPORT_ID,ORIGIN_AIRPORT_SEQ_ID, ORIGIN_CITY_MARKET_ID are highly correlated. 

DEST_AIRPORT_ID, DEST_AIRPORT_SEQ_ID, DEST_CITY_MARKET_ID are highly correlated.

Unnamed: 41 is not needed column.

ITIN_GEO_TYPE, MKT_GEO_TYPE are highly correlated.

MARKET_DISTANCE, DISTANCE_GROUP, MARKET_MILES_FLOWN, NONSTOP_MILES are highly correlated.

The travelled miles should be equal to the flown miles. After giving this filter we can remove any of MARKET_MILES_FLOWN and MARKET_DISTANCE as they are same. 

BULK_FARE is a categorical variable suggesting multiple tickets bought or not. Which is same as number of PASSENGERS. So we can remove one of both. 

We are good with one reference of origin and destination so removing extra columns suggesting same. Which are ORIGIN_COUNTRY, ORIGIN_STATE_FIPS, ORIGIN_STATE_ABR, ORIGIN_STATE_NM, DEST_COUNTRY, DEST_STATE_FIPS, DEST_STATE_ABR, DEST_STATE_NM.

Ticketing, Operating and Reporting carrier should be same as we can ignore the customers, who changed the carrier. And there should not be any career change. Hence TK_CARRIER_CHANGE and OP_CARRIER_CHANGE should be 0 and then we can remove both the columns. 

We can also drop carrier groups for ticketing, operating and reporting. Hence, TK_CARRIER_GROUP and OP_CARRIER_GROUP can be dropped. 

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Missing value Handling 

There are no missing values after data filtering. 

![image](https://user-images.githubusercontent.com/74948720/165873345-ede55a75-a731-420b-8f71-b5412983f1a9.png)

As per the boxplot we can remove the values less than 50 and above 1000 for fare price. 

![image](https://user-images.githubusercontent.com/74948720/165873381-86592818-7263-4069-b9e3-9c0e31532757.png)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Additional Screen 

We are also planning to add pandas profiling report as a separate screen which will be using the raw data set. 

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Data Modeling Plan and Prediction

We will be dividing data into two parts, one for training the model and other for the testing. 

We will be using logistic regression and random forest regression models to train and predict the data.

Based on the accuracy, we will select the most efficient model for predicting the prices based on the parameters given.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Conclusion

We will follow the plan mentioned above and if required add additional functionalities based on the requirements. 

The app will not look exactly like the given format but we are sure that we will add more than what we have committed in this presentation. 

We will host our application on herokuapp.com and will update our code and dataset on github. 

This project can be used to implement the system where prices of the air travel needs to be predicted, where user can see prices for different future time and based on that they can plan the journey or vacation. For example, booking.com can implement this model to predict the estimated price.












