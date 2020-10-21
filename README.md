# airbnb_boston_ds
Airbnb Boston Data science project using Kaggle data sets


## Project Motivation
Learn to follow the CRISP-DM process to answer questions and potentially make predictions about Airbnb Boston data.

For this project the primary question is to compare Airbnb prices to hotel prices in Boston circa 2016-2017 to find out which costs more.

Secondary questions are around property locations by neighborhood.  What do prices look like by neighborhood, how many properties are in a neighborhood, how big are the properties in the neighborhoods.

Another area of exploration relates to what types of properties are available for rent from Airbnb Boston.

Conclusions:
* With some data cleaning and feature engineering, the data is suitable for predicting rental prices.
* Random Forest Regression modeling worked much better than Linear Regression.  There might be better results if more different types of models are explored.
* With various plots and queries, was able to find answers to many questions to find insights about prices per neighborhood in Boston, and property types and sizes, and even find the cost savings that could be realized by renting an Airbnb property rather than a hotel.

Read the blog post on Medium to see the results: https://medium.com/@alexanli.wk/this-will-make-you-think-twice-before-booking-a-hotel-in-boston-cdfb209c2093

## Installations
Jupyter notebook and helper files build using Python v3.8.3

Libraries Included:
* pandas
* numpy
* matplotlib.pyplot
* seaborn
* KMeans from sklearn.clusters
* PCA from sklearn.decomposition
* RandomForestRegressor from sklearn.ensemble
* RandomizedSearchCV from sklearn.model_selection
* sklearn.preprocessing
* sklearn.linear_model
* test_train_split from sklearn.model_selection
* r2_score from sklearn.metrics


## File Descriptions
* boston_airbnb_exploration.ipynb - Jupyter notebook used to follow CRSIP-DM process with Airbnb Boston data
* helper_data_wrangling.py - Python file with helper functions used to gather, clean, and prepare the data
* helper_modeling.py - Python file with helper functions used to create and tune predictive models
* helper_visuals.py - Python file with helper functions used to visualize the data

## How to use
* Download the project files
* Download the Airbnb Data from Kaggle into the same folder as the project files
* Launch Jupyter notebook from the folder where the project resides
* Follow the code and comments in the notebook

## Authors, Acknowledgements, Etc
* Author:  Lincoln Alexander
* Acknowledgements:  Udacity made me do it

* Primary Data Source: https://www.kaggle.com/airbnb/boston  download: calendar.csv and listings.csv

* Hotel Price Data Source: https://www.statista.com/statistics/202376/average-daily-rate-of-hotels-in-boston/
