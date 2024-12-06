# Predicting NFL Match Spreads Using Historical Match Data and Neural Networks

## Project’s function:
Predicting NFL game outcomes has long been of interest to sports analysts and enthusiasts alike. the project centers on forecasting the point spread of NFL games by leveraging historical data on game scores, weather conditions, and stadium information. Using a Keras-based neural network model, this project aims to predict future spreads by examining trends in historical data. The goal of the project is to develop a predictive model that can estimate the point spread of NFL games based on historical data, specifically incorporating final game scores, weather details, stadium information, and home-field advantage. The model’s accuracy will be evaluated using Mean Squared Error (MSE), with comparisons between predicted and actual spreads. 90\% of the dataset will be allocated for training the model, with the remaining 10\% used for testing and evaluation.

## Dataset
The data set used was developed by spreadspoke and was located on kaggle. The following description of the dataset is provided by the owner:
"National Football League (NFL) game results since 1966 with betting odds information since 1979. Dataset was created from a variety of sources including games and scores from a variety of public websites such as ESPN, NFL.com, and Pro Football Reference. Weather information is from NOAA data with NFLweather.com a good cross reference. Betting data was used from http://www.repole.com/sun4cast/data.html for 1978-2013 seasons. Pro-football-reference.com data was then cross referenced for betting lines and odds as well as weather data. From 2013 on betting data reflects lines available at sportsline.com and aussportsbetting.com."
Additionally, the dataset was altered to include a column that shows the resulting spread of the game and adjusted the column for the predicted (betting line, not NN) spread by making it relative to the home team.

## Pipeline / Architecture
The project employs an Apache Airflow pipeline for systematic data cleaning and preprocessing.

## Data Quality Assessment
The dataset demonstrates a moderate-to-high quality status, assessed across dimensions of accuracy, completeness, consistency, timeliness, and reliability. Its strengths include comprehensive game results since 1966, cross-referenced data from reputable sources like ESPN, NFL.com, NOAA, and Pro Football Reference, and betting odds spanning 1978 onward. However, gaps exist in pre-1978 betting data, and source shifts over time create potential inconsistencies, especially in weather and betting information. Timeliness depends on regular updates, and while game results and weather data are reliable, betting odds have moderate reliability due to limited historical sources. This assessment was based on examining the dataset’s source diversity, the consistency of reported data, and its coverage over time. To enhance quality, the dataset would benefit from standardizing formats, filling data gaps, and improving documentation for better analytical utility and predictive accuracy.


## Data Transformation Models used
For data cleaning the following methods were used to transform the model. 
  Missing Data: Rows with missing values in essential columns are removed using the dropna function. This step ensured that each record in the dataset is complete and reliable. 
  Duplicate Removal: Duplicate records are dropped, via dropna, to reduce the redundancy of the dataset. 
  Outlier Filtering: Outliers in point spreads are removed to prevent excessively skewed results in the model’s predictions.
  Data Selection – Only relevant columns (such as game scores, weather details, and stadium data) are retained.

## Architecture
![System Architecture](https://github.com/user-attachments/assets/f033b044-be98-47f4-a858-bb29c0b22cac)


##Thorough Investigation
