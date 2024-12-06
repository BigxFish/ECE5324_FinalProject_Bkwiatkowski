# Predicting NFL Match Spreads Using Historical Match Data and Neural Networks

## Project’s function:
Predicting NFL game outcomes has long been of interest to sports analysts and enthusiasts alike. The project centers on forecasting the point spread of NFL games by leveraging historical data on game scores, weather conditions, and stadium information. Using a Keras-based neural network model, this project aims to predict future spreads by examining trends in historical data. The goal of the project is to develop a predictive model that can estimate the point spread of NFL games based on historical data, specifically incorporating final game scores, weather details, stadium information, and home-field advantage. The model’s accuracy will be evaluated using Mean Squared Error (MSE), with comparisons between predicted and actual spreads. 90\% of the dataset will be allocated for training the model, with the remaining 10\% used for testing and evaluation.

## Dataset
The data set used was developed by spreadspoke and was located on kaggle. The following description of the dataset is provided by the owner:
"National Football League (NFL) game results since 1966 with betting odds information since 1979. Dataset was created from a variety of sources including games and scores from a variety of public websites such as ESPN, NFL.com, and Pro Football Reference. Weather information is from NOAA data with NFLweather.com a good cross reference. Betting data was used from http://www.repole.com/sun4cast/data.html for 1978-2013 seasons. Pro-football-reference.com data was then cross referenced for betting lines and odds as well as weather data. From 2013 on betting data reflects lines available at sportsline.com and aussportsbetting.com."
Additionally, the dataset was altered to include a column that shows the resulting spread of the game and adjusted the column for the predicted (betting line, not NN) spread by making it relative to the home team.

## Pipeline / Architecture
The project utilized Apache Airflow to build and automate the data pipeline, orchestrating tasks such as data ingestion, cleaning, and transformation. Airflow's flexibility  ensured efficient scheduling and monitoring of workflows, supporting the project's nature. For the predictive modeling, the Keras neural network (NN) method was employed, leveraging its simplicity and power to achieve accurate NFL spread predictions. The combination of Airflow for pipeline management and Keras for modeling ensured a streamlined, scalable, and effective workflow for the project.

## Data Quality Assessment
The dataset demonstrates a moderate-to-high quality status, assessed across dimensions of accuracy, completeness, consistency, timeliness, and reliability. Its strengths include comprehensive game results since 1966, cross-referenced data from reputable sources like ESPN, NFL.com, NOAA, and Pro Football Reference, and betting odds spanning 1978 onward. However, gaps exist in pre-1978 betting data, and source shifts over time create potential inconsistencies, especially in weather and betting information. Timeliness depends on regular updates, and while game results and weather data are reliable, betting odds have moderate reliability due to limited historical sources. This assessment was based on examining the dataset’s source diversity, the consistency of reported data, and its coverage over time. To enhance quality, the dataset would benefit from standardizing formats, filling data gaps, and improving documentation for better analytical utility and predictive accuracy.


## Data Transformation Models used
For data cleaning the following methods were used to transform the model. 
  Missing Data: Rows with missing values in essential columns are removed using the dropna function. This step ensured that each record in the dataset is complete and reliable. 
  Duplicate Removal: Duplicate records are dropped, via dropna, to reduce the redundancy of the dataset. 
  Outlier Filtering: Outliers in point spreads are removed to prevent excessively skewed results in the model’s predictions.
  Data Selection – Only relevant columns (such as game scores, weather details, and stadium data) are retained.

## Instructions
In order for the project to be run correctly, similarly to lab 3, the .pkl files must be developed using an initial apache airflow run. Then, using the final part of the dag to generate the keras model.
Before anything else, to replicate the data.pkl file, the excel_pkl_transform.py file must be run in pycharm or vscode. Then, adjusting the dag, transform, featureExtraction, and build_train_model files accordingly, the project can be recreated. The graphing_file.py and testing_model.py files can be used to create the graphics and find the mse for the trained dataset.

## Architecture
![System Architecture](https://github.com/user-attachments/assets/f033b044-be98-47f4-a858-bb29c0b22cac)

## Results
![System Architecture (1)](https://github.com/user-attachments/assets/49b67375-3192-41a1-aa3f-5ce93714740d)

The project resulted in a very accurate training system resulting with the following errors:
Mean Squared Error (MSE): 0.0002537434708363646
Mean Absolute Error (MAE): 0.01132325333643287

## Thorough Investigation
### Scaling the Project
To scale this project, the dataset must be expanded and the model’s robustness improved. Current results show high accuracy, but the limited scope of historical game results and betting odds restricts generalization. Incorporating recent NFL seasons, contextual factors like injuries and team dynamics, and live betting odds is essential. Cloud platforms like AWS or Google Cloud could handle increased computational demands, while automated pipelines for deployment and monitoring would streamline scalability.

### Innovativeness
This project demonstrates strong innovation by applying neural networks to a domain traditionally reliant on intuition and analytics. Its integration of diverse data streams like game results, betting odds, and weather data offers a unique approach. To elevate its impact, future iterations could focus on real-time decision-making tools or extend predictions to other sports and betting contexts.

### Challenges and Limitations
Key challenges include data quality issues, such as inconsistencies in betting odds and gaps in historical records, as well as risks of overfitting due to rapid MSE convergence. Computational demands will increase with dataset scaling, requiring distributed systems or specialized hardware. Additionally, integrating and cleaning data from various sources remains a time-intensive process.

### Next Steps
Future efforts should focus on expanding the dataset with richer features like player and team stats and testing advanced architectures like LSTMs or transformers. Automating data pipelines would improve efficiency, while broadening use cases to include team performance forecasts or injury analyses could extend the project’s value. Collaborating with domain experts would further refine features and ensure actionable insights.
