Project Title: Cryptocurrency Price Prediction using Time Series Analysis

Project Goal: To predict cryptocurrency prices using historical data and time series modeling techniques.

Data Sources:

coin_gecko_2022-03-16.csv
coin_gecko_2022-03-17 (1).csv
Key Libraries Used:

pandas
matplotlib
seaborn
sklearn (for splitting, scaling, and model evaluation)
statsmodels (for ARIMA model)
pmdarima (for auto-ARIMA)
joblib (for saving the model)
Project Steps:

Data Loading and Merging: Load two CSV files and concatenate them into a single pandas DataFrame.
Data Exploration and Preprocessing (EDA):
Display basic information (.info(), .describe(), .shape).
Check for and handle duplicate rows.
Identify and fill missing values (using the median).
Convert the 'date' column to datetime objects.
Perform initial visualizations (histograms, scatter plots, box plots) to understand data distributions and relationships.
Generate detailed summary statistics for key columns.
Analyze categorical data (e.g., 'symbol' counts).
Visualize time series trends.
Generate a correlation matrix for numerical columns.
Feature Engineering:
Create lagged price features (price_lag_1, price_lag_7).
Create rolling window features (price_rolling_mean_7, price_rolling_std_7).
Extract time-based features from the 'date' column (year, month, day, day of week, day of year).
Create interaction features (price_diff_rolling_mean_7).
Model Selection: Based on the data characteristics and the problem (time series prediction), the ARIMA model and RandomForestRegressor are considered.
Data Splitting and Scaling:
Define features (X) and target (y).
Split the data chronologically into training and testing sets.
Scale numerical features using MinMaxScaler to a range between 0 and 1.
Model Training and Evaluation:
RandomForestRegressor:
Use a Pipeline for scaling and model training.
Perform hyperparameter tuning using GridSearchCV with TimeSeriesSplit for cross-validation.
Evaluate the best model using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
Analyze feature importances.
ARIMA:
Train an ARIMA model on the target variable (y_train).
Evaluate the ARIMA model's forecast using MSE, RMSE, and MAE.
Model Comparison: Compare the evaluation metrics of the trained models (RandomForestRegressor and ARIMA) to determine the best performing model. The ARIMA model is deemed the best fit for this problem based on the results.
Model Saving: Save the trained ARIMA model using joblib for future use.
Conclusion: The ARIMA model demonstrated better performance for this time series price prediction task. The trained model is saved for deployment or further analysis.
