# Work Sample for Data Engineer

To effectively solve the following data pipeline problems, it is essential to use a DAG (Directed Acyclic Graph) oriented tool. DAG tools like Pachyderm, Airflow, Dagster, etc., can help streamline data processing and management with tracking data lineage, ensuring data integrity, and minimizing errors during processing.

To provide more context and clarity, including pipeline specs and diagrams can be helpful. These artifacts can help visualize the DAG and its components, provide information on how data flows through the pipeline, and highlight the dependencies between tasks.

## Problem 1: Raw Data Processing

**Objective**: Ingest and process raw stock market datasets.

### Tasks:
1. Download the ETF and stock datasets from the primary dataset available at https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset.
2. Setup a data structure to retain all data from ETFs and stocks with the following columns.
    ```
    Symbol: string
    Security Name: string
    Date: string (YYYY-MM-DD)
    Open: float
    High: float
    Low: float
    Close: float
    Adj Close: float
    Volume: int
    ```
3. Convert the resulting dataset into a structured format (e.g. Parquet).

## Problem 2: Feature Engineering

**Objective**: Build some feature engineering on top of the dataset from Problem 1.

### Tasks:
1. Calculate the moving average of the trading volume (`Volume`) of 30 days per each stock and ETF, and retain it in a newly added column `vol_moving_avg`.
2. Similarly, calculate the rolling median and retain it in a newly added column `adj_close_rolling_med`.
3. Retain the resulting dataset into the same format as Problem 1, but in its own stage/directory distinct from the first.
4. (Bonus) Write unit tests for any relevant logic.

## Problem 3: Integrate ML Training

**Objective**: Integrate an ML predictive model training step into the data pipeline.

You can use the following simple Random Forest model as a reference:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Assume `data` is loaded as a Pandas DataFrame
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Remove rows with NaN values
data.dropna(inplace=True)

# Select features and target
features = ['vol_moving_avg', 'adj_close_rolling_med']
target = 'Volume'

X = data[features]
y = data[target]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate the Mean Absolute Error and Mean Squared Error
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
```

You may come up with your own process with any choice of model architectures, algorithms, libraries, and training configurations.

### Tasks:
1. Integrate the ML training process as a part of the data pipeline.
2. Save the resulting model to disk.
3. Persist any training metrics, such as loss and error values as log files.
4. (Bonus) If you choose your own model implementation, articulate why it's better as a part of your submission.
