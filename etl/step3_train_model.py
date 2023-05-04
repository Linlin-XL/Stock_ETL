import joblib
import os
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import tree

# import matplotlib.pyplot as plt
# import seaborn as sns


def main(staging_data_path, joblib_stock_model_path):
    # Load the staging parquet
    data_df = pd.read_parquet(staging_data_path)

    # Assume `data` is loaded as a Pandas DataFrame
    data_df['Date'] = pd.to_datetime(data_df['Date'])
    data_df.set_index('Date', inplace=True)

    # Remove rows with NaN values
    data_df.dropna(inplace=True)

    # Select features and target
    features = ['vol_moving_avg', 'adj_close_rolling_med']
    target = 'Volume'

    # Descriptive statistics for each column
    desc_df = data_df[features].describe()

    X = data_df[features]
    y = data_df[target]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f'Split the train data: {len(X_train)} - test data: {len(X_test)}')

    # Create a RandomForest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, joblib_stock_model_path)
    print('Save the regression model:', joblib_stock_model_path)

    # Load the model
    loaded_regr = joblib.load(joblib_stock_model_path)
    print('Load the regression model:', joblib_stock_model_path)

    # Make predictions on test data
    y_pred = loaded_regr.predict(X_test)

    # Calculate the Mean Absolute Error and Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    # accuracy = 100 - np.mean(mae)

    # Calculate and display accuracy
    train_output = [
        f'Mean Absolute Error: {mae}',
        f'Mean Squared Error: {mse}',
        f'Root Mean Squared Error: {np.sqrt(mse)}',
        # f'Accuracy: {round(accuracy, 2)}',
    ]
    print('Train Output:', train_output)
    output_df = pd.DataFrame([[i] for i in train_output], columns=['Metrics'])

    output_dict = {
        'Output': output_df, 'Data Describe': desc_df,
    }

    # Obtain the first regression tree
    tree_text = tree.export_text(loaded_regr.estimators_[0], feature_names=features, max_depth=20)
    # _ = tree.plot_tree(regr_tree, feature_names=features, fontsize=8, filled=True,  rounded=True);

    logs_path = os.path.dirname(joblib_stock_model_path)
    export_model_training_result(logs_path, output_dict, tree_text)


def export_model_training_result(logs_path, train_dict, tree_text):
    today = datetime.today().strftime('%Y%m%d')
    logs_output = os.path.join(logs_path, f'stock_regression_output-{today}.log')
    with open(logs_output, "a") as fp:
        # Write the model result
        for key, df in train_dict.items():
            fp.write(f"\n{key} Summary\n{'-'*60}\n")
            output_text = df.to_string(header=True, index=False)
            fp.write(output_text)

        fp.write(f"\n{'-'*60}\n")
        # Write the Tree
        fp.write(f"\nRegression Tree:\n{'-' * 60}\n{tree_text}\n\n\n")
    print('Write the regression model output:', logs_output)

