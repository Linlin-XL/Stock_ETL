import joblib


def main(vol_moving_avg, adj_close_rolling_med):
    joblib_stock_model_path = 'data/output/stock_regression_v01.joblib'

    # Load the model
    regression_model = joblib.load(joblib_stock_model_path)
    print('Load the regression model:', joblib_stock_model_path)

    # Make predictions
    request_data = [[vol_moving_avg, adj_close_rolling_med]]
    print('Request_data:', request_data)
    y_pred = regression_model.predict(request_data)
    print('Prediction: %d' % y_pred[0])
    return y_pred[0]


if __name__ == '__main__':
    main(12345, 25)
