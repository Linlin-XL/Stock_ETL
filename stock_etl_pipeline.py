import copy
import pandas as pd
from etl.spark.session import get_spark_session, spark_default_config
from etl import step1_ingest_data, step2_extract_features, step3_train_model
import os

def read_stock_price_parquet(parquet_path):
    pddf = pd.read_parquet(parquet_path)
    print(pddf.columns)
    print(pddf.head(5))


def main(spark, stock_csv, symbol_csv, stock_price_daily, stock_price_staging, joblib_stock_model):
    # Step 1 - Ingest the CSV file
    print('Step 1 - Ingest the CSV file')
    step1_ingest_data.main(spark, stock_csv, symbol_csv, stock_price_daily)
    read_stock_price_parquet(stock_price_daily)

    # Step 2 - Feature Engineering
    print('Step 2 - Feature Engineering')
    step2_extract_features.main(spark, stock_price_daily, stock_price_staging)
    read_stock_price_parquet(stock_price_staging)

    # Step 3 - ML Training
    print('Step 3 - ML Training')
    step3_train_model.main(stock_price_staging, joblib_stock_model)


if __name__ == '__main__':
    spark_config = copy.deepcopy(spark_default_config)
    spark = get_spark_session('Stock ETL Pipeline', config=spark_config, log_level='Warn')
    try:
        main(spark, ['data/input/etfs', 'data/input/stocks'], 'data/input/symbols_valid_meta.csv',
             'data/output/stock_daily_source',  'data/output/stock_daily_staging',
             'data/output/stock_regression_v01.joblib')
    finally:
       spark.stop()

