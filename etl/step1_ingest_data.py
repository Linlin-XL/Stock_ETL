import logging
import os
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, FloatType, IntegerType
import pyspark.sql.functions as F
from pyspark.sql.utils import ParseException
from etl.spark.udf import get_symbol_filename_udf

logger = logging.getLogger(__name__)


def main(spark, stock_csv, symbol_meta_csv, data_output_path):
    # Define Stock Price Schema
    schema = StructType([
        StructField("Date", StringType()),
        StructField("Open", FloatType()),
        StructField("High", FloatType()),
        StructField("Low", FloatType()),
        StructField("Close", FloatType()),
        StructField("Adj Close", FloatType()),
        StructField("Volume", IntegerType())
    ])

    try:
        # Load the Security Daily Price CSV
        df_daily = spark.read.csv(stock_csv, schema=schema, sep=',', header=True)
    except ParseException as e:
        logger.error(str(e))
        raise e

    # Update the Symbol
    df_daily = df_daily.withColumn('Symbol', get_symbol_filename_udf(F.input_file_name()))
    print(df_daily.show(10))

    try:
        # Load the Security Name CSV
        df_symbol = spark.read.csv(symbol_meta_csv, sep=',', inferSchema=True, header=True)
    except ParseException as e:
        logger.error(str(e))
        raise e

    # Update the Security Name
    df_out = df_daily.join(df_symbol.select(['Symbol', 'Security Name']), on='Symbol', how='left')
    print(df_out.show(10))
    df_out.printSchema()

    # Export the parquet
    data_dir = os.path.dirname(data_output_path)
    os.makedirs(data_dir, exist_ok=True)
    # df_out.write.partitionBy('Symbol').mode('overwrite').parquet(data_output_path)
    df_out.write.mode('overwrite').parquet(data_output_path)
    print('Export:', data_output_path)
