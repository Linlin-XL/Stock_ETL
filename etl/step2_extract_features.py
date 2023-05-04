import logging
import os
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException

logger = logging.getLogger(__name__)


def main(spark, stock_price_daily, stock_price_staging):
    try:
        df_daily = spark.read.parquet(stock_price_daily)
    except AnalysisException as e:
        logger.warning(f'file {stock_price_daily} does not exists!')
        raise e

    df_daily.printSchema()

    # we need this timestampGMT as seconds for our Window time frame
    df_daily = df_daily.withColumn('TimeStamp', F.unix_timestamp(F.to_timestamp('Date')).cast('long'))

    days = lambda i: i * 86400

    # we need this timestampGMT as seconds for our Window time frame
    # Calculate the moving average of the trading volume (Volume) of 30 days per each stock and ETF
    window_30days = Window.partitionBy('Symbol').orderBy(F.col("TimeStamp")).rangeBetween(-days(30), 0)

    # Update the moving average volume and median adj_close in 30days window
    # Note the OVER clause added to AVG(), to define a windowing column.
    df_daily = df_daily.withColumn('vol_moving_avg', F.avg('Volume').over(window_30days).cast('int')) \
        .withColumn('adj_close_rolling_med', F.percentile_approx('Adj Close', 0.5).over(window_30days).cast('float'))

    df_daily.show(20)

    # Export the staging parquet
    data_dir = os.path.dirname(stock_price_staging)
    os.makedirs(data_dir, exist_ok=True)
    df_daily.drop('TimeStamp').write.partitionBy('Symbol').mode("overwrite").parquet(stock_price_staging)
