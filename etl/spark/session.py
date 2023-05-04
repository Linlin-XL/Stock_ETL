import logging
import os
import pathlib
import platform
from pyspark import SparkConf
from pyspark.sql import SparkSession
from etl.spark.utils import get_login_user

login_user = get_login_user(default_login='default_user')
if platform.system() == 'Windows':
    _spark_local_dir = os.path.join(pathlib.path.home(), 'spark-local')
    _spark_warehouse_dir = os.path.join(pathlib.path.home(), 'spark-warehouse')
else:
    _spark_local_dir = os.path.join('/tmp', 'spark-local')
    _spark_warehouse_dir = os.path.join('/tmp', 'spark-warehouse')

spark_default_config = {
    'spark.app.name': 'PySparkETL',
    'spark.master': f'local[{(os.cpu_count() - 1)}]',
    'spark.driver.memory': '1G',
    'spark.executor.memory': '1G',
    'spark.local.dir': _spark_local_dir,
    'spark.sql.warehouse.dir': _spark_warehouse_dir,
    'spark.sql.file.maxPartitionBytes': 32 * 1024 * 1024,
}


def get_spark_session(app_name, config=None, master=None, log_level='INFO'):
    spark_config = config or spark_default_config
    spark_config['spark.app.name'] = app_name
    spark_config['spark.master'] = master or spark_config['spark.master']

    conf = SparkConf().setAll(spark_config.items())
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    sc = spark.sparkContext
    sc.setLogLevel(log_level)

    logging.info(f'PySaprk version: {sc.version}')

    return spark
