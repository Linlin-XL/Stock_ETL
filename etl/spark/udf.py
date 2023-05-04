import pyspark.sql.functions as F
from pyspark.sql.types import StringType, FloatType, IntegerType
import os
import numpy as np


def get_last_element(data):
    return data[-1]


get_last_element_udf = F.udf(get_last_element)


def get_symbol_filename(data):
    file = os.path.basename(data)
    return file[:-4]


get_symbol_filename_udf = F.udf(get_symbol_filename, StringType())
