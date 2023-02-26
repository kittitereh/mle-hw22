from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import Normalizer
from numpy import load
from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import IDF
import numpy as np
import pandas as pd
from pyspark.sql.functions import col, collect_list
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, IntegerType
from pyspark.ml.feature import Normalizer
import pyspark.sql.functions as psf
from pyspark.sql.types import DoubleType
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pyspark.sql.functions import udf


class Train:

    def load_data_make_df(self, train_data, test_data):
        train_data = load(train_data)['arr_0'][:10000]
        test_data = load(test_data)['arr_0'][:10000]
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)
        return train_df, test_df
    
    def make_spark_df(self, train_df, test_df):

        mySchema = StructType([ StructField("user", IntegerType(), True),StructField("film", IntegerType(), True)])

        train = spark.createDataFrame(train_df, schema=mySchema)
        test = spark.createDataFrame(test_df, schema=mySchema)

        train.select('user').distinct().rdd.map(lambda r: r[0]).collect()

        # получаем df с 2 столбцами: film и collect_list(user)
        films_by_users= train.groupBy("film").agg(collect_list("user"))

        return  train, test
        
    def tf_idf(self, train):
        
        hashingTF = HashingTF(inputCol="collect_list(user)", outputCol="tf",numFeatures=10000, )
        tf = hashingTF.transform(films_by_users)
        idf = IDF(inputCol="tf", outputCol="idf")
        tfidf = idf.fit(tf).transform(tf)
        
        # compute L2 norm
        normalizer = Normalizer(inputCol="idf", outputCol="norm")
        data = normalizer.transform(tfidf)

        # multiply the table with itself to get the cosine similarity as the dot product of two by two L2norms
        dot_udf = psf.udf(lambda x,y: float(x.dot(y)), DoubleType())
        res = data.alias("i").join(data.alias("j"), psf.col("i.film") < psf.col("j.film"))\
            .select(
                psf.col("i.film").alias("i"), 
                psf.col("j.film").alias("j"), 
                dot_udf("i.norm", "j.norm").alias("dot"))\
            .sort("i", "j")

        mat = IndexedRowMatrix(
            data.select("film", "norm")\
                .rdd.map(lambda row: IndexedRow(row.film, row.norm.toArray()))).toBlockMatrix()
        dot = mat.multiply(mat.transpose())
        res = tfidf.rdd.map(lambda x : (x.film, x["collect_list(user)"], x.tf, x.idf,(None if x.idf is None else x.idf.values.sum())))
        sum_ = udf(lambda v: float(v.values.sum()), DoubleType())

        return res
    
    

if __name__ == "__main__":
    
    # Create SparkSession
    conf = SparkConf().setMaster("local[*]").setAppName("TFIDF")
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    
    train = Train()
    train, test = train.load_data_make_df('trainx16x32_0.npz','testx16x32_0.npz')
    train_df = train.make_spark_df(train, test)
    tfidf = train.tf_idf(train_df)
    
