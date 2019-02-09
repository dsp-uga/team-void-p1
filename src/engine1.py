import os.path
import re
import numpy as np
from operator import add
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import DoubleType
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer

DEBUG = True

def cal_accuracy(label_list, pred_list):
    cnt = 0
    ttl = len(label_list)
    for doc in range(ttl):
        pred_list[doc] = str(pred_list[doc])
        if pred_list[doc] == label_list[doc]: 
            cnt += 1
    accuracy = cnt // ttl
    return accuracy


def output_file(pred_list, output_path):
    with open(output_path, "w") as f:
        for pred_label in pred_list:
            f.write('%d\n' % pred_label)
    

if __name__ == "__main__":
    sc = SparkContext()
    spark = SparkSession.builder.master("local").appName("Word Count").config("spark.some.config.option", "some-value").getOrCreate()
    
    num_trees = 50
    max_depth = 10

    if DEBUG:
        fea1_train_path = '/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p1/team-void-p1/features/segment_train' 
        fea1_test_path = '/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p1/team-void-p1/features/segment_test'
    else:
        fea1_train_path = 'gs://uga-8360-projects/features/segment_train_jx' 
        fea1_test_path = 'gs://uga-8360-projects/features/segment_test_jx'
    

    rdd_train = spark.read.parquet(fea1_train_path).rdd.map(tuple)
    rdd_test = spark.read.parquet(fea1_test_path).rdd.map(tuple)

    rdd_train = rdd_train
    rdd_test = rdd_test.map(lambda x: (x[1], x[0], x[2]))


    print('***** Training Dataframe ******************************************')
    df_train = spark.createDataFrame(rdd_train, ['docid', 'hash', 'label', 'features'])
    print(df_train.show(n=2, truncate=140))
    print('***** Testing Dataframe ******************************************')
    df_test = spark.createDataFrame(rdd_test, ['docid', 'hash', 'features'])
    print(df_test.show(n=2, truncate=140))

    # ------------------------------------------------------------------------


    # Random Forest Classification
    ##########################################################################
    rf = RandomForestClassifier(numTrees=num_trees, maxDepth=max_depth)
    model = rf.fit(df_train.withColumn("label", df_train["label"].cast(DoubleType())))

    pred = model.transform(df_test)
    pred = pred.withColumn("prediction", pred["prediction"].cast("int"))

    y_test = pred.select("docid", "prediction").rdd.map(tuple).sortByKey().map(lambda x: x[1]).collect()

    # Accuracy
    # rdd_ytest = sc.textFile('gs://uga-dsp/project1/files/y_small_test.txt')
    # accuracy = cal_accuracy(rdd_ytest.collect(), y_test)
    # print('**********************************************')
    # print('Testing Accuracy: %.2f %%' % (accuracy*100))
    # print('**********************************************')

    # Output file
    output_file(y_test, 'prediction1.txt')
