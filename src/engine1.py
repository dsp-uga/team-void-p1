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
# import xgboost as xgb


DEBUG = False
NUM_FEA = 2

if DEBUG:
    FEA1_TRAIN_PATH = '/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p1/team-void-p1/features/segment_train' 
    FEA1_TEST_PATH = '/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p1/team-void-p1/features/segment_test'
    FEA2_TRAIN_PATH = '/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p1/team-void-p1/features/bytes_2gram_train' 
    FEA2_TEST_PATH = '/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p1/team-void-p1/features/bytes_2gram_test'
    FEA3_TRAIN_PATH = '/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p1/team-void-p1/features/bytes_3gram_train' 
    FEA3_TEST_PATH = '/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p1/team-void-p1/features/bytes_3gram_test'
else:
    FEA1_TRAIN_PATH = 'gs://uga-8360-projects/features/segment_train' 
    FEA1_TEST_PATH = 'gs://uga-8360-projects/features/segment_test'
    FEA2_TRAIN_PATH = 'gs://uga-8360-projects/features/bytes_2gram_train' 
    FEA2_TEST_PATH = 'gs://uga-8360-projects/features/bytes_2gram_test'
    FEA3_TRAIN_PATH = 'gs://uga-8360-projects/features/bytes_3gram_train' 
    FEA3_TEST_PATH = 'gs://uga-8360-projects/features/bytes_3gram_test'


def _format_(rdd, feature_path):
    f = feature_path.lower()
    if 'train' in f:
        if 'segment' in f: 
            rdd_fea = rdd.map(lambda x: ((x[0], x[1], int(x[2])), x[3]))
        elif 'gram' in f: 
            rdd_fea = rdd.map(lambda x: ((x[0], x[1], x[3]), Vectors.dense(list(x[2].toArray()))))
    if 'test' in f:
        if 'segment' in f: 
            rdd_fea = rdd.map(lambda x: ((x[1], x[0]), x[2]))
        elif 'gram' in f: 
            rdd_fea = rdd.map(lambda x: ((x[0], x[1]), Vectors.dense(list(x[2].toArray()))))
    return rdd_fea


def get_dataframe(NUM_FEA):
    if NUM_FEA == 1:
        ## features: (segments)
        # ------------------------------------------------------------------------
        rdd_train = spark.read.parquet(FEA1_TRAIN_PATH).rdd.map(tuple)
        rdd_test = spark.read.parquet(FEA1_TEST_PATH).rdd.map(tuple)

        if 'segment' in FEA1_TRAIN_PATH.lower():
            rdd_test = rdd_test.map(lambda x: (x[1], x[0], x[2]))
        if 'gram' in FEA1_TRAIN_PATH.lower():
            rdd_train = rdd_train.map(lambda x: (x[0], x[1], x[3], Vectors.dense(list(x[2].toArray()))))
            rdd_test = rdd_test.map(lambda x: (x[0], x[1], Vectors.dense(list(x[2].toArray()))))

    elif NUM_FEA == 2:
        ## features, (segments, bytes_2gram)
        # ------------------------------------------------------------------------
        fea1_train_rdd = spark.read.parquet(FEA1_TRAIN_PATH).rdd.map(tuple)
        fea1_train = _format_(fea1_train_rdd, FEA1_TRAIN_PATH.split('/')[-1])
        fea2_train_rdd = spark.read.parquet(FEA2_TRAIN_PATH).rdd.map(tuple)
        fea2_train = _format_(fea2_train_rdd, FEA2_TRAIN_PATH.split('/')[-1])
        
        rdd_train = fea1_train.leftOuterJoin(fea2_train).map(lambda x: (x[0][0], x[0][1], x[0][2], Vectors.dense(list(x[1][0]) + list(x[1][1]))))

        fea1_test_rdd = spark.read.parquet(FEA1_TEST_PATH).rdd.map(tuple)
        fea1_test = _format_(fea1_test_rdd, FEA1_TEST_PATH.split('/')[-1])
        fea2_test_rdd = spark.read.parquet(FEA2_TEST_PATH).rdd.map(tuple)
        fea2_test = _format_(fea2_test_rdd, FEA2_TEST_PATH.split('/')[-1])
        
        rdd_test = fea1_test.leftOuterJoin(fea2_test).map(lambda x: (x[0][0], x[0][1], Vectors.dense(list(x[1][0]) + list(x[1][1]))))

    elif NUM_FEA == 3:
        ## features, (segments, bytes_2gram, bytes_3gram)
        # ------------------------------------------------------------------------
        fea1_train_rdd = spark.read.parquet(FEA1_TRAIN_PATH).rdd.map(tuple)
        fea1_train = _format_(fea1_train_rdd, FEA1_TRAIN_PATH)
        fea2_train_rdd = spark.read.parquet(FEA2_TRAIN_PATH).rdd.map(tuple)
        fea2_train = _format_(fea2_train_rdd, FEA2_TRAIN_PATH)
        fea3_train_rdd = spark.read.parquet(FEA3_TRAIN_PATH).rdd.map(tuple)
        fea3_train = _format_(fea3_train_rdd, FEA3_TRAIN_PATH)
        rdd_train = fea1_train.leftOuterJoin(fea2_train)\
                        .map(lambda x: (x[0], Vectors.dense(list(x[1][0]) + list(x[1][1]))))\
                        .leftOuterJoin(fea3_train)\
                        .map(lambda x: (x[0][0], x[0][1], x[0][2], Vectors.dense(list(x[1][0]) + list(x[1][1]))))

        fea1_test_rdd = spark.read.parquet(FEA1_TEST_PATH).rdd.map(tuple)
        fea1_test = _format_(fea1_test_rdd, FEA1_TEST_PATH)
        fea2_test_rdd = spark.read.parquet(FEA2_TEST_PATH).rdd.map(tuple)
        fea2_test = _format_(fea2_test_rdd, FEA2_TEST_PATH)
        fea3_test_rdd = spark.read.parquet(FEA3_TEST_PATH).rdd.map(tuple)
        fea3_test = _format_(fea3_test_rdd, FEA3_TEST_PATH)
        rdd_test = fea1_test.leftOuterJoin(fea2_test)\
                        .map(lambda x: (x[0], Vectors.dense(list(x[1][0]) + list(x[1][1]))))\
                        .leftOuterJoin(fea3_test)\
                        .map(lambda x: (x[0][0], x[0][1], Vectors.dense(list(x[1][0]) + list(x[1][1]))))
    else:
        exit()

    print('***********************************************')
    print('Training Dataframe')
    df_train = spark.createDataFrame(rdd_train, ['docid', 'hash', 'label', 'features'])
    # print(df_train.show(n=2, truncate=140))
    
    print('***********************************************')
    print('Testing Dataframe')
    df_test = spark.createDataFrame(rdd_test, ['docid', 'hash', 'features'])
    # print(df_test.show(n=2, truncate=140))

    return df_train, df_test


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
    max_depth = 25

    df_train, df_test = get_dataframe(NUM_FEA)

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
    # print('Testing Accuracy: %.2f %%' % (accuracy*100))
    # print('**********************************************')

    # Output file
    output_file(y_test, 'prediction.txt')
