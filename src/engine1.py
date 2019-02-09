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

def _format_(rdd, feature_path):
    f = feature_path.lower()
    if 'train' in f:
        if 'segment' in f: 
            rdd_fea = rdd.map(lambda x: ((x[0], x[1], int(x[2])), x[3]))
        elif 'gram' in f: 
            rdd_fea = rdd.map(lambda x: ((x[0], x[1], x[3]), Vectors.dense(list(x[2].toArray()))))
        elif 'file' in f: 
            rdd_fea = rdd.map(lambda x: ((x[0], x[1], x[5]), Vectors.dense(list((x[2], x[3], x[4])))))
    if 'test' in f:
        if 'segment' in f: 
            rdd_fea = rdd.map(lambda x: ((x[1], x[0]), x[2]))
        elif 'gram' in f: 
            rdd_fea = rdd.map(lambda x: ((x[0], x[1]), Vectors.dense(list(x[2].toArray()))))
        elif 'file' in f: 
            rdd_fea = rdd.map(lambda x: ((x[0], x[1]), Vectors.dense(list((x[2], x[3], x[4])))))
    return rdd_fea


def RF_format(df):
    # Input >> DF(docid, hash, label, vector.dense(features))
    stringIndexer = StringIndexer(inputCol = "hash", outputCol = "indexed")
    si_model = stringIndexer.fit(df)
    td = si_model.transform(df)
    return td


def RF_model(td, n, m):
    td_new = td.withColumn("label", td["label"].cast(DoubleType()))
    rf = RandomForestClassifier(numTrees=n, maxDepth=m)
    model = rf.fit(td_new)
    return model


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
        fea1_train_path = '/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p1/team-void-p1/features/segment_train_jx' 
        fea1_test_path = '/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p1/team-void-p1/features/segment_test_jx'
        fea2_train_path = '/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p1/team-void-p1/features/opcode_1gram_train' 
        fea2_test_path = '/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p1/team-void-p1/features/opcode_1gram_test'
    else:
        fea1_train_path = 'gs://uga-8360-projects/features/segment_train_jx' 
        fea1_test_path = 'gs://uga-8360-projects/features/segment_test_jx'
        fea2_train_path = 'gs://uga-8360-projects/features/opcode_1gram_train' 
        fea2_test_path = 'gs://uga-8360-projects/features/opcode_1gram_test'

    ## features, (segment, opcode-bigram)
    # ------------------------------------------------------------------------
    fea1_train_rdd = spark.read.parquet(fea1_train_path).rdd.map(tuple)
    fea1_train = _format_(fea1_train_rdd, fea1_train_path.split('/')[-1])
    fea2_train_rdd = spark.read.parquet(fea2_train_path).rdd.map(tuple)
    fea2_train = _format_(fea2_train_rdd, fea2_train_path.split('/')[-1])
    
    rdd_train = fea1_train.leftOuterJoin(fea2_train).map(lambda x: (x[0][0], x[0][1], x[0][2], Vectors.dense(list(x[1][0]) + list(x[1][1]))))

    fea1_test_rdd = spark.read.parquet(fea1_test_path).rdd.map(tuple)
    fea1_test = _format_(fea1_test_rdd, fea1_test_path.split('/')[-1])
    fea2_test_rdd = spark.read.parquet(fea2_test_path).rdd.map(tuple)
    fea2_test = _format_(fea2_test_rdd, fea2_test_path.split('/')[-1])
    
    rdd_test = fea1_test.leftOuterJoin(fea2_test).map(lambda x: (x[0][0], x[0][1], Vectors.dense(list(x[1][0]) + list(x[1][1]))))

    print('***********************************************')
    print('Training Dataframe')
    df_train = spark.createDataFrame(rdd_train, ['docid', 'hash', 'label', 'features'])
    # print(df_train.show(n=2, truncate=140))
    
    print('***********************************************')
    print('Testing Dataframe')
    df_test = spark.createDataFrame(rdd_test, ['docid', 'hash', 'features'])
    # print(df_test.show(n=2, truncate=140))

    # ------------------------------------------------------------------------


    # Random Forest Classification
    ##########################################################################
    model = RF_model(df_train, n=num_trees, m=max_depth)
    # pred = model.transform(RF_format(df_test))
    pred = model.transform(df_test)
    pred = pred.withColumn("prediction", pred["prediction"].cast("int"))
    print(pred.show(n=10))
    y_test = pred.select("docid", "prediction").rdd.map(tuple).sortByKey().map(lambda x: x[1]).collect()

    # Accuracy
    # rdd_ytest = sc.textFile('files/y_small_test.txt')
    # accuracy = cal_accuracy(rdd_ytest.collect(), y_test)
    # print('Testing Accuracy: %.2f %%' % (accuracy*100))
    # print('**********************************************')

    # Output file
    output_file(y_test, 'prediction1.txt')
