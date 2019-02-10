import os.path
import re
import numpy as np
from operator import add
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import NGram
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import RandomForestClassifier


DEBUG = False
NUM_GRAM = 2

def get_specified_index_counts(feature_list, n):
    """
    feature_list: type list, a list contains (segment_idx, cnt)
    n: type int, total number of distince segments
    return: type list, a list contains 0s for non-specified index, but feature counts for specified index.
    """
    feature_array = np.zeros((n,), dtype = np.int)
    if feature_list == None:
        feature_array = np.zeros((n,), dtype = np.int)
    else:
        f = np.asarray(feature_list)
        feature_array[f[:,0]] = f[:,1]
    return list(feature_array)

def bytes_detect(bytes_content):
    """
    Detects bytess of the content of bytes file.
    Returns a bytess list.
    """
    return [word for word in bytes_content.split() if len(word) == 2 and word != '??']



def bytes_ngram(df_bytes, n):
    """
    Generates n-grams bytes by bytes data frame.
    Returns n-grams bytes in RDD((hash, n-gram), total_counts)
    """
    ngrams = NGram(n=n, inputCol="bytes", outputCol="ngrams")
    df_ngrams = ngrams.transform(df_bytes)
    rdd_ngrams = df_ngrams.select("hash", "ngrams").rdd.map(tuple).flatMapValues(lambda x: x)\
                    .map(lambda x: ((x[0], x[1]), 1)).reduceByKey(add)
    return rdd_ngrams


def RF_features_select(rdd_feature_vd, n=10, m=7):
    """
    Implements random forest classifier to the bytess counts in each document
    Returns the importance of each bytess
    >> Input (hash, label, features), Output (features, importance)
    """
    data_feature = rdd_feature_vd.map(lambda x: (x[1], x[2], x[3]))
    df = spark.createDataFrame(data_feature, ["hash", "label", "features"])

    stringIndexer = StringIndexer(inputCol = "hash", outputCol = "indexed")
    si_model = stringIndexer.fit(df)
    td = si_model.transform(df)
    rf = RandomForestClassifier(numTrees=n, maxDepth=m, labelCol = "label")
    td_new = td.withColumn("label", td["label"].cast(DoubleType()))
    model = rf.fit(td_new)
    feature_imp = model.featureImportances
    return feature_imp

def feature_filter(rdd_feature_imp, rdd_feature_distinct, rdd_feature_cnt, rdd_train):
    """
    Filters out the features with importance = 0
    Returns the same format as (docid, hash, label, feature_list)
    """
    # >> (index, feature_imp)
    rdd_feature_imp = rdd_feature_imp.zipWithIndex().map(lambda x: (x[1], x[0]))
    # >> (index, (feature, feature_imp)) >> (feature, new_index)
    rdd_feature_choose = rdd_feature_distinct.map(lambda x: (x[1], x[0])).leftOuterJoin(rdd_feature_imp)\
                            .filter(lambda x: x[1][1] != 0).map(lambda x: x[1][0]).zipWithIndex()
    num = rdd_feature_choose.count()
    # >> (docid, hash, label, vector.dense(bytes))
    rdd = rdd_feature_cnt.map(lambda x: (x[0][1], (x[0][0], x[1])))\
                .leftOuterJoin(rdd_feature_choose).filter(lambda x: x[1][1]!=None)\
                .map(lambda x: (x[1][0][0], (x[1][1], x[1][0][1])))\
                .groupByKey().map(lambda x: (x[0], list(x[1])))
    feature = rdd_train.map(lambda x: (x[1], (x[0], x[2]))).leftOuterJoin(rdd)\
                .map(lambda x: (x[1][0][0], x[0], x[1][0][1], Vectors.dense(get_specified_index_counts(x[1][1], N))))
    return feature, rdd_feature_choose, num


if __name__ == "__main__":
    sc = SparkContext()
    spark = SparkSession.builder.master("local").appName("Word Count").config("spark.some.config.option", "some-value").getOrCreate()


    if DEBUG:
        file_path = '/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p1/small_data/files/'
        bytes_path = '/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p1/small_data/bytes/'
        output_path = '/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p1/small_data/features/'
        size = '_tiny'
    else:
        file_path = 'gs://uga-dsp/project1/files/'
        bytes_path = 'gs://uga-dsp/project1/data/bytes/'
        output_path = 'gs://uga-8360-projects/features/'
        size = '_small'

    # .txt files
    print('***** Reading txt files ********************************')
    rdd_Xtrain = sc.textFile(file_path + 'X' + size + '_train.txt').zipWithIndex().map(lambda x: (x[1], x[0]))
    rdd_ytrain = sc.textFile(file_path + 'y' + size + '_train.txt').zipWithIndex().map(lambda x: (x[1], x[0]))
    # >> (id, hash, label)
    rdd_train = rdd_Xtrain.join(rdd_ytrain).sortByKey().map(lambda x: (x[0],) + x[1])
    # >> (id, hash)
    rdd_Xtest = sc.textFile(file_path + 'X' + size + '_test.txt').zipWithIndex().map(lambda x: (x[1], x[0]))

    n_train = rdd_Xtrain.count()
    n_test = rdd_Xtest.count()

    # .bytes files
    print('***** Reading bytes training files *********************')
    files_train = rdd_Xtrain.map(lambda x: bytes_path + x[1] + '.bytes').reduce(lambda accum, x: accum + ',' + x)
    # >> (hash, content)
    rdd_bytes_train = sc.wholeTextFiles(files_train).map(lambda x: (os.path.basename(x[0]).replace('.bytes', ''), x[1]))
    
    print('***** Reading bytes testing files **********************')
    files_test = rdd_Xtest.map(lambda x: bytes_path + x[1] + '.bytes').reduce(lambda accum, x: accum + ',' + x)
    # >> (hash, content)
    rdd_bytes_test = sc.wholeTextFiles(files_test).map(lambda x: (os.path.basename(x[0]).replace('.bytes', ''), x[1]))


    # Training set
    # -------------------------------------------------------------------------
    print('***** Training set starts ******************************')
    print('***** Detecting bytess *********************************')
    # >> (hash, bytes) _not distinct
    rdd_bytes_detect = rdd_bytes_train.map(lambda x: (x[0], bytes_detect(x[1])))

    print('***** Generating n-gram bytes **************************')
    df_bytes = spark.createDataFrame(rdd_bytes_detect).toDF("hash", "bytes")
    # >> ((hash, bytes_ngrams), count)
    rdd_bytes_cnt = bytes_ngram(df_bytes, NUM_GRAM)

    print('***** Creating distinct n-grams bytes list *************')
    # >> (bytes, index)
    rdd_bytes_distinct = rdd_bytes_cnt.map(lambda x: x[0][1]).distinct().sortBy(lambda x: x).zipWithIndex()
    N = rdd_bytes_distinct.count()

    print('***** Creating bytes list for each document ************')
    # >> (bytes, ((docid, hash, label), cnt))
    rdd_bytes = rdd_bytes_cnt.map(lambda x: (x[0][1], (x[0][0], x[1])))\
                        .leftOuterJoin(rdd_bytes_distinct)\
                        .map(lambda x: (x[1][0][0], (x[1][1], x[1][0][1])))\
                        .groupByKey().map(lambda x: (x[0], list(x[1])))

    print('**** Creating bytes list with document information *****')
    # >> (docid, hash, label, vector.dense(bytes))
    bytes = rdd_train.map(lambda x: (x[1], (x[0], x[2]))).leftOuterJoin(rdd_bytes)\
                    .map(lambda x: (x[1][0][0], x[0], x[1][0][1], Vectors.dense(get_specified_index_counts(x[1][1], N))))


    print('***** RF feature selection ****************************')
    bytes_imp = RF_features_select(bytes)
    # >> (index, feature_importance)
    rdd_bytes_imp = sc.parallelize(bytes_imp)
    # bytes_r >> (docid, hash, label, vectors.dense(bytes))
    # rdd_bytes_distinct_r >> (bytes, index_r)
    bytes_r, rdd_bytes_distinct_r, N_r = feature_filter(rdd_bytes_imp, rdd_bytes_distinct, rdd_bytes_cnt, rdd_train)


    print('***** Transforming RDD into Dateframe ******************')
    df_bytes_train_r = spark.createDataFrame(bytes_r)
    print('***** Outputing parquet file ***************************')
    df_bytes_train_r.write.parquet(output_path + "bytes_" + str(NUM_GRAM) + 'gram' + size + "_train/")


    # Testing set
    # -------------------------------------------------------------------------
    print('***** Testing set starts *******************************')
    print('***** Detecting bytes **********************************')
    # >> ((hash, segment), count)
    rdd_bytes_detect_test = rdd_bytes_test.map(lambda x: (x[0], bytes_detect(x[1])))


    print('***** Generating n-gram bytes **************************')
    df_bytes_test = spark.createDataFrame(rdd_bytes_detect_test).toDF("hash", "bytes")
    # >> ((hash, bytes), count)
    rdd_bytes_cnt_test = bytes_ngram(df_bytes_test, NUM_GRAM)

    print('***** Creating bytes list for each document ************')
    # >> (hash, (bytes_index, bytes_cnt))
    rdd_bytes_test = rdd_bytes_cnt_test.map(lambda x: (x[0][1], (x[0][0], x[1])))\
                            .leftOuterJoin(rdd_bytes_distinct_r).filter(lambda x: x[1][1]!=None)\
                            .map(lambda x: (x[1][0][0], (x[1][1], x[1][0][1])))\
                            .groupByKey().map(lambda x: (x[0], list(x[1])))

    print('***** Creating bytes list with document information ****')
    # >> (docid, hash, vector.dense(bytes))
    bytes_test = rdd_Xtest.map(lambda x: (x[1], x[0]))\
                        .leftOuterJoin(rdd_bytes_test)\
                        .map(lambda x: (x[1][0], x[0], Vectors.dense(get_specified_index_counts(x[1][1], N_r))))

    print('***** Transforming RDD into Dateframe ******************')
    df_bytes_test = spark.createDataFrame(bytes_test)
    print('***** Outputing parquet file **************************************************************')
    df_bytes_test.write.parquet(output_path + "bytes_" + str(NUM_GRAM) + 'gram' + size + "_test/")
