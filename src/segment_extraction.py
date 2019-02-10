# python packages
import argparse
import os.path
import re
import numpy as np
from operator import add

# pyspark packages
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import NGram
from pyspark.ml.linalg import Vectors


DEBUG = False

def get_segments(asm_content):
    """
    Detects segment words (e.g. 'text', 'data', 'idata', 'rdata' ...) in the asm file content.
    
    asm_content: type str, the raw content of asm file
    return: type list, a list of segment words.
    """
    pattern = re.compile(r'([A-Za-z]+):[0-9A-Z]{8}[\s+]')
    pattern_list = pattern.findall(asm_content)
    return pattern_list

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


if __name__ == "__main__":
    sc = SparkContext()
    spark = SparkSession.builder.master("local").appName("Word Count").config("spark.some.config.option", "some-value").getOrCreate()


    if DEBUG:
        file_path = '/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p1/small_data/files/'
        asm_path = '/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p1/small_data/asm/'
        output_path = '/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p1/small_data/features/'
        size = '_tiny'
    else:
        file_path = 'gs://uga-dsp/project1/files/'
        asm_path = 'gs://uga-dsp/project1/data/asm/'
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


    # .asm files
    print('***** Reading asm training files ***********************')
    files_train = rdd_Xtrain.map(lambda x: asm_path + x[1] + '.asm').reduce(lambda accum,x: accum + ',' + x)
    # >> (hash, content)
    rdd_asm_train = sc.wholeTextFiles(files_train).map(lambda x: (os.path.basename(x[0]).replace('.asm', ''), x[1]))
    
    print('***** Reading asm testing files ************************')
    files_test = rdd_Xtest.map(lambda x: asm_path + x[1] + '.asm').reduce(lambda accum,x: accum + ',' + x)
    # >> (hash, content)
    rdd_asm_test = sc.wholeTextFiles(files_test).map(lambda x: (os.path.basename(x[0]), x[1])).map(lambda x: (x[0].replace('.asm', ''), x[1]))


    # Training set
    # -------------------------------------------------------------------------
    print('***** Training set starts ******************************')
    print('***** Detecting segments *******************************')
    # >> ((hash, segment), count)
    rdd_segment_cnt = rdd_asm_train.map(lambda x: (x[0], get_segments(x[1]))).flatMapValues(lambda x: x).map(lambda x: (x, 1)).reduceByKey(add)

    print('***** Creating distinct segments list ******************')
    # >> (segment, index)
    rdd_segment_distinct = rdd_segment_cnt.map(lambda x: x[0][1]).distinct().sortBy(lambda x: x).zipWithIndex()
    N = rdd_segment_distinct.count()

    print('***** Creating segments list for each document *********')
    # >> (hash, [(segment_index, cnt), (segment_index, cnt), ...])
    rdd_segment = rdd_segment_cnt.map(lambda x: (x[0][1], (x[0][0], x[1])))\
                    .leftOuterJoin(rdd_segment_distinct)\
                    .map(lambda x: (x[1][0][0], (x[1][1], x[1][0][1])))\
                    .groupByKey().map(lambda x: (x[0], list(x[1])))

    print('*** Creating segments list with document information ***')
    # >> (docid, hash, label, vector.dense(segment))
    segment = rdd_train.map(lambda x: (x[1], (x[0], x[2]))).leftOuterJoin(rdd_segment)\
                .map(lambda x: (x[1][0][0], x[0], x[1][0][1], Vectors.dense(get_specified_index_counts(x[1][1], N))))
                
    print('***** Transforming RDD into Dateframe ******************')
    df_segment_train = spark.createDataFrame(segment)
    print('***** Outputing parquet file ***************************')
    df_segment_train.write.parquet(output_path + "segment" + size + "_train/")

    # Testing set
    # -------------------------------------------------------------------------
    print('***** Testing set starts *******************************')
    print('***** Detecting segments *******************************')
    # >> ((filename, segment), count)
    rdd_segment_cnt_test = rdd_asm_test.map(lambda x: (x[0], get_segments(x[1]))).flatMapValues(lambda x: x).map(lambda x: (x,1)).reduceByKey(add)

    print('***** Creating segments list for each document *********')
    # >> (hash, (segment_index, segment_cnt))
    rdd_segment_test = rdd_segment_distinct\
                            .leftOuterJoin(rdd_segment_cnt_test.map(lambda x: (x[0][1], (x[0][0], x[1]))))\
                            .filter(lambda x: x[1][1] != None)\
                            .map(lambda x: (x[1][1][0], (x[1][0], x[1][1][1])))\
                            .groupByKey().map(lambda x: (x[0], list(x[1])))

    print('*** Creating segments list with document information ***')
    # >> (docid, hash, vector.dense(segment))
    segment_test = rdd_Xtest.map(lambda x: (x[1], x[0])).leftOuterJoin(rdd_segment_test)\
                        .map(lambda x: (x[0], x[1][0], Vectors.dense(get_specified_index_counts(x[1][1], N))))

    print('***** Transforming RDD into Dateframe ******************')
    df_segment_test = spark.createDataFrame(segment_test)
    print('***** Outputing parquet file ***************************')
    df_segment_test.write.parquet(output_path + "segment" + size + "_test/")
