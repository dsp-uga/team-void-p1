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


DEBUG = True
NUM_GRAM = 2
PERCENTAGE = 0.8

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

def opcode_detect(asm_content):
    """
    Detects opcodes of the content of asm file.
    Returns a opcodes list.
    """
    pattern = re.compile(r'[\s][A-F0-9]{2}[\s]+([a-z]+)[\s+]')
    pattern_list = pattern.findall(asm_content)
    return pattern_list


def opcode_ngram(df_opcode, n):
    """
    Generates n-grams opcode by opcode data frame.
    Returns n-grams opcode in RDD((hash, n-gram), total_counts)
    """
    ngrams = NGram(n=n, inputCol="opcode", outputCol="ngrams")
    df_ngrams = ngrams.transform(df_opcode)
    rdd_ngrams = df_ngrams.select("hash", "ngrams").rdd.map(tuple).flatMapValues(lambda x: x)\
                    .map(lambda x: ((x[0], x[1]), 1)).reduceByKey(add)
    return rdd_ngrams


def feature_IDF(rdd_feature_detect):
    """
    Generates IDF values of each opcodes
    >>> Input (hash, feature), Output (feature, IDF)
    """
    rdd_feature_IDF = rdd_feature_detect.distinct().map(lambda x: (x[1], x[0])).groupByKey()\
                        .map(lambda x: (x[0], len(list(x[1])))).filter(lambda x: x[1] < (n_train * PERCENTAGE))\
                        .map(lambda x: (x[0], np.log(n_train / x[1]))).filter(lambda x: x[1] != 0)
    return rdd_feature_IDF


def RF_features_select(rdd_feature_vd, n=10, m=7):
    """
    Implements random forest classifier to the opcodes counts in each document
    Returns the importance of each opcodes
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
    # >> (docid, hash, label, vector.dense(opcode))
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
        asm_path = '/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p1/small_data/asm/'
        output_path = '/Users/jerryhui/Desktop/Files/UGA/CS/8360DataPracticum/Projects/p1/small_data/features/'
        size = '_tiny'
    else:
        file_path = 'gs://uga-dsp/project1/files/'
        asm_path = 'gs://uga-dsp/project1/data/asm/'
        output_path = 'gs://uga-8360-projects/features/'
        size = ''

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

    # .asm files
    print('***** Reading asm training files ***********************')
    files_train = rdd_Xtrain.map(lambda x: asm_path + x[1] + '.asm').reduce(lambda accum, x: accum + ',' + x)
    # >> (hash, content)
    rdd_asm_train = sc.wholeTextFiles(files_train).map(lambda x: (os.path.basename(x[0]).replace('.asm', ''), x[1]))
    
    print('***** Reading asm testing files ************************')
    files_test = rdd_Xtest.map(lambda x: asm_path + x[1] + '.asm').reduce(lambda accum, x: accum + ',' + x)
    # >> (hash, content)
    rdd_asm_test = sc.wholeTextFiles(files_test).map(lambda x: (os.path.basename(x[0]).replace('.asm', ''), x[1]))


    # Training set
    # -------------------------------------------------------------------------
    print('***** Training set starts ******************************')
    print('***** Detecting opcodes ********************************')
    # >> (hash, opcode) _not distinct
    rdd_opcode_detect = rdd_asm_train.map(lambda x: (x[0], opcode_detect(x[1]))).flatMapValues(lambda x: x).map(lambda x: (x[0], x[1]))

    print('***** Creating opcodes list by IDF values **************')
    # >> (opcode) _distinct
    opcode_list = feature_IDF(rdd_opcode_detect).map(lambda x: x[0]).collect()

    print('***** Filtering out opcodes with low IDF values ********')
    # >> (hash, [opcodes_list])
    rdd_opcode_list = rdd_opcode_detect.filter(lambda x: x[1] in opcode_list).groupByKey().map(lambda x: (x[0], list(x[1])))

    print('***** Generating n-gram opcodes ************************')
    df_opcode = spark.createDataFrame(rdd_opcode_list).toDF("hash", "opcode")
    # >> ((hash, opcode_ngrams), count)
    rdd_opcode_cnt = opcode_ngram(df_opcode, NUM_GRAM)


    print('***** Creating distinct n-grams opcodes list ***********')
    # >> (opcode, index)
    rdd_opcode_distinct = rdd_opcode_cnt.map(lambda x: x[0][1]).distinct().sortBy(lambda x: x).zipWithIndex()
    N = rdd_opcode_distinct.count()

    print('***** Creating opcodes list for each document **********')
    # >> (opcode, ((docid, hash, label), cnt))
    rdd_opcode = rdd_opcode_cnt.map(lambda x: (x[0][1], (x[0][0], x[1])))\
                        .leftOuterJoin(rdd_opcode_distinct)\
                        .map(lambda x: (x[1][0][0], (x[1][1], x[1][0][1])))\
                        .groupByKey().map(lambda x: (x[0], list(x[1])))

    print('**** Creating opcodes list with document information ***')
    # >> (docid, hash, label, vector.dense(opcode))
    opcode = rdd_train.map(lambda x: (x[1], (x[0], x[2]))).leftOuterJoin(rdd_opcode)\
                    .map(lambda x: (x[1][0][0], x[0], x[1][0][1], Vectors.dense(get_specified_index_counts(x[1][1], N))))


    print('***** RF feature selection *****************************')
    opcode_imp = RF_features_select(opcode)
    # >> (index, feature_importance)
    rdd_opcode_imp = sc.parallelize(opcode_imp)
    # opcode_r >> (docid, hash, label, vectors.dense(opcode))
    # rdd_opcode_distinct_r >> (opcode, index_r)
    opcode_r, rdd_opcode_distinct_r, N_r = feature_filter(rdd_opcode_imp, rdd_opcode_distinct, rdd_opcode_cnt, rdd_train)


    print('***** Transforming RDD into Dateframe ******************')
    df_opcode_train_r = spark.createDataFrame(opcode_r)
    print('***** Outputing parquet file ***************************')
    df_opcode_train_r.write.parquet(output_path + "opcode_" + str(NUM_GRAM) + 'gram' + size + "_train/")


    # Testing set
    # -------------------------------------------------------------------------
    print('***** Testing set starts *******************************')
    print('***** Detecting segments *******************************')
    # >> ((hash, segment), count)
    rdd_opcode_detect_test = rdd_asm_test.map(lambda x: (x[0], opcode_detect(x[1]))).flatMapValues(lambda x: x).map(lambda x: (x[0], x[1]))

    print('***** Creating opcodes list for each document **********')
    # >> (hash, [opcodes_list])
    rdd_opcode_list_test = rdd_opcode_detect_test.filter(lambda x: x[1] in opcode_list).groupByKey().map(lambda x: (x[0], list(x[1])))

    print('***** Generating n-gram opcodes ************************')
    df_opcode_test = spark.createDataFrame(rdd_opcode_list_test).toDF("hash", "opcode")
    # >> ((hash, opcode), count)
    rdd_opcode_cnt_test = opcode_ngram(df_opcode_test, NUM_GRAM)

    print('***** Creating opcodes list for each document **********')
    # >> (hash, (opcode_index, opcode_cnt))
    rdd_opcode_test = rdd_opcode_cnt_test.map(lambda x: (x[0][1], (x[0][0], x[1])))\
                            .leftOuterJoin(rdd_opcode_distinct_r).filter(lambda x: x[1][1]!=None)\
                            .map(lambda x: (x[1][0][0], (x[1][1], x[1][0][1])))\
                            .groupByKey().map(lambda x: (x[0], list(x[1])))

    print('***** Creating opcodes list with document information **')
    # >> (docid, hash, vector.dense(opcode))
    opcode_test = rdd_Xtest.map(lambda x: (x[1], x[0]))\
                        .leftOuterJoin(rdd_opcode_test)\
                        .map(lambda x: (x[1][0], x[0], Vectors.dense(get_specified_index_counts(x[1][1], N_r))))

    print('***** Transforming RDD into Dateframe ******************')
    df_opcode_test = spark.createDataFrame(opcode_test)
    print('***** Outputing parquet file **************************************************************')
    df_opcode_test.write.parquet(output_path + "opcode_" + str(NUM_GRAM) + 'gram' + size + "_test/")
