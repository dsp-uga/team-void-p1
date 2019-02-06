from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import numpy as np
import Feature_Extraction as fe


def get_hash_label_map(X_file_path, y_file_path):
    """
    X_file_path: type str, the path for the X data document, which contains the hash codes
    y_file_path: type str, the path for the y data document, which contains the labels
    return: type dict, a map where the key is the hash code, the value is the label
    """
    hash_label_map = {}
    X_files = sc.textFile(X_file_path).zipWithIndex().map(lambda x:(x[1],x[0]))
    y_files = sc.textFile(y_file_path).zipWithIndex().map(lambda x:(x[1],x[0]))
    zippedRdd = X_files.join(y_files).map(lambda x: (x[1][0], int(x[1][1]))) #(hash, (label, docid))
    for hash_code, label in zippedRdd.collect():
        hash_label_map[hash_code] = label - 1
    return hash_label_map


def get_train_data_rdd(sc, X_train_path, y_train_path):
    """
    sc: type sparckContext
    X_train_path: type str, the path for the X_train data document, which contains the hash codes
    y_train_path: type str, the path for the y_train data document, which contains the labels
    return: type rdd, a rdd where each record contains a document content and its label
    """
    print('****************************')
    print('Reading Training Data\n') 

    hash_label_map = get_hash_label_map(X_train_path, y_train_path)
    hash_label_map_bc = sc.broadcast(hash_label_map)
    X_train_files = sc.textFile(X_train_path).map(lambda row: 'gs://uga-dsp/project1/data/bytes/' + row + '.bytes').reduce(lambda a, b: a + ',' + b)
    data_train_rdd = sc.wholeTextFiles(X_train_files).map(lambda row: (row[1], hash_label_map_bc.value[row[0].split('/')[-1][:-6]]))

    print('****************************')
    print('Done Read Training Data\n')

    return data_train_rdd


def get_test_data_rdd(sc, X_test_path):
    """
    sc: type sparckContext
    X_test_path: type str, the path for the X_test data document, which contains the hash codes
    return: type rdd, a rdd where each record contains a document content and its zipped index in ascendind order (the zipped index will be used for output predictions)
    """
    print('****************************')
    print('Reading Test Data\n') 

    X_test = sc.textFile(X_test_path)
    X_test_idx = X_test.zipWithIndex()
    X_test_files = X_test.map(lambda row: 'gs://uga-dsp/project1/data/bytes/' + row + '.bytes').reduce(lambda a, b: a + ',' + b)
    X_test_content_rdd = sc.wholeTextFiles(X_test_files).map(lambda row: (row[0].split('/')[-1][:-6], row[1]))
    X_test_rdd = X_test_content_rdd.join(X_test_idx).map(lambda row: row[1])

    print('****************************')
    print('Done Read Test Data\n')

    return X_test_rdd


def main(sc, X_train_path, y_train_path, X_test_path, y_test_path=None):
    # file processing
    # train_df is a dataframe containing 2 columns, text content and label
    train_raw_rdd = get_train_data_rdd(sc, X_train_path, y_train_path)
    test_raw_rdd = get_test_data_rdd(sc, X_test_path)
    print(train_raw_rdd.count())
    print(test_raw_rdd.count())

    # feature extraction
    feature_extraction = fe.Feature_Extraction()
    train_df = feature_extraction.extract_featrues(input_rdd=train_raw_rdd, is_train=True)
    test_df = feature_extraction.extract_featrues(input_rdd=test_raw_rdd, is_train=False)
    # print(train_df.show(n=5, truncate=100))
    # print(test_df.show(n=5, truncate=100))


    print('****************************')
    print('Train Model with NaiveBayes\n')
    nb = NaiveBayes(smoothing=1)
    model = nb.fit(train_df)

    print('****************************')
    print('Testing Unseen Data\n')
    predictions = model.transform(test_df)

    pred_list = [int(row.prediction) + 1 for row in predictions.sort('doc_id').select('prediction').collect()]
    with open('prediction.txt', 'w') as f:
        for pred_label in pred_list:
            f.write('%d\n' % pred_label)

    if y_test_path:
        y_test_data = sc.textFile(y_test_path).collect()
        cnt = 0
        for i in range(len(y_test_data)):
            if int(y_test_data[i]) == pred_list[i]:
                cnt += 1
        print('Accuracy: %f, %d/%d' % (cnt * 1.0 / len(y_test_data), cnt, len(y_test_data)))


if __name__ == '__main__':
    X_train_path = 'gs://uga-dsp/project1/files/X_small_train.txt'
    y_train_path = 'gs://uga-dsp/project1/files/y_small_train.txt'
    X_test_path = 'gs://uga-dsp/project1/files/X_small_test.txt'
    y_test_path = 'gs://uga-dsp/project1/files/y_small_test.txt'
    
    sc = SparkContext(pyFiles=['Feature_Extraction.py'])
    sql_context = SQLContext(sc)
    main(sc, X_train_path, y_train_path, X_test_path, y_test_path=y_test_path)
    sc.stop()

