from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.ml.feature import HashingTF, IDF, NGram, StringIndexer, Tokenizer
from pyspark.ml import Pipeline


class Feature_Extraction(object):
    # def __init__(self):


    def extract_featrues(self, input_rdd=None, is_train=True):
        """
        input_rdd: type rdd, the raw rdd of data
        is_train: type boolean, whether or not the input_rdd is corresponding to train data
        return: type data frame, a data frame where each record contains the extracred features
        """
        print('****************************')
        print('Feature Extraction: TF-IDF\n')

        if is_train:
            # input_df contains 2 columns, text content and label
            data_raw_df = input_rdd.map(lambda row: (self.convert(row[0]), row[1])).toDF(['lines', 'label'])
        else:
            # input_df contains 2 columns, text content and doc_id
            data_raw_df = input_rdd.map(lambda row: (self.convert(row[0]), row[1])).toDF(['lines', 'doc_id'])

        # regexTokenizer = RegexTokenizer(inputCol="doc", outputCol="words", pattern="\\W")
        # tokenizer = Tokenizer(inputCol="lines", outputCol="words")
        # ngram = NGram(n=2, inputCol="lines", outputCol="ngrams")
        hashing_tf = HashingTF(inputCol="lines", outputCol="raw_features")
        idf = IDF(inputCol="raw_features", outputCol="features", minDocFreq=3)

        # label_string_idx = StringIndexer(inputCol = "catogory", outputCol = "label")
        pipeline = Pipeline(stages=[hashing_tf, idf])

        pipeline_fit = pipeline.fit(data_raw_df)
        data_df = pipeline_fit.transform(data_raw_df).drop('lines', 'raw_features')
        return data_df


    def convert(self, content):
        """
        content: type str, the raw content of a document
        return: type list, a list which contains the cleaned info
        """
        lines = []
        for line in content.splitlines():
            if '?' in line:
                continue
            lines.append(' '.join(line.split()[1:]))
        return lines
