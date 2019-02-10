from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.ml.feature import HashingTF, IDF, NGram, StringIndexer, Tokenizer, CountVectorizer
from pyspark.ml import Pipeline


class Feature_Extraction(object):
    # def __init__(self):


    def extract_featrues(self, train_rdd=None, test_rdd=None):
        """
        train_rdd: type rdd, the raw rdd of train data (text content, label)
        test_rdd: type rdd, the raw rdd of test data (text content, doc_id)
        return: type data frame, a data frame where each record contains the extracred features
        """
        print('****************************')
        print('Feature Extraction: TF-IDF\n')

        train_raw_df = train_rdd.map(lambda row: (self.convert(row[0]), row[1])).toDF(['words', 'label'])
        test_raw_df = test_rdd.map(lambda row: (self.convert(row[0]), row[1])).toDF(['words', 'doc_id'])

        ngram = NGram(n=2, inputCol="words", outputCol="ngrams")
        train_ngram_df = ngram.transform(train_raw_df).drop('words')
        test_ngram_df = ngram.transform(test_raw_df).drop('words')

        hashing_tf = HashingTF(inputCol='ngrams', outputCol='raw_features')
        train_raw_featured_data = hashing_tf.transform(train_ngram_df).drop('ngrams')
        test_raw_featured_data = hashing_tf.transform(test_ngram_df).drop('ngrams')

        idf = IDF(inputCol='raw_features', outputCol='features')
        idf_model = idf.fit(train_raw_featured_data)

        train_df = idf_model.transform(train_raw_featured_data).drop('raw_features')
        test_df = idf_model.transform(test_raw_featured_data).drop('raw_features')

        return (train_df, test_df)


    def convert(self, content):
        """
        content: type str, the raw content of a document
        return: type list, a list which contains the cleaned info
        """
        # lines = []
        # for line in content.splitlines():
        #     if '?' in line:
        #         continue
        #     lines.append(' '.join(line.split()[1:]))
        # return lines
        return [word for word in content.split() if len(word) == 2 and word != '??']
