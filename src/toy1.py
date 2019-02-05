from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.ml.feature import HashingTF, IDF, NGram, StringIndexer, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

sc = SparkContext()
sql_context = SQLContext(sc)


def helper(content):
    lines = []
    for line in content.splitlines():
        if '?' in line:
            continue
        lines.append(' '.join(line.split()[1:]))
    return lines
    # return [word for word in content.split() if len(word) == 2 and word != '??']
    # return content

def update_hash_label(X_file_path, y_file_path, hash_label):
    X_files = sc.textFile(X_file_path).zipWithIndex().map(lambda x:(x[1],x[0]))
    y_files = sc.textFile(y_file_path).zipWithIndex().map(lambda x:(x[1],x[0]))
    zippedRdd = X_files.join(y_files).map(lambda x: (x[1][0], int(x[1][1]))) #(hash,(label,docid))
    for hash_code, label in zippedRdd.collect():
        hash_label[hash_code] = label - 1

hash_label = {}
# update_hash_label('gs://uga-dsp/project1/files/X_small_train.txt', 'gs://uga-dsp/project1/files/y_small_train.txt', hash_label)
# update_hash_label('gs://uga-dsp/project1/files/X_small_test.txt', 'gs://uga-dsp/project1/files/y_small_test.txt', hash_label)
update_hash_label('gs://uga-dsp/project1/files/X_small_train.txt', 'gs://uga-dsp/project1/files/y_small_train.txt', hash_label)
update_hash_label('gs://uga-dsp/project1/files/X_small_test.txt', 'gs://uga-dsp/project1/files/y_small_test.txt', hash_label)
hash_label_bc = sc.broadcast(hash_label)


print('****************************')
print('Reading Training Data\n')

X_train_files = sc.textFile('gs://uga-dsp/project1/files/X_small_train.txt').map(lambda a: "gs://uga-dsp/project1/data/bytes/" + a + ".bytes").reduce(lambda a, b: a + ',' + b)
# X_train_bytes = sc.wholeTextFiles(X_train_files).zipWithIndex().map(lambda a: (a[1], [word for word in a[0][1].split() if len(word) == 2 and word != '??'])).toDF(['id', 'words'])
data_train_df = sc.wholeTextFiles(X_train_files).map(lambda row: (helper(row[1]), hash_label_bc.value[row[0].split('/')[-1][:-6]], row[0].split('/')[-1][:-6])).toDF(['lines', 'label', 'hash'])

print('****************************')
print('Done Read Training Data\n')

print('****************************')
print('Reading Test Data\n')

X_test_files = sc.textFile('gs://uga-dsp/project1/files/X_small_test.txt').map(lambda a: "gs://uga-dsp/project1/data/bytes/" + a + ".bytes").reduce(lambda a, b: a + ',' + b)
# X_train_bytes = sc.wholeTextFiles(X_train_files).zipWithIndex().map(lambda a: (a[1], [word for word in a[0][1].split() if len(word) == 2 and word != '??'])).toDF(['id', 'words'])
data_test_df = sc.wholeTextFiles(X_test_files).map(lambda row: (helper(row[1]), hash_label_bc.value[row[0].split('/')[-1][:-6]], row[0].split('/')[-1][:-6])).toDF(['lines', 'label', 'hash'])

# X_test_files = sc.textFile('gs://uga-dsp/project1/files/X_small_test.txt').map(lambda a: "gs://uga-dsp/project1/data/bytes/" + a + ".bytes").reduce(lambda a, b: a + ',' + b)
# # X_test_files = sc.textFile('gs://uga-dsp/project1/files/X_small_test.txt').collect()[:20]
# # X_test_files = ','.join(["gs://uga-dsp/project1/data/bytes/" + a + ".bytes" for a in X_test_files])

# # X_test_bytes = sc.wholeTextFiles(X_test_files).zipWithIndex().map(lambda a: (a[1], [word for word in a[0][1].split() if len(word) == 2 and word != '??'])).toDF(['id', 'words'])

# # X_test_bytes = sc.wholeTextFiles(X_test_files).zipWithIndex().map(lambda a: (a[1], [' '.join(line.split()[1:]) for line in a[0][1].splitlines()])).toDF(['id', 'words'])

# X_test_bytes = sc.wholeTextFiles(X_test_files).zipWithIndex().map(helper).toDF(['id', 'words'])

# y_test = sc.textFile('gs://uga-dsp/project1/files/y_small_test.txt').zipWithIndex().map(lambda a: (a[1], int(a[0]) - 1)).toDF(['id', 'label'])
# data_test_df = X_test_bytes.join(y_test, ['id'])

print('****************************')
print('Done Read Test Data\n')

# print(data_train_df.show(n=5, truncate=100))
# print(data_test_df.show(n=5, truncate=100))


print('****************************')
print('Feature Extraction: BiGram, TF-IDF\n')

# regexTokenizer = RegexTokenizer(inputCol="doc", outputCol="words", pattern="\\W")
# tokenizer = Tokenizer(inputCol="lines", outputCol="words")
# ngram = NGram(n=2, inputCol="lines", outputCol="ngrams")
hashing_tf = HashingTF(inputCol="lines", outputCol="raw_features")
idf = IDF(inputCol="raw_features", outputCol="features", minDocFreq=3)

# label_string_idx = StringIndexer(inputCol = "catogory", outputCol = "label")
pipeline = Pipeline(stages=[hashing_tf, idf])

pipeline_fit_train = pipeline.fit(data_train_df)
data_train = pipeline_fit_train.transform(data_train_df)

pipeline_fit_test = pipeline.fit(data_test_df)
data_test = pipeline_fit_test.transform(data_test_df)

print('****************************')
print('Train Model with NaiveBayes\n')

nb = NaiveBayes(smoothing=1)
model = nb.fit(data_train)

# print('****************************')
# print('Train Model with Logistic Regression\n')
# lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0.8, family="multinomial")
# model = lr.fit(data_train)

print('****************************')
print('Testing Unseen Data\n')

predictions = model.transform(data_test)
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
print(accuracy)

predictions.select('label', 'prediction').toPandas().to_csv('output.csv', header=False, index=False)

sc.stop()
