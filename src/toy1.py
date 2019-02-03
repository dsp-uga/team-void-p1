from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.ml.feature import HashingTF, IDF, NGram, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

sc = SparkContext()
sql_context = SQLContext(sc)


def helper(a):
    lines = []
    for line in a[0][1].splitlines():
        if '?' in line:
            continue
        lines.append(' '.join(line.split()[1:]))
    return (a[1], lines)

print('****************************')
print('Reading Training Data\n')

X_train_files = sc.textFile('gs://uga-dsp/project1/files/X_small_train.txt').map(lambda a: "gs://uga-dsp/project1/data/bytes/" + a + ".bytes").reduce(lambda a, b: a + ',' + b)
# X_train_bytes = sc.wholeTextFiles(X_train_files).zipWithIndex().map(lambda a: (a[1], [word for word in a[0][1].split() if len(word) == 2 and word != '??'])).toDF(['id', 'words'])
X_train_bytes = sc.wholeTextFiles(X_train_files).zipWithIndex().map(helper).toDF(['id', 'words'])
y_train = sc.textFile('gs://uga-dsp/project1/files/y_small_train.txt').zipWithIndex().map(lambda a: (a[1], int(a[0]) - 1)).toDF(['id', 'label'])
data_train_df = X_train_bytes.join(y_train, ['id'])

print('****************************')
print('Done Read Training Data\n')

print('****************************')
print('Reading Test Data\n')

X_test_files = sc.textFile('gs://uga-dsp/project1/files/X_small_test.txt').map(lambda a: "gs://uga-dsp/project1/data/bytes/" + a + ".bytes").reduce(lambda a, b: a + ',' + b)
# X_test_files = sc.textFile('gs://uga-dsp/project1/files/X_small_test.txt').collect()[:20]
# X_test_files = ','.join(["gs://uga-dsp/project1/data/bytes/" + a + ".bytes" for a in X_test_files])

# X_test_bytes = sc.wholeTextFiles(X_test_files).zipWithIndex().map(lambda a: (a[1], [word for word in a[0][1].split() if len(word) == 2 and word != '??'])).toDF(['id', 'words'])

# X_test_bytes = sc.wholeTextFiles(X_test_files).zipWithIndex().map(lambda a: (a[1], [' '.join(line.split()[1:]) for line in a[0][1].splitlines()])).toDF(['id', 'words'])

X_test_bytes = sc.wholeTextFiles(X_test_files).zipWithIndex().map(helper).toDF(['id', 'words'])

y_test = sc.textFile('gs://uga-dsp/project1/files/y_small_test.txt').zipWithIndex().map(lambda a: (a[1], int(a[0]) - 1)).toDF(['id', 'label'])
data_test_df = X_test_bytes.join(y_test, ['id'])

print('****************************')
print('Done Read Test Data\n')


print('****************************')
print('Feature Extraction: BiGram, TF-IDF\n')

# regexTokenizer = RegexTokenizer(inputCol="doc", outputCol="words", pattern="\\W")
# tokenizer = Tokenizer(inputCol="doc", outputCol="words")
# ngram = NGram(n=2, inputCol="words", outputCol="ngrams")
# hashing_tf = HashingTF(inputCol="ngrams", outputCol="rawFeatures")
# idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=3)

# label_string_idx = StringIndexer(inputCol = "catogory", outputCol = "label")
pipeline = Pipeline(stages=[ngram, hashing_tf, idf])

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

predictions.select('id', 'label', 'prediction').toPandas().to_csv('output.csv', header=False, index=False)

sc.stop()
