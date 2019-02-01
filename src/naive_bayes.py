from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

sc = SparkContext()
sql_context = SQLContext(sc)

print('****************************')
print('Reading Training Data\n')

X_train_files = sc.textFile('gs://uga-dsp/project1/files/X_small_train.txt').map(lambda a: "gs://uga-dsp/project1/data/bytes/" + a + ".bytes").reduce(lambda a, b: a + ',' + b)
X_train_bytes = sc.wholeTextFiles(X_train_files).zipWithIndex().map(lambda a: (a[1], a[0][1].split())).toDF(['id', 'words'])
y_train = sc.textFile('gs://uga-dsp/project1/files/y_small_train.txt').zipWithIndex().map(lambda a: (a[1], int(a[0]))).toDF(['id', 'label'])
data_train_df = X_train_bytes.join(y_train, ['id'])

print('****************************')
print('Done Read Training Data\n')

print('****************************')
print('Reading Test Data\n')

X_test_files = sc.textFile('gs://uga-dsp/project1/files/X_small_test.txt').map(lambda a: "gs://uga-dsp/project1/data/bytes/" + a + ".bytes").reduce(lambda a, b: a + ',' + b)
X_test_bytes = sc.wholeTextFiles(X_test_files).zipWithIndex().map(lambda a: (a[1], a[0][1].split())).toDF(['id', 'words'])
y_test = sc.textFile('gs://uga-dsp/project1/files/y_small_test.txt').zipWithIndex().map(lambda a: (a[1], int(a[0]))).toDF(['id', 'label'])
data_test_df = X_test_bytes.join(y_test, ['id'])

print('****************************')
print('Done Read Test Data\n')


print('****************************')
print('Feature Extraction: TF-IDF\n')

# regexTokenizer = RegexTokenizer(inputCol="doc", outputCol="words", pattern="\\W")
# tokenizer = Tokenizer(inputCol="doc", outputCol="words")
hashing_tf = HashingTF(inputCol="words", outputCol="rawFeatures")
idf = IDF(inputCol="rawFeatures", outputCol="features")
pipeline = Pipeline(stages=[hashing_tf, idf])

pipeline_fit_train = pipeline.fit(data_train_df)
data_train = pipeline_fit_train.transform(data_train_df)

pipeline_fit_test = pipeline.fit(data_test_df)
data_test = pipeline_fit_test.transform(data_test_df)

print('****************************')
print('Training Model with NaiveBayes\n')

nb = NaiveBayes(smoothing=1)
model = nb.fit(data_train)

print('****************************')
print('Testing Unseen Data\n')

predictions = model.transform(data_test)
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
print(accuracy)

sc.stop()
