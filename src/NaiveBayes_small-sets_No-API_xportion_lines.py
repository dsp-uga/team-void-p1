### Sampling the train set and only use 2.5 grams of begining of each line on bytes for training and predicting
### 79% accuracy on small test set

## For running on google cloud dataproc:
## Initialization actions: gs://uga-dsp/scripts/conda-dataproc-bootstrap.sh
## python -m pip install --user google-cloud-storage
## pyspark --master yarn
## spark.conf.set("spark.executor.memory", "18g")

from pyspark import SparkContext
#need to be configured based on memory of workers:
spark.conf.set("spark.executor.memory", "18g")
sc = SparkContext()
sqlContext = SQLContext(sc)


x_small_train = sc.textFile("gs://uga-dsp/project1/files/X_small_train.txt").collect()
x_small_test = sc.textFile("gs://uga-dsp/project1/files/X_small_test.txt").collect()
Y_small_train = sc.textFile("gs://uga-dsp/project1/files/y_small_train.txt").collect()
Y_small_test = sc.textFile("gs://uga-dsp/project1/files/y_small_test.txt").collect()
print("!!!Importing the small datasets is done!!!")

n_fs = 200 #len(x_small_train)
print("Number of train file: ",n_fs)
n_tfs = len(x_small_test)
print("Number of test file: ",n_tfs)
classes = len(set(Y_small_train)) #distinct number of values in Y train file
print("Number of classes: ",classes)
#The function of adding elements of two same size list of integers, which is used in reduce
def add_a(a,b):
   print (a,b)
   if len(a)==len(b):
      for ii in range(0,len(a)):
         a[ii]=int(a[ii])+int(b[ii])
   return a

#Initializing an RDD to union result of each term frequncy calculation into specific index of the value list
a = [1]*(classes)
from pyspark.sql.types import *
field = [StructField('term',StringType(), True),StructField('tf_classes', ArrayType(DoubleType(), True),True)]
schema = StructType(field)


#The function to calculate TF for each term for each class using the list of term frequncy calculated above. 
def cal_tf_per_cf(arr):
   sum = 0
   for a in arr:
      sum += a
   for k in range (0,len(arr)):
      arr[k]=arr[k]/sum
   return arr  

#Running term frequncy calculation while cleaning the address in each line and removing terms including "?"; then creat a RDD of all terms with list of TF for each class

wc3 = sc.parallelize([])
for i in range(0,n_fs):
	k += 1
	f = sc.textFile("gs://uga-dsp/project1/data/bytes/"+x_small_train[i]+".bytes")
	arr = [0]*(classes)
	arr[int(Y_small_train[i])-1]=1
	wc= f.map(lambda x: x[9:16]).filter(lambda x: "?" not in x).map(lambda x: (x,arr))
	wc3 = wc.union(wc3)


wc4 = wc3.combineByKey(list,add_a,add_a).map(lambda x: (x[0],cal_tf_per_cf(x[1])))
wc4.cache()

#The function for calculating and accumulating the probabablity of each terms in test doc using the created bag of words from traning documents
def cal_prob(arr):
	for i in range(0,len(arr[1])):
		arr[1][i]=arr[1][i]*arr[0]
	return arr[1]

#Reading test document and calculate NB probabablity using above function and the created bag of words from traning documents
Pr = [0]*n_tfs
from operator import add
for j in range(0,n_tfs):
	ttf = sc.textFile("gs://uga-dsp/project1/data/bytes/"+x_small_test[j]+".bytes")
	ttwc= ttf.flatMap(lambda x: x.split('\n')).map(lambda x: x[9:16]).filter(lambda x: "?" not in x[0]).map(lambda x: (x,1)).reduceByKey(add)
	prob = sc.parallelize([])
	prob = ttwc.leftOuterJoin(wc4).filter(lambda x: x[1][1]!= None).map(lambda x: (j,cal_prob(x[1]))).reduceByKey(add_a)
	pr2 = prob.collect()
	Pr[pr2[0][0]] = pr2[0][1] 



Pre = [0]*n_tfs
for i in range(0,n_tfs):
	if len(Pr[i])>1:
		Pre[i] = (Pr[i].index(max(Pr[i])))+1


print (Pre)

#Calculating accuracy
k=0
acc = 0.0
for i in range(0,n_tfs):
	if int(Pre[i])==int(Y_small_test[i]):
		k += 1

acc = (k/i)
print (acc)

sc.stop() 

with open("/tt.txt", "w") as output:
    output.write(str(Pre))
	
with open("/acc.txt", "w") as output:
    output.write(str(acc))


sc.stop() 



