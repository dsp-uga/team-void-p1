## 56% accuracy on small test set

## For running on google cloud dataproc:
## Initialization actions: gs://uga-dsp/scripts/conda-dataproc-bootstrap.sh
## python -m pip install --user google-cloud-storage
## pyspark --master yarnexit

import math #Needed for calculating Log
import pyspark
sc = pyspark.SparkContext()
x_small_train = sc.textFile("gs://uga-dsp/project1/files/X_small_train.txt").collect()
x_small_test = sc.textFile("gs://uga-dsp/project1/files/X_small_test.txt").collect()
Y_small_train = sc.textFile("gs://uga-dsp/project1/files/y_small_train.txt").collect()
Y_small_test = sc.textFile("gs://uga-dsp/project1/files/y_small_test.txt").collect()
print("!!!Importing the small datasets is done!!!")

n_fs = 200 #len(x_small_train)
n_tfs = len(x_small_test)
classes = len(set(Y_small_train)) #distinct number of values in Y train file
norm = 1 #00
cutoftf = 20.0

#The function of adding elements of two same size list of integers, which is used in reduce
def add_a(a,b):
   if len(a)==len(b):
      for ii in range(0,len(a)):
         a[ii]=int(a[ii])+int(b[ii])
   return a

#Initializing an RDD to union result of each term frequncy calculation into specific index of the value list
a = [1]*(classes)
wc3 = sc.parallelize([])

#Running term frequncy calculation while cleaning the address in each line and removing terms including "?"; then creat a RDD of all terms with list of TF for each class
for i in range(0,n_fs):
   f = sc.textFile("gs://uga-dsp/project1/data/bytes/"+x_small_train[i]+".bytes")
   arr = [0]*(classes)
   arr[int(Y_small_train[i])-1]=1
   wc= f.flatMap(lambda x: x.split('\n')).map(lambda x: x[9:]).map(lambda x: (x,arr)).combineByKey(list,add_a,add_a)
   wc2 = wc.filter(lambda x: "?" not in x[0])
   wc3 = wc2.union(wc3).combineByKey(list,add_a,add_a)

   
#The function to calculate TF-IDF for each term for each document using the list of term frequncy calculated above. It replaces the TF for each doc in the list with TF-IDF of such term for the doc
def cal_tf_idf(arr):
   dn = 0
   for a in arr:
      if a > 0: dn +=1
   for k in range (0,len(arr)):
      arr[k]=arr[k]*math.log(len(arr)/dn)
   return arr  

#Calling the TF-IDF function in RDD format
wc4 = wc3.map(lambda x: (x[0],cal_tf_idf(x[1])))


#Reduce the size of bag of words by excluding the terms that doesn't have any tf-idf bigger than the cut of value set at begining, so the prediction could work faster
def cut_of_tf(arr):
	for a in arr:
		if a > cutoftf:
			return True
	return False

wc5 = wc4.filter(lambda x: cut_of_tf(x[1]))

wc5.cache()


#The function for calculating and accumulating the probabablity of each terms in test doc using the created bag of words from traning documents and applying the normalization value
def cal_prob(arr):
	for i in range(0,len(arr[1])):
		arr[1][i]=1+arr[1][i]*arr[0] #*norm
	return arr[1]

#Reading test document and calculate NB probabablity using above function and the created bag of words from traning documents
Pr = [0]*n_tfs
from operator import add
for j in range(0,n_tfs):
	ttf = sc.textFile("gs://uga-dsp/project1/data/bytes/"+x_small_test[j]+".bytes")
	ttwc= ttf.flatMap(lambda x: x.split('\n')).map(lambda x: x[9:]).filter(lambda x: "?" not in x[0]).map(lambda x: (x,1)).reduceByKey(add)
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

with open("/tt.txt", "w") as output:
    output.write(str(PredictY))


with open("/acc.txt", "w") as output:
    output.write(str(acc))

sc.stop() 