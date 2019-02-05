## For running on google cloud dataproc:
## Initialization actions: gs://uga-dsp/scripts/conda-dataproc-bootstrap.sh
## python -m pip install --user google-cloud-storage
## gs://dataproc-b74c834c-1332-45d7-ac8e-17eb96d4b33e-us-east1/res/NaiveBayes_small-sets_No-API.py	

import pyspark
sc = pyspark.SparkContext()
x_small_train = sc.textFile("gs://uga-dsp/project1/files/X_small_train.txt").collect()
x_small_test = sc.textFile("gs://uga-dsp/project1/files/X_small_test.txt").collect()
Y_small_train = sc.textFile("gs://uga-dsp/project1/files/y_small_train.txt").collect()
Y_small_test = sc.textFile("gs://uga-dsp/project1/files/y_small_test.txt").collect()
print("!!!Importing the small datasets is done!!!")

n_fs = len(x_small_train)
n_tfs = len(x_small_test)
classes = 9 #distinct number of values in Y train file

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

   
#The function to calculate TF for each term for each class using the list of term frequncy calculated above. 
def cal_tf_per_cf(arr):
   sum = 0
   for a in arr:
      sum += a
   for k in range (0,len(arr)):
      arr[k]=arr[k]/sum
   return arr  

#Calling the TF function in RDD format
wc4 = wc3.map(lambda x: (x[0],cal_tf_per_cf(x[1])))
wc4.cache()

#The function for calculating and accumulating the probabablity of each terms in test doc using the created bag of words from traning documents
def cal_prob(arr):
	for i in range(0,len(arr[1])):
		arr[1][i]=arr[1][i]*arr[0]
	return arr[1]

#Reading test document and calculate NB probabablity using above function and the created bag of words from traning documents
probarr=sc.parallelize([])
from operator import add
for j in range(0,n_tfs):
	ttf = sc.textFile("gs://uga-dsp/project1/data/bytes/"+x_small_test[j]+".bytes")
	ttwc= ttf.flatMap(lambda x: x.split('\n')).map(lambda x: x[9:]).map(lambda x: (x,1)).reduceByKey(add)
	ttwc2 = ttwc.filter(lambda x: "?" not in x[0])
	prob = ttwc2.leftOuterJoin(wc4)
	prob2 = prob.filter(lambda x: x[1][1]!= None)
	prob3 = prob2.map(lambda x: (j,cal_prob(x[1])))
	prob4 = prob3.reduceByKey(add_a)
	probarr = prob4.union(probarr)
	probarr.cache()

#The function for classify each test doc by finding the largest calculate probabablity of classes
def predict_id(arr):
	l=arr[0]
	index = 0
	for m in range(1,len(arr)):
		if arr[m]>l :
			l=arr[m]
			index = m
	return (index+1)

#Calling the above mentioned function and create the predicted list of classes by order of given test docs
predict_list = probarr.sortBy(lambda x: x[0]).map(lambda x: predict_id(x[1]))
PredictY = predict_list.collect()

#Calculating accuracy
k=0
for i in range(0,len(PredictY)):
	if int(PredictY[i])==int(Y_small_test[i]):
		k += 1
acc = k/i

print (acc)

with open("/home/mohammadreza_im/tt.txt", "w") as output:
    output.write(str(PredictY))
	
with open("/home/mohammadreza_im/acc.txt", "w") as output:
    output.write(str(acc))


sc.stop() 