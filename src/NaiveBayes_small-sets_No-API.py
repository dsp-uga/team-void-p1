## 78% accuracy on small test set

## For running on google cloud dataproc:
## Initialization actions: gs://uga-dsp/scripts/conda-dataproc-bootstrap.sh
## python -m pip install --user google-cloud-storage

from pyspark import SparkContext
sc = SparkContext()
sqlContext = SQLContext(sc)


x_small_train = sc.textFile("gs://uga-dsp/project1/files/X_small_train.txt").collect()
x_small_test = sc.textFile("gs://uga-dsp/project1/files/X_small_test.txt").collect()
Y_small_train = sc.textFile("gs://uga-dsp/project1/files/y_small_train.txt").collect()
Y_small_test = sc.textFile("gs://uga-dsp/project1/files/y_small_test.txt").collect()
print("!!!Importing the small datasets is done!!!")

n_fs = 100 #len(x_small_train)
n_tfs = 30 #len(x_small_test)
classes = len(set(Y_small_train)) #distinct number of values in Y train file

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
   wc= f.flatMap(lambda x: x.split('\n')).map(lambda x: x[9:]).filter(lambda x: "?" not in x).map(lambda x: (x,arr))
   wc3 = wc.union(wc3)

   
#The function to calculate TF for each term for each class using the list of term frequncy calculated above. 
def cal_tf_per_cf(arr):
   sum = 0
   for a in arr:
      sum += a
   for k in range (0,len(arr)):
      arr[k]=arr[k]/sum
   return arr  

#Calling the TF function in RDD format
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
with open("tt.txt", "w") as output:
    output.write(str(PredictY))
	
with open("acc.txt", "w") as output:
    output.write(str(acc))


sc.stop() 