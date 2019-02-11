# Project 1 CSCI-8360 : Malware Classification
## Team: team-void-p1
#### Members: 
* Jiahao Xu
* Yang Shi
* Mohammadreza Iman

## Classifiers: 
* Naive-Bayes
* Random Forest
* Logistic Regression

## Technologies Used:
* Apache Spark on Google Cloud Platform
  * Packages: spark.ml, spark.sql
* pySpark 2.3.2
* Python 3.7.2

## Overview:
The project details and explanation is accessible via [Project1-CSCI8360-UGA](https://github.com/dsp-uga/sp19/blob/master/projects/p1/project1.pdf).

This repository contains the source codes of different methods that we implement to address the project goal. Briefly, the project goal is to find the best machine learning method for classifying malware files, which should be implemented using Spark platform and could handle a large (Big Data) amount of document as training set and testing set.

## Dataset:
The details of the datasets are accessible in the project description document, [Project1-CSCI8360-UGA](https://github.com/dsp-uga/sp19/blob/master/projects/p1/project1.pdf). There are 9 classes of malware. The datasets  are accessible via the Google Cloud Bucket:
### Small set:
	1. Training set. 379 files available both in bytes and asm formats.
	1. Test set. 169 files available both in bytes and asm formats.
	1. The list of classes for the training set and testing set.

### Large set:
	1. Training set. 8147 files available both in bytes and asm formats.
	1. Test set. 2721 files available both in bytes and asm formats.
	1. The list of classes for the training set.
To verify the accuracy of the implemented model, we should submit the predicted list to the leaderboard site for this activity and retrieve  the result.

## Project Summary:
The biggest challenge of this project underlies in the size of datasets (hundreds of gigabytes), which makes the process of feature selection crucial for this project.

We explored several different approaches for this problem, like the use of implemented libraries of n-gram, TF_IDF in spark.ml package as well as trying to implement a specific Naive Bayes and feature extraction methods tailored for this specific project and the documents format.

The memory issue for feature extraction due to the size of datasets was so challenging. Through this process, we learned that on bytes since the files contain hexadecimal words (256 possible words):

	- Unigram ends with 256 features
	- Bi-grams returns 256*256 = 65,536 features
	- Three-grams means 256^3 = 16,777,216 possible features 
	- Four-grams rise the number of possible features to 256^4 = 4,294,967,296

Such exponential possible number of features shows the reason of memory issues all team faced for this project. Same time it lead us to try to find the best possible competition of selected features.


## Source codes (/src):
We tried to add enough comments and directions to each source code to let others be able to run them and adjust them for different platforms and datasets. The codes we are sharing in this repository are listed below with a brief explanation:

	- Engine.py	(The engine code for defining the datasets access, selecting the feature selection method, and running the prediction model)
	- engine1.py	(The second version of engine code, which supports Random Forest)
	- Feature_Extraction.py	(The implementation of feature selection)
	- naive_bayes.py	(The implementation of the naive bayes classifier)
	- ngram_bytes_extraction.py	(The implementation of the n-gram(s) word extractor)
	- ngram_opcode_extraction.py	(The implementation of the n-gram(s) word extractor with opcode method)
	- segment_extraction.py	(The implementation of the segment features extractor)
	- NaiveBayes_TF-IDF_small-sets_No-API.py	(A tailored implementation of TF_IDF and naive bayes for this type of datasets, with no use of available ML libraries)
	- NaiveBayes_small-sets_No-API.py	(A tailored implementation of TF and naive bayes for this type of datasets, with no use of available ML libraries)
	- NaiveBayes_small-sets_No-API_xportion_lines.py	(A tailored implementation of partial TF and naive bayes for this type of datasets, with no use of available ML libraries)

## Some of the accuracy results:
* Small datasets:
	- NaiveBayes:
		- bytes & unigram			0.2662
		- bytes & bigram 			0.6331
		- bytes & unigram of lines	0.5799
	- Random Forest	
		- bytes & unigram of lines	0.6450
		- bytes & bigram			0.8580

* Large datasets:
	- Random Forest (numtree=50, maxdepth=25)
		- only segment									0.912
		- only bigrams with spark RF featureImportance	0.965
		- only 3-grams with spark RF featureImportance 	0.945
		- bytes & bigrams with segment count			**0.989**
		- bytes & 3-grams with segment count			0.981