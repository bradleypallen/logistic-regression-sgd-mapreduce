logistic-regression-sgd-mapreduce
=================================

# Overview
Python scripts for building binary classifiers using logistic regression with stochastic gradient descent, packaged for use with map-reduce platforms supporting Hadoop streaming.

# ￼Algorithm
* Distributed regularized binary logistic regression with stochastic gradient descent [[1]], [[2]]
	* Competitive with best extant large-scale supervised learning algorithms
	* Results provide direct estimation of probability of class membership 
* Implementation supports large-scale learning using Hadoop streaming	* Map/reduce implementation [[3]] allows embarrassingly parallel processing of training and test datasets to reduce wall clock time	* Supports the use of the hashing trick [[4]], [[5]] which allows memory requirements to be fixed with minimal increase of error	* Margin-based certainty [[6]] based on probability estimates supports active learning workflow [[7]] to reduce human annotation costs
	# Installation
    $ git clone https://github.com/bradleypallen/logistic-regression-sgd-mapreduce.git
    $ cd logistic-regression-sgd-mapreduce    $ chmod +x *.py

# Data formats

## Instances
    <instance> ::= { "class": <class>, "features": { <feature>: <value>, … , <feature>: <value> } }
    <class>    ::= 0 | 1
    <feature>  ::= a JSON string
    <value>    ::= a JSON float in the interval [0.0, 1.0]   

## Models
    <model> ::= { <feature>: <weight>, … , <feature>: <weight> }
    <weight> ::= a JSON float in the interval [0.0, 1.0]

## Confusion matrices
    <confusion_matrix> ::= { "TP": <count>, "FP": <count>, "FN": <count>, "TN": <count> }
    <count>            ::= a JSON int in the interval [0, inf)
    
# Usage
The Python scripts implement the key parts of a complete active learning workflow:

* train,
* test, and
* predict.

## From a UNIX shell
For small data sets, the scripts can be run from the command line.

### Convert data in SVM<sup><i>Light</i></sup> format into instances
Convert a file with data in SVM<sup><i>Light</i></sup> [[8]] format into a file containing instances. Awk, sort and cut are used here to randomly shuffle the data set, which is required to correctly train the model.

    $  cat train.data.svmlight | ./parse_svmlight_examples.py | awk 'BEGIN{srand();} {printf "%06d %s\n", rand()*1000000, $0;}' | sort -n | cut -c8- > train.data
    $  cat test.data.svmlight | ./parse_svmlight_examples.py | awk 'BEGIN{srand();} {printf "%06d %s\n", rand()*1000000, $0;}' | sort -n | cut -c8- > test.data

### Train a model
Generate a model from training data.

    $ cat train.data | ./train_map.py | sort | ./train_reduce.py > /path/to/your/model
    
Hyperparameters can be optionally set as environment variables:

    $ export MU=0.002   # the regularization parameter
    $ export ETA=0.5    # the learning rate
    $ export N=100000   # the number of instances in the training set

### Test a model
Generate a confusion matrix based on running a model against test data. The location of the model is passed as an environment variable that is a valid file URL.

    $ export MODEL=file:///path/to/your/model
    $ cat test.data | ./test_map.py | sort | ./test_reduce.py > confusion-matrix
    
### Predict
Generate a tab-separated file containing a line per instance in the input file with the following fields:

1. the margin,
2. the estimated probability of class membership,
3. the prediction of class membership (0 or 1), and
4. the JSON representation of the instance.
 
The output file is intended to support active learning workflows; smaller margin implies greater uncertainty given the model, so given the output is sorted in increasing order of margin, the first line in the file can be used as the most informative instance to provide to a subject matter expert for review to determine the correct class of the instance. Additionally, instances over a threshold margin can be automatically labled with the predicted class and added to a training set to refine the model.

    $ export MODEL=file:///path/to/your/model
    $ cat test.data | ./predict_map.py | sort | ./predict_reduce.py > predictions
    
## Using Elastic MapReduce
For large-scale data sets, the scripts can be run using Hadoop streaming in Elastic MapReduce. First, upload the Python scripts and data files to a bucket in S3.

### Train a model
    $ ./elastic-mapreduce --create --stream \
		--input s3n://path/to/your/bucket/train.data \
		--mapper s3n://path/to/your/bucket/train_map.py \
		--reducer s3n://path/to/your/bucket/train_reduce.py \
		--output s3n://path/to/your/bucket/model

### Test a model
    $ ./elastic-mapreduce --create --stream \
		--input s3n://path/to/your/bucket/test.data \
		--mapper s3n://path/to/your/bucket/test_map.py \
		--reducer s3n://path/to/your/bucket/test_reduce.py \
		--output s3n://path/to/your/bucket/confusion-matrix
		--cmdenv MODEL=https://s3.amazonaws.com/path/to/your/bucket/model/part-00000
		
### Predict
    $ ./elastic-mapreduce --create --stream \
		--input s3n://path/to/your/bucket/unlabeled.data \
		--mapper s3n://path/to/your/bucket/predict_map.py \
		--reducer s3n://path/to/your/bucket/predict_reduce.py \
		--output s3n://path/to/your/bucket/predictions
		--cmdenv MODEL=https://s3.amazonaws.com/path/to/your/bucket/model/part-00000
    
# References

[[1]] Cohen, W. Stochastic Gradient Descent. <i>Downloaded from http://www.cs.cmu.edu/~wcohen/10-605/notes/sgd-notes.pdf</i> (2012).
[[2]] Zinkevich, M., Smola, A. & Weimer, M. Parallelized Stochastic Gradient Descent. <i>Advances in Neural Information Processing Systems 23, 1-9</i> (2010).[[3]] Lin, J. & Kolcz, A. Large-Scale Machine Learning at Twitter. <i>SIGMOD</i> (2012).[[4]] Weinberger, K., Dasgupta, A., Attenberg, J., Langford, J. & Smola, A. Feature hashing for large scale multitask learning. <i>ICML</i> (2009).[[5]] Attenberg, J., Weinberger, K., Dasgupta, A., Smola, A. & Zinkevich, M. Collaborative Email-Spam Filtering with the Hashing-Trick. <i>CEAS 2009</i> (2009).[[6]] Schein, A. & Ungar, L. Active Learning for Logistic Regression: An Evaluation. <i>Machine Learning 68(3), 235-265</i> (2007).[[7]] Sculley, D., Otey, M.E., Pohl, M., Spitznagel, B. & Hainsworth, J. Detecting Adversarial Advertisements in the Wild. <i>KDD’11</i> (2011).
[[8]] Joachims, T. SVM<sup><i>Light</i></sup> Support Vector Machine. <i>Downloaded from http://svmlight.joachims.org/</i> (2008).

[1]: http://www.cs.cmu.edu/~wcohen/10-605/notes/sgd-notes.pdf "Cohen, W. Stochastic Gradient Descent. Downloaded from http://www.cs.cmu.edu/~wcohen/10-605/notes/sgd-notes.pdf. (2012)."[2]: http://www.martin.zinkevich.org/publications/nips2010.pdf "Zinkevich, M., Smola, A. & Weimer, M. Parallelized Stochastic Gradient Descent. Advances in Neural Information Processing Systems 23, 1-9 (2010)."[3]: http://www.umiacs.umd.edu/~jimmylin/publications/Lin_Kolcz_SIGMOD2012.pdf "Lin, J. & Kolcz, A. Large-Scale Machine Learning at Twitter. SIGMOD (2012)."[4]: http://arxiv.org/pdf/0902.2206.pdf "Weinberger, K., Dasgupta, A., Attenberg, J., Langford, J. & Smola, A. Feature hashing for large scale multitask learning. ICML (2009)."[5]: http://ceas.cc/2009/papers/ceas2009-paper-11.pdf "Attenberg, J., Weinberger, K., Dasgupta, A., Smola, A. & Zinkevich, M. Collaborative Email-Spam Filtering with the Hashing-Trick. CEAS 2009 (2009)."[6]: http://www.andrewschein.com/publications/scheinML2007.pdf "Schein, A. & Ungar, L. Active Learning for Logistic Regression: An Evaluation. Machine Learning 68(3), 235-265 (2007)."[7]: http://www.eecs.tufts.edu/~dsculley/papers/adversarial-ads.pdf "Sculley, D., Otey, M.E., Pohl, M., Spitznagel, B. & Hainsworth, J. Detecting Adversarial Advertisements in the Wild. KDD’11 (2011)."
[8]: http://svmlight.joachims.org/ "Joachims, T. SVM<sup><i>Light</i></sup> Support Vector Machine. Downloaded from http://svmlight.joachims.org/. (2008)."