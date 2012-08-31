logistic-regression-sgd-mapreduce
=================================
<p>
Bradley P. Allen</br>
Elsevier Labs</br>
2012-08-30
</p>

# Overview
This repository contains Python scripts for building binary classifiers using logistic regression with stochastic gradient descent, packaged for use with map-reduce platforms supporting Hadoop streaming. This implementation is closely based on the algorithm described by William Cohen in his class notes on implementing stochastic gradient descent in his Machine Learning with Large Datasets 10-605 course at Carnegie-Mellon University in Spring 2012.

# ￼Algorithm
* Distributed regularized binary logistic regression with stochastic gradient descent [[1]], [[2]]
	* Competitive with best extant large-scale supervised learning algorithms
	* Results provide direct estimation of probability of class membership 
* Implementation supports large-scale learning using Hadoop streaming	* Map/reduce implementation [[3]] allows embarrassingly parallel processing of training and test datasets to reduce wall clock time	* Supports the use of the hashing trick [[4]], [[5]] which allows memory requirements to be fixed with minimal increase of error	* Margin-based certainty [[6]] based on probability estimates supports active learning workflow [[7]] to reduce human annotation costs
# Installation
    $ git clone https://github.com/elsevierlabs/logistic-regression-sgd-mapreduce.git
    $ cd logistic-regression-sgd-mapreduce    $ chmod +x *.py

# Objects
The scripts use JSON objects to represent instances, models, tests and predictions. These objects can have additional keys associated with them beyond the ones specified below; for example, each instance can have a key/value pair providing an identifier, contain key/value pairs with additional provenance information, etc.

## Instances
    <instance>           ::= <labeled-instance> | <unlabeled-instance>
    <labeled-instance>   ::= { "date_created": <iso-date>, "random_key": <value>, "class": <class>, "features": <features> }
    <unlabeled-instance> ::= { "date_created": <iso-date>, "random_key": <value>, "features": <features> }
    <iso-date>           ::= a JSON string that is an ISO 8601 datetime with Zulu (GMT) time zone
    <value>              ::= a JSON float in the interval [0.0, 1.0]   
    <class>              ::= 0 | 1
    <features>           ::= { <fv-pair>, … <fv-pair>, <fv-pair> }
    <fv-pair>            ::= <feature>: <value>
    <feature>            ::= a JSON string

## Models
    <model>              ::= { "id": <uuid>, "date_created": <iso-date>, "mu": <float>, "eta": <float>, "N": <positive-integer>, "parameters": <parameters> }
    <uuid>               ::= a JSON string that is a UUID
    <float>              ::= a JSON float
    <count>              ::= a JSON int in the interval [0, inf)
    <parameters>         ::= { <parameter>, … <parameter>, <parameter> }
    <parameter>          ::= <feature>: <weight>
    <weight>             ::= a JSON float in the interval (-inf, inf)

## Tests
    <test>               ::= { "model": <uuid>, "date_created": <iso-date>, "confusion_matrix": <confusion-matrix> }
    <confusion-matrix>   ::= { "TP": <count>, "FP": <count>, "FN": <count>, "TN": <count> }
    
## Predictions
    <prediction>         ::= { "model": <uuid>, "date_created": <iso-date>, "margin": <margin>, "p": <p>, "prediction": <class>, "instance": <instance> }
    <margin>             ::= a JSON float in the interval [0.0, 1.0]
    <p>                  ::= a JSON float in the interval [0.0, 1.0]    
    
# Usage
The Python scripts implement the key parts of a complete active learning workflow: train, validate/test, predict and query.

## From a UNIX shell
For small data sets, the scripts can be run from the command line. This allows the user to evaluate the use of the scripts before using them in a Hadoop streaming environment.

### Converting data in SVM<sup><i>Light</i></sup> format into labeled instances
Convert a file with data in SVM<sup><i>Light</i></sup> [[8]] format into a file containing instances. The mapper uses a randomly-generated number between 0 and 1 as a key; the file produced by the reducer is then a shuffled list of the training instances, making train/test partitioning to be a simple matter of specifying a split percentage. In the below example, we'll download and use the example data described in the section "Getting started: some Example Problems / Inductive SVM" from [[8]]. 

    $ wget http://download.joachims.org/svm_light/examples/example1.tar.gz
    $ gunzip -c example1.tar.gz | tar xvf -
    $ cat example1/train.dat | ./parse_svmlight_mapper.py  | sort | ./parse_svmlight_reducer.py > train.data
    $ cat example1/test.dat | ./parse_svmlight_mapper.py  | sort | ./parse_svmlight_reducer.py > test.data

### Training a model
Generate a model by running a single pass of the learning algorithm over a training set of labeled instances. 

Three hyperparameters (MU, ETA and N) can be optionally set using environment variables. The SPLIT environment variable determines the train/test split; only those labeled instances with random_key greater than or equal to SPLIT are used to update the model parameters. The mapper produces the feature string as the key with the trained weight; the reducer averages the weights associated with each feature to generate the final set of parameters. Unlabeled instances are not processed.

    $ export MU=0.002   # the regularization parameter
    $ export ETA=0.5    # the learning rate
    $ export N=2000000  # the number of instances in the training set
    $ export SPLIT=0.3  # the fraction of the total set of labelled instances sampled for testing (this setting yields a 70/30 train/test split)
    $ cat train.data | ./train_mapper.py | sort | ./train_reducer.py > /path/to/your/model

### Validating and testing a model
Generate a timestamped record with a confusion matrix, computed by running a model against a test set of labeled instances. 

The location of the model is passed as an environment variable whose value is a valid URL. 

Validation is performed against a training set generated as shown above in the section "Converting data in SVM<sup><i>Light</i></sup> format into labeled instances". A model produced as shown above in the section "Training a model" is passed to the mapper through the MODEL environment variable. Labeled instances with random_key less than SPLIT are used to generate predictions. Unlabeled instances are not processed.

    $ export MODEL=file:///path/to/your/model # in this example we're loading from a file on the local system; note that this is expressed as a file: URL
    $ export SPLIT=0.3  # the fraction of the total set of labelled instances sampled for testing
    $ cat test.data | ./test_mapper.py | sort | ./test_reducer.py > validation
    
Additionally, a model can be tested against a separately created hold-out test set of labeled instances using the test_map.py mapper. All of the labelled instances in the test set will be used to generate predictions.
    
    $ export MODEL=file:///path/to/your/model # in this example we're loading from a file on the local system; note that this is expressed as a file URL
    $ export SPLIT=0.3  # the fraction of the total set of labelled instances sampled for testing
    $ cat test.data | ./test_mapper.py | sort | ./test_reducer.py > test
    
The utility script display_stats.py can be used to process the resulting JSON object in the files produced by test_reducer.py to display a summary of the test run with accuracy, recall, precision and F1 metrics:

    $ cat validation | ./display_stats.py
    $ cat test | ./display_stats.py
    
### Predicting classes for a set of instances
Generate a file containing prediction for each instance in an input set of instances.

Each prediction is represented as a JSON object with the following keys:

1. model: the UUID of the model used to generate the predictions
2. date_created: the date and time that the predictions were made
2. p: the estimated probability of class membership (i.e., p(y=1|x))
2. margin: a measure of the certainty of the prediction, calculated as abs(p(y=1|x) - p(y=0|x))
3. prediction: the prediction of class membership (0 or 1)
4. instance: the JSON representation of the instance
 
The output file is intended to support active learning workflows; smaller margin implies greater uncertainty given the model, so given the output is sorted in increasing order of margin, the first line in the file can be used as the most informative instance to provide to a subject matter expert for review to determine the correct class of the instance. Additionally, instances over a threshold margin can be automatically labled with the predicted class and added to a training set to refine the model. Note that in the example below we use the test.data training set as an example; though those examples are labeled the labeling is ignored by the mapper, and alternatively a file containing unlabeled instances as defined above can be supplied.

    $ export MODEL=file:///path/to/your/model
    $ cat test.data | ./predict_mapper.py | sort | ./predict_reducer.py > predictions
    
### Generating queries to pass to a human oracle
Generate a file that is a sampled set of instance predictions to pass to a (human) oracle for labeling.

Given a file generated as in the previous section, we use a priority queue to randomly sample a set of predictions, emitting them with the uncertainty margin as the key. The reducer then emits the top k in decreasing order of uncertainty margin.

    $ export K=30  # the number of queries to be sampled for labeling by an oracle
	$ cat predictions | ./query_mapper.py | sort | ./query_reducer.py > queries
    
## Using Elastic MapReduce (EMR)
For large-scale data sets, the scripts can be run using Hadoop streaming in Elastic MapReduce. First, install the AWS Ruby libraries for running EMR command from the shell. Then upload the Python scripts and data files to a bucket in S3. For each of the above steps, one supplies the appropriate links to the mapper and reducer scripts (as an "s3n" type URI,) provide the input and output files/buckets, and set environment variables as appropriate using the --cmdenv argument. For example the EMR command that parallels the example in the section "Training a model" above would be:

    $ ./elastic-mapreduce --create --stream \
		--input s3n://path/to/your/bucket/train.data \
		--mapper s3n://path/to/your/bucket/train_map.py \
		--reducer s3n://path/to/your/bucket/train_reduce.py \
		--output s3n://path/to/your/bucket/model
		--cmdenv N=2000
		
The result of running this EMR command would yield a bucket containing a file part-00000 that is a file with a single line consisting of the JSON object describing the trained model.
		
As another example, the EMR command that parallels the test example in the section "Validating and testing a model" above would be:

    $ ./elastic-mapreduce --create --stream \
		--input s3n://path/to/your/bucket/test.data \
		--mapper s3n://path/to/your/bucket/test_map.py \
		--reducer s3n://path/to/your/bucket/test_reduce.py \
		--output s3n://path/to/your/bucket/test
		--cmdenv MODEL=https://s3.amazonaws.com/path/to/your/bucket/model/part-00000
		
# License
This code is provided under the terms of an MIT License [[9]]. See the LICENSE file for the copyright notice.
    
# References

[[1]] Cohen, W. Stochastic Gradient Descent. <i>Downloaded from http://www.cs.cmu.edu/~wcohen/10-605/notes/sgd-notes.pdf</i> (2012).
[[2]] Zinkevich, M., Smola, A. & Weimer, M. Parallelized Stochastic Gradient Descent. <i>Advances in Neural Information Processing Systems 23, 1-9</i> (2010).[[3]] Lin, J. & Kolcz, A. Large-Scale Machine Learning at Twitter. <i>SIGMOD</i> (2012).[[4]] Weinberger, K., Dasgupta, A., Attenberg, J., Langford, J. & Smola, A. Feature hashing for large scale multitask learning. <i>ICML</i> (2009).[[5]] Attenberg, J., Weinberger, K., Dasgupta, A., Smola, A. & Zinkevich, M. Collaborative Email-Spam Filtering with the Hashing-Trick. <i>CEAS 2009</i> (2009).[[6]] Schein, A. & Ungar, L. Active Learning for Logistic Regression: An Evaluation. <i>Machine Learning 68(3), 235-265</i> (2007).[[7]] Sculley, D., Otey, M.E., Pohl, M., Spitznagel, B. & Hainsworth, J. Detecting Adversarial Advertisements in the Wild. <i>KDD’11</i> (2011).
[[8]] Joachims, T. SVM<sup><i>Light</i></sup> Support Vector Machine. <i>Downloaded from http://svmlight.joachims.org/</i> (2008).[[9]] Open Source Initiative (OSI). The MIT License. <i>Downloaded from http://www.opensource.org/licenses/mit-license.php</i> (2012).

[1]: http://www.cs.cmu.edu/~wcohen/10-605/notes/sgd-notes.pdf "Cohen, W. Stochastic Gradient Descent. Downloaded from http://www.cs.cmu.edu/~wcohen/10-605/notes/sgd-notes.pdf. (2012)."[2]: http://www.martin.zinkevich.org/publications/nips2010.pdf "Zinkevich, M., Smola, A. & Weimer, M. Parallelized Stochastic Gradient Descent. Advances in Neural Information Processing Systems 23, 1-9 (2010)."
[3]: http://www.umiacs.umd.edu/~jimmylin/publications/Lin_Kolcz_SIGMOD2012.pdf "Lin, J. & Kolcz, A. Large-Scale Machine Learning at Twitter. SIGMOD (2012)."[4]: http://arxiv.org/pdf/0902.2206.pdf "Weinberger, K., Dasgupta, A., Attenberg, J., Langford, J. & Smola, A. Feature hashing for large scale multitask learning. ICML (2009)."[5]: http://ceas.cc/2009/papers/ceas2009-paper-11.pdf "Attenberg, J., Weinberger, K., Dasgupta, A., Smola, A. & Zinkevich, M. Collaborative Email-Spam Filtering with the Hashing-Trick. CEAS 2009 (2009)."[6]: http://www.andrewschein.com/publications/scheinML2007.pdf "Schein, A. & Ungar, L. Active Learning for Logistic Regression: An Evaluation. Machine Learning 68(3), 235-265 (2007)."[7]: http://www.eecs.tufts.edu/~dsculley/papers/adversarial-ads.pdf "Sculley, D., Otey, M.E., Pohl, M., Spitznagel, B. & Hainsworth, J. Detecting Adversarial Advertisements in the Wild. KDD’11 (2011)."
[8]: http://svmlight.joachims.org/ "Joachims, T. SVM<sup><i>Light</i></sup> Support Vector Machine. Downloaded from http://svmlight.joachims.org/. (2008)."
[9]: http://www.opensource.org/licenses/mit-license.php "Open Source Initiative OSI - The MIT License"
