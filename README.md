logistic-regression-sgd-mapreduce
=================================

# Overview
Python scripts for building binary classifiers using logistic regression with stochastic gradient descent, packaged for use with Hadoop map-reduce environments.

# ￼Algorithm
* Distributed regularized binary logistic regression with stochastic gradient descent [[1]], [[2]]
	* Competitive with best extant large-scale supervised learning algorithms
	* Results provide direct estimation of probability of class membership 
* Implementation supports large-scale learning	* Map/reduce implementation [[3]] allows embarrassingly parallel processing of training and test datasets to reduce wall clock time	* The hashing trick [[4]], [[5]] allows memory requirements to be fixed with minimal reduction of accuracy	* Margin uncertainty [[6]] based on probability estimates supports active learning workflow [[7]] to reduce human annotation costs

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

## Convert data in SVMLight format into instances
Convert a file with data in SVM<super>Light</super> format into a file containing instances.

    $  cat train.data.svmlight | parse_svmlight_examples.py | awk 'BEGIN{srand();} {printf "%06d %s\n", rand()*1000000, $0;}' | sort -n | cut -c8- > train.data
    $  cat test.data.svmlight | parse_svmlight_examples.py | awk 'BEGIN{srand();} {printf "%06d %s\n", rand()*1000000, $0;}' | sort -n | cut -c8- > test.data

## Train a model
Generate a model from training data.

    $ cat train.data | ./train_map.py | sort | ./train_reduce.py > model

## Test a model
Generate a confusion matrix based on running a model against test data.

    $ cat test.data | ./test_map.py model | sort | ./test_reduce.py
    
## Predict
Generate a tab-separated file containing a margin uncertainty for each instance in a data file.

    $ cat test.data | ./predict_map.py model | sort | ./predict_reduce.py
    
# References

[[1]] Cohen, W. Stochastic Gradient Descent. Downloaded from http://www.cs.cmu.edu/~wcohen/10-605/notes/sgd-notes.pdf. (2012).
[[2]] Zinkevich, M., Smola, A. & Weimer, M. Parallelized Stochastic Gradient Descent. Advances in Neural Information Processing Systems 23, 1-9 (2010).[[3]] Lin, J. & Kolcz, A. Large-Scale Machine Learning at Twitter. SIGMOD (2012).[[4]] Weinberger, K., Dasgupta, A., Attenberg, J., Langford, J. & Smola, A. Feature hashing for large scale multitask learning. ICML (2009)."[[5]] Attenberg, J., Weinberger, K., Dasgupta, A., Smola, A. & Zinkevich, M. Collaborative Email-Spam Filtering with the Hashing-Trick. CEAS 2009 (2009).[[6]] Schein, A. & Ungar, L. Active Learning for Logistic Regression: An Evaluation. Machine Learning 68(3), 235-265 (2007).[[7]] Sculley, D., Otey, M.E., Pohl, M., Spitznagel, B. & Hainsworth, J. Detecting Adversarial Advertisements in the Wild. KDD’11 (2011)

[1]: http://www.cs.cmu.edu/~wcohen/10-605/notes/sgd-notes.pdf "Cohen, W. Stochastic Gradient Descent. Downloaded from http://www.cs.cmu.edu/~wcohen/10-605/notes/sgd-notes.pdf. (2012)."[2]: http://www.martin.zinkevich.org/publications/nips2010.pdf "Zinkevich, M., Smola, A. & Weimer, M. Parallelized Stochastic Gradient Descent. Advances in Neural Information Processing Systems 23, 1-9 (2010)."[3]: http://www.umiacs.umd.edu/~jimmylin/publications/Lin_Kolcz_SIGMOD2012.pdf "Lin, J. & Kolcz, A. Large-Scale Machine Learning at Twitter. SIGMOD (2012)."[4]: http://arxiv.org/pdf/0902.2206.pdf "Weinberger, K., Dasgupta, A., Attenberg, J., Langford, J. & Smola, A. Feature hashing for large scale multitask learning. ICML (2009)."[5]: http://ceas.cc/2009/papers/ceas2009-paper-11.pdf "Attenberg, J., Weinberger, K., Dasgupta, A., Smola, A. & Zinkevich, M. Collaborative Email-Spam Filtering with the Hashing-Trick. CEAS 2009 (2009)."[6]: http://www.andrewschein.com/publications/scheinML2007.pdf "Schein, A. & Ungar, L. Active Learning for Logistic Regression: An Evaluation. Machine Learning 68(3), 235-265 (2007)."[7]: http://www.eecs.tufts.edu/~dsculley/papers/adversarial-ads.pdf "Sculley, D., Otey, M.E., Pohl, M., Spitznagel, B. & Hainsworth, J. Detecting Adversarial Advertisements in the Wild. KDD’11 (2011)."