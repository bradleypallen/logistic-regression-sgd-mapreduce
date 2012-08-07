import math, json

class ConfusionMatrix(object):

    def __init__(self, true_positives=0, false_positives=0, false_negatives=0, true_negatives=0):
        self.true_positives = true_positives
        self.false_positives = false_positives
        self.true_negatives = true_negatives
        self.false_negatives = false_negatives
        self.positives = true_positives + false_negatives
        self.negatives = false_positives + true_negatives
        self.trials = self.positives + self.negatives

    def reset(self):
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        self.positives = 0
        self.negatives = 0
        self.trials = 0

    def load(self, filename):
        with open(filename, 'rb') as file:
            object = json.load(file)
            self.true_positives = object["true_positives"]
            self.false_positives = object["false_positives"]
            self.true_negatives = object["true_negatives"]
            self.false_negatives = object["false_negatives"]
            self.positives = true_positives + false_negatives
            self.negatives = false_positives + true_negatives
            self.trials = self.positives + self.negatives

    def save(self, filename):
        object = { "true_positives" : self.true_positives, "false_positives" : self.false_positives, "true_negatives" : self.true_negatives, "false_negatives" : self.false_negatives }
        with open(filename, 'wb') as file:
            json.dump(object, file)
        
    def update(self, hypothesis, klass):
        self.trials += 1
        if klass == 1:
            self.positives += 1
            if hypothesis == 1:
                self.true_positives += 1
            else:
                self.false_negatives += 1
        else:
            self.negatives += 1
            if hypothesis == 0:
                self.true_negatives += 1
            else:
                self.false_positives += 1

    def accuracy(self):
        try:
            return float(self.true_positives + self.true_negatives) / float(self.trials)
        except ZeroDivisionError:
            return float('nan')

    def error_rate(self):
        return 1. - self.accuracy()

    def precision(self):
        try:
            return float(self.true_positives) / float(self.true_positives + self.false_positives)
        except ZeroDivisionError:
            return float('nan')

    def recall(self):
        try:
            return float(self.true_positives) / float(self.positives)
        except ZeroDivisionError:
            return float('nan')

    def f_measure(self, beta=1.):
        beta_squared = math.pow(beta, 2.)
        p = self.precision()
        r = self.recall()
        return (1. + beta_squared) * ((p * r) / ((beta_squared * p) + r))

    def false_positive_rate(self):
        try:
            return float(self.false_positives) / float(self.negatives)
        except ZeroDivisionError:
            return float('nan')

    def false_negative_rate(self):
        try:
            return float(self.false_negatives) / float(self.positives)
        except ZeroDivisionError:
            return float('nan')

    def positive_predictive_value(self):
        return self.precision()

    def false_discovery_rate(self):
        try:
            return float(self.false_positives) / float(self.true_positives + self.false_positives)
        except ZeroDivisionError:
            return float('nan')

    def negative_predictive_value(self):
        try:
            return float(self.true_negatives) / float(self.true_negatives + self.false_negatives)
        except ZeroDivisionError:
            return float('nan')

    def sensitivity(self):
        return self.recall()

    def specificity(self):
        try:
            return float(self.true_negatives) / float(self.negatives)
        except ZeroDivisionError:
            return float('nan')

    def balanced_classification_rate(self):
        return (self.sensitivity() + self.specificity()) / 2.

    def matthews_correlation_coefficient(self):
        try:
            return float((self.true_positives * self.true_negatives) - (self.false_positives * self.false_negatives)) / math.sqrt(float((self.true_positives + self.false_positives) * (self.true_positives + self.false_negatives) * (self.true_negatives + self.false_positives) * (self.true_negatives + self.false_negatives)))
        except ZeroDivisionError:
            return float('nan')

    def __str__(self):
        return "%d trials: TP=%d, FP=%d, FN=%d, TN=%d, acc=%3.2f, P=%3.2f, R=%3.2f, F1=%3.2f" % (self.trials, self.true_positives, self.false_positives, self.false_negatives, self.true_negatives, self.accuracy(), self.precision(), self.recall(), self.f_measure(1.0))

