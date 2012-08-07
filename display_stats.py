#!/usr/bin/env python

import sys, json
from confusionmatrix import ConfusionMatrix as CM

def main():
    for line in sys.stdin:
        test = json.loads(line)
        test_cm = test["confusion_matrix"]
        print CM(test_cm["TP"], test_cm["FP"], test_cm["FN"], test_cm["TN"])
        

if __name__ == '__main__':
    main()
