#!/usr/bin/env python
 
import math, sys, json, datetime, urllib2, os
 
def main():
    matrix = { "TP": 0, "FP": 0, "TN": 0, "FN": 0 }
    for line in sys.stdin:
        trial = line.strip().split('\t')
        prediction = int(trial[1])
        klass = int(trial[2])
        if klass == 1:
            if prediction == 1:
                matrix["TP"] += 1
            else:
                matrix["FN"] += 1
        else:
            if prediction == 0:
                matrix["TN"] += 1
            else:
                matrix["FP"] += 1
    print json.dumps(matrix)
 
if __name__ == "__main__":
    main()
