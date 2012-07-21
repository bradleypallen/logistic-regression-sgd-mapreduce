#!/usr/bin/env python
 
import collections, math, sys, json
 
def main(separator='\t'):
    matrix = collections.defaultdict(int)
    for line in sys.stdin:
        trial = line.strip().split('\t')
        prediction = int(trial[0])
        klass = int(trial[1])
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
