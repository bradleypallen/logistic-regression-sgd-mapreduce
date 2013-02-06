#!/usr/bin/env python

import sys, json

def main():
    matrix = { "TP": 0, "FP": 0, "TN": 0, "FN": 0 }
    for line in sys.stdin:
        test_cm = json.loads(line)
        if test_cm.has_key("TP") and test_cm.has_key("FP") and test_cm.has_key("FN") and test_cm.has_key("TN"):
        	matrix["TP"] += test_cm["TP"]
        	matrix["FP"] += test_cm["FP"]
        	matrix["FN"] += test_cm["FN"]
        	matrix["TN"] += test_cm["TN"]
    print json.dumps(matrix)
        
if __name__ == '__main__':
    main()