#!/usr/bin/env python

import sys, json
from confusionmatrix import ConfusionMatrix as CM

def main():
    for line in sys.stdin:
        cm = json.loads(line)
        print CM(cm["TP"], cm["FP"], cm["FN"], cm["TN"])
        
if __name__ == '__main__':
    main()
