#!/usr/bin/env python
 
from itertools import groupby
from operator import itemgetter
import sys, json, datetime, uuid, os
 
def read_mapper_output(file, separator='\t'):
    for line in file:
        yield line.rstrip().split(separator, 1)
 
def main(separator='\t'):
    data = read_mapper_output(sys.stdin, separator=separator)
    for feature, group in groupby(data, itemgetter(0)):
        print "%s\t%17.16f" % (feature, sum([ float(weight) for feature, weight in group ]))
 
if __name__ == "__main__":
    main()
