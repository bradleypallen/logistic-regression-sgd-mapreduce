#!/usr/bin/env python
 
from itertools import groupby
from operator import itemgetter
import sys, json
 
def read_mapper_output(file, separator='\t'):
    for line in file:
        yield line.rstrip().split(separator, 1)
 
def main(separator='\t'):
    data = read_mapper_output(sys.stdin, separator=separator)
    parameters = {}
    for current_parameter, group in groupby(data, itemgetter(0)):
        try:
            weights = [ float(weight) for current_parameter, weight in group ]
            parameters[current_parameter] = sum(weights) / float(len(weights))
        except ValueError:
            pass
    print json.dumps(parameters)
 
if __name__ == "__main__":
    main()
