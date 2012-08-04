#!/usr/bin/env python
 
from itertools import groupby
from operator import itemgetter
import sys, json, datetime, uuid, os
 
def read_mapper_output(file, separator='\t'):
    for line in file:
        yield line.rstrip().split(separator, 1)
 
def main(separator='\t'):
    id = str(uuid.uuid1())
    date_created = datetime.datetime.utcnow().isoformat() + 'Z'
    mu = float(os.environ['MU']) if os.environ.has_key('MU') else 0.002
    eta = float(os.environ['ETA']) if os.environ.has_key('ETA') else 0.5
    N = os.environ['N'] if os.environ.has_key('N') else 10000
    parameters = {}
    data = read_mapper_output(sys.stdin, separator=separator)
    for feature, group in groupby(data, itemgetter(0)):
        try:
            weights = [ float(weight) for feature, weight in group ]
            parameters[feature] = sum(weights) / float(len(weights))
        except ValueError:
            pass
    print json.dumps({
        "id": id,
        "date_created": date_created,
        "mu": mu,
        "eta": eta,
        "N": N,
        "parameters": parameters
        })
 
if __name__ == "__main__":
    main()
