#!/usr/bin/env python

import collections, math, sys, json, urllib2, os

def main():
    model = json.loads(urllib2.urlopen(os.environ['MODEL']).readline().strip())
    W = collections.defaultdict(float)
    for f, w in model['parameters'].items():
        W[f] = w
    for line in sys.stdin:
        x = json.loads(line)
        sigma = sum([W[j] * x["features"][j] for j in x["features"].keys()])
        p = 1. / (1. + math.exp(-sigma)) if -100. < sigma else sys.float_info.min
        prediction = 1 if 0. < sigma else 0
        print '%17.16f\t%17.16f\t%d\t%s' % (abs(p - (1. - p)), p, prediction, json.dumps(x))
    
if __name__ == '__main__':
    main()
