#!/usr/bin/env python

import collections, math, sys, json, urllib2, os

def main():
    model = json.loads(urllib2.urlopen(os.environ['MODEL']).readline().strip())
    split = float(os.environ['SPLIT']) if os.environ.has_key('SPLIT') else 0.3
    W = collections.defaultdict(float)
    for f, w in model['parameters'].items():
        W[f] = w
    for line in sys.stdin:
        x = json.loads(line)
        if x.has_key('class') and x["random_key"] <= split:
            prediction = 1 if 0. < sum([W[j] * x["features"][j] for j in x["features"].keys()]) else 0
            print '%d\t%d' % (prediction, x["class"])

if __name__ == '__main__':
    main()
