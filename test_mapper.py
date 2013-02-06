#!/usr/bin/env python

import collections, math, sys, json, urllib2, os

def main():
    model = json.loads(urllib2.urlopen(os.environ['MODEL']).readline().strip())
    W = collections.defaultdict(float)
    for f, w in model['parameters'].items():
        W[f] = w
    for line in sys.stdin:
        x = json.loads(line)
        prediction = 1 if 0. < sum([W[j] * x["features"][j] for j in x["features"].keys()]) else 0
        print '%s\t%d\t%d' % (x["id"], prediction, x["class"])

if __name__ == '__main__':
    main()
