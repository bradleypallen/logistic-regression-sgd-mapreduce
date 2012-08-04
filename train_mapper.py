#!/usr/bin/env python

import collections, math, sys, json, os

def main():
    mu = float(os.environ['MU']) if os.environ.has_key('MU') else 0.002
    eta = float(os.environ['ETA']) if os.environ.has_key('ETA') else 0.5
    N = float(os.environ['N']) if os.environ.has_key('N') else 10000.
    t = 0
    W = collections.defaultdict(float)
    A = collections.defaultdict(int)
    for line in sys.stdin:
        x = json.loads(line)
        sigma = sum([W[j] * x["features"][j] for j in x["features"].keys()])
        p = 1. / (1. + math.exp(-sigma)) if -100. < sigma else sys.float_info.min
        t += 1
        lambd4 = eta / (1. + (float(t) / N))
        penalty = 1. - (2. * lambd4 * mu)
        for j in x["features"].keys():
            W[j] *= math.pow(penalty, t - A[j])
            W[j] += lambd4 * (float(x["class"]) - p) * x["features"][j]
            A[j] = t
    lambd4 = eta / (1. + (float(t) / N))
    penalty = 1. - (2. * lambd4 * mu)
    for j in W.keys():
        W[j] *= math.pow(penalty, t - A[j])
        print "%s\t%17.16f" % (j, W[j])
    
if __name__ == '__main__':
    main()
