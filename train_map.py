#!/usr/bin/env python

import collections, math, sys, json

def main():
    mu = 0.002
    eta = 0.5
    t = 0
    N = 300.
    W = collections.defaultdict(float)
    A = collections.defaultdict(int)
    for line in sys.stdin:
        x = json.loads(line)
        sigma = sum([W[j] * x["features"][j] for j in x["features"].keys()])
        p = 1. / (1. + math.exp(-sigma)) if -100. < sigma else sys.float_info.min
        t += 1
        lambd4 = eta / (1 + (t / N))
        penalty = 1. - (2. * lambd4 * mu)
        for j in x["features"].keys():
            W[j] *= math.pow(penalty, t - A[j])
            W[j] += lambd4 * (float(x["class"]) - p) * x["features"][j]
            A[j] = t
    lambd4 = eta / (1 + (t / N))
    penalty = 1. - (2. * lambd4 * mu)
    for j in W.keys():
        W[j] *= math.pow(penalty, t - A[j])
        print "%s\t%17.16f" % (j, W[j])
    
if __name__ == '__main__':
    main()
