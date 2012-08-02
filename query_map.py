#!/usr/bin/env python

import sys, json, random, os, heapq

def main():
    k = int(os.environ['K']) if os.environ.has_key('K') else 100
    heap = []
    for line in sys.stdin:
        heapq.heappush(heap, (random.random(), line.strip()))
        if len(heap) > k:
            heapq.heappop(heap)
    for sample in heap:
        prediction = json.loads(sample[1])
        print "%17.16f\t%s" % (prediction["margin"], sample[1])
    
if __name__ == '__main__':
    main()
