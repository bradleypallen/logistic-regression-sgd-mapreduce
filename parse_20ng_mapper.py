#!/usr/bin/env python

import sys, json, random, uuid

def main():
    for line in sys.stdin:
        example = line.strip().split(' ')
        id = str(uuid.uuid1())
        newsgroup = example[0]
        random_key = random.random()
        features = {}
        for fv in example[1:]:
            features[fv] = 1.
        print "%17.16f\t%s\t%s\t%s" % (random_key, id, newsgroup, json.dumps(features))
    
if __name__ == '__main__':
    main()
