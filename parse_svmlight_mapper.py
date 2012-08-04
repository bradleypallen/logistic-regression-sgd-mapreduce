#!/usr/bin/env python

import sys, json, random, uuid

def main():
    for line in sys.stdin:
        example = line[0:line.find('#')].strip().split(' ')
        id = str(uuid.uuid1())
        klass = 0 if example[0] == '-1' else 1
        random_key = random.random()
        features = {}
        for fv in [ pair.split(':') for pair in example[1:] ]:
            features[fv[0]] = float(fv[1])
        print "%17.16f\t%s\t%d\t%s" % (random_key, id, klass, json.dumps(features))
    
if __name__ == '__main__':
    main()
