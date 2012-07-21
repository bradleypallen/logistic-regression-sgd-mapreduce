#!/usr/bin/env python

import sys, json

def main():
    for line in sys.stdin:
        x = {}
        example = line[0:line.find('#')].strip().split(' ')
        x['class'] = 0 if example[0] == '-1' else 1
        x['features'] = {}
        for fv in [ pair.split(':') for pair in example[1:] ]:
            x['features'][fv[0]] = float(fv[1])
        print json.dumps(x)
    
if __name__ == '__main__':
    main()
