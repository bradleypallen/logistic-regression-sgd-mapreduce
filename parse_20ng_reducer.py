#!/usr/bin/env python

import sys, json, random, datetime

def main():
    date_created = datetime.datetime.utcnow().isoformat() + 'Z'
    for line in sys.stdin:
        example = line.strip().split('\t')
        print json.dumps({
            "date_created": date_created,
            "random_key": float(example[0]),
            "id": example[1],
            "newsgroup": example[2],
            "features": json.loads(example[3])
            })
    
if __name__ == '__main__':
    main()
