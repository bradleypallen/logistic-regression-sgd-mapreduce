#!/usr/bin/env python

import sys, os

def main():
    k = int(os.environ['K']) if os.environ.has_key('K') else 100
    for i in range(k):
        print sys.stdin.readline().strip()
    
if __name__ == '__main__':
    main()
