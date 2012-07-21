#!/usr/bin/env python
 
import sys
 
def main():
    with sys.stdin as file:
        for line in file:
            try:
                print line.strip()
            except ValueError:
                pass
 
if __name__ == "__main__":
    main()
