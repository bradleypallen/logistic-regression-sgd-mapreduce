#!/usr/bin/env python
 
import collections, math, sys, json, datetime, urllib2, os
 
def main():
    model = json.loads(urllib2.urlopen(os.environ['MODEL']).readline().strip())
    date_created = datetime.datetime.utcnow().isoformat() + 'Z'
    matrix = collections.defaultdict(int)
    for line in sys.stdin:
        trial = line.strip().split('\t')
        prediction = int(trial[0])
        klass = int(trial[1])
        if klass == 1:
            if prediction == 1:
                matrix["TP"] += 1
            else:
                matrix["FN"] += 1
        else:
            if prediction == 0:
                matrix["TN"] += 1
            else:
                matrix["FP"] += 1
    print json.dumps({
            "model": model["id"],
            "date_created": date_created,
            "confusion_matrix": matrix
        })
 
if __name__ == "__main__":
    main()
