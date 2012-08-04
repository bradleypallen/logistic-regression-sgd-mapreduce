#!/usr/bin/env python
 
import sys, json, datetime, urllib2, os
 
def main():
    model = json.loads(urllib2.urlopen(os.environ['MODEL']).readline().strip())
    date_created = datetime.datetime.utcnow().isoformat() + 'Z'
    for line in sys.stdin:
        prediction = line.strip().split('\t')
        print json.dumps({
            "model": model["id"],
            "date_created": date_created,
            "margin": float(prediction[0]),
            "p": float(prediction[1]),
            "prediction": int(prediction[2]),
            "instance": json.loads(prediction[3])
            })
 
if __name__ == "__main__":
    main()
