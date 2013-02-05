#!/usr/bin/env python
 
import sys, json, datetime, uuid, os
 
def main(separator='\t'):
    id = str(uuid.uuid1())
    date_created = datetime.datetime.utcnow().isoformat() + 'Z'
    mu = float(os.environ['MU']) if os.environ.has_key('MU') else 0.002
    eta = float(os.environ['ETA']) if os.environ.has_key('ETA') else 0.5
    n_models_key = os.environ['N_MODELS_KEY'] if os.environ.has_key('N_MODELS_KEY') else 'MODEL'
    N = os.environ['N'] if os.environ.has_key('N') else 10000
    parameters = {}
    for line in sys.stdin:
        (feature, sigma) = line.strip().split(separator)
        parameters[feature] = float(sigma)
    n_models = float(parameters[n_models_key])
    for f, sigma in parameters.items():
        parameters[f] = parameters[f] / n_models
    del parameters[n_models_key]
    print json.dumps({
        "id": id,
        "date_created": date_created,
        "models": n_models,
        "mu": mu,
        "eta": eta,
        "N": N,
        "parameters": parameters
        })
 
if __name__ == "__main__":
    main()