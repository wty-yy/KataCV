import json

def read_json(path):
    with open(path, 'r') as file:
        ret = json.load(file)
    return ret