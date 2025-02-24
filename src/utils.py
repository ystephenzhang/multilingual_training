import os, json
import numpy as np
def deduplicate(neuron_target, neuron_delete):
    for set in neuron_target:
        for layer in neuron_target[set]:
            neuron_target[set][layer] = neuron_target[set][layer] - neuron_delete[set][layer]

    return neuron_target

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.int64):  # numpy int 类型转换为 Python int
        return int(obj)
    elif isinstance(obj, np.floating):  # numpy float 类型转换为 Python float
        return float(obj)
    else:
        return obj

def save_neuron(activate_neurons, path):
    for group in activate_neurons:
        entry = activate_neurons[group]
        activate_neurons[group] = {key: list(value) if isinstance(value, set) else value for key, value in entry.items()}
    with open(path, 'w') as f:
        json.dump(activate_neurons, f)

def read_neuron(path):
    with open(path, 'r') as f:
        data = json.load(f)
    for group in data:
        entry = data[group]
        data[group] = {key: set(value) if isinstance(value, list) else value for key, value in entry.items()}

    return data