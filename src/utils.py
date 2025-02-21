def deduplicate(neuron_target, neuron_delete):
    for set in neuron_target:
        for layer in neuron_target[set]:
            neuron_target[set][layer] = neuron_target[set][layer] - neuron_delete[set][layer]

    return neuron_target