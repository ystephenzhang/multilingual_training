import pdb, json, itertools
def read_neuron(path, layers_strat, types_strat, top_k = -1):
        layers_mapping = {
            "all": range(32),
            "front": range(10),
            "middle": range(10, 20),
            "back": range(20, 32),
            "none": []
        }
        types_mapping = {
            "all": ['fwd_up', 'fwd_down', 'attn_q', 'attn_k', 'attn_v', 'attn_o'],
            "atten_only": ['attn_q', 'attn_k', 'attn_v', 'attn_o'],
            "ffn_only": ['fwd_up', 'fwd_down']
        }
        activate_layers = layers_mapping[layers_strat]
        activate_types = types_mapping[types_strat]
        with open(path, 'r') as f:
            data = json.load(f)
        activte_neurons = {}
        for group in activate_types:
            entry = data[group]
            activte_neurons[group] = {key: set(value) if isinstance(value, list) else value for key, value in entry.items()}
            for key in activte_neurons[group]:
                if int(key) not in activate_layers:
                    activte_neurons[group][key] = {}
        pdb.set_trace()
        return activte_neurons