import random, os, json, sys
from itertools import groupby
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)
from transformers import AutoModelForCausalLM, AutoTokenizer

def detect_key_neurons(model, tokenizer, lang, test_size=1000, candidate_layers=range(32)) -> dict:
    """Detects neurons key to the language *lang* and writes to ../output/model_lang_neuron.txt 

    Args:
        model (AutoModelForCausalLM): loaded hf model.
        tokenizer (AutoTokenizer): loaded hf tokenizer.
        lang (str): one of ['english', 'chinese', 'french', 'russian']
        test_size (int, optional): number of entries used when detecting.
        candidate_layers (list, optional): list of layers to examine.
    """
    
    with open('./corpus_all/' + lang + '.txt', 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    #lines = random.sample(lines, test_size)
    lines = lines[:test_size] #Because using the same corpus for detection and training now, separating them.

    activate_key_sets = {
        "fwd_up" : [],
        "fwd_down" : [],
        "attn_q" : [],
        "attn_k" : [],
        "attn_v" : [],
        "attn_o" : []
    }

    for prompt in lines:
        count = 0
        try:
            hidden, answer, activate, o_layers = _prompting(model, tokenizer, prompt, candidate_layers)
            for key in activate.key():
                activate_key_sets[key].append(activate[key])
        except Exception as e:
            count += 1
            # Handle the OutOfMemoryError here
            print(count)
            print(e)

    print("Detection query complete; error: ", count)
    
    for set in activate_key_sets.keys():
        entries = activate_key_sets[set]
        common_layers = {}
        for layer in entries[0].keys():
            if all(layer in d for d in entries):
                arrays = [d[key] for d in entries]
                common_elements = set.intersection(*map(set, arrays))

                common_layers[layer] = common_elements
        activate_key_sets[set] = common_layers
        #final structure of important neurons: {"param_set": {"layer1": [neuron1, neuron2, ...], ...}, ...}
    
    file_path = "./output/" + model.name_or_path.split('/')[-1] + '_' + lang + '.json'
    with open(file_path, 'w') as f:
        json.load(f, activate_key_sets)
        
    return activate_key_sets
        

def _prompting(model, tokenizer, prompt, candidate_premature_layers):
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    hidden_states, outputs, activate, o_layers = model.generate(**{'input_ids':inputs.input_ids, 'attention_mask':inputs.attention_mask, 'max_new_tokens':1, 'candidate_premature_layers':candidate_premature_layers})
    hidden_embed = {}

    for i, early_exit_layer in enumerate(candidate_premature_layers):
        hidden_embed[early_exit_layer] = tokenizer.decode(hidden_states[early_exit_layer][0])
    answer = tokenizer.decode(outputs[0]).replace('<pad> ', '')
    answer = answer.replace('</s>', '')
    
    return hidden_embed, answer, activate, o_layers




    
    