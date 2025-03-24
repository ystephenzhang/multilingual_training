import random, os, json, sys
from itertools import groupby
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)
from transformers import AutoModelForCausalLM, AutoTokenizer
from .utils import *
import pdb

def detect_key_neurons(model, tokenizer, lang, test_size=3000, candidate_layers=[]) -> dict:
    """Detects neurons key to the language *lang* and writes to ../output/model_lang_neuron.txt 

    Args:
        model (AutoModelForCausalLM): loaded hf model.
        tokenizer (AutoTokenizer): loaded hf tokenizer.
        lang (str): one of ['english', 'chinese', 'french', 'russian']
        test_size (int, optional): number of entries used when detecting.
        candidate_layers (list, optional): list of layers to examine.
    """
    if not len(candidate_layers):
        candidate_layers = range(model.config.num_hidden_layers)
    
    with open('./corpus_all/' + lang + '.txt', 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    lines = random.sample(lines, test_size)
    #lines = lines[:test_size] #Because using the same corpus for detection and training now, separating them.

    activate_key_sets = {
        "fwd_up" : [],
        "fwd_down" : [],
        "attn_q" : [],
        "attn_k" : [],
        "attn_v" : [],
        "attn_o" : []
    }
    count = 0
    for prompt in tqdm(lines):
        #hidden, answer, activate, o_layers = detection_prompting(model, tokenizer, prompt, candidate_layers)
        try:
            hidden, answer, activate, o_layers = detection_prompting(model, tokenizer, prompt, candidate_layers)
            for key in activate.keys():
                activate_key_sets[key].append(activate[key])
        except Exception as e:
            count += 1
            # Handle the OutOfMemoryError here
            print(count)
            print(e)

    print("Detection query complete; error: ", count)
    
    for group in activate_key_sets.keys():
        entries = activate_key_sets[group]
        common_layers = {}
        for layer in entries[0].keys():
            if all(layer in d for d in entries):
                arrays = [d[layer] for d in entries]
                common_elements = set.intersection(*map(set, arrays))
                common_elements = {int(x) for x in common_elements}

                common_layers[layer] = common_elements
        activate_key_sets[group] = common_layers
        '''
        DETECTION SET TO LAYER

        from collections import Counter
        threshold = test_size // 100
        
        common_layers = {}
        for layer in entries[0].keys():
            neuron_counter = Counter(neuron for d in entries for neuron in d[layer][0])
            frequent_neurons = [neuron for neuron, count in neuron_counter.items() if count > threshold]
            common_layers[layer] = set([int(x) for x in frequent_neurons])

        activate_key_sets[group] = common_layers'''
        
        
        #final structure of important neurons: {"param_set": {"layer1": [neuron1, neuron2, ...], ...}, ...}
    
    file_path = "./output/" + model.name_or_path.split('/')[-1] + '_' + lang + '.json'
    save_neuron(activate_key_sets, file_path)
        
    return activate_key_sets
        

def detection_prompting(model, tokenizer, prompt, candidate_premature_layers):
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    hidden_states, outputs, activate, o_layers = model.generate(**{'input_ids':inputs.input_ids, 'attention_mask':inputs.attention_mask, 'max_new_tokens':1, 'candidate_premature_layers':candidate_premature_layers})
    hidden_embed = {}

    for i, early_exit_layer in enumerate(candidate_premature_layers):
        hidden_embed[early_exit_layer] = tokenizer.decode(hidden_states[early_exit_layer][0])
    answer = tokenizer.decode(outputs[0]).replace('<pad> ', '')
    answer = answer.replace('</s>', '')
    
    return hidden_embed, answer, activate, o_layers




    
    