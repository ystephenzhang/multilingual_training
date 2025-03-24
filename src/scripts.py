from src.enhance import *
from src.utils import *
from src.detection import detect_key_neurons
from src.evaluation import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import pdb
import math
def baseline_experiment(model_name, lang, _lang):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only = True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", local_files_only = True)

    tokenizer.pad_token = tokenizer.eos_token

    #acc = evaluate(model, tokenizer, _lang, full_record=True, suffix='before-training', n=8)
    #print("before training acc: ", acc)
    if not os.path.exists("./output/" + model.name_or_path.split('/')[-1] + '_' + lang + '.json'):
        neurons = detect_key_neurons(model, tokenizer, lang, test_size=10)
        print(len(neurons["attn_q"].keys()), len(neurons["attn_q"][0]))
    if not os.path.exists('./models/' + model_name.split('/')[-1] + '_' + lang):
        trained_model = enhanced_training(model, tokenizer, lang)
    else:
        checkpoint_path = get_latest_checkpoint('./models/' + model_name.split('/')[-1] + '_' + lang)
        trained_model = AutoModelForCausalLM.from_pretrained(checkpoint_path, device_map="auto")
        trained_model.name_or_path = './models/' + model_name.split('/')[-1] + '_' + lang
    acc = evaluate_gsm(trained_model, tokenizer, _lang, full_record=True, suffix='naively-trained', n=8)
    print("after naive training acc: ", acc)

def reverse_experiment(model_name, m_lang, lang, _lang):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only = True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", local_files_only = True)

    tokenizer.pad_token = tokenizer.eos_token

    #acc = evaluate_gsm(model, tokenizer, _lang, full_record=True, suffix='before-training', n=8)
    #print("before training acc: ", acc)
    if not os.path.exists("./output/" + model.name_or_path.split('/')[-1] + '_' + m_lang + '.json'):
        neurons = detect_key_neurons(model, tokenizer, m_lang, test_size=5000)
    if not os.path.exists('./models/' + model_name.split('/')[-1] + '_' + m_lang + '-to-' + _lang):
        #trained_model = reverse_training(model, tokenizer, n_lang = m_lang, lang = _lang)
        reverse_training(model, tokenizer, n_lang = m_lang, lang = _lang)
    checkpoint_path = get_latest_checkpoint('./models/' + model_name.split('/')[-1] + '_' + m_lang + '-to-' + _lang)
    trained_model = AutoModelForCausalLM.from_pretrained(checkpoint_path, device_map="auto")
    trained_model.name_or_path = './models/' + model_name.split('/')[-1] + '_' + m_lang + '-to-' + _lang
    acc = evaluate_gsm(trained_model, tokenizer, _lang, full_record=True, suffix='reversed', n=8)
    print("after reversion acc: ", acc)

def detection_all(model_name, lang):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only = True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", local_files_only = True)
    for l in lang:
        print("Detecting neurons for", l)
        neurons = detect_key_neurons(model, tokenizer, l, test_size=1000)
        print(l, "complete", len(neurons["attn_q"].keys()), len(neurons["attn_q"][0]))

def quick_eval(model_name, lang, _lang, shots=8):
    #tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only = True)
    #model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", local_files_only = True)
    model, tokenizer = load_model_from_name(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    acc = evaluate_gsm(model, tokenizer, _lang, full_record=True, suffix='before-training', n=shots)
    print("before training acc: ", acc)

def inspect_hidden_state(model_name, prompt, lang, log_dir = './output/hidden_state_log/'):
    model, tokenizer = load_model_from_name(model_name)
    
    batch = tokenizer(prompt, truncation=False, padding=False)
    input = {'input_ids':torch.tensor(batch["input_ids"]).to('cuda'), "attention_mask":torch.tensor(batch["attention_mask"]).to('cuda'), "max_new_tokens": 50}
    output = model.generate(**input, return_dict_in_generate=True, output_hidden_states=True, temperature=0.7, top_p=0.9, do_sample=True)
    hidden_states = output['hidden_states']
    hidden_states_by_layer = [torch.cat([step_hidden[layer_idx] for step_hidden in hidden_states], dim=1)
                              for layer_idx in range(len(hidden_states[-1]))] 
    logits = [model.lm_head(x) for x in hidden_states_by_layer]  # shape: (num_layers, batch_size, seq_len, vocab_size)
    predicted_token_ids = [torch.argmax(logit, dim=-1) for logit in logits]  # shaoe: (num_layers, batch_size, seq_len)
    decoded_tokens = [tokenizer.batch_decode(predicted)[0] for predicted in predicted_token_ids]
    english_ratio = [math.log10(count_english_ratio(text) + 1e-6) for text in decoded_tokens]
    if lang == "zh":
        input_lang_ratio = [math.log10(count_chinese_ratio(text=text) + 1e-6)  for text in decoded_tokens]
    print("writing into ", log_dir + model_name.split('/')[-1] + '_hidden_states_decoded.txt')
    with open(log_dir + model_name.split('/')[-1] + '_hidden_states_decoded.txt', 'w') as f:
        for i, entry in enumerate(decoded_tokens):
            f.write(f"\nLAYER{i}:")
            f.write(entry)
            f.write(f"END_OF_LAYER{i}\n") 
        f.write("LANGUAGE_RATIO: \n")
        for i, x in enumerate(input_lang_ratio):
            f.write(f"({i}: {x:.3f})\n")
    plot_two_lines(english_ratio, input_lang_ratio, "English", "Input Language", "Change of Ratios within Model: " + model_name.split('/')[-1])
    return decoded_tokens


        


    
    