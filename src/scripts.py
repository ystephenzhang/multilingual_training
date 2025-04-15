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

def reverse_experiment(model_name, m_lang, lang, training_args, eval_method:Literal["sequential", "parallel"] = "parallel",
                       eval_dataset = ["gsm", "mmlu", "ppl"], force_retrain = True, training_mode="swift",
                       train_data_path="./assets", test_data_path="./test_data", output_path="./models/trained/"):
    for dataset in eval_dataset:
        '''try:
            acc = evaluate(model_name, mode=eval_method, dataset=dataset, lang=lang,
                        full_record=True, log_name=model_name.split('/')[-1], path=test_data_path)
            print("before reversion acc ", dataset, acc)
        except:
            print("Evaluation failed ", dataset, lang)'''
        acc = evaluate(model_name, mode=eval_method, dataset=dataset, lang=lang,
                        full_record=True, log_name=model_name.split('/')[-1], path=test_data_path)
        print("before reversion acc ", dataset, acc)
    if not os.path.exists("./output/"+ model_name.split('/')[-1] + '_' + m_lang + '.json'):
        model, tokenizer = load_model_from_name(model_name)
        neurons = detect_key_neurons(model, tokenizer, m_lang, test_size=5000)
    if not os.path.exists(output_path + model_name.split('/')[-1] + '_' + m_lang + '-to-' + lang) or force_retrain:
        #trained_model = reverse_training(model, tokenizer, n_lang = m_lang, lang = _lang)
        reverse_training(model_name, n_lang = m_lang, lang = lang, mode=training_mode, data_path=train_data_path,
                         output_path=output_path, kwargs=training_args)
        print("Training done for ", output_path + model_name.split('/')[-1] + '_' + m_lang + '-to-' + lang)
    else:
        print("Training already done for ", output_path + model_name.split('/')[-1] + '_' + m_lang + '-to-' + lang)
    if training_mode=="hf":
        checkpoint_path = get_hf_checkpoints(output_path + model_name.split('/')[-1] + '_' + m_lang + '-to-' + lang)
    elif training_mode=="swift":
        checkpoint_path = get_swift_checkpoints(output_path + model_name.split('/')[-1] + '_' + m_lang + '-to-' + lang)
    for dataset in eval_dataset:
        for c in checkpoint_path:
            '''try:
                acc = evaluate(c, mode=eval_method, dataset=dataset, lang=lang,
                            full_record=True, log_name=model_name.split('/')[-1] + "_" + c.split('/')[-1], suffix="reversed", path=test_data_path)
                print("reversed reversion acc ", dataset, acc)
            except:
                print("Evaluation failed ", dataset, lang)'''
            acc = evaluate(c, mode=eval_method, dataset=dataset, lang=lang,
                            full_record=True, log_name=model_name.split('/')[-1] + "_" + c.split('/')[-1], suffix="reversed", path=test_data_path)
            print("reversed reversion acc ", dataset, acc)
        #acc = evaluate(checkpoint_path, mode=eval_method, dataset=dataset, lang=lang,
        #                full_record=True, log_name=model_name.split('/')[-1], suffix="reversed", path=test_data_path)
        print("reversed reversion acc ", dataset, acc)


def detection_all(model_name, lang, atten_num=4000, ffn_num=12000, test_size=-1, suffix=""):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only = True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", local_files_only = True)
    for l in lang:
        print("Detecting neurons for", l)
        neurons = detect_key_neurons(model, tokenizer, l, atten_num=atten_num, ffn_num=ffn_num, test_size=test_size, suffix=suffix)
        print(l, "complete", len(neurons["attn_q"].keys()), len(neurons["attn_q"][0]))

def quick_eval(model_name, lang, mode="sequential"):
    #tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only = True)
    #model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", local_files_only = True)
    #model, tokenizer = load_model_from_name(model_name)

    #acc = evaluate_gsm(model, tokenizer, _lang, bsz=bsz, full_record=True, suffix='before-training', n=shots)
    checkpoint = get_latest_checkpoint(model_name)
    acc = evaluate(checkpoint, mode=mode, dataset="gsm", lang=lang,
                   full_record=True, log_name=model_name.split('/')[-1])
    print("before training acc: ", acc)

def complete_eval(model_pref, langs, datasets, mode="sequential"):
    for l in langs:
        #model_name = "Llama-3-8B_english-to-" + l
        model_name = model_pref + l
        model_path = get_latest_checkpoint(model_name)
        
        for d in datasets:
            acc = evaluate(model_path, mode=mode, dataset=d, lang=l, full_record=True)
            print(f"Acc of {l} on {d} is {acc}")
            
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


        


    