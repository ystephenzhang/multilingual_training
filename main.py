from src.enhance import enhanced_training
from src.detection import detect_key_neurons
from src.evaluation import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

if __name__ == "__main__":
    #model_name = "andrijdavid/Llama3-1B-Base"
    model_name = "meta-llama/Meta-Llama-3-8B"
    lang = 'chinese'
    _lang = 'zh'
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", local_files_only=True)

    tokenizer.pad_token = tokenizer.eos_token

    '''
    acc = evaluate(model, tokenizer, _lang, full_record=True, suffix='before-training')
    print("before training acc: ", acc)
    '''
    if not os.path.exists("./output/" + model.name_or_path.split('/')[-1] + '_' + lang + '.json'):
        neurons = detect_key_neurons(model, tokenizer, lang, test_size=1000)
        print(len(neurons["attn_q"].keys()), len(neurons["attn_q"][0]))
    #trained_model = enhanced_training(model, tokenizer, lang)
    trained_model = AutoModelForCausalLM.from_pretrained('./model/Meta-Llama-3-8B_chinese/checkpoint-31')
    acc = evaluate(trained_model, tokenizer, _lang, full_record=True, suffix='naively-trained')
    print("after naive training acc: ", acc)