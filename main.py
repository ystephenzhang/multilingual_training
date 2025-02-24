from src.enhance import enhanced_training
from src.detection import detect_key_neurons
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

if __name__ == "__main__":
    #model_name = "andrijdavid/Llama3-1B-Base"
    model_name = "meta-llama/Meta-Llama-3-8B"
    lang = 'russian'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    if not os.path.exists("./output/" + model.name_or_path.split('/')[-1] + '_' + lang + '.json'):
        neurons = detect_key_neurons(model, tokenizer, lang, test_size=4)
        print(len(neurons["attn_q"].keys()), len(neurons["attn_q"][0]))
    enhanced_training(model, tokenizer, lang)