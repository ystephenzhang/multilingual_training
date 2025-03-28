from src.scripts import *
import os

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
    model_name = "./models/base/Llama-3-8B"
    #model_name = "./models/base/Llama-3.2-3B"
    #model_name = "./models/base/Llama-3.2-1B"
    #model_name = "./models/Llama-3-8B_french"
    #model_name = "./models/Llama-3-8B_english-to-sw"
    model_pref1 = "/mnt/file1/zhangyang/multilingual_data/models/trained/Llama-3-8B_english-to-"
    model_pref2 = "/mnt/file2/zhangyang/multilingual_data/models/trained/Llama-3-8B_english-to-"
    
    '''for l in ["sw", "zh", "fr", "de", "th"]:
        quick_eval(model_name, l, mode="parallel")
    complete_eval(model_pref1, ["sw", "zh"], ["gsm"], "parallel") 
    complete_eval(model_pref2, ["fr", "de"], ["gsm"], "parallel")'''
    acc = evaluate(model_name, "parallel", "mmlu", full_record=True, suffix="test-mmlu-imple")

    #reverse_experiment(model_name, "english", "chinese", "zh")