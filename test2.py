from src.scripts import *
import os

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
    model_name = "./models/trained/Llama-3.2-1B_english-to-"
    #model_name = "./models/base/Llama-3.2-3B"
    model_base = "./models/base/Llama-3-8B"
    #model_name = "./models/Llama-3-8B_french"
    #model_name = "./models/Llama-3-8B_english-to-sw"
    model_pref1 = "/mnt/file1/zhangyang/multilingual_data/models/trained/Llama-3-8B_english-to-"
    model_pref2 = "/mnt/file2/zhangyang/multilingual_data/models/trained/Llama-3-8B_english-to-"
    
    '''
    acc = evaluate(model_base, "parallel", "gsm", full_record=True, shots=4, suffix="base")
    print(f"acc {acc}")
    '''
    for l in ["sw","zh"]:
        acc = evaluate(model_base, "perplexity", "ppl", lang=l, bsz=4, full_record=True, shots=4, suffix="base", path="test_data/")
        print(f"lang {l}, base acc {acc}")
        acc = evaluate(get_latest_checkpoint(model_pref1 + l), "perplexity", "ppl", lang=l, bsz=4, full_record=True, shots=4, suffix="reversed", path="test_data/")
        print(f"lang {l}, rev acc {acc}")
    for l in ["fr","de"]:
        acc = evaluate(model_base, "perplexity", "ppl", lang=l, bsz=4, full_record=True, shots=4, suffix="base", path="test_data/")
        print(f"lang {l}, base acc {acc}")
        acc = evaluate(get_latest_checkpoint(model_pref2 + l), "perplexity", "ppl", lang=l, bsz=4, full_record=True, shots=4, suffix="reversed", path="test_data/")
        print(f"lang {l}, rev acc {acc}") 
    '''
    complete_eval(model_pref1, ["sw", "zh"], ["gsm"], "parallel") 
    complete_eval(model_pref2, ["fr", "de"], ["gsm"], "parallel")
    '''

    #reverse_experiment(model_name, "english", "chinese", "zh")