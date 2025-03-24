from src.scripts import *

if __name__ == "__main__":
    #model_name = "./models/base/Llama-3.2-1B"
    #model_name = "./models/base/Llama-3.2-3B"
    model_name = "./models/base/Llama-3-8B"
    #model_name = "./models/Llama-3-8B_french"
    '''
    lang_set = [("chinese", "zh"), ("french", "fr"), ("german", "de"), ("thai", "th"), ("swahili", "sw")]
    for x in lang_set[-1]:
        #baseline_experiment(model_name, x[0], x[1])
        #quick_eval(model_name, x[0], x[1])
        reverse_experiment(model_name, "english", x[0], x[1])
    '''
    '''quick_eval(model_name, "chinese", "zh")
    model_name = "./models/Llama-3-8B_english-to-chinese"
    quick_eval(model_name, "chinese", "zh")'''
    reverse_experiment(model_name, "english", "chinese", "zh")