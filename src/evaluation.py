from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import torch
import pandas as pd
import re, os
from tqdm import tqdm
from .mgsm_exemplars import MGSM_EXEMPLARS

import pdb

def answer_mapping(decoded, num_shots):
    answer = re.findall(r'answer is\s(.+)', decoded)[num_shots:]
    if len(answer):
        #answer = [i for text in answer for i in re.findall(r'\$?\d{1,3}(?:,\d{3})*', text) ]
        answer = [prd.replace(',', '').replace('$', '') for prd in answer] 
        answer = [int(i) for text in answer for i in re.findall(r'\d+', text)]
        #answer = [int(prd.split('.')[0].replace(',', '').replace('$', '')) for prd in answer]
    else:
        answer = re.findall(r"\d+\,?\.?\d*", decoded)
        if len(answer):
            answer = [int(answer[-1].replace(',', '').rstrip('.'))]
        else:
            answer = [-1]
    
    return answer

def _answer_mapping(decoded, num_shots = 0):
    try:
        answer = re.findall(r'answer is\s(.+)', decoded)[0]
        prd = re.findall(r"\d+\,?\.?\d*",answer)[-1]
        prd = float(prd.replace(',', '').rstrip('.')) if prd else prd
        answer = int(prd)

    except:
        try:
            prd = re.findall(r"\d+\,?\.?\d*",decoded)[-1]
            prd = float(prd.replace(',', '').rstrip('.')) if prd else prd
            answer = int(prd)
        except:
            answer = -1
    
    return answer

def answer_mapping_batch(decoded_batch, num_shots):
    ret = []
    for i, row in decoded_batch.iterrows():
        generated_answer = answer_mapping(row["generated_answer"], num_shots)
        print("Mapping: ", generated_answer, row["generated_answer"])
        answer = row["answer"]
        ret.append(int(answer in generated_answer))
    return ret

def construct_inference_dataset(path, tokenizer, examplar, lang, shots=4):
    df = pd.read_csv(path, sep="\t", names=["question","answer"]) 
    dataset = Dataset.from_pandas(df)
    def tokenize_function(example):
        templated_questions = [few_shot_inference(q, examplar, lang, n=shots) for q in example["question"]] 
        #templated_questions = [zero_shot_inference(q, lang) for q in example["question"]] 
        entry = tokenizer(templated_questions, truncation=True, padding="max_length", max_length=128 * shots)
        return entry
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['question','answer'])
    #tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    return tokenized_dataset, df

def few_shot_inference(question, examplar, lang, n=3) -> str:
    """
    Examplar is expected to be a dict: {"zh": {"1": {...}, ...}, ...}
    """
    instruction_set = {"zh":"请仿照给出的例子，逐步解答下面的问题\n", "en":"Please follow the examples and answer the given question step-by-step.\n"}
    template_set = {"zh":"{q}; {a}.\n", "en": "{q}; {a}\n"}
    question_set = {"zh":"问题：{q}; 逐步解答：", "en": "Question: {q}; Answer step-by-step:"}
    illustration = ""
    for i in range(n):
        q = examplar[lang][str(i+1)]["q"]
        a = examplar[lang][str(i+1)]["a"]
        filled = template_set[lang].format(q=q, a=a)
        illustration += filled
    question_filled = question_set[lang].format(q=question) 
    return instruction_set[lang] + "\n" + illustration + "\n" + question_filled

def zero_shot_inference(question, lang) -> str:
    question_set = {"zh":"问题：{q}; 逐步解答：", "en": "Question: {q};step-by-step Answer:"}
    return question_set[lang].format(q=question) 

def evaluate(model, tokenizer, lang, bsz=2, full_record=False, suffix='', n=4):
    path = './url-nlp/mgsm/mgsm_' + lang + '.tsv'
    dataset, dataframe = construct_inference_dataset(path, tokenizer, MGSM_EXEMPLARS, lang, shots=n)
    #dataset = construct_inference_dataset(path, tokenizer, MGSM_EXEMPLARS, lang)
    
    num_batches = len(dataset) // bsz + (1 if len(dataset) % bsz > 0 else 0)
    correctness = []
    log_path = './output/eval_log/' + model.name_or_path.split('/')[-1] + '_' + lang + '_' + suffix + '.txt'
    if os.path.exists(log_path):
        os.remove(log_path)
    for i in tqdm(range(num_batches), desc="Evaluating"):
        batch = dataset.select(range(i * bsz, min((i + 1) * bsz, len(dataset))))
        #print(len(batch["input_ids"]))
        input = {'input_ids':torch.tensor(batch["input_ids"]).to('cuda'), "attention_mask":torch.tensor(batch["attention_mask"]).to('cuda'), "max_new_tokens": 128}
        with torch.no_grad():
            #outputs = model.generate(**input, max_new_tokens=512)
            outputs = model.generate(**input, temperature=0.2)
        generated = tokenizer.batch_decode(outputs, skip_special_tokens=True) 
        batch_df = dataframe.loc[range(i * bsz, min((i + 1) * bsz, len(dataset)))]
        batch_df['generated_answer'] = generated
        correctness.extend(answer_mapping_batch(batch_df, num_shots=n))
        try:
            '''decoded_input = tokenizer.batch_decode(input['input_ids'], skip_special_tokens=True) 
            batch_df['decoded_input'] = decoded_input'''
            #pdb.set_trace()
            if full_record:
                with open(log_path, 'a+') as f:
                    for i, b in batch_df.iterrows():
                        f.write('INPUT: ' + b['question'] + '\t GT_ANS: ' + b['answer'] + '\t GENERATED_ANS: ' + b['generated_answer'] + 'END_OF_GENERATED\n')
        except:
            print("Invalid batch, ", len(batch_df), len(generated))
        
    #assert len(correctness) == len(dataset)
    correct = correctness.count(1)
    with open(log_path, 'a+') as f:
        f.write("correct, " + str(correct) + "\n")
        f.write("wrong, " + str(len(correctness) - correct) + "\n")
        f.write("accuracy, " + str(correct / len(correctness)) + "\n") 
    return correct / len(correctness)
        
def single_evaluation(model, tokenizer, lang):
    path = './url-nlp/mgsm/mgsm_' + lang + '.tsv'
    dataset, dataframe = construct_inference_dataset(path, tokenizer, MGSM_EXEMPLARS, lang)
    entry = dataset[0]
    input = {'input_ids':torch.tensor(entry["input_ids"]).to('cuda'), "attention_mask":torch.tensor(entry["attention_mask"]).to('cuda'), "max_new_tokens": 50}
    with torch.no_grad():
        outputs = model.generate(**input)
    generated = tokenizer.decode(outputs)
    print("ANSWER:\n\n", generated)

