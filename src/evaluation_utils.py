from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
import torch
import pandas as pd
import re, os, json
from tqdm import tqdm
from .mgsm_exemplars import MGSM_EXEMPLARS
from .utils import *
import pdb

class Evaluator:
    def __init__(self, benchmark_path, type, model_name, lang, output_path="./eval_output", shots=8):
        self.b_path = benchmark_path
        self.o_path = output_path
        self.type = type
        self.model, self.tokenizer = load_model_from_name(model_name)
        self.lang = lang
        self.shots = shots
        self.exemplar = self.generate_exemplar()
    
    def update_evaluator(self, benchmark_path, type, lang):
        self.b_path = benchmark_path
        self.type = type
        self.lang = lang
    
    def generate_exemplar(self):
        if self.type == "gsm":
            return MGSM_EXEMPLARS
        if self.type == "squad":
            return self._squad_exemplar()
    
    def answer_mapping_batch(self, decoded_batch):
        ret = []

        mapper = None
        if self.type == "gsm":
            mapper = self._gsm_mapping
        if self.type == "squad":
            print("Not Implemented.")
            return ret

        for i, row in decoded_batch.iterrows():
            generated_answer = mapper(row["generated_answer"], self.shots, lang=self.lang)
            answer = str(row["answer"]).replace(",", "")
            #print("checking, ", answer, generated_answer)
            ret.append(int(answer in generated_answer))
        return ret
    
    def _gsm_mapping(decoded, num_shots, lang="en", ignore_decimal=True):
        head = {"en":"answer is", "fr":"La r\u00e9ponse est", "zh":"\u7b54\u6848\u662f", "th":"\u0e04\u0e33\u0e15\u0e2d\u0e1a\u0e04\u0e37\u0e2d", "sw":"Jibu ni", "de":"Die Antwort ist"}
        x = head[lang]
        answer = re.findall(fr'{x}\s(.+)', decoded)[num_shots:]
        #print("mapping answer heads: ", answer)
        if len(answer):
            #answer = [i for text in answer for i in re.findall(r'\$?\d{1,3}(?:,\d{3})*', text) ]
            answer = [prd.replace(',', '').replace('$', '') for prd in answer] 
            if not ignore_decimal:
                answer = [i.rstrip('.') for text in answer for i in re.findall(r'\d+\.?\d*', text)]
            else:
                answer = [i.split('.')[0] for text in answer for i in re.findall(r'\d+\.?\d*', text)]
            #print("results from answer heads", answer)
            #answer = [int(prd.split('.')[0].replace(',', '').replace('$', '')) for prd in answer]
        else:
            answer = re.findall(r"\d+\,?\.?\d*", decoded)
            #print("mapping numbers", answer)
            if len(answer):
                if not ignore_decimal:
                    answer = [x.replace(',', '').rstrip('.') for x in answer[-5:]]
                else:
                    answer = [x.replace(',', '').split('.')[0] for x in answer[-5:]]
            else:
                answer = ['-1']
            #print("results from numbers", answer) 
        #print('\n')
        return answer

    def _squad_mapping():
        return None        

    def _squad_exemplar(self):
        print("Not Implemented.")
        return None

    def construct_gsm_dataset(self):
        df = pd.read_csv(self.b_path, sep="\t", names=["question","answer"]) 
        dataset = Dataset.from_pandas(df)
        def tokenize_function(example):
            templated_questions = [self._few_shot(q, self.examplar, self.lang, n=self.shots) for q in example["question"]] 
            #templated_questions = [zero_shot_inference(q, lang) for q in example["question"]] 
            entry = self.tokenizer(templated_questions, truncation=True, padding="max_length", max_length=128 * self.shots)
            return entry
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['question','answer'])    

        return tokenized_dataset, df
    
    def construct_squad_dataset(self):
        with open(self.b_path, 'r') as f:
            inference_dataset = json.load(f)
            inference_dataset = inference_dataset["data"]
            
        template = {"en": "Context: {c}\nQuestion: {q}\nAnswer:{a}", "fr": "Contexte: {c}\nQuestion: {q}\nRéponse:{a}",
                        "de":"Kontext: {c}\nFrage: {q}\nAntwort:{a}", "zh":"上下文: {c}\n问题: {q}\n答案:{a}",
                        "th":"บริบท: {c}\nคำถาม: {q}\nคำตอบ:{a}", "sw":"Muktadha: {c}\nSwali: {q}\nJibu:{a}"}
        template = template[self.lang]

        input_dict = {"text":[], "id":[]}
        for passage in tqdm(inference_dataset):
            for para in passage["paragraphs"]:
                context = para["context"]
                for qa in para["qas"]:

                    question = template.format(c=context, q=qa["question"])
                    input_dict["text"].append()
                    input_dict["id"].append(qa["id"])

        input_dataset = Dataset.from_dict(input_dict)
        def tokenize_function(example):
            entry = self.tokenizer(example["text"], truncation=False, padding=True, return_tensors="pt")
            return entry
        tokenized_dataset = input_dataset.map(tokenize_function, batched=True)

        return tokenized_dataset                 

    def _few_shot(self, question) -> str:
        """
        Examplar is expected to be a dict: {"zh": {"1": {...}, ...}, ...}
        """
        if self.type == 'gsm':
            instruction_set = {"zh":"请仿照给出的例子，逐步解答下面的问题\n", "en":"Please follow the examples and answer the given question step-by-step.\n", "fr":"Veuillez suivre les exemples et répondre à la question donnée étape par étape.",
                            "th": "ปรดทำตามตัวอย่างและตอบคำถามที่ให้มาเป็นขั้นตอน.\n", "sw":"Tafadhali fuata mifano na ujibu swali lililotolewa hatua kwa hatua.\n",
                            "de": "Bitte folgen Sie den Beispielen und beantworten Sie die gegebene Frage Schritt für Schritt."}
            template = "{q}; {a}.\n"
            question_set = {"zh":"问题：{q}; 逐步解答：", "en": "Question: {q}; Answer step-by-step:Question: {q}; Answer step-by-step:", "fr":"Question: {q}; Réponse étape par étape:",
                            "th":"คำถาม: {q}; คำตอบทีละขั้นตอน:", "sw":"Swali: {q}; Jibu hatua kwa hatua:", "de":"Frage: {q}; Antwort Schritt für Schritt:"}
        if self.type == 'squad':
            raise NotImplementedError("SQUAD")

        illustration = ""
        for i in range(n):
            q = self.examplar[self.lang][str(i+1)]["q"]
            a = self.examplar[self.lang][str(i+1)]["a"]
            filled = template.format(q=q, a=a)
            illustration += filled
        question_filled = question_set[self.lang].format(q=question) 
        return instruction_set[self.lang] + "\n" + illustration + "\n" + question_filled    
    
    def serial_inference(self, batch):
        input = {'input_ids':torch.tensor(batch["input_ids"]).to('cuda'), "attention_mask":torch.tensor(batch["attention_mask"]).to('cuda'), "max_new_tokens": 120 if self.type == "gsm" else 15}
        with torch.no_grad():
            outputs = self.model.generate(**input, temperature=0.3, top_p=0.9)
        generated = self.tokenizer.batch_decode(outputs, skip_special_tokens=True) 
        return generated
    
    def evaluate(self):
        
        dataset, dataframe = construct_gsm_dataset(path, tokenizer, MGSM_EXEMPLARS, lang, shots=n)
    
        num_batches = len(dataset) // bsz + (1 if len(dataset) % bsz > 0 else 0)
        correctness = []
        log_path = './output/eval_log/' + model.name_or_path.split('/')[-1] + '_' + lang + '_' + suffix + '_gsm.json'

def answer_mapping_mmlu(decoded, lang="en"):
    head = {"zh":"答案：", "de":"Antwort:", "fr":"Réponse:", "sw":"Jibu:", "en":"Answer:"}
    answer = decoded.split(head[lang])[-1]
    mapped_answer = re.search(r"\s*([A-Da-d])\b", answer)
    #print("Generated", decoded, "End of Generated.", mapped_answer)
    if mapped_answer:
        return [mapped_answer.group(1).upper()]
    return ['-1']
    
def answer_mapping_gsm(decoded, lang="en", ignore_decimal=True):
    head = {"en":"answer is", "fr":"La r\u00e9ponse", "zh":"\u7b54\u6848\u662f", "th":"\u0e04\u0e33\u0e15\u0e2d\u0e1a\u0e04\u0e37\u0e2d", "sw":"Jibu ni", "de":"Die Antwort"}
    x = head[lang]
    answer = re.findall(fr'{x}\s(.+)', decoded)
    #print("mapping answer heads: ", answer)
    if len(answer):
        #answer = [i for text in answer for i in re.findall(r'\$?\d{1,3}(?:,\d{3})*', text) ]
        answer = [prd.replace(',', '').replace('$', '') for prd in answer] 
        answer = [re.sub(r'(\d)\s+(\d)', r'\1\2', text) for text in answer]
        if not ignore_decimal:
            answer = [i.rstrip('.') for text in answer for i in re.findall(r'\d+\.?\d*', text)]
        else:
            answer = [i.split('.')[0] for text in answer for i in re.findall(r'\d+\.?\d*', text)]
        #print("results from answer heads", answer)
        #answer = [int(prd.split('.')[0].replace(',', '').replace('$', '')) for prd in answer]
    else:
        answer = re.findall(r"\d+\,?\.?\d*", decoded)
        #print("mapping numbers", answer)
        if len(answer):
            if not ignore_decimal:
                answer = [x.replace(',', '').rstrip('.') for x in answer[-8:]]
            else:
                answer = [x.replace(',', '').split('.')[0] for x in answer[-8:]]
        else:
            answer = ['-1']
        #print("results from numbers", answer) 
    #print('\n')
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

def answer_mapping_batch(decoded_batch, lang="en", dataset="gsm"):
    ret = []
    log = []
    for i, row in decoded_batch.iterrows():
        if dataset=="gsm":
            generated_answer = answer_mapping_gsm(row["generated_answer"], lang=lang)
            answer = str(row["answer"]).replace(",", "")
        if dataset=="mmlu":
            generated_answer = answer_mapping_mmlu(row["generated_answer"], lang=lang)    
            answer = row["answer"]
        #print("checking, ", answer, generated_answer)
        ret.append(int(answer in generated_answer))
        log.append(generated_answer)
    return ret, log

def _answer_mapping_batch_mmlu(answer, generated, num_shots, lang="en"):
    ret = []
    log = []
    for gt, gen in zip(answer, generated):
        answer = answer_mapping_mmlu(gen, num_shots, lang)
        #print("checking, ", gt, answer)
        ret.append(int(gt in answer))
        log.append(gen)
    return ret, log

def construct_gsm_dataset(path, tokenizer, examplar, lang, shots=4):
    df = pd.read_csv(path, sep="\t", names=["question","answer"]) 
    dataset = Dataset.from_pandas(df)
    def tokenize_function(example):
        templated_questions = [few_shot_gsm(q, examplar, lang, n=shots) for q in example["question"]] 
        #templated_question = few_shot_gsm(example["question"], examplar, lang, n=shots)
        #templated_questions = [zero_shot_inference(q, lang) for q in example["question"]] 
        entry = tokenizer(templated_questions, padding=True, return_tensors="pt")
        return entry
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['question','answer'])
    #tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    return tokenized_dataset, df

def _construct_squad_dataset(path, tokenizer, lang, training=None, shots=0):
    with open(path, 'r') as f:
        inference_dataset = json.load(f)
        inference_dataset = inference_dataset["data"]
        
    template = {"en": "Context: {c}\nQuestion: {q}\nAnswer:{a}", "fr": "Contexte: {c}\nQuestion: {q}\nRéponse:{a}",
                    "de":"Kontext: {c}\nFrage: {q}\nAntwort:{a}", "zh":"上下文: {c}\n问题: {q}\n答案:{a}",
                    "th":"บริบท: {c}\nคำถาม: {q}\nคำตอบ:{a}", "sw":"Muktadha: {c}\nSwali: {q}\nJibu:{a}"}
    template = template[lang]

    input_dict = {"text":[], "id":[]}
    for passage in tqdm(inference_dataset):
        for para in passage["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:

                question = template.format(c=context, q=qa["question"])
                input_dict["text"].append()
                input_dict["id"].append(qa["id"])

    input_dataset = Dataset.from_dict(input_dict)
    def tokenize_function(example):
        entry = tokenizer(example["text"], truncation=False, padding=True, return_tensors="pt")
        return entry
    tokenized_dataset = input_dataset.map(tokenize_function, batched=True)

    return tokenized_dataset                 

def construct_mmlu_dataset(tokenizer, lang, path="openai/MMMLU", shots=8):
    mapping = {"zh":"ZH_CN", "de":"DE_DE", "fr":"FR_FR","sw":"SW_KE"}
    dataset = load_dataset(path, mapping[lang], split="test")
    examplar = dataset.select(random.sample(range(len(dataset)), shots))
    def tokenize_function(example):
        #pdb.set_trace()
        batch = [dict(zip(example.keys(), values)) for values in zip(*example.values())]
        templated_questions = [few_shot_mmlu(q, examplar, lang) for q in batch] 
        entry = tokenizer(templated_questions, padding=True, return_tensors="pt")
        return entry
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['Subject'])
    return tokenized_dataset

def few_shot_gsm(question, examplar, lang, n=3) -> str:
    """
    Examplar is expected to be a dict: {"zh": {"1": {...}, ...}, ...}
    """
    instruction_set = {"zh":"请仿照给出的例子，逐步解答下面的问题\n\n", "en":"Please follow the examples and answer the given question step-by-step.\n\n", "fr":"Veuillez suivre les exemples et répondre à la question donnée étape par étape.\n\n",
                       "th": "ปรดทำตามตัวอย่างและตอบคำถามที่ให้มาเป็นขั้นตอน.\n\n", "sw":"Tafadhali fuata mifano na ujibu swali lililotolewa hatua kwa hatua.\n\n",
                       "de": "Bitte folgen Sie den Beispielen und beantworten Sie die gegebene Frage Schritt für Schritt.\n\n"}
    template = "{q}\n{a}.\n\n"
    question_set = {"zh":"问题：{q}\n逐步解答：", "en": "Question: {q}\nStep-by-step Answer:", "fr":"Question: {q}\nRéponse étape par étape:",
                    "th":"คำถาม: {q}\nคำตอบทีละขั้นตอน:", "sw":"Swali: {q}\nJibu hatua kwa hatua:", "de":"Frage: {q}\nAntwort Schritt für Schritt:"}
    illustration = ""
    for i in range(n):
        q = examplar[lang][str(i+1)]["q"]
        a = examplar[lang][str(i+1)]["a"]
        filled = template.format(q=q, a=a)
        illustration += filled
    question_filled = question_set[lang].format(q=question) 
    return instruction_set[lang] + "\n" + illustration + "\n" + question_filled

def zero_shot_gsm(question, lang) -> str:
    question_set = {"zh":"问题：{q}; 逐步解答：", "en": "Question: {q}; step-by-step Answer:", "fr":"Question: {q}; Réponse étape par étape:"}
    return question_set[lang].format(q=question) 

def few_shot_mmlu(question, examplar, lang):
    instruction = {"zh":"请仿照例子，选择问题的正确答案。你只能回答一个字母。\n\n", "de":"Bitte wählen Sie nach dem Beispiel die richtige Antwort auf die Frage aus. Antworten Sie nur mit einem einzelnen Buchstaben.\n\n",
                   "fr":"Veuillez choisir la bonne réponse à la question en suivant l’exemple. Vous devez répondre par une seule lettre.\n\n",
                   "sw":"Tafadhali chagua jibu sahihi la swali kulingana na mfano. Jibu lako linapaswa kuwa herufi moja pekee.\n\n"}
    template = {"zh":"{q}\nA.{a}\nB.{b}\nC.{c}\nD.{d}\n答案：{s}.\n\n", "fr":"{q}\nA.{a}\nB.{b}\nC.{c}\nD.{d}\nRéponse:{s}.\n\n", 
                "de":"{q}\nA.{a}\nB.{b}\nC.{c}\nD.{d}\nAntwort:{s}.\n\n", "sw":"{q}\nA.{a}\nB.{b}\nC.{c}\nD.{d}\nJibu:{s}.\n\n"}
    template_q = {"zh":"{q}\nA.{a}\nB.{b}\nC.{c}\nD.{d}\n答案：", "fr":"{q}\nA.{a}\nB.{b}\nC.{c}\nD.{d}\nRéponse:", 
                "de":"{q}\nA.{a}\nB.{b}\nC.{c}\nD.{d}\nAntwort:", "sw":"{q}\nA.{a}\nB.{b}\nC.{c}\nD.{d}\nJibu:"}
    illustration = instruction[lang]
    for i, example in examplar.iterrows():
        #pdb.set_trace()
        filled = template[lang].format(q=example["Question"], a=example["A"],
                                      b=example["B"], c=example["C"], d=example["D"],
                                      s=example["answer"])
        illustration += filled
    #pdb.set_trace()
    ret = illustration + template_q[lang].format(q=question["Question"], a=question["A"],
                                               b=question["B"], c=question["C"], d=question["C"])
    return ret 

def evaluate_gsm(model, tokenizer, lang, bsz=16, full_record=False, suffix='', n=4):
    path = './url-nlp/mgsm/mgsm_' + lang + '.tsv'
    dataset, dataframe = construct_gsm_dataset(path, tokenizer, MGSM_EXEMPLARS, lang, shots=n)
    #dataset = construct_inference_dataset(path, tokenizer, MGSM_EXEMPLARS, lang)
    
    num_batches = len(dataset) // bsz + (1 if len(dataset) % bsz > 0 else 0)
    correctness = []
    log_path = './output/eval_log/' + model.name_or_path.split('/')[-1] + '_' + lang + '_' + suffix + '_gsm.json'
    '''
    if os.path.exists(log_path):
        os.remove(log_path)
    '''
    with open(log_path, 'w') as f:
        json.dump([], f)

    for i in tqdm(range(num_batches), desc="Evaluating"):
        batch = dataset.select(range(i * bsz, min((i + 1) * bsz, len(dataset))))
        #print(len(batch["input_ids"]))
        input = {'input_ids':torch.tensor(batch["input_ids"]).to("cuda"), 'attention_mask':torch.tensor(batch["attention_mask"]).to("cuda"), "max_new_tokens": 180}
        with torch.no_grad():
            #outputs = model.generate(**input, max_new_tokens=512)
            outputs = model.generate(**input, temperature=0.3, top_p=0.9, do_sample=True)
        generated = tokenizer.batch_decode(outputs, skip_special_tokens=True) 
        batch_df = dataframe.loc[range(i * bsz, min((i + 1) * bsz, len(dataset)))]
        batch_df['generated_answer'] = generated
        mapped, mapped_log = answer_mapping_batch(batch_df, num_shots=n, lang=lang)
        correctness.extend(mapped)
        batch_df['mapped_answer'] = mapped_log
        try:
            '''decoded_input = tokenizer.batch_decode(input['input_ids'], skip_special_tokens=True) 
            batch_df['decoded_input'] = decoded_input'''
            #pdb.set_trace()
            if full_record:
                with open(log_path, 'r') as f:
                    log = json.load(f)
                    #pdb.set_trace()
                for i, b in batch_df.iterrows():
                    log.append({"INPUT": b['question'], "GT": b['answer'], "GENERATED": b['generated_answer'], "MAPPED": b['mapped_answer']})
                        #f.write('INPUT: ' + b['question'] + '\t GT_ANS: ' + b['answer'] + '\t GENERATED_ANS: ' + b['generated_answer'] + 'END_OF_GENERATED\n')
                with open(log_path, 'w') as f:
                    json.dump(log, f)
        except:
            print("Invalid batch, ", len(batch_df), len(generated))
        
    #assert len(correctness) == len(dataset)
    correct = correctness.count(1)
    with open(log_path, 'r') as f:
        log = json.load(f)
    log.append({"correct": correct, "wrong": len(correctness) - correct, "accuracy": correct / len(correctness)})
    with open(log_path, 'w') as f:
        json.dump(log, f)
    return correct / len(correctness)
        
def evaluate_mmlu(model, tokenizer, lang, bsz=16, full_record=False, suffix='', n=4):
    dataset = construct_mmlu_dataset(tokenizer, lang)
    num_batches = len(dataset) // bsz + (1 if len(dataset) % bsz > 0 else 0)
    correctness = []
    generated = []
    mapped = []
    log_path = './output/eval_log/' + model.name_or_path.split('/')[-1] + '_' + lang + '_' + suffix + '_mmlu.json'

    with open(log_path, 'w') as f:
        json.dump([], f)

    for i in tqdm(range(num_batches), desc="Evaluating"):
        batch = dataset.select(range(i * bsz, min((i + 1) * bsz, len(dataset))))
        input = {'input_ids':torch.tensor(batch["input_ids"]).to("cuda"), 'attention_mask':torch.tensor(batch["attention_mask"]).to("cuda"), "max_new_tokens": 3}
        with torch.no_grad():
            #outputs = model.generate(**input, max_new_tokens=512)
            outputs = model.generate(**input, temperature=0.0, do_sample=False)
        generated_this = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if full_record:
            generated.extend(generated_this)
            mapped_bool, mapped_log = answer_mapping_batch_mmlu(batch["answer"], generated_this, num_shots=n, lang=lang)
            correctness.extend(mapped_bool)
            mapped.extend(mapped_log)
        else:
            mapped_bool, _ = answer_mapping_batch_mmlu(batch["answer"], generated_this, num_shots=n, lang=lang)
            correctness.extend(mapped_bool)
    try:
        if full_record:
            with open(log_path, 'r') as f:
                log = json.load(f)
            for i, b in enumerate(dataset):
                log.append({"GENERATED": generated[i], "GT": b['answer'], "MAPPED": mapped[i]})
            with open(log_path, 'w') as f:
                json.dump(log, f)
    except:
        print("Invalid batch, ", len(mapped), len(generated))
    
    correct = correctness.count(1)
    with open(log_path, 'r') as f:
        log = json.load(f)
    log.append({"correct": correct, "wrong": len(correctness) - correct, "accuracy": correct / len(correctness)})
    with open(log_path, 'w') as f:
        json.dump(log, f)
    return correct / len(correctness)
        
def _evaluate_squad(model, tokenizer, lang, bsz=16, suffix='', n=4):
    #First, training is required. But nah
   
    #Inference based on given format 
    path = "./xquad/xquad." + lang + ".json"
    dataset = construct_squad_dataset(path, tokenizer, lang, shots=n)
    prediction_path = './output/eval_log/' + model.name_or_path.split('/')[-1] + '_' + lang + '_' + suffix + '_squad.json'

    prediction = {}
    num_batches = len(dataset) // bsz + (1 if len(dataset) % bsz > 0 else 0) 
    for i in tqdm(range(3), desc="Evaluating"):
        batch = dataset.select(range(i * bsz, min((i + 1) * bsz, len(dataset))))
        input = {'input_ids':torch.tensor(batch["input_ids"]).to('cuda'), "attention_mask":torch.tensor(batch["attention_mask"]).to('cuda'), "max_new_tokens": 12}
        with torch.no_grad():
            #outputs = model.generate(**input, max_new_tokens=512)
            outputs = model.generate(**input, temperature=0.2, top_p=0.9)
        generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)  
        for j, id in enumerate(batch["id"]):
            prediction[id] = generated[j]
        '''for j, id in enumerate(batch["id"]):
            prediction[id] = ""'''

    from xquad.evaluate import evaluate
    with open(path, 'r') as f:
        data_dict = json.load(f)
    ret = evaluate(data_dict["data"], prediction)
    #print(ret) 
    with open(prediction_path, 'w') as f:
        json.dump([prediction, ret], f)

    return ret["f1"]

def single_evaluation_gsm(model, tokenizer, lang):
    path = './url-nlp/mgsm/mgsm_' + lang + '.tsv'
    dataset, dataframe = construct_gsm_dataset(path, tokenizer, MGSM_EXEMPLARS, lang)
    entry = dataset[0]
    input = {'input_ids':torch.tensor(entry["input_ids"]).to('cuda'), "attention_mask":torch.tensor(entry["attention_mask"]).to('cuda'), "max_new_tokens": 50}
    with torch.no_grad():
        outputs = model.generate(**input)
    generated = tokenizer.decode(outputs)
    print("ANSWER:\n\n", generated)

