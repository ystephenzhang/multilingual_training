from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import re

def answer_mapping(decoded):
    try:
        answer = re.findall(r'####\s(.+)', answer)[0]
        prd = re.findall(r"\d+\,?\.?\d*",answer)[-1]
        prd = float(prd.replace(',', '').rstrip('.')) if prd else prd
        prd = int(prd)
        answer = int(prd)

    except:
        try:
            prd = re.findall(r"\d+\,?\.?\d*",answer)[-1]
            prd = float(prd.replace(',', '').rstrip('.')) if prd else prd
            answer = int(prd)
        except:
            answer = -1
    
    return answer
