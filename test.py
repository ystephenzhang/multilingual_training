from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def test_single_query(model, tokenizer, prompt, lang="chinese"):
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**{'input_ids':inputs.input_ids, 'attention_mask':inputs.attention_mask, 'max_new_tokens':512}, temperature=0.7
                ,deactivation_params=deactivate_params)