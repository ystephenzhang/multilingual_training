import subprocess, os
from vllm import LLM, SamplingParams
import torch
import numpy as np
from tqdm import tqdm 
from transformers import AutoTokenizer
import pdb
sft = """swift pt \
    --model ./models/base/Llama-3.2-1B \
    --train_type full \
    --dataset swift/chinese-c4 \
    --torch_dtype bfloat16 \
    --streaming true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps $(expr 256 / 4) \
    --warmup_ratio 0.03 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --deepspeed zero3 \
    --max_length 8192 \
    --max_steps 100000
"""

def prepare_vllm(model="./models/Llama-3-8B", temperature=0.3, top_p = 0.9, max_tokens=64, tensor_parallel_size=None, stop = None, seed=42, max_model_len = None):
    print("vllm mnt: ", max_tokens)
    sampling_params = SamplingParams(temperature=temperature, top_p = top_p, max_tokens=max_tokens, stop=None, seed = seed)
    tokenizer = AutoTokenizer.from_pretrained(model,trust_remote_code=True)
    tensor_parallel_size = torch.cuda.device_count() if tensor_parallel_size is None else tensor_parallel_size
    llm = LLM(model=model,tensor_parallel_size=tensor_parallel_size,trust_remote_code=True, gpu_memory_utilization=0.8, max_model_len=max_model_len)
    return llm, sampling_params, tokenizer

def get_vllm_completion(llm, prompts, sampling_params):
    outputs = llm.generate(prompts, sampling_params)
    '''prompt = outputs[1].prompt
    generated_text = outputs[1].outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")'''
    responses = [output.outputs[0].text for output in outputs]
    return responses

def prompt_to_messages(role, prompt, messages=[], model_id="gpt-3.5-turbo-0125"):
    # user input (e.g., questions) to message format
    if "claude" in model_id:
        message = {"role": role, "content": [{"type": "text", "text": prompt}]}
        messages.append(message)
    elif "Sailor" in model_id:
        roles_map = {"system": "system", "user": "question", "assistant": "answer"}
        message = {"role": roles_map[role], "content": prompt}
        messages.append(message)
    else:
        message = {"role": role, "content": prompt}
        messages.append(message)
    return messages

def messages_to_prompt(messages, tokenizer):
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=True)
    else:
        prompt = ' '.join([f"{m['content']}" for m in messages])
    return prompt

def sequential_inference_hf(model, tokenizer, prompts, max_new_tokens=180, batch_size=16, seed=42):
    torch.manual_seed(seed)
    model.eval()
    all_responses = []
    
    # sort the prompts based on length and batch them accordingly, after completion sort them back to original order
    length_sorted_idx = np.argsort([len(sen) for sen in prompts])[::-1]
    prompts_sorted = [prompts[idx] for idx in length_sorted_idx]
    for i in tqdm(range(0, len(prompts_sorted), batch_size)):
        batch_prompts = prompts_sorted[i:i+batch_size]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            output_sequences = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                              max_new_tokens=max_new_tokens, temperature=0.6, top_p=0.95, do_sample=True)
        # responses = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        responses = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        outputs = [response.replace(p, '') for response, p in zip(responses, batch_prompts)]
        all_responses.extend(outputs)
    all_responses = [all_responses[idx] for idx in np.argsort(length_sorted_idx)]
    
    return all_responses 

def parallel_inference_vllm(llm, sampling_params, prompts):
    responses = get_vllm_completion(llm, prompts, sampling_params)
    return responses

def test_vllm():
    llm, sampling_params, tokenizer = prepare_vllm()
    prompt = "What is the capital of France?"
    messages = prompt_to_messages('user', prompt, messages=[])
    prompt = messages_to_prompt(messages, tokenizer)
    responses = get_vllm_completion(llm, prompt, sampling_params)
    #print(f"VLLM: {responses}")
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICE"] = "4,5,6,7"
    os.environ["NPROC_PER_NODE"] = "4"
    subprocess.run(test)
    
