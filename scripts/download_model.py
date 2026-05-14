import os
# ✅ 关键：国内镜像，解决超时！速度飞起来
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 模型名称
model_name = "Qwen/Qwen2.5-3B-Instruct-AWQ"
# 模型下载到这个文件夹
cache_dir = "/home/user/Qwen_codegen_web/models/Qwen2.5-3B"

print("loading model...")

# ✅ 自动用GPU + 国内高速下载 + 自定义缓存路径
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    device_map="auto",    
    dtype="auto",        # ✅ 自动用GPU（没有GPU才用CPU）
    trust_remote_code=True
)

print("loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    trust_remote_code=True
)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

print("building prompt...")
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("generating...")
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=256,
    do_sample=True,
    top_p=0.8,
    temperature=0.7
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("\n=== 回答 ===")
print(response)
