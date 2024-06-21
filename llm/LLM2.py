import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

os.environ['TRANSFORMERS_CACHE'] = 'D:\\transformers_cache\\'

torch.random.manual_seed(0)

# model = AutoModelForCausalLM.from_pretrained(
#     "microsoft/Phi-3-mini-128k-instruct", 
#     device_map="cuda", 
#     torch_dtype="auto", 
#     trust_remote_code=True, 
# )
# tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

# messages = [
#     {"role": "system", "content": "You russian native speaker who don't like talk much so you keep things straight and simple"},
#     {"role": "user", "content": "Расскажи мне, как приготовить пельмени?"},
# ]

# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.bfloat16, 
#     device_map="auto"
# )

# generation_args = {
#     "max_new_tokens": 250,
#     "return_full_text": True,
#     "temperature": 0.0,
#     "do_sample": False
# }

# output = pipe(messages, **generation_args)
# print(output[0]['generated_text'])

class LLMphi3():
    def __init__(self,
                 max_context_len=30,
                 max_new_tokens=100) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-128k-instruct", 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True, 
        )
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
        self.messages = [{"role": "system", "content": "You russian native speaker who don't like talk much so you keep things straight and simple"}]
        self.generation_args = {
            "max_new_tokens": max_new_tokens,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False
        }
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        self.max_context_len = max_context_len
    
    def clean_context(self):
        self.messages = self.messages[0]

    def answer(self, request):
        if len(self.messages) > self.max_context_len:
            self.clean_context()
        self.messages.append({"role": "user", "content": request})
        output = self.pipe(self.messages, **self.generation_args)
        return output[0]['generated_text']

