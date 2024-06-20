import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

class LLM():
    def __init__(self) -> None:
        self.model = T5ForConditionalGeneration.from_pretrained("cointegrated/rut5-small-chitchat")
        self.tokenizer = T5Tokenizer.from_pretrained("cointegrated/rut5-small-chitchat")
    
    def answer(self, request):
        inputs = self.tokenizer(request, return_tensors='pt')
        with torch.no_grad():
            hypotheses = self.model.generate(
                **inputs, 
                do_sample=True, top_p=0.5, num_return_sequences=3, 
                repetition_penalty=2.5,
                max_length=32,
            )
        # for h in hypotheses:
        #     print(self.tokenizer.decode(h, skip_special_tokens=True))
        return self.tokenizer.decode(hypotheses[0], skip_special_tokens=True)