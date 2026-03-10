import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class LLMInferenceEngine:
    def __init__(self, model_id, device="auto"):
        print(f"Initializing LLM Engine: {model_id}...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map=device,
            dtype=torch.float16
        )

    def generate(self, prompt, max_new_tokens=10, temperature=0.1):
        """プロンプトを受け取り、LLMの回答を返す"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_length=None,
                max_new_tokens=max_new_tokens, 
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 入力部分を除いた生成テキストのみをデコード
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[-1]:], 
            skip_special_tokens=True
        )
        return response.strip()