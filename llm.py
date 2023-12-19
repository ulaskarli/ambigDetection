import numpy as np
import time,torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, CodeLlamaTokenizer, GenerationConfig

class LLM():
    def __init__(self,model="gpt2",temperature=0.6,p=0.9 ,sequence_length=1024, llama_size="7b") -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seq_length = sequence_length
        self.bar = None
        
        if model == 'llama':
            self.model_path = "meta-llama/Llama-2-"+llama_size+"-chat-hf"
            access_token = "ENTER YOUR ACCESS TOKEN HERE"
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto", load_in_4bit=True,  use_auth_token=access_token)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True, use_auth_token=access_token)
            self.seq_length = 4096
            print("Model in use: "+self.model_path)
        elif model == 'codeLlama':
            self.tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-13b-Instruct-hf")
            self.model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-13b-Instruct-hf")
            self.seq_length = 4096
            print("Model in use: codeLlama")
        else:
            self.model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            print("Model in use: gpt2")

        self.model.eval()

    def multi_sample(self,n_samples,prompt_text,max_token = 300):
        prompt = [prompt_text for _ in range(n_samples)]
        self.sample_size = n_samples
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self.input_length = 1 if self.model.config.is_encoder_decoder else model_inputs.input_ids.shape[1]
        conf = GenerationConfig.from_pretrained(self.model_path, num_return_sequences=n_samples)
        output = self.model.generate(**model_inputs, max_new_tokens=max_token, generation_config = conf, return_dict_in_generate=True, output_scores=True)

        return output
    
    def analyze(self,output):
        transition_scores = self.model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True)
        generated_tokens = output.sequences[:, self.input_length:]
        seq_prob = np.zeros(self.sample_size)
        result = []
        for i in range(self.sample_size):
            l_out = 0.0
            for tok, score in zip(generated_tokens[i], transition_scores[i]):
                # | token | token string | logits | probability
                if not np.exp(score.item()) == 0.0:
                    seq_prob[i]+=score.item()
                    l_out+=1.0

            result.append((self.tokenizer.decode(output.sequences[i,self.input_length:], skip_special_tokens=True), np.exp(seq_prob[i]/l_out)))

        return result
    
    def tokenize_object_space(self,obj_list):
        object_token_dict={}
        for obj in obj_list:
            object_token_dict[obj] = self.tokenizer.encode(obj, return_tensors="pt").numpy()[0][1:]

        return object_token_dict

    def forward_pass(self,prompt):
        torch.cuda.empty_cache()
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = model_inputs.input_ids.shape[1]
        return (self.model.generate(**model_inputs, return_dict_in_generate=True, output_scores=True), input_length)
    
    def decode_single_token(self,token_id):
        return self.tokenizer.decode(token_id)

