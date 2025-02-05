from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig, AutoModelForCausalLM
import torch
from dotenv import load_dotenv
import os

def get_quantization_config():
    bnb_config = BitsAndBytesConfig(
        #load_in_8bit = True
        load_in_4bit = True, 
        bnb_4bit_quant_type = "nf4", 
        bnb_4bit_use_double_quant = True
    )
    return bnb_config

def load_model_and_tokenizer(model_name:str):
    load_dotenv()
    # if (model_name == "mistralai/Mistral-7B-Instruct-v0.3"):
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_name, 
    #         quantization_config = get_quantization_config(), 
    #         torch_dtype = torch.bfloat16,  
    #         trust_remote_code = True,
    #         token = os.getenv("HF_KEY")
    #     )
    # #za varijantu koja je fine-tjunovana
    # else:
    #     model = AutoPeftModelForCausalLM.from_pretrained(
    #         model_name,
    #         low_cpu_mem_usage=True,
    #         torch_dtype=torch.float16,
    #         load_in_4bit=True,
    #         token = os.getenv("HF_KEY")
    #     )
    model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            quantization_config = get_quantization_config(), 
            torch_dtype = torch.bfloat16,  
            trust_remote_code = True,
            token = os.getenv("HF_KEY_Newest")
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        token = os.getenv("HF_KEY_Newest")
        )
    return model, tokenizer 

def load_pipeline(model, tokenizer):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer = tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return pipe

def get_model_response(pipe, context, question):
    prompt = f"""<s>[INST] You are a helpful assistant.
        You need to answer in Serbian language.
        Use the following context to answer the question below comprehensively:
        context = {context}
        [/INST] </s>question = {question}
        If you cannot answer with the context, just say "Nemam dovoljno informacija."
        """
    prompt3 = f"""<s>[INST] Odgovori na postavljeno pitanje samo na osnovu prosledjenog konteksta. 
                            kontekst = {context}
                            [/INST] </s>pitanje = {question}
                            """
    prompt_template = f"""
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    You need to answer in Serbian language.
    Use just the following context and no prior knowledge to answer the question below comprehensively:
    context = {context}
    Question: {question}<|eot_id|>
    If you cannot answer using the context, dont answer. 
    <|start_header_id|>assistant<|end_header_id|>
    """
    prompt_template2 = f"""<start_of_turn>user
        Odgovori detaljno na pitanje samo na osnovu prosledjenog konteksta, a ne na osnovu prethodnog znanja. 
        Odgovaraj samo na srpskom jeziku i to na cirilici.  
        Kontekst = {context} 
        Pitanje = {question}<end_of_turn>
        <start_of_turn>model"""

    sequences = pipe(
        prompt3,
        #prompt_template2,
        do_sample=True,
        max_new_tokens=300,
        temperature=0.6,
        top_k=50,
        top_p=0.90,
        num_return_sequences=1,

    ) 
    ret_string = sequences[0]["generated_text"]
    answer_without_prompt = ret_string.replace(prompt3, " ")
    return answer_without_prompt
    #return ret_string
