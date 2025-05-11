from langchain import llms, chains
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModel


model_id = 'meta-llama/Llama-3.2-1B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
prompt = "Give me a list of 5 animal species. Only answer with the list of species nammes."

pipe = pipeline(
    "text2text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=100
)

local_llm = llms.HuggingFacePipeline(pipeline=pipe)

llm_chain = chains.LLMChain(
    prompt=prompt, 
    llm=local_llm
)

print(llm_chain.run(prompt))