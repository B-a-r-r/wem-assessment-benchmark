from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
import torch


class LanguageModelHandler:
    """
    A class to handle the loading, configuration and use of a language model from HuggingFace.
    
    Attributes
    ----------
    model_name : str
        The name of the model to load from HuggingFace.
        
    __auth_token : str
        The HuggingFace authentication token.
    
    __model : AutoModelForCausalLM
        The language model to load from HuggingFace.
        
    __tokenizer : AutoTokenizer
        The tokenizer for the language model, which is the default one for the given model.
    
    __device : torch.device
        The hardware that should handle the computation (CPU or GPU, default is GPU).
    
    __logits_processor : LogitsProcessorList
        A list of orchestration parameters that will be used to process the logits (partly determining/adapting the output).
    """
    
    model_name: str
    __auth_token: str
    __model: AutoModelForCausalLM
    __tokenizer: AutoTokenizer
    __device: torch.device
    __logits_processor: LogitsProcessorList
    
    def __init__(self, model_name: str, auth_token: str) -> None:
        """
        Initialize a handler for a given LLM.
        
        Parameters
        ----------
        model_name : str
            The name of the model to load from HuggingFace.
            
        auth_token : str
            The HuggingFace authentication token.
        """
        
        try:
            login(auth_token)
            self.__auth_token = auth_token
        
        except Exception as e:
            print(f"Error during authentication: {e}")
            raise
        
        self.__tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.model_name = model_name
   
    def get_model(self) -> AutoModelForCausalLM:
        return self.__model
    
    def get_tokenizer(self) -> AutoTokenizer:
        return self.__tokenizer
    
    def get_device(self) -> torch.device:
        return self.__device
    
    def get_logits_processor(self) -> LogitsProcessorList:
        return self.__logits_processor     
        
    def configure_model(self, 
        device: str ="gpu",
        temperature: float =1.0, 
        top_p: float =1.0, 
        fp16: bool =False,
        quantization: int =None,
    ) -> None:
        """
        Configure the model with the given parameters.
        
        Parameters
        ----------
        device : str, optional
            The hardware that should handle the computation (CPU or GPU, default is GPU).
            
        temperature : float, optional
            An exploration variable which favors creativity.
            
        top_p : float, optional
            Only keep the tokens with a probability lower than top_p.
        
        fp16 : bool, optional
            Use half precision (16 bits) instead of full precision (32 bits) for non critical calculations.

        quantization : int, optional
            The quantization level to use for the model (4 or 8 bits).
        """
        
        #check if gpu is available before setting the device to gpu
        self.__device = torch.device(device if device == "cpu" else "cuda" if torch.cuda.is_available() else "cpu")
        
        self.__model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            device_map="auto",
            load_in_8bit=quantization == 8,
            load_in_4bit=quantization == 4,
        )
        
        if (fp16):
            self.__model.half()
            
        self.__logits_processor = LogitsProcessorList([
            TemperatureLogitsWarper(temperature),
            TopPLogitsWarper(top_p),
        ])
        
        
    def generate_response(self, prompt: str, rep_penalty: float =1.0, num_return_sequences: int =1, attention_mask: bool =False) -> str:
        """
        Generate a response from the model based on the given prompt.
        
        Parameters
        ----------
        prompt : str
            The input text to generate a response for.
            
        rep_penalty : float, optional
            The repetition penalty to apply to the generated text. This fovors diversity in the output.
            
        num_return_sequences : int, optional
            The number of sequences to return. If set to 1, only one sequence will be returned.
            
        attention_mask : bool, optional
            If True, the attention mask will be applied to the input text (padding + truncation).
            
        Returns
        -------
        str
            The generated response from the model.
        """
        
        inputs = self.__tokenizer(prompt, return_tensors="pt", padding=attention_mask, truncation=attention_mask).to(self.__device)
        
        # Generate the response
        with torch.no_grad():
            outputs = self.__model.generate(
                inputs["input_ids"],
                num_return_sequences = num_return_sequences,
                repetition_penalty = rep_penalty,
                logits_processor = self.__logits_processor,
                attention_mask=inputs["attention_mask"], 
                pad_token_id=self.__tokenizer.eos_token_id
            )
        
        response = self.__tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response
            

if __name__ == "__main__":
    
    def print_response(model_name, prompt, response):
        print(f"""\n
            Using the model {model_name} on {handler.get_device().type}:\n
            \n
            Prompt:\n {prompt}\n
            Response:\n {response}\n
        """)
    
    auth_token = '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'
    prompt = "Give me an animal with wings. Return only the name of the animal"
    
    model_name = "mistralai/Mistral-7B-v0.1"
    handler = LanguageModelHandler(model_name, auth_token)
    handler.configure_model(device="cuda")
    response = handler.generate_response(prompt)
    print_response(model_name, prompt, response)
    
    model_name = "meta-llama/Llama-4-Maverick-17B-128E"
    handler = LanguageModelHandler(model_name, auth_token)
    handler.configure_model(device="cuda")
    response = handler.generate_response(prompt)
    print_response(model_name, prompt, response)
        
        