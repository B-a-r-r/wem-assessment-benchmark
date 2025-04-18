from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, TemperatureLogitsWarper, AutoConfig, TopPLogitsWarper
from torch import cuda, device, no_grad, float32 
from accelerate import disk_offload, init_empty_weights
from dotenv import dotenv_values
from gc import collect
from os import remove, listdir

#
# author:   Clément BARRIERE
# github:   @B-a-r-r
#

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
    offload_dir: str
    __auth_token: str
    __model: AutoModelForCausalLM
    __tokenizer: AutoTokenizer
    __device: device
    __logits_processor: LogitsProcessorList
    
    def __init__(self, model_name: str, auth_token: str, offload_dir: str ="./.llm-offloads") -> None:
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
        
        self.__tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name,
        )
        self.__tokenizer.pad_token = self.__tokenizer.eos_token #allow to normalize the oken sequences to enhanced open-ended generation
        
        self.model_name = model_name
        self.offload_dir = offload_dir
        self.__model = None
        self.__device = None
        self.__logits_processor = None
        
   
    def get_model(self) -> AutoModelForCausalLM:
        return self.__model
    
    def get_tokenizer(self) -> AutoTokenizer:
        return self.__tokenizer
    
    def get_device(self) -> device:
        return self.__device
    
    def get_logits_processor(self) -> LogitsProcessorList:
        return self.__logits_processor     
        
    def configure_model(self, 
        use_gpu: bool =True,
        temperature: float =0.1, 
        top_p: float =1.0, 
        quantization: int =None,
        local_offload: bool =False,
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

        quantization : int, optional
            The quantization level to use for the model (4 or 8 bits).
            
        local_offload : bool, optional
            If True, the model will be offloaded to the local machine's disk
        """
        assert self.__model is None, "Model is already cnofigured and freezed."
        
        #check if gpu is available before setting the device to gpu
        self.__device = device("cpu" if (not use_gpu or not cuda.is_available()) else "cuda")
    
        if ( #the local offload is applyed only if the wanted hardware is available
            (self.__device.type == "cuda" and local_offload)
            or (not use_gpu and local_offload)
        ):
            config = AutoConfig.from_pretrained(self.model_name, torch_dtype=float32)
            with init_empty_weights():
                self.__model = AutoModelForCausalLM.from_config(config)
            
            self.__model = disk_offload(self.__model, offload_dir=self.offload_dir)
            
        #if the wanted hardware is not available, or the local offload is not wanted, load the model normally
        else:
            self.__model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path = self.model_name,
                device_map = "auto",
                load_in_8bit = quantization == 8,
                load_in_4bit = quantization == 4,
                torch_dtype = float32,
            )
            
        self.__logits_processor = LogitsProcessorList([
            TemperatureLogitsWarper(temperature),
            TopPLogitsWarper(top_p),
        ])
        
        self.__model.to_empty(device=self.__device)
        
        
    def generate_response(self, 
        prompt: str, 
        rep_penalty: float =1.0, 
        num_return_sequences: int =1, 
        attention_mask: bool =True, 
        max_lenght_input: int =None, 
        max_lenght_output: int =None,
        max_new_tokens: int =50,
    ) -> str:
        """
        Generate a response from the model based on the given prompt.
        
        Parameters
        ----------
        prompt : str
            The input text to generate a response for.
            
        rep_penalty : float, optionals
            The repetition penalty to apply to the generated text. This fovors diversity in the output.
            
        num_return_sequences : int, optional
            The number of sequences to return. If set to 1, only one sequence will be returned.
            
        attention_mask : bool, optional
            If True, the attention mask will be applied to the input text (padding + truncation).
            
        max_lenght_input : int, optional
            The maximum length of the input text. If the input text is longer than this value, it will be truncated.
            If None, the LLM might generate a response too long, going beyong the expectations of the prompt.
            
        max_lenght_output : int, optional
            The maximum length of the output text. If the output text is longer than this value, it will be truncated.
            If None, the LLM might generate a response too long, going beyong the expectations of the prompt.
            
        max_new_tokens : int, optional
            The maximum number of new tokens (that are not in the initial prompt) to generate.
            
        Returns
        -------
        str
            The generated response from the model.
        """
        if (self.__model is None):
            print("The model was not configured. Launching with default parameters.")
            self.configure_model()
        
        #turn the prompt into a LLM understandable formated input
        inputs = self.__tokenizer(
            prompt, 
            return_tensors = "pt", 
            padding = attention_mask, 
            truncation = attention_mask,
            max_length = max_lenght_input,
        ).to(self.__device)
        
        inputs["input_ids"] = inputs["input_ids"].to(self.__device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.__device) if attention_mask else None
        
        #submit parameters to the model and generate the response
        with no_grad():
            outputs = self.__model.generate(
                inputs["input_ids"],
                num_return_sequences = num_return_sequences,
                repetition_penalty = rep_penalty,
                logits_processor = self.__logits_processor,
                attention_mask = inputs["attention_mask"] if attention_mask else None, 
                pad_token_id = self.__tokenizer.eos_token_id if attention_mask else None,
                max_new_tokens = max_new_tokens,
            )
        
        #turn the tokenized response in the tokenizer in a human readable string
        response = self.__tokenizer.decode(outputs[0], skip_special_tokens=True, max_length=max_lenght_output)
        
        return response
    
    def __del__(self) -> None:
        """
        Destructor for the LanguageModelHandler class.
        WARNING: this is a final operation and will erase this instance and CUDA's cache from the memory.
        """
        # for file in listdir(self.offload_dir):
        #     remove(f"{self.offload_dir}/{file}")
        
        cuda.empty_cache()
        
        del self.__model
        del self.__tokenizer
        del self.__logits_processor
        del self.__device
        del self.__auth_token
        del self
        collect()
        
    def __repr__(self) -> str:
        return f"LanguageModelHandler(model_name={self.model_name}, device={self.__device.type}, offload_dir={self.offload_dir})"
        

if __name__ == "__main__":
    def example_usage() -> None:
        """Example usage of the LanguageModelHandler class."""
        
        auth_token = dotenv_values(".env")["HF_AUTH_TOKEN"]
        prompt = "Répondez uniquement par le nom de la capitale : Quelle est la capitale de la France?"
        
        model_name = "mistralai/Mistral-7B-v0.1"
        handler = LanguageModelHandler(model_name, auth_token)
        handler.configure_model(use_gpu=True, local_offload=False)
        print_response(handler, model_name, prompt)
        handler.__del__()
        
        cuda.empty_cache()
        
        model_name = "mistralai/Mistral-7B-v0.1"
        handler = LanguageModelHandler(model_name, auth_token)
        handler.configure_model(use_gpu=True, local_offload=True)
        print_response(handler, model_name, prompt)
        handler.__del__()
        
    
    def print_response(handler: LanguageModelHandler, model_name: str, prompt: str) -> None:
        """Print a response from the model."""
        
        print(f"""\n
            Using the model {model_name} on {handler.get_device().type}:\n
            \n
            Prompt:\n {prompt}\n
            Response:\n {handler.generate_response(prompt=prompt, max_lenght_input=50, attention_mask=True)}\n
        """)
        
    
    example_usage()
    # print(cuda.is_available())
    # print(get_gpu_local_config())
    # print(get_cpu_local_config())
        
        