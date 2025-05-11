from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, TemperatureLogitsWarper, AutoConfig, pipeline, TextStreamer
from torch import cuda, device, float32, no_grad
from accelerate import disk_offload, init_empty_weights
from dotenv import dotenv_values
from gc import collect
from os import path, makedirs
from sys import exit
from requests.exceptions import ConnectionError

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
    
    __model : AutoModel
        The language model to load from HuggingFace.
        
    __tokenizer : AutoTokenizer
        The tokenizer for the language model, which is the default one for the given model.
    
    __device : torch.device
        The hardware that should handle the computation (CPU or GPU, default is GPU).
    
    __logits_processor : LogitsProcessorList
        A list of orchestration parameters that will be used to process the logits (partly determining/adapting the output).
        
    __log_event : callable
        A function to log events during the model's configuration and usage.
    """
    
    model_name: str
    offload_dir: str
    __auth_token: str
    __model: AutoModelForCausalLM
    __tokenizer: AutoTokenizer
    __device: device
    __logits_processor: LogitsProcessorList
    __log_event: callable
    
    def __init__(self, model_name: str, auth_token: str, offload_dir: str ="./.llm-offloads", log_event: callable = None) -> None:
        """
        Initialize a handler for a given LLM.
        
        Parameters
        ----------
        model_name : str
            The name of the model to load from HuggingFace.
            
        auth_token : str
            The HuggingFace authentication token.
        """
        self.__log_event = log_event if log_event is not None else lambda **kwargs: None
        
        connexion_tryer = 0 #Avoid occasionnal crash at launch due to connection issues
        while connexion_tryer < 3:
            try:
                login(auth_token)
                self.__auth_token = auth_token
                connexion_tryer = 3
            
            except ConnectionError as e:
                self.__log_event(event=f"Error during authentication: {e}", source="LanguageModelHandler", indent="\t", color="\033[91m")
                self.__log_event(event="Retrying...", source="LanguageModelHandler", indent="\t")
                connexion_tryer += 1
                continue
            
            except Exception as e:
                self.__log_event(event=f"Error during authentication: {e}", source="LanguageModelHandler", indent="\t", color="\033[91m")
                exit(1)
        
        self.model_name = model_name
        self.offload_dir = path.join(path.dirname(__file__),  offload_dir)
        self.__model = None
        self.__device = None
        self.__logits_processor = None
        
        makedirs(path.join(path.dirname(__file__), ".llm-cache")) if not path.exists(path.join(path.dirname(__file__), ".llm-cache")) else None
    
        self.__log_event(event=f"Resolving model auto tokenizer from pretrained...", source="LanguageModelHandler", indent="\t")
        self.__tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name,
            force_download=True,
            trust_remote_code=True,
            use_fast=True,
            cache_dir=path.join(path.dirname(__file__), ".llm-cache")
        )
        self.__tokenizer.pad_token = self.__tokenizer.eos_token 
        
   
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
        temperature: float =1.,
        quantization: int =None,
        local_offload: bool =False,
    ) -> None:
        """
        Configure the model with the given parameters.
        
        Parameters
        ----------
        use_gpu : bool, optional
            If True, the model will be loaded on the GPU (if available).
            
        temperature : float, optional
            Allows to give more or less importance to the low-probability tokens.
            This can encourage originality in the output, and therefore creativity.

        quantization : int, optional
            The quantization level to use for the model (4 or 8 bits).
            Allows to minimize the model size and speed up the inference time.
            
        local_offload : bool, optional
            If True, the model will be offloaded to the local machine's disk.
            This parameter will be ignored if the expected hardware is not available.
        """
        if self.__model is not None:
            self.__log_event(event="Model is already configured and freezed.", source="LanguageModelHandler", indent="\t", color="\033[91m")
            raise RuntimeError(f"Model is already configured and freezed.")
        
        #check if gpu is available before setting the device to gpu
        self.__device = device("cpu" if (not use_gpu or not cuda.is_available()) else "cuda")
        self.__log_event(event=f"Model will be loaded on {self.__device.type}.", source="LanguageModelHandler", indent="\t")
        
        self.__log_event(event=f"Resolving auto model from pretrained...", source="LanguageModelHandler", indent="\t")
        if ( #the local offload is applyed only if the wanted hardware is available
            (self.__device.type == "cuda" and local_offload)
            or (not use_gpu and local_offload)
        ):
            with init_empty_weights():
                self.__model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path = self.model_name,
                    pad_token_id=self.__tokenizer.eos_token_id,
                    device_map = "auto",
                    load_in_8bit = quantization == 8,
                    load_in_4bit = quantization == 4,
                    torch_dtype = float32,
                )
            self.__model = disk_offload(self.__model, offload_dir=self.offload_dir)
            
            self.__log_event(event=f"Model offloaded to {self.offload_dir}.", source="LanguageModelHandler", indent="\t")
            
        #if the wanted hardware is not available, or the local offload is not wanted, load the model normally
        else:
            # self.__model = AutoModelForCausalLM.from_config(self.__model_config)
            self.__model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path = self.model_name,
                #device_map = "auto",
                load_in_8bit = quantization == 8,
                load_in_4bit = quantization == 4,
            )
                       
            self.__log_event(event=f"Model not offloaded.", source="LanguageModelHandler", indent="\t")
            
        self.__logits_processor = LogitsProcessorList([
            TemperatureLogitsWarper(temperature),
        ])
        
        #self.__model.to_empty(device=self.__device)
        
        
    def generate_response(self, 
        prompt: str, 
        rep_penalty: float =1.,
        min_new_tokens: int =0,
        max_new_tokens: int =50,
        batch_size: int =None,
    ) -> str:
        """
        Generate a response from the model based on the given prompt.
        
        Parameters
        ----------
        prompt : str
            The input text to generate a response for.
            
        rep_penalty : float, optionals
            Each repetition of the same token is more or less penalized.
            This could encourage the model to diversify the terms in the output.
            
        attention_mask : bool, optional
            If True, the attention mask will be applied to the input text (padding + truncation).
            
        Returns
        -------
        str
            The generated response from the model.
        """
        if self.__model is None:
            self.__log_event(
                event="The model was not configured. Launching with default parameters.",
                source="LanguageModelHandler",
                indent="\t",
                color="\033[91m",
            )
            self.configure_model()

        generator = pipeline(
            task="text-generation",
            #task="text2text-generation",
            model=self.__model,
            tokenizer=self.__tokenizer,
            device=self.__device,
            torch_dtype="auto",
            logits_processor=self.__logits_processor,
        )
        
        outputs = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            truncation=True,
            repetition_penalty=rep_penalty,
            do_sample=True,
            batch_size=batch_size,
            num_return_sequences=1,
            eos_token_id=self.__tokenizer.eos_token_id,
            pad_token_id=self.__tokenizer.eos_token_id,
        )
        
        response = outputs[0]["generated_text"].replace(prompt, "")

        
        # # tokenize the prompt
        # inputs = self.__tokenizer(
        #     self.__tokenizer.tokenize(prompt),
        #     return_tensors="pt",
        #     padding=attention_mask,
        #     truncation=attention_mask,
        #     max_length=2048,
        # ).to(self.__device)

        # with no_grad():
        #     # generate response
        #     outputs = self.__model.generate(
        #         **inputs,
        #         repetition_penalty=rep_penalty,
        #         logits_processor=self.__logits_processor
        #     )

        # # decode response
        # response = self.__tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "")

        return response
    
    def clear(self) -> None:
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
        del self
        collect()
    
    def __repr__(self) -> str:
        return f"LanguageModelHandler(model_name={self.model_name}, device={self.__device.type}, offload_dir={self.offload_dir})"
        

if __name__ == "__main__":
    def example_usage():
        """Example usage of the LanguageModelHandler class."""
        
        try:
            auth_token = dotenv_values(path.abspath("../.env"))["HF_AUTH_TOKEN"]
        except KeyError:
            auth_token = dotenv_values(path.abspath("./.env"))["HF_AUTH_TOKEN"]
        
        def specify_format(n: int) -> str:
            res = ""
            for i in range(1, n+1):
                if i > 1:
                    res += ""
                res += f"name{i}"
                if i < n:
                    res += "/"
            return res 
        
        words_list_length = 5
        base_prompt = "USER: You are a versatile AI with deep language and comparative analysis skills, adept at generating diverse word and phrase lists and to take into account nuances between terms. You are obedient and follow the generation rules specified in the prompt, whithout any additional information and whithout introdction. #prompt#. \nASSISTANT:"
        #task_prompt = "Could you create a non-overlapping list of #N# distinct existing animal species? Please separate the names with slashes and present them in a single line without numbering. ## Format name1 / name2 / name3 / ..."
        #task_prompt = "Which one is stronger 'Koala' or 'Giraffe'? Answer only with 'Koala' xor 'Giraffe'."
        task_prompt = "Could you create a non-overlapping list of #N# existing animal species similar to 'Giraffe' but slightly or significantly different. Separate the names with slashes in one single line, without any list style, as the following format #Format#"
        prompt = base_prompt.replace("#prompt#", task_prompt)\
            .replace("#N#", str(words_list_length)).replace("#Format#", specify_format(words_list_length)).strip()
        
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        handler = LanguageModelHandler(model_name, auth_token)
        handler.configure_model(use_gpu=True, local_offload=False)
        response = handler.generate_response(
            prompt=prompt, 
            min_new_tokens=5, 
            max_new_tokens=25, 
            rep_penalty=1.,
            batch_size=None
        )
        print_response(handler, model_name, prompt, response, words_list_length)
        return format_response(response, words_list_length)
        
        handler.clear()
    
    def print_response(handler: LanguageModelHandler, model_name: str, prompt: str, response: str, n: int) -> None:
        """Print a response from the model."""
        
        print(f"\nUsing the model {model_name} on {handler.get_device().type}:\n-------------\n")
        print(f"Prompt:\n-------------\n{prompt}\n")
        print(f"Response:\n-------------\n{response}\n")
        print(f"Formatted response:\n-------------\n{format_response(response, n)}\n")
        
    def format_response(response: str, n: int) -> str:
        res = response.lower().strip().split("\n")
        
        #avoid empty lines
        tmp = 0
        while len(res[tmp]) == 0:
            tmp += 1
        res = res[tmp]
        
        res = res.replace(".","").replace("*"," ").replace("_"," ").strip()
        while res.startswith(("-", "+", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
            res = res[1:]
        
        res = res.split("/")[:n]
        for i in range(0, len(res)):
            res[i] = res[i].strip()
        return res
    
    example_usage()
    # print(cuda.is_available())
    # print(get_gpu_local_config())
    # print(get_cpu_local_config())
    
    # responses = []
    # for i in range(10):
    #     responses.append(example_usage())
    # for response in responses:
    #     print(response)
        