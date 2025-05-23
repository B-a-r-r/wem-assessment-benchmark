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
    
    Objects attributes
    ----------
    model_name : str
        The name of the model to load from HuggingFace.
        
    _auth_token : str
        The HuggingFace authentication token.
    
    _model : AutoModel
        The model resolved from model_name.
        
    _tokenizer : AutoTokenizer
        The tokenizer resolved from model_name.
    
    _device : torch.device
        The hardware that should handle the computation (CPU or GPU, default is GPU).
    
    _logits_processor : LogitsProcessorList
        A list of parameters that will be used to process the logits (partly determining/adapting the output).
        
    _log_event : callable
        A callable function to log events.
    """
    
    def __init__(self, model_name: str, auth_token: str =None, offload_dir: str ="./.llm-offloads", log_event: callable = None) -> None:
        """
        Initialize a handler for a given LLM.
        
        Parameters
        ----------
        model_name : str
            The name of the model to load from HuggingFace.
            
        auth_token : str, optional
            The HuggingFace authentication token. If not provided, the environment variable will try to be used.
            
        offload_dir : str, optional
            The directory where the model will be offloaded. Default is "./.llm-offloads".
            
        log_event : callable, optional
            A callable function to log events. If not provided, a no-op function will be used.
        """        
        self._log_event: callable = log_event if log_event is not None else lambda **kwargs: print("--- WARNING - No log function set for the LanguageModelHandler. ---")
        
        self._auth_token: str = None
        connexion_tryer = 0 #Avoid occasionnal crash at launch due to connection issues
        while self._auth_token is None:
            try:
                if auth_token is None:
                    #search for the environment variable file in the project 
                    try:
                        #if the app is run from a sub directory of the root directory
                        auth_token = dotenv_values(path.abspath("../.env"))["HF_AUTH_TOKEN"]
                    except KeyError:
                        try:
                            #if the app is run from the root directory
                            auth_token = dotenv_values(path.abspath("./.env"))["HF_AUTH_TOKEN"]
                        except KeyError:
                            try:
                                #try desperately to find the file using original name of the project
                                auth_token = dotenv_values(path.abspath("wem-assessment-benchmark/.env"))["HF_AUTH_TOKEN"]
                            except Exception as e:
                                self._log_event(
                                    event=f"Could not resolve the authentication token: {e}",
                                    color=""
                                )
                                raise e
                        
                login(auth_token)
                self._auth_token = auth_token
            
            except ConnectionError as e:
                self._log_event(event=f"Server error during authentication: {e}", source="LanguageModelHandler", indent="\t", type="WARNING")
                self._log_event(event="Retrying to login...", source="LanguageModelHandler", indent="\t")
                connexion_tryer += 1
                continue
            
            except Exception as e:
                self._log_event(event=f"Unexpected error during authentication: {e}", source="LanguageModelHandler", indent="\t", type="FATAL")
                raise e
            
            #avoid infinite loop
            if connexion_tryer > 3:
                self._log_event(event="Too many connection attempts. Exiting...", source="LanguageModelHandler", indent="\t", type="FATAL")
                raise ConnectionError("Too many connection attempts. Exiting...")
        
        self.model_name: str = model_name
        self.offload_dir: str = path.join(path.dirname(__file__),  offload_dir)
        self._model: AutoModelForCausalLM = None
        self._device: device = None
        self._logits_processor: LogitsProcessorList = None
        
        #cache directory for the model
        makedirs(path.join(path.dirname(__file__), ".llm-cache")) if not path.exists(path.join(path.dirname(__file__), ".llm-cache")) else None
    
        self._log_event(event=f"Resolving model auto tokenizer from pretrained...", source="LanguageModelHandler")
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name,
            force_download=True,
            trust_remote_code=True,
            use_fast=True,
            cache_dir=path.join(path.dirname(__file__), ".llm-cache")
        )
        self._tokenizer.pad_token = self._tokenizer.eos_token 
        
        self._log_event(event=f"LanguageModelHandler for {self.model_name} initialized. Tokenizer reolved.", source="LanguageModelHandler")
        
    @property
    def get_model(self) -> AutoModelForCausalLM:
        return self._model
    
    @property
    def get_tokenizer(self) -> AutoTokenizer:
        return self._tokenizer
    
    @property
    def get_device(self) -> device:
        return self._device
    
    @property
    def get_logits_processor(self) -> LogitsProcessorList:
        return self._logits_processor     
     
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
        #once the model is configured, it cannot be reconfigured
        if self._model is not None:
            self._log_event(event="Model is already configured and freezed. Ignoring this call.", source="LanguageModelHandler", indent="\t", type="WARNING")
            return
        
        #check if gpu is available before setting the device to gpu
        self._device = device("cpu" if (not use_gpu or not cuda.is_available()) else "cuda")
        self._log_event(event=f"Model will be loaded on {self._device.type}.", source="LanguageModelHandler")
        
        self._log_event(event=f"Resolving auto model from pretrained...", source="LanguageModelHandler")
        if ( #the local offload is applyed only if the wanted hardware is available
            (self._device.type == "cuda" and local_offload)
            or (not use_gpu and local_offload)
        ):
            with init_empty_weights():
                self._model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path = self.model_name,
                    pad_token_id=self._tokenizer.eos_token_id,
                    device_map = "auto",
                    load_in_8bit = quantization == 8,
                    load_in_4bit = quantization == 4,
                    torch_dtype = float32,
                )
            self._model = disk_offload(self._model, offload_dir=self.offload_dir)
            
            self._log_event(event=f"Model offloaded to {self.offload_dir}.", source="LanguageModelHandler", indent="\t")
            
        #if the wanted hardware is not available, or the local offload is not wanted, load the model normally
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path = self.model_name,
                load_in_8bit = quantization == 8,
                load_in_4bit = quantization == 4,
            )
                       
            self._log_event(event=f"Model not offloaded.", source="LanguageModelHandler", indent="\t")
            self._log_event(event=f"Model initialized.", source="LanguageModelHandler")
            
        self._logits_processor = LogitsProcessorList([
            TemperatureLogitsWarper(temperature),
        ])

        
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
            The input text to generate a response from.
            
        rep_penalty : float, optionals
            Each repetition of the same token is more or less penalized.
            This could encourage the model to diversify the terms in the output.
            
        min_new_tokens : int, optional
            This parameter may help to avoid empty responses and force the model to not just repeat the prompt.
            
        max_new_tokens : int, optional
            This parameter may help to avoid too long responses and force the model stay moer or less on the topic.
            
        batch_size : int, optional
            The number of samples to generate in parallel.
            
        Returns
        -------
        str
            The generated (raw) response from the model.
        """
        if self._model is None:
            self._log_event(
                event="The model was not configured before trying to generate a response. Launching with default parameters.",
                source="LanguageModelHandler",
                indent="\t",
                type="WARNING",
            )
            self.configure_model()

        generator = pipeline(
            task="text-generation",
            #task="text2text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            device=self._device,
            torch_dtype="auto",
            logits_processor=self._logits_processor,
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
            eos_token_id=self._tokenizer.eos_token_id,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        
        response = outputs[0]["generated_text"].replace(prompt, "") #remove the prompt if the model repeats it

        return response
    
    def clear(self) -> None:
        """
        Destructor for the LanguageModelHandler class.
        WARNING: this is a final operation and will erase this instance and CUDA's cache from the memory.
        """
        # for file in listdir(self.offload_dir):
        #     remove(f"{self.offload_dir}/{file}")
        
        cuda.empty_cache()
        
        del self._model
        del self._tokenizer
        del self._logits_processor
        del self._device
        del self
        collect()
    
    def __repr__(self) -> str:
        return f"LanguageModelHandler(model_name={self.model_name}, device={self._device.type}, offload_dir={self.offload_dir})"
        

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
        #model_name = "bartowski/gemma-2-9b-it-GGUF"
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
        