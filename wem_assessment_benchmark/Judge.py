from random import choice as rnd_choice
import json
from gc import collect
from LanguageModelHandler import LanguageModelHandler
from sys import exit
import re
from json import dump
from json import loads as encode_dict

#
# author:       Reiji SUZUKI et al.
# refactor:     Clément BARRIERE
#


class Judge:
    """
    This class is a judging system powered by a LLM model.
    It is used to compare words according to a given criteria.
    
    Objects attributes     
    ----------
    _model : LanguageModelHandler
        The llm model to be used for judging.
        
    judgments_history : dict
        A dictionary to store the results of the matches.
        
    criteria : str
        The judging criteria.
        
    config : dict
        A dictionary to store some config parameters.
        See config.json for more details on the format.
        
    log_event : callable
        A callable function to log events.
    """
    
    def __init__(self,
        config: dict,
        log_event: callable =None,
    ):
        """
        Initializes a Judge object.

        Parameters
        ----------
        criteria : str
            The judging criteria.
            
        config: dict
            A dictionary containing the configuration parameters.
            
        log_event : callable, optional
            A callable function to log events (default is None).
        """
        self._log_event: callable = log_event if log_event is not None else lambda x: print("--- WARNING - No log function set for the Judge. ---")
        self.judgments_history: dict = {}
        
        self.config: dict = config
        self.verify_config()
        
        self.criteria: str = self.config["simulation"]["CRITERIA"]
        
        self._model: LanguageModelHandler = LanguageModelHandler(
            model=self.config["model"]["name"],
            log_event=self._log_event,
        )
        
        #load the model with the given config
        self._model.configure_model(
            local_offload= self.config["model"]["local_offload"],
            quantization= self.config["model"]["quantization"], 
            temperature= self.config["model"]["temperature"],
            use_gpu= self.config["model"]["use_gpu"],
        )
        
        self._log_event(event=f"Judge '{self.criteria}' loaded.")
    
    def verify_config(self):
        """
        Verify the config file and check if all the required keys are present.
        May proceed to some adjustments according to the inputed config.
        """
        self._log_event(f"Chcking requirements from config...", "Judge", "\t")
        
        try:
            assert self.config["model"]["name"] is not None, "Missing model name in the 'model' section of the config file."
            assert self.config["model"]["temperature"] is not None, "Missing temperature parameter in the 'model' section of the config file."
            
            assert self.config["prompts"]["judge"] is not None, "The judge prompt template must be set in the 'prompts' section of the config file."
            assert self.config["prompts"]["judge"].find("#criteria#") != -1, "The judge prompt template must contain #criteria#."
            assert self.config["prompts"]["judge"].find("#word1#") != -1, "The judge prompt template must contain #word1#."
            assert self.config["prompts"]["judge"].find("#word2#") != -1, "The judge prompt template must contain #word2#."
            assert self.config["prompts"]["judge"].find("#format#") != -1, "The judge prompt template must contain #format#."
            
            assert self.config["prompts"]["prefix"] is not None, "The prefix prompt template must be set in the 'prompts' section of the config file."
            assert self.config["prompts"]["prefix"].find("#prompt#") != -1, "The prefix prompt template must contain #prompt#."
            
            assert self.config["prompts"]["create"] is not None, "The create prompt template must be set in the 'prompts' section of the config file."
            assert self.config["prompts"]["create"].find("#A#") != -1, "The create prompt template must contain #A#."
            assert self.config["prompts"]["create"].find("#format#") != -1, "The create prompt template must contain #format#."
            
            assert self.config["prompts"]["mutate"] is not None, "The mutate prompt template must be set in the 'prompts' section of the config file."
            assert self.config["prompts"]["mutate"].find("#word#") != -1, "The mutate prompt template must contain #word#."
            assert self.config["prompts"]["mutate"].find("#B#") != -1, "The mutate prompt template must contain #B#."
            assert self.config["prompts"]["mutate"].find("#format#") != -1, "The mutate prompt template must contain #format#."
            
            assert self.config["simulation"]["CRITERIA"] is not None, "The criteria must be set in the 'simulation' section of the config file."
            assert self.config["simulation"]["N"] is not None, "The N parameter must be set in the 'simulation' section of the config file."
            assert self.config["simulation"]["B"] is not None, "The B parameter must be set in the 'simulation' section of the config file."
            assert self.config["simulation"]["WORD_MIN_LENGHT"] is not None, "The WORD_MIN_LENGHT parameter must be set in the 'simulation' section of the config file."
            assert self.config["simulation"]["WORD_MAX_LENGHT"] is not None, "The WORD_MAX_LENGHT parameter must be set in the 'simulation' section of the config file."
            
            assert self.config["workspace"]["exp_dir"] is not None, "The EXP_DIR parameter must be set in the 'workspace' section of the config file."

            if self.config["model"]["quantization"] is None:
                self.config["model"]["quantization"] = -1
                
            if self.config["model"]["local_offload"] is None:
                self.config["model"]["local_offload"] = False
                
            if self.config["model"]["use_gpu"] is None:
                self.config["model"]["use_gpu"] = False
            
        except Exception as e:
            self._log_event(f"\nFailed to log config: {e}. Please check the config file.", source="Judge", indent="\t", type="FATAL")
            raise e
    
    def judge(self, word1: str, word2: str) -> str:
        """
        The model is used to compare two words and return the best according to the given criteria.

        Parameters
        ----------
        word1 : str
            The first word to compare.
            
        word2 : str
            The second word to compare.

        Returns
        ---------
        str
            The word chosen by the model.
        """
        #check if the words were already judged, if so return the cause law
        if (word1, word2) in self.judgments_history:
            return self.judgments_history[(word1, word2)]
        
        #adapt the prompt template to the criteria and given words
        prompt = self.config["prompts"]["prefix"][:].replace(
            "#prompt#", 
            self.config["prompts"]["judge"][:]
        ).replace("#criteria#", self.criteria).replace("#word1#", word1).replace("#word2#", word2)
    
        chosen_one= self.request_intervention(prompt, 1, distinct=True)
        
        #store the result in the dictionary
        if ((word1==chosen_one) or (word2==chosen_one)):
            self.judgments_history[(word1, word2)] = chosen_one
            self.judgments_history[(word2, word1)] = chosen_one
            return chosen_one
        
        else:
            return "--Nomatch--"
    
    def create_word_list(self, verbose: bool =False) -> list:
        """ 
        Create a list of n words using the model.
        
        Parameters
        ----------
        verbose : bool, optional
            If True, print the words created (default is False).
            
        Returns
        -------
        list
            A list of n words created by the model.
        """
        n = self.config["simulation"]["N"]
        prompt = self.config["prompts"]["prefix"][:].replace(
            "#prompt#", 
            self.config["prompts"]["create"][:]
        ).replace("#A#", str(n)).replace("#format#", self.forge_response_format(n))
             
        words = []
        while len(words) < n:
            self._log_event(f"Creating {n} words...", "Judge", "\t")
            
            words= [a for a in self.request_intervention(prompt, n, distinct=True) if (
                #minimie the risk to introduce phrases or sentences
                len(a)<=self.config["simulation"]["WORD_MAX_LENGHT"] 
                and len(a)>=self.config["simulation"]["WORD_MIN_LENGHT"]
            )]

            self._log_event(f"Created words: {words}", "Judge", "\t")
            if len(words) < n: self._log_event(f"Not enough words created, trying again...", "Judge", "\t", type="WARNING")
            
        if verbose : print(f"Words created: {words}")
        return list(words)

    def mutate(self, 
        word: str,
        verbose: bool =False
    ) -> str:
        """
        Mutate a word using the model.
        
        Parameters
        ----------
        word : str
            The word to mutate.
            
        verbose : bool, optional
            If True, print the mutated word (default is False).
            
        Returns
        -------
        str
            The mutated word.
        """
        n = self.config["simulation"]["B"]
        
        prompt = self.config["prompts"]["prefix"][:].replace(
            "#prompt#", 
            self.config["prompts"]["mutate"][:]
        ).replace("#B#", str(n)).replace("#word#", word).replace("#format#", self.forge_response_format(n))
        
        words= []
        while len(words) < n:
            words= [a for a in self.request_intervention(prompt, n) if (
                #minimie the risk to introduce phrases or sentences
                len(a)<=self.config["simulation"]["WORD_MAX_LENGHT"] 
                and len(a)>=self.config["simulation"]["WORD_MIN_LENGHT"]
            )]
            
            self._log_event(f"Mutation list: {words}", "Judge", "\t")
            if len(words) < n: self._log_event(f"Not enough words created, trying again...", "Judge", "\t", type="WARNING")
        
        chosen_word= rnd_choice(words)
        
        if verbose : print(f"{word}->{chosen_word}, {words}")
        return(chosen_word)
    
    def request_intervention(self,
        prompt: str,
        n: int,
        distinct: bool =False,
        verbose: bool =False
    ) -> str:
        """ 
        Request an intervention from the model.
        
        Parameters
        ----------
        prompt : str
            The prompt to send to the model.
            
        n : int
            The number of words to generate.
            
        distinct : bool, optional
            If True, the formatted response will be cleared of duplicates (default is False).
            
        verbose : bool, optional
            If True, print the raw and formatted response (default is False).
            
        Returns
        -------
        str
            The formatted version of theresponse from the model.
        """
        self._log_event(f"Requesting intervention...", "Judge", "\t")
        
        response = self._model.generate_response(
            prompt, 
            min_new_tokens=n, 
            max_new_tokens=self.config["model"]["max_tokens_per_word"]*n,
            rep_penalty=self.config["model"]["rep_penalty"],
        )
        
        f_response = Judge.format_response(response, n, distinct)
        
        if verbose:
            print(f"Raw response: {response}")
            print(f"Formatted response: {f_response}")
        
        return f_response
    
    @staticmethod
    def format_response(response: str, n: int, distinct: bool) -> list:
        """ 
        Format the response from the model.
        
        Parameters
        ----------
        response : str
            The raw response from the model.
            
        n : int
            The number of words expected.
            
        distinct : bool
            If True, the formatted response will be cleared of duplicates.
            
        Returns
        -------
        list
            The formatted response as a list of words.
        """
        #avoid useless special characters in the response
        res = response.lower().replace(".","").replace("*"," ").replace("_"," ") \
            .replace("-"," ").replace("+"," ").replace("#","").replace("!","") \
            .replace("~","").replace("@","").replace("$","").replace("|","") \
            .replace("%","").replace("^","").replace("&","").replace("(","") \
            .replace(")","").replace("[","").replace("]","").replace("{","") \
            .replace("}","").replace(";","").replace(":","").replace(",","") \
            .replace("?","").replace("<","").replace(">","").replace("\n","") \
        
        #avoid empty lines
        res = re.sub(r'^\s+$', '', res, flags=re.MULTILINE)
        
        #avoid certain characters at the beginning of the response, such as numbers
        res = re.sub(r'^\d+', '', res) 
        
        #convert to list
        res = res.split("/")
        
        #strip each word individually
        for i in range(0, len(res)):
            res[i] = res[i].strip()
        
        #remove duplicates if needed
        if distinct:
            res = list(set(res))
        
        #only keep the right number of words
        return res[:n]
    
    @staticmethod
    def forge_response_format(n: int, choices: list =None) -> str:
        """
        Forge the response format to add to a prompt.
        
        Parameters
        ----------
        n : int
            The number of words expected.
            
        choices : list, optional
            A list of specific choices to specify in the format (default is None).
            
        Returns
        -------
        str
            The response format as a string.
        """
        res = ""
        #if only one word is expeced, as when judging
        if n == 1:
            res += f"Answer only with "
            #if specific choices are given
            if choices is not None:
                for i in len(choices):
                    res += f"{choices[i]}"
                    if i < len(choices)-1:
                        res += " or "
            else:
                res += "one word"
        #if multiple words are expected, as when creating or mutating
        else:
            for i in range(1, n+1):
                if i > 1:
                    res += ""
                res += f"name{i}"
                if i < n:
                    res += "/"
        return res 
    
    def clear(self):
        """
        Destructor for the Judge class.
        """
        self._model.clear()
        del self._model
        del self.judgments_history
        del self.config
        del self
        
        collect()
        
    def log_case_law(self, log_label: str =""):
        """
        Log the judgments history to a file.
        
        Parameters
        ----------
        log_label : str, optional
            The label to add to the file name (default is "").
        """
        with open(f"{self.config["workspace"]["exp_dir"]}/judgments_history_{log_label}.json", "w", encoding="utf-8") as tmp:
            dump(encode_dict(self.judgments_history), tmp, indent=2)
        
        self._log_event(f"Judgments history saved in {tmp.name}", "Judge")
        
    def __repr__(self):
        return f"Judge({self.criteria}) using {self._model.model_name}."