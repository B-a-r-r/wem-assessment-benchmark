from random import choice as rnd_choice
import json
from gc import collect
from LanguageModelHandler import LanguageModelHandler
from sys import exit
import re

#
# author:       Reiji SUZUKI et al.
# refactor:     Clément BARRIERE
#


class Judge:
    """
    This class is a judging system powered by a LLM model.
    It is used to compare words according to a given criteria.
    
    Attributes     
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
    
    _model: LanguageModelHandler
    judgments_history: dict
    criteria: str
    config: dict
    _log_event: callable
    expected_max_tokens_per_word: int
    
    
    def __init__(self,
        model: LanguageModelHandler, 
        criteria: str, 
        config_path: str ="config.json",
        log_event: callable =None,
        expected_max_tokens_per_word: int =5
    ):
        """
        Initializes a Judge object.

        Parameters
        ----------
        model : LanguageModelHandler
            The llm model to be used for judging.
            
        criteria : str
            The judging criteria.
            
        config_path : str, optional
            The path to the config JSON file (default is "config.json").
            
        log_event : callable, optional
            A callable function to log events (default is None).
        """
        
        self.judgments_history = {}
        self.criteria = criteria
        self.expected_max_tokens_per_word = expected_max_tokens_per_word
        self._log_event = log_event if log_event is not None else lambda x: None
        
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        self._model = model
        
        #load the model with the given config
        try:
            self._model.configure_model(
                local_offload= self.config["model"]["local_offload"],
                quantization= self.config["model"]["quantization"], 
                temperature= self.config["model"]["temperature"],
                use_gpu= self.config["model"]["use_gpu"],
            )
            
        except KeyError as e:
            self._log_event(f"\nMissing key in config: {e}. Please check the config file.", source="Judge", indent="\t", color="\033[91m")
            exit(1)
        
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
        ##check if the words were already judged
        if (word1, word2) in self.judgments_history:
            return self.judgments_history[(word1, word2)]
        
        try:
            #check if the prompt template is well formed
            assert self.config["prompts"]["judge"].find("#criteria#") != -1, "The prompt template for judging must contain #criteria#."
            assert self.config["prompts"]["judge"].find("#word1#") != -1, "The prompt template for judging must contain #word1#."
            assert self.config["prompts"]["judge"].find("#word2#") != -1, "The prompt template for judging must contain #word2#."
            assert self.config["model"]["max_lenght_output"] is not None, "The max_lenght_output parameter must be set in the config file."
            assert self.config["prompts"]["prefix"] is not None and self.config["prompts"]["prefix"].find("#prompt#") != -1, "The prefix prompt is set but must contain #prompt#."
        
        except Exception as e:
            self._log_event(f"\nFailed to log config: {e}. Please check the config file.", source="Judge", indent="\t", color="\033[91m")
            exit(1)
        
        #adapt the prompt template to the criteria and given words
        prompt = self.config["prompts"]["prefix"][:].replace(
            "#prompt#", 
            self.config["prompts"]["judge"][:]
        ).replace("#criteria#", self.config["simulation"]["CRITERIA"]).replace("#word1#", word1).replace("#word2#", word2)
    
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
        n : int
            The number of words to create.
            
        word_min_lenght : int
            The minimum length of the words.
            
        word_max_lenght : int
            The maximum length of the words.
        """
        try:
            #check if the prompt template is well formed
            assert self.config["prompts"]["create"].find("#N#") != -1, "The prompt template for creating words must contain #N#."
            assert self.config["model"]["max_lenght_output"] is not None, "The max_lenght_output parameter must be set in the config file."
            assert self.config["prompts"]["prefix"] is not None and self.config["prompts"]["prefix"].find("#prompt#") != -1, "The prefix prompt is set but must contain #prompt#."

        except Exception as e:
            self._log_event(f"\nFailed to log config: {e}. Please check the config file.", source="Judge", indent="\t", color="\033[91m")
            exit(1)
        
        n = self.config["simulation"]["N"]
        prompt = self.config["prompts"]["prefix"][:].replace(
            "#prompt#", 
            self.config["prompts"]["create"][:]
        ).replace("#N#", str(n)).replace("#format#", self.forge_response_format(n))
             
        words = []
        while len(words) < n:
            self._log_event(f"Creating {n} words...", "Judge", "\t")
            
            words= [a for a in self.request_intervention(prompt, n, distinct=True) if (
                #minimie the risk to introduce phrases or sentences
                len(a)<=self.config["simulation"]["WORD_MAX_LENGHT"] 
                and len(a)>=self.config["simulation"]["WORD_MIN_LENGHT"]
            )]

            self._log_event(f"Created words: {words}", "Judge", "\t")
            if len(words) < n: self._log_event(f"Not enough words created, trying again...", "Judge", "\t")
            
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
            
        word_min_lenght : int
            The minimum length of the mutated words.
            
        word_max_lenght : int
            The maximum length of the mutated words.
            
        n : int, optional
            The number of words to create, among which the mutation will be chosen.
        """
        self._log_event(f"Chcking requirements from config...", "Judge", "\t")
        try:
            #check if the prompt template is well formed
            assert self.config["prompts"]["mutate"].find("#word#") != -1, "The prompt template for mutating words must contain #word#."
            assert self.config["model"]["max_lenght_output"] is not None, "The max_lenght_output parameter must be set in the config file."
            assert self.config["prompts"]["prefix"] is not None and self.config["prompts"]["prefix"].find("#prompt#") != -1, "The prefix prompt is set but must contain #prompt#."
        
        except Exception as e:
            self._log_event(f"\nFailed to log config in mutation process: {e}. Please check the config file.", source="Judge", indent="\t", color="\033[91m")
            exit(1)
        
        
        n = self.config["simulation"]["B"]
        
        prompt = self.config["prompts"]["prefix"][:].replace(
            "#prompt#", 
            self.config["prompts"]["mutate"][:]
        ).replace("#N#", str(n)).replace("#word#", word).replace("#format#", self.forge_response_format(n))
        
        words= []
        while len(words) < n:
            words= [a for a in self.request_intervention(prompt, n) if (
                #minimie the risk to introduce phrases or sentences
                len(a)<=self.config["simulation"]["WORD_MAX_LENGHT"] 
                and len(a)>=self.config["simulation"]["WORD_MIN_LENGHT"]
            )]
            
            self._log_event(f"Mutation list: {words}", "Judge", "\t")
            if len(words) < n: self._log_event(f"Not enough words created, trying again...", "Judge", "\t")
        
        chosen_word= rnd_choice(words)
        
        if verbose : print(f"{word}->{chosen_word}, {words}")
        return(chosen_word)
    
    def request_intervention(self,
        prompt: str,
        n: int,
        distinct: bool =False
    ) -> str:
        response = self._model.generate_response(
            prompt, 
            min_new_tokens=n, 
            max_new_tokens=self.expected_max_tokens_per_word*n,
            rep_penalty=self.config["model"]["rep_penalty"],
        )
        # print(f"Raw response: {response}")
        # print(f"Formatted response: {Judge.format_response(response, n, distinct)}")
        
        return Judge.format_response(response, n, distinct)
    
    @staticmethod
    def format_response(response: str, n: int, distinct: bool) -> list:
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
    def forge_response_format(n: int) -> str:
        res = ""
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
        