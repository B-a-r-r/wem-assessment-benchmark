from random import choice as rnd_choice
from gc import collect
from LanguageModelHandler import LanguageModelHandler
from os import path
import re
from json import dump
from io import TextIOWrapper
from WemActor import WemActor

#
# author:       Reiji SUZUKI et al.
# refactor:     Clément BARRIERE
#

class Judge(WemActor):
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
    _jugements_history_logs : TextIOWrapper | None
        A file to log the judgments history.
    """
    
    def __init__(self,
        config: dict,
        enable_logs: bool =True,
    ):
        """
        Initializes a Judge object.

        Parameters
        ----------
        config: dict
            A dictionary containing the configuration parameters.
        log_event : callable, optional
            A callable function to log events (default is None).
        """
        print("--- WARNING - No log function set for the Judge. ---") if not enable_logs else None
        self.logs_path = path.join(path.abspath(self.config["workspace"]["exp_dir"]), "logs.txt") if enable_logs else None
        self._logs_file: TextIOWrapper = None
        
        self.config: dict = config
        self.verify_config()
        
        self.judgments_history: dict[int, dict[tuple[str, str], str]] = {}
        self._jugements_history_logs: TextIOWrapper | None = None
        self.criteria: str = self.config["simulation"]["CRITERIA"]
        
        self._model: LanguageModelHandler = LanguageModelHandler(
            model_name=self.config["model"]["name"],
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
    
    def _log_event(self, 
        event: str, 
        source: str =None, 
        indent: str ="", 
        underline: bool =False,
        type: str =None,
    ) -> None:
        super()._log_event(
            event=event, source=source, indent=indent, underline=underline, type=type, 
            logs_path=self.config["workspace"]["exp_dir"], logs_file=self._logs_file
        )
    
    def verify_config(self):
        """
        Verify the config file and check if all the required keys are present.
        May proceed to some adjustments according to the inputed config.
        """
        try:
            assert self.config["model"]["name"] is not None, "Missing model name in the 'model' section of the config file."
            assert self.config["model"]["temperature"] is not None, "Missing temperature parameter in the 'model' section of the config file."
            
            assert self.config["prompts"]["judge"] is not None, "The judge prompt template must be set in the 'prompts' section of the config file."
            assert self.config["prompts"]["judge"].find("#criteria#") != -1, "The judge prompt template must contain #criteria#."
            assert self.config["prompts"]["judge"].find("#word1#") != -1, "The judge prompt template must contain #word1#."
            assert self.config["prompts"]["judge"].find("#word2#") != -1, "The judge prompt template must contain #word2#."
            assert self.config["prompts"]["judge"].find("#format#") == -1, "Formatting not suppoted yet for the judge prompt template. Please remove #format# and hardcode this format in the prompt."
            
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

            try:
                self.config["model"]["quantization"]
            except KeyError:
                self.config["model"]["quantization"] = -1
                
            try: 
                self.config["model"]["local_offload"]
            except KeyError:
                self.config["model"]["local_offload"] = False
                
            try:
                self.config["model"]["use_gpu"]
            except KeyError:
                self.config["model"]["use_gpu"] = False
                
            try:
                self.config["simulation"]["A_ALLOWED_DELTA"]
            except KeyError:
                self.config["simulation"]["A_ALLOWED_DELTA"] = 0
                
            try:
                self.config["simulation"]["B_ALLOWED_DELTA"]
            except KeyError:
                self.config["simulation"]["B_ALLOWED_DELTA"] = 0
            
        except Exception as e:
            print(f"FATAL - from Judge - Failed to load config: {e}. Please check the config file.")
            raise e
    
    def judge(self, word1: str, word2: str, gen: int) -> str:
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
        for res in self.judgments_history.values():
            if res.__contains__((word1, word2)):
                return res[(word1, word2)]
        
        #adapt the prompt template to the criteria and given words
        prompt = self.config["prompts"]["prefix"].replace(
            "#prompt#", 
            self.config["prompts"]["judge"]
        ).replace("#criteria#", self.criteria).replace("#word1#", word1).replace("#word2#", word2)
    
        chosen_one = self._request_intervention(prompt, 1, distinct=True)[0]
        
        if ((chosen_one.__contains__(word1)) or (chosen_one.__contains__(word2))):
            chosen_one = word1 if chosen_one.__contains__(word1) else word2
            #store the result in the history dictionary
            if not self.judgments_history.keys().__contains__(gen):
                self.judgments_history[gen] = {}
            self.judgments_history[gen][(word1, word2)] = chosen_one
            self.judgments_history[gen][(word2, word1)] = chosen_one
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
        n = self.config["simulation"]["A"]
        
        prompt = self.config["prompts"]["prefix"].replace(
            "#prompt#", 
            self.config["prompts"]["create"]
        ).replace("#A#", str(n)).replace("#format#", self.forge_response_format(n))
             
        words = []
        while len(words) < n - self.config["simulation"]["A_ALLOWED_DELTA"]:
            self._log_event(f"Creating {n} words...", "Judge", "\t")
            
            words= [a for a in self._request_intervention(prompt, n, distinct=True) if (
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
    ) -> tuple[str, list[str]]:
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
        tupl(str, list[str])
            A tuple containing the mutated word and the list of possible mutations.
        """
        n = self.config["simulation"]["B"]
        
        prompt = self.config["prompts"]["prefix"].replace(
            "#prompt#", 
            self.config["prompts"]["mutate"]
        ).replace("#B#", str(n)).replace("#word#", word).replace("#format#", self.forge_response_format(n))
        
        words= []
        while len(words) < n - self.config["simulation"]["B_ALLOWED_DELTA"]:
            words= [a for a in self._request_intervention(prompt, n) if (
                #minimie the risk to introduce phrases or sentences
                len(a)<=self.config["simulation"]["WORD_MAX_LENGHT"] 
                and len(a)>=self.config["simulation"]["WORD_MIN_LENGHT"]
            )]
            
            self._log_event(f"Mutation list: {words}", "Judge", "\t")
            if len(words) < n: self._log_event(f"Not enough words created, trying again...", "Judge", "\t", type="WARNING")
        
        chosen_word= rnd_choice(words)
        
        if verbose : print(f"{word} -> {chosen_word}, {words}")
        return (chosen_word, words)
    
    def _request_intervention(self,
        prompt: str,
        n: int,
        distinct: bool =False,
        verbose: bool =False
    ) -> list[str]:
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
        list[str]
            The formatted version of the response from the model.
        """
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
            .replace("?","").replace("<","").replace(">","").replace("\\","")
        
        #avoid empty lines
        res = re.sub(r'^\s+$', '', res, flags=re.MULTILINE)
        
        #avoid certain characters at the beginning of the response, such as numbers
        res = re.sub(r'^\d+', '', res) 
        
        #convert to list
        res = res.split("\n")[0]
        
        if n > 1:
            res = res.split("/")
        else:
            res = res.split(" ")
        
        #remove empty strings
        res = [a for a in res if a != ""]
        
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
        """
        Forge the response format to add to a prompt.
        
        Parameters
        ----------
        n : int
            The number of words expected.
            
        Returns
        -------
        str
            The response format as a string.
        """
        assert n > 1, "One element format specifying is not supported yet."
        
        res = ""
        for i in range(1, n+1):
            if i > 1:
                res += ""
            res += f"name{i}"
            if i < n:
                res += "/"
                
        return res 
    
    def _clear(self):
        """
        Destructor for the Judge class.
        """
        if self._jugements_history_logs is not None:
            self._jugements_history_logs.close()
            self._jugements_history_logs = None
        self._model._clear()
        del self._model
        del self.judgments_history
        del self.config
        del self
        
        collect()
        
    def log_case_law(self, t: int):
        """
        Log the judgments history to a file.
        
        Parameters
        ----------
        t : int
            The current generation number.
        """
        if self._jugements_history_logs is None:
            self._jugements_history_logs = open(f"{path.abspath(self.config['workspace']['exp_dir'])}/judgments_history_{t}.json", "w", encoding="utf-8")
            self._log_event(f"Judgments history log file created.", "Judge")
        
        self._jugements_history_logs.seek(0)
        self._jugements_history_logs.truncate()
        dump({
            gen: {
                str(k): v for k, v in cases.items()
            } for gen, cases in self.judgments_history.items()
        }, self._jugements_history_logs, indent=2)
               
        self._log_event(f"Judgments history logs updated.", "Judge")
        
    def __repr__(self) -> str:
        return f"Judge({self.criteria}) using {self._model.model_name}."