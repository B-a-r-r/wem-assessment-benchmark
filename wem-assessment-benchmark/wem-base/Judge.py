from random import choice as rnd_choice
import json
from ..LanguageModelHandler import LanguageModelHandler
from gc import collect


#
# author:       Reiji SUZUKI et al.
# refactor:     Clément BARRIERE
#


class Judge:
    """
    This class is a judging system powered by a LLM model.
    It is used to compare words according to a given criteria.
    
    Attributes      collect()
    ----------
    __model : LanguageModelHandler
        The llm model to be used for judging.
        
    judgments_history : dict
        A dictionary to store the results of the matches.
        
    criteria : str
        The judging criteria.
        
    config : dict
        A dictionary to store some config parameters.
        See config.json for more details on the format.
    """

    __model: LanguageModelHandler
    judgments_history: dict
    criteria: str
    config: dict
    
    def __init__(self, model: LanguageModelHandler, criteria: str, config_path: str = "config.json"):
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
        """
        
        self.judgments_history = {}
        self.criteria = criteria
        self.config = self.__load_config(config_path=config_path)
        self.__model = model
        
        #load the model with the given config
        try:
            self.__model.configure_model(
                local_offload= self.config["model"]["local_offload"],
                quantization= self.config["model"]["quantization"], 
                temperature= self.config["model"]["temperature"],
                top_p= self.config["model"]["top_p"],
                use_gpu= self.config["model"]["use_gpu"],
            )
        except KeyError as e:
            raise KeyError(f"Missing key in config: {e}. Please check the config file.") from e
        
        
    def __load_config(self, config_path: str):
        """
        Load the config parameters from the given file path.

        Parameters
        ----------
        config_path : str
            The path to the config JSON file.
        """
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        
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
        
        except KeyError as e:
            raise KeyError(f"Missing key in config: {e}. Please check the config file.") from e
        
        #adapt the prompt template to the criteria and given words
        prompt = self.config["prompts"]["judge"][:] \
            .replace("#criteria#", self.criteria) \
            .replace("#word1#", word1) \
            .replace("#word2#", word2)
        
        #ask the model
        response= self.__model.generate_response(
            prompt, 
            max_lenght_output= int(self.config["model"]["max_lenght_output"])
        )
        #response= random.choice([word1, word2])
        
        #picking up the answer from the model
        ans= response.lower().split("\n")[0].strip().replace("/","")

        #store the result in the dictionary
        if ((word1==ans) or (word2==ans)):
            self.judgments_history[(word1, word2)] = ans
            self.judgments_history[(word2, word1)] = ans
            return ans
        
        else:
            return "--Nomatch--"
    
    def create_word_list(self, 
        n: int, 
        word_min_lenght: int,
        word_max_lenght: int, 
        verbose: bool =False
    ) -> list:
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

        except KeyError as e:
            raise KeyError(f"Missing key in config: {e}. Please check the config file.") from e
        
        prompt = self.config["prompts"]["create"][:].replace("#N#", str(n))
        words = []
        
        while len(words)<4:
            response = self.__model.generate_response(
                prompt, 
                max_tokens= int(self.config["model"]["max_lenght_output"])*n
            ).lower()
            
            words= [a.strip() for a in response.replace("\n"," ").split("/")]
            words= [a for a in words if (len(a)<=word_max_lenght and len(a)>=word_min_lenght)]
            words= words[:n] if len(words) > n else words
        
        if verbose : print(f"Words created: {words}")
        return words

    def mutate(self, 
        word: str, 
        word_min_lenght: int, 
        word_max_lenght: int, 
        verbose: bool =False
    ) -> str:
        """
        Mutate a word using the model.
        
        Parameters
        ----------
        word : str
            The word to mutate.
            
        word_min_lenght : int
            The minimum length of the mutated word.
            
        word_max_lenght : int
            The maximum length of the mutated word.
        """
        
        try:
            #check if the prompt template is well formed
            assert self.config["prompts"]["mutate"].find("#word#") != -1, "The prompt template for mutating words must contain #word#."
            assert self.config["model"]["max_lenght_output"] is not None, "The max_lenght_output parameter must be set in the config file."
        
        except KeyError as e:
            raise KeyError(f"Missing key in config: {e}. Please check the config file.") from e
        
        prompt = self.config["prompts"]["mutate"][:].replace("#word#", word)
        
        words= []
        while words == []:
            response = self.__model.generate_response(
                prompt,
                max_tokens= int(self.config["model"]["max_lenght_output"])*2
            ).lower()
            
            words= [a.strip() for a in response.replace("\n"," ").replace("*"," ").replace("_"," ").split("/")]
            words= [a for a in words if (len(a)<=word_max_lenght and len(a)>=word_min_lenght)]
        
        chosen_word= rnd_choice(words)
        
        if verbose : print(f"{word}->{chosen_word}, {words}")
        return(chosen_word)
    
    def __del__(self):
        """
        Destructor for the Judge class.
        """
        self.__model.__del__()
        del self.__model
        del self.judgments_history
        del self.config
        del self
        
        collect()
        