from abc import ABC, abstractmethod
from datetime import datetime
from os import path
from io import TextIOWrapper

class WemActor(ABC):
    """
    An abstract class as an interface for actors in the WEM application.
    """
    def __init__(self, logs_path: str =None):
        self.logs_path: str = logs_path
        self._logs_file: TextIOWrapper = None
    
    @abstractmethod
    def verify_config(self):
        """
        Each actor must ensure that the config file contains all necessary parameters.
        """
        pass
    
    def _log_event(self, 
        event: str, 
        source: str =None, 
        indent: str ="", 
        underline: bool =False,
        type: str =None,
    ) -> None:
        """
        Logs an event to the log file with some 
        additional information.

        Parameters
        ----------
        event : str
            The event to log. 
        source : str, optional
            The source file of the event.
        indent : str, optional
            The indentation to use for the log message. 
            Use "/t" for each indentation level (relpace the "/" with a backslash).
        underline : bool, optional
            Whether to underline the log message or not.
        type : str, optional
            Specific type of the event (FATAL, WARNING, etc...)
        """
        if self._logs_file is None:
            #if the logs path is not set
            if self.logs_path is None:
                print("--- WARNING - Logs disabled for the simulation. No information will be logged. ---")
                return
            
            #create and/or open the log file
            self._logs_file = open(
                self.logs_path,
                "w+",
                encoding="utf-8"
            )
            #if the file is not empty, overwrite the content
            if self._logs_file.readlines() != []:
                self._logs_file.write("\n")
            
            self._log_event(event="Log file initialized.", underline= True)
        
        self._logs_file.writelines(f"{indent}{datetime.now().hour}:{datetime.now().minute}:{datetime.now().second}{f" - {type}" if type is not None else ""}{f" - from {source}" if source is not None else ""} - {event}\n{indent}{"------------\n" if underline else ""}")
        self._logs_file.flush()
        
    @abstractmethod
    def _clear(self):
        """
        Free all resources used by the actor.
        """
        pass