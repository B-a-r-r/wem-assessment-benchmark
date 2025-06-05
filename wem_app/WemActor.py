from abc import ABC, abstractmethod
from datetime import datetime
from os import path
from io import TextIOWrapper

class WemActor(ABC):
    """
    An abstract class as an interface for actors in the WEM application.
    """
    
    @abstractmethod
    def verify_config(self):
        """
        Each actor must ensure that the config file contains all necessary parameters.
        """
        pass
    
    @abstractmethod
    def _log_event(self, 
        event: str, 
        source: str =None, 
        indent: str ="", 
        underline: bool =False,
        type: str =None,
        logs_path: str =None,
        logs_file: TextIOWrapper =None
    ) -> TextIOWrapper:
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
        if logs_file is None:
            #if the logs path is not set
            if logs_path is None:
                print("--- WARNING - Logs disabled for the simulation. No information will be logged. ---")
                return
            
            #create and/or open the log file
            logs_file = open(
                logs_path,
                "w+",
                encoding="utf-8"
            )
            #if the file is not empty, overwrite the content
            if logs_file.readlines() != []:
                logs_file.write("\n")
        
        logs_file.writelines(f"{indent}{datetime.now().hour}:{datetime.now().minute}:{datetime.now().second}{f" - {type}" if type is not None else ""}{f" - from {source}" if source is not None else ""} - {event}\n{indent}{"------------\n" if underline else ""}")
        logs_file.flush()
        
        return logs_file
        
    @abstractmethod
    def _clear(self):
        """
        Free all resources used by the actor.
        """
        pass