from os import path, listdir
from warnings import warn
from makegraphs import ExpTrajectoryAnalyzer, ExpSpatialAnalyzer, ExpTopBAnalyzer, ExpEmergenceAnalyzer

#
# author: Clément BARRIERE
# github: B-a-r-r
#

#TODO: Implement and link the VisualsMakerUI class for cli visuals selection.

class VisualsMakerUI:
    
    def __init__(self):
        self.available_makers = self.get_available_makers()
        
        
        
    
    def get_available_makers(self):
        found = [
                file_name.removeprefix("Exp").removesuffix(".py") for file_name in listdir(path.dirname(__file__)) 
                if file_name.startswith("Exp") 
                and file_name.endswith('.py')
        ]
        
        if len(found) == 0:
            warn("No visuals maker found for ui.")
            
        return found