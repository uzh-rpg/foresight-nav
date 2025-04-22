"""
Base class for exploration models.
"""
from omegaconf import OmegaConf

class ExplorationModel:
    def __init__(self, conf: OmegaConf):
        pass

    def get_model(self):
        raise NotImplementedError

    def get_model_output(self, input):
        raise NotImplementedError
    
    def reset_eval(self, *args, **kwargs):
        """
        Reset the model for evaluation.
        """
        pass
    
    def get_goal(self, input):
        """
        All derived classes should implement this function to return the goal for the given input.
        Input: input (dict): Input data for the model.
            Keys: 
                - pred_map (np.array): Occupancy map
                - x (int): Current x-coordinate
                - y (int): Current y-coordinate
                - [Optional] clipfeat_map (np.array): CLIP features for ImaginationModel
        
        Output: output (dict)
            Keys:
                - goal_x (int): Goal x-coordinate
                - goal_y (int): Goal y-coordinate
                - roll_back (bool): Whether to roll back the goal selection
                - [Optional] pred_occ (np.array): Predicted occupancy map for ImaginationModel
        """
        raise NotImplementedError