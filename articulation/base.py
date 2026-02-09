from PIL import Image as PILImage
import numpy as np
import omegaconf

from articulation.iTACO import iTACO
from articulation.Artipoint import Artipoint

from typing import List, Dict, Any, Tuple


class ArticulationEstimation:
    def __init__(self, config: omegaconf.DictConfig):
        self.config = config
    
    def articulation_estimation(self, rgb_frame_list: List[PILImage.Image], reconstruction_results: Dict, part_masks: np.ndarray) -> Dict:
        raise NotImplementedError("Subclasses should implement this method to perform articulation estimation.")
    

def build_articulation_estimation_model(config: omegaconf.DictConfig) -> ArticulationEstimation:
    if config.name == "iTACO":
        return iTACO(config)
    elif config.name == "Artipoint":
        return Artipoint(config)
    else:
        raise ValueError(f"Unsupported articulation estimation model: {config.name}")