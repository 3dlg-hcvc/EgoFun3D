from PIL import Image as PILImage
import numpy as np
import omegaconf

from typing import List, Dict, Any, Tuple, TYPE_CHECKING

# Avoid importing concrete implementations at module import time to prevent
# circular imports (those modules subclass ArticulationEstimation and import
# this file). Use TYPE_CHECKING for static analysis while deferring runtime
# imports to the factory function below.
if TYPE_CHECKING:  # pragma: no cover
    from articulation.iTACO import iTACO  # noqa: F401
    from articulation.Artipoint import Artipoint  # noqa: F401


class ArticulationEstimation:
    def __init__(self, config: omegaconf.DictConfig):
        self.config = config
    
    def articulation_estimation(self, rgb_frame_list: List[np.ndarray], reconstruction_results: Dict, part_masks: np.ndarray) -> Dict:
        raise NotImplementedError("Subclasses should implement this method to perform articulation estimation.")
    

def build_articulation_estimation_model(config: omegaconf.DictConfig) -> ArticulationEstimation:
    if config.articulation_method == "iTACO":
        from articulation.iTACO import iTACO
        return iTACO(config)
    elif config.articulation_method == "Artipoint":
        from articulation.Artipoint import Artipoint
        return Artipoint(config)
    else:
        raise ValueError(f"Unsupported articulation estimation model: {config.articulation_method}")
