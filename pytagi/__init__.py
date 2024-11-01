from pytagi.tagi_utils import Normalizer, HRCSoftmax, Utils, exponential_scheduler, SeedManager
from pytagi.metric import HRCSoftmaxMetric
import cutagi
from .__version import __version__

# Create a single instance of SeedManager
_seed_manager = SeedManager()

def manual_seed(seed: int) -> None:
    """Set the random seed for reproducibility."""
    _seed_manager.manual_seed(seed)