__package_name__ = 'Adversarial_attack_defense'
__version__ = '1.0.0'

import os
from .attack import *
from .dataset import *
from .defense import *
from .models.model import *
from .utils import *
from .utils.registry import registry


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache_dir = os.environ['AAD_CACHE'] if os.environ.get('AAD_CACHE') else os.path.join(root_dir, 'cache')

registry.register_path("root_dir", root_dir)
registry.register_path("cache_dir", cache_dir)
