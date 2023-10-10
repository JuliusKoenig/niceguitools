import os
from pathlib import Path

__module_path__ = os.path.dirname(os.path.abspath(__file__))
__module_files__ = []
__author__ = "Julius Koenig"
__version_file__ = str(Path(__module_path__) / "__version__")
__version__ = open(__version_file__).read()

from .refreshables import *
