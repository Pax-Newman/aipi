
import os
import importlib
import pkgutil

# Dynamically import all the models in the current directory
# This makes it easier to set model import targets in the config
for module in pkgutil.iter_modules([os.path.dirname(__file__)]):
    if module.name == '__init__':
        continue
    globals().update(importlib.import_module('.' + module.name, __package__).__dict__)
importlib.invalidate_caches()

