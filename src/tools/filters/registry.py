import importlib
import pkgutil
import src.tools.filters.filters

_filter_registry = {}

def _discover_filters():
    """Dynamically discover all subclasses of Filter in filters.filters."""
    for _, module_name, _ in pkgutil.iter_modules(src.tools.filters.filters.__path__):
        module = importlib.import_module(f'src.tools.filters.filters.{module_name}')
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if isinstance(obj, type):
                if hasattr(obj, 'apply'):
                    _filter_registry[obj.__name__] = obj

# Call on import
_discover_filters()

def create_filter(name: str, **kwargs):
    """Create a filter instance by name."""
    cls = _filter_registry.get(name)
    if cls is None:
        raise ValueError(f"No filter found for name '{name}'")
    return cls(**kwargs)
