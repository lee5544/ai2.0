"""ML model implementations and registry."""

__all__ = [
    "BaseModel",
    "BRFModel",
    "LGBModel",
    "ModelInterface",
    "RFModel",
    "XGBModel",
]


def __getattr__(name: str):
    modules = {
        "BaseModel": (".base", "BaseModel"),
        "BRFModel": (".brf", "BRFModel"),
        "LGBModel": (".lgb", "LGBModel"),
        "ModelInterface": (".base", "ModelInterface"),
        "RFModel": (".rf", "RFModel"),
        "XGBModel": (".xgb", "XGBModel"),
    }
    if name not in modules:
        raise AttributeError(name)
    from importlib import import_module

    module_name, attribute = modules[name]
    return getattr(import_module(module_name, package=__name__), attribute)
