
from fastapi import FastAPI, Request
from skeletonkey import instantiate, unlock

def set_model(app: FastAPI, name: str, model_type: str):
    assert model_type in app.state.config.models, f"Model type {model_type} not found in config"
    assert name in getattr(app.state.config.models, model_type), f"Model {name} not found in config"

    # FIXME we need to check if the model_types are the same as well
    if app.state.model_name == name:
        return app.state.model

    models = getattr(app.state.config.models, model_type)
    model_settings = getattr(models, name)

    app.state.model = instantiate(model_settings)
    app.state.model_name = name

    return app.state.model

def get_app_instance(request: Request) -> FastAPI:
    return request.app

def inject_models_to_enum(enum_locals: dict, values: dict):
    # Dynamically fill an enum with values
    for key, value in values.items():
        enum_locals[key] = value.name

def load_config(path):
    @unlock(path)
    def inner(config):
        return config
    return inner()

