
from fastapi import FastAPI, Request
from skeletonkey import instantiate

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
