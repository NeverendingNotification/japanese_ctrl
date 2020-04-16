
import torch

from transformers import AutoConfig
from transformers import CTRLLMHeadModel

def get_models(params, datasets):
    model_params = params["model_params"]
    model_name = model_params.get("transformer_name", "ctrl")
    n_gpus = model_params.get("n_gpus", 0)
    config_params = model_params.get("config_params", None)

    if n_gpus ==0 or (not torch.cuda.is_available()):
        device = "cpu"
        device_ids = []
    else:
        device = "cuda"
        device_ids = range(min(n_gpus, torch.cuda.device_count()))

    print("gpu device ids ", device_ids)

    if model_name == "ctrl":
        config = AutoConfig.from_pretrained("ctrl")
        if config_params is not None:
            for k, v in config_params.items():
                if hasattr(config, k):
                    setattr(config, k, v)
#            print(config)        
        model = CTRLLMHeadModel(config).to(device)
    else:
        raise NotImplementedError()

    models = {"model": model, "device":device, "device_ids": device_ids}
    return models
