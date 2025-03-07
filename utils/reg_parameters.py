from clearml import Task
from omegaconf import OmegaConf

def register_all_parameters(task, config, prefix=""):

    for key, value in config.items():
        current_key = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            register_all_parameters(task, value, current_key)
        else:

            task.set_parameter(current_key, value)

def dict_to_omegaconf(original_dict):
    nested_dict = {}
    for key, value in original_dict.items():
        new_key = key.replace("General/", "")
        parts = new_key.split(".")
        current_level = nested_dict
        for part in parts[:-1]:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]
        current_level[parts[-1]] = value

    return OmegaConf.create(nested_dict)