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

    def parse_value(value_str):
        try:
            return eval(value_str)  # Ïðåîáðàçóåò "False" > False, "1e-5" > 0.00001
        except:
            return value_str

    nested_dict = {}
    for key, value in original_dict.items():
        # new_key = key.replace("General/", "")
        parts = key.split(".")
        current_level = nested_dict
        for part in parts[:-1]:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]
        current_level[parts[-1]] = parse_value(value)

    return OmegaConf.create(nested_dict)