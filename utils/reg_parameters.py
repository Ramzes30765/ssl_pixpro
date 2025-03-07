from clearml import Task

def register_all_parameters(task, config, prefix=""):

    for key, value in config.items():
        current_key = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            register_all_parameters(task, value, current_key)
        else:

            task.set_parameter(current_key, value)