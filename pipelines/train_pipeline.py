import os
import json
from clearml import PipelineDecorator, Task
from omegaconf import OmegaConf
from src.train import train

# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

@PipelineDecorator.component(return_values=["config"], cache=False)
def load_config(base_config_path: str, override_params: dict):

    print(f"OVERRIDE PARAMS - {override_params}")
    print(f"OVERRIDE PARAMS TYPE- {type(override_params)}")

    working_dir = os.environ.get("CLEARML_TASK_WORKING_DIR")
    if not working_dir:
        working_dir = ''
    working_base_config_path = os.path.join(working_dir, base_config_path)

    base_conf = OmegaConf.load(working_base_config_path)

    if override_params.strip() and override_params.strip() not in ["{}", ""]:
        try:
            override_dict = json.loads(override_params)
            overrides_conf = OmegaConf.create(override_dict)
        except Exception as e:
            try:
                overrides_conf = OmegaConf.from_yaml(override_params)
            except Exception as ye:
                raise ValueError(f"Cannot parse override_params: {e}; {ye}")
    else:
        overrides_conf = OmegaConf.create({})

    merged_conf = OmegaConf.merge(base_conf, overrides_conf)
    print('CONFIG SUCCESFULLY LOADED')
    return merged_conf

@PipelineDecorator.component(return_values=["model"], cache=False)
def train_step(config, parent_task_id):

    model = train(config, parent_task_id)
    return model

@PipelineDecorator.pipeline(name="SSL Pipeline", project="PixPro")
def pipeline_flow(
    base_config_path: str = "configs/config.yaml",
    override_params: str = "{}"
):
    cfg = load_config(base_config_path, override_params)

    pretrained = 'pretrained' if cfg.model.pretrained else 'not_pretrained'
    pipeline = PipelineDecorator.get_current_pipeline()
    pipeline.connect_configuration(configuration=cfg)
    
    pipeline.add_tags(
        [cfg.model.backbone, pretrained, f'epochs-{cfg.train.epoch}', cfg.data.dataset_name]
        )

    parent_task_id = Task.current_task().id if Task.current_task() is not None else None
    model = train_step(cfg, parent_task_id)
    return model

if __name__ == "__main__":
    
    PipelineDecorator.run_locally()
    pipeline_flow()
