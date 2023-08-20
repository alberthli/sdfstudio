from pathlib import Path

import numpy as np
from sdfstudio.configs.base_config import Config, TrainerConfig, ViewerConfig
from sdfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from sdfstudio.data.dataparsers.sdfstudio_dataparser import SdfStudioDataParserConfig
from sdfstudio.engine.optimizers import AdamOptimizerConfig
from sdfstudio.engine.schedulers import MultiStepSchedulerConfig, NeuSSchedulerConfig
from sdfstudio.fields.sdf_field import SDFFieldConfig
from sdfstudio.models.neus_facto import NeuSFactoModelConfig
from sdfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from sdfstudio.scripts.train import main as train_main

max_num_iterations = 200
config = Config(
    method_name="neus-facto",
    data=Path("/root/test/data/sdfstudio-demo-data/dtu-scan65"),
    trainer=TrainerConfig(
        steps_per_eval_image=max_num_iterations + 1,  # no eval
        steps_per_eval_batch=max_num_iterations + 1,
        steps_per_save=max_num_iterations - 1,  # save right before finishing
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=max_num_iterations,
        mixed_precision=False,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=SdfStudioDataParserConfig(),
            train_num_rays_per_batch=2048,
            eval_num_rays_per_batch=1024,
        ),
        model=NeuSFactoModelConfig(
            sdf_field=SDFFieldConfig(
                use_grid_feature=True,
                num_layers=2,
                num_layers_color=2,
                hidden_dim=256,
                bias=0.5,
                beta_init=0.3,
                use_appearance_embedding=False,
                inside_outside=False,
                encoding_type="periodic",
            ),
            background_model="none",
            eval_num_rays_per_chunk=1024,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(max_steps=20000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="tensorboard",
)
config.pipeline.datamanager.camera_optimizer.mode = "off"
config.pipeline.datamanager.camera_optimizer.optimizer = AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
train_main(config)  # this does the training
breakpoint()