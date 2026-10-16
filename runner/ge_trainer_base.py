import argparse
import json
import logging
import os
import random
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import List

import diffusers
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DeepSpeedPlugin,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter

from utils import import_custom_class, init_logging
from utils.data_utils import get_text_conditions
from utils.memory_utils import free_memory
from utils.model_utils import (
    count_model_parameters,
    load_condition_models,
    load_diffusion_model,
    load_latent_models,
    load_vae_models,
    unwrap_model,
)
from utils.path_utils import load_resolved_config


os.environ["TOKENIZERS_PARALLELISM"] = "false"

LOG_LEVEL = "INFO"
logger = get_logger("wm_runner")
logger.setLevel(LOG_LEVEL)


class State:
    seed: int = None
    model_name: str = None
    accelerator: Accelerator = None
    weight_dtype: torch.dtype = None
    train_epochs: int = None
    train_steps: int = None
    overwrote_max_train_steps: bool = False
    num_trainable_parameters: int = 0
    learning_rate: float = None
    train_batch_size: int = None
    generator: torch.Generator = None
    repo_id: str = None
    output_dir: str = None


class BaseGETrainer:
    """Common GE trainer plumbing.

    This base intentionally owns only infrastructure: config parsing,
    distributed setup, datasets, model loading, common prepare hooks, and
    checkpoint helpers. Concrete trainers own their batch semantics, losses,
    validation, and train loop.
    """

    def __init__(self, config_file, to_log=True, output_dir=None) -> None:
        self.config_file = str(config_file)
        cd = load_resolved_config(config_file)
        args = argparse.Namespace(**cd)
        args.lr = float(args.lr)
        args.epsilon = float(args.epsilon)
        args.weight_decay = float(args.weight_decay)
        self.args = args

        self.label_filter = getattr(self.args, "label_filter", "all")
        if self.label_filter not in ("all", "only_normal"):
            raise ValueError(f"unsupported label_filter={self.label_filter}; use 'all' or 'only_normal'")

        if output_dir is not None:
            self.args.output_dir = output_dir
        if self.args.load_weights is False:
            print("You are not loading the pretrained weights, please check the code.")

        self.state = State()
        self.tokenizer = None
        self.text_encoder = None
        self.diffusion_model = None
        self.unet = None
        self.vae = None
        self.scheduler = None

        self._init_distributed()
        self._init_logging()
        self._init_directories_and_repositories()
        self.state.model_name = self.args.model_name
        self._init_save_folder(to_log=to_log)
        init_logging(self.save_folder, rank=self.state.accelerator.process_index)

    def _init_save_folder(self, to_log=True):
        if self.state.accelerator.is_main_process:
            from datetime import datetime

            start_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            self.save_folder = os.path.join(self.args.output_dir, start_time)
            if getattr(self.args, "sub_folder", False):
                self.save_folder = os.path.join(self.args.output_dir, self.args.sub_folder)
            os.makedirs(self.save_folder, exist_ok=True)

            args_dict = vars(deepcopy(self.args))
            for key, value in args_dict.items():
                args_dict[key] = str(value)
            with open(os.path.join(self.save_folder, "config.json"), "w", encoding="utf-8") as file:
                json.dump(args_dict, file, indent=4, sort_keys=False)
            self._save_input_config_snapshot()

            self.writer = SummaryWriter(log_dir=self.save_folder) if to_log else None
            save_folder_bytes = self.save_folder.encode()
            folder_len_tensor = torch.tensor([len(save_folder_bytes)], device=self.state.accelerator.device)
            dist.broadcast(folder_len_tensor, src=0)
            folder_tensor = torch.ByteTensor(list(save_folder_bytes)).to(self.state.accelerator.device)
            dist.broadcast(folder_tensor, src=0)
        else:
            folder_len_tensor = torch.tensor([0], device=self.state.accelerator.device)
            dist.broadcast(folder_len_tensor, src=0)
            folder_tensor = torch.empty(folder_len_tensor.item(), dtype=torch.uint8, device=self.state.accelerator.device)
            dist.broadcast(folder_tensor, src=0)
            self.save_folder = bytes(folder_tensor.tolist()).decode()

    def _save_input_config_snapshot(self):
        source_path = Path(self.config_file).expanduser()
        with open(os.path.join(self.save_folder, "input_config_path.txt"), "w", encoding="utf-8") as file:
            file.write(str(source_path) + "\n")
        if not source_path.is_file():
            logger.warning(f"Input config file does not exist, cannot snapshot verbatim: {source_path}")
            return
        with open(source_path, "rb") as src, open(os.path.join(self.save_folder, "input_config.yaml"), "wb") as dst:
            dst.write(src.read())

    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)
        project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout)
        )
        mixed_precision = "no" if torch.backends.mps.is_available() else self.args.mixed_precision
        report_to = None if self.args.report_to.lower() == "none" else self.args.report_to

        if getattr(self.args, "use_deepspeed", False):
            per_device_bs = self.args.batch_size
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            grad_accum = self.args.gradient_accumulation_steps
            self.args.deepspeed["train_batch_size"] = per_device_bs * world_size * grad_accum
            ds_plugin = DeepSpeedPlugin(
                hf_ds_config=self.args.deepspeed,
                gradient_accumulation_steps=grad_accum,
            )
        else:
            ds_plugin = None

        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
            deepspeed_plugin=ds_plugin,
        )
        if torch.backends.mps.is_available():
            accelerator.native_amp = False
        self.state.accelerator = accelerator

        if self.args.seed is not None:
            self.state.seed = self.args.seed
            set_seed(self.args.seed)

        weight_dtype = torch.float32
        if self.state.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.state.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.state.weight_dtype = weight_dtype

    def _init_logging(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=LOG_LEVEL,
        )
        if self.state.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        logger.info("Initialized Trainer")
        logger.info(self.state.accelerator.state, main_process_only=False)

    def _init_directories_and_repositories(self):
        if self.state.accelerator.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)
            self.state.output_dir = self.args.output_dir

    def prepare_dataset(self) -> None:
        logger.info(f"Training Dataset: {self.args.train_data_class}")
        train_dataset_class = import_custom_class(self.args.train_data_class, self.args.train_data_class_path)
        self.train_dataset = train_dataset_class(**self.args.data["train"])

        loader_kwargs = {}
        pin_memory = getattr(self.args, "pin_memory", False)
        if self.args.dataloader_num_workers > 0:
            loader_kwargs["persistent_workers"] = getattr(self.args, "persistent_workers", False)
            loader_kwargs["prefetch_factor"] = getattr(self.args, "prefetch_factor", 2)
        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=pin_memory,
            multiprocessing_context=None,
            **loader_kwargs,
        )
        logger.info(f">>>>>>>>>>>>>Total Train Eps: {len(self.train_dataset)}<<<<<<<<<<<<<<<<<<\n")

        if "val" in self.args.data:
            self.prepare_val_dataset()

    def prepare_val_dataset(self) -> None:
        if not hasattr(self.args, "val_data_class"):
            self.args.val_data_class = self.args.train_data_class
        if not hasattr(self.args, "val_data_class_path"):
            self.args.val_data_class_path = self.args.train_data_class_path
        logger.info(f"Validation Dataset: {self.args.val_data_class}")

        val_dataset_class = import_custom_class(self.args.val_data_class, self.args.val_data_class_path)
        self.val_dataset = val_dataset_class(**self.args.data["val"])

        self.val_index = [random.randint(0, len(self.val_dataset) - 1) for _ in range(self.args.batch_size)]
        if self.state.accelerator.is_main_process:
            with open(os.path.join(self.save_folder, "idx.txt"), "w", encoding="utf-8") as file:
                file.write(", ".join(map(str, self.val_index)))

        subset = torch.utils.data.Subset(self.val_dataset, self.val_index)
        val_loader_kwargs = {}
        pin_memory = getattr(self.args, "pin_memory", False)
        if self.args.dataloader_num_workers > 0:
            val_loader_kwargs["persistent_workers"] = getattr(self.args, "persistent_workers", False)
            val_loader_kwargs["prefetch_factor"] = getattr(self.args, "prefetch_factor", 2)
        self.val_dataloader = torch.utils.data.DataLoader(
            subset,
            batch_size=self.args.batch_size,
            shuffle=getattr(self.args, "val_shuffle", False),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=pin_memory,
            multiprocessing_context=None,
            **val_loader_kwargs,
        )
        logger.info(f">>>>>>>>>>>>>Total Validatoin Eps: {len(self.val_dataset)}<<<<<<<<<<<<<<<<<<\n")

    def prepare_models(self):
        logger.info("Initializing models")
        device = self.state.accelerator.device
        dtype = self.state.weight_dtype

        tokenizer_class = import_custom_class(
            self.args.tokenizer_class, getattr(self.args, "tokenizer_class_path", "transformers")
        )
        textenc_class = import_custom_class(
            self.args.textenc_class, getattr(self.args, "textenc_class_path", "transformers")
        )
        cond_models = load_condition_models(
            tokenizer_class,
            textenc_class,
            self.args.pretrained_model_name_or_path
            if not hasattr(self.args, "tokenizer_pretrained_model_name_or_path")
            else self.args.tokenizer_pretrained_model_name_or_path,
            load_weights=True,
        )
        self.tokenizer, text_encoder = cond_models["tokenizer"], cond_models["text_encoder"]
        self.text_encoder = text_encoder.to(device, dtype=dtype).eval()
        self.text_uncond = get_text_conditions(self.tokenizer, self.text_encoder, prompt="")
        self.uncond_prompt_embeds = self.text_uncond["prompt_embeds"]
        self.uncond_prompt_attention_mask = self.text_uncond["prompt_attention_mask"]

        vae_class = import_custom_class(self.args.vae_class, getattr(self.args, "vae_class_path", "transformers"))
        if getattr(self.args, "vae_path", False):
            self.vae = load_vae_models(vae_class, self.args.vae_path).to(device, dtype=dtype).eval()
        else:
            self.vae = load_latent_models(vae_class, self.args.pretrained_model_name_or_path)["vae"].to(
                device, dtype=dtype
            ).eval()
        if isinstance(self.vae.latents_mean, List):
            self.vae.latents_mean = torch.FloatTensor(self.vae.latents_mean)
        if isinstance(self.vae.latents_std, List):
            self.vae.latents_std = torch.FloatTensor(self.vae.latents_std)
        if self.vae is not None:
            if self.args.enable_slicing:
                self.vae.enable_slicing()
            if self.args.enable_tiling:
                self.vae.enable_tiling()
        self.SPATIAL_DOWN_RATIO = self.vae.spatial_compression_ratio
        self.TEMPORAL_DOWN_RATIO = self.vae.temporal_compression_ratio
        logger.info(f"SPATIAL_DOWN_RATIO of VAE :{self.SPATIAL_DOWN_RATIO}")
        logger.info(f"TEMPORAL_DOWN_RATIO of VAE :{self.TEMPORAL_DOWN_RATIO}")

        diffusion_model_class = import_custom_class(
            self.args.diffusion_model_class, getattr(self.args, "diffusion_model_class_path", "transformers")
        )
        self.diffusion_model = load_diffusion_model(
            model_cls=diffusion_model_class,
            model_dir=self.args.diffusion_model["model_path"],
            load_weights=self.args.load_weights and getattr(self.args, "load_diffusion_model_weights", True),
            **self.args.diffusion_model["config"],
        ).to(device, dtype=dtype)
        logger.info(f"Total parameters for transformer model:{count_model_parameters(self.diffusion_model)}")

        diffusion_scheduler_class = import_custom_class(
            self.args.diffusion_scheduler_class, getattr(self.args, "diffusion_scheduler_class_path", "diffusers")
        )
        if hasattr(self.args, "diffusion_scheduler_args"):
            self.scheduler = diffusion_scheduler_class(**self.args.diffusion_scheduler_args)
        else:
            self.scheduler = diffusion_scheduler_class()

        self.pipeline_class = import_custom_class(
            self.args.pipeline_class, getattr(self.args, "pipeline_class_path", "diffusers")
        )

    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")
        if torch.backends.mps.is_available() and self.state.weight_dtype == torch.bfloat16:
            raise ValueError("Mixed precision training with bfloat16 is not supported on MPS.")
        if self.args.gradient_checkpointing:
            self.diffusion_model.enable_gradient_checkpointing()
        if self.args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

    def prepare_for_training(self):
        self.diffusion_model, self.optimizer, self.train_dataloader, self.lr_scheduler = (
            self.state.accelerator.prepare(
                self.diffusion_model,
                self.optimizer,
                self.train_dataloader,
                self.lr_scheduler,
            )
        )

    def prepare_trackers(self):
        logger.info("Initializing trackers")
        tracker_name = self.args.tracker_name or "model_train"
        folder_name = os.path.basename(str(self.save_folder))
        configured_run_name = getattr(self.args, "run_name", None)
        run_name = f"{folder_name}-{configured_run_name}" if configured_run_name else folder_name
        init_kwargs = {}
        if str(getattr(self.args, "report_to", "")).lower() == "wandb":
            init_kwargs["wandb"] = {"name": run_name}
        self.state.accelerator.init_trackers(
            tracker_name,
            config=self.args.__dict__,
            init_kwargs=init_kwargs or None,
        )

        
    def save_diffusion_model(self, global_step):
        self.state.accelerator.wait_for_everyone()
        if not self.state.accelerator.is_main_process:
            return
        model_to_save = unwrap_model(self.state.accelerator, self.diffusion_model)
        model_save_dir = os.path.join(self.save_folder, f"step_{global_step}")
        model_to_save.save_pretrained(model_save_dir, safe_serialization=True)
        del model_to_save

    def cleanup_training(self):
        if self.diffusion_model is not None:
            del self.diffusion_model
        if self.scheduler is not None:
            del self.scheduler
        free_memory()
