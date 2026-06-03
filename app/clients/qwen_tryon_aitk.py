from __future__ import annotations

import importlib
import logging
import os
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import Settings, get_enabled_tryon_specialists
from app.runtime.tryon_types import TryonRunResult, TryonRuntimeStatus

logger = logging.getLogger("glamify-ai")


class TryonRuntimeError(RuntimeError):
    pass


class TryonGenerationError(RuntimeError):
    pass


SPECIALIST_CATEGORIES: tuple[str, ...] = ("top", "bottom", "dress", "multi")


class QwenTryonAitkClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._aitk_root = Path(settings.ai_toolkit_root).expanduser()
        self._model_path = Path(settings.qwen_image_edit_model_path).expanduser()
        self._use_specialists = bool(settings.tryon_use_specialists)
        self._enabled_specialists = (
            get_enabled_tryon_specialists(settings)
            if self._use_specialists
            else SPECIALIST_CATEGORIES
        )
        self._checkpoint_path = Path(settings.tryon_lora_path).expanduser()
        self._specialist_paths: dict[str, Path] = {
            "top": Path(settings.tryon_lora_top_path).expanduser(),
            "bottom": Path(settings.tryon_lora_bottom_path).expanduser(),
            "dress": Path(settings.tryon_lora_dress_path).expanduser(),
            "multi": Path(settings.tryon_lora_multi_path).expanduser(),
        }
        self._pipeline: Any | None = None
        self._network: Any | None = None
        self._loaded_checkpoint: str | None = None
        self._specialist_state_dicts: dict[str, OrderedDict[str, Any]] = {}
        self._active_specialist: str | None = None
        self._torch_module: Any | None = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._device = "cpu"
        self._dtype_name = "float32"
        self._toolkit: dict[str, Any] | None = None

    def warmup(self) -> None:
        self._ensure_ready()
        if self._use_specialists:
            self._load_specialist_state_dicts()
        else:
            self._load_checkpoint_if_needed()

    def status(self) -> TryonRuntimeStatus:
        lora_loaded = (
            bool(self._specialist_state_dicts)
            if self._use_specialists
            else self._loaded_checkpoint is not None
        )
        return TryonRuntimeStatus(
            loaded=self._pipeline is not None and self._network is not None,
            backend="ai_toolkit_exact" if self._pipeline is not None else None,
            lora_loaded=lora_loaded,
        )

    def set_active_specialist(self, category: str) -> None:
        if not self._use_specialists:
            return
        if category not in self._enabled_specialists:
            raise TryonRuntimeError(f"Try-on specialist is not enabled: {category}")
        self._ensure_ready()
        self._load_specialist_state_dicts()
        if self._active_specialist == category:
            return
        if self._network is None:
            raise TryonRuntimeError("AI-Toolkit LoRA network is not initialized.")
        state = self._specialist_state_dicts[category]
        self._network.load_state_dict(state, strict=False)
        self._active_specialist = category
        logger.info("Switched try-on specialist LoRA to '%s'", category)

    def run_tryon(
        self,
        *,
        person_image_path: str,
        garment_reference_path: str,
        prompt: str,
        steps: int,
        guidance_scale: float,
        seed: int,
        output_path: str,
        output_width: int,
        output_height: int,
        lora_key: str | None = None,
    ) -> TryonRunResult:
        self._ensure_ready()
        if (
            self._pipeline is None
            or self._network is None
            or self._toolkit is None
            or self._torch_module is None
        ):
            raise TryonRuntimeError("AI-Toolkit try-on runtime is not loaded.")

        if self._use_specialists:
            if lora_key is None:
                raise TryonRuntimeError(
                    "lora_key is required when TRYON_USE_SPECIALISTS is enabled.",
                )
            self.set_active_specialist(lora_key)
            active_checkpoint = str(self._specialist_paths[lora_key])
        else:
            self._load_checkpoint_if_needed()
            active_checkpoint = str(self._checkpoint_path)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        started_at = time.perf_counter()
        conf = self._toolkit["GenerateImageConfig"](
            prompt=str(prompt).strip(),
            width=int(output_width),
            height=int(output_height),
            negative_prompt="",
            seed=int(seed),
            guidance_scale=float(guidance_scale),
            guidance_rescale=float(self._settings.tryon_guidance_rescale),
            num_inference_steps=int(steps),
            network_multiplier=float(self._settings.tryon_lora_scale),
            output_path=str(output_file),
            output_ext="jpg",
            ctrl_img_1=str(person_image_path),
            ctrl_img_2=str(garment_reference_path),
            do_cfg_norm=bool(self._settings.tryon_do_cfg_norm),
        )

        with self._infer_lock, self._torch_module.inference_mode():
            self._pipeline.generate_images(
                [conf],
                sampler=str(self._settings.tryon_sampler),
            )

        if not output_file.exists():
            raise TryonGenerationError("No image was generated by the try-on runtime.")

        image = Image.open(output_file).convert("RGB")
        metadata = {
            "backend": "ai_toolkit_exact",
            "architecture": "qwen_image_edit_plus",
            "model_source": str(self._model_path),
            "ai_toolkit_root": str(self._aitk_root),
            "checkpoint_path": active_checkpoint,
            "lora_key": lora_key if self._use_specialists else None,
            "lora_rank": int(self._settings.tryon_lora_rank),
            "lora_alpha": int(self._settings.tryon_lora_alpha),
            "network_multiplier": float(self._settings.tryon_lora_scale),
            "guidance_scale": float(guidance_scale),
            "guidance_rescale": float(self._settings.tryon_guidance_rescale),
            "do_cfg_norm": bool(self._settings.tryon_do_cfg_norm),
            "sampler": str(self._settings.tryon_sampler),
            "seed": int(seed),
            "steps": int(steps),
            "control_order": {
                "ctrl_img_1": "person",
                "ctrl_img_2": "garment_reference",
            },
            "output_size": {
                "width": int(output_width),
                "height": int(output_height),
            },
            "dtype": self._dtype_name,
            "device": self._device,
        }
        return TryonRunResult(
            image=image,
            metadata=metadata,
            wall_seconds=float(round(time.perf_counter() - started_at, 3)),
        )

    def _ensure_ready(self) -> None:
        if self._pipeline is not None and self._network is not None and self._toolkit is not None:
            return
        with self._load_lock:
            if self._pipeline is None or self._network is None or self._toolkit is None:
                self._load_runtime()

    def _load_runtime(self) -> None:
        if not self._aitk_root.exists():
            raise TryonRuntimeError(f"AI-Toolkit root does not exist: {self._aitk_root}")
        if not self._model_path.exists():
            raise TryonRuntimeError(f"Qwen model path does not exist: {self._model_path}")

        os.environ.setdefault("DISABLE_TELEMETRY", "YES")
        os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
        # Match the validated RunPod AI-Toolkit startup path. This is process-global,
        # so the rest of the service must keep using absolute paths.
        os.chdir(self._aitk_root)
        aitk_root_str = str(self._aitk_root)
        if aitk_root_str not in sys.path:
            sys.path.insert(0, aitk_root_str)

        try:
            config_module = importlib.import_module("toolkit.config_modules")
            lora_module = importlib.import_module("toolkit.lora_special")
            train_tools_module = importlib.import_module("toolkit.train_tools")
            model_module = importlib.import_module("toolkit.util.get_model")
        except Exception as exc:
            raise TryonRuntimeError(
                f"Unable to import AI-Toolkit modules from {self._aitk_root}: {exc}",
            ) from exc

        ModelConfig = config_module.ModelConfig
        NetworkConfig = config_module.NetworkConfig
        GenerateImageConfig = config_module.GenerateImageConfig
        LoRASpecialNetwork = lora_module.LoRASpecialNetwork
        get_torch_dtype = train_tools_module.get_torch_dtype
        get_model_class = model_module.get_model_class

        model_config = ModelConfig(
            **{
                "name_or_path": str(self._model_path),
                "arch": "qwen_image_edit_plus",
                "quantize": False,
                "quantize_te": False,
                "low_vram": False,
            },
        )
        net_config = NetworkConfig(
            **{
                "type": "lora",
                "linear": int(self._settings.tryon_lora_rank),
                "linear_alpha": int(self._settings.tryon_lora_alpha),
            },
        )
        torch_module = importlib.import_module("torch")
        self._torch_module = torch_module
        self._device = "cuda:0" if torch_module.cuda.is_available() else "cpu"
        self._dtype_name = "bfloat16" if self._device.startswith("cuda") else "float32"

        ModelClass = get_model_class(model_config)
        sampler = ModelClass.get_train_scheduler()
        dtype = get_torch_dtype("bf16" if self._device.startswith("cuda") else "fp32")

        sd = ModelClass(
            device=self._device,
            model_config=model_config,
            dtype=dtype,
            noise_scheduler=sampler,
        )
        sd.load_model()
        if hasattr(sd, "pipeline") and self._device.startswith("cuda"):
            sd.pipeline.to(self._device, dtype)

        network_kwargs = dict(getattr(net_config, "network_kwargs", {}) or {})
        if hasattr(sd, "target_lora_modules"):
            network_kwargs["target_lin_modules"] = sd.target_lora_modules

        network = LoRASpecialNetwork(
            text_encoder=sd.text_encoder,
            unet=sd.get_model_to_train(),
            lora_dim=net_config.linear,
            multiplier=float(self._settings.tryon_lora_scale),
            alpha=net_config.linear_alpha,
            train_unet=True,
            train_text_encoder=False,
            conv_lora_dim=net_config.conv,
            conv_alpha=net_config.conv_alpha,
            is_sdxl=model_config.is_xl or model_config.is_ssd,
            is_v2=model_config.is_v2,
            is_v3=model_config.is_v3,
            is_pixart=model_config.is_pixart,
            is_auraflow=model_config.is_auraflow,
            is_flux=model_config.is_flux,
            is_lumina2=model_config.is_lumina2,
            is_ssd=model_config.is_ssd,
            is_vega=model_config.is_vega,
            dropout=net_config.dropout,
            use_text_encoder_1=model_config.use_text_encoder_1,
            use_text_encoder_2=model_config.use_text_encoder_2,
            network_config=net_config,
            network_type=net_config.type,
            transformer_only=net_config.transformer_only,
            is_transformer=sd.is_transformer,
            base_model=sd,
            **network_kwargs,
        )
        network.force_to(sd.device_torch, dtype=torch_module.float32)
        sd.network = network
        network._update_torch_multiplier()
        network.apply_to(sd.text_encoder, sd.get_model_to_train(), False, True)

        self._pipeline = sd
        self._network = network
        self._toolkit = {
            "GenerateImageConfig": GenerateImageConfig,
        }
        logger.info(
            "Qwen try-on AI-Toolkit runtime ready (device=%s, specialists=%s)",
            self._device,
            self._use_specialists,
        )

    def _load_checkpoint_if_needed(self) -> None:
        if self._network is None:
            raise TryonRuntimeError("AI-Toolkit LoRA network is not initialized.")
        checkpoint = str(self._checkpoint_path)
        if not self._checkpoint_path.exists():
            raise TryonRuntimeError(f"Try-on checkpoint does not exist: {self._checkpoint_path}")
        if self._loaded_checkpoint == checkpoint:
            return
        self._network.load_weights(checkpoint)
        self._loaded_checkpoint = checkpoint
        logger.info("Loaded try-on checkpoint: %s", checkpoint)

    def _load_specialist_state_dicts(self) -> None:
        if not self._use_specialists:
            return
        if self._network is None:
            raise TryonRuntimeError("AI-Toolkit LoRA network is not initialized.")
        if self._specialist_state_dicts:
            return
        for category in self._enabled_specialists:
            path = self._specialist_paths[category]
            if not path.exists():
                raise TryonRuntimeError(
                    f"Try-on specialist checkpoint missing for '{category}': {path}",
                )
            self._network.load_weights(str(path))
            snapshot = OrderedDict(
                (key, tensor.detach().clone())
                for key, tensor in self._network.state_dict().items()
            )
            self._specialist_state_dicts[category] = snapshot
            logger.info("Cached specialist LoRA weights: %s -> %s", category, path)
        first = SPECIALIST_CATEGORIES[0]
        self._network.load_state_dict(self._specialist_state_dicts[first], strict=False)
        self._active_specialist = first
