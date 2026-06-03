from __future__ import annotations

import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from PIL import Image

from app.clients.qwen_tryon_aitk import (
    SPECIALIST_CATEGORIES,
    QwenTryonAitkClient,
    TryonGenerationError,
    TryonRuntimeError,
)
from app.clients.qwen_wardrobe_aitk import (
    WARDROBE_CATEGORIES,
    WardrobeGenerationError,
    WardrobeRuntimeError,
)
from app.config import Settings
from app.constants import wardrobe as wardrobe_constants
from app.runtime.tryon_types import TryonRunResult, TryonRuntimeStatus
from app.runtime.wardrobe_types import WardrobeRunResult, WardrobeRuntimeStatus

logger = logging.getLogger("glamify-ai")


class QwenSharedAitkClient(QwenTryonAitkClient):
    """Single resident Qwen Image Edit Plus runtime shared by try-on and wardrobe."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._wardrobe_specialist_paths: dict[str, Path] = {
            "top": Path(settings.wardrobe_lora_top_path).expanduser(),
            "bottom": Path(settings.wardrobe_lora_bottom_path).expanduser(),
            "dress": Path(settings.wardrobe_lora_dress_path).expanduser(),
        }
        self._wardrobe_specialist_state_dicts: dict[str, OrderedDict[str, Any]] = {}
        self._active_lora_namespace: str | None = None
        self._active_lora_key: str | None = None

    def warmup(self) -> None:
        self._ensure_ready()
        self._load_wardrobe_specialist_state_dicts()
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
            backend="ai_toolkit_exact_shared_plus" if self._pipeline is not None else None,
            lora_loaded=lora_loaded,
        )

    def wardrobe_status(self) -> WardrobeRuntimeStatus:
        return WardrobeRuntimeStatus(
            loaded=self._pipeline is not None and self._network is not None,
            backend="ai_toolkit_exact_shared_plus" if self._pipeline is not None else None,
            loras_loaded=bool(self._wardrobe_specialist_state_dicts),
        )

    def set_active_specialist(self, category: str) -> None:
        if not self._use_specialists:
            return
        if category not in self._enabled_specialists:
            raise TryonRuntimeError(f"Try-on specialist is not enabled: {category}")
        self._ensure_ready()
        self._load_specialist_state_dicts()
        if self._active_lora_namespace == "tryon" and self._active_lora_key == category:
            return
        if self._network is None:
            raise TryonRuntimeError("AI-Toolkit LoRA network is not initialized.")
        state = self._specialist_state_dicts[category]
        self._network.load_state_dict(state, strict=False)
        self._network._update_torch_multiplier()
        self._active_lora_namespace = "tryon"
        self._active_lora_key = category
        self._active_specialist = category
        logger.info("Switched shared Qwen LoRA to try-on '%s'", category)

    def run_extract(
        self,
        *,
        input_image_path: str,
        prompt: str,
        garment_type: str,
        output_path: str,
    ) -> WardrobeRunResult:
        self._ensure_ready()
        if (
            self._pipeline is None
            or self._network is None
            or self._toolkit is None
            or self._torch_module is None
        ):
            raise WardrobeRuntimeError("Shared Qwen wardrobe runtime is not loaded.")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        started_at = time.perf_counter()
        conf = self._toolkit["GenerateImageConfig"](
            prompt=str(prompt).strip(),
            width=wardrobe_constants.OUTPUT_WIDTH,
            height=wardrobe_constants.OUTPUT_HEIGHT,
            negative_prompt="",
            seed=wardrobe_constants.GENERATION_SEED,
            guidance_scale=wardrobe_constants.GENERATION_GUIDANCE_SCALE,
            guidance_rescale=wardrobe_constants.GENERATION_GUIDANCE_RESCALE,
            num_inference_steps=wardrobe_constants.GENERATION_STEPS,
            network_multiplier=wardrobe_constants.GENERATION_NETWORK_MULTIPLIER,
            output_path=str(output_file),
            output_ext="jpg",
            ctrl_img_1=str(input_image_path),
            do_cfg_norm=wardrobe_constants.GENERATION_DO_CFG_NORM,
        )

        with self._infer_lock, self._torch_module.inference_mode():
            self.set_active_wardrobe_specialist(garment_type)
            self._pipeline.generate_images(
                [conf],
                sampler=wardrobe_constants.GENERATION_SAMPLER,
            )

        if not output_file.exists():
            raise WardrobeGenerationError("No image was generated by the wardrobe runtime.")

        image = Image.open(output_file).convert("RGB")
        metadata = {
            "backend": "ai_toolkit_exact_shared_plus",
            "architecture": "qwen_image_edit_plus",
            "model_source": str(self._model_path),
            "ai_toolkit_root": str(self._aitk_root),
            "checkpoint_path": str(self._wardrobe_specialist_paths[garment_type]),
            "lora_namespace": "wardrobe",
            "lora_key": garment_type,
            "lora_rank": wardrobe_constants.LORA_RANK,
            "lora_alpha": wardrobe_constants.LORA_ALPHA,
            "network_multiplier": wardrobe_constants.GENERATION_NETWORK_MULTIPLIER,
            "guidance_scale": wardrobe_constants.GENERATION_GUIDANCE_SCALE,
            "guidance_rescale": wardrobe_constants.GENERATION_GUIDANCE_RESCALE,
            "do_cfg_norm": wardrobe_constants.GENERATION_DO_CFG_NORM,
            "sampler": wardrobe_constants.GENERATION_SAMPLER,
            "seed": wardrobe_constants.GENERATION_SEED,
            "steps": wardrobe_constants.GENERATION_STEPS,
            "control_order": {"ctrl_img_1": "garment_input"},
            "output_size": {
                "width": wardrobe_constants.OUTPUT_WIDTH,
                "height": wardrobe_constants.OUTPUT_HEIGHT,
            },
            "dtype": self._dtype_name,
            "device": self._device,
        }
        return WardrobeRunResult(
            image=image,
            metadata=metadata,
            wall_seconds=float(round(time.perf_counter() - started_at, 3)),
        )

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
            raise TryonRuntimeError("Shared Qwen try-on runtime is not loaded.")

        if self._use_specialists:
            if lora_key is None:
                raise TryonRuntimeError(
                    "lora_key is required when TRYON_USE_SPECIALISTS is enabled.",
                )
            active_checkpoint = str(self._specialist_paths[lora_key])
        else:
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
            if self._use_specialists:
                self.set_active_specialist(str(lora_key))
            else:
                self._load_checkpoint_if_needed()
                self._active_lora_namespace = "tryon"
                self._active_lora_key = "default"
            self._pipeline.generate_images(
                [conf],
                sampler=str(self._settings.tryon_sampler),
            )

        if not output_file.exists():
            raise TryonGenerationError("No image was generated by the try-on runtime.")

        image = Image.open(output_file).convert("RGB")
        metadata = {
            "backend": "ai_toolkit_exact_shared_plus",
            "architecture": "qwen_image_edit_plus",
            "model_source": str(self._model_path),
            "ai_toolkit_root": str(self._aitk_root),
            "checkpoint_path": active_checkpoint,
            "lora_namespace": "tryon",
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

    def set_active_wardrobe_specialist(self, category: str) -> None:
        if category not in WARDROBE_CATEGORIES:
            raise WardrobeRuntimeError(f"Unknown wardrobe specialist category: {category}")
        self._ensure_ready()
        self._load_wardrobe_specialist_state_dicts()
        if self._active_lora_namespace == "wardrobe" and self._active_lora_key == category:
            return
        if self._network is None:
            raise WardrobeRuntimeError("AI-Toolkit LoRA network is not initialized.")
        state = self._wardrobe_specialist_state_dicts[category]
        self._network.load_state_dict(state, strict=False)
        self._network._update_torch_multiplier()
        self._active_lora_namespace = "wardrobe"
        self._active_lora_key = category
        logger.info("Switched shared Qwen LoRA to wardrobe '%s'", category)

    def _load_wardrobe_specialist_state_dicts(self) -> None:
        if self._network is None:
            raise WardrobeRuntimeError("AI-Toolkit LoRA network is not initialized.")
        if self._wardrobe_specialist_state_dicts:
            return
        for category in WARDROBE_CATEGORIES:
            path = self._wardrobe_specialist_paths[category]
            if not path.exists():
                raise WardrobeRuntimeError(
                    f"Wardrobe specialist checkpoint missing for '{category}': {path}",
                )
            self._network.load_weights(str(path))
            snapshot = OrderedDict(
                (key, tensor.detach().clone())
                for key, tensor in self._network.state_dict().items()
            )
            self._wardrobe_specialist_state_dicts[category] = snapshot
            logger.info(
                "Cached wardrobe LoRA weights in shared Qwen runtime: %s -> %s",
                category,
                path,
            )
        first = WARDROBE_CATEGORIES[0]
        self._network.load_state_dict(self._wardrobe_specialist_state_dicts[first], strict=False)
        self._network._update_torch_multiplier()
        self._active_lora_namespace = "wardrobe"
        self._active_lora_key = first

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
            logger.info(
                "Cached try-on LoRA weights in shared Qwen runtime: %s -> %s",
                category,
                path,
            )
        first = (
            self._enabled_specialists[0]
            if self._enabled_specialists
            else SPECIALIST_CATEGORIES[0]
        )
        self._network.load_state_dict(self._specialist_state_dicts[first], strict=False)
        self._network._update_torch_multiplier()
        self._active_lora_namespace = "tryon"
        self._active_lora_key = first
        self._active_specialist = first
