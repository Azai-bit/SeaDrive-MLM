#!/usr/bin/env python3
"""Fine-tune Qwen3-VL on COLREG chat JSONL data and save a merged model."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import gc
import inspect
import io
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune Qwen3-VL with HF JSONL chat data."
    )
    parser.add_argument(
        "--model-path",
        default="/home/ubuntu/tze/LLModels/Qwen/Qwen3-VL-2B-Instruct",
        help="Base Qwen3-VL model directory.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="tmp/hf_success_dataset",
        help="Dataset directory containing train.jsonl and optional validation.jsonl.",
    )
    parser.add_argument(
        "--output-model-path",
        default="/home/ubuntu/tze/LLModels/Qwen/Qwen3-VL-2B-COLREG",
        help="Final merged model directory.",
    )
    parser.add_argument(
        "--work-dir",
        default="tmp/qwen3_vl_colreg_sft_work",
        help="Intermediate trainer/checkpoint directory.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="",
        help="Directory for metrics, log history, plots, and validation generations. Default: work-dir/training_results.",
    )
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of scheduler warmup steps. Use 0 to disable warmup.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target module names.",
    )
    parser.add_argument(
        "--no-merge-lora",
        action="store_true",
        help="Save only the LoRA adapter instead of merging it into a full model.",
    )
    parser.add_argument(
        "--merge-lora-device",
        choices=("cpu-reload", "cpu", "cuda", "auto"),
        default="cpu-reload",
        help="Device used for merge_and_unload. cpu-reload reloads base+adapter on CPU and avoids GPU use during merge.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing. Default: true.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-label-tokens",
        type=int,
        default=16,
        help="Fail fast if a sample has fewer supervised assistant tokens after masking.",
    )
    parser.add_argument(
        "--label-stats-log-steps",
        type=int,
        default=25,
        help="Print supervised-token statistics every N collator calls; <=0 disables it.",
    )
    parser.add_argument(
        "--label-preflight-samples",
        type=int,
        default=32,
        help="Scan the first N training rows for supervised-token counts before starting Trainer; <=0 disables it.",
    )
    parser.add_argument(
        "--image-max-side",
        type=int,
        default=384,
        help="Resize VL images so the longest side is at most this many pixels before processing; <=0 keeps original size.",
    )
    parser.add_argument(
        "--image-min-pixels",
        type=int,
        default=3136,
        help="Minimum VL image pixel budget passed to processors that support min_pixels; <=0 leaves the processor default.",
    )
    parser.add_argument(
        "--image-max-pixels",
        type=int,
        default=262144,
        help="Maximum VL image pixel budget passed to processors that support max_pixels; <=0 leaves the processor default.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Attention implementation passed to from_pretrained, e.g. sdpa, flash_attention_2, eager, or auto.",
    )
    parser.add_argument(
        "--eval-generate-samples",
        type=int,
        default=8,
        help="Generate predictions for the first N validation samples after training; <=0 disables it.",
    )
    parser.add_argument(
        "--eval-generate-max-new-tokens",
        type=int,
        default=180,
        help="Max new tokens for each validation sample generation.",
    )
    parser.add_argument(
        "--reasoning-max-words",
        type=int,
        default=24,
        help="Normalize assistant reasoning to at most this many whitespace-separated words.",
    )
    return parser.parse_args()


def optional_attn_kwargs(attn_implementation: str) -> dict[str, str]:
    value = str(attn_implementation or "").strip()
    if not value or value.lower() == "auto":
        return {}
    return {"attn_implementation": value}


def processor_pixel_kwargs(min_pixels: int, max_pixels: int) -> dict[str, int]:
    kwargs: dict[str, int] = {}
    if int(min_pixels) > 0:
        kwargs["min_pixels"] = int(min_pixels)
    if int(max_pixels) > 0:
        kwargs["max_pixels"] = int(max_pixels)
    return kwargs


def is_attn_implementation_error(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "attn_implementation" in message
        or "flash_attention" in message
        or ("sdpa" in message and ("support" in message or "available" in message))
    )


def from_pretrained_with_optional_attn(model_cls, model_path: str, kwargs: dict[str, Any]):
    try:
        return model_cls.from_pretrained(model_path, **kwargs)
    except Exception as e:
        if "attn_implementation" not in kwargs or not is_attn_implementation_error(e):
            raise
        print(
            f"Model loader rejected attn_implementation={kwargs['attn_implementation']!r}; "
            f"retrying with model default attention. error={e}",
            flush=True,
        )
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("attn_implementation", None)
        return model_cls.from_pretrained(model_path, **fallback_kwargs)


def load_model(model_path: str, attn_implementation: str):
    common_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto" if torch.cuda.is_available() else None,
        "low_cpu_mem_usage": True,
        **optional_attn_kwargs(attn_implementation),
    }
    common_kwargs = {k: v for k, v in common_kwargs.items() if v is not None}

    try:
        from transformers import AutoModelForImageTextToText

        return from_pretrained_with_optional_attn(
            AutoModelForImageTextToText,
            model_path,
            common_kwargs,
        )
    except Exception:
        from transformers import AutoModelForCausalLM

        return from_pretrained_with_optional_attn(
            AutoModelForCausalLM,
            model_path,
            common_kwargs,
        )


def load_model_on_cpu(model_path: str):
    common_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": None,
        "low_cpu_mem_usage": True,
    }
    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(model_path, **common_kwargs)
    except Exception:
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(model_path, **common_kwargs)


def load_processor_and_tokenizer(model_path: str, min_pixels: int, max_pixels: int):
    pixel_kwargs = processor_pixel_kwargs(min_pixels, max_pixels)
    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            **pixel_kwargs,
        )
    except TypeError as e:
        if not pixel_kwargs:
            raise
        print(
            f"Processor rejected pixel budget {pixel_kwargs}; "
            f"retrying with processor defaults. error={e}",
            flush=True,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        processor = None

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return processor, tokenizer


def normalize_content_for_template(content: Any) -> Any:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        normalized: list[Any] = []
        for item in content:
            if not isinstance(item, dict):
                normalized.append({"type": "text", "text": str(item)})
                continue
            item_type = item.get("type")
            if item_type == "text":
                normalized.append({"type": "text", "text": str(item.get("text", ""))})
            elif item_type == "image":
                copied = dict(item)
                copied.setdefault("type", "image")
                normalized.append(copied)
            elif item_type == "video":
                copied = dict(item)
                copied.setdefault("type", "video")
                normalized.append(copied)
            else:
                normalized.append(dict(item))
        return normalized
    return [{"type": "text", "text": str(content)}]


def normalize_messages_for_template(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        normalized.append(
            {
                "role": message.get("role", "user"),
                "content": normalize_content_for_template(message.get("content", "")),
            }
        )
    return normalized


def apply_chat_template(tokenizer, messages: list[dict[str, Any]], *, prompt: bool) -> str:
    template_owner = tokenizer
    if getattr(template_owner, "chat_template", None) or hasattr(
        template_owner, "apply_chat_template"
    ):
        return template_owner.apply_chat_template(
            normalize_messages_for_template(messages),
            tokenize=False,
            add_generation_prompt=prompt,
        )

    rendered: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = content_text(message.get("content", ""))
        rendered.append(f"{role}: {content}")
    if prompt:
        rendered.append("assistant:")
    return "\n".join(rendered)


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif item.get("type") == "image":
                    parts.append("<image>")
                elif item.get("type") == "video":
                    parts.append("<video>")
        return "\n".join(part for part in parts if part)
    return str(content)


COURSE_ACTIONS = {"KEEP_COURSE", "TURN_STARBOARD", "TURN_PORT"}
SPEED_ACTIONS = {"SLOW_DOWN", "SPEED_UP", "EMERGENCY_STOP"}


def truncate_words(text: str, max_words: int) -> str:
    text = " ".join(str(text or "").split())
    if max_words <= 0:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def parse_assistant_payload(raw_text: str) -> dict[str, Any] | None:
    raw_text = str(raw_text or "").strip()
    if not raw_text:
        return None
    try:
        value = json.loads(raw_text)
        return value if isinstance(value, dict) else None
    except Exception:
        pass
    try:
        value = ast.literal_eval(raw_text)
        return value if isinstance(value, dict) else None
    except Exception:
        pass
    return None


def normalize_colreg_answer(raw_content: Any, reasoning_max_words: int) -> str | None:
    raw_text = content_text(raw_content).strip()
    payload = parse_assistant_payload(raw_text)
    if payload is None:
        return None

    course_action = str(payload.get("course_action", "")).strip().upper()
    speed_action = str(payload.get("speed_action", "")).strip().upper()
    if course_action not in COURSE_ACTIONS or speed_action not in SPEED_ACTIONS:
        return None

    try:
        confidence = float(payload.get("confidence", 0.8))
    except Exception:
        confidence = 0.8
    confidence = max(0.0, min(1.0, confidence))
    reasoning = truncate_words(str(payload.get("reasoning", "COLREG decision")), reasoning_max_words)
    if not reasoning:
        reasoning = "COLREG decision"

    normalized = {
        "confidence": round(confidence, 3),
        "reasoning": reasoning,
        "course_action": course_action,
        "speed_action": speed_action,
    }
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def is_valid_colreg_row(row: dict[str, Any], reasoning_max_words: int) -> bool:
    try:
        messages = row_messages(row)
    except Exception:
        return False
    if not messages or messages[-1].get("role") != "assistant":
        return False
    return normalize_colreg_answer(messages[-1].get("content", ""), reasoning_max_words) is not None


def split_prompt_and_full_text(
    template_owner,
    messages: list[dict[str, Any]],
    *,
    reasoning_max_words: int,
) -> tuple[str, str]:
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Each row must end with an assistant message.")
    prompt_text = apply_chat_template(template_owner, messages[:-1], prompt=True)
    answer_text = normalize_colreg_answer(messages[-1].get("content", ""), reasoning_max_words)
    if answer_text is None:
        raise ValueError("Assistant message is not a valid COLREG JSON answer.")
    full_text = prompt_text + answer_text
    return prompt_text, full_text


def row_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    if "messages" in row:
        return row["messages"]
    elif {"instruction", "output"}.issubset(row):
        return [
            {"role": "user", "content": row["instruction"]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(row["output"])}],
            },
        ]
    raise ValueError("Dataset row must contain messages or instruction/output fields.")


def row_image_paths(row: dict[str, Any], dataset_dir: Path) -> list[Path | str]:
    paths: list[Path | str] = []
    for message in row_messages(row):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                raw_path = str(item.get("image") or item.get("path") or "").strip()
                if raw_path.startswith("data:image"):
                    paths.append(raw_path)
                    continue
                if raw_path.startswith("file://"):
                    raw_path = raw_path[7:]
                if not raw_path:
                    continue
                path = Path(raw_path)
                if not path.is_absolute():
                    path = dataset_dir / path
                paths.append(path)
    return paths


def resolve_image_ref(raw_ref: str, dataset_dir: Path) -> Path | str | None:
    raw_ref = str(raw_ref or "").strip()
    if raw_ref.startswith("data:image"):
        return raw_ref
    if raw_ref.startswith("file://"):
        raw_ref = raw_ref[7:]
    if not raw_ref:
        return None
    path = Path(raw_ref)
    if not path.is_absolute():
        path = dataset_dir / path
    return path


def row_video_paths(row: dict[str, Any], dataset_dir: Path) -> list[list[Path | str]]:
    videos: list[list[Path | str]] = []
    for message in row_messages(row):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not (isinstance(item, dict) and item.get("type") == "video"):
                continue
            raw_video = item.get("video", [])
            if isinstance(raw_video, str):
                raw_video = [raw_video]
            if not isinstance(raw_video, list):
                continue
            refs: list[Path | str] = []
            for raw_ref in raw_video:
                ref = resolve_image_ref(str(raw_ref), dataset_dir)
                if ref is not None:
                    refs.append(ref)
            if refs:
                videos.append(refs)
    return videos


def open_image_ref(image_ref: Path | str) -> Image.Image:
    if isinstance(image_ref, str) and image_ref.startswith("data:image"):
        try:
            _, payload = image_ref.split(",", 1)
        except ValueError as exc:
            raise ValueError("Invalid image data URL: missing comma") from exc
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    path = Path(image_ref)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    return Image.open(path).convert("RGB")


def resize_image_max_side(image: Image.Image, max_side: int) -> Image.Image:
    max_side = int(max_side)
    if max_side <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = float(max_side) / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
    return image.resize(new_size, resample=resample)


def open_images_with_sizes(
    paths: list[Path | str], max_side: int = 0
) -> tuple[list[Image.Image], list[tuple[int, int]], list[tuple[int, int]]]:
    images: list[Image.Image] = []
    original_sizes: list[tuple[int, int]] = []
    resized_sizes: list[tuple[int, int]] = []
    for path in paths:
        image = open_image_ref(path)
        original_sizes.append(image.size)
        image = resize_image_max_side(image, max_side)
        resized_sizes.append(image.size)
        images.append(image)
    return images, original_sizes, resized_sizes


def open_images(paths: list[Path | str], max_side: int = 0) -> list[Image.Image]:
    images, _, _ = open_images_with_sizes(paths, max_side)
    return images


class ColregDataCollator:
    def __init__(
        self,
        *,
        processor,
        tokenizer,
        dataset_dir: Path,
        max_seq_length: int,
        min_label_tokens: int,
        label_stats_log_steps: int,
        image_max_side: int,
        reasoning_max_words: int,
    ) -> None:
        self.processor = processor
        self.tokenizer = tokenizer
        self.dataset_dir = dataset_dir
        self.max_seq_length = max_seq_length
        self.min_label_tokens = max(0, int(min_label_tokens))
        self.label_stats_log_steps = int(label_stats_log_steps)
        self.image_max_side = int(image_max_side)
        self.reasoning_max_words = int(reasoning_max_words)
        self.template_owner = processor if processor is not None else tokenizer
        self._collator_calls = 0

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        messages_batch = [row_messages(feature) for feature in features]
        prompt_texts: list[str] = []
        full_texts: list[str] = []
        image_paths_batch: list[list[Path | str]] = []
        video_paths_batch: list[list[list[Path | str]]] = []
        original_sizes_batch: list[list[tuple[int, int]]] = []
        resized_sizes_batch: list[list[tuple[int, int]]] = []

        for feature, messages in zip(features, messages_batch):
            prompt_text, full_text = split_prompt_and_full_text(
                self.template_owner,
                messages,
                reasoning_max_words=self.reasoning_max_words,
            )
            if self.tokenizer.eos_token and not full_text.endswith(self.tokenizer.eos_token):
                full_text += self.tokenizer.eos_token
            prompt_texts.append(prompt_text)
            full_texts.append(full_text)
            image_paths_batch.append(row_image_paths(feature, self.dataset_dir))
            video_paths_batch.append(row_video_paths(feature, self.dataset_dir))

        image_counts = [len(paths) for paths in image_paths_batch]
        video_counts = [len(videos) for videos in video_paths_batch]
        has_images = any(count > 0 for count in image_counts)
        has_videos = any(count > 0 for count in video_counts)
        if has_images and has_videos:
            raise ValueError("Do not mix image and video samples in the same batch.")
        if has_images and not all(count == 1 for count in image_counts):
            raise ValueError(
                "This trainer expects exactly one image per VL sample. "
                "Regenerate the dataset with --include-images --copy-images --require-images."
            )
        if has_videos and not all(count == 1 for count in video_counts):
            raise ValueError(
                "This trainer expects exactly one video per VL sample. "
                "Regenerate the dataset with --include-images --require-images."
            )

        if has_videos:
            if self.processor is None:
                raise RuntimeError("A processor is required for VL samples with videos.")
            videos = []
            for video_refs in video_paths_batch:
                frames, original_sizes, resized_sizes = open_images_with_sizes(
                    video_refs[0], self.image_max_side
                )
                videos.append(frames)
                original_sizes_batch.append(original_sizes)
                resized_sizes_batch.append(resized_sizes)
            model_inputs = self.processor(
                text=full_texts,
                videos=videos,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            prompt_lens: list[int] = []
            for prompt_text, video in zip(prompt_texts, videos):
                prompt_inputs = self.processor(
                    text=[prompt_text],
                    videos=[video],
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                )
                prompt_lens.append(int(prompt_inputs["input_ids"].shape[1]))
        elif has_images:
            if self.processor is None:
                raise RuntimeError("A processor is required for VL samples with images.")
            images = []
            for paths in image_paths_batch:
                row_images, original_sizes, resized_sizes = open_images_with_sizes(
                    paths, self.image_max_side
                )
                images.append(row_images[0])
                original_sizes_batch.append(original_sizes)
                resized_sizes_batch.append(resized_sizes)
            model_inputs = self.processor(
                text=full_texts,
                images=images,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            prompt_lens: list[int] = []
            for prompt_text, image in zip(prompt_texts, images):
                prompt_inputs = self.processor(
                    text=[prompt_text],
                    images=[image],
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                )
                prompt_lens.append(int(prompt_inputs["input_ids"].shape[1]))
        else:
            original_sizes_batch = [[] for _ in features]
            resized_sizes_batch = [[] for _ in features]
            model_inputs = self.tokenizer(
                full_texts,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            prompt_lens = [
                len(
                    self.tokenizer(
                        prompt_text,
                        truncation=True,
                        max_length=self.max_seq_length,
                    )["input_ids"]
                )
                for prompt_text in prompt_texts
            ]

        labels = model_inputs["input_ids"].clone()
        labels[model_inputs["attention_mask"] == 0] = -100
        label_token_counts: list[int] = []
        full_lens = [
            int(torch.count_nonzero(model_inputs["attention_mask"][i]).item())
            for i in range(labels.shape[0])
        ]
        for row_index, prompt_len in enumerate(prompt_lens):
            prompt_len = min(prompt_len, labels.shape[1])
            labels[row_index, :prompt_len] = -100
            diag = (
                f"row={row_index} full_len={full_lens[row_index]} "
                f"prompt_len={prompt_len} max_seq_length={self.max_seq_length} "
                f"image_original={original_sizes_batch[row_index]} "
                f"image_resized={resized_sizes_batch[row_index]} "
                f"image_max_side={self.image_max_side}"
            )
            if torch.all(labels[row_index] == -100):
                raise ValueError(
                    "No supervised assistant tokens remain after truncation. "
                    "Increase --max-seq-length, reduce image resolution/pixels, or shorten prompts. "
                    + diag
                )
            label_count = int(torch.count_nonzero(labels[row_index] != -100).item())
            label_token_counts.append(label_count)
            if label_count < self.min_label_tokens:
                preview_ids = labels[row_index][labels[row_index] != -100][
                    :80
                ].detach().cpu().tolist()
                preview = self.tokenizer.decode(preview_ids, skip_special_tokens=False)
                raise ValueError(
                    "Too few supervised assistant tokens after masking: "
                    f"{label_count} < {self.min_label_tokens}. "
                    "Increase --max-seq-length, lower --image-max-side, or inspect the chat template. "
                    + diag
                    + f" label_preview={preview[:300]!r}"
                )
        self._collator_calls += 1
        if self.label_stats_log_steps > 0 and (
            self._collator_calls == 1
            or self._collator_calls % self.label_stats_log_steps == 0
        ):
            preview_ids = labels[0][labels[0] != -100][:80].detach().cpu().tolist()
            preview = self.tokenizer.decode(preview_ids, skip_special_tokens=False)
            print(
                "[label_stats] "
                f"call={self._collator_calls} "
                f"label_tokens=min/avg/max "
                f"{min(label_token_counts)}/{sum(label_token_counts) / len(label_token_counts):.1f}/{max(label_token_counts)} "
                f"preview={preview[:300]!r}",
                flush=True,
            )
        model_inputs["labels"] = labels
        return model_inputs


def load_jsonl_dataset(dataset_dir: Path):
    data_files = {"train": str(dataset_dir / "train.jsonl")}
    validation_path = dataset_dir / "validation.jsonl"
    if validation_path.is_file():
        data_files["validation"] = str(validation_path)
    return load_dataset("json", data_files=data_files)


def filter_colreg_dataset(raw_dataset, reasoning_max_words: int):
    filtered = {}
    for split_name, dataset in raw_dataset.items():
        before = len(dataset)
        kept = dataset.filter(
            lambda row: is_valid_colreg_row(row, reasoning_max_words),
            desc=f"Filtering {split_name} COLREG labels",
            load_from_cache_file=False,
        )
        after = len(kept)
        dropped = before - after
        print(
            f"[dataset_filter] split={split_name} kept={after}/{before} dropped={dropped}",
            flush=True,
        )
        if after <= 0:
            raise ValueError(f"No valid COLREG samples remain in split {split_name}.")
        filtered[split_name] = kept
    return filtered


def run_label_preflight(collator: ColregDataCollator, dataset, sample_count: int) -> None:
    if sample_count <= 0:
        return
    total = min(int(sample_count), len(dataset))
    if total <= 0:
        return
    counts: list[int] = []
    for idx in range(total):
        try:
            batch = collator([dataset[idx]])
        except Exception as e:
            raise RuntimeError(f"Label preflight failed at dataset index {idx}: {e}") from e
        labels = batch["labels"]
        counts.append(int(torch.count_nonzero(labels[0] != -100).item()))
    print(
        "[label_preflight] "
        f"samples={total} label_tokens=min/avg/max "
        f"{min(counts)}/{sum(counts) / len(counts):.1f}/{max(counts)}",
        flush=True,
    )


def build_training_args(args: argparse.Namespace) -> TrainingArguments:
    kwargs = {
        "output_dir": args.work_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "fp16": bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        "optim": "adamw_torch",
        "report_to": "none",
        "remove_unused_columns": False,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "torch_empty_cache_steps": 10,
        "dataloader_num_workers": 0,
        "seed": args.seed,
    }
    signature = inspect.signature(TrainingArguments)
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in signature.parameters})


def save_processor(processor, tokenizer, output_path: Path) -> None:
    if processor is not None:
        processor.save_pretrained(output_path)
    else:
        tokenizer.save_pretrained(output_path)


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=json_default)
        f.write("\n")


def write_log_history_csv(path: Path, log_history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    preferred = ["step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm"]
    for key in preferred:
        if any(key in row for row in log_history):
            fieldnames.append(key)
            seen.add(key)
    for row in log_history:
        for key in sorted(row):
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in log_history:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_training_curves(log_history: list[dict[str, Any]], artifact_dir: Path) -> None:
    import contextlib
    import io

    import_stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(import_stderr):
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
    except Exception as e:
        detail = import_stderr.getvalue().strip()
        lines = [f"matplotlib unavailable: {type(e).__name__}: {e}"]
        if detail:
            lines.extend(["", "stderr during matplotlib import:", detail])
        (artifact_dir / "plots_unavailable.txt").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        print(
            f"Skipping training curve plots; see {artifact_dir / 'plots_unavailable.txt'}",
            flush=True,
        )
        return

    train_loss = [
        (row.get("step"), row.get("loss"))
        for row in log_history
        if row.get("step") is not None and row.get("loss") is not None
    ]
    eval_loss = [
        (row.get("step"), row.get("eval_loss"))
        for row in log_history
        if row.get("step") is not None and row.get("eval_loss") is not None
    ]
    if train_loss or eval_loss:
        plt.figure(figsize=(8, 4.8))
        if train_loss:
            xs, ys = zip(*train_loss)
            plt.plot(xs, ys, label="train_loss", linewidth=1.8)
        if eval_loss:
            xs, ys = zip(*eval_loss)
            plt.plot(xs, ys, label="eval_loss", marker="o", linewidth=1.8)
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Training Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(artifact_dir / "loss_curve.png", dpi=160)
        plt.close()

    lr_points = [
        (row.get("step"), row.get("learning_rate"))
        for row in log_history
        if row.get("step") is not None and row.get("learning_rate") is not None
    ]
    if lr_points:
        xs, ys = zip(*lr_points)
        plt.figure(figsize=(8, 4.8))
        plt.plot(xs, ys, label="learning_rate", linewidth=1.8)
        plt.xlabel("step")
        plt.ylabel("learning rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(artifact_dir / "learning_rate_curve.png", dpi=160)
        plt.close()


def write_training_summary(
    *,
    artifact_dir: Path,
    train_metrics: dict[str, Any],
    eval_metrics: dict[str, Any] | None,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Qwen3-VL COLREG Fine-tuning Results",
        "",
        "## Run",
        "",
        f"- base_model: `{args.model_path}`",
        f"- dataset_dir: `{args.dataset_dir}`",
        f"- output_model_path: `{args.output_model_path}`",
        f"- work_dir: `{args.work_dir}`",
        f"- learning_rate: `{args.learning_rate}`",
        f"- epochs: `{args.num_train_epochs}`",
        f"- batch/grad_accum: `{args.per_device_train_batch_size}/{args.gradient_accumulation_steps}`",
        "",
        "## Metrics",
        "",
    ]
    for key in sorted(train_metrics):
        lines.append(f"- train.{key}: `{train_metrics[key]}`")
    if eval_metrics:
        for key in sorted(eval_metrics):
            lines.append(f"- eval.{key}: `{eval_metrics[key]}`")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `train_metrics.json`",
            "- `eval_metrics.json`",
            "- `trainer_state.json`",
            "- `log_history.json`",
            "- `log_history.csv`",
            "- `loss_curve.png`",
            "- `learning_rate_curve.png`",
            "- `validation_generations.jsonl`",
        ]
    )
    (artifact_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def tensor_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_tensors_to_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def generate_validation_samples(
    *,
    model,
    processor,
    tokenizer,
    eval_dataset,
    dataset_dir: Path,
    artifact_dir: Path,
    sample_count: int,
    max_new_tokens: int,
    image_max_side: int,
    reasoning_max_words: int,
) -> None:
    if eval_dataset is None or sample_count <= 0:
        return
    if processor is None:
        raise RuntimeError("Validation generation requires a processor for VL samples.")

    output_path = artifact_dir / "validation_generations.jsonl"
    template_owner = processor if processor is not None else tokenizer
    device = tensor_device(model)
    was_training = bool(getattr(model, "training", False))
    model.eval()
    rows_written = 0

    with output_path.open("w", encoding="utf-8") as f:
        for row_index in range(min(sample_count, len(eval_dataset))):
            row = eval_dataset[row_index]
            messages = row_messages(row)
            prompt_text = apply_chat_template(
                template_owner,
                messages[:-1],
                prompt=True,
            )
            expected = normalize_colreg_answer(
                messages[-1].get("content", ""),
                reasoning_max_words,
            )
            image_paths = row_image_paths(row, dataset_dir)
            video_paths = row_video_paths(row, dataset_dir)
            if video_paths:
                if len(video_paths) != 1:
                    raise ValueError(
                        f"Validation row {row_index} must have exactly one video, got {len(video_paths)}."
                    )
                video = open_images(video_paths[0], image_max_side)
                inputs = processor(
                    text=[prompt_text],
                    videos=[video],
                    return_tensors="pt",
                )
                image_record = f"video_frames={len(video_paths[0])}"
            elif len(image_paths) != 1:
                raise ValueError(
                    f"Validation row {row_index} must have exactly one image, got {len(image_paths)}."
                )
            else:
                image = open_images(image_paths, image_max_side)[0]
                inputs = processor(
                    text=[prompt_text],
                    images=[image],
                    return_tensors="pt",
                )
                image_record = (
                    f"data:image/*;base64,<len={len(image_paths[0])}>"
                    if isinstance(image_paths[0], str)
                    and image_paths[0].startswith("data:image")
                    else str(image_paths[0])
                )
            input_len = int(inputs["input_ids"].shape[1])
            inputs = move_tensors_to_device(inputs, device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0][input_len:]
            prediction = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
            record = {
                "row_index": row_index,
                "image": image_record,
                "expected": expected,
                "prediction": prediction,
                "prediction_empty": not bool(prediction.strip()),
            }
            f.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")
            rows_written += 1

    if was_training:
        model.train()
    print(f"Saved validation generations: {output_path} ({rows_written} rows)")


def save_training_artifacts(
    *,
    trainer: Trainer,
    artifact_dir: Path,
    train_metrics: dict[str, Any],
    eval_metrics: dict[str, Any] | None,
    args: argparse.Namespace,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_history = list(trainer.state.log_history)
    write_json(artifact_dir / "train_metrics.json", train_metrics)
    if eval_metrics is not None:
        write_json(artifact_dir / "eval_metrics.json", eval_metrics)
    if hasattr(trainer.state, "to_dict"):
        trainer_state = trainer.state.to_dict()
    elif hasattr(trainer.state, "to_json_string"):
        trainer_state = json.loads(trainer.state.to_json_string())
    else:
        trainer_state = dict(vars(trainer.state))
    write_json(artifact_dir / "trainer_state.json", trainer_state)
    write_json(artifact_dir / "log_history.json", log_history)
    write_log_history_csv(artifact_dir / "log_history.csv", log_history)
    write_json(
        artifact_dir / "run_config.json",
        {
            "model_path": args.model_path,
            "dataset_dir": args.dataset_dir,
            "output_model_path": args.output_model_path,
            "work_dir": args.work_dir,
            "max_seq_length": args.max_seq_length,
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "image_max_side": args.image_max_side,
            "image_min_pixels": args.image_min_pixels,
            "image_max_pixels": args.image_max_pixels,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": args.lora_target_modules,
            "merge_lora_device": args.merge_lora_device,
            "attn_implementation": args.attn_implementation,
            "seed": args.seed,
            "min_label_tokens": args.min_label_tokens,
            "label_preflight_samples": args.label_preflight_samples,
            "reasoning_max_words": args.reasoning_max_words,
        },
    )
    plot_training_curves(log_history, artifact_dir)
    write_training_summary(
        artifact_dir=artifact_dir,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        args=args,
    )
    print(f"Saved training artifacts: {artifact_dir}")


def copy_artifacts_to_model_dir(artifact_dir: Path, output_model_path: Path) -> None:
    if not artifact_dir.exists():
        return
    target = output_model_path / "training_results"
    if artifact_dir.resolve() == target.resolve():
        return
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(artifact_dir, target)
    print(f"Copied training artifacts to: {target}")


def release_runtime_memory(label: str) -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    allocated_gib = torch.cuda.memory_allocated() / (1024**3)
    reserved_gib = torch.cuda.memory_reserved() / (1024**3)
    print(
        f"[cuda_memory] {label}: allocated={allocated_gib:.2f}GiB "
        f"reserved={reserved_gib:.2f}GiB",
        flush=True,
    )


def clear_trainer_runtime_memory(trainer: Trainer) -> None:
    trainer.optimizer = None
    trainer.lr_scheduler = None
    if hasattr(trainer, "accelerator"):
        try:
            trainer.accelerator.free_memory()
        except Exception as e:
            print(f"Trainer accelerator free_memory failed: {e}", flush=True)
    if hasattr(trainer, "_train_batch_size"):
        trainer._train_batch_size = None
    release_runtime_memory("after clearing trainer runtime")


def merge_lora_model_for_save(
    model,
    *,
    merge_device: str,
    base_model_path: Path,
    adapter_path: Path,
):
    merge_device = str(merge_device or "cpu").lower()
    if merge_device == "auto":
        merge_device = "cuda" if torch.cuda.is_available() else "cpu"
    if merge_device == "cpu-reload":
        print("Reloading base model and LoRA adapter on CPU before merge_and_unload.")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        base_model = load_model_on_cpu(str(base_model_path))
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=False)
        merged = peft_model.merge_and_unload()
        del peft_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return merged
    if merge_device == "cpu":
        print("Moving LoRA model to CPU before merge_and_unload to avoid GPU OOM.")
        model = model.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("Merging LoRA model on CUDA; this may require several GiB of free GPU memory.")
    return model.merge_and_unload()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model_path)
    dataset_dir = Path(args.dataset_dir)
    output_model_path = Path(args.output_model_path)
    work_dir = Path(args.work_dir)
    adapter_path = work_dir / "lora_adapter"
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else work_dir / "training_results"

    if not model_path.is_dir():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not (dataset_dir / "train.jsonl").is_file():
        raise FileNotFoundError(f"Missing dataset file: {dataset_dir / 'train.jsonl'}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    processor, tokenizer = load_processor_and_tokenizer(
        str(model_path),
        args.image_min_pixels,
        args.image_max_pixels,
    )
    model = load_model(str(model_path), args.attn_implementation)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        gradient_checkpointing_kwargs = {}
        try:
            signature = inspect.signature(model.gradient_checkpointing_enable)
            if "gradient_checkpointing_kwargs" in signature.parameters:
                gradient_checkpointing_kwargs["gradient_checkpointing_kwargs"] = {
                    "use_reentrant": False,
                }
        except Exception:
            pass
        model.gradient_checkpointing_enable(**gradient_checkpointing_kwargs)
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_targets = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    raw_dataset = filter_colreg_dataset(
        load_jsonl_dataset(dataset_dir),
        args.reasoning_max_words,
    )
    train_dataset = raw_dataset["train"]
    eval_dataset = raw_dataset["validation"] if "validation" in raw_dataset else None
    data_collator = ColregDataCollator(
        processor=processor,
        tokenizer=tokenizer,
        dataset_dir=dataset_dir,
        max_seq_length=args.max_seq_length,
        min_label_tokens=args.min_label_tokens,
        label_stats_log_steps=args.label_stats_log_steps,
        image_max_side=args.image_max_side,
        reasoning_max_words=args.reasoning_max_words,
    )
    run_label_preflight(data_collator, train_dataset, args.label_preflight_samples)

    trainer = Trainer(
        model=model,
        args=build_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    train_output = trainer.train()
    train_metrics = dict(train_output.metrics)
    train_metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    eval_metrics = None
    if eval_dataset is not None:
        eval_metrics = dict(trainer.evaluate(eval_dataset=eval_dataset))
        eval_metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    save_training_artifacts(
        trainer=trainer,
        artifact_dir=artifact_dir,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        args=args,
    )

    try:
        generate_validation_samples(
            model=trainer.model,
            processor=processor,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            dataset_dir=dataset_dir,
            artifact_dir=artifact_dir,
            sample_count=args.eval_generate_samples,
            max_new_tokens=args.eval_generate_max_new_tokens,
            image_max_side=args.image_max_side,
            reasoning_max_words=args.reasoning_max_words,
        )
    except Exception as e:
        error_path = artifact_dir / "validation_generation_error.txt"
        error_path.write_text(str(e) + "\n", encoding="utf-8")
        print(f"Validation generation failed; wrote: {error_path}")

    adapter_path.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_path)
    save_processor(processor, tokenizer, adapter_path)

    if args.no_merge_lora:
        output_model_path.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(output_model_path)
        save_processor(processor, tokenizer, output_model_path)
        copy_artifacts_to_model_dir(artifact_dir, output_model_path)
        print(f"Saved LoRA adapter to: {output_model_path}")
        return 0

    clear_trainer_runtime_memory(trainer)
    if args.merge_lora_device == "cpu-reload":
        print("Releasing CUDA training model before CPU reload merge.")
        trainer_model = trainer.model
        trainer.model = None
        if hasattr(trainer, "model_wrapped"):
            trainer.model_wrapped = None
        del trainer_model
        del trainer
        del model
        model_for_merge = None
        release_runtime_memory("after dropping CUDA training model")
    else:
        model_for_merge = trainer.model
        del trainer
        del model
        release_runtime_memory("before in-process LoRA merge")
    merged_model = merge_lora_model_for_save(
        model_for_merge,
        merge_device=args.merge_lora_device,
        base_model_path=model_path,
        adapter_path=adapter_path,
    )
    output_model_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(
        output_model_path,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    save_processor(processor, tokenizer, output_model_path)
    copy_artifacts_to_model_dir(artifact_dir, output_model_path)
    print(f"Saved merged fine-tuned model to: {output_model_path}")
    print(f"Saved intermediate LoRA adapter to: {adapter_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
