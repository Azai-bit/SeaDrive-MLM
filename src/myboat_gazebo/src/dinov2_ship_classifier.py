#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import json

import cv2
import numpy as np
from PIL import Image


class DinoV2ShipClassifier:
    VESSEL_TYPES = ("Lifeboat", "USV", "Fishing")

    def __init__(
        self,
        model_name="facebook/dinov2-base",
        prototypes_dir="",
        device="",
        min_similarity=0.34,
        similarity_margin=0.035,
        heuristic_only=False,
    ):
        self.model_name = str(model_name or "facebook/dinov2-base").strip()
        self.prototypes_dir = os.path.expanduser(str(prototypes_dir or "").strip())
        self.device = str(device or "").strip().lower()
        self.min_similarity = float(min_similarity)
        self.similarity_margin = float(similarity_margin)
        self.heuristic_only = bool(heuristic_only)

        self._torch = None
        self._processor = None
        self._model = None
        self._prototype_vectors = {}
        self._prototype_color_vectors = {}
        self._prototype_files = {}
        self._runtime_backend = "heuristic"
        self._status = "not_initialized"

    @property
    def runtime_backend(self):
        return self._runtime_backend

    @property
    def status(self):
        return self._status

    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    @staticmethod
    def _l2_normalize(vec):
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-8:
            return arr
        return arr / norm

    @staticmethod
    def _safe_float(v, default_v=None):
        try:
            return float(v)
        except Exception:
            return default_v

    @staticmethod
    def _pil_to_rgb(image):
        if image is None:
            raise ValueError("empty_image")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        arr = np.asarray(image)
        if arr.ndim != 3:
            raise ValueError("invalid_image_ndim")
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return Image.fromarray(arr.astype(np.uint8), mode="RGB")

    def _list_prototype_images(self, vessel_type):
        base = os.path.join(self.prototypes_dir, vessel_type)
        if not os.path.isdir(base):
            return []
        out = []
        for name in sorted(os.listdir(base)):
            lower = name.lower()
            if lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                out.append(os.path.join(base, name))
        return out

    def _ensure_color_prototypes(self):
        if self._prototype_color_vectors:
            return bool(self._prototype_color_vectors)
        prototype_vectors = {}
        prototype_files = {}
        for vessel_type in self.VESSEL_TYPES:
            files = self._list_prototype_images(vessel_type)
            if not files:
                continue
            feats = []
            for path in files:
                with Image.open(path) as img:
                    feats.append(self._hist_feature(img.convert("RGB")))
            if feats:
                prototype_vectors[vessel_type] = self._l2_normalize(np.mean(np.stack(feats, axis=0), axis=0))
                prototype_files[vessel_type] = files
        if prototype_vectors:
            self._prototype_color_vectors = prototype_vectors
            if not self._prototype_files:
                self._prototype_files = prototype_files
        return bool(self._prototype_color_vectors)

    def _lazy_init_dino(self):
        if self.heuristic_only:
            self._runtime_backend = "dinov2"
            self._status = "invalid_config:heuristic_only_not_supported"
            return False
        if self._model is not None and self._prototype_vectors:
            self._runtime_backend = "dinov2"
            self._status = "ready"
            return True
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except Exception as exc:
            self._runtime_backend = "dinov2"
            self._status = "import_failed:%s" % str(exc)
            return False

        try:
            if self.device in ("cpu", "cuda"):
                device = self.device
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = AutoImageProcessor.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            model = model.to(device)
            model.eval()
        except Exception as exc:
            self._runtime_backend = "dinov2"
            self._status = "model_load_failed:%s" % str(exc)
            return False

        prototype_vectors = {}
        prototype_files = {}
        try:
            for vessel_type in self.VESSEL_TYPES:
                files = self._list_prototype_images(vessel_type)
                if not files:
                    continue
                vectors = []
                for path in files:
                    with Image.open(path) as img:
                        vec = self._embed_with_model(img.convert("RGB"), torch, processor, model, device)
                    vectors.append(vec)
                if vectors:
                    mean_vec = self._l2_normalize(np.mean(np.stack(vectors, axis=0), axis=0))
                    prototype_vectors[vessel_type] = mean_vec
                    prototype_files[vessel_type] = files
        except Exception as exc:
            self._runtime_backend = "dinov2"
            self._status = "prototype_load_failed:%s" % str(exc)
            return False

        if len(prototype_vectors) < len(self.VESSEL_TYPES):
            self._runtime_backend = "dinov2"
            self._status = "prototype_missing"
            return False

        self._torch = torch
        self._processor = processor
        self._model = model
        self._prototype_vectors = prototype_vectors
        self._prototype_files = prototype_files
        self.device = device
        self._runtime_backend = "dinov2"
        self._status = "ready"
        return True

    @staticmethod
    def _embed_with_model(image, torch, processor, model, device):
        inputs = processor(images=image, return_tensors="pt")
        for key, value in inputs.items():
            inputs[key] = value.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state
            embedding = last_hidden[:, 0]
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        return embedding[0].detach().cpu().numpy().astype(np.float32)

    def _embed_image(self, image):
        if not self._lazy_init_dino():
            raise RuntimeError(self._status)
        return self._embed_with_model(
            image,
            self._torch,
            self._processor,
            self._model,
            self.device,
        )

    @staticmethod
    def _region_x_hint(target):
        x01 = DinoV2ShipClassifier._safe_float(target.get("image_x_hint_01"), None)
        if x01 is not None:
            return DinoV2ShipClassifier._clamp(x01, 0.05, 0.95)
        hint = str(target.get("image_region_hint", "center") or "center").strip().lower()
        if hint == "far_left":
            return 0.08
        if hint == "left":
            return 0.26
        if hint == "right":
            return 0.74
        if hint == "far_right":
            return 0.92
        return 0.5

    def _candidate_crops(self, rgb_image, target):
        width, height = rgb_image.size
        x_hint = self._region_x_hint(target)
        distance_m = self._safe_float(target.get("distance_m"), 20.0)
        size_proxy = max(
            self._safe_float(target.get("size_x_m"), 0.0) or 0.0,
            self._safe_float(target.get("size_y_m"), 0.0) or 0.0,
            self._safe_float(target.get("size_z_m"), 0.0) or 0.0,
            self._safe_float(target.get("size"), 0.0) or 0.0,
        )

        width_frac = 0.09 + max(0.0, 20.0 - distance_m) * 0.005 + min(8.0, size_proxy) * 0.008
        height_frac = 0.08 + max(0.0, 20.0 - distance_m) * 0.004 + min(8.0, size_proxy) * 0.007
        width_frac = self._clamp(width_frac, 0.08, 0.30)
        height_frac = self._clamp(height_frac, 0.08, 0.26)

        y_center = int(height * 0.21)
        if distance_m is not None and distance_m < 10.0:
            y_center = int(height * 0.19)
        elif distance_m is not None and distance_m > 24.0:
            y_center = int(height * 0.23)

        base_w = width * width_frac
        base_h = height * height_frac
        shifts = (0.0, -0.035, 0.035)
        scales = (0.85, 1.0, 1.25)
        out = []
        for shift in shifts:
            for scale in scales:
                cx = int(width * self._clamp(x_hint + shift, 0.05, 0.95))
                crop_w = int(base_w * scale)
                crop_h = int(base_h * scale)
                x0 = max(0, min(width - 1, cx - crop_w // 2))
                y0 = max(0, min(height - 1, y_center - crop_h // 2))
                x1 = max(x0 + 4, min(width, x0 + crop_w))
                y1 = max(y0 + 4, min(height, y0 + crop_h))
                crop = rgb_image.crop((x0, y0, x1, y1)).convert("RGB")
                center_x01 = 0.5 * (float(x0 + x1) / max(1.0, float(width)))
                alignment = 1.0 - min(1.0, abs(center_x01 - x_hint) / 0.5)
                out.append(
                    {
                        "crop": crop,
                        "box": [int(x0), int(y0), int(x1), int(y1)],
                        "alignment": round(self._clamp(alignment, 0.0, 1.0), 3),
                    }
                )
        return out

    @staticmethod
    def _crop_color_features(crop):
        arr = np.asarray(crop, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[0] < 2 or arr.shape[1] < 2:
            return {"red": 0.0, "yellow": 0.0, "blue": 0.0, "valid": 0.0}
        focus = arr[: max(2, int(arr.shape[0] * 0.8)), :, :]
        hsv = cv2.cvtColor(focus, cv2.COLOR_RGB2HSV)
        sat_mask = hsv[:, :, 1] >= 40
        val_mask = hsv[:, :, 2] >= 35
        mask = sat_mask & val_mask
        denom = float(np.count_nonzero(mask))
        if denom < 8.0:
            denom = float(mask.size)
            mask = np.ones(mask.shape, dtype=bool)
        red_mask = ((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 168)) & mask
        yellow_mask = (hsv[:, :, 0] >= 16) & (hsv[:, :, 0] <= 40) & mask
        blue_mask = (hsv[:, :, 0] >= 92) & (hsv[:, :, 0] <= 138) & mask
        return {
            "red": float(np.count_nonzero(red_mask)) / max(1.0, denom),
            "yellow": float(np.count_nonzero(yellow_mask)) / max(1.0, denom),
            "blue": float(np.count_nonzero(blue_mask)) / max(1.0, denom),
            "valid": min(1.0, denom / max(64.0, 0.25 * float(mask.size))),
        }

    @staticmethod
    def _hist_feature(crop):
        arr = np.asarray(crop, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[0] < 2 or arr.shape[1] < 2:
            return np.zeros((16 * 4 * 4,), dtype=np.float32)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256])
        hist = hist.reshape(-1).astype(np.float32)
        return DinoV2ShipClassifier._l2_normalize(hist)

    def _heuristic_classify(self, crop, target, alignment):
        colors = self._crop_color_features(crop)
        size_x = self._safe_float(target.get("size_x_m"), 0.0) or 0.0
        size_y = self._safe_float(target.get("size_y_m"), 0.0) or 0.0
        size_z = self._safe_float(target.get("size_z_m"), 0.0) or 0.0
        size_proxy = max(size_x, size_y, size_z, self._safe_float(target.get("size"), 0.0) or 0.0)
        point_count = self._safe_float(target.get("point_count"), 0.0) or 0.0

        lifeboat_score = 0.70 * colors["red"] + 0.20 * max(0.0, 2.2 - size_proxy) / 2.2 + 0.10 * alignment
        usv_score = 0.65 * colors["yellow"] + 0.15 * max(0.0, 4.0 - abs(size_proxy - 2.5)) / 4.0 + 0.20 * alignment
        fishing_score = 0.58 * colors["blue"] + 0.24 * min(1.0, size_proxy / 6.5) + 0.10 * min(1.0, point_count / 40.0) + 0.08 * alignment

        score_map = {
            "Lifeboat": lifeboat_score,
            "USV": usv_score,
            "Fishing": fishing_score,
        }
        sorted_items = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
        best_label, best_score = sorted_items[0]
        second_score = sorted_items[1][1]
        confidence = self._clamp(0.35 + 0.55 * best_score + 0.10 * colors["valid"], 0.0, 1.0)
        if best_score < 0.16 or (best_score - second_score) < 0.03:
            best_label = "Unknown"
            confidence = self._clamp(0.20 + 0.45 * best_score, 0.0, 0.7)
        return {
            "label": best_label,
            "confidence": round(confidence, 3),
            "margin": round(float(best_score - second_score), 3),
            "score_map": {k: round(float(v), 3) for k, v in score_map.items()},
            "backend": "heuristic",
            "colors": {k: round(float(v), 3) for k, v in colors.items()},
        }

    def _template_fallback_classify(self, crop, target, alignment):
        heuristic = self._heuristic_classify(crop, target, alignment)
        if not self._ensure_color_prototypes():
            return heuristic
        hist = self._hist_feature(crop)
        colors = heuristic.get("colors", {}) or {}
        size_proxy = max(
            self._safe_float(target.get("size_x_m"), 0.0) or 0.0,
            self._safe_float(target.get("size_y_m"), 0.0) or 0.0,
            self._safe_float(target.get("size_z_m"), 0.0) or 0.0,
            self._safe_float(target.get("size"), 0.0) or 0.0,
        )
        x_hint = self._region_x_hint(target)
        score_map = {}
        for vessel_type, proto in self._prototype_color_vectors.items():
            score_map[vessel_type] = float(np.dot(hist, proto))
        merged = {}
        heuristic_scores = heuristic.get("score_map", {}) or {}
        for vessel_type in self.VESSEL_TYPES:
            proto_score = score_map.get(vessel_type, 0.0)
            heur_score = float(heuristic_scores.get(vessel_type, 0.0))
            merged[vessel_type] = 0.52 * proto_score + 0.48 * heur_score

        red = float(colors.get("red", 0.0))
        yellow = float(colors.get("yellow", 0.0))
        blue = float(colors.get("blue", 0.0))
        if red > 0.01:
            merged["Lifeboat"] += 0.20 + 0.70 * red
        if yellow > 0.01:
            merged["USV"] += 0.18 + 0.60 * yellow
        if blue > 0.08 and size_proxy > 4.0:
            merged["Fishing"] += 0.12 + 0.35 * blue
        if size_proxy >= 5.0:
            merged["Fishing"] += 0.10
        if size_proxy <= 2.6:
            merged["Lifeboat"] += 0.05 * (1.0 - min(1.0, size_proxy / 2.6))
            merged["USV"] += 0.08
        if x_hint < 0.40 and size_proxy <= 3.0 and red >= 0.02:
            merged["Lifeboat"] += 0.18
            merged["USV"] -= 0.06
        if 0.40 <= x_hint <= 0.60:
            merged["USV"] += 0.03
            if yellow >= 0.04:
                merged["USV"] += 0.08
        if x_hint >= 0.86:
            merged["Fishing"] += 0.05
        sorted_items = sorted(merged.items(), key=lambda item: item[1], reverse=True)
        best_label, best_score = sorted_items[0]
        second_score = sorted_items[1][1]
        confidence = self._clamp(0.60 * best_score + 0.20 * alignment + 0.20 * float(heuristic.get("confidence", 0.0)), 0.0, 1.0)
        if best_score < 0.24 or (best_score - second_score) < 0.02:
            best_label = heuristic.get("label", "Unknown")
            confidence = max(confidence, float(heuristic.get("confidence", 0.0)))
        return {
            "label": best_label,
            "confidence": round(confidence, 3),
            "margin": round(float(best_score - second_score), 3),
            "score_map": {k: round(float(v), 4) for k, v in merged.items()},
            "backend": "template_fallback",
            "colors": heuristic.get("colors", {}),
        }

    def _dinov2_classify(self, crop, alignment):
        emb = self._embed_image(crop)
        score_map = {}
        for vessel_type, proto in self._prototype_vectors.items():
            score_map[vessel_type] = float(np.dot(emb, proto))
        sorted_items = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
        best_label, best_score = sorted_items[0]
        second_score = sorted_items[1][1] if len(sorted_items) > 1 else -1.0
        score_term = self._clamp((best_score - self.min_similarity) / max(1e-5, 1.0 - self.min_similarity), 0.0, 1.0)
        margin_term = self._clamp((best_score - second_score) / 0.18, 0.0, 1.0)
        confidence = self._clamp(0.50 * score_term + 0.20 * margin_term + 0.30 * alignment, 0.0, 1.0)
        return {
            "label": best_label,
            "confidence": round(confidence, 3),
            "margin": round(float(best_score - second_score), 3),
            "score_map": {k: round(float(v), 4) for k, v in score_map.items()},
            "backend": "dinov2",
        }

    def classify(self, image, targets):
        rgb_image = self._pil_to_rgb(image)
        normalized_targets = []
        for item in targets or []:
            if isinstance(item, dict):
                normalized_targets.append(dict(item))

        if not self._lazy_init_dino():
            raise RuntimeError(self._status)
        track_classifications = []
        diagnostics = []
        confidences = []
        for target in normalized_targets:
            target_id = str(target.get("target_id") or target.get("id") or "")
            crops = self._candidate_crops(rgb_image, target)
            best = None
            for candidate in crops:
                crop = candidate["crop"]
                alignment = float(candidate["alignment"])
                result = self._dinov2_classify(crop, alignment)
                result["alignment"] = round(alignment, 3)
                result["box"] = candidate["box"]
                if best is None:
                    best = result
                    continue
                if float(result.get("confidence", 0.0)) > float(best.get("confidence", 0.0)):
                    best = result

            if best is None:
                raise RuntimeError("no_valid_candidate_crop:%s" % target_id)

            vessel_type = str(best.get("label", self.VESSEL_TYPES[0]))
            assoc = self._clamp(float(best.get("confidence", 0.0)), 0.0, 1.0)
            track_classifications.append(
                {
                    "target_id": target_id,
                    "vessel_type": vessel_type,
                    "association_confidence": round(assoc, 3),
                }
            )
            confidences.append(assoc)
            diagnostics.append(
                {
                    "target_id": target_id,
                    "vessel_type": vessel_type,
                    "association_confidence": round(assoc, 3),
                    "backend": best.get("backend", self._runtime_backend),
                    "margin": best.get("margin", 0.0),
                    "alignment": best.get("alignment", 0.0),
                    "crop_box": best.get("box", [0, 0, 0, 0]),
                    "score_map": best.get("score_map", {}),
                }
            )

        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        confidence = self._clamp(mean_conf, 0.0, 1.0)
        return {
            "confidence": round(confidence, 3),
            "reasoning": "DINOv2原型匹配",
            "track_classifications": track_classifications,
            "source": "dinov2",
            "backend_runtime": self._runtime_backend,
            "backend_status": self._status,
            "diagnostics": diagnostics,
            "prototype_files": self._prototype_files,
        }

    def diagnostics_json(self):
        return json.dumps(
            {
                "model_name": self.model_name,
                "runtime_backend": self._runtime_backend,
                "status": self._status,
                "prototypes_dir": self.prototypes_dir,
                "prototype_files": self._prototype_files,
            },
            ensure_ascii=True,
        )