"""Reproduce HF image geometry once, before mtmd normalization and encoding."""

import hashlib
import math
from importlib.metadata import PackageNotFoundError, version


PREPROCESSING_VERSION = 1
DEFAULT_RESIZE_BACKEND = "torchvision"


def preprocessing_identity(backend=DEFAULT_RESIZE_BACKEND, *, enabled=True):
    if enabled and backend not in ("torchvision", "pillow"):
        raise ValueError(f"unsupported image resize backend: {backend!r}")
    backend = backend if enabled else "native"
    packages = ["pillow"] + (["torch", "torchvision"] if backend == "torchvision" else [])
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            raise ValueError(f"{package} is required for image resize backend {backend!r}; install the skymizer vision extra") from None
    return {"version": PREPROCESSING_VERSION, "resize_backend": backend, "packages": versions}


def effective_image_config(config):
    ip = dict(config.get("hf_image_processor", {}))
    for key in ("gemma4_vision_token_config", "kimi_vl_vision_token_config"):
        ip.update(config.get(key, {}))
    overrides = config.get("vllm_mm_processor_kwargs", {})
    if not isinstance(overrides, dict):
        raise ValueError("vllm_mm_processor_kwargs must be an object")
    for key, value in overrides.items():
        if key == "size":
            if not isinstance(value, dict):
                raise ValueError("image size override must be an object")
            ip["size"] = {**ip.get("size", {}), **value}
            for old, new in (("min_pixels", "shortest_edge"), ("max_pixels", "longest_edge")):
                if new in value:
                    ip.pop(old, None)
        else:
            ip[key] = value
    return ip


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


def image_geometry(width, height, ip):
    """Return resize size, padded size, and merged grid, all in (width, height) order."""
    positive_int(width, "image width")
    positive_int(height, "image height")
    patch = positive_int(ip.get("patch_size"), "patch_size")
    family = ip.get("image_processor_type", "")
    if family == "KimiVLImageProcessor":
        kernel = ip.get("merge_kernel_size")
        if not isinstance(kernel, (list, tuple)) or len(kernel) != 2:
            raise ValueError("merge_kernel_size must have two entries")
        kh, kw = [positive_int(x, "merge_kernel_size") for x in kernel]
        if kh != kw or ip.get("pad_input", True) is not True:
            raise ValueError("Kimi alignment requires square merging and pad_input=true")
        cap = positive_int(ip.get("in_token_limit"), "in_token_limit")
        patches = (width // patch) * (height // patch)
        rw, rh = width, height
        if patches > cap:
            scale = math.sqrt(cap / patches)
            rw, rh = int(width * scale), int(height * scale)
        aw, ah = patch * kw, patch * kh
        ow, oh = math.ceil(rw / aw) * aw, math.ceil(rh / ah) * ah
        if ow // patch >= 512 or oh // patch >= 512:
            raise ValueError("Kimi image exceeds the vision position embedding range")
    elif family in ("Gemma4ImageProcessor", "Gemma4ImageProcessorPil"):
        merge = positive_int(ip.get("pooling_kernel_size"), "pooling_kernel_size")
        cap = positive_int(ip.get("max_soft_tokens"), "max_soft_tokens")
        if cap not in (70, 140, 280, 560, 1120):
            raise ValueError(f"unsupported Gemma max_soft_tokens: {cap}")
        aw = ah = patch * merge
        scale = math.sqrt(cap * aw * ah / (width * height))
        ow, oh = math.floor(width * scale / aw) * aw, math.floor(height * scale / ah) * ah
        if ow == 0:
            ow, oh = aw, min(math.floor(height / width) * ah, cap * ah)
        elif oh == 0:
            oh, ow = ah, min(math.floor(width / height) * aw, cap * aw)
        rw, rh = ow, oh
    elif family in ("Qwen2VLImageProcessor", "Qwen2VLImageProcessorFast", "Qwen2VLImageProcessorPil",
                    "Glm46VImageProcessor", "Glm46VImageProcessorPil"):
        merge = positive_int(ip.get("merge_size"), "merge_size")
        aw = ah = patch * merge
        size = ip.get("size", {})
        lower = positive_int(ip.get("min_pixels", size.get("shortest_edge")), "min_pixels")
        upper = positive_int(ip.get("max_pixels", size.get("longest_edge")), "max_pixels")
        if upper < lower:
            raise ValueError("max_pixels must be at least min_pixels")
        temporal = 1
        if family.startswith("Glm"):
            temporal = positive_int(ip.get("temporal_patch_size"), "temporal_patch_size")
            if height < ah or width < aw:
                scale = max(ah / height, aw / width)
                height, width = int(height * scale), int(width * scale)
        if max(width, height) / min(width, height) > 200:
            raise ValueError("image aspect ratio exceeds 200")
        ow, oh = round(width / aw) * aw, round(height / ah) * ah
        if temporal * ow * oh > upper:
            scale = math.sqrt(temporal * width * height / upper)
            ow = max(aw, math.floor(width / scale / aw) * aw)
            oh = max(ah, math.floor(height / scale / ah) * ah)
        elif temporal * ow * oh < lower:
            scale = math.sqrt(lower / (temporal * width * height))
            ow, oh = math.ceil(width * scale / aw) * aw, math.ceil(height * scale / ah) * ah
        rw, rh = ow, oh
    else:
        raise ValueError(f"unsupported image processor: {family!r}")
    if min(rw, rh, ow, oh) <= 0:
        raise ValueError("image processor produced an empty image")
    if ip.get("do_resize", True) is not True and family != "KimiVLImageProcessor":
        raise ValueError("alignment requires do_resize=true")
    return (rw, rh), (ow, oh), (ow // aw, oh // ah)


def prepare_images(images, config, backend=DEFAULT_RESIZE_BACKEND, *, enabled=True):
    from PIL import Image

    ip = effective_image_config(config) if enabled else {}
    actual_backend = "pillow" if ip.get("image_processor_type") == "KimiVLImageProcessor" else backend
    identity = preprocessing_identity(actual_backend, enabled=enabled)
    prepared, records = [], []
    for image in images:
        original_size = image.size
        resized_size, output_size, grid = (image_geometry(*original_size, ip) if enabled
                                           else (original_size, original_size, None))
        image = image.convert("RGB")
        source_hash = hashlib.sha256(image.tobytes()).hexdigest()
        resample = ip.get("resample", 3)
        if resample not in (0, 2, 3):
            raise ValueError(f"unsupported image resample: {resample!r}")
        if image.size != resized_size:
            if actual_backend == "pillow":
                image = image.resize(resized_size, Image.Resampling(resample))
            else:
                from torchvision.transforms.v2 import functional as F
                interpolation = {0: F.InterpolationMode.NEAREST, 2: F.InterpolationMode.BILINEAR,
                                 3: F.InterpolationMode.BICUBIC}[resample]
                pixels = F.resize(F.pil_to_tensor(image), list(reversed(resized_size)),
                                  interpolation=interpolation, antialias=True)
                image = F.to_pil_image(pixels)
        if resized_size != output_size:
            padded = Image.new("RGB", output_size, (0, 0, 0))
            padded.paste(image, (0, 0))
            image = padded
        record = {
            "original_size": list(original_size), "resized_size": list(resized_size),
            "width": output_size[0], "height": output_size[1],
            "source_rgb_sha256": source_hash, "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        }
        if grid is not None:
            record.update(grid_width=grid[0], grid_height=grid[1], vision_tokens=math.prod(grid))
        records.append(record)
        prepared.append(image)
    return prepared, {**identity, "images": records}
