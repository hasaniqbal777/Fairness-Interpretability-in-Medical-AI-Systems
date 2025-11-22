from typing import Optional, Callable
import io
import torch
from PIL import Image, ImageEnhance

def gaussian_noise(img: torch.Tensor, severity: int = 1) -> torch.Tensor:
    # img: (B,C,H,W) normalized tensor; we add noise in normalized space approximate
    # Map to [0,1] first
    x = img.clone()
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    sigmas = {1: 0.05, 2: 0.10, 3: 0.20}
    sigma = sigmas.get(severity, 0.10)
    noise = torch.randn_like(x) * sigma
    x = torch.clamp(x + noise, 0.0, 1.0)
    # return clamped normalized back to original mean/std is not straightforward; keep in [0,1] and re-normalize outside if needed
    return x

def jpeg_compress_pil(pil_img: Image.Image, severity: int = 1) -> Image.Image:
    qualities = {1: 50, 2: 25, 3: 10}
    q = qualities.get(severity, 50)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    return out

def brightness_contrast_pil(pil_img: Image.Image, severity: int = 1) -> Image.Image:
    # Use moderate ranges
    factors = {1: (1.1, 1.1), 2: (1.25, 1.25), 3: (1.4, 1.4)}
    b, c = factors.get(severity, (1.1,1.1))
    pil_img = ImageEnhance.Brightness(pil_img).enhance(b)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(c)
    return pil_img
