"""Image-adjustment helpers for RawImageBatchLab.

All functions expect and return float32 BGR arrays in the [0, 1] range.
Tone-region controls operate in linear light and re-apply changes through a
luminance ratio so color and hue remain stable under strong edits.
"""
import math

import cv2
import numpy as np

NEUTRAL_TEMP = 5050
NEUTRAL_TINT = 8


def compute_wb_scalers(temperature, tint):
    """Return multiplicative (r, g, b) factors around the neutral point."""
    log_ratio = math.log(max(1, temperature) / NEUTRAL_TEMP)
    r = math.exp(-0.8 * log_ratio)
    b = math.exp(0.8 * log_ratio)
    g = 1.0 - (tint - NEUTRAL_TINT) / 200.0
    return r, max(g, 0.1), b


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _srgb_to_linear(x):
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)


def _linear_to_srgb(x):
    a = 0.055
    return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * np.power(x, 1 / 2.4) - a)


def _to_linear(img_srgb):
    return _srgb_to_linear(np.clip(img_srgb, 0.0, 1.0))


def _to_srgb(img_lin):
    return np.clip(_linear_to_srgb(np.clip(img_lin, 0.0, 1.0)), 0.0, 1.0)


def _luminance_bgr_linear(lin):
    return 0.0722 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.2126 * lin[..., 2]


def _robust_percentile(x, q):
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return 0.0
    return float(np.percentile(finite, q))


def _soft_compress(x, shoulder=0.92):
    """Gently compress values above shoulder instead of hard clipping."""
    x = np.maximum(x, 0.0)
    hi = x > shoulder
    out = x.copy()
    headroom = max(1e-6, 1.0 - shoulder)
    out[hi] = shoulder + headroom * (1.0 - np.exp(-(x[hi] - shoulder) / headroom))
    return np.clip(out, 0.0, 1.0)


def _guided_refine(mask, guide, radius=40, eps=1e-3):
    """Refine a float mask [0, 1] using a guided filter; fallback to bilateral."""
    mask32 = np.clip(mask, 0.0, 1.0).astype(np.float32)
    guide32 = np.clip(guide, 0.0, 1.0).astype(np.float32)
    try:
        import cv2.ximgproc as xi

        refined = xi.guidedFilter(guide=guide32, src=mask32, radius=radius, eps=eps)
    except Exception:
        refined = cv2.bilateralFilter(mask32, 9, 0.1, 75)
    return np.clip(refined, 0.0, 1.0)


def _apply_luma_ratio(lin, y_old, y_new):
    """Hue-preserving scaling by luminance ratio in linear space."""
    scale = (y_new / (y_old + 1e-6))[..., None]
    return np.clip(lin * scale, 0.0, 1.0)


def adjust_exposure(img, stops):
    """Scene-like exposure: multiply linear light by 2^stops."""
    if stops == 0:
        return img
    lin = _to_linear(img)
    return _to_srgb(np.clip(lin * float(2.0 ** stops), 0.0, 1.0))


def adjust_contrast(img, amount):
    """Midtone contrast around perceptual gray with hue preservation."""
    if amount == 0:
        return img
    lin = _to_linear(img)
    y = _luminance_bgr_linear(lin)
    a = np.clip(amount / 100.0, -1.0, 1.0)
    pivot = 0.18
    slope = 1.0 + 1.45 * a if a >= 0 else 1.0 / (1.0 + 1.8 * -a)
    y_target = pivot + (y - pivot) * slope
    midtone_weight = _smoothstep(0.015, 0.12, y) * (1.0 - _smoothstep(0.78, 0.98, y))
    y_new = y * (1.0 - midtone_weight) + _soft_compress(y_target) * midtone_weight
    return _to_srgb(_apply_luma_ratio(lin, y, y_new))


def adjust_highlights(img, amount):
    """Recover or expand upper tones with a soft, edge-aware luminance mask."""
    if amount == 0:
        return img
    lin = _to_linear(img)
    y = _luminance_bgr_linear(lin)
    knee = max(0.32, _robust_percentile(y, 62))
    mask = _smoothstep(knee * 0.78, min(0.98, knee * 1.28), y)
    mask = _guided_refine(mask, y, radius=40)

    a = np.clip(amount / 100.0, -1.0, 1.0)
    if a >= 0:
        y_target = _soft_compress(y * (2.0 ** (0.9 * a)), shoulder=0.86)
    else:
        compressed = y / (1.0 + (-a) * 1.8 * np.maximum(y - knee, 0.0))
        y_target = np.minimum(y, compressed)

    y_new = y * (1.0 - mask) + y_target * mask
    return _to_srgb(_apply_luma_ratio(lin, y, y_new))


def adjust_shadows(img, amount):
    """Lift or deepen lower tones with a soft, edge-aware luminance mask."""
    if amount == 0:
        return img
    lin = _to_linear(img)
    y = _luminance_bgr_linear(lin)
    knee = min(0.45, max(0.08, _robust_percentile(y, 42)))
    mask = 1.0 - _smoothstep(max(0.0, knee * 0.45), min(0.85, knee * 1.65), y)
    mask = _guided_refine(mask, y, radius=40)

    a = np.clip(amount / 100.0, -1.0, 1.0)
    if a >= 0:
        curve = 2.0 + 5.0 * a
        lift = (1.0 - np.exp(-y * curve)) / (1.0 - np.exp(-curve))
        y_target = np.maximum(y, lift)
    else:
        y_target = np.power(np.clip(y, 0.0, 1.0), 1.0 + 1.8 * -a)

    y_new = y * (1.0 - mask) + y_target * mask
    return _to_srgb(_apply_luma_ratio(lin, y, y_new))


def adjust_whites(img, amount):
    """Adjust the upper tonal endpoint without moving the full image equally."""
    if amount == 0:
        return img
    lin = _to_linear(img)
    y = _luminance_bgr_linear(lin)
    white_point = max(0.25, _robust_percentile(y, 99.5))
    mask = _smoothstep(white_point * 0.55, min(0.995, white_point * 1.02), y)
    mask = _guided_refine(mask, y, radius=30)

    a = np.clip(amount / 100.0, -1.0, 1.0)
    y_target = _soft_compress(y * (2.0 ** (0.7 * a)), shoulder=0.9 if a >= 0 else 0.98)
    if a < 0:
        y_target = np.minimum(y, y_target)

    y_new = y * (1.0 - mask) + y_target * mask
    return _to_srgb(_apply_luma_ratio(lin, y, y_new))


def adjust_blacks(img, amount):
    """Adjust the lower tonal endpoint without crushing color channels."""
    if amount == 0:
        return img
    lin = _to_linear(img)
    y = _luminance_bgr_linear(lin)
    black_point = max(0.005, _robust_percentile(y, 2.0))
    mask = 1.0 - _smoothstep(max(0.0, black_point * 0.45), min(0.22, black_point * 4.0 + 0.015), y)
    mask = _guided_refine(mask, y, radius=50)

    a = np.clip(amount / 100.0, -1.0, 1.0)
    if a >= 0:
        floor = 0.055 * a
        y_target = np.maximum(y, floor + y * (1.0 - floor))
    else:
        y_target = np.power(np.clip(y, 0.0, 1.0), 1.0 + 3.2 * -a)

    y_new = y * (1.0 - mask) + y_target * mask
    return _to_srgb(_apply_luma_ratio(lin, y, y_new))


def adjust_texture(img, amount):
    """Fine-detail emphasis; remains display-referred."""
    if amount == 0:
        return img
    f = amount / 100.0
    detail = img - cv2.GaussianBlur(img, (0, 0), 3)
    return np.clip(img + detail * f, 0.0, 1.0)


def adjust_clarity(img, amount):
    """Local midtone contrast with reduced shadow/highlight halos."""
    if amount == 0:
        return img
    f = amount / 100.0
    blur = cv2.bilateralFilter(img.astype(np.float32), 0, 0.08, 24)
    local = img - blur
    y = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2GRAY)
    weight = _smoothstep(0.08, 0.35, y) * (1.0 - _smoothstep(0.72, 0.96, y))
    return np.clip(img + local * (f * 1.4) * weight[..., None], 0.0, 1.0)


def adjust_saturation(img, amount):
    if amount == 0:
        return img
    f = amount / 100.0
    hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + f), 0.0, 1.0)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_vibrance(img, amount):
    if amount == 0:
        return img
    f = amount / 100.0
    hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]
    value = hsv[..., 2]
    protection = 1.0 - np.clip(sat, 0.0, 1.0)
    highlight_guard = 1.0 - _smoothstep(0.85, 1.0, value)
    hsv[..., 1] = np.clip(sat * (1.0 + f * protection * highlight_guard), 0.0, 1.0)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def dehaze_image(img, amount):
    if amount == 0:
        return img

    f = amount / 100.0
    if f > 0:
        kernel = np.ones((15, 15), np.uint8)
        dark = cv2.erode(np.min(img, 2), kernel)

        flat = dark.ravel()
        n_bright = max(1, flat.size // 1000)
        idx = np.argpartition(flat, -n_bright)[-n_bright:]
        atmosphere = np.clip(np.mean(img.reshape(-1, 3)[idx], axis=0), 0.7, 1.0)

        omega = 0.95
        t0 = 1.0 - omega * f * dark / np.maximum(dark.max(), 1e-6)
        gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2GRAY)
        transmission = _guided_refine(t0.astype(np.float32), gray, radius=60, eps=1e-3)
        transmission = np.clip(transmission, 0.2, 1.0)[..., None]
        out = (img - atmosphere) / transmission + atmosphere
    else:
        alpha = -f
        haze = cv2.GaussianBlur(img, (0, 0), 8)
        haze = np.clip(haze + 0.5, 0.0, 1.0)
        out = img * (1.0 - alpha) + haze * alpha

    return np.clip(out, 0.0, 1.0)


ADJUST_SEQ = [
    ("exposure", adjust_exposure),
    ("contrast", adjust_contrast),
    ("whites", adjust_whites),
    ("blacks", adjust_blacks),
    ("highlights", adjust_highlights),
    ("shadows", adjust_shadows),
    ("texture", adjust_texture),
    ("clarity", adjust_clarity),
    ("dehaze", dehaze_image),
    ("vibrance", adjust_vibrance),
    ("saturation", adjust_saturation),
]


def apply_adjustments(img, params):
    """Apply the full chain in a Camera Raw-like order."""
    for name, func in ADJUST_SEQ:
        img = np.clip(func(img, params[name]), 0.0, 1.0).astype(np.float32, copy=False)
    return img
