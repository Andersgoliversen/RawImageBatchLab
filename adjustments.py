"""
Image-adjustment helpers for RawImageBatchLab
All functions expect / return a float32 **BGR** array in the [0, 1] range.

2025-07-01
• Re-implemented Dehaze with dark-channel + guided-filter refinement
  (no artefacts at strong +values).
• Restored accidentally-omitted basic slider functions so ADJUST_SEQ
  is complete again.
"""
import math
import numpy as np
import cv2

# ------------------------------------------------------------------ #
# White-balance helpers                                              #
# ------------------------------------------------------------------ #
NEUTRAL_TEMP = 5050
NEUTRAL_TINT = 8


def compute_wb_scalers(temperature, tint):
    """Return multiplicative (r,g,b) factors around the neutral point."""
    log_ratio = math.log(max(1, temperature) / NEUTRAL_TEMP)
    r = math.exp(-0.8 * log_ratio)
    b = math.exp( 0.8 * log_ratio)
    g = 1.0 - (tint - NEUTRAL_TINT) / 200.0
    return r, max(g, 0.1), b


# ------------------------------------------------------------------ #
# Small helpers                                                      #
# ------------------------------------------------------------------ #
def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _luminance_bgr(img):
    return 0.0722 * img[..., 0] + 0.7152 * img[..., 1] + 0.2126 * img[..., 2]


# ------------------------------------------------------------------ #
# Basic sliders – all now restored                                   #
# ------------------------------------------------------------------ #
def adjust_exposure(img, stops):
    """Linear gain, ±5 stops in GUI."""
    return np.clip(img * (2.0 ** stops), 0.0, 1.0)


def adjust_contrast(img, amount):
    f = 1 + amount / 100.0         # ±100 → ×0 … ×2
    return np.clip(0.5 + (img - 0.5) * f, 0.0, 1.0)


def adjust_highlights(img, amount):
    if amount == 0:
        return img
    f    = amount / 100.0
    Y    = _luminance_bgr(img)
    mask = _smoothstep(0.55, 1.0, Y)[..., None]
    tgt  = 1.0 if f > 0 else 0.5
    return np.clip(img + mask * abs(f) * (tgt - img), 0.0, 1.0)


def adjust_shadows(img, amount):
    if amount == 0:
        return img
    f    = amount / 100.0
    Y    = _luminance_bgr(img)
    mask = (1.0 - _smoothstep(0.0, 0.45, Y))[..., None]
    tgt  = 0.5 if f > 0 else 0.0
    return np.clip(img + mask * abs(f) * (tgt - img), 0.0, 1.0)


def adjust_whites(img, amount):
    if amount == 0:
        return img
    Y        = _luminance_bgr(img)
    white_p  = np.percentile(Y, 99)
    if white_p < 1e-6:
        return img.copy()
    gain = 2.0 ** (amount / 100.0)          # ±1 stop
    mask = _smoothstep(white_p * 0.8, white_p, Y)[..., None]
    out  = img * (1 - mask) + img * gain * mask
    return np.clip(out, 0.0, 1.0)


def adjust_blacks(img, amount):
    if amount == 0:
        return img
    f    = amount / 100.0
    Y    = _luminance_bgr(img)
    mask = (1.0 - _smoothstep(0.0, 0.15, Y))[..., None]
    tgt  = 0.15 if f > 0 else 0.0
    return np.clip(img + mask * abs(f) * (tgt - img), 0.0, 1.0)


def adjust_texture(img, amount):
    f   = amount / 100.0
    out = img + (img - cv2.GaussianBlur(img, (0, 0), 3)) * f
    return np.clip(out, 0.0, 1.0)


def adjust_clarity(img, amount):
    f   = amount / 100.0
    lap = img - cv2.GaussianBlur(img, (0, 0), 3)
    out = img + lap * f * (1 - np.abs(img - 0.5) * 2)
    return np.clip(out, 0.0, 1.0)


def adjust_saturation(img, amount):
    f   = amount / 100.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + f), 0, 1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_vibrance(img, amount):
    f   = amount / 100.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + f * (1 - hsv[..., 1])), 0, 1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ------------------------------------------------------------------ #
# Dehaze (dark-channel + guided-filter)                              #
# ------------------------------------------------------------------ #
def dehaze_image(img, amount):
    if amount == 0:
        return img

    f = amount / 100.0
    if f > 0:                                            # remove haze
        kernel = np.ones((15, 15), np.uint8)
        dark   = cv2.erode(np.min(img, 2), kernel)

        flat   = dark.ravel()
        n_bright = max(1, flat.size // 1000)
        idx    = np.argpartition(flat, -n_bright)[-n_bright:]
        A      = np.clip(np.mean(img.reshape(-1, 3)[idx], axis=0), 0.7, 1.0)

        omega  = 0.95
        t0     = 1.0 - omega * f * dark / np.maximum(dark.max(), 1e-6)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        try:
            import cv2.ximgproc as xi
            t = xi.guidedFilter(guide=gray, src=t0.astype(np.float32),
                                radius=60, eps=1e-3)
        except Exception:
            t = cv2.bilateralFilter(t0.astype(np.float32), 9, 0.1, 75)

        t   = np.clip(t, 0.2, 1.0)[..., None]
        out = (img - A) / t + A
    else:                                                # add haze
        alpha = -f
        haze  = cv2.GaussianBlur(img, (0, 0), 8)
        haze  = np.clip(haze + 0.5, 0.0, 1.0)
        out   = img * (1 - alpha) + haze * alpha

    return np.clip(out, 0.0, 1.0)


# ------------------------------------------------------------------ #
# Adjustment pipeline                                                #
# ------------------------------------------------------------------ #
ADJUST_SEQ = [
    ("exposure",   adjust_exposure),
    ("contrast",   adjust_contrast),
    ("highlights", adjust_highlights),
    ("shadows",    adjust_shadows),
    ("whites",     adjust_whites),
    ("blacks",     adjust_blacks),
    ("texture",    adjust_texture),
    ("clarity",    adjust_clarity),
    ("dehaze",     dehaze_image),
    ("vibrance",   adjust_vibrance),
    ("saturation", adjust_saturation),
]


def apply_adjustments(img, params):
    """Apply the full chain in Camera-Raw order."""
    for name, func in ADJUST_SEQ:
        img = np.clip(func(img, params[name]), 0.0, 1.0)
    return img
