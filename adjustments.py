"""
Image-adjustment helpers.
Every function expects / returns a float32 **BGR** array in [0, 1].

The “Highlights” and “Whites” implementations emulate Adobe Camera Raw and
avoid the artefacts seen earlier.
"""
import math
import numpy as np
import cv2

# ------------------------------------------------------------------
# white-balance helpers
# ------------------------------------------------------------------
NEUTRAL_TEMP = 5050      # GUI default = camera As-Shot
NEUTRAL_TINT = 8


def compute_wb_scalers(temperature, tint):
    """Return multiplicative scalers (r,g,b) around the neutral point."""
    log_ratio = math.log(max(1, temperature) / NEUTRAL_TEMP)
    r = math.exp(-0.8 * log_ratio)
    b = math.exp( 0.8 * log_ratio)
    g = 1.0 - (tint - NEUTRAL_TINT) / 200.0
    return r, max(g, 0.1), b       # keep green ≥ 0.1


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
    return t*t*(3.0 - 2.0*t)


def _luminance_bgr(img):
    return 0.0722*img[..., 0] + 0.7152*img[..., 1] + 0.2126*img[..., 2]


# ------------------------------------------------------------------
# basic sliders
# ------------------------------------------------------------------
def adjust_exposure(img, stops):
    return img * (2.0 ** stops)


def adjust_contrast(img, amount):
    return 0.5 + (img - 0.5) * (1 + amount/100.0)


# ------------------------------------------------------------------
# highlights
# ------------------------------------------------------------------
def adjust_highlights(img, amount):
    if amount == 0:
        return img
    f     = amount / 100.0
    Y     = _luminance_bgr(img)
    mask  = _smoothstep(0.55, 1.0, Y)[..., None]
    out   = img + mask * (f if f > 0 else -f) * ((1.0 if f > 0 else 0.5) - img)
    return np.clip(out, 0.0, 1.0)


# ------------------------------------------------------------------
# whites
# ------------------------------------------------------------------
def adjust_whites(img, amount):
    if amount == 0:
        return img
    Y        = _luminance_bgr(img)
    white_p  = np.percentile(Y, 99)
    if white_p < 1e-6:
        return img.copy()
    gain     = 2.0 ** (amount/100.0)          # ±1 stop
    mask     = _smoothstep(white_p*0.8, white_p, Y)[..., None]
    out      = img*(1 - mask) + img*gain*mask
    return np.clip(out, 0.0, 1.0)


# ------------------------------------------------------------------
# remaining sliders (unchanged)
# ------------------------------------------------------------------
def adjust_shadows(img, amount):
    f, m = amount/100.0, img < 0.5
    out  = img.copy()
    out[m] += out[m]*f
    return out


def adjust_blacks(img, amount):
    f, lo = amount/100.0, np.percentile(img, 5)
    return np.clip((img - lo*f)/(1 - lo*f), 0, 1)


def adjust_texture(img, amount):
    f = amount/100.0
    return img + (img - cv2.GaussianBlur(img, (0, 0), 3))*f


def adjust_clarity(img, amount):
    f   = amount/100.0
    lap = img - cv2.GaussianBlur(img, (0, 0), 3)
    return img + lap*f*(1 - np.abs(img - 0.5)*2)


def dehaze_image(img, amount):
    f    = max(0, amount/100.0)
    dark = cv2.erode(np.min(img, 2), np.ones((15, 15), np.uint8))
    t    = np.clip(1 - f*dark, 0.1, 1.0)
    return np.clip((img - 1)/t[..., None] + 1, 0, 1)


def adjust_saturation(img, amount):
    f   = amount/100.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1]*(1 + f), 0, 1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_vibrance(img, amount):
    f   = amount/100.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1]*(1 + f*(1 - hsv[..., 1])), 0, 1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ------------------------------------------------------------------
# adjustment pipeline
# ------------------------------------------------------------------
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
    for name, func in ADJUST_SEQ:
        img = np.clip(func(img, params[name]), 0.0, 1.0)
    return img
