"""
Image-adjustment helpers for RawImageBatchLab
All functions expect / return a float32 **BGR** array in the [0, 1] range.

2025-08-27
• Tone-region sliders (Exposure/Highlights/Shadows/Whites/Blacks) now operate
  in linear light with soft-knee curves and hue preservation (luminance ratio).
• 16-bit decode is supported by pipeline; these ops avoid banding and plates.
• Kept UI options, Dehaze/Clarity/Texture/Vibrance/Saturation behavior.
"""
import math
import numpy as np
import cv2

# ------------------------------------------------------------------ #
# White-balance helpers (unchanged)                                  #
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
    # BGR order; BT.709 luminance in **linear** light
    return 0.0722 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.2126 * lin[..., 2]


def _guided_refine(mask, guide, radius=40, eps=1e-3):
    """Refine a float mask [0..1] using a guided filter; fallback to bilateral."""
    mask32 = mask.astype(np.float32)
    guide32 = guide.astype(np.float32)
    try:
        import cv2.ximgproc as xi
        return xi.guidedFilter(guide=guide32, src=mask32, radius=radius, eps=eps)
    except Exception:
        return cv2.bilateralFilter(mask32, 9, 0.1, 75)


def _apply_luma_ratio(lin, Y_old, Y_new):
    """Hue-preserving scaling by luminance ratio in linear space."""
    scale = (Y_new / (Y_old + 1e-6))[..., None]
    return np.clip(lin * scale, 0.0, 1.0)


# ------------------------------------------------------------------ #
# Basic sliders – linear-light where tone-mapping is intended        #
# ------------------------------------------------------------------ #
def adjust_exposure(img, stops):
    """Scene-like exposure: multiply linear light by 2^stops."""
    if stops == 0:
        return img
    lin = _to_linear(img)
    gain = float(2.0 ** stops)
    out = np.clip(lin * gain, 0.0, 1.0)
    return _to_srgb(out)


def adjust_contrast(img, amount):
    """Simple mid-gray contrast around 18% (linear)."""
    if amount == 0:
        return img
    lin = _to_linear(img)
    m = 0.18
    k = 1.0 + amount / 100.0  # ±100 → ×[0,2]
    out = np.clip(m + (lin - m) * k, 0.0, 1.0)
    return _to_srgb(out)


def adjust_highlights(img, amount):
    """Roll-off or expand highlights with a soft knee; hue-preserving."""
    if amount == 0:
        return img
    lin = _to_linear(img)
    Y = _luminance_bgr_linear(lin)

    # Mask brightest 35% by default, with soft edge
    knee = np.percentile(Y, 65)
    mask = _smoothstep(knee * 0.9, min(1.0, knee * 1.15), Y)
    mask = _guided_refine(mask, Y, radius=40)
    mask = mask[..., None]

    a = np.clip(amount / 100.0, -1.0, 1.0)
    if a >= 0:  # brighten/expand highlights
        k = 1.0 + 2.0 * a
        # Use a root curve, the inverse of the power curve used for compression.
        # This smoothly expands highlights instead of clipping them.
        Y_tgt = np.power(Y, 1.0 / k)
    else:       # compress/darken highlights
        k = 1.0 + 2.0 * (-a)
        Y_tgt = np.power(Y, k)

    Y_new = Y * (1.0 - mask[..., 0]) + Y_tgt * mask[..., 0]
    out = _apply_luma_ratio(lin, Y, Y_new)
    return _to_srgb(out)


def adjust_shadows(img, amount):
    """Lift or deepen shadows with a soft knee; hue-preserving."""
    if amount == 0:
        return img
    lin = _to_linear(img)
    Y = _luminance_bgr_linear(lin)

    # Mask darkest ~40% with soft edge
    knee = np.percentile(Y, 40)
    mask = 1.0 - _smoothstep(max(0.0, knee * 0.9), min(1.0, knee * 1.1), Y)
    mask = _guided_refine(mask, Y, radius=40)
    mask = mask[..., None]

    a = np.clip(amount / 100.0, -1.0, 1.0)
    if a >= 0:  # lift
        k = 1.0 + 2.0 * a
        Y_tgt = np.power(Y, 1.0 / k)
    else:       # deepen
        k = 1.0 + 2.0 * (-a)
        Y_tgt = np.power(Y, k)

    Y_new = Y * (1.0 - mask[..., 0]) + Y_tgt * mask[..., 0]
    out = _apply_luma_ratio(lin, Y, Y_new)
    return _to_srgb(out)

def adjust_whites(img, amount):
    """Adjust the white point via linear gain."""
    if amount == 0:
        return img

    lin = _to_linear(img)
    if amount > 0:
        # Pushing whites brighter is equivalent to exposure
        gain = 1.0 + amount / 100.0 * 0.5 # Scale effect
        out = lin * gain
    else:
        # Pulling whites down re-maps a high percentile to a lower value
        Y = _luminance_bgr_linear(lin)
        # Find the current near-white point
        white_point = np.percentile(Y, 99.8)
        if white_point < 1e-5:
            return img # Avoid division by zero on black images

        # Calculate gain needed to pull the white point down
        # -100 maps the white point to ~0.85, recovering headroom
        target_point = white_point * (1.0 + amount / 200.0)
        gain = target_point / white_point
        out = lin * gain

    return _to_srgb(out)


def adjust_blacks(img, amount):
    """Adjust near-black region with soft knee and hue preservation (linear)."""
    if amount == 0:
        return img

    lin = _to_linear(img)
    Y = _luminance_bgr_linear(lin)

    black_p = np.percentile(Y, 1.0)
    m0 = max(0.0, black_p * 0.8)
    m1 = black_p * 1.2 + 1e-6
    mask = 1.0 - _smoothstep(m0, m1, Y)
    mask = _guided_refine(mask, Y, radius=50)
    mask = mask[..., None]

    a = np.clip(amount / 100.0, -1.0, 1.0)
    if a >= 0:  # lift blacks
        k = 1.0 + 2.5 * a
        Y_tgt = np.power(Y, 1.0 / k)
    else:       # crush blacks
        k = 1.0 + 3.0 * (-a)
        Y_tgt = np.power(Y, k)

    Y_new = Y * (1.0 - mask[..., 0]) + Y_tgt * mask[..., 0]
    out = _apply_luma_ratio(lin, Y, Y_new)
    return _to_srgb(out)


def adjust_texture(img, amount):
    """Fine-detail emphasis; remains display-referred."""
    if amount == 0:
        return img
    f = amount / 100.0
    out = img + (img - cv2.GaussianBlur(img, (0, 0), 3)) * f
    return np.clip(out, 0.0, 1.0)


def adjust_clarity(img, amount):
    """Midtone contrast emphasis with edge-aware weighting (display-referred)."""
    if amount == 0:
        return img
    f = amount / 100.0
    lap = img - cv2.GaussianBlur(img, (0, 0), 3)
    out = img + lap * f * (1 - np.abs(img - 0.5) * 2)
    return np.clip(out, 0.0, 1.0)


def adjust_saturation(img, amount):
    if amount == 0:
        return img
    f = amount / 100.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + f), 0, 1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_vibrance(img, amount):
    if amount == 0:
        return img
    f = amount / 100.0
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
        t = _guided_refine(t0.astype(np.float32), gray, radius=60, eps=1e-3)

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
    ("whites",     adjust_whites),
    ("blacks",     adjust_blacks),
    ("highlights", adjust_highlights),
    ("shadows",    adjust_shadows),
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
