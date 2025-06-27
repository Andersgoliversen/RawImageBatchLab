"""
Image-adjustment helpers.
Every function expects / returns a float32 BGR array in [0, 1].
"""

import math
import numpy as np
import cv2

# ----------------------------------------------------------------------
# white-balance helpers
# ----------------------------------------------------------------------

NEUTRAL_TEMP  = 5050      # GUI defaults = camera As-Shot
NEUTRAL_TINT  = 8

def compute_wb_scalers(temperature, tint):
    """
    Return multiplicative scalers (r, g, b) that express the *offset*
    from the neutral point (5050 K, +8).

    At the GUI defaults this function returns (1, 1, 1).
    Positive temp ⇒ cooler image, negative ⇒ warmer.
    Positive tint ⇒ magenta, negative ⇒ green.
    """
    log_ratio = math.log(max(1, temperature) / NEUTRAL_TEMP)

    # Symmetric exponential model, similar to LibRaw’s
    r = math.exp(-0.8 * log_ratio)
    b = math.exp( 0.8 * log_ratio)

    # Linear green/magenta around the neutral tint
    g = 1.0 - (tint - NEUTRAL_TINT) / 200.0
    return r, max(g, 0.1), b        # clamp g to stay positive


# ----------------------------------------------------------------------
# tone / colour adjustments   (unchanged)
# ----------------------------------------------------------------------

def adjust_exposure(img, stops):
    return img * (2.0 ** stops)

def adjust_contrast(img, amount):
    return 0.5 + (img - 0.5) * (1 + amount / 100.0)

def adjust_highlights(img, amount):
    f, m = amount / 100.0, img > 0.5
    img[m] += (1.0 - img[m]) * f
    return img

def adjust_shadows(img, amount):
    f, m = amount / 100.0, img < 0.5
    img[m] += img[m] * f
    return img

def adjust_whites(img, amount):
    f, hi = amount / 100.0, np.percentile(img, 95)
    return np.clip((img - hi * f) / (1 - hi * f), 0, 1)

def adjust_blacks(img, amount):
    f, lo = amount / 100.0, np.percentile(img, 5)
    return np.clip((img - lo * f) / (1 - lo * f), 0, 1)

def adjust_texture(img, amount):
    f = amount / 100.0
    return img + (img - cv2.GaussianBlur(img, (0,0), 3)) * f

def adjust_clarity(img, amount):
    f   = amount / 100.0
    lap = img - cv2.GaussianBlur(img, (0,0), 3)
    return img + lap * f * (1 - np.abs(img - .5) * 2)

def dehaze_image(img, amount):
    f   = max(0, amount / 100.0)
    dark = cv2.erode(np.min(img, 2), np.ones((15,15), np.uint8))
    t    = np.clip(1 - f * dark, 0.1, 1.0)
    return np.clip((img - 1) / t[...,None] + 1, 0, 1)

def adjust_saturation(img, amount):
    f   = amount / 100.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] = np.clip(hsv[...,1] * (1 + f), 0, 1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def adjust_vibrance(img, amount):
    f   = amount / 100.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] = np.clip(hsv[...,1] * (1 + f * (1 - hsv[...,1])), 0, 1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
