"""
RAW-to-JPEG batch pipeline + live-preview helper.
"""

import os
import cv2
import rawpy
import numpy as np

from adjustments import (
    compute_wb_scalers,
    apply_adjustments,
    NEUTRAL_TEMP, NEUTRAL_TINT,
)

RAW_EXTS = (".nef",".cr2",".crw",".raf",".dng",
            ".dcr",".mrw",".orf",".pef",".srf")

# mapping from user-friendly names to rawpy colour spaces
_COLOR_MAP = {
    "sRGB": rawpy.ColorSpace.sRGB,
    "Adobe": rawpy.ColorSpace.Adobe,
    "ProPhoto": rawpy.ColorSpace.ProPhoto,
}

def _map_color_space(name):
    """Return rawpy colour space constant for given name."""
    return _COLOR_MAP.get(name, rawpy.ColorSpace.sRGB)

# ----------------------------------------------------------------------
# RAW → float32 BGR [0–1]   (preview + processing start-point)
# ----------------------------------------------------------------------

def raw2bgr(raw, *, kelvin, tint, color_space="sRGB"):
    """
    * If sliders are at the neutral point (5050 K, +8) we simply let
      LibRaw apply the camera’s As-Shot multipliers – identical to ACR.
    * Otherwise we scale those multipliers by the *relative* factors
      returned from `compute_wb_scalers()`.
    """
    if kelvin == NEUTRAL_TEMP and tint == NEUTRAL_TINT:
        rgb8 = raw.postprocess(
            no_auto_bright=True,
            output_color = _map_color_space(color_space),
            gamma        = (2.222, 4.5),
            output_bps   = 8,
            use_camera_wb=True,
        )
    else:
        cam_wb = raw.camera_whitebalance
        r_s, g_s, b_s = compute_wb_scalers(kelvin, tint)
        user_wb = (cam_wb[0]*r_s, cam_wb[1]*g_s, cam_wb[2]*b_s, cam_wb[3]*g_s)

        rgb8 = raw.postprocess(
            no_auto_bright=True,
            output_color = _map_color_space(color_space),
            gamma        = (2.222, 4.5),
            output_bps   = 8,
            use_camera_wb=False,
            user_wb      = user_wb,
        )

    # OpenCV prefers BGR; convert and normalise to [0,1] float32
    return rgb8[..., ::-1].astype(np.float32) / 255.0

# ----------------------------------------------------------------------
# Batch processing (unchanged except for the raw2bgr call)
# ----------------------------------------------------------------------

def process_images(file_list, params, options):
    """Process a list of RAW files using the given parameters and options."""
    for path in file_list:
        if not path.lower().endswith(RAW_EXTS):          # skip non-RAW
            continue

        try:
            with rawpy.imread(path) as raw:
                img = raw2bgr(
                    raw,
                    kelvin=params["temperature"],
                    tint=params["tint"],
                    color_space=options["color_space"],
                )
        except Exception as e:
            print(f"[skip] {os.path.basename(path)}: {e}")
            continue

        # tone / colour adjustments
        img = apply_adjustments(img, params)

        # optional resize
        if options["size"]:
            w, h = options["size"]
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # optional sharpening
        if options["sharpening"] != "None":
            s = {"Low":0.5,"Standard":1.0,"High":1.5}[options["sharpening"]]
            blur = cv2.GaussianBlur(img, (0,0), 1)
            img  = np.clip(img + (img - blur) * s, 0, 1)

        out = (img * 255 + 0.5).astype(np.uint8)
        stem, _ = os.path.splitext(os.path.basename(path))
        name = f"{stem}{options['file_naming']}.{options['format'].lower()}"
        cv2.imwrite(os.path.join(options["dest_folder"], name), out)
        print(f"[saved] {name}")
