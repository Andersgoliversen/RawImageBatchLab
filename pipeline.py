"""RAW-to-image batch pipeline plus live-preview helper."""
import os

import cv2
import numpy as np
import rawpy
from PIL import Image, ImageCms

from adjustments import (
    NEUTRAL_TEMP,
    NEUTRAL_TINT,
    apply_adjustments,
    compute_wb_scalers,
)

RAW_EXTS = (
    ".nef", ".cr2", ".crw", ".raf", ".dng",
    ".dcr", ".mrw", ".orf", ".pef", ".srf",
    ".arw", ".rw2", ".rwl", ".srw", ".x3f",
)

_COLOR_MAP = {
    "sRGB": rawpy.ColorSpace.sRGB,
    "Adobe": rawpy.ColorSpace.Adobe,
    "ProPhoto": rawpy.ColorSpace.ProPhoto,
}


def _map_color_space(name):
    return _COLOR_MAP.get(name, rawpy.ColorSpace.sRGB)


def _rawpy_attr(group, name, fallback):
    return getattr(getattr(rawpy, group), name, fallback)


def _srgb_icc_bytes():
    """Return an sRGB ICC profile as bytes, or None if unavailable."""
    try:
        prof = ImageCms.createProfile("sRGB")
        return ImageCms.ImageCmsProfile(prof).tobytes()
    except Exception:
        return None


def raw2bgr(raw, *, kelvin, tint, color_space="sRGB"):
    """Decode a RAW file to float32 BGR in [0, 1].

    The decode favors deterministic, high-quality batch rendering: no per-image
    auto-brightening, 16-bit output, high-quality demosaic, LibRaw highlight
    blending, and camera white balance unless the user moves WB sliders.
    """
    common_kwargs = dict(
        no_auto_bright=True,
        bright=1.0,
        output_color=_map_color_space(color_space),
        gamma=(2.222, 4.5),
        output_bps=16,
        demosaic_algorithm=_rawpy_attr("DemosaicAlgorithm", "AMAZE", rawpy.DemosaicAlgorithm.AHD),
        highlight_mode=_rawpy_attr("HighlightMode", "Blend", rawpy.HighlightMode.Clip),
        fbdd_noise_reduction=_rawpy_attr("FBDDNoiseReductionMode", "Light", rawpy.FBDDNoiseReductionMode.Off),
        median_filter_passes=0,
    )

    if kelvin == NEUTRAL_TEMP and tint == NEUTRAL_TINT:
        rgb16 = raw.postprocess(use_camera_wb=True, **common_kwargs)
    else:
        cam_wb = raw.camera_whitebalance
        r_scale, g_scale, b_scale = compute_wb_scalers(kelvin, tint)
        user_wb = (
            cam_wb[0] * r_scale,
            cam_wb[1] * g_scale,
            cam_wb[2] * b_scale,
            cam_wb[3] * g_scale,
        )
        rgb16 = raw.postprocess(use_camera_wb=False, user_wb=user_wb, **common_kwargs)

    if rgb16 is None or getattr(rgb16, "size", 0) == 0:
        raise RuntimeError("LibRaw returned no pixel data")

    return rgb16[..., ::-1].astype(np.float32) / 65535.0


def _save_jpeg(out8_bgr, dest_path, dpi, color_space):
    rgb = cv2.cvtColor(out8_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)

    save_kwargs = {"quality": 95, "subsampling": 0, "optimize": True}
    if dpi:
        save_kwargs["dpi"] = (int(dpi), int(dpi))

    icc = _srgb_icc_bytes() if color_space == "sRGB" else None
    if icc:
        save_kwargs["icc_profile"] = icc

    im.save(dest_path, format="JPEG", **save_kwargs)


def _save_high_bitdepth(out16_bgr, dest_path, fmt, dpi):
    params = []
    if fmt == "PNG":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    elif fmt == "TIFF" and dpi:
        params = [
            cv2.IMWRITE_TIFF_RESUNIT, 2,  # inches
            cv2.IMWRITE_TIFF_XDPI, int(dpi),
            cv2.IMWRITE_TIFF_YDPI, int(dpi),
        ]
    if not cv2.imwrite(dest_path, out16_bgr, params):
        raise RuntimeError(f"OpenCV failed to write {dest_path}")


def _save_output(img_bgr, dest_path, fmt, dpi, color_space):
    fmt = fmt.upper()
    img_bgr = np.clip(img_bgr, 0.0, 1.0)
    if fmt == "JPEG":
        out8 = (img_bgr * 255.0 + 0.5).astype(np.uint8)
        _save_jpeg(out8, dest_path, dpi, color_space)
    elif fmt in {"PNG", "TIFF"}:
        out16 = (img_bgr * 65535.0 + 0.5).astype(np.uint16)
        _save_high_bitdepth(out16, dest_path, fmt, dpi)
    else:
        raise ValueError(f"Unsupported output format: {fmt}")


def process_images(file_list, params, options):
    os.makedirs(options["dest_folder"], exist_ok=True)

    for path in file_list:
        if not path.lower().endswith(RAW_EXTS):
            continue
        try:
            with rawpy.imread(path) as raw:
                img = raw2bgr(
                    raw,
                    kelvin=params["temperature"],
                    tint=params["tint"],
                    color_space=options["color_space"],
                )
        except Exception as exc:
            print(f"[skip] {os.path.basename(path)} - {exc}")
            continue

        img = apply_adjustments(img, params)

        if options["size"]:
            w, h = options["size"]
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)

        if options["sharpening"] != "None":
            strength = {"Low": 0.35, "Standard": 0.75, "High": 1.10}[options["sharpening"]]
            blur = cv2.GaussianBlur(img, (0, 0), 0.8)
            img = np.clip(img + (img - blur) * strength, 0.0, 1.0)

        stem, _ = os.path.splitext(os.path.basename(path))
        name = f"{stem}{options['file_naming']}.{options['format'].lower()}"
        dest_path = os.path.join(options["dest_folder"], name)

        try:
            _save_output(img, dest_path, options["format"], options.get("dpi", 300), options["color_space"])
        except Exception as exc:
            print(f"[skip] {os.path.basename(path)} - save failed: {exc}")
            continue

        print(f"[saved] {name}")
