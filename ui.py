"""
ui.py – Modernised Tkinter front-end for RawImageBatchLab
2025-07-01
• Switch to ttk “clam” theme and Segoe UI 10 pt font for a flatter look.
• Single-window layout with ttk.PanedWindow:
      ┌ sliders ┐│── live preview ──│
• Extra spacing / separators for visual grouping.
No functional behaviour was removed.
"""
import os, json, tkinter as tk
from tkinter import ttk, filedialog

import rawpy, cv2, numpy as np
from PIL import Image, ImageTk

import pipeline
from pipeline import raw2bgr, RAW_EXTS
from adjustments import apply_adjustments

# ----------------------------------------------------------------------
# Appearance tweaks – keep tiny, no external dependency
# ----------------------------------------------------------------------
root = tk.Tk()
root.title("Image Batch Lab")

style = ttk.Style(root)
try:
    style.theme_use("clam")                     # flat, light-grey theme
except tk.TclError:
    pass                                        # fallback to default
style.configure(".", font=("Segoe UI", 10))     # system fallback if absent

# *DO NOT* hard-code colours so the theme can decide.

# ----------------------------------------------------------------------
# Constants / state
# ----------------------------------------------------------------------
PREVIEW_MAX = 900
DEFAULT_TEMP, DEFAULT_TINT = 5050, 8

selected_images, preview_idx = [], 0
preview_label = preview_change_btn = None
status_var = tk.StringVar(value="No images loaded")

options = {                    # identical semantics
    "dest_folder": os.getcwd(),
    "file_naming": "_edited",
    "format": "JPEG",
    "color_space": "sRGB",
    "size": None,
    "sharpening": "None",
}

# ----------------------------------------------------------------------
# Layout: PanedWindow → sliders | preview
# ----------------------------------------------------------------------
paned = ttk.PanedWindow(root, orient="horizontal")   # fills whole window
paned.pack(fill="both", expand=True)

# Left – sliders area
left = ttk.Frame(paned, padding=16)
left.columnconfigure(1, weight=1)                   # make scales stretch
paned.add(left, weight=0)                           # fixed-ish width

# Right – preview area
right = ttk.Frame(paned, padding=10)
right.columnconfigure(0, weight=1)
right.rowconfigure(0, weight=1)
paned.add(right, weight=1)

# ----------------------------------------------------------------------
# Helper: add one slider row
# ----------------------------------------------------------------------
row = 0
slider_vars = {}                # name → (scaleVar, entryVar)

def add_slider(label, frm, to, default, name):
    """
    Create a labelled ttk.Scale with a small entry showing its value.
    Store references in slider_vars for gather_params().
    """
    global row
    ttk.Label(left, text=label).grid(row=row, column=0, sticky="w")

    var = tk.DoubleVar(value=default)
    ent_var = tk.StringVar(value=str(default))

    def on_scale(raw_val):
        ent_var.set(f"{float(raw_val):.2f}")
        update_preview()

    scale = ttk.Scale(left, from_=frm, to=to,
                      variable=var, command=on_scale)
    scale.grid(row=row, column=1, sticky="ew", padx=4)

    entry = ttk.Entry(left, textvariable=ent_var, width=7, justify="center")
    entry.grid(row=row, column=2, padx=(2, 0))

    def on_entry(_=None):
        try:
            v = float(ent_var.get())
            v = max(min(v, to), frm)
            var.set(v); update_preview()
        except ValueError:
            ent_var.set(f"{var.get():.2f}")
    entry.bind("<Return>", on_entry)

    slider_vars[name] = (var, ent_var)
    row += 1

# ----------------------------------------------------------------------
# Build sliders (keep order = Camera Raw)
# ----------------------------------------------------------------------
add_slider("Temperature (K)", 2000, 50000, DEFAULT_TEMP, "temperature")
add_slider("Tint",           -100,   100, DEFAULT_TINT, "tint")

ttk.Separator(left, orient="horizontal").grid(columnspan=3, pady=8, sticky="ew"); row += 1

add_slider("Exposure (stops)",  -5,    5, 0, "exposure")
add_slider("Contrast",        -100,  100, 0, "contrast")

ttk.Separator(left, orient="horizontal").grid(columnspan=3, pady=8, sticky="ew"); row += 1

add_slider("Highlights",      -100,  100, 0, "highlights")
add_slider("Shadows",         -100,  100, 0, "shadows")
add_slider("Whites",          -100,  100, 0, "whites")
add_slider("Blacks",          -100,  100, 0, "blacks")

ttk.Separator(left, orient="horizontal").grid(columnspan=3, pady=8, sticky="ew"); row += 1

add_slider("Texture",         -100,  100, 0, "texture")
add_slider("Clarity",         -100,  100, 0, "clarity")
add_slider("Dehaze",          -100,  100, 0, "dehaze")
add_slider("Vibrance",        -100,  100, 0, "vibrance")
add_slider("Saturation",      -100,  100, 0, "saturation")

# ----------------------------------------------------------------------
# Gather current parameters dictionary
# ----------------------------------------------------------------------
def gather_params():
    return {k: slider_vars[k][0].get() for k in slider_vars}

# ----------------------------------------------------------------------
# Preview widgets
# ----------------------------------------------------------------------
preview_label = ttk.Label(right, text="No image", anchor="center")
preview_label.grid(row=0, column=0, sticky="nsew")

preview_change_btn = ttk.Button(right, text="Next Image",
                                command=lambda: change_preview_image())
preview_change_btn.grid(row=1, column=0, pady=6)
preview_change_btn.state(["disabled"])

# ----------------------------------------------------------------------
# Preview logic
# ----------------------------------------------------------------------
def update_preview():
    """Decode current RAW, apply adjustments, show in preview pane."""
    if not selected_images:
        preview_label.config(text="No image", image="")
        preview_change_btn.state(["disabled"])
        return
    path = selected_images[preview_idx]

    # 1 – decode RAW
    try:
        with rawpy.imread(path) as raw:
            img_base = raw2bgr(raw,
                               kelvin=slider_vars["temperature"][0].get(),
                               tint=slider_vars["tint"][0].get(),
                               color_space=options["color_space"])
    except Exception as e:
        preview_label.config(text=f"Decode error: {e}", image="")
        return

    if img_base is None or getattr(img_base, "size", 0) == 0:
        preview_label.config(text="Decode error: no pixel data", image="")
        return

    # 2 – resize for preview
    h, w = img_base.shape[:2]
    scale = min(PREVIEW_MAX / max(h, w), 1.0)
    if scale < 1.0:
        img_base = cv2.resize(img_base, (int(w*scale), int(h*scale)),
                              interpolation=cv2.INTER_AREA)

    # 3 – apply adjustments & display
    img_adj = apply_adjustments(img_base, gather_params())
    disp    = (np.clip(img_adj, 0, 1) * 255).astype(np.uint8)[..., ::-1]
    imgtk   = ImageTk.PhotoImage(Image.fromarray(disp))

    preview_label.config(image=imgtk, text="")
    preview_label.image = imgtk      # keep reference
    preview_change_btn.state(["!disabled"] if len(selected_images) > 1 else ["disabled"])

def change_preview_image():
    global preview_idx
    if selected_images:
        preview_idx = (preview_idx + 1) % len(selected_images)
        update_preview()

# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------
def on_add_images():
    global selected_images, preview_idx
    files = filedialog.askopenfilenames(
        title="Select RAW images",
        filetypes=[("RAW files", " ".join(f"*{e}" for e in RAW_EXTS)),
                   ("All files", "*.*")]
    )
    if not files:
        return
    selected_images = [p for p in files if p.lower().endswith(RAW_EXTS)]
    if not selected_images:
        return
    preview_idx = 0
    status_var.set(f"Loaded {len(selected_images)} image" +
                   ("s" if len(selected_images) != 1 else ""))
    update_preview()

def on_process_images():
    pipeline.process_images(selected_images, gather_params(), options)

# ----------------------------------------------------------------------
# Bottom toolbar
# ----------------------------------------------------------------------
toolbar = ttk.Frame(left, padding=(0, 12, 0, 0))
toolbar.grid(row=row, column=0, columnspan=3, sticky="ew")
toolbar.columnconfigure(0, weight=1)

ttk.Button(toolbar, text="Add Images", command=on_add_images)         .grid(row=0, column=0, padx=4)
ttk.Button(toolbar, text="Process",    command=on_process_images)     .grid(row=0, column=1, padx=4)

ttk.Label(root, textvariable=status_var, anchor="w").pack(fill="x", padx=8, pady=(4, 8))

# ----------------------------------------------------------------------
# Start event loop
# ----------------------------------------------------------------------
root.minsize(720, 480)
root.mainloop()
