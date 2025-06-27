# ui.py
"""
GUI for batch RAW editing with live preview:
- Sliders with numeric Entry centered above each slider
- Add / Process Images
- Save Options, Save/Load Presets
- Live Preview window updating in real time
"""

import os
import math
import tkinter as tk
from tkinter import filedialog, ttk
import json

import rawpy
import cv2
import numpy as np
from PIL import Image, ImageTk

import pipeline
from pipeline import raw2bgr
from adjustments import apply_adjustments

# supported RAW extensions
RAW_EXTS = (".nef", ".cr2", ".crw", ".raf", ".dng",
            ".dcr", ".mrw", ".orf", ".pef", ".srf")

# default save-options
options = {
    'dest_folder': os.getcwd(),
    'file_naming': '_edited',
    'format': 'JPEG',
    'color_space': 'sRGB',
    'size': None,
    'sharpening': 'None'
}

# main window
root = tk.Tk()
root.title("Image Batch Lab")

# main frame for controls
frame = ttk.Frame(root, padding=10)
frame.grid(sticky="nsew")
frame.columnconfigure(0, weight=0)
frame.columnconfigure(1, weight=1)

# current grid row index
r = 0

# Set defaults to match the "As Shot" values from the reference image.
DEFAULT_TEMP = 5050
DEFAULT_TINT = 8

# globals for live preview
selected_images = []
preview_idx = 0
preview_win = None
preview_label = None
preview_change_btn = None

# utility: create slider with centered numeric entry
def add_slider(label, frm, to, res, default):
    global r
    ttk.Label(frame, text=label).grid(row=r, column=0, sticky="w")
    r += 1

    var_entry = tk.StringVar(value=str(default))
    entry = ttk.Entry(frame, textvariable=var_entry, width=6)
    entry.grid(row=r, column=0, columnspan=2)
    r += 1

    var_slider = tk.DoubleVar(value=default)
    def on_slide(val):
        try:
            v = float(val)
            prec = abs(int(-math.log10(res))) if isinstance(res, float) and res < 1.0 else 0
            var_entry.set(str(round(v, prec)))
        except ValueError:
            var_entry.set(val)

    slider = tk.Scale(
        frame, from_=frm, to=to,
        orient="horizontal", resolution=res,
        variable=var_slider, showvalue=False,
        command=lambda val: (on_slide(val), update_preview())
    )
    slider.grid(row=r, column=0, columnspan=2, sticky="ew")
    slider.bind("<ButtonRelease-1>", lambda e: update_preview())

    def on_enter(event):
        try:
            v = float(var_entry.get())
            v = max(frm, min(to, v))
            slider.set(v)
            update_preview()
        except ValueError:
            pass

    entry.bind("<Return>", on_enter)
    r += 1

    return var_slider, var_entry

# create all sliders
temp_scale, temp_entry             = add_slider("Temperature (K)", 2000, 50000, 100, DEFAULT_TEMP)
tint_scale, tint_entry             = add_slider("Tint", -100, 100, 1, DEFAULT_TINT)
exposure_scale, exposure_entry     = add_slider("Exposure (stops)", -5, 5, 0.1, 0.0)
contrast_scale, contrast_entry     = add_slider("Contrast", -100, 100, 1, 0)
highlights_scale, highlights_entry = add_slider("Highlights", -100, 100, 1, 0)
shadows_scale, shadows_entry       = add_slider("Shadows", -100, 100, 1, 0)
whites_scale, whites_entry         = add_slider("Whites", -100, 100, 1, 0)
blacks_scale, blacks_entry         = add_slider("Blacks", -100, 100, 1, 0)
texture_scale, texture_entry       = add_slider("Texture", -100, 100, 1, 0)
clarity_scale, clarity_entry       = add_slider("Clarity", -100, 100, 1, 0)
dehaze_scale, dehaze_entry         = add_slider("Dehaze", -100, 100, 1, 0)
vibrance_scale, vibrance_entry     = add_slider("Vibrance", -100, 100, 1, 0)
saturation_scale, saturation_entry = add_slider("Saturation", -100, 100, 1, 0)

# gather current slider parameters
def gather_params():
    return {
        'temperature': temp_scale.get(),
        'tint': tint_scale.get(),
        'exposure': exposure_scale.get(),
        'contrast': contrast_scale.get(),
        'highlights': highlights_scale.get(),
        'shadows': shadows_scale.get(),
        'whites': whites_scale.get(),
        'blacks': blacks_scale.get(),
        'texture': texture_scale.get(),
        'clarity': clarity_scale.get(),
        'dehaze': dehaze_scale.get(),
        'vibrance': vibrance_scale.get(),
        'saturation': saturation_scale.get(),
    }

# status label
status_var = tk.StringVar(value="No images loaded.")

# create live preview window
def create_preview_window():
    global preview_win, preview_label, preview_change_btn
    preview_win = tk.Toplevel(root)
    preview_win.title("Live Preview")

    preview_label = ttk.Label(preview_win, text="")
    preview_label.pack(padx=5, pady=5)

    preview_change_btn = ttk.Button(
        preview_win, text="Next Image",
        command=change_preview_image
    )
    preview_change_btn.pack(pady=5)
    preview_change_btn.state(["disabled"])

# change preview index
def change_preview_image():
    global preview_idx
    preview_idx = (preview_idx + 1) % len(selected_images)
    update_preview()

# update the live preview
def update_preview():
    if not selected_images or preview_win is None or not preview_win.winfo_exists():
        if preview_label:
            preview_label.config(image='', text="No image selected or preview window closed.")
            preview_label.image = None
        return

    path = selected_images[preview_idx]
    try:
        with rawpy.imread(path) as raw:
            img_base = raw2bgr(
                raw,
                kelvin=temp_scale.get(),
                tint=tint_scale.get(),
                color_space=options["color_space"],
            )
    except Exception as e:
        preview_label.config(text=f"Error: {e}")
        return

    params = gather_params()
    h, w = img_base.shape[:2]
    scale = min(800/w, 800/h, 1.0)
    img_for_preview = (cv2.resize(img_base, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
                       if scale < 1.0 else img_base.copy())

    # apply adjustment chain
    img = apply_adjustments(img_for_preview, params)

    # convert to 8-bit and swap channels once for display
    disp = (img * 255).astype(np.uint8)[..., ::-1]  # BGR->RGB

    pil_img = Image.fromarray(disp)
    imgtk = ImageTk.PhotoImage(pil_img)
    preview_label.config(image=imgtk, text="")
    preview_label.image = imgtk

# handle adding images
def on_add_images():
    global selected_images, preview_idx
    files = filedialog.askopenfilenames(
        title="Select RAW images",
        filetypes=[("RAW files", " ".join(RAW_EXTS)), ("All files", "*.*")]
    )
    if not files:
        return

    selected_images = [p for p in files if p.lower().endswith(RAW_EXTS)]
    preview_idx = 0
    status_var.set(f"Editing {len(selected_images)} image{'s' if len(selected_images)!=1 else ''}")

    if preview_win is None or not preview_win.winfo_exists():
        create_preview_window()
    preview_change_btn.state(["!disabled"] if len(selected_images) > 1 else ["disabled"])
    update_preview()

# save options window
def on_save_options():
    """Open a window for adjusting file saving preferences."""
    win = tk.Toplevel(root)
    win.title("Save Options")
    win.grab_set()

    # -- destination folder --
    dest_var = tk.StringVar(value=options["dest_folder"])

    def choose_dest():
        d = filedialog.askdirectory(initialdir=dest_var.get())
        if d:
            dest_var.set(d)

    ttk.Label(win, text="Destination Folder").grid(row=0, column=0, sticky="w")
    dest_entry = ttk.Entry(win, textvariable=dest_var, width=40)
    dest_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    ttk.Button(win, text="Browse", command=choose_dest).grid(row=0, column=2, padx=5)

    # -- file naming suffix --
    ttk.Label(win, text="File Naming Suffix").grid(row=1, column=0, sticky="w")
    name_var = tk.StringVar(value=options["file_naming"])
    ttk.Entry(win, textvariable=name_var).grid(row=1, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

    # -- output format --
    ttk.Label(win, text="Format").grid(row=2, column=0, sticky="w")
    fmt_var = tk.StringVar(value=options["format"])
    ttk.Combobox(win, textvariable=fmt_var, state="readonly",
                 values=["JPEG", "TIFF", "PNG"]).grid(row=2, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

    # -- colour space --
    ttk.Label(win, text="Color Space").grid(row=3, column=0, sticky="w")
    color_var = tk.StringVar(value=options["color_space"])
    ttk.Combobox(win, textvariable=color_var, state="readonly",
                 values=["sRGB", "Adobe", "ProPhoto"]).grid(row=3, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

    # -- image sizing --
    ttk.Label(win, text="Resize (WxH)").grid(row=4, column=0, sticky="w")
    w_var = tk.StringVar()
    h_var = tk.StringVar()
    if options["size"]:
        w_var.set(str(options["size"][0]))
        h_var.set(str(options["size"][1]))
    ttk.Entry(win, textvariable=w_var, width=6).grid(row=4, column=1, sticky="w", padx=(5,2), pady=5)
    ttk.Entry(win, textvariable=h_var, width=6).grid(row=4, column=1, sticky="e", padx=(2,5), pady=5)

    # -- output sharpening --
    ttk.Label(win, text="Output Sharpening").grid(row=5, column=0, sticky="w")
    sharp_var = tk.StringVar(value=options["sharpening"])
    ttk.Combobox(win, textvariable=sharp_var, state="readonly",
                 values=["None", "Low", "Standard", "High"]).grid(row=5, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

    # save the options back to the global dict
    def save_and_close():
        options["dest_folder"] = dest_var.get()
        options["file_naming"] = name_var.get()
        options["format"] = fmt_var.get()
        options["color_space"] = color_var.get()
        try:
            w = int(w_var.get())
            h = int(h_var.get())
            options["size"] = (w, h) if w > 0 and h > 0 else None
        except ValueError:
            options["size"] = None
        options["sharpening"] = sharp_var.get()
        win.destroy()

    ttk.Button(win, text="Save", command=save_and_close).grid(row=6, column=1, pady=10)
    ttk.Button(win, text="Cancel", command=win.destroy).grid(row=6, column=2, pady=10)

    win.columnconfigure(1, weight=1)


# save preset
def on_save_preset():
    preset = gather_params()
    path = filedialog.asksaveasfilename(
        title="Save Preset", defaultextension=".json", filetypes=[("JSON","*.json")]
    )
    if not path:
        return
    with open(path, "w") as f:
        json.dump(preset, f, indent=2)

# load preset
def on_load_preset():
    path = filedialog.askopenfilename(
        title="Load Preset", filetypes=[("JSON","*.json")]
    )
    if not path:
        return
    with open(path) as f:
        data = json.load(f)
    
    controls = {
        "temperature": (temp_scale, temp_entry),
        "tint":       (tint_scale, tint_entry),
        "exposure":   (exposure_scale, exposure_entry),
        "contrast":   (contrast_scale, contrast_entry),
        "highlights": (highlights_scale, highlights_entry),
        "shadows":    (shadows_scale, shadows_entry),
        "whites":     (whites_scale, whites_entry),
        "blacks":     (blacks_scale, blacks_entry),
        "texture":    (texture_scale, texture_entry),
        "clarity":    (clarity_scale, clarity_entry),
        "dehaze":     (dehaze_scale, dehaze_entry),
        "vibrance":   (vibrance_scale, vibrance_entry),
        "saturation": (saturation_scale, saturation_entry),
    }

    for name, (scale_var, entry_var) in controls.items():
        if name in data:
            value = data[name]
            scale_var.set(value)
            res = scale_var.cget('resolution')
            prec = abs(int(-math.log10(res))) if isinstance(res, float) and res < 1.0 else 0
            entry_var.set(str(round(float(value), prec)))
            
    update_preview()

# buttons
btn_frame = ttk.Frame(frame)
btn_frame.grid(row=r, column=0, columnspan=2, pady=10, sticky="ew")
ttk.Button(btn_frame, text="Add Images",    command=on_add_images).pack(side="left", padx=5)
ttk.Button(btn_frame, text="Process Images",
           command=lambda: pipeline.process_images(selected_images, gather_params(), options)
          ).pack(side="left", padx=5)
ttk.Button(btn_frame, text="Save Options",  command=on_save_options).pack(side="left", padx=5)
ttk.Button(btn_frame, text="Save Preset",   command=on_save_preset).pack(side="left", padx=5)
ttk.Button(btn_frame, text="Load Preset",   command=on_load_preset).pack(side="left", padx=5)

r += 1
ttk.Label(frame, textvariable=status_var).grid(row=r, column=0, columnspan=2, sticky="w")

root.mainloop()
