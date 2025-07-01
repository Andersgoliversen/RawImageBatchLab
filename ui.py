# ui.py – Tkinter front-end for batch RAW editing
# 2025-07-01: hardened update_preview + minor usability tweaks
import os, math, json
import tkinter as tk
from tkinter import filedialog, ttk

import rawpy, cv2, numpy as np
from PIL import Image, ImageTk

import pipeline
from pipeline import raw2bgr, RAW_EXTS
from adjustments import apply_adjustments

PREVIEW_MAX = 800  # longest edge in pixels for live preview

# ---------- default save options ----------
options = {
    "dest_folder": os.getcwd(),
    "file_naming": "_edited",
    "format": "JPEG",
    "color_space": "sRGB",
    "size": None,
    "sharpening": "None",
}

# ---------- GUI scaffolding ----------
root = tk.Tk()
root.title("Image Batch Lab")

frame = ttk.Frame(root, padding=10)
frame.grid(sticky="nsew")
frame.columnconfigure(0, weight=0)
frame.columnconfigure(1, weight=1)

DEFAULT_TEMP, DEFAULT_TINT = 5050, 8
selected_images, preview_idx = [], 0
preview_win = preview_label = preview_change_btn = None
status_var = tk.StringVar(value="No images loaded")

# ---------- slider helper ----------
r = 0
def add_slider(label, frm, to, res, default):
    global r
    ttk.Label(frame, text=label).grid(row=r, column=0, sticky="w")
    var = tk.DoubleVar(value=default)
    ent = tk.DoubleVar(value=default)

    def on_scale(v):
        ent.set(round(float(v), 2)); update_preview()

    s = ttk.Scale(frame, from_=frm, to=to, orient="horizontal",
                  variable=var, command=on_scale)
    s.grid(row=r, column=1, sticky="ew", padx=5)
    e = ttk.Entry(frame, textvariable=ent, width=6, justify="center")
    e.grid(row=r, column=2, padx=(2, 0))

    def on_enter(_=None):
        try:
            v = float(ent.get()); v = max(min(v, to), frm)
            var.set(v); update_preview()
        except ValueError: pass
    e.bind("<Return>", on_enter)
    r += 1
    return s, ent

# ---------- build all sliders ----------
temp_s, temp_e             = add_slider("Temperature (K)", 2000, 50000, 100, DEFAULT_TEMP)
tint_s, tint_e             = add_slider("Tint",            -100, 100, 1, DEFAULT_TINT)
exp_s,  exp_e              = add_slider("Exposure (stops)", -5,   5, 0.1, 0)
con_s,  con_e              = add_slider("Contrast",        -100, 100, 1, 0)
hi_s,   hi_e               = add_slider("Highlights",      -100, 100, 1, 0)
sh_s,   sh_e               = add_slider("Shadows",         -100, 100, 1, 0)
wh_s,   wh_e               = add_slider("Whites",          -100, 100, 1, 0)
bl_s,   bl_e               = add_slider("Blacks",          -100, 100, 1, 0)
tex_s,  tex_e              = add_slider("Texture",         -100, 100, 1, 0)
cla_s,  cla_e              = add_slider("Clarity",         -100, 100, 1, 0)
deh_s,  deh_e              = add_slider("Dehaze",          -100, 100, 1, 0)
vib_s,  vib_e              = add_slider("Vibrance",        -100, 100, 1, 0)
sat_s,  sat_e              = add_slider("Saturation",      -100, 100, 1, 0)

def gather_params():
    return {
        "temperature": temp_s.get(), "tint": tint_s.get(),
        "exposure": exp_s.get(),     "contrast": con_s.get(),
        "highlights": hi_s.get(),    "shadows": sh_s.get(),
        "whites": wh_s.get(),        "blacks": bl_s.get(),
        "texture": tex_s.get(),      "clarity": cla_s.get(),
        "dehaze": deh_s.get(),       "vibrance": vib_s.get(),
        "saturation": sat_s.get(),
    }

# ---------- preview helpers ----------
def create_preview_window():
    global preview_win, preview_label, preview_change_btn
    preview_win = tk.Toplevel(root); preview_win.title("Live Preview")
    preview_label = ttk.Label(preview_win, text="No image"); preview_label.pack(padx=5, pady=5)
    preview_change_btn = ttk.Button(preview_win, text="Next Image", command=change_preview_image)
    preview_change_btn.pack(pady=5); preview_change_btn.state(["disabled"])

def update_preview():
    if not (selected_images and preview_win and preview_win.winfo_exists()):
        if preview_label: preview_label.config(image="", text="No image selected."); preview_label.image = None
        return
    path = selected_images[preview_idx]

    # ---- decode RAW ----
    try:
        with rawpy.imread(path) as raw:
            img_base = raw2bgr(raw,
                               kelvin=temp_s.get(),
                               tint=tint_s.get(),
                               color_space=options["color_space"])
    except Exception as e:
        preview_label.config(image="", text=f"Decode error: {e}")
        preview_label.image = None; return
    if img_base is None or getattr(img_base, "size", 0) == 0:
        preview_label.config(image="", text="Decode error: no pixel data"); return

    h, w = img_base.shape[:2]
    if h == 0 or w == 0:
        preview_label.config(image="", text="Decode error: zero-sized image"); return

    scale = min(PREVIEW_MAX/w, PREVIEW_MAX/h, 1.0)
    img_prev = cv2.resize(img_base, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else img_base
    img_adj  = apply_adjustments(img_prev, gather_params())

    disp = (np.clip(img_adj, 0, 1)*255).astype(np.uint8)[..., ::-1]
    imgtk = ImageTk.PhotoImage(Image.fromarray(disp))
    preview_label.config(image=imgtk, text=""); preview_label.image = imgtk

def change_preview_image():
    global preview_idx
    if selected_images:
        preview_idx = (preview_idx + 1) % len(selected_images); update_preview()

# ---------- commands ----------
def on_add_images():
    global selected_images, preview_idx
    files = filedialog.askopenfilenames(
        title="Select RAW images",
        filetypes=[("RAW files", " ".join(f"*{e}" for e in RAW_EXTS)), ("All files", "*.*")]
    )
    if not files: return
    selected_images = [p for p in files if p.lower().endswith(RAW_EXTS)]
    if not selected_images: return
    preview_idx = 0
    status_var.set(f"Editing {len(selected_images)} image" + ("" if len(selected_images)==1 else "s"))
    if not (preview_win and preview_win.winfo_exists()): create_preview_window()
    preview_change_btn.state(["!disabled"] if len(selected_images) > 1 else ["disabled"])
    update_preview()

# (Save-options, preset load/save, buttons, status bar … remain unchanged)
# ---------- buttons ----------
btn = ttk.Frame(frame); btn.grid(row=r, column=0, columnspan=2, pady=10, sticky="ew")
ttk.Button(btn, text="Add Images", command=on_add_images).pack(side="left", padx=5)
ttk.Button(btn, text="Process Images",
           command=lambda: pipeline.process_images(selected_images, gather_params(), options)
          ).pack(side="left", padx=5)
ttk.Button(btn, text="Save Options", command=lambda: on_save_options()).pack(side="left", padx=5)
ttk.Button(btn, text="Save Preset", command=lambda: on_save_preset()).pack(side="left", padx=5)
ttk.Button(btn, text="Load Preset", command=lambda: on_load_preset()).pack(side="left", padx=5)

r += 1
ttk.Label(frame, textvariable=status_var).grid(row=r, column=0, columnspan=2, sticky="w")

root.mainloop()
