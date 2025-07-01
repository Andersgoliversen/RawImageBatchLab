"""
ui.py – Modernised Tkinter front-end for RawImageBatchLab
2025-07-01
• Restore:   Save Options window (export settings, like Camera Raw).
• Add:       Save Preset / Load Preset (adjustment parameter JSON).
• Toolbar now: Add Images · Process · Save Options · Save Preset · Load Preset.
"""
import os, json, tkinter as tk
from tkinter import ttk, filedialog

import rawpy, cv2, numpy as np
from PIL import Image, ImageTk

import pipeline
from pipeline import raw2bgr, RAW_EXTS
from adjustments import apply_adjustments

# ----------------------------------------------------------------------
# Appearance
# ----------------------------------------------------------------------
root = tk.Tk()
root.title("Image Batch Lab")

style = ttk.Style(root)
try:                     # flat, neutral-grey theme
    style.theme_use("clam")
except tk.TclError:
    pass
style.configure(".", font=("Segoe UI", 10))

# ----------------------------------------------------------------------
# State
# ----------------------------------------------------------------------
PREVIEW_MAX = 900
DEFAULT_TEMP, DEFAULT_TINT = 5050, 8

selected_images, preview_idx = [], 0
status_var = tk.StringVar(value="No images loaded")

# Export-options dictionary (mirrors Camera Raw “Save Options” panel)
options = {
    "dest_folder": os.getcwd(),
    "file_naming": "_edited",
    "format": "JPEG",          # JPEG / TIFF / PNG
    "color_space": "sRGB",     # sRGB / Adobe / ProPhoto
    "size": None,              # (w,h) or None
    "sharpening": "None",      # None / Low / Standard / High
}

# ----------------------------------------------------------------------
# Layout : sliders | preview
# ----------------------------------------------------------------------
paned = ttk.PanedWindow(root, orient="horizontal")
paned.pack(fill="both", expand=True)

left = ttk.Frame(paned, padding=16); left.columnconfigure(1, weight=1)
right = ttk.Frame(paned, padding=10); right.columnconfigure(0, weight=1); right.rowconfigure(0, weight=1)
paned.add(left, weight=0); paned.add(right, weight=1)

# ----------------------------------------------------------------------
# Slider helper
# ----------------------------------------------------------------------
row = 0
slider_vars = {}

def add_slider(label, frm, to, default, name):
    """Create a labelled Scale + Entry row and register its vars."""
    global row
    ttk.Label(left, text=label).grid(row=row, column=0, sticky="w")
    var = tk.DoubleVar(value=default)
    ent = tk.StringVar(value=str(default))

    def on_scale(v):
        ent.set(f"{float(v):.2f}"); update_preview()

    s = ttk.Scale(left, from_=frm, to=to, variable=var, command=on_scale)
    s.grid(row=row, column=1, sticky="ew", padx=4)

    e = ttk.Entry(left, textvariable=ent, width=7, justify="center")
    e.grid(row=row, column=2, padx=(2,0))
    def on_entry(_=None):
        try:
            v = float(ent.get()); v = max(min(v, to), frm)
            var.set(v); update_preview()
        except ValueError:
            ent.set(f"{var.get():.2f}")
    e.bind("<Return>", on_entry)

    slider_vars[name] = (var, ent)
    row += 1

# ----------------------------------------------------------------------
# Build sliders (grouped like Camera Raw)
# ----------------------------------------------------------------------
add_slider("Temperature (K)", 2000, 50000, DEFAULT_TEMP, "temperature")
add_slider("Tint",           -100, 100,    DEFAULT_TINT, "tint")
ttk.Separator(left, orient="horizontal").grid(columnspan=3, pady=8, sticky="ew"); row += 1
add_slider("Exposure (stops)", -5,   5, 0, "exposure")
add_slider("Contrast",        -100, 100, 0, "contrast")
ttk.Separator(left, orient="horizontal").grid(columnspan=3, pady=8, sticky="ew"); row += 1
add_slider("Highlights",      -100, 100, 0, "highlights")
add_slider("Shadows",         -100, 100, 0, "shadows")
add_slider("Whites",          -100, 100, 0, "whites")
add_slider("Blacks",          -100, 100, 0, "blacks")
ttk.Separator(left, orient="horizontal").grid(columnspan=3, pady=8, sticky="ew"); row += 1
add_slider("Texture",         -100, 100, 0, "texture")
add_slider("Clarity",         -100, 100, 0, "clarity")
add_slider("Dehaze",          -100, 100, 0, "dehaze")
add_slider("Vibrance",        -100, 100, 0, "vibrance")
add_slider("Saturation",      -100, 100, 0, "saturation")

def gather_params():
    """Return current adjustment parameter dict."""
    return {k: slider_vars[k][0].get() for k in slider_vars}

# ----------------------------------------------------------------------
# Preview widgets
# ----------------------------------------------------------------------
preview_label = ttk.Label(right, text="No image", anchor="center")
preview_label.grid(row=0, column=0, sticky="nsew")
preview_change_btn = ttk.Button(right, text="Next Image")
preview_change_btn.grid(row=1, column=0, pady=6)
preview_change_btn.state(["disabled"])

# ----------------------------------------------------------------------
# Preview logic
# ----------------------------------------------------------------------
def update_preview():
    if not selected_images:
        preview_label.config(text="No image", image=""); preview_change_btn.state(["disabled"]); return

    path = selected_images[preview_idx]
    # 1. Decode RAW
    try:
        with rawpy.imread(path) as raw:
            base = raw2bgr(raw,
                           kelvin=slider_vars["temperature"][0].get(),
                           tint=slider_vars["tint"][0].get(),
                           color_space=options["color_space"])
    except Exception as e:
        preview_label.config(text=f"Decode error: {e}", image=""); return

    if base is None or base.size == 0:
        preview_label.config(text="Decode error: no pixel data", image=""); return

    # 2. Resize if needed
    h, w = base.shape[:2]
    scale = min(PREVIEW_MAX / max(h, w), 1.0)
    if scale < 1.0:
        base = cv2.resize(base, (int(w*scale), int(h*scale)),
                          interpolation=cv2.INTER_AREA)

    # 3. Apply adjustments and display
    adj  = apply_adjustments(base, gather_params())
    disp = (np.clip(adj, 0, 1) * 255).astype(np.uint8)[..., ::-1]
    imgtk = ImageTk.PhotoImage(Image.fromarray(disp))
    preview_label.config(image=imgtk, text=""); preview_label.image = imgtk
    preview_change_btn.state(["!disabled"] if len(selected_images) > 1 else ["disabled"])

def change_preview_image():
    global preview_idx
    preview_idx = (preview_idx + 1) % len(selected_images)
    update_preview()

preview_change_btn.configure(command=change_preview_image)

# ----------------------------------------------------------------------
# File / batch commands
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
# Save Options dialog (export settings)
# ----------------------------------------------------------------------
def on_save_options():
    """Open modal window that edits the global `options` dict."""
    win = tk.Toplevel(root); win.title("Save Options"); win.grab_set()
    win.columnconfigure(1, weight=1)

    # Destination folder
    ttk.Label(win, text="Destination Folder").grid(row=0, column=0, sticky="w")
    dest_var = tk.StringVar(value=options["dest_folder"])
    ttk.Entry(win, textvariable=dest_var, width=40).grid(row=0, column=1, sticky="ew", padx=5, pady=5)
    ttk.Button(win, text="Browse", command=lambda:
               dest_var.set(filedialog.askdirectory(initialdir=dest_var.get()) or dest_var.get())
               ).grid(row=0, column=2, padx=5)

    # File naming suffix
    ttk.Label(win, text="File Naming Suffix").grid(row=1, column=0, sticky="w")
    name_var = tk.StringVar(value=options["file_naming"])
    ttk.Entry(win, textvariable=name_var).grid(row=1, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

    # Format
    ttk.Label(win, text="Format").grid(row=2, column=0, sticky="w")
    fmt_var = tk.StringVar(value=options["format"])
    ttk.Combobox(win, textvariable=fmt_var, state="readonly",
                 values=["JPEG", "TIFF", "PNG"]
                 ).grid(row=2, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

    # Colour space
    ttk.Label(win, text="Color Space").grid(row=3, column=0, sticky="w")
    color_var = tk.StringVar(value=options["color_space"])
    ttk.Combobox(win, textvariable=color_var, state="readonly",
                 values=["sRGB", "Adobe", "ProPhoto"]
                 ).grid(row=3, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

    # Resize
    ttk.Label(win, text="Resize (W×H)").grid(row=4, column=0, sticky="w")
    w_var = tk.StringVar()
    h_var = tk.StringVar()
    if options["size"]:
        w_var.set(str(options["size"][0])); h_var.set(str(options["size"][1]))
    ttk.Entry(win, textvariable=w_var, width=6).grid(row=4, column=1, sticky="w", padx=(5,2), pady=5)
    ttk.Entry(win, textvariable=h_var, width=6).grid(row=4, column=1, sticky="e", padx=(2,5), pady=5)

    # Output sharpening
    ttk.Label(win, text="Output Sharpening").grid(row=5, column=0, sticky="w")
    sharp_var = tk.StringVar(value=options["sharpening"])
    ttk.Combobox(win, textvariable=sharp_var, state="readonly",
                 values=["None", "Low", "Standard", "High"]
                 ).grid(row=5, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

    def save_and_close():
        options["dest_folder"] = dest_var.get()
        options["file_naming"] = name_var.get()
        options["format"]      = fmt_var.get()
        options["color_space"] = color_var.get()
        try:
            w = int(w_var.get()); h = int(h_var.get())
            options["size"] = (w, h) if w > 0 and h > 0 else None
        except ValueError:
            options["size"] = None
        options["sharpening"] = sharp_var.get()
        win.destroy()

    ttk.Button(win, text="Save",   command=save_and_close).grid(row=6, column=1, pady=10)
    ttk.Button(win, text="Cancel", command=win.destroy).grid(row=6, column=2, pady=10)

# ----------------------------------------------------------------------
# Preset handling (adjustment parameters)
# ----------------------------------------------------------------------
def on_save_preset():
    """Write current adjustment parameters to JSON."""
    path = filedialog.asksaveasfilename(
        title="Save Preset", defaultextension=".json",
        filetypes=[("JSON", "*.json")]
    )
    if not path:
        return
    with open(path, "w") as f:
        json.dump(gather_params(), f, indent=2)
    status_var.set(f"Preset saved to {os.path.basename(path)}")

def on_load_preset():
    """Load parameters from JSON and update all sliders."""
    path = filedialog.askopenfilename(
        title="Load Preset", filetypes=[("JSON", "*.json")]
    )
    if not path:
        return
    try:
        with open(path) as f:
            data = json.load(f)
        for k, v in data.items():
            if k in slider_vars:
                slider_vars[k][0].set(v)          # update Scale variable
                slider_vars[k][1].set(f"{v:.2f}") # update Entry display
        update_preview()
        status_var.set(f"Preset loaded: {os.path.basename(path)}")
    except Exception as e:
        status_var.set(f"Failed to load preset: {e}")

# ----------------------------------------------------------------------
# Toolbar
# ----------------------------------------------------------------------
toolbar = ttk.Frame(left, padding=(0, 12, 0, 0))
toolbar.grid(row=row, column=0, columnspan=3, sticky="ew")
toolbar.columnconfigure(0, weight=1)

ttk.Button(toolbar, text="Add Images",    command=on_add_images) .grid(row=0, column=0, padx=4)
ttk.Button(toolbar, text="Process",       command=on_process_images).grid(row=0, column=1, padx=4)
ttk.Button(toolbar, text="Save Options",  command=on_save_options) .grid(row=0, column=2, padx=4)
ttk.Button(toolbar, text="Save Preset",   command=on_save_preset)  .grid(row=0, column=3, padx=4)
ttk.Button(toolbar, text="Load Preset",   command=on_load_preset)  .grid(row=0, column=4, padx=4)

# ----------------------------------------------------------------------
# Status bar
# ----------------------------------------------------------------------
ttk.Label(root, textvariable=status_var, anchor="w").pack(fill="x", padx=8, pady=(4, 8))

# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
root.minsize(720, 480)
root.mainloop()
