from PIL import Image, ImageTk
import tkinter as tk
from pathlib import Path
import math

def select_image_gui(image_paths):
    root = tk.Tk()
    root.title("Select Generated Image")
    root.attributes("-fullscreen", True)

    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()

    selected = {"path": None}

    def on_click(path):
        selected["path"] = path
        root.destroy()

    n_images = len(image_paths)

    # ---------- GRID LOGIC ----------
    if n_images == 2:
        cols = 2
        rows = 1
    else:
        cols = math.ceil(math.sqrt(n_images))
        rows = math.ceil(n_images / cols)

    # ---------- IMAGE SIZE ----------
    max_w = screen_w // cols
    max_h = screen_h // rows

    # Make 2-image view feel bigger
    if n_images == 2:
        max_w = int(max_w * 0.95)
        max_h = int(max_h * 0.95)

    thumbs = []

    for idx, path in enumerate(image_paths):
        img = Image.open(path)

        img.thumbnail((max_w, max_h), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)

        btn = tk.Button(
            root,
            image=tk_img,
            command=lambda p=path: on_click(p),
            borderwidth=0,
            highlightthickness=0
        )

        r = idx // cols
        c = idx % cols
        btn.grid(row=r, column=c, padx=10, pady=10, sticky="nsew")

        thumbs.append(tk_img)  # prevent GC

    # ---------- GRID EXPANSION ----------
    for r in range(rows):
        root.grid_rowconfigure(r, weight=1)
    for c in range(cols):
        root.grid_columnconfigure(c, weight=1)

    root.mainloop()
    return selected["path"]
