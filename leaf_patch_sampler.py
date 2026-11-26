from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw


EPS = 1e-8


def _indices(patch):
    """Calculate the RGB measurements for one patch."""
    pixels = np.asarray(patch, dtype=np.float32)
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]

    brightness = (red + green + blue) / 3
    valid = (brightness >= 25) & (brightness <= 245)
    if not valid.any():
        raise ValueError("No valid pixels in patch")

    red = red[valid]
    green = green[valid]
    blue = blue[valid]

    total = red + green + blue + EPS
    exg = 2 * green - red - blue
    gli = exg / (2 * green + red + blue + EPS)
    ngrdi = (green - red) / (green + red + EPS)
    vari = (green - red) / (green + red - blue + EPS)

    return {
        "R_mean": float(red.mean()),
        "G_mean": float(green.mean()),
        "B_mean": float(blue.mean()),
        "r_median": float(np.median(red / total)),
        "g_median": float(np.median(green / total)),
        "b_median": float(np.median(blue / total)),
        "GLI_median": float(np.median(gli)),
        "NGRDI_median": float(np.median(ngrdi)),
        "VARI_median": float(np.median(vari)),
        "ExG_median": float(np.median(exg)),
        "valid_fraction": float(valid.mean()),
    }


def summarize_patches(table, measurements, image_info):
    """Combine the patch measurements into one row for the image."""
    summary = image_info.copy()
    summary["n_patches"] = len(table)

    for name in measurements:
        values = table[name].dropna()
        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        std = 0.0
        if len(values) > 1:
            std = values.std(ddof=1)

        stats = {
            "median": values.median(),
            "mean": values.mean(),
            "std": std,
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
            "min": values.min(),
            "max": values.max(),
        }
        for statistic, value in stats.items():
            summary[f"{name}__{statistic}_across_patches"] = float(value)

    return summary


def save_summary(path, summary):
    """Replace an earlier row for the same tree, timepoint and image."""
    new_row = pd.DataFrame([summary])
    if path.exists():
        old_rows = pd.read_csv(path)
        same_tree = old_rows["tree_id"].astype(str) == summary["tree_id"]
        same_time = old_rows["timepoint"].astype(str) == summary["timepoint"]
        same_image = old_rows["image_name"].astype(str) == summary["image_name"]
        keep = ~(same_tree & same_time & same_image)
        new_row = pd.concat([old_rows[keep], new_row], ignore_index=True)

    new_row.to_csv(path, index=False)


class PatchSampler:
    def __init__(self, image_path, output_root, tree_id, timepoint, species, patch_size=40, n_patches=8, save_patches=True):
        self.image_path = Path(image_path)
        self.output_root = Path(output_root)
        self.tree_id = str(tree_id)
        self.timepoint = str(timepoint)
        self.species = str(species)
        self.patch_size = int(patch_size)
        self.n_patches = int(n_patches)
        self.save_patches = bool(save_patches)

        with Image.open(self.image_path) as image:
            self.image = image.convert("RGB")
        self.points = []
        self.fig = None
        self.ax = None

    def start(self):
        self.fig, self.ax = plt.subplots(figsize=(16, 9))
        self._show_image()
        self.fig.canvas.mpl_connect("button_press_event", self._click)
        plt.show()
        return self

    def _box(self, x, y):
        half = self.patch_size // 2
        return x - half, y - half, x + half, y + half

    def _show_image(self):
        self.ax.imshow(self.image)
        self.ax.axis("off")
        self.ax.set_title(f"{self.tree_id} | {self.timepoint} | {self.species}\n" "Left click: add patch | Right click: remove last")

    def _redraw(self):
        self.ax.clear()
        self._show_image()
        for number, (x, y) in enumerate(self.points, start=1):
            left, top = self._box(x, y)[:2]
            square = Rectangle((left, top), self.patch_size, self.patch_size, fill=False, linewidth=2)
            self.ax.add_patch(square)
            self.ax.text(left, top - 5, str(number), fontsize=10)

        self.fig.canvas.draw_idle()

    def _click(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == 3:  # Right click removes the last patch.
            if self.points:
                self.points.pop()
                self._redraw()
            return

        if event.button != 1 or len(self.points) >= self.n_patches:
            return

        x = int(event.xdata)
        y = int(event.ydata)
        left, top, right, bottom = self._box(x, y)
        width, height = self.image.size
        if left < 0 or top < 0 or right >= width or bottom >= height:
            print("Too close to image border")
            return

        self.points.append((x, y))
        self._redraw()
        print(f"Patch {len(self.points)}/{self.n_patches}")
        if len(self.points) == self.n_patches:
            print("Selection complete. Run sampler.save().")

    def save(self):
        if not self.points:
            raise RuntimeError("No patches selected")

        folder = self.output_root / self.timepoint / self.tree_id
        folder.mkdir(parents=True, exist_ok=True)
        image_info = {"tree_id": self.tree_id, "timepoint": self.timepoint, "species": self.species, "image_name":self.image_path.name, }

        rows = []
        overlay = self.image.copy()
        draw = ImageDraw.Draw(overlay)

        for number, (x, y) in enumerate(self.points, start=1):
            box = self._box(x, y)
            patch = self.image.crop(box)
            measurements = _indices(patch)

            row = image_info.copy()
            row["patch_id"] = number
            row["center_x"] = x
            row["center_y"] = y
            row["patch_size"] = self.patch_size
            row.update(measurements)
            if self.save_patches:
                path = folder / f"patch_{number:02d}.png"
                patch.save(path)
                row["patch_path"] = str(path)
            rows.append(row)

            left, top = box[:2]
            draw.rectangle(box, outline="red", width=3)
            draw.text((left, top - 15), str(number), fill="red")

        patch_df = pd.DataFrame(rows)
        patch_df.to_csv(folder / "patch_indices.csv", index=False)
        summary = summarize_patches(patch_df, measurements.keys(), image_info)
        save_summary(self.output_root / "image_level_summary.csv", summary)

        path = folder / "selected_patches_overlay.tiff"
        overlay.save(path, format="TIFF", compression="tiff_lzw", dpi=(300, 300))
        print("Saved to:", path)
        return patch_df, summary








        
