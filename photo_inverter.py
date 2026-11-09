"""
Batch TIFF Processor
=====================

Standalone desktop app for a lab worker to batch-process multi-page 16-bit
TIFF stacks: 16->8 bit conversion, inversion, percentile-based level
adjustment, then concatenation of every frame from every selected file
into one combined output TIFF.

Run directly with Python, or package with PyInstaller (see README.md).
"""

import os
import threading
import traceback
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor, as_completed
from tkinter import filedialog, messagebox, ttk

import numpy as np
import tifffile
from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
# Image processing pipeline
# ---------------------------------------------------------------------------

def to_8bit(frame: np.ndarray) -> np.ndarray:
    """Downsample a 16-bit frame to 8-bit by linearly scaling 0-65535 -> 0-255."""
    frame32 = frame.astype(np.float32)
    scaled = frame32 / 65535.0 * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def invert(frame: np.ndarray) -> np.ndarray:
    """Invert an 8-bit image (255 - pixel)."""
    return 255 - frame


def adjust_levels(frame: np.ndarray, percentile: float) -> np.ndarray:
    """
    Set the black point to the given percentile of the frame's histogram and
    stretch the remaining values to fill the full 0-255 range.

    percentile=0 means the black point is the absolute darkest pixel (no-op
    on the low end beyond a standard stretch to the frame's own min/max).
    """
    low = np.percentile(frame, percentile)
    high = frame.max()

    if high <= low:
        # Degenerate/flat frame: nothing to stretch, avoid divide-by-zero.
        return np.zeros_like(frame, dtype=np.uint8)

    stretched = (frame.astype(np.float32) - low) / (high - low) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


def process_frame(frame: np.ndarray, percentile: float) -> np.ndarray:
    """Run the full per-frame pipeline: downsample -> invert -> level adjust."""
    frame = to_8bit(frame)
    frame = invert(frame)
    frame = adjust_levels(frame, percentile)
    return frame


def _read_and_process_file(path, percentile):
    """Open one file, process every frame, and return the list of processed frames."""
    stack = tifffile.imread(path)  # shape (n_frames, H, W) for a multi-page tif
    if stack.ndim == 2:
        # A "multi-page" file that turned out to have only one page.
        stack = stack[np.newaxis, ...]
    return [process_frame(stack[i], percentile) for i in range(stack.shape[0])]


# Leave headroom below the true 4GiB (2**32 bytes) classic-TIFF limit for header/IFD overhead.
CLASSIC_TIFF_SAFE_LIMIT = 3 * 1024**3 + 900 * 1024**2  # ~3.9 GB


def _file_byte_size(path):
    """
    A file's expected 8-bit output byte size (sum of page width*height), found
    by reading only TIFF header/IFD metadata, not pixel data.
    """
    with tifffile.TiffFile(path) as tif:
        return sum(page.shape[0] * page.shape[1] for page in tif.pages)


def _estimate_output_bytes(file_paths):
    return sum(_file_byte_size(path) for path in file_paths)


def _human_size(num_bytes):
    """Format a byte count as a short human-readable string (e.g. '3.9 GB')."""
    value = float(num_bytes)
    for unit in ("bytes", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "bytes" else f"{int(value)} bytes"
        value /= 1024


def _split_into_parts(file_paths, max_bytes):
    """
    Group files (in original order) into batches that each fit under max_bytes,
    without splitting any single file's frames across two output parts.
    """
    parts = []
    current_group = []
    current_bytes = 0
    for path in file_paths:
        size = _file_byte_size(path)
        if size > max_bytes:
            raise ValueError(
                f"'{os.path.basename(path)}' alone produces {_human_size(size)} of output, which exceeds the "
                f"max file size of {_human_size(max_bytes)}. Increase the max size, enable BigTIFF, or "
                "exclude this file."
            )
        if current_group and current_bytes + size > max_bytes:
            parts.append(current_group)
            current_group = []
            current_bytes = 0
        current_group.append(path)
        current_bytes += size
    if current_group:
        parts.append(current_group)
    return parts


def _part_output_path(output_path, part_number, total_parts):
    if total_parts == 1:
        return output_path
    base, ext = os.path.splitext(output_path)
    return f"{base}_part{part_number}of{total_parts}{ext}"


def process_files(
    file_paths, percentile, output_path, progress_callback=None, allow_bigtiff=True, max_bytes_per_file=None
):
    """
    Process every frame of every input file and write the result to output_path.

    Files are read and processed concurrently (each file is independent), then
    written out in the original file order as results become available so the
    whole batch never needs to be held in memory at once.

    If allow_bigtiff is True (default), BigTIFF (64-bit offsets) is used
    automatically if the combined output would exceed the classic 4GB TIFF
    limit. If False, classic TIFF is always used, and the output is instead
    split across multiple numbered files at file boundaries whenever the next
    file would push a part over max_bytes_per_file (which defaults to the
    classic-TIFF safe limit and is capped at it, since a classic TIFF can never
    exceed 4GB regardless).

    progress_callback(files_done, total_files, message) is called after each
    file finishes processing so the GUI can update its progress bar/status text.

    Returns the list of output file path(s) actually written.
    """
    total_files = len(file_paths)
    max_workers = min(8, os.cpu_count() or 4)

    if allow_bigtiff:
        # Streaming writes (vs. one imwrite of a fully-built array) mean tifffile
        # can't see the final size upfront to switch formats itself, so estimate
        # it ourselves from cheap header-only reads.
        use_bigtiff = _estimate_output_bytes(file_paths) > CLASSIC_TIFF_SAFE_LIMIT
        parts = [file_paths]
    else:
        use_bigtiff = False
        # A classic TIFF can never exceed the 4GB safe limit, so cap the
        # user-chosen max there even if they asked for more.
        limit = CLASSIC_TIFF_SAFE_LIMIT if max_bytes_per_file is None else min(max_bytes_per_file, CLASSIC_TIFF_SAFE_LIMIT)
        parts = _split_into_parts(file_paths, limit)

    total_parts = len(parts)
    part_output_paths = [_part_output_path(output_path, i + 1, total_parts) for i in range(total_parts)]

    files_done = 0
    written_paths = []

    try:
        for part_files, part_output_path in zip(parts, part_output_paths):
            written_paths.append(part_output_path)
            pending_results = {}  # file_index (within this part) -> processed frames, awaiting write
            next_to_write = 0

            with tifffile.TiffWriter(part_output_path, bigtiff=use_bigtiff) as writer:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_index = {
                        executor.submit(_read_and_process_file, path, percentile): i
                        for i, path in enumerate(part_files)
                    }

                    try:
                        for future in as_completed(future_to_index):
                            index = future_to_index[future]
                            pending_results[index] = future.result()

                            files_done += 1
                            if progress_callback:
                                progress_callback(
                                    files_done, total_files, f"Processed {os.path.basename(part_files[index])}"
                                )

                            # Flush any run of results that are now ready, in original order.
                            while next_to_write in pending_results:
                                for frame in pending_results.pop(next_to_write):
                                    writer.write(frame, photometric="minisblack", contiguous=True)
                                next_to_write += 1
                    except Exception:
                        # Stop launching new work; already-running workers (at most
                        # max_workers of them) are left to finish, not killed mid-read.
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
    except Exception:
        # Don't leave partial/corrupt output files behind after a failed batch.
        for path in written_paths:
            if os.path.exists(path):
                os.remove(path)
        raise

    return part_output_paths


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

PREVIEW_MAX_SIZE = 260  # max width/height (px) of the preview thumbnail

# Range (in GB) for the "max size per output file" slider, shown when BigTIFF is disabled.
MAX_SIZE_SLIDER_MIN_GB = 0.5
MAX_SIZE_SLIDER_MAX_GB = CLASSIC_TIFF_SAFE_LIMIT / 1024**3  # ~3.9, the classic-TIFF ceiling


class PhotoInverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Batch TIFF Processor")
        self.root.geometry("820x560")
        self.root.resizable(False, False)

        self.selected_files = []
        self.preview_source_frame = None  # raw 16-bit first frame of the first selected file
        self.preview_load_error = None
        self.preview_photo = None  # kept as an attribute so Tk doesn't garbage-collect it

        # --- Layout: left column for controls, right column for preview ---
        left_column = ttk.Frame(root)
        left_column.pack(side="left", fill="both", expand=True, padx=(12, 6), pady=12)

        right_column = ttk.Frame(root)
        right_column.pack(side="left", fill="both", padx=(6, 12), pady=12)

        # --- File selection ---
        file_frame = ttk.LabelFrame(left_column, text="1. Select input files")
        file_frame.pack(fill="x", pady=(0, 6))

        self.select_button = ttk.Button(file_frame, text="Select .tif Files...", command=self.select_files)
        self.select_button.pack(side="left", padx=8, pady=8)

        self.files_label = ttk.Label(file_frame, text="No files selected")
        self.files_label.pack(side="left", padx=8)

        # --- Percentile adjustment ---
        settings_frame = ttk.LabelFrame(left_column, text="2. Percentile level adjustment")
        settings_frame.pack(fill="x", pady=6)

        ttk.Label(settings_frame, text="Black point percentile:").pack(side="left", padx=8, pady=8)

        self.percentile_var = tk.DoubleVar(value=0.0)
        self.percentile_spinbox = ttk.Spinbox(
            settings_frame,
            from_=0,
            to=99,
            increment=0.01,
            textvariable=self.percentile_var,
            width=6,
            command=self.update_preview,
        )
        self.percentile_spinbox.pack(side="left", padx=8)
        # Catch typed changes too (command= only fires on the spinbox arrows).
        self.percentile_var.trace_add("write", lambda *_: self.update_preview())

        # --- Output format ---
        output_format_frame = ttk.LabelFrame(left_column, text="3. Output format")
        output_format_frame.pack(fill="x", pady=6)

        self.allow_bigtiff_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            output_format_frame,
            text="Allow BigTIFF for outputs over 4GB\n(uncheck to split into multiple files instead)",
            variable=self.allow_bigtiff_var,
            command=self._on_bigtiff_toggle,
        ).pack(anchor="w", padx=8, pady=(8, 4))

        # Max size per output file (only used when BigTIFF is disabled).
        self.max_size_var = tk.DoubleVar(value=MAX_SIZE_SLIDER_MAX_GB)  # in GB

        self.max_size_label = ttk.Label(output_format_frame, text="Max size per file:")
        self.max_size_label.pack(anchor="w", padx=8)

        slider_row = ttk.Frame(output_format_frame)
        slider_row.pack(fill="x", padx=8, pady=(0, 8))

        self.max_size_slider = ttk.Scale(
            slider_row,
            from_=MAX_SIZE_SLIDER_MIN_GB,
            to=MAX_SIZE_SLIDER_MAX_GB,
            variable=self.max_size_var,
            command=lambda _=None: self._update_max_size_label(),
        )
        self.max_size_slider.pack(side="left", fill="x", expand=True)

        self.max_size_value_label = ttk.Label(slider_row, text="", width=8)
        self.max_size_value_label.pack(side="left", padx=(8, 0))

        self._update_max_size_label()
        self._on_bigtiff_toggle()  # set initial enabled/disabled state

        # --- Start processing ---
        action_frame = ttk.LabelFrame(left_column, text="4. Process")
        action_frame.pack(fill="x", pady=6)

        self.start_button = ttk.Button(action_frame, text="Start Processing", command=self.start_processing)
        self.start_button.pack(padx=8, pady=8)

        # --- Progress / status ---
        status_frame = ttk.LabelFrame(left_column, text="Status")
        status_frame.pack(fill="both", expand=True, pady=6)

        self.progress_bar = ttk.Progressbar(status_frame, mode="determinate", maximum=100)
        self.progress_bar.pack(fill="x", padx=8, pady=(8, 4))

        self.status_label = ttk.Label(status_frame, text="Idle.")
        self.status_label.pack(padx=8, pady=(0, 8), anchor="w")

        # --- Live preview ---
        preview_frame = ttk.LabelFrame(right_column, text="Preview (first frame)")
        preview_frame.pack(fill="both", expand=True)

        self.preview_label = ttk.Label(
            preview_frame,
            text="Select files to see a preview",
            anchor="center",
            justify="center",
            wraplength=PREVIEW_MAX_SIZE,
        )
        self.preview_label.pack(padx=10, pady=10, expand=True)

    # -- UI callbacks --

    def select_files(self):
        paths = filedialog.askopenfilenames(
            title="Select multi-page .tif files",
            filetypes=[("TIFF files", "*.tif *.tiff")],
        )
        if paths:
            self.selected_files = list(paths)
            self.files_label.config(text=f"{len(self.selected_files)} file(s) selected")
            self._load_preview_source()
            self.update_preview()

    def _load_preview_source(self):
        """Load the first frame of the first selected file for use as the preview source."""
        try:
            first_page = tifffile.imread(self.selected_files[0], key=0)
            self.preview_source_frame = first_page
            self.preview_load_error = None
        except Exception as exc:
            self.preview_source_frame = None
            self.preview_load_error = str(exc)
            self.preview_label.config(image="", text=f"Could not load preview:\n{exc}")

    def update_preview(self):
        if self.preview_source_frame is None:
            return

        try:
            percentile = float(self.percentile_var.get())
        except (tk.TclError, ValueError):
            return
        if not (0 <= percentile <= 99):
            return

        processed = process_frame(self.preview_source_frame, percentile)

        image = Image.fromarray(processed, mode="L")
        image.thumbnail((PREVIEW_MAX_SIZE, PREVIEW_MAX_SIZE))
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview_label.config(image=self.preview_photo, text="")

    def _update_max_size_label(self):
        self.max_size_value_label.config(text=f"{self.max_size_var.get():.1f} GB")

    def _on_bigtiff_toggle(self):
        """Enable the max-size slider only when BigTIFF is disabled (i.e. splitting mode)."""
        state = "disabled" if self.allow_bigtiff_var.get() else "normal"
        self.max_size_slider.config(state=state)
        self.max_size_label.config(state=state)
        self.max_size_value_label.config(state=state)

    def start_processing(self):
        if not self.selected_files:
            messagebox.showwarning("No files", "Please select at least one .tif file first.")
            return

        try:
            percentile = float(self.percentile_var.get())
        except (TypeError, ValueError):
            messagebox.showerror("Invalid input", "Percentile must be a number between 0 and 99.")
            return

        if not (0 <= percentile <= 99):
            messagebox.showerror("Invalid input", "Percentile must be between 0 and 99.")
            return

        output_path = filedialog.asksaveasfilename(
            title="Save combined output as...",
            defaultextension=".tif",
            filetypes=[("TIFF files", "*.tif")],
            initialfile="combined_output.tif",
        )
        if not output_path:
            return

        allow_bigtiff = self.allow_bigtiff_var.get()
        # Convert the slider's GB value to bytes; only meaningful when splitting.
        max_bytes_per_file = int(self.max_size_var.get() * 1024**3)

        self.start_button.config(state="disabled")
        self.select_button.config(state="disabled")
        self.progress_bar["value"] = 0
        self.status_label.config(text="Starting...")

        thread = threading.Thread(
            target=self._run_processing,
            args=(list(self.selected_files), percentile, output_path, allow_bigtiff, max_bytes_per_file),
            daemon=True,
        )
        thread.start()

    def _run_processing(self, files, percentile, output_path, allow_bigtiff, max_bytes_per_file):
        try:
            def progress_callback(step, total, message):
                percent = int(step / total * 100) if total else 0
                self.root.after(0, self._update_progress, percent, message)

            output_paths = process_files(
                files, percentile, output_path, progress_callback, allow_bigtiff, max_bytes_per_file
            )
            self.root.after(0, self._on_success, output_paths)
        except Exception as exc:
            error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self.root.after(0, self._on_error, str(exc), error_text)

    def _update_progress(self, percent, message):
        self.progress_bar["value"] = percent
        self.status_label.config(text=message)

    def _on_success(self, output_paths):
        self.progress_bar["value"] = 100
        self.start_button.config(state="normal")
        self.select_button.config(state="normal")

        if len(output_paths) == 1:
            self.status_label.config(text=f"Done. Saved to: {output_paths[0]}")
            messagebox.showinfo("Complete", f"Processing complete.\nSaved to:\n{output_paths[0]}")
        else:
            listing = "\n".join(output_paths)
            self.status_label.config(text=f"Done. Saved {len(output_paths)} files (split at the 4GB limit).")
            messagebox.showinfo(
                "Complete",
                f"Processing complete.\nOutput was split into {len(output_paths)} files "
                f"because BigTIFF was disabled and the 4GB classic-TIFF limit was reached:\n\n{listing}",
            )

    def _on_error(self, short_message, full_traceback):
        self.status_label.config(text=f"Error: {short_message}")
        self.start_button.config(state="normal")
        self.select_button.config(state="normal")
        print(full_traceback)  # goes to console/log for debugging
        messagebox.showerror("Processing failed", f"An error occurred:\n\n{short_message}")


def main():
    root = tk.Tk()
    app = PhotoInverterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
