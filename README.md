# Batch TIFF Processor

A standalone desktop app that automates image preparation for a cancer research lab. I built it as a volunteer project, and the lab uses it daily.

## The problem

Lab staff work with long multi-page TIFF files of live-cell images. Preparing them meant going through hundreds to thousands of images and editing each one by hand. This app reduces that work to a few clicks.

## What it does

For every frame of every selected file, the app:

1. Converts the frame from 16-bit to 8-bit
2. Inverts it
3. Adjusts levels, setting the black point at a chosen percentile and stretching the rest to the full range

It then combines all frames into one output TIFF, in the original file order.

## Features

- Simple desktop interface, with no coding needed
- Live preview of the first frame while adjusting the percentile
- Multithreaded processing of input files, with a progress bar
- Streaming, in-order writes, so a whole batch is never held in memory at once
- BigTIFF for outputs over the 4 GB classic TIFF limit, or splitting into numbered files
- Partial output files are removed if a batch fails

## Running it

Requires Python 3 with Tkinter.

```bash
pip install -r requirements.txt
python photo_inverter.py
```

## Building a standalone app

```bash
pyinstaller photo_inverter.spec
```

The packaged app is created in `dist/`.
