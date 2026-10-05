# Pixelator

Turns an image into pixel art.

## Run

```
pip install streamlit numpy pillow tqdm
streamlit run app.py
```

Run it from the repo root (`palettes.json` is loaded by relative path).

## How it works

1. **Palette:** every pixel is replaced by the closest colour in the chosen palette (`palettes.json`).
2. **Blocks:** the image is split into square blocks, and each block is filled with a single colour.
3. **Edges (optional):** dark outlines are drawn on top.

Code is in `core/`: `ApplyPalatte.py` does step 1, `ApplyPixelise.py` does steps 2 and 3.

## Mean vs mode

Decide which colour a block gets.

- **mode:** the most common colour in the block. The colour is always one from the original palette. Keeps colours sharp.
- **mean:** the average colour of the block. Blends colours, so it can produce colours that are not in the original palette.

## Settings

- **Palette:** the set of colours the image is limited to.
- **Block size:** width and height of each block in pixels. Bigger = chunkier, lower detail.
- **Method:** mean or mode (above).
- **Apply edges:** detects edges in the original image and darkens the blocks they fall in.
- **Edge strength:** how dark those blocks get (0 = no change, 0.9 = nearly black). Only shown when edges are on.
- **Brightness / Saturation:** adjust the result after pixelation. 1.0 = unchanged. They update instantly without pressing Run.
