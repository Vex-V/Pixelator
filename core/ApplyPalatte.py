import json
from tqdm import tqdm
import numpy as np
from pathlib import Path
from PIL import Image



def load_palettes(json_path="palettes.json"):
    """
    Load palettes from a JSON file.
    Generates dynamic palettes if missing.
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Palette file not found: {json_path}")

    with open(json_path, "r") as f:
        palettes = json.load(f)

    return palettes


def apply_palette(
    img_array: np.ndarray,
    palette: str,
    chunk_size: int = 64,
):
    """
    Map an image array to the closest colors in a given palette.

    Args:
        img_array: (H, W, 3) uint8 or float array
        palette: palette name
        chunk_size: vertical chunk size

    Returns:
        (H, W, 3) uint8 numpy array
    """
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    palettes = load_palettes("palettes.json")

    if palette not in palettes:
        fallback = next(iter(palettes))
        print(f"palette '{palette}' not found. Defaulting to '{fallback}'.")
        palette = fallback

    img = img_array.astype(np.float32)
    pal = np.array(palettes[palette], dtype=np.float32)
    pal_u8 = pal.astype(np.uint8)
    pal_sq = (pal * pal).sum(axis=1)

    H, W, _ = img.shape
    output = np.empty((H, W, 3), dtype=np.uint8)

    num_chunks = (H + chunk_size - 1) // chunk_size

    for i in tqdm(range(num_chunks), desc="Mapping Pixels", unit="chunk"):
        y0 = i * chunk_size
        y1 = min(y0 + chunk_size, H)

        # |x-p|^2 = |x|^2 - 2x.p + |p|^2; |x|^2 is constant per pixel, so drop it
        pixels = img[y0:y1].reshape(-1, 3)
        closest = np.argmin(pal_sq - 2 * pixels @ pal.T, axis=1)
        output[y0:y1] = pal_u8[closest].reshape(y1 - y0, W, 3)

    return output


