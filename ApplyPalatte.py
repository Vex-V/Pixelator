import json
import torch
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
    device: str | None = None,
):
    """
    Map an image array to the closest colors in a given palette.

    Args:
        img_array: (H, W, 3) uint8 or float array
        palette: palette name
        palettes_path: path to palettes.json
        chunk_size: vertical chunk size
        device: "cuda", "cpu", or None (auto)

    Returns:
        (H, W, 3) uint8 numpy array
    """
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3] 

    img = torch.as_tensor(img_array, dtype=torch.float32, device=device)
    palettes_path="palettes.json"
    palettes = load_palettes(palettes_path)

    if palette not in palettes:
        print(f"palette '{palette}' not found. Defaulting to '16bit'.")
        palette = "16bit"

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    img = torch.as_tensor(img_array, dtype=torch.float32, device=device)
    palette = torch.tensor(palettes[palette], dtype=torch.float32, device=device)

    H, W, _ = img.shape
    output = torch.empty((H, W, 3), dtype=torch.uint8, device=device)

    num_chunks = (H + chunk_size - 1) // chunk_size

    for i in tqdm(range(num_chunks), desc="Mapping Pixels", unit="chunk"):
        y0 = i * chunk_size
        y1 = min(y0 + chunk_size, H)

        chunk = img[y0:y1] 

        diff = chunk[:, :, None, :] - palette[None, None, :, :]
        distances = torch.linalg.norm(diff, dim=3)

        closest = torch.argmin(distances, dim=2)
        output[y0:y1] = palette[closest].to(torch.uint8)

    return output.cpu().numpy()


