import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm
import torch



def pad_image_to_block(img_array: np.ndarray, block_size: int) -> np.ndarray:
    """
    Pads image so height and width are divisible by block_size.
    Padding uses edge values to avoid artificial edges.
    """
    h, w, c = img_array.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size

    return np.pad(
        img_array,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="edge"
    )

def pixelate_image(    img_array,
    block_size,
    method="mode",

):
    device="cuda"
    chunk_rows=64

    img = torch.from_numpy(img_array).to(device)
    H, W, C = img.shape
    b = block_size

    Hc = H // b * b
    Wc = W // b * b
    img = img[:Hc, :Wc]

    blocks = img.view(Hc // b, b, Wc // b, b, C)
    blocks = blocks.permute(0, 2, 1, 3, 4)
    blocks = blocks.reshape(Hc // b, Wc // b, b * b, C)

    colors = torch.empty(
        (Hc // b, Wc // b, 1, C),
        device=device,
        dtype=img.dtype
    )

    for i in tqdm(
        range(0, Hc // b, chunk_rows),
        desc="Pixelating",
        unit="chunk"
    ):
        chunk = blocks[i:i + chunk_rows]

        if method == "mean":
            colors[i:i + chunk_rows] = (chunk.float().mean(dim=2, keepdim=True).round().clamp(0, 255).to(torch.uint8) )
        else:
            colors[i:i + chunk_rows] = torch.mode(chunk, dim=2, keepdim=True).values

    output = (
        colors.expand(-1, -1, b * b, -1)
        .reshape(Hc // b, Wc // b, b, b, C)
        .permute(0, 2, 1, 3, 4)
        .reshape(Hc, Wc, C)
    )

    result = img_array.copy()
    result[:Hc, :Wc] = output.cpu().numpy()
    return result

def extract_pixelated_edges(
    img_array: np.ndarray,
    block_size: int,
    edge_threshold: int = 30,
    density_threshold: float = 0.15
):
    """
    Extracts high-resolution edges and snaps them to the block grid.
    Returns a boolean edge mask and the high-res debug edge image.
    """
    img = Image.fromarray(img_array.astype(np.uint8))
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_data = np.array(edges)

    binary_edges = (edge_data > edge_threshold).astype(np.uint8)
    debug_edge_img = Image.fromarray(binary_edges * 255, mode="L")

    h, w = binary_edges.shape
    b = block_size

    blocks = binary_edges.reshape(h // b, b, w // b, b)
    densities = blocks.mean(axis=(1, 3))

    block_mask = densities > density_threshold
    upscaled_mask = np.kron(block_mask, np.ones((b, b)))

    return upscaled_mask.astype(bool), debug_edge_img


def blend_edges(
    img_array: np.ndarray,
    edge_mask: np.ndarray,
    blend_strength: float = 0.5,
    edge_color=(0, 0, 0)
) -> np.ndarray:
    """
    Soft-blends an edge color into the image using the provided mask.
    """
    output = img_array.astype(float)
    edge_layer = np.full_like(output, edge_color, dtype=float)

    idx = np.where(edge_mask)
    output[idx] = (
        output[idx] * (1 - blend_strength)
        + edge_layer[idx] * blend_strength
    )

    return output.astype(np.uint8)




def create_pixelated(
    img_array: np.ndarray,
    block_size: int,
    method: str,
    apply_edges: bool,
    blend_strength: float,

):
    """
    DRIVER FUNCTION

    Parameters
    ----------
    img_array : np.ndarray
        Input image array (H, W, 3)
    block_size : int
        Pixelation block size
    method : str
        "mean" or "mode"
    apply_edges : bool
        Enable edge preservation
    edge_threshold : int
        Edge detection sensitivity
    density_threshold : float
        Edge density per block
    blend_strength : float
        Edge blend intensity
    edge_color : tuple
        RGB edge color
    return_debug_edges : bool
        If True, also returns high-res edge image

    Returns
    -------
    np.ndarray or (np.ndarray, PIL.Image)
        Final pixelated image (and optional debug edge image)
    """
    edge_color=(0, 0, 0),
    return_debug_edges: bool = False
    edge_threshold: int = 40,
    density_threshold: float = 0.15,

    padded = pad_image_to_block(img_array, block_size)
    pixelated = pixelate_image(padded, block_size, method)



    if apply_edges:
        edge_mask, debug_edges = extract_pixelated_edges(
            padded,
            block_size,
            edge_threshold,
            density_threshold
        )

        pixelated = blend_edges(
            pixelated,
            edge_mask,
            blend_strength,
            edge_color
        )

        if return_debug_edges:
            return pixelated, debug_edges

    return pixelated




