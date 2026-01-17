from ApplyPalatte import apply_palette
from ApplyPixelise import create_pixelated

import numpy as np
from PIL import Image

def pixelate(
    img_array: np.ndarray,
    palette: str,
    block_size: int,
    method: str,
    apply_edges: bool,
    blend_strength: float,

):
    """
    Pixelates an image array using a specified palette and method.
    
    Args:
        img_array: A numpy array representing the image.
        palette: String identifier for the color palette to use.
        block_size: The size of the pixel blocks.
        method: The statistical method to downsample (e.g., "mode", "mean").
        apply_edges: Whether to enhance edges during the process.
        blend_strength: Float determining the intensity of the effect.
    """

    palette_arr = apply_palette(img_array,palette)
    pixelated_arr = create_pixelated(palette_arr,block_size,method,apply_edges,blend_strength)
    return pixelated_arr

'''
image_path = "Images/flowrs.png"

img = np.array(Image.open(image_path))

style = "stylized"

blocks = 16 

method = "mode"

edges = False

b_strength = 0.5



final_arr = pixelate(img,style,blocks,method,edges,b_strength)
final_img = Image.fromarray(final_arr.astype(np.uint8))
final_img.show()
final_img.save("out.png")

'''