"""Utility functions for image handling and canvas operations."""
import os
from typing import Optional, Tuple
from PIL import Image, ImageTk
import config


# Image cache to avoid repeated disk I/O
_image_cache = {}


def load_mouse_image(cell_size: int, angle: int = 0) -> Optional[ImageTk.PhotoImage]:
    """Load, resize, and rotate mouse icon with caching.
    
    Args:
        cell_size: Size of the cell for scaling
        angle: Rotation angle in degrees (0=up, 90=right, 180=down, 270=left)
    
    Returns:
        PhotoImage object or None if image not found
    """
    cache_key = (cell_size, angle)
    
    # Return cached image if available
    if cache_key in _image_cache:
        return _image_cache[cache_key]
    
    if not os.path.exists(config.MOUSE_ICON_PATH):
        return None
    
    try:
        img = Image.open(config.MOUSE_ICON_PATH)
        # Resize to fit in cell with padding
        size = max(10, cell_size - 10)
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        
        # Rotate if needed
        if angle != 0:
            img = img.rotate(-angle, expand=False, resample=Image.Resampling.BICUBIC)
        
        mouse_photo = ImageTk.PhotoImage(img)
        
        # Cache the result
        _image_cache[cache_key] = mouse_photo
        
        return mouse_photo
    except Exception as e:
        print(f"Warning: Failed to load mouse icon: {e}")
        return None


def clear_image_cache() -> None:
    """Clear the image cache. Useful when cell size changes significantly."""
    global _image_cache
    _image_cache.clear()


def get_direction_angle(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
    """Get rotation angle based on movement direction.
    
    Args:
        from_pos: Starting position (row, col)
        to_pos: Ending position (row, col)
    
    Returns:
        Angle in degrees: 0=up, 90=right, 180=down, 270=left
    """
    if from_pos == to_pos:
        return config.DIRECTION_ANGLES["right"]
    
    dr = to_pos[0] - from_pos[0]
    dc = to_pos[1] - from_pos[1]
    
    if dc > 0:  # Moving right
        return config.DIRECTION_ANGLES["right"]
    elif dc < 0:  # Moving left
        return config.DIRECTION_ANGLES["left"]
    elif dr > 0:  # Moving down
        return config.DIRECTION_ANGLES["down"]
    elif dr < 0:  # Moving up
        return config.DIRECTION_ANGLES["up"]
    
    return config.DIRECTION_ANGLES["right"]


def calculate_cell_size(canvas_width: int, canvas_height: int, rows: int, cols: int) -> int:
    """Calculate optimal cell size for a given canvas and grid dimensions.
    
    Args:
        canvas_width: Width of canvas in pixels
        canvas_height: Height of canvas in pixels
        rows: Number of grid rows
        cols: Number of grid columns
    
    Returns:
        Cell size in pixels
    """
    if canvas_width <= 1 or canvas_height <= 1:
        return config.DEFAULT_CELL_SIZE
    
    size_by_width = canvas_width // cols
    size_by_height = canvas_height // rows
    
    return max(config.MIN_CELL_SIZE, min(size_by_width, size_by_height))
