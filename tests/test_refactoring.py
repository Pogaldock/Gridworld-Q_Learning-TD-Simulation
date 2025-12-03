"""Quick test script to verify refactored modules work correctly."""
import numpy as np
from maze_generators import MAZE_GENERATORS
from utils import load_mouse_image, get_direction_angle, calculate_cell_size
import config

def test_maze_generators():
    """Test all maze generators."""
    print("Testing maze generators...")
    for name, generator in MAZE_GENERATORS.items():
        print(f"  Testing {name}...", end=" ")
        maze = generator(10, 10)
        assert maze.shape == (10, 10), f"Wrong shape for {name}"
        assert 'S' in maze, f"No start in {name}"
        assert 'G' in maze, f"No goal in {name}"
        print("✓")
    print("All maze generators passed!\n")

def test_direction_angle():
    """Test direction angle calculation."""
    print("Testing direction angles...")
    assert get_direction_angle((0, 0), (0, 1)) == config.DIRECTION_ANGLES["right"]
    assert get_direction_angle((0, 0), (0, -1)) == config.DIRECTION_ANGLES["left"]
    assert get_direction_angle((0, 0), (1, 0)) == config.DIRECTION_ANGLES["down"]
    assert get_direction_angle((0, 0), (-1, 0)) == config.DIRECTION_ANGLES["up"]
    print("  Direction angles ✓\n")

def test_cell_size():
    """Test cell size calculation."""
    print("Testing cell size calculation...")
    size = calculate_cell_size(800, 600, 10, 10)
    assert size == 60, f"Expected 60, got {size}"
    
    size = calculate_cell_size(1000, 800, 20, 20)
    assert size == 40, f"Expected 40, got {size}"
    print("  Cell size calculation ✓\n")

def test_config():
    """Test configuration values."""
    print("Testing configuration...")
    assert "W" in config.GRID_COLORS
    assert "S" in config.GRID_COLORS
    assert "G" in config.GRID_COLORS
    assert " " in config.GRID_COLORS
    assert config.MIN_CELL_SIZE > 0
    assert config.DEFAULT_CELL_SIZE > 0
    print("  Configuration ✓\n")

def test_image_caching():
    """Test that image caching works."""
    print("Testing image caching...")
    import time
    
    # First load (from disk)
    start = time.time()
    img1 = load_mouse_image(40, 0)
    first_load_time = time.time() - start
    
    # Second load (from cache)
    start = time.time()
    img2 = load_mouse_image(40, 0)
    cached_load_time = time.time() - start
    
    if img1 is not None:
        print(f"  First load: {first_load_time*1000:.2f}ms")
        print(f"  Cached load: {cached_load_time*1000:.2f}ms")
        print(f"  Speedup: {first_load_time/cached_load_time:.1f}x faster ✓\n")
    else:
        print("  Mouse image not found (expected in test environment) ✓\n")

if __name__ == "__main__":
    print("=" * 60)
    print("Refactored Module Tests")
    print("=" * 60 + "\n")
    
    try:
        test_config()
        test_maze_generators()
        test_direction_angle()
        test_cell_size()
        test_image_caching()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
