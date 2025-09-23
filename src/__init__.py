"""Core algorithms and utilities for linelaser.

This module re-exports commonly used functions so that scripts can import
them concisely, e.g.:

    from ..src import debayer_image, find_circles
"""

# Debayer
from .debayer import debayer_image

# Merge / homography
from .merge import getHomography_bgimage, stitch_images

# Detection and visualization
from .detect_circles import find_circles, detect_circles
from .visualization import plot_circles, plot_matched_circles

# Matching and calibration utilities
from .match import match_all_circles, match_all_circles_normalized
from .calibration_circle import generate_circle_array

__all__ = [
    # debayer
    "debayer_image",
    # merge
    "getHomography_bgimage",
    "stitch_images",
    # detection
    "find_circles",
    "detect_circles",
    # visualization
    "plot_circles",
    "plot_matched_circles",
    # match
    "match_all_circles",
    "match_all_circles_normalized",
    # calibration
    "generate_circle_array",
]
