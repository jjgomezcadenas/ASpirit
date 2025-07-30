import numpy as np

def line_from_points(x1, y1, x2, y2):
    """
    Returns a function y(x) representing the line passing through (x1, y1) and (x2, y2)
    """
    if x1 == x2:
        raise ValueError("Vertical line: slope is undefined.")
    
    m = (y2 - y1) / (x2 - x1)  # slope
    b = y1 - m * x1            # intercept

    def y(x):
        return m * x + b

    return y

