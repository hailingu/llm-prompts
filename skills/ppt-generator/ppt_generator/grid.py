"""Grid and layout helpers."""
from typing import Dict, Tuple

from .helpers import px_to_inches


class GridSystem:
    """Simple grid system to compute columns and usable area."""

    def __init__(self, spec: Dict) -> None:
        grid = (spec.get('design_system', {}).get('grid_system')
                or spec.get('grid_system', {}))
        self.slide_w = grid.get('slide_width_inches', 13.333)
        self.slide_h = grid.get('slide_height_inches', 7.5)
        self.margin_h = px_to_inches(grid.get('margin_horizontal', 80))
        self.gutter = px_to_inches(grid.get('gutter', 24))
        self.columns = grid.get('columns', 12)
        self.usable_w = self.slide_w - 2 * self.margin_h
        self.col_w = (self.usable_w - self.gutter * (self.columns - 1)) / self.columns

    def col_span(self, n_cols: int, start_col: int = 0) -> Tuple[float, float]:
        """Return (left_in_inches, width_in_inches) for n_cols starting at start_col."""
        left = self.margin_h + start_col * (self.col_w + self.gutter)
        width = n_cols * self.col_w + max(0, n_cols - 1) * self.gutter
        return left, width
