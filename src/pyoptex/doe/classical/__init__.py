"""
Classical designs: factorial, screening, response surface, Taguchi,
space-filling and mixture designs, and their evaluation.

Every function accepts the factors of any pyoptex module. The exported
``Factor`` is the generic :py:class:`pyoptex.utils.Factor`. To compare a
classical design with an optimal one, create the factors with the
optimizer's own class instead (e.g.
:py:class:`pyoptex.doe.fixed_structure.Factor`), since the optimizers only
accept their own factor class; the classical functions accept it as well.

For two-level factors with explicit levels, the levels are taken in the
order given: the first is the low (-1) level and the second the high (+1)
level. Listing them high-first (e.g. ``levels=[20, 10]``) therefore gives
the other half of a fraction (e.g. D = -ABC instead of D = ABC): a valid
design with the same resolution. This applies to all two-level designs
(full and fractional factorials, Plackett-Burman and supersaturated
designs, split-plot designs, ...), and so also to their foldovers.
"""

# ruff: noqa: F401
from ...utils.factor import Factor
from .box_behnken import box_behnken
from .central_composite import add_axial_block, central_composite
from .definitive_screening import definitive_screening
from .doehlert import doehlert
from .factorial import full_factorial
from .foldover import foldover
from .fractional_factorial import fractional_factorial, min_aberration_generators
from .latin_hypercube import latin_hypercube
from .latin_square import latin_square
from .mixture import extreme_vertices, simplex_centroid, simplex_lattice
from .plackett_burman import plackett_burman
from .run_order import gray_code_order, level_changes
from .space_filling import halton, sobol, uniform
from .split_plot import (
    min_aberration_split_plot_generators,
    split_plot_central_composite,
    split_plot_fractional_factorial,
)
from .supersaturated import supersaturated
from .taguchi import choose_orthogonal_array, crossed_array, taguchi
from .utils import WHOLE_PLOT_COLUMN, randomize
