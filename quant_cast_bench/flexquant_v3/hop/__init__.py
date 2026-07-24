"""flexquant_v3 HOP -- the TRITON_TEMPLATE backend's out-of-tree HigherOrderOperator path.

Importing this package fires three side-effects:
  1. Defines flex_tile_map_hop (the HOP) and registers its eager +
     ProxyTorchDispatchMode dispatch keys.
  2. Monkey-patches torch._dynamo.variables.higher_order_ops._hop_name_to_variable_class
     so Dynamo dispatches our HOP through FlexTileMapHigherOrderVariable.
  3. Registers the Inductor lowering for the HOP.
"""

from .hop import flex_tile_map_hop
from . import dynamo_variable  # registers the Dynamo variable class  # noqa: F401
from . import inductor_lowering  # registers the lowering  # noqa: F401

__all__ = ["flex_tile_map_hop"]
