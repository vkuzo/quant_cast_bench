"""flexquant HOP — out-of-tree HigherOrderOperator path.

Importing this package fires three side-effects:
  1. Defines flex_cast_quant_dense_with_hop (the HOP) and registers eager +
     ProxyTorchDispatchMode dispatch keys.
  2. Monkey-patches torch._dynamo.variables.higher_order_ops._hop_name_to_variable_class
     so Dynamo dispatches our HOP through FlexQuantHigherOrderVariable.
  3. Registers the Inductor lowering for the HOP.
"""

from .hop import flex_cast_quant_dense_with_hop
from . import dynamo_variable  # registers the Dynamo variable class  # noqa: F401
from . import inductor_lowering  # registers the lowering  # noqa: F401

__all__ = ["flex_cast_quant_dense_with_hop"]
