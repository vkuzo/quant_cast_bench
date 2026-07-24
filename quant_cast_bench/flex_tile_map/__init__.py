"""flex_tile_map package.

Importing the package fires the registration side-effects for both HOP paths and auto-installs the
mm -> flex_gemm fusion pass:
  - ``.hop``           the hand-rolled Triton-template HOP (+ its Dynamo variable and lowering).
  - ``.reference_hop`` the fusible BaseHOP for the REFERENCE backend.
  - ``.flex_gemm_to_tile_map_fusion`` the post-grad fusion+inline pass; importing it calls
                       ``_auto_install()``, which claims Inductor's
                       ``post_grad_custom_post_pass`` slot.
"""

from . import hop  # noqa: F401  registers the Triton-template HOP path
from . import reference_hop  # noqa: F401  registers the fusible reference HOP
from . import (  # noqa: F401  auto-installs the post-grad fusion pass
    flex_gemm_to_tile_map_fusion,
)

flex_gemm_to_tile_map_fusion._auto_install()
