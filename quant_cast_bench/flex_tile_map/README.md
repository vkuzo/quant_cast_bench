# flex_tile_map

## context

This is a study of how to express quantization of a tensor in a tile 
invariant way, to inform:

1. what could a general tensor quantization API in PyTorch look like 
   (`flex_tile_map` below), and whether this makes sense to build
2. what are requirements that other flex* projects 
   (flex_gemm, flex_ep, flex_moe) should consider to cover quantization kernel
   authoring
