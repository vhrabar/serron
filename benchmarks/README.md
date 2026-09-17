# Benchmarks



## NVIDIA GeForce RTX 5070Ti & AMD Ryzen 9 9950X3D

### Benchmarks
| test                                | params                       | min (ms) | mean (ms) | median (ms) | stddev (ms) | rounds |    ops/s |
| ----------------------------------- | ---------------------------- | -------: | --------: | ----------: | ----------: | -----: | -------: |
| test_forward_throughput             | erosion-1x1x256x256-k3       |    0.010 |     0.012 |       0.012 |       0.002 |  10091 | 82,229.9 |
| test_forward_throughput             | erosion-1x1x256x256-k7       |    0.013 |     0.014 |       0.014 |       0.001 |  36684 | 69,153.7 |
| test_forward_throughput             | erosion-1x1x256x256-k15      |    0.022 |     0.023 |       0.023 |       0.003 |  34771 | 42,560.6 |
| test_forward_throughput             | erosion-8x3x512x512-k3       |    0.181 |     0.183 |       0.183 |       0.005 |   5051 |  5,453.4 |
| test_forward_throughput             | erosion-8x3x512x512-k7       |    0.304 |     0.306 |       0.305 |       0.002 |   3181 |  3,272.0 |
| test_forward_throughput             | erosion-8x3x512x512-k15      |    0.992 |     0.994 |       0.993 |       0.002 |    999 |  1,006.5 |
| test_forward_throughput             | erosion-16x3x1024x1024-k3    |    1.535 |     1.538 |       1.537 |       0.002 |    644 |    650.4 |
| test_forward_throughput             | erosion-16x3x1024x1024-k7    |    2.490 |     2.494 |       2.493 |       0.003 |    400 |    401.0 |
| test_forward_throughput             | erosion-16x3x1024x1024-k15   |    7.843 |     7.867 |       7.850 |       0.052 |    127 |    127.1 |
| test_forward_throughput             | dilation-1x1x256x256-k3      |    0.010 |     0.012 |       0.011 |       0.001 |  36338 | 85,241.7 |
| test_forward_throughput             | dilation-1x1x256x256-k7      |    0.012 |     0.014 |       0.014 |       0.001 |  51707 | 71,433.9 |
| test_forward_throughput             | dilation-1x1x256x256-k15     |    0.021 |     0.023 |       0.023 |       0.001 |  27840 | 43,244.9 |
| test_forward_throughput             | dilation-8x3x512x512-k3      |    0.181 |     0.183 |       0.182 |       0.002 |   5034 |  5,468.0 |
| test_forward_throughput             | dilation-8x3x512x512-k7      |    0.304 |     0.306 |       0.305 |       0.004 |   3209 |  3,266.7 |
| test_forward_throughput             | dilation-8x3x512x512-k15     |    0.992 |     0.994 |       0.993 |       0.006 |   1000 |  1,005.7 |
| test_forward_throughput             | dilation-16x3x1024x1024-k3   |    1.533 |     1.536 |       1.535 |       0.003 |    648 |    651.1 |
| test_forward_throughput             | dilation-16x3x1024x1024-k7   |    2.490 |     2.498 |       2.492 |       0.024 |    400 |    400.3 |
| test_forward_throughput             | dilation-16x3x1024x1024-k15  |    7.843 |     7.886 |       7.870 |       0.052 |    127 |    126.8 |
| test_forward_throughput             | opening-1x1x256x256-k3       |    0.016 |     0.018 |       0.018 |       0.002 |  34400 | 54,935.3 |
| test_forward_throughput             | opening-1x1x256x256-k7       |    0.018 |     0.021 |       0.020 |       0.004 |  41340 | 48,590.4 |
| test_forward_throughput             | opening-1x1x256x256-k15      |    0.036 |     0.038 |       0.037 |       0.003 |  24546 | 26,577.6 |
| test_forward_throughput             | opening-8x3x512x512-k3       |    0.377 |     0.380 |       0.379 |       0.003 |   2592 |  2,631.3 |
| test_forward_throughput             | opening-8x3x512x512-k7       |    0.618 |     0.624 |       0.620 |       0.016 |   1564 |  1,601.6 |
| test_forward_throughput             | opening-8x3x512x512-k15      |    1.977 |     1.980 |       1.979 |       0.004 |    501 |    505.0 |
| test_forward_throughput             | opening-16x3x1024x1024-k3    |    3.063 |     3.069 |       3.066 |       0.016 |    324 |    325.8 |
| test_forward_throughput             | opening-16x3x1024x1024-k7    |    4.973 |     4.976 |       4.975 |       0.003 |    201 |    201.0 |
| test_forward_throughput             | opening-16x3x1024x1024-k15   |   15.677 |    15.684 |      15.681 |       0.011 |     64 |     63.8 |
| test_forward_throughput             | closing-1x1x256x256-k3       |    0.016 |     0.017 |       0.017 |       0.002 |  39479 | 57,275.0 |
| test_forward_throughput             | closing-1x1x256x256-k7       |    0.018 |     0.020 |       0.019 |       0.002 |  36874 | 51,009.7 |
| test_forward_throughput             | closing-1x1x256x256-k15      |    0.036 |     0.038 |       0.037 |       0.005 |  24349 | 26,614.9 |
| test_forward_throughput             | closing-8x3x512x512-k3       |    0.377 |     0.379 |       0.379 |       0.003 |   2599 |  2,638.9 |
| test_forward_throughput             | closing-8x3x512x512-k7       |    0.617 |     0.620 |       0.619 |       0.003 |   1605 |  1,614.2 |
| test_forward_throughput             | closing-8x3x512x512-k15      |    1.976 |     1.978 |       1.978 |       0.002 |    505 |    505.5 |
| test_forward_throughput             | closing-16x3x1024x1024-k3    |    3.063 |     3.066 |       3.065 |       0.007 |    325 |    326.2 |
| test_forward_throughput             | closing-16x3x1024x1024-k7    |    4.970 |     4.974 |       4.972 |       0.006 |    201 |    201.1 |
| test_forward_throughput             | closing-16x3x1024x1024-k15   |   15.676 |    15.682 |      15.679 |       0.010 |     64 |     63.8 |
| test_forward_throughput             | gradient-1x1x256x256-k3      |    0.017 |     0.021 |       0.019 |       0.004 |   4474 | 48,036.0 |
| test_forward_throughput             | gradient-1x1x256x256-k7      |    0.019 |     0.021 |       0.021 |       0.001 |  37665 | 47,403.4 |
| test_forward_throughput             | gradient-1x1x256x256-k15     |    0.037 |     0.039 |       0.038 |       0.001 |  20239 | 25,836.6 |
| test_forward_throughput             | gradient-8x3x512x512-k3      |    0.455 |     0.460 |       0.460 |       0.002 |   2166 |  2,171.8 |
| test_forward_throughput             | gradient-8x3x512x512-k7      |    0.696 |     0.701 |       0.701 |       0.003 |   1426 |  1,426.0 |
| test_forward_throughput             | gradient-8x3x512x512-k15     |    2.054 |     2.058 |       2.058 |       0.002 |    483 |    485.9 |
| test_forward_throughput             | gradient-16x3x1024x1024-k3   |    4.077 |     4.099 |       4.099 |       0.010 |    243 |    243.9 |
| test_forward_throughput             | gradient-16x3x1024x1024-k7   |    5.979 |     6.013 |       6.009 |       0.019 |    167 |    166.3 |
| test_forward_throughput             | gradient-16x3x1024x1024-k15  |   16.685 |    16.751 |      16.748 |       0.047 |     60 |     59.7 |
| test_forward_throughput             | top_hat-1x1x256x256-k3       |    0.017 |     0.020 |       0.019 |       0.004 |  36996 | 51,236.6 |
| test_forward_throughput             | top_hat-1x1x256x256-k7       |    0.019 |     0.021 |       0.021 |       0.003 |  39620 | 46,869.0 |
| test_forward_throughput             | top_hat-1x1x256x256-k15      |    0.037 |     0.039 |       0.039 |       0.001 |  23053 | 25,702.3 |
| test_forward_throughput             | top_hat-8x3x512x512-k3       |    0.453 |     0.458 |       0.458 |       0.002 |   2160 |  2,184.4 |
| test_forward_throughput             | top_hat-8x3x512x512-k7       |    0.692 |     0.699 |       0.698 |       0.005 |   1417 |  1,431.4 |
| test_forward_throughput             | top_hat-8x3x512x512-k15      |    2.051 |     2.057 |       2.056 |       0.003 |    485 |    486.2 |
| test_forward_throughput             | top_hat-16x3x1024x1024-k3    |    4.051 |     4.072 |       4.072 |       0.008 |    246 |    245.6 |
| test_forward_throughput             | top_hat-16x3x1024x1024-k7    |    5.961 |     5.979 |       5.977 |       0.009 |    168 |    167.3 |
| test_forward_throughput             | top_hat-16x3x1024x1024-k15   |   16.668 |    16.695 |      16.691 |       0.020 |     60 |     59.9 |
| test_forward_throughput             | black_hat-1x1x256x256-k3     |    0.017 |     0.019 |       0.019 |       0.002 |  33841 | 52,434.3 |
| test_forward_throughput             | black_hat-1x1x256x256-k7     |    0.019 |     0.021 |       0.021 |       0.002 |  38492 | 47,159.1 |
| test_forward_throughput             | black_hat-1x1x256x256-k15    |    0.037 |     0.039 |       0.038 |       0.002 |  22852 | 25,641.7 |
| test_forward_throughput             | black_hat-8x3x512x512-k3     |    0.452 |     0.458 |       0.457 |       0.007 |   2171 |  2,183.7 |
| test_forward_throughput             | black_hat-8x3x512x512-k7     |    0.692 |     0.700 |       0.698 |       0.012 |   1424 |  1,429.5 |
| test_forward_throughput             | black_hat-8x3x512x512-k15    |    2.051 |     2.060 |       2.056 |       0.016 |    486 |    485.5 |
| test_forward_throughput             | black_hat-16x3x1024x1024-k3  |    4.046 |     4.111 |       4.083 |       0.067 |    245 |    243.2 |
| test_forward_throughput             | black_hat-16x3x1024x1024-k7  |    5.964 |     5.992 |       5.981 |       0.038 |    165 |    166.9 |
| test_forward_throughput             | black_hat-16x3x1024x1024-k15 |   16.679 |    16.711 |      16.700 |       0.028 |     60 |     59.8 |
| test_forward_backward_throughput    | erosion-1x1x256x256          |    0.062 |     0.071 |       0.066 |       0.015 |   1570 | 14,121.4 |
| test_forward_backward_throughput    | erosion-8x3x512x512          |    1.786 |     1.808 |       1.804 |       0.018 |    550 |    553.0 |
| test_forward_backward_throughput    | dilation-1x1x256x256         |    0.061 |     0.076 |       0.066 |       0.021 |  11226 | 13,170.1 |
| test_forward_backward_throughput    | dilation-8x3x512x512         |    1.789 |     1.823 |       1.820 |       0.019 |    541 |    548.7 |
| test_forward_backward_throughput    | opening-1x1x256x256          |    0.109 |     0.165 |       0.167 |       0.018 |   4720 |  6,049.5 |
| test_forward_backward_throughput    | opening-8x3x512x512          |    4.668 |     4.736 |       4.729 |       0.056 |    210 |    211.1 |
| test_forward_backward_throughput    | closing-1x1x256x256          |    0.107 |     0.126 |       0.117 |       0.022 |   4779 |  7,940.3 |
| test_forward_backward_throughput    | closing-8x3x512x512          |    4.651 |     4.691 |       4.682 |       0.038 |    214 |    213.2 |
| test_forward_backward_throughput    | gradient-1x1x256x256         |    0.106 |     0.128 |       0.115 |       0.031 |   3008 |  7,801.9 |
| test_forward_backward_throughput    | gradient-8x3x512x512         |    3.757 |     3.789 |       3.780 |       0.031 |    261 |    263.9 |
| test_forward_backward_throughput    | top_hat-1x1x256x256          |    0.116 |     0.150 |       0.139 |       0.034 |   3420 |  6,648.5 |
| test_forward_backward_throughput    | top_hat-8x3x512x512          |    4.824 |     4.888 |       4.868 |       0.069 |    206 |    204.6 |
| test_forward_backward_throughput    | black_hat-1x1x256x256        |    0.118 |     0.144 |       0.133 |       0.043 |   4779 |  6,965.9 |
| test_forward_backward_throughput    | black_hat-8x3x512x512        |    4.948 |     4.981 |       4.975 |       0.023 |    201 |    200.7 |
| test_dtype_forward_throughput       | erosion-fp32                 |    0.304 |     0.308 |       0.306 |       0.013 |   3159 |  3,245.3 |
| test_dtype_forward_throughput       | erosion-fp16                 |    0.318 |     0.321 |       0.319 |       0.010 |   3018 |  3,114.5 |
| test_dtype_forward_throughput       | erosion-bf16                 |    0.317 |     0.319 |       0.319 |       0.002 |   3003 |  3,130.8 |
| test_dtype_forward_throughput       | dilation-fp32                |    0.304 |     0.307 |       0.305 |       0.007 |   3194 |  3,259.6 |
| test_dtype_forward_throughput       | dilation-fp16                |    0.317 |     0.320 |       0.319 |       0.005 |   3045 |  3,128.7 |
| test_dtype_forward_throughput       | dilation-bf16                |    0.317 |     0.320 |       0.319 |       0.008 |   3047 |  3,126.2 |
| test_se_kind_forward_throughput     | erosion-flat                 |    0.992 |     0.995 |       0.993 |       0.004 |    998 |  1,005.4 |
| test_se_kind_forward_throughput     | erosion-grayscale            |    0.992 |     0.994 |       0.993 |       0.002 |    968 |  1,006.4 |
| test_se_kind_forward_throughput     | dilation-flat                |    0.992 |     0.994 |       0.993 |       0.007 |    999 |  1,006.2 |
| test_se_kind_forward_throughput     | dilation-grayscale           |    0.992 |     0.995 |       0.993 |       0.006 |   1001 |  1,005.4 |
| test_border_forward_throughput      | REFLECT                      |    0.326 |     0.328 |       0.327 |       0.002 |   2961 |  3,051.8 |
| test_border_forward_throughput      | REPLICATE                    |    0.303 |     0.305 |       0.305 |       0.002 |   3217 |  3,275.3 |
| test_border_forward_throughput      | CONSTANT                     |    0.305 |     0.308 |       0.307 |       0.006 |   3156 |  3,244.7 |
| test_layer_training_step_throughput | Erosion2d                    |    2.012 |     2.035 |       2.033 |       0.016 |    457 |    491.4 |
| test_layer_training_step_throughput | Dilation2d                   |    2.007 |     2.032 |       2.025 |       0.023 |    491 |    492.2 |
| test_layer_training_step_throughput | Opening2d                    |    5.360 |    18.004 |      20.568 |       4.920 |    194 |     55.5 |
| test_layer_training_step_throughput | Closing2d                    |    5.429 |    17.934 |      20.531 |       4.921 |    191 |     55.8 |


### Against PyTorch, SciPy & Kornia
| op       | shape        |   k | serron (ms) | torch (ms) | vs torch | scipy (ms) | vs scipy | kornia (ms) | vs kornia | cupy (ms) | vs cupy |
| -------- | ------------ | --: | ----------: | ---------: | -------: | ---------: | -------: | ----------: | --------: | --------: | ------: |
| erosion  | 1x1x512x512  |   3 |       0.011 |      0.014 |    1.25x |      3.953 |  352.28x |       0.060 |     5.37x |     0.029 |   2.55x |
| erosion  | 1x1x512x512  |  15 |       0.045 |      0.063 |    1.38x |      4.108 |   90.69x |       1.025 |    22.62x |     0.059 |   1.30x |
| erosion  | 8x3x512x512  |   7 |       0.311 |      0.470 |    1.51x |    101.772 |  327.33x |       5.886 |    18.93x |     0.637 |   2.05x |
| erosion  | 8x32x256x256 |   5 |       0.643 |      0.988 |    1.54x |    240.197 |  373.43x |       9.068 |    14.10x |     1.290 |   2.01x |
| erosion  | 1x1x512x512  |  31 |       0.036 |      0.228 |    6.24x |      3.932 |  107.78x |       4.122 |   113.00x |     0.110 |   3.02x |
| erosion  | 1x1x512x512  |  63 |       0.044 |      0.858 |   19.63x |      3.440 |   78.67x |      16.351 |   373.90x |     0.218 |   5.00x |
| erosion  | 8x3x512x512  |  31 |       0.506 |      4.860 |    9.60x |     91.716 |  181.26x |         OOM |         - |     2.417 |   4.78x |
| erosion  | 8x3x512x512  |  63 |       0.560 |     18.773 |   33.51x |     86.768 |  154.90x |         OOM |         - |     4.764 |   8.51x |
| erosion  | 8x3x512x512  | 127 |       0.524 |     70.373 |  134.30x |     95.029 |  181.36x |         OOM |         - |     9.456 |  18.05x |
| dilation | 1x1x512x512  |   3 |       0.013 |      0.008 |    0.60x |      4.922 |  386.23x |       0.062 |     4.84x |     0.030 |   2.32x |
| dilation | 1x1x512x512  |  15 |       0.045 |      0.058 |    1.29x |      4.274 |   94.80x |       1.019 |    22.60x |     0.058 |   1.29x |
| dilation | 8x3x512x512  |   7 |       0.319 |      0.347 |    1.09x |    124.270 |  390.09x |       5.895 |    18.50x |     0.643 |   2.02x |
| dilation | 8x32x256x256 |   5 |       0.649 |      0.558 |    0.86x |    293.524 |  451.95x |       9.060 |    13.95x |     1.302 |   2.00x |
| dilation | 1x1x512x512  |  31 |       0.042 |      0.224 |    5.27x |      4.400 |  103.58x |       4.126 |    97.12x |     0.109 |   2.56x |
| dilation | 1x1x512x512  |  63 |       0.044 |      0.851 |   19.24x |      4.741 |  107.25x |      16.317 |   369.16x |     0.228 |   5.16x |
| dilation | 8x3x512x512  |  31 |       0.505 |      4.734 |    9.37x |    131.435 |  260.06x |         OOM |         - |     2.414 |   4.78x |
| dilation | 8x3x512x512  |  63 |       0.572 |     18.646 |   32.59x |    122.621 |  214.32x |         OOM |         - |     4.786 |   8.37x |
| dilation | 8x3x512x512  | 127 |       0.526 |     70.261 |  133.49x |     88.290 |  167.74x |         OOM |         - |     9.452 |  17.96x |

## NVIDIA B200 SXM6 &

### Benchmarks

### Against PyTorch, SciPy & Kornia
