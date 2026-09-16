# Benchmarks



## NVIDIA GeForce RTX 5070Ti & AMD Ryzen 9 9950X3D

### Benchmarks

| test                                | params                       | min (ms) | mean (ms) | median (ms) | stddev (ms) | rounds |    ops/s |
| ----------------------------------- | ---------------------------- | -------: | --------: | ----------: | ----------: | -----: | -------: |
| test_forward_throughput             | erosion-1x1x256x256-k3       |    0.011 |     0.014 |       0.013 |       0.008 |   3628 | 70,358.1 |
| test_forward_throughput             | erosion-1x1x256x256-k7       |    0.013 |     0.015 |       0.015 |       0.002 |  37908 | 66,826.8 |
| test_forward_throughput             | erosion-1x1x256x256-k15      |    0.022 |     0.024 |       0.023 |       0.003 |  32734 | 41,828.9 |
| test_forward_throughput             | erosion-8x3x512x512-k3       |    0.189 |     0.191 |       0.190 |       0.002 |   4950 |  5,243.8 |
| test_forward_throughput             | erosion-8x3x512x512-k7       |    0.310 |     0.312 |       0.312 |       0.005 |   3124 |  3,203.6 |
| test_forward_throughput             | erosion-8x3x512x512-k15      |    0.992 |     0.997 |       0.994 |       0.014 |   1001 |  1,003.4 |
| test_forward_throughput             | erosion-16x3x1024x1024-k3    |    1.538 |     1.546 |       1.542 |       0.025 |    644 |    646.7 |
| test_forward_throughput             | erosion-16x3x1024x1024-k7    |    2.492 |     2.500 |       2.497 |       0.009 |    400 |    400.0 |
| test_forward_throughput             | erosion-16x3x1024x1024-k15   |    7.848 |     7.923 |       7.898 |       0.076 |    126 |    126.2 |
| test_forward_throughput             | dilation-1x1x256x256-k3      |    0.011 |     0.013 |       0.012 |       0.001 |  28146 | 79,121.3 |
| test_forward_throughput             | dilation-1x1x256x256-k7      |    0.013 |     0.015 |       0.014 |       0.002 |  53107 | 67,212.7 |
| test_forward_throughput             | dilation-1x1x256x256-k15     |    0.022 |     0.024 |       0.023 |       0.002 |  30770 | 41,893.7 |
| test_forward_throughput             | dilation-8x3x512x512-k3      |    0.189 |     0.191 |       0.190 |       0.002 |   4861 |  5,234.9 |
| test_forward_throughput             | dilation-8x3x512x512-k7      |    0.310 |     0.313 |       0.312 |       0.003 |   3130 |  3,198.7 |
| test_forward_throughput             | dilation-8x3x512x512-k15     |    0.992 |     0.995 |       0.993 |       0.006 |   1001 |  1,005.4 |
| test_forward_throughput             | dilation-16x3x1024x1024-k3   |    1.537 |     1.540 |       1.540 |       0.003 |    648 |    649.3 |
| test_forward_throughput             | dilation-16x3x1024x1024-k7   |    2.491 |     2.498 |       2.495 |       0.016 |    400 |    400.3 |
| test_forward_throughput             | dilation-16x3x1024x1024-k15  |    7.843 |     7.866 |       7.845 |       0.050 |    127 |    127.1 |
| test_forward_throughput             | opening-1x1x256x256-k3       |    0.017 |     0.019 |       0.018 |       0.002 |  37736 | 52,107.8 |
| test_forward_throughput             | opening-1x1x256x256-k7       |    0.019 |     0.022 |       0.021 |       0.006 |  37651 | 45,715.3 |
| test_forward_throughput             | opening-1x1x256x256-k15      |    0.036 |     0.038 |       0.038 |       0.004 |  22447 | 26,010.4 |
| test_forward_throughput             | opening-8x3x512x512-k3       |    0.383 |     0.387 |       0.385 |       0.010 |   2563 |  2,582.6 |
| test_forward_throughput             | opening-8x3x512x512-k7       |    0.623 |     0.625 |       0.624 |       0.006 |   1574 |  1,599.3 |
| test_forward_throughput             | opening-8x3x512x512-k15      |    1.977 |     1.980 |       1.979 |       0.005 |    504 |    505.0 |
| test_forward_throughput             | opening-16x3x1024x1024-k3    |    3.061 |     3.090 |       3.067 |       0.063 |    325 |    323.7 |
| test_forward_throughput             | opening-16x3x1024x1024-k7    |    4.972 |     4.994 |       4.978 |       0.041 |    199 |    200.3 |
| test_forward_throughput             | opening-16x3x1024x1024-k15   |   15.703 |    15.776 |      15.749 |       0.071 |     64 |     63.4 |
| test_forward_throughput             | closing-1x1x256x256-k3       |    0.017 |     0.019 |       0.018 |       0.003 |  39402 | 52,879.3 |
| test_forward_throughput             | closing-1x1x256x256-k7       |    0.019 |     0.021 |       0.021 |       0.002 |  30176 | 46,771.1 |
| test_forward_throughput             | closing-1x1x256x256-k15      |    0.036 |     0.038 |       0.038 |       0.002 |  24261 | 26,318.9 |
| test_forward_throughput             | closing-8x3x512x512-k3       |    0.383 |     0.385 |       0.385 |       0.006 |   2571 |  2,594.1 |
| test_forward_throughput             | closing-8x3x512x512-k7       |    0.623 |     0.626 |       0.624 |       0.012 |   1583 |  1,597.8 |
| test_forward_throughput             | closing-8x3x512x512-k15      |    1.977 |     1.981 |       1.979 |       0.005 |    505 |    504.9 |
| test_forward_throughput             | closing-16x3x1024x1024-k3    |    3.060 |     3.065 |       3.063 |       0.014 |    326 |    326.2 |
| test_forward_throughput             | closing-16x3x1024x1024-k7    |    4.971 |     4.974 |       4.973 |       0.004 |    201 |    201.1 |
| test_forward_throughput             | closing-16x3x1024x1024-k15   |   15.677 |    15.689 |      15.681 |       0.027 |     64 |     63.7 |
| test_forward_throughput             | gradient-1x1x256x256-k3      |    0.019 |     0.022 |       0.021 |       0.003 |   3517 | 45,730.7 |
| test_forward_throughput             | gradient-1x1x256x256-k7      |    0.021 |     0.023 |       0.022 |       0.001 |  31153 | 44,399.6 |
| test_forward_throughput             | gradient-1x1x256x256-k15     |    0.038 |     0.039 |       0.039 |       0.002 |  22952 | 25,362.6 |
| test_forward_throughput             | gradient-8x3x512x512-k3      |    0.462 |     0.467 |       0.467 |       0.004 |   2121 |  2,140.8 |
| test_forward_throughput             | gradient-8x3x512x512-k7      |    0.703 |     0.706 |       0.706 |       0.005 |   1406 |  1,416.0 |
| test_forward_throughput             | gradient-8x3x512x512-k15     |    2.054 |     2.058 |       2.058 |       0.003 |    485 |    485.8 |
| test_forward_throughput             | gradient-16x3x1024x1024-k3   |    4.081 |     4.105 |       4.105 |       0.010 |    243 |    243.6 |
| test_forward_throughput             | gradient-16x3x1024x1024-k7   |    5.984 |     6.017 |       6.017 |       0.011 |    166 |    166.2 |
| test_forward_throughput             | gradient-16x3x1024x1024-k15  |   16.699 |    16.787 |      16.758 |       0.073 |     60 |     59.6 |
| test_forward_throughput             | top_hat-1x1x256x256-k3       |    0.019 |     0.021 |       0.020 |       0.003 |  32960 | 47,350.5 |
| test_forward_throughput             | top_hat-1x1x256x256-k7       |    0.021 |     0.023 |       0.022 |       0.004 |  36484 | 42,962.9 |
| test_forward_throughput             | top_hat-1x1x256x256-k15      |    0.038 |     0.040 |       0.039 |       0.003 |  22671 | 25,247.4 |
| test_forward_throughput             | top_hat-8x3x512x512-k3       |    0.459 |     0.463 |       0.463 |       0.004 |   2146 |  2,157.9 |
| test_forward_throughput             | top_hat-8x3x512x512-k7       |    0.698 |     0.703 |       0.703 |       0.002 |   1418 |  1,422.4 |
| test_forward_throughput             | top_hat-8x3x512x512-k15      |    2.053 |     2.057 |       2.057 |       0.005 |    486 |    486.1 |
| test_forward_throughput             | top_hat-16x3x1024x1024-k3    |    4.042 |     4.056 |       4.055 |       0.006 |    245 |    246.6 |
| test_forward_throughput             | top_hat-16x3x1024x1024-k7    |    5.957 |     5.968 |       5.967 |       0.009 |    168 |    167.6 |
| test_forward_throughput             | top_hat-16x3x1024x1024-k15   |   16.654 |    16.668 |      16.667 |       0.007 |     60 |     60.0 |
| test_forward_throughput             | black_hat-1x1x256x256-k3     |    0.019 |     0.021 |       0.020 |       0.001 |  34294 | 48,768.8 |
| test_forward_throughput             | black_hat-1x1x256x256-k7     |    0.020 |     0.023 |       0.022 |       0.002 |  36698 | 44,420.7 |
| test_forward_throughput             | black_hat-1x1x256x256-k15    |    0.038 |     0.040 |       0.039 |       0.002 |  20886 | 25,243.9 |
| test_forward_throughput             | black_hat-8x3x512x512-k3     |    0.458 |     0.462 |       0.462 |       0.002 |   2152 |  2,162.3 |
| test_forward_throughput             | black_hat-8x3x512x512-k7     |    0.699 |     0.703 |       0.702 |       0.002 |   1413 |  1,423.2 |
| test_forward_throughput             | black_hat-8x3x512x512-k15    |    2.055 |     2.058 |       2.058 |       0.002 |    485 |    485.8 |
| test_forward_throughput             | black_hat-16x3x1024x1024-k3  |    4.045 |     4.068 |       4.070 |       0.013 |    245 |    245.9 |
| test_forward_throughput             | black_hat-16x3x1024x1024-k7  |    5.942 |     5.956 |       5.954 |       0.008 |    168 |    167.9 |
| test_forward_throughput             | black_hat-16x3x1024x1024-k15 |   16.652 |    16.670 |      16.664 |       0.035 |     60 |     60.0 |
| test_forward_backward_throughput    | erosion-1x1x256x256          |    0.064 |     0.072 |       0.068 |       0.020 |   1836 | 13,795.0 |
| test_forward_backward_throughput    | erosion-8x3x512x512          |    1.832 |     1.858 |       1.858 |       0.009 |    539 |    538.4 |
| test_forward_backward_throughput    | dilation-1x1x256x256         |    0.127 |     0.136 |       0.134 |       0.006 |   5412 |  7,377.2 |
| test_forward_backward_throughput    | dilation-8x3x512x512         |    1.828 |     1.846 |       1.842 |       0.016 |    536 |    541.7 |
| test_forward_backward_throughput    | opening-1x1x256x256          |    0.110 |     0.124 |       0.117 |       0.020 |   5283 |  8,086.6 |
| test_forward_backward_throughput    | opening-8x3x512x512          |    4.725 |     4.759 |       4.757 |       0.023 |    210 |    210.1 |
| test_forward_backward_throughput    | closing-1x1x256x256          |    0.110 |     0.141 |       0.120 |       0.035 |   5586 |  7,080.1 |
| test_forward_backward_throughput    | closing-8x3x512x512          |    4.727 |     4.762 |       4.757 |       0.029 |    211 |    210.0 |
| test_forward_backward_throughput    | gradient-1x1x256x256         |    0.107 |     0.125 |       0.115 |       0.025 |   2186 |  8,017.4 |
| test_forward_backward_throughput    | gradient-8x3x512x512         |    3.791 |     3.809 |       3.805 |       0.015 |    261 |    262.6 |
| test_forward_backward_throughput    | top_hat-1x1x256x256          |    0.117 |     0.132 |       0.126 |       0.019 |   5375 |  7,582.5 |
| test_forward_backward_throughput    | top_hat-8x3x512x512          |    4.886 |     4.925 |       4.919 |       0.029 |    202 |    203.0 |
| test_forward_backward_throughput    | black_hat-1x1x256x256        |    0.120 |     0.188 |       0.207 |       0.038 |   3823 |  5,317.5 |
| test_forward_backward_throughput    | black_hat-8x3x512x512        |    4.988 |     5.033 |       5.022 |       0.081 |    200 |    198.7 |
| test_dtype_forward_throughput       | erosion-fp32                 |    0.310 |     0.315 |       0.312 |       0.012 |   3062 |  3,177.6 |
| test_dtype_forward_throughput       | erosion-fp16                 |    0.318 |     0.323 |       0.320 |       0.013 |   3047 |  3,097.1 |
| test_dtype_forward_throughput       | erosion-bf16                 |    0.318 |     0.320 |       0.319 |       0.005 |   3062 |  3,123.5 |
| test_dtype_forward_throughput       | dilation-fp32                |    0.311 |     0.313 |       0.312 |       0.002 |   3016 |  3,198.4 |
| test_dtype_forward_throughput       | dilation-fp16                |    0.318 |     0.320 |       0.320 |       0.002 |   3032 |  3,120.4 |
| test_dtype_forward_throughput       | dilation-bf16                |    0.318 |     0.320 |       0.320 |       0.005 |   2854 |  3,122.4 |
| test_se_kind_forward_throughput     | erosion-flat                 |    0.992 |     0.994 |       0.993 |       0.003 |    954 |  1,005.9 |
| test_se_kind_forward_throughput     | erosion-grayscale            |    0.992 |     0.994 |       0.994 |       0.002 |    980 |  1,005.9 |
| test_se_kind_forward_throughput     | dilation-flat                |    0.992 |     0.994 |       0.993 |       0.001 |    998 |  1,006.2 |
| test_se_kind_forward_throughput     | dilation-grayscale           |    0.992 |     0.994 |       0.994 |       0.002 |   1000 |  1,005.7 |
| test_border_forward_throughput      | REFLECT                      |    0.332 |     0.334 |       0.334 |       0.004 |   2944 |  2,991.3 |
| test_border_forward_throughput      | REPLICATE                    |    0.310 |     0.313 |       0.312 |       0.006 |   3141 |  3,198.9 |
| test_border_forward_throughput      | CONSTANT                     |    0.313 |     0.315 |       0.314 |       0.003 |   2617 |  3,178.3 |
| test_layer_training_step_throughput | Erosion2d                    |    2.014 |     2.051 |       2.050 |       0.018 |    458 |    487.6 |
| test_layer_training_step_throughput | Dilation2d                   |    2.030 |     2.054 |       2.049 |       0.022 |    486 |    486.9 |
| test_layer_training_step_throughput | Opening2d                    |    5.572 |    18.057 |      20.692 |       4.945 |    185 |     55.4 |
| test_layer_training_step_throughput | Closing2d                    |    5.699 |    18.063 |      20.773 |       4.961 |    185 |     55.4 |


### Against PyTorch, SciPy & Kornia
| op       | shape        |   k | serron (ms) | torch (ms) | vs torch | scipy (ms) | vs scipy | kornia (ms) | vs kornia | cupy (ms) | vs cupy |
| -------- | ------------ | --: | ----------: | ---------: | -------: | ---------: | -------: | ----------: | --------: | --------: | ------: |
| erosion  | 1x1x512x512  |   3 |       0.011 |      0.012 |    1.09x |      3.853 |  361.75x |       0.061 |     5.68x |     0.034 |   3.16x |
| erosion  | 1x1x512x512  |  15 |       0.045 |      0.063 |    1.38x |      4.340 |   95.53x |       1.014 |    22.33x |     0.058 |   1.27x |
| erosion  | 8x3x512x512  |   7 |       0.307 |      0.469 |    1.53x |    100.203 |  326.34x |       5.888 |    19.17x |     0.645 |   2.10x |
| erosion  | 8x32x256x256 |   5 |       0.644 |      0.986 |    1.53x |    239.808 |  372.51x |       9.069 |    14.09x |     1.290 |   2.00x |
| erosion  | 1x1x512x512  |  31 |       0.039 |      0.228 |    5.89x |      3.758 |   97.22x |       4.117 |   106.51x |     0.110 |   2.85x |
| erosion  | 1x1x512x512  |  63 |       0.044 |      0.859 |   19.40x |      3.625 |   81.83x |      16.382 |   369.80x |     0.218 |   4.93x |
| erosion  | 8x3x512x512  |  31 |       0.532 |      4.867 |    9.14x |     91.086 |  171.10x |         OOM |         - |     2.404 |   4.52x |
| erosion  | 8x3x512x512  |  63 |       0.585 |     18.767 |   32.07x |     88.857 |  151.83x |         OOM |         - |     4.760 |   8.13x |
| erosion  | 8x3x512x512  | 127 |       0.732 |     70.377 |   96.20x |     85.474 |  116.83x |         OOM |         - |     9.455 |  12.92x |
| dilation | 1x1x512x512  |   3 |       0.010 |      0.007 |    0.76x |      3.730 |  387.43x |       0.062 |     6.42x |     0.028 |   2.87x |
| dilation | 1x1x512x512  |  15 |       0.045 |      0.058 |    1.29x |      4.417 |   97.89x |       1.020 |    22.61x |     0.058 |   1.28x |
| dilation | 8x3x512x512  |   7 |       0.310 |      0.341 |    1.10x |    103.323 |  333.69x |       5.874 |    18.97x |     0.661 |   2.14x |
| dilation | 8x32x256x256 |   5 |       0.643 |      0.553 |    0.86x |    242.871 |  377.47x |       9.062 |    14.09x |     1.300 |   2.02x |
| dilation | 1x1x512x512  |  31 |       0.043 |      0.224 |    5.25x |      3.645 |   85.41x |       4.096 |    95.98x |     0.110 |   2.58x |
| dilation | 1x1x512x512  |  63 |       0.044 |      0.854 |   19.38x |      3.468 |   78.67x |      16.309 |   370.02x |     0.218 |   4.95x |
| dilation | 8x3x512x512  |  31 |       0.534 |      4.731 |    8.86x |     89.552 |  167.77x |         OOM |         - |     2.406 |   4.51x |
| dilation | 8x3x512x512  |  63 |       0.585 |     18.644 |   31.85x |     86.819 |  148.32x |         OOM |         - |     4.766 |   8.14x |
| dilation | 8x3x512x512  | 127 |       0.734 |     70.262 |   95.79x |     86.006 |  117.25x |         OOM |         - |     9.449 |  12.88x |


## NVIDIA B200 SXM6 &

### Benchmarks

### Against PyTorch, SciPy & Kornia
