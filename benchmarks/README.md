# Benchmarks



## NVIDIA GeForce RTX 5070Ti & AMD Ryzen 9 9950X3D

### Benchmarks

| test                                | params                       | min (ms) | mean (ms) | median (ms) | stddev (ms) | rounds |    ops/s |
| ----------------------------------- | ---------------------------- | -------: | --------: | ----------: | ----------: | -----: | -------: |
| test_forward_throughput             | erosion-1x1x256x256-k3       |    0.011 |     0.013 |       0.012 |       0.001 |  11815 | 78,597.4 |
| test_forward_throughput             | erosion-1x1x256x256-k7       |    0.013 |     0.015 |       0.014 |       0.001 |  41170 | 67,670.6 |
| test_forward_throughput             | erosion-1x1x256x256-k15      |    0.016 |     0.017 |       0.017 |       0.001 |  36564 | 57,316.1 |
| test_forward_throughput             | erosion-1x1x256x256-k31      |    0.016 |     0.018 |       0.018 |       0.001 |  40323 | 55,306.6 |
| test_forward_throughput             | erosion-1x1x256x256-k63      |    0.017 |     0.018 |       0.018 |       0.001 |  39185 | 55,276.8 |
| test_forward_throughput             | erosion-8x3x512x512-k3       |    0.189 |     0.192 |       0.191 |       0.005 |   4938 |  5,206.6 |
| test_forward_throughput             | erosion-8x3x512x512-k7       |    0.311 |     0.314 |       0.312 |       0.007 |   3110 |  3,186.9 |
| test_forward_throughput             | erosion-8x3x512x512-k15      |    0.476 |     0.482 |       0.482 |       0.008 |   2037 |  2,075.0 |
| test_forward_throughput             | erosion-8x3x512x512-k31      |    0.498 |     0.501 |       0.501 |       0.002 |   1962 |  1,996.4 |
| test_forward_throughput             | erosion-8x3x512x512-k63      |    0.548 |     0.550 |       0.550 |       0.002 |   1782 |  1,817.1 |
| test_forward_throughput             | erosion-16x3x1024x1024-k3    |    1.538 |     1.540 |       1.539 |       0.003 |    645 |    649.5 |
| test_forward_throughput             | erosion-16x3x1024x1024-k7    |    2.492 |     2.494 |       2.493 |       0.002 |    399 |    401.0 |
| test_forward_throughput             | erosion-16x3x1024x1024-k15   |    3.748 |     3.806 |       3.800 |       0.032 |    266 |    262.8 |
| test_forward_throughput             | erosion-16x3x1024x1024-k31   |    3.777 |     3.790 |       3.788 |       0.006 |    263 |    263.9 |
| test_forward_throughput             | erosion-16x3x1024x1024-k63   |    4.007 |     4.015 |       4.010 |       0.014 |    248 |    249.1 |
| test_forward_throughput             | dilation-1x1x256x256-k3      |    0.011 |     0.012 |       0.012 |       0.001 |  39683 | 80,838.2 |
| test_forward_throughput             | dilation-1x1x256x256-k7      |    0.013 |     0.015 |       0.014 |       0.002 |  44287 | 67,775.0 |
| test_forward_throughput             | dilation-1x1x256x256-k15     |    0.016 |     0.018 |       0.017 |       0.002 |  38611 | 56,652.5 |
| test_forward_throughput             | dilation-1x1x256x256-k31     |    0.016 |     0.018 |       0.018 |       0.002 |  39124 | 55,081.5 |
| test_forward_throughput             | dilation-1x1x256x256-k63     |    0.017 |     0.018 |       0.018 |       0.001 |  37412 | 55,124.6 |
| test_forward_throughput             | dilation-8x3x512x512-k3      |    0.189 |     0.191 |       0.191 |       0.001 |   4811 |  5,233.4 |
| test_forward_throughput             | dilation-8x3x512x512-k7      |    0.311 |     0.313 |       0.312 |       0.002 |   3110 |  3,199.3 |
| test_forward_throughput             | dilation-8x3x512x512-k15     |    0.477 |     0.481 |       0.480 |       0.003 |   2052 |  2,080.2 |
| test_forward_throughput             | dilation-8x3x512x512-k31     |    0.496 |     0.499 |       0.498 |       0.002 |   1952 |  2,005.8 |
| test_forward_throughput             | dilation-8x3x512x512-k63     |    0.548 |     0.551 |       0.550 |       0.006 |   1791 |  1,815.2 |
| test_forward_throughput             | dilation-16x3x1024x1024-k3   |    1.536 |     1.539 |       1.539 |       0.003 |    646 |    649.6 |
| test_forward_throughput             | dilation-16x3x1024x1024-k7   |    2.492 |     2.494 |       2.493 |       0.004 |    399 |    401.0 |
| test_forward_throughput             | dilation-16x3x1024x1024-k15  |    3.747 |     3.774 |       3.773 |       0.009 |    264 |    265.0 |
| test_forward_throughput             | dilation-16x3x1024x1024-k31  |    3.758 |     3.771 |       3.770 |       0.011 |    265 |    265.1 |
| test_forward_throughput             | dilation-16x3x1024x1024-k63  |    4.007 |     4.011 |       4.010 |       0.006 |    249 |    249.3 |
| test_forward_throughput             | opening-1x1x256x256-k3       |    0.017 |     0.019 |       0.018 |       0.002 |  41546 | 53,970.0 |
| test_forward_throughput             | opening-1x1x256x256-k7       |    0.019 |     0.021 |       0.020 |       0.001 |  26261 | 48,262.1 |
| test_forward_throughput             | opening-1x1x256x256-k15      |    0.024 |     0.026 |       0.025 |       0.004 |  29206 | 38,582.2 |
| test_forward_throughput             | opening-1x1x256x256-k31      |    0.025 |     0.027 |       0.026 |       0.002 |  33289 | 37,513.4 |
| test_forward_throughput             | opening-1x1x256x256-k63      |    0.025 |     0.027 |       0.026 |       0.004 |  31437 | 37,135.2 |
| test_forward_throughput             | opening-8x3x512x512-k3       |    0.377 |     0.379 |       0.379 |       0.003 |   2577 |  2,637.0 |
| test_forward_throughput             | opening-8x3x512x512-k7       |    0.618 |     0.620 |       0.620 |       0.001 |   1595 |  1,612.7 |
| test_forward_throughput             | opening-8x3x512x512-k15      |    0.921 |     0.927 |       0.927 |       0.002 |   1078 |  1,078.3 |
| test_forward_throughput             | opening-8x3x512x512-k31      |    0.960 |     0.964 |       0.963 |       0.005 |   1035 |  1,037.7 |
| test_forward_throughput             | opening-8x3x512x512-k63      |    1.071 |     1.074 |       1.074 |       0.002 |    918 |    930.8 |
| test_forward_throughput             | opening-16x3x1024x1024-k3    |    3.064 |     3.067 |       3.067 |       0.003 |    326 |    326.0 |
| test_forward_throughput             | opening-16x3x1024x1024-k7    |    4.976 |     4.981 |       4.978 |       0.011 |    200 |    200.8 |
| test_forward_throughput             | opening-16x3x1024x1024-k15   |    7.506 |     7.553 |       7.552 |       0.018 |    133 |    132.4 |
| test_forward_throughput             | opening-16x3x1024x1024-k31   |    7.527 |     7.542 |       7.543 |       0.007 |    133 |    132.6 |
| test_forward_throughput             | opening-16x3x1024x1024-k63   |    7.975 |     7.984 |       7.979 |       0.012 |    125 |    125.3 |
| test_forward_throughput             | closing-1x1x256x256-k3       |    0.017 |     0.019 |       0.018 |       0.002 |  40307 | 53,405.9 |
| test_forward_throughput             | closing-1x1x256x256-k7       |    0.019 |     0.021 |       0.020 |       0.002 |  35125 | 47,966.0 |
| test_forward_throughput             | closing-1x1x256x256-k15      |    0.024 |     0.026 |       0.025 |       0.002 |  31280 | 38,692.7 |
| test_forward_throughput             | closing-1x1x256x256-k31      |    0.025 |     0.026 |       0.026 |       0.002 |  30507 | 37,807.4 |
| test_forward_throughput             | closing-1x1x256x256-k63      |    0.025 |     0.027 |       0.026 |       0.002 |  30405 | 37,586.6 |
| test_forward_throughput             | closing-8x3x512x512-k3       |    0.377 |     0.379 |       0.379 |       0.003 |   2603 |  2,638.3 |
| test_forward_throughput             | closing-8x3x512x512-k7       |    0.618 |     0.620 |       0.620 |       0.004 |   1594 |  1,612.4 |
| test_forward_throughput             | closing-8x3x512x512-k15      |    0.913 |     0.926 |       0.927 |       0.004 |   1078 |  1,079.3 |
| test_forward_throughput             | closing-8x3x512x512-k31      |    0.960 |     0.964 |       0.963 |       0.002 |   1035 |  1,037.8 |
| test_forward_throughput             | closing-8x3x512x512-k63      |    1.071 |     1.074 |       1.073 |       0.003 |    922 |    931.3 |
| test_forward_throughput             | closing-16x3x1024x1024-k3    |    3.064 |     3.067 |       3.066 |       0.004 |    326 |    326.1 |
| test_forward_throughput             | closing-16x3x1024x1024-k7    |    4.971 |     4.976 |       4.973 |       0.009 |    201 |    201.0 |
| test_forward_throughput             | closing-16x3x1024x1024-k15   |    7.435 |     7.538 |       7.545 |       0.027 |    134 |    132.7 |
| test_forward_throughput             | closing-16x3x1024x1024-k31   |    7.529 |     7.558 |       7.549 |       0.039 |    133 |    132.3 |
| test_forward_throughput             | closing-16x3x1024x1024-k63   |    7.976 |     7.983 |       7.980 |       0.010 |    125 |    125.3 |
| test_forward_throughput             | gradient-1x1x256x256-k3      |    0.018 |     0.020 |       0.020 |       0.001 |   6665 | 49,665.1 |
| test_forward_throughput             | gradient-1x1x256x256-k7      |    0.021 |     0.022 |       0.022 |       0.001 |  34615 | 44,821.5 |
| test_forward_throughput             | gradient-1x1x256x256-k15     |    0.025 |     0.027 |       0.027 |       0.003 |  28466 | 36,549.5 |
| test_forward_throughput             | gradient-1x1x256x256-k31     |    0.026 |     0.028 |       0.027 |       0.004 |  27771 | 35,530.6 |
| test_forward_throughput             | gradient-1x1x256x256-k63     |    0.026 |     0.028 |       0.028 |       0.003 |  26512 | 35,286.2 |
| test_forward_throughput             | gradient-8x3x512x512-k3      |    0.459 |     0.466 |       0.465 |       0.014 |   2106 |  2,146.9 |
| test_forward_throughput             | gradient-8x3x512x512-k7      |    0.699 |     0.706 |       0.705 |       0.010 |   1403 |  1,416.5 |
| test_forward_throughput             | gradient-8x3x512x512-k15     |    1.022 |     1.030 |       1.028 |       0.006 |    959 |    971.0 |
| test_forward_throughput             | gradient-8x3x512x512-k31     |    1.056 |     1.066 |       1.065 |       0.004 |    936 |    938.3 |
| test_forward_throughput             | gradient-8x3x512x512-k63     |    1.160 |     1.174 |       1.177 |       0.010 |    852 |    851.7 |
| test_forward_throughput             | gradient-16x3x1024x1024-k3   |    4.075 |     4.097 |       4.097 |       0.008 |    244 |    244.1 |
| test_forward_throughput             | gradient-16x3x1024x1024-k7   |    5.982 |     5.999 |       5.999 |       0.007 |    166 |    166.7 |
| test_forward_throughput             | gradient-16x3x1024x1024-k15  |    8.496 |     8.570 |       8.562 |       0.045 |    116 |    116.7 |
| test_forward_throughput             | gradient-16x3x1024x1024-k31  |    8.528 |     8.552 |       8.548 |       0.016 |    117 |    116.9 |
| test_forward_throughput             | gradient-16x3x1024x1024-k63  |    8.971 |     9.089 |       9.106 |       0.044 |    111 |    110.0 |
| test_forward_throughput             | top_hat-1x1x256x256-k3       |    0.018 |     0.021 |       0.020 |       0.003 |  30166 | 48,146.9 |
| test_forward_throughput             | top_hat-1x1x256x256-k7       |    0.020 |     0.022 |       0.022 |       0.002 |  31716 | 44,543.0 |
| test_forward_throughput             | top_hat-1x1x256x256-k15      |    0.025 |     0.027 |       0.027 |       0.002 |  28280 | 36,511.7 |
| test_forward_throughput             | top_hat-1x1x256x256-k31      |    0.026 |     0.028 |       0.027 |       0.004 |  29516 | 35,299.8 |
| test_forward_throughput             | top_hat-1x1x256x256-k63      |    0.026 |     0.028 |       0.028 |       0.004 |  27941 | 35,234.6 |
| test_forward_throughput             | top_hat-8x3x512x512-k3       |    0.454 |     0.459 |       0.459 |       0.003 |   2154 |  2,179.4 |
| test_forward_throughput             | top_hat-8x3x512x512-k7       |    0.697 |     0.703 |       0.703 |       0.007 |   1388 |  1,422.1 |
| test_forward_throughput             | top_hat-8x3x512x512-k15      |    1.003 |     1.018 |       1.020 |       0.013 |    979 |    982.6 |
| test_forward_throughput             | top_hat-8x3x512x512-k31      |    1.035 |     1.041 |       1.041 |       0.003 |    949 |    960.3 |
| test_forward_throughput             | top_hat-8x3x512x512-k63      |    1.154 |     1.171 |       1.172 |       0.005 |    859 |    853.8 |
| test_forward_throughput             | top_hat-16x3x1024x1024-k3    |    4.066 |     4.079 |       4.079 |       0.007 |    245 |    245.2 |
| test_forward_throughput             | top_hat-16x3x1024x1024-k7    |    5.968 |     5.982 |       5.981 |       0.006 |    167 |    167.2 |
| test_forward_throughput             | top_hat-16x3x1024x1024-k15   |    8.570 |     8.621 |       8.622 |       0.011 |    117 |    116.0 |
| test_forward_throughput             | top_hat-16x3x1024x1024-k31   |    8.587 |     8.612 |       8.611 |       0.011 |    117 |    116.1 |
| test_forward_throughput             | top_hat-16x3x1024x1024-k63   |    9.078 |     9.099 |       9.092 |       0.024 |    110 |    109.9 |
| test_forward_throughput             | black_hat-1x1x256x256-k3     |    0.018 |     0.021 |       0.020 |       0.003 |  25840 | 47,008.4 |
| test_forward_throughput             | black_hat-1x1x256x256-k7     |    0.020 |     0.023 |       0.022 |       0.003 |  34929 | 43,230.1 |
| test_forward_throughput             | black_hat-1x1x256x256-k15    |    0.025 |     0.028 |       0.027 |       0.003 |  25368 | 35,629.8 |
| test_forward_throughput             | black_hat-1x1x256x256-k31    |    0.026 |     0.028 |       0.028 |       0.005 |  30003 | 35,193.3 |
| test_forward_throughput             | black_hat-1x1x256x256-k63    |    0.026 |     0.029 |       0.028 |       0.004 |  29543 | 34,466.7 |
| test_forward_throughput             | black_hat-8x3x512x512-k3     |    0.454 |     0.459 |       0.459 |       0.003 |   2159 |  2,179.5 |
| test_forward_throughput             | black_hat-8x3x512x512-k7     |    0.697 |     0.702 |       0.702 |       0.002 |   1413 |  1,424.6 |
| test_forward_throughput             | black_hat-8x3x512x512-k15    |    1.013 |     1.021 |       1.021 |       0.004 |    978 |    979.4 |
| test_forward_throughput             | black_hat-8x3x512x512-k31    |    1.036 |     1.050 |       1.054 |       0.007 |    941 |    952.1 |
| test_forward_throughput             | black_hat-8x3x512x512-k63    |    1.153 |     1.171 |       1.171 |       0.005 |    860 |    854.2 |
| test_forward_throughput             | black_hat-16x3x1024x1024-k3  |    4.069 |     4.083 |       4.079 |       0.019 |    245 |    244.9 |
| test_forward_throughput             | black_hat-16x3x1024x1024-k7  |    5.970 |     5.984 |       5.983 |       0.009 |    167 |    167.1 |
| test_forward_throughput             | black_hat-16x3x1024x1024-k15 |    8.595 |     8.627 |       8.625 |       0.012 |    115 |    115.9 |
| test_forward_throughput             | black_hat-16x3x1024x1024-k31 |    8.595 |     8.614 |       8.614 |       0.011 |    116 |    116.1 |
| test_forward_throughput             | black_hat-16x3x1024x1024-k63 |    9.074 |     9.100 |       9.091 |       0.033 |    110 |    109.9 |
| test_forward_backward_throughput    | erosion-1x1x256x256-k7       |    0.060 |     0.103 |       0.113 |       0.023 |   2404 |  9,746.9 |
| test_forward_backward_throughput    | erosion-1x1x256x256-k31      |    0.065 |     0.122 |       0.121 |       0.010 |   5979 |  8,212.5 |
| test_forward_backward_throughput    | erosion-8x3x512x512-k7       |    1.745 |     1.767 |       1.767 |       0.007 |    560 |    565.9 |
| test_forward_backward_throughput    | erosion-8x3x512x512-k31      |    1.803 |     1.826 |       1.826 |       0.008 |    546 |    547.7 |
| test_forward_backward_throughput    | dilation-1x1x256x256-k7      |    0.060 |     0.104 |       0.113 |       0.022 |   5289 |  9,615.3 |
| test_forward_backward_throughput    | dilation-1x1x256x256-k31     |    0.063 |     0.088 |       0.074 |       0.032 |   5171 | 11,371.0 |
| test_forward_backward_throughput    | dilation-8x3x512x512-k7      |    1.741 |     1.753 |       1.750 |       0.031 |    569 |    570.5 |
| test_forward_backward_throughput    | dilation-8x3x512x512-k31     |    1.797 |     1.813 |       1.809 |       0.031 |    549 |    551.4 |
| test_forward_backward_throughput    | opening-1x1x256x256-k7       |    0.102 |     0.134 |       0.121 |       0.047 |   6819 |  7,462.1 |
| test_forward_backward_throughput    | opening-1x1x256x256-k31      |    0.109 |     0.156 |       0.171 |       0.030 |   4867 |  6,424.9 |
| test_forward_backward_throughput    | opening-8x3x512x512-k7       |    4.558 |     4.585 |       4.586 |       0.011 |    218 |    218.1 |
| test_forward_backward_throughput    | opening-8x3x512x512-k31      |    4.978 |     4.999 |       4.997 |       0.013 |    198 |    200.0 |
| test_forward_backward_throughput    | closing-1x1x256x256-k7       |    0.102 |     0.117 |       0.109 |       0.020 |   6654 |  8,583.4 |
| test_forward_backward_throughput    | closing-1x1x256x256-k31      |    0.105 |     0.130 |       0.114 |       0.029 |   7695 |  7,680.9 |
| test_forward_backward_throughput    | closing-8x3x512x512-k7       |    4.560 |     4.602 |       4.604 |       0.013 |    218 |    217.3 |
| test_forward_backward_throughput    | closing-8x3x512x512-k31      |    4.999 |     5.032 |       5.029 |       0.019 |    200 |    198.7 |
| test_forward_backward_throughput    | gradient-1x1x256x256-k7      |    0.099 |     0.161 |       0.168 |       0.025 |   2574 |  6,192.4 |
| test_forward_backward_throughput    | gradient-1x1x256x256-k31     |    0.106 |     0.172 |       0.179 |       0.027 |   4473 |  5,804.0 |
| test_forward_backward_throughput    | gradient-8x3x512x512-k7      |    3.680 |     3.713 |       3.709 |       0.020 |    269 |    269.3 |
| test_forward_backward_throughput    | gradient-8x3x512x512-k31     |    3.782 |     3.805 |       3.805 |       0.011 |    263 |    262.8 |
| test_forward_backward_throughput    | top_hat-1x1x256x256-k7       |    0.108 |     0.129 |       0.128 |       0.044 |   3642 |  7,730.5 |
| test_forward_backward_throughput    | top_hat-1x1x256x256-k31      |    0.119 |     0.157 |       0.143 |       0.031 |   6150 |  6,367.6 |
| test_forward_backward_throughput    | top_hat-8x3x512x512-k7       |    4.704 |     4.743 |       4.742 |       0.024 |    210 |    210.8 |
| test_forward_backward_throughput    | top_hat-8x3x512x512-k31      |    5.124 |     5.160 |       5.160 |       0.014 |    195 |    193.8 |
| test_forward_backward_throughput    | black_hat-1x1x256x256-k7     |    0.112 |     0.172 |       0.184 |       0.029 |   6957 |  5,821.2 |
| test_forward_backward_throughput    | black_hat-1x1x256x256-k31    |    0.194 |     0.203 |       0.201 |       0.006 |   4345 |  4,933.4 |
| test_forward_backward_throughput    | black_hat-8x3x512x512-k7     |    4.817 |     4.847 |       4.843 |       0.021 |    207 |    206.3 |
| test_forward_backward_throughput    | black_hat-8x3x512x512-k31    |    5.257 |     5.285 |       5.281 |       0.022 |    189 |    189.2 |
| test_dtype_forward_throughput       | erosion-fp32                 |    0.311 |     0.314 |       0.313 |       0.008 |   3107 |  3,181.2 |
| test_dtype_forward_throughput       | erosion-fp16                 |    0.318 |     0.321 |       0.320 |       0.006 |   3025 |  3,116.3 |
| test_dtype_forward_throughput       | erosion-bf16                 |    0.318 |     0.320 |       0.319 |       0.001 |   2977 |  3,127.9 |
| test_dtype_forward_throughput       | dilation-fp32                |    0.310 |     0.313 |       0.312 |       0.004 |   3111 |  3,196.1 |
| test_dtype_forward_throughput       | dilation-fp16                |    0.318 |     0.320 |       0.320 |       0.003 |   3008 |  3,120.9 |
| test_dtype_forward_throughput       | dilation-bf16                |    0.318 |     0.320 |       0.319 |       0.007 |   3018 |  3,121.5 |
| test_se_kind_forward_throughput     | erosion-flat                 |    0.495 |     0.506 |       0.504 |       0.013 |   1965 |  1,976.5 |
| test_se_kind_forward_throughput     | erosion-grayscale            |    4.017 |     4.039 |       4.024 |       0.035 |    246 |    247.6 |
| test_se_kind_forward_throughput     | dilation-flat                |    0.499 |     0.509 |       0.504 |       0.065 |   1958 |  1,964.0 |
| test_se_kind_forward_throughput     | dilation-grayscale           |    4.017 |     4.029 |       4.020 |       0.026 |    248 |    248.2 |
| test_border_forward_throughput      | REFLECT                      |    0.332 |     0.334 |       0.334 |       0.002 |   2910 |  2,989.8 |
| test_border_forward_throughput      | REPLICATE                    |    0.311 |     0.314 |       0.312 |       0.008 |   3094 |  3,182.5 |
| test_border_forward_throughput      | CONSTANT                     |    0.313 |     0.317 |       0.315 |       0.011 |   3064 |  3,156.5 |
| test_layer_training_step_throughput | Erosion2d                    |    1.997 |     2.029 |       2.020 |       0.034 |    447 |    493.0 |
| test_layer_training_step_throughput | Dilation2d                   |    1.995 |     2.038 |       2.031 |       0.031 |    490 |    490.6 |
| test_layer_training_step_throughput | Opening2d                    |    5.764 |    18.024 |      20.632 |       4.935 |    188 |     55.5 |
| test_layer_training_step_throughput | Closing2d                    |    5.376 |    18.064 |      20.692 |       4.915 |    193 |     55.4 |

### Against PyTorch, SciPy,  Kornia & CuPy

| op       | shape        |   k | serron (ms) | torch (ms) | vs torch | scipy (ms) | vs scipy | kornia (ms) | vs kornia | cupy (ms) | vs cupy |
| -------- | ------------ | --: | ----------: | ---------: | -------: | ---------: | -------: | ----------: | --------: | --------: | ------: |
| erosion  | 1x1x512x512  |   3 |       0.010 |      0.012 |    1.22x |      4.078 |  399.97x |       0.062 |     6.10x |     0.029 |   2.85x |
| erosion  | 1x1x512x512  |  15 |       0.024 |      0.063 |    2.66x |      4.172 |  175.50x |       1.067 |    44.89x |     0.058 |   2.45x |
| erosion  | 8x3x512x512  |   7 |       0.313 |      0.475 |    1.52x |    102.342 |  327.43x |       5.935 |    18.99x |     0.653 |   2.09x |
| erosion  | 8x32x256x256 |   5 |       0.644 |      0.992 |    1.54x |    250.579 |  389.10x |       9.103 |    14.13x |     1.295 |   2.01x |
| erosion  | 1x1x512x512  |  31 |       0.024 |      0.228 |    9.41x |      3.909 |  161.21x |       4.123 |   170.06x |     0.111 |   4.57x |
| erosion  | 1x1x512x512  |  63 |       0.028 |      0.854 |   30.97x |      3.951 |  143.36x |      16.398 |   594.97x |     0.217 |   7.87x |
| erosion  | 8x3x512x512  |  31 |       0.490 |      4.859 |    9.92x |     93.130 |  190.07x |         OOM |         - |     2.410 |   4.92x |
| erosion  | 8x3x512x512  |  63 |       0.542 |     18.767 |   34.63x |     91.415 |  168.70x |         OOM |         - |     4.772 |   8.81x |
| erosion  | 8x3x512x512  | 127 |       0.501 |     70.357 |  140.48x |     88.268 |  176.25x |         OOM |         - |     9.451 |  18.87x |
| dilation | 1x1x512x512  |   3 |       0.010 |      0.007 |    0.76x |      3.998 |  413.96x |       0.061 |     6.31x |     0.027 |   2.79x |
| dilation | 1x1x512x512  |  15 |       0.022 |      0.058 |    2.61x |      4.287 |  192.21x |       1.031 |    46.22x |     0.058 |   2.62x |
| dilation | 8x3x512x512  |   7 |       0.310 |      0.341 |    1.10x |    103.450 |  333.68x |       5.910 |    19.06x |     0.654 |   2.11x |
| dilation | 8x32x256x256 |   5 |       0.646 |      0.558 |    0.86x |    246.545 |  381.76x |       9.080 |    14.06x |     1.295 |   2.01x |
| dilation | 1x1x512x512  |  31 |       0.024 |      0.223 |    9.45x |      3.717 |  157.26x |       4.128 |   174.65x |     0.110 |   4.67x |
| dilation | 1x1x512x512  |  63 |       0.027 |      0.848 |   31.34x |      3.630 |  134.12x |      16.403 |   606.01x |     0.214 |   7.91x |
| dilation | 8x3x512x512  |  31 |       0.494 |      4.731 |    9.58x |     93.566 |  189.43x |         OOM |         - |     2.407 |   4.87x |
| dilation | 8x3x512x512  |  63 |       0.545 |     18.644 |   34.19x |     86.724 |  159.03x |         OOM |         - |     4.763 |   8.73x |
| dilation | 8x3x512x512  | 127 |       0.507 |     70.252 |  138.63x |     84.477 |  166.70x |         OOM |         - |     9.452 |  18.65x |

## NVIDIA H100 (PCIe) & Intel Xeon Platinum 8480+

### Benchmarks

| test                                | params                       | min (ms) | mean (ms) | median (ms) | stddev (ms) | rounds |    ops/s |
| ----------------------------------- | ---------------------------- | -------: | --------: | ----------: | ----------: | -----: | -------: |
| test_forward_throughput             | erosion-1x1x256x256-k3       |    0.033 |     0.037 |       0.035 |       0.005 |   6064 | 26,937.5 |
| test_forward_throughput             | erosion-1x1x256x256-k7       |    0.033 |     0.037 |       0.035 |       0.006 |  11918 | 27,293.7 |
| test_forward_throughput             | erosion-1x1x256x256-k15      |    0.040 |     0.044 |       0.042 |       0.009 |  14483 | 22,902.1 |
| test_forward_throughput             | erosion-1x1x256x256-k31      |    0.039 |     0.045 |       0.042 |       0.008 |    665 | 22,462.1 |
| test_forward_throughput             | erosion-1x1x256x256-k63      |    0.039 |     0.044 |       0.042 |       0.005 |  12773 | 22,849.4 |
| test_forward_throughput             | erosion-8x3x512x512-k3       |    0.200 |     0.211 |       0.211 |       0.014 |   3783 |  4,728.8 |
| test_forward_throughput             | erosion-8x3x512x512-k7       |    0.340 |     0.349 |       0.349 |       0.004 |   2676 |  2,862.0 |
| test_forward_throughput             | erosion-8x3x512x512-k15      |    0.804 |     0.812 |       0.812 |       0.003 |   1197 |  1,231.7 |
| test_forward_throughput             | erosion-8x3x512x512-k31      |    0.245 |     0.251 |       0.250 |       0.004 |   3538 |  3,979.1 |
| test_forward_throughput             | erosion-8x3x512x512-k63      |    0.334 |     0.341 |       0.338 |       0.014 |   2787 |  2,932.1 |
| test_forward_throughput             | erosion-16x3x1024x1024-k3    |    1.463 |     1.467 |       1.465 |       0.005 |    666 |    681.8 |
| test_forward_throughput             | erosion-16x3x1024x1024-k7    |    2.580 |     2.585 |       2.583 |       0.017 |    385 |    386.8 |
| test_forward_throughput             | erosion-16x3x1024x1024-k15   |    6.228 |     6.231 |       6.231 |       0.003 |    160 |    160.5 |
| test_forward_throughput             | erosion-16x3x1024x1024-k31   |    1.571 |     1.578 |       1.578 |       0.004 |    622 |    633.6 |
| test_forward_throughput             | erosion-16x3x1024x1024-k63   |    2.166 |     2.171 |       2.170 |       0.004 |    455 |    460.6 |
| test_forward_throughput             | dilation-1x1x256x256-k3      |    0.023 |     0.025 |       0.025 |       0.003 |  12978 | 39,521.1 |
| test_forward_throughput             | dilation-1x1x256x256-k7      |    0.026 |     0.028 |       0.027 |       0.003 |  18391 | 36,360.5 |
| test_forward_throughput             | dilation-1x1x256x256-k15     |    0.033 |     0.035 |       0.035 |       0.003 |  18935 | 28,329.0 |
| test_forward_throughput             | dilation-1x1x256x256-k31     |    0.030 |     0.033 |       0.032 |       0.009 |  16829 | 30,255.9 |
| test_forward_throughput             | dilation-1x1x256x256-k63     |    0.029 |     0.031 |       0.030 |       0.004 |  16634 | 31,928.2 |
| test_forward_throughput             | dilation-8x3x512x512-k3      |    0.202 |     0.208 |       0.206 |       0.005 |   4158 |  4,816.7 |
| test_forward_throughput             | dilation-8x3x512x512-k7      |    0.345 |     0.351 |       0.350 |       0.004 |   2628 |  2,846.2 |
| test_forward_throughput             | dilation-8x3x512x512-k15     |    0.805 |     0.813 |       0.812 |       0.014 |   1197 |  1,229.8 |
| test_forward_throughput             | dilation-8x3x512x512-k31     |    0.246 |     0.252 |       0.250 |       0.004 |   3669 |  3,976.0 |
| test_forward_throughput             | dilation-8x3x512x512-k63     |    0.334 |     0.341 |       0.339 |       0.008 |   2772 |  2,932.0 |
| test_forward_throughput             | dilation-16x3x1024x1024-k3   |    1.462 |     1.467 |       1.465 |       0.012 |    668 |    681.6 |
| test_forward_throughput             | dilation-16x3x1024x1024-k7   |    2.579 |     2.584 |       2.582 |       0.009 |    384 |    386.9 |
| test_forward_throughput             | dilation-16x3x1024x1024-k15  |    6.230 |     6.234 |       6.233 |       0.005 |    160 |    160.4 |
| test_forward_throughput             | dilation-16x3x1024x1024-k31  |    1.572 |     1.579 |       1.579 |       0.005 |    628 |    633.2 |
| test_forward_throughput             | dilation-16x3x1024x1024-k63  |    2.166 |     2.172 |       2.171 |       0.004 |    456 |    460.5 |
| test_forward_throughput             | opening-1x1x256x256-k3       |    0.039 |     0.043 |       0.041 |       0.005 |  13698 | 23,317.5 |
| test_forward_throughput             | opening-1x1x256x256-k7       |    0.042 |     0.045 |       0.044 |       0.013 |  13980 | 22,071.6 |
| test_forward_throughput             | opening-1x1x256x256-k15      |    0.050 |     0.054 |       0.052 |       0.004 |  12152 | 18,689.9 |
| test_forward_throughput             | opening-1x1x256x256-k31      |    0.051 |     0.056 |       0.054 |       0.007 |  12865 | 17,987.5 |
| test_forward_throughput             | opening-1x1x256x256-k63      |    0.050 |     0.054 |       0.053 |       0.006 |  12863 | 18,394.5 |
| test_forward_throughput             | opening-8x3x512x512-k3       |    0.386 |     0.393 |       0.392 |       0.005 |   2487 |  2,547.8 |
| test_forward_throughput             | opening-8x3x512x512-k7       |    0.669 |     0.676 |       0.677 |       0.005 |   1439 |  1,478.9 |
| test_forward_throughput             | opening-8x3x512x512-k15      |    1.592 |     1.595 |       1.595 |       0.003 |    619 |    626.8 |
| test_forward_throughput             | opening-8x3x512x512-k31      |    0.472 |     0.480 |       0.480 |       0.007 |   2032 |  2,082.0 |
| test_forward_throughput             | opening-8x3x512x512-k63      |    0.644 |     0.652 |       0.653 |       0.004 |   1495 |  1,533.1 |
| test_forward_throughput             | opening-16x3x1024x1024-k3    |    2.896 |     2.900 |       2.899 |       0.003 |    343 |    344.8 |
| test_forward_throughput             | opening-16x3x1024x1024-k7    |    5.131 |     5.135 |       5.134 |       0.003 |    194 |    194.7 |
| test_forward_throughput             | opening-16x3x1024x1024-k15   |   12.425 |    12.448 |      12.454 |       0.012 |     81 |     80.3 |
| test_forward_throughput             | opening-16x3x1024x1024-k31   |    3.128 |     3.141 |       3.141 |       0.008 |    316 |    318.3 |
| test_forward_throughput             | opening-16x3x1024x1024-k63   |    4.296 |     4.303 |       4.300 |       0.006 |    231 |    232.4 |
| test_forward_throughput             | closing-1x1x256x256-k3       |    0.040 |     0.043 |       0.042 |       0.004 |  13233 | 23,145.6 |
| test_forward_throughput             | closing-1x1x256x256-k7       |    0.042 |     0.045 |       0.044 |       0.005 |  11577 | 22,311.9 |
| test_forward_throughput             | closing-1x1x256x256-k15      |    0.050 |     0.053 |       0.052 |       0.004 |  14201 | 18,821.1 |
| test_forward_throughput             | closing-1x1x256x256-k31      |    0.051 |     0.055 |       0.054 |       0.004 |  12618 | 18,054.3 |
| test_forward_throughput             | closing-1x1x256x256-k63      |    0.050 |     0.054 |       0.053 |       0.005 |  13401 | 18,529.1 |
| test_forward_throughput             | closing-8x3x512x512-k3       |    0.386 |     0.391 |       0.390 |       0.005 |   2474 |  2,554.5 |
| test_forward_throughput             | closing-8x3x512x512-k7       |    0.668 |     0.675 |       0.676 |       0.006 |   1448 |  1,480.7 |
| test_forward_throughput             | closing-8x3x512x512-k15      |    1.592 |     1.595 |       1.594 |       0.004 |    620 |    627.1 |
| test_forward_throughput             | closing-8x3x512x512-k31      |    0.472 |     0.480 |       0.480 |       0.006 |   2041 |  2,082.3 |
| test_forward_throughput             | closing-8x3x512x512-k63      |    0.645 |     0.653 |       0.654 |       0.004 |   1505 |  1,530.7 |
| test_forward_throughput             | closing-16x3x1024x1024-k3    |    2.902 |     2.905 |       2.905 |       0.002 |    343 |    344.2 |
| test_forward_throughput             | closing-16x3x1024x1024-k7    |    5.141 |     5.145 |       5.144 |       0.013 |    194 |    194.4 |
| test_forward_throughput             | closing-16x3x1024x1024-k15   |   12.449 |    12.454 |      12.453 |       0.007 |     81 |     80.3 |
| test_forward_throughput             | closing-16x3x1024x1024-k31   |    3.128 |     3.140 |       3.139 |       0.005 |    318 |    318.5 |
| test_forward_throughput             | closing-16x3x1024x1024-k63   |    4.295 |     4.309 |       4.300 |       0.021 |    232 |    232.1 |
| test_forward_throughput             | gradient-1x1x256x256-k3      |    0.048 |     0.054 |       0.052 |       0.005 |   6305 | 18,644.7 |
| test_forward_throughput             | gradient-1x1x256x256-k7      |    0.048 |     0.053 |       0.051 |       0.005 |  12327 | 18,958.5 |
| test_forward_throughput             | gradient-1x1x256x256-k15     |    0.053 |     0.057 |       0.056 |       0.005 |  12430 | 17,586.3 |
| test_forward_throughput             | gradient-1x1x256x256-k31     |    0.056 |     0.063 |       0.061 |       0.006 |  11694 | 15,856.6 |
| test_forward_throughput             | gradient-1x1x256x256-k63     |    0.057 |     0.064 |       0.062 |       0.005 |  11306 | 15,724.0 |
| test_forward_throughput             | gradient-8x3x512x512-k3      |    0.425 |     0.432 |       0.432 |       0.006 |   2253 |  2,315.9 |
| test_forward_throughput             | gradient-8x3x512x512-k7      |    0.708 |     0.716 |       0.717 |       0.004 |   1358 |  1,396.5 |
| test_forward_throughput             | gradient-8x3x512x512-k15     |    1.631 |     1.634 |       1.634 |       0.002 |    608 |    611.8 |
| test_forward_throughput             | gradient-8x3x512x512-k31     |    0.512 |     0.518 |       0.518 |       0.003 |   1870 |  1,929.2 |
| test_forward_throughput             | gradient-8x3x512x512-k63     |    0.680 |     0.690 |       0.690 |       0.004 |   1410 |  1,450.1 |
| test_forward_throughput             | gradient-16x3x1024x1024-k3   |    3.212 |     3.216 |       3.215 |       0.003 |    310 |    310.9 |
| test_forward_throughput             | gradient-16x3x1024x1024-k7   |    5.445 |     5.451 |       5.449 |       0.016 |    184 |    183.5 |
| test_forward_throughput             | gradient-16x3x1024x1024-k15  |   12.732 |    12.737 |      12.736 |       0.003 |     79 |     78.5 |
| test_forward_throughput             | gradient-16x3x1024x1024-k31  |    3.449 |     3.461 |       3.460 |       0.005 |    289 |    289.0 |
| test_forward_throughput             | gradient-16x3x1024x1024-k63  |    4.615 |     4.642 |       4.623 |       0.033 |    216 |    215.4 |
| test_forward_throughput             | top_hat-1x1x256x256-k3       |    0.047 |     0.053 |       0.051 |       0.006 |  12895 | 18,999.9 |
| test_forward_throughput             | top_hat-1x1x256x256-k7       |    0.047 |     0.053 |       0.051 |       0.006 |  11799 | 18,967.6 |
| test_forward_throughput             | top_hat-1x1x256x256-k15      |    0.053 |     0.057 |       0.056 |       0.005 |  12964 | 17,471.2 |
| test_forward_throughput             | top_hat-1x1x256x256-k31      |    0.057 |     0.063 |       0.061 |       0.009 |  11685 | 15,863.8 |
| test_forward_throughput             | top_hat-1x1x256x256-k63      |    0.056 |     0.063 |       0.061 |       0.007 |  11102 | 15,870.2 |
| test_forward_throughput             | top_hat-8x3x512x512-k3       |    0.426 |     0.432 |       0.432 |       0.011 |   2239 |  2,313.5 |
| test_forward_throughput             | top_hat-8x3x512x512-k7       |    0.708 |     0.716 |       0.717 |       0.007 |   1365 |  1,396.5 |
| test_forward_throughput             | top_hat-8x3x512x512-k15      |    1.630 |     1.635 |       1.634 |       0.003 |    607 |    611.8 |
| test_forward_throughput             | top_hat-8x3x512x512-k31      |    0.511 |     0.518 |       0.519 |       0.004 |    800 |  1,928.9 |
| test_forward_throughput             | top_hat-8x3x512x512-k63      |    0.683 |     0.692 |       0.692 |       0.003 |   1424 |  1,445.8 |
| test_forward_throughput             | top_hat-16x3x1024x1024-k3    |    3.220 |     3.225 |       3.225 |       0.003 |    310 |    310.1 |
| test_forward_throughput             | top_hat-16x3x1024x1024-k7    |    5.459 |     5.465 |       5.464 |       0.007 |    183 |    183.0 |
| test_forward_throughput             | top_hat-16x3x1024x1024-k15   |   12.764 |    12.773 |      12.769 |       0.023 |     79 |     78.3 |
| test_forward_throughput             | top_hat-16x3x1024x1024-k31   |    3.453 |     3.461 |       3.461 |       0.004 |    288 |    288.9 |
| test_forward_throughput             | top_hat-16x3x1024x1024-k63   |    4.614 |     4.640 |       4.622 |       0.034 |    216 |    215.5 |
| test_forward_throughput             | black_hat-1x1x256x256-k3     |    0.047 |     0.053 |       0.051 |       0.005 |  12837 | 19,047.3 |
| test_forward_throughput             | black_hat-1x1x256x256-k7     |    0.048 |     0.053 |       0.051 |       0.005 |  10049 | 18,877.9 |
| test_forward_throughput             | black_hat-1x1x256x256-k15    |    0.054 |     0.058 |       0.057 |       0.006 |  12522 | 17,258.3 |
| test_forward_throughput             | black_hat-1x1x256x256-k31    |    0.056 |     0.063 |       0.061 |       0.006 |  11078 | 15,832.5 |
| test_forward_throughput             | black_hat-1x1x256x256-k63    |    0.057 |     0.064 |       0.062 |       0.013 |  10267 | 15,549.9 |
| test_forward_throughput             | black_hat-8x3x512x512-k3     |    0.425 |     0.432 |       0.432 |       0.007 |   2226 |  2,314.4 |
| test_forward_throughput             | black_hat-8x3x512x512-k7     |    0.708 |     0.717 |       0.717 |       0.011 |   1369 |  1,395.1 |
| test_forward_throughput             | black_hat-8x3x512x512-k15    |    1.631 |     1.635 |       1.634 |       0.003 |    605 |    611.8 |
| test_forward_throughput             | black_hat-8x3x512x512-k31    |    0.510 |     0.520 |       0.520 |       0.003 |   1881 |  1,924.5 |
| test_forward_throughput             | black_hat-8x3x512x512-k63    |    0.684 |     0.693 |       0.694 |       0.004 |   1413 |  1,442.9 |
| test_forward_throughput             | black_hat-16x3x1024x1024-k3  |    3.220 |     3.225 |       3.224 |       0.007 |    310 |    310.1 |
| test_forward_throughput             | black_hat-16x3x1024x1024-k7  |    5.459 |     5.465 |       5.464 |       0.005 |    183 |    183.0 |
| test_forward_throughput             | black_hat-16x3x1024x1024-k15 |   12.721 |    12.754 |      12.744 |       0.038 |     79 |     78.4 |
| test_forward_throughput             | black_hat-16x3x1024x1024-k31 |    3.452 |     3.462 |       3.462 |       0.004 |    289 |    288.8 |
| test_forward_throughput             | black_hat-16x3x1024x1024-k63 |    4.617 |     4.623 |       4.621 |       0.004 |    216 |    216.3 |
| test_forward_backward_throughput    | erosion-1x1x256x256-k7       |    0.333 |     0.637 |       0.670 |       0.140 |   1155 |  1,571.1 |
| test_forward_backward_throughput    | erosion-1x1x256x256-k31      |    0.703 |     0.754 |       0.756 |       0.030 |   1008 |  1,326.7 |
| test_forward_backward_throughput    | erosion-8x3x512x512-k7       |    1.555 |     1.593 |       1.577 |       0.040 |    550 |    627.6 |
| test_forward_backward_throughput    | erosion-8x3x512x512-k31      |    1.005 |     1.098 |       1.087 |       0.068 |    883 |    910.9 |
| test_forward_backward_throughput    | dilation-1x1x256x256-k7      |    0.544 |     0.693 |       0.703 |       0.044 |   1295 |  1,443.2 |
| test_forward_backward_throughput    | dilation-1x1x256x256-k31     |    0.700 |     0.749 |       0.748 |       0.045 |    953 |  1,335.7 |
| test_forward_backward_throughput    | dilation-8x3x512x512-k7      |    1.555 |     1.591 |       1.579 |       0.040 |    538 |    628.4 |
| test_forward_backward_throughput    | dilation-8x3x512x512-k31     |    1.003 |     1.084 |       1.081 |       0.064 |    914 |    922.7 |
| test_forward_backward_throughput    | opening-1x1x256x256-k7       |    0.733 |     0.937 |       0.944 |       0.056 |   1053 |  1,067.2 |
| test_forward_backward_throughput    | opening-1x1x256x256-k31      |    0.962 |     1.013 |       0.999 |       0.102 |    795 |    987.1 |
| test_forward_backward_throughput    | opening-8x3x512x512-k7       |    3.854 |     3.888 |       3.874 |       0.049 |    250 |    257.2 |
| test_forward_backward_throughput    | opening-8x3x512x512-k31      |    2.675 |     2.724 |       2.722 |       0.024 |    365 |    367.1 |
| test_forward_backward_throughput    | closing-1x1x256x256-k7       |    0.604 |     0.929 |       0.949 |       0.110 |   1412 |  1,076.2 |
| test_forward_backward_throughput    | closing-1x1x256x256-k31      |    0.964 |     1.027 |       1.004 |       0.086 |    786 |    974.1 |
| test_forward_backward_throughput    | closing-8x3x512x512-k7       |    3.868 |     3.901 |       3.894 |       0.030 |    250 |    256.3 |
| test_forward_backward_throughput    | closing-8x3x512x512-k31      |    2.662 |     2.726 |       2.718 |       0.105 |    366 |    366.8 |
| test_forward_backward_throughput    | gradient-1x1x256x256-k7      |    0.696 |     1.056 |       1.055 |       0.084 |   1020 |    947.1 |
| test_forward_backward_throughput    | gradient-1x1x256x256-k31     |    0.740 |     1.146 |       1.138 |       0.111 |    703 |    872.9 |
| test_forward_backward_throughput    | gradient-8x3x512x512-k7      |    3.125 |     3.157 |       3.146 |       0.028 |    302 |    316.8 |
| test_forward_backward_throughput    | gradient-8x3x512x512-k31     |    1.961 |     1.991 |       1.992 |       0.011 |    497 |    502.2 |
| test_forward_backward_throughput    | top_hat-1x1x256x256-k7       |    0.699 |     1.005 |       1.003 |       0.116 |   1025 |    995.1 |
| test_forward_backward_throughput    | top_hat-1x1x256x256-k31      |    0.714 |     1.122 |       1.119 |       0.069 |    726 |    891.7 |
| test_forward_backward_throughput    | top_hat-8x3x512x512-k7       |    3.930 |     3.959 |       3.950 |       0.027 |    246 |    252.6 |
| test_forward_backward_throughput    | top_hat-8x3x512x512-k31      |    2.729 |     2.790 |       2.789 |       0.031 |    353 |    358.4 |
| test_forward_backward_throughput    | black_hat-1x1x256x256-k7     |    0.686 |     1.005 |       1.043 |       0.120 |   1188 |    995.4 |
| test_forward_backward_throughput    | black_hat-1x1x256x256-k31    |    0.755 |     1.158 |       1.154 |       0.064 |    701 |    863.8 |
| test_forward_backward_throughput    | black_hat-8x3x512x512-k7     |    3.951 |     3.980 |       3.970 |       0.072 |    246 |    251.3 |
| test_forward_backward_throughput    | black_hat-8x3x512x512-k31    |    2.778 |     2.842 |       2.842 |       0.027 |    353 |    351.9 |
| test_dtype_forward_throughput       | erosion-fp32                 |    0.355 |     0.362 |       0.360 |       0.005 |   2573 |  2,760.0 |
| test_dtype_forward_throughput       | erosion-fp16                 |    0.361 |     0.369 |       0.366 |       0.011 |   2518 |  2,712.5 |
| test_dtype_forward_throughput       | erosion-bf16                 |    0.361 |     0.369 |       0.367 |       0.006 |   2476 |  2,712.0 |
| test_dtype_forward_throughput       | dilation-fp32                |    0.355 |     0.363 |       0.359 |       0.008 |   2559 |  2,757.1 |
| test_dtype_forward_throughput       | dilation-fp16                |    0.361 |     0.368 |       0.366 |       0.005 |   2485 |  2,715.5 |
| test_dtype_forward_throughput       | dilation-bf16                |    0.353 |     0.363 |       0.362 |       0.007 |   2480 |  2,755.2 |
| test_se_kind_forward_throughput     | erosion-flat                 |    0.247 |     0.252 |       0.250 |       0.007 |   3622 |  3,972.3 |
| test_se_kind_forward_throughput     | erosion-grayscale            |    2.451 |     2.456 |       2.454 |       0.006 |    403 |    407.2 |
| test_se_kind_forward_throughput     | dilation-flat                |    0.246 |     0.251 |       0.250 |       0.004 |   3692 |  3,980.1 |
| test_se_kind_forward_throughput     | dilation-grayscale           |    2.432 |     2.453 |       2.453 |       0.007 |    403 |    407.7 |
| test_border_forward_throughput      | REFLECT                      |    0.335 |     0.342 |       0.339 |       0.016 |   1284 |  2,925.3 |
| test_border_forward_throughput      | REPLICATE                    |    0.345 |     0.351 |       0.349 |       0.013 |   2671 |  2,849.1 |
| test_border_forward_throughput      | CONSTANT                     |    0.320 |     0.326 |       0.324 |       0.008 |   2931 |  3,068.1 |
| test_layer_training_step_throughput | Erosion2d                    |    1.735 |     1.763 |       1.752 |       0.111 |    474 |    567.1 |
| test_layer_training_step_throughput | Dilation2d                   |    1.689 |     1.729 |       1.708 |       0.049 |    555 |


### Against PyTorch, SciPy, Kornia & CuPy

| op       | shape        |   k | serron (ms) | torch (ms) | vs torch | scipy (ms) | vs scipy | kornia (ms) | vs kornia | cupy (ms) | vs cupy |
| -------- | ------------ | --: | ----------: | ---------: | -------: | ---------: | -------: | ----------: | --------: | --------: | ------: |
| erosion  | 1x1x512x512  |   3 |       0.031 |      0.028 |    0.90x |      8.682 |  280.49x |       0.109 |     3.51x |     0.094 |   3.04x |
| erosion  | 1x1x512x512  |  15 |       0.026 |      0.075 |    2.89x |      6.678 |  257.59x |       0.459 |    17.71x |     0.095 |   3.65x |
| erosion  | 8x3x512x512  |   7 |       0.300 |      0.421 |    1.40x |    158.795 |  529.50x |       2.801 |     9.34x |     0.203 |   0.68x |
| erosion  | 8x32x256x256 |   5 |       0.612 |      0.742 |    1.21x |    380.023 |  621.44x |       4.561 |     7.46x |     0.424 |   0.69x |
| erosion  | 1x1x512x512  |  31 |       0.025 |      0.280 |   11.19x |      7.183 |  287.39x |       1.665 |    66.64x |     0.092 |   3.68x |
| erosion  | 1x1x512x512  |  63 |       0.025 |      1.026 |   41.37x |      6.826 |  275.29x |       6.488 |   261.67x |     0.098 |   3.96x |
| erosion  | 8x3x512x512  |  31 |       0.226 |      5.457 |   24.12x |    150.676 |  666.07x |      41.873 |   185.10x |     0.577 |   2.55x |
| erosion  | 8x3x512x512  |  63 |       0.308 |     21.475 |   69.66x |    143.885 |  466.70x |         OOM |         - |     1.053 |   3.42x |
| erosion  | 8x3x512x512  | 127 |       0.246 |     81.027 |  329.53x |    141.945 |  577.28x |         OOM |         - |     2.002 |   8.14x |
| dilation | 1x1x512x512  |   3 |       0.021 |      0.009 |    0.44x |      5.944 |  283.11x |       0.086 |     4.12x |     0.093 |   4.42x |
| dilation | 1x1x512x512  |  15 |       0.026 |      0.070 |    2.73x |      6.744 |  262.50x |       0.460 |    17.91x |     0.092 |   3.60x |
| dilation | 8x3x512x512  |   7 |       0.300 |      0.366 |    1.22x |    154.894 |  516.78x |       2.801 |     9.34x |     0.203 |   0.68x |
| dilation | 8x32x256x256 |   5 |       0.612 |      0.591 |    0.97x |    387.827 |  634.00x |       4.563 |     7.46x |     0.424 |   0.69x |
| dilation | 1x1x512x512  |  31 |       0.025 |      0.275 |   10.98x |      5.822 |  232.14x |       1.670 |    66.58x |     0.090 |   3.59x |
| dilation | 1x1x512x512  |  63 |       0.025 |      1.023 |   40.91x |      5.694 |  227.81x |       6.494 |   259.81x |     0.094 |   3.76x |
| dilation | 8x3x512x512  |  31 |       0.226 |      5.402 |   23.89x |    143.629 |  635.16x |      41.916 |   185.36x |     0.574 |   2.54x |
| dilation | 8x3x512x512  |  63 |       0.308 |     21.420 |   69.51x |    138.888 |  450.75x |         OOM |         - |     1.051 |   3.41x |
| dilation | 8x3x512x512  | 127 |       0.246 |     80.966 |  329.25x |    136.215 |  553.92x |         OOM |         - |     2.003 |   8.14x |