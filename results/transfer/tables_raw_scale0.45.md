## 1. Gain over random walk, reach top 1% (points). Pooled over 10 held-out assays x 10 seeds; ± = SE over the 100 (assay, seed) pairs

| arm | gain over random | gain, top-10% reach |
|---|---|---|
| chemotaxis | +29.0 ± 1.3 | +3.9 ± 0.5 |
| zero | -0.2 ± 0.9 | -0.4 ± 0.3 |
| zero_mem | -1.8 ± 1.7 | -0.7 ± 0.6 |
| pre | -13.1 ± 2.3 | -2.0 ± 0.7 |
| pre_mem | -7.1 ± 2.3 | -1.0 ± 0.6 |
| pre_frozen | -12.4 ± 2.3 | -2.7 ± 0.8 |
| shuf_pre | -15.2 ± 2.1 | -7.0 ± 0.8 |
| shuf_pre_mem | -12.0 ± 2.2 | -4.7 ± 0.7 |
| pre_a0.1 | +0.5 ± 0.9 | +0.0 ± 0.3 |
| pre_a0.3 | +0.1 ± 1.1 | +0.1 ± 0.4 |
| pre_mem_a0.1 | +0.0 ± 1.7 | -0.5 ± 0.6 |
| pre_mem_a0.3 | -1.0 ± 1.9 | +0.3 ± 0.6 |

## 2. Paired differences (points, top 1%); ± = SE over (assay, seed) pairs; 'wins' = held-out assays where the seed-mean difference is > 0

| comparison | mean diff | wins (of held-out assays) |
|---|---|---|
| pre − zero | -12.9 ± 2.2 | 3 |
| pre_mem − zero_mem | -5.3 ± 2.2 | 3 |
| pre_mem − zero | -6.9 ± 2.2 | 4 |
| pre_frozen − zero | -12.2 ± 2.2 | 3 |
| pre − shuf_pre | +2.1 ± 2.2 | 6 |
| pre_mem − shuf_pre_mem | +4.9 ± 2.3 | 7 |
| zero_mem − zero | -1.6 ± 1.7 | 5 |
| pre_a0.1 − zero | +0.7 ± 0.8 | 7 |
| pre_a0.3 − zero | +0.3 ± 1.0 | 5 |
| pre_mem_a0.1 − zero_mem | +1.8 ± 1.4 | 7 |
| pre_mem_a0.3 − zero_mem | +0.7 ± 1.6 | 6 |

## 3. Per held-out assay: gain over random (mean over seeds)

arm         chemotaxis  zero  zero_mem   pre  pre_mem  pre_a0.1  pre_mem_a0.1  shuf_pre
BLAT 2014         28.0   2.0      15.6 -26.4     -8.2       2.2          11.2     -15.8
DYR 2023          32.6   0.2       5.8  -3.2      3.4       0.8           1.0      -6.0
KKA2 2014         12.0  -7.6     -28.0 -38.0    -33.4      -9.8         -21.2     -34.0
MLAC 2023         27.0   1.4       7.0 -17.4     -8.4       2.2           7.8     -14.2
NUD15 2020        20.2   4.4      -7.2 -28.4    -26.0       5.8          -5.0     -38.4
RASH 2017         37.8   0.8       8.4   9.2     11.4       4.4          10.4      -2.6
RNC 2023          35.6   3.4      -4.4   5.0      4.8       1.2          -1.2      10.6
TPK1 2017         23.0  -6.0     -23.0 -18.2    -18.2      -7.2         -22.4     -23.2
TPMT 2018         30.2   1.4      -0.2 -15.4     -3.4       4.0          12.2     -23.6
UBC9 2017         43.8  -1.8       8.4   1.8      7.4       1.2           7.2      -4.4

## 4. Zero-shot value of pretrained weights on the held-out assay (Spearman of w·code vs fitness, mean ± std over seeds)

             mean             std         
arm           pre shuf_pre    pre shuf_pre
BLAT 2014   0.082   -0.031  0.039    0.083
DYR 2023    0.043   -0.009  0.055    0.065
KKA2 2014   0.101    0.012  0.056    0.076
MLAC 2023   0.074   -0.017  0.043    0.034
NUD15 2020  0.144    0.037  0.044    0.057
RASH 2017   0.035   -0.001  0.041    0.061
RNC 2023    0.112   -0.000  0.040    0.044
TPK1 2017   0.053   -0.009  0.024    0.019
TPMT 2018   0.109    0.005  0.040    0.048
UBC9 2017   0.133    0.011  0.040    0.062
