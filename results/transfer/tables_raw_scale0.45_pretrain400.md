## 1. Gain over random walk, reach top 1% (points). Pooled over 10 held-out assays x 5 seeds; ± = SE over the 50 (assay, seed) pairs

| arm | gain over random | gain, top-10% reach |
|---|---|---|
| chemotaxis | +30.0 ± 1.5 | +4.2 ± 0.6 |
| zero | +0.4 ± 1.1 | +0.6 ± 0.4 |
| zero_mem | -0.9 ± 2.6 | +0.0 ± 0.7 |
| pre | -14.9 ± 2.7 | -6.7 ± 1.1 |
| pre_mem | -10.1 ± 3.1 | -2.4 ± 0.9 |
| pre_frozen | -15.5 ± 2.9 | -8.9 ± 1.4 |
| shuf_pre | -19.8 ± 2.8 | -11.8 ± 1.2 |
| shuf_pre_mem | -16.7 ± 2.7 | -8.2 ± 1.2 |
| pre_a0.1 | +2.1 ± 1.4 | +0.9 ± 0.6 |
| pre_a0.3 | +0.6 ± 1.9 | +0.0 ± 0.6 |
| pre_mem_a0.1 | -0.9 ± 2.4 | +0.2 ± 0.8 |
| pre_mem_a0.3 | +2.7 ± 2.3 | +0.4 ± 0.9 |

## 2. Paired differences (points, top 1%); ± = SE over (assay, seed) pairs; 'wins' = held-out assays where the seed-mean difference is > 0

| comparison | mean diff | wins (of held-out assays) |
|---|---|---|
| pre − zero | -15.4 ± 2.8 | 1 |
| pre_mem − zero_mem | -9.2 ± 3.3 | 4 |
| pre_mem − zero | -10.6 ± 2.8 | 3 |
| pre_frozen − zero | -16.0 ± 2.9 | 1 |
| pre − shuf_pre | +4.8 ± 2.8 | 8 |
| pre_mem − shuf_pre_mem | +6.6 ± 3.4 | 7 |
| zero_mem − zero | -1.3 ± 2.3 | 5 |
| pre_a0.1 − zero | +1.7 ± 1.2 | 5 |
| pre_a0.3 − zero | +0.2 ± 1.8 | 5 |
| pre_mem_a0.1 − zero_mem | -0.0 ± 2.0 | 3 |
| pre_mem_a0.3 − zero_mem | +3.6 ± 2.5 | 5 |

## 3. Per held-out assay: gain over random (mean over seeds)

arm         chemotaxis  zero  zero_mem   pre  pre_mem  pre_a0.1  pre_mem_a0.1  shuf_pre
BLAT 2014         28.0   2.4      17.6 -27.6    -15.2       0.4          12.0     -21.6
DYR 2023          37.2   1.6      10.8  -6.0      3.2      11.6           6.0     -10.0
KKA2 2014         16.8  -8.0     -28.8 -38.0    -40.8      -6.0         -24.4     -41.2
MLAC 2023         32.0   5.2      10.4 -15.2    -13.2       4.4           3.6     -23.6
NUD15 2020        20.4   2.0      -5.2 -32.8    -24.4       0.8         -10.0     -33.2
RASH 2017         37.6   4.0      10.0  -8.4     -4.4       4.0           6.0       2.4
RNC 2023          32.8   3.2      -4.4   1.2     -2.0      -0.8          -6.8      -3.2
TPK1 2017         22.8  -7.2     -21.6 -10.0    -18.4      -3.6         -22.8     -24.0
TPMT 2018         29.2   2.8      -2.0 -10.8      7.2       5.6          15.2     -28.0
UBC9 2017         43.2  -1.6       4.4  -1.6      6.8       4.8          12.0     -15.2

## 4. Zero-shot value of pretrained weights on the held-out assay (Spearman of w·code vs fitness, mean ± std over seeds)

             mean             std         
arm           pre shuf_pre    pre shuf_pre
BLAT 2014   0.088    0.016  0.056    0.040
DYR 2023   -0.002    0.040  0.061    0.036
KKA2 2014   0.101   -0.022  0.034    0.069
MLAC 2023   0.090   -0.032  0.034    0.056
NUD15 2020  0.116    0.042  0.021    0.066
RASH 2017   0.069    0.017  0.066    0.046
RNC 2023    0.101    0.015  0.036    0.064
TPK1 2017   0.046    0.003  0.033    0.027
TPMT 2018   0.115    0.007  0.027    0.032
UBC9 2017   0.079    0.047  0.050    0.030
