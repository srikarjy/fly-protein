## 1. Gain over random walk, reach top 1% (points). Pooled over 10 held-out assays x 10 seeds; ± = SE over the 100 (assay, seed) pairs

| arm | gain over random | gain, top-10% reach |
|---|---|---|
| chemotaxis | +18.7 ± 1.9 | +3.2 ± 0.5 |
| zero | -5.6 ± 1.1 | -1.4 ± 0.4 |
| zero_mem | -10.9 ± 2.0 | -2.8 ± 0.8 |
| pre | -24.6 ± 1.9 | -16.2 ± 1.1 |
| pre_mem | -21.2 ± 2.1 | -13.5 ± 1.0 |
| pre_frozen | -25.5 ± 1.9 | -16.3 ± 1.1 |
| shuf_pre | -26.1 ± 1.9 | -21.5 ± 1.3 |
| shuf_pre_mem | -23.9 ± 2.0 | -17.2 ± 1.1 |

## 2. Paired differences (points, top 1%); ± = SE over 100 pairs; 'wins' = held-out assays where the seed-mean difference is > 0

| comparison | mean diff | wins (of 10 assays) |
|---|---|---|
| pre − zero | -19.0 ± 1.7 | 0 |
| pre_mem − zero_mem | -10.4 ± 1.9 | 2 |
| pre_mem − zero | -15.7 ± 1.8 | 2 |
| pre_frozen − zero | -20.0 ± 1.8 | 0 |
| pre − shuf_pre | +1.5 ± 1.6 | 9 |
| pre_mem − shuf_pre_mem | +2.7 ± 1.7 | 8 |
| zero_mem − zero | -5.3 ± 1.5 | 4 |

## 3. Per held-out assay: gain over random (mean over 10 seeds)

arm         chemotaxis  zero  zero_mem   pre  pre_mem  shuf_pre
BLAT 2014         19.4  -1.0       1.8 -28.8    -20.6     -33.4
DYR 2023          38.2  -0.0      -3.6 -17.8    -16.4     -18.4
KKA2 2014         -7.0 -23.0     -40.4 -48.2    -48.0     -49.8
MLAC 2023         30.0  -3.8      -3.8 -34.6    -35.2     -27.2
NUD15 2020        -4.4  -2.2     -24.2 -35.0    -34.2     -38.0
RASH 2017         27.0   1.6       4.0 -10.0     -3.8     -11.8
RNC 2023          44.2   3.0      -2.2  -4.2      3.6      -5.0
TPK1 2017          4.4 -15.6     -28.4 -24.2    -25.0     -24.2
TPMT 2018         11.6 -11.4      -9.6 -37.4    -30.4     -40.0
UBC9 2017         23.8  -3.2      -2.2  -5.4     -2.2     -13.0

## 4. Zero-shot value of pretrained weights on the held-out assay (Spearman of w·code vs fitness, mean ± std over 10 seeds)

             mean             std         
arm           pre shuf_pre    pre shuf_pre
BLAT 2014   0.096    0.022  0.060    0.044
DYR 2023    0.033   -0.013  0.045    0.054
KKA2 2014   0.077   -0.002  0.048    0.060
MLAC 2023   0.059   -0.005  0.038    0.032
NUD15 2020  0.139   -0.024  0.084    0.084
RASH 2017   0.045   -0.006  0.056    0.057
RNC 2023    0.060    0.034  0.074    0.064
TPK1 2017   0.036   -0.009  0.034    0.031
TPMT 2018   0.087    0.010  0.046    0.060
UBC9 2017   0.084   -0.012  0.030    0.054
