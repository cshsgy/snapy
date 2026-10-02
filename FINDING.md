# study/250 O3

## Question

On the 50 m Bryan bubble (cfl 0.9, tlim 1000 s, CPU), D = −min(θ_e′) is 0.2576904296875 K with the default smooth5 density reference. θ_e′ is absolute Bryan θ_e minus the t = 0 left-edge column. O3 asks whether that cold drop is caused by the x1 density reference (the density restored on the face is not on the same thermodynamic path as the pressure reference), and whether a different reference, moisture inside that reference, or vertical WENO scaling removes it.

## Arms and code

Every snapy number below is taken from the posts in thread 1790815809.274189. Nothing was rerun for this note. D0 is the posted smooth5 value, 0.2576904296875 K (step 5: bitwise identical to the earlier D0 file). One thread, except the moist arms, which the step 5 launch set to 8 intra-op threads. Positivity fallback on every posted bubble arm: il = 0, ir = 0.

| arm | what ran | code |
| --- | --- | --- |
| smooth5 | default 5-point density reference, vertical scale false | `21bc2a1512e66bffc8f22cd266a5b36041f086e3` |
| N | both density references zeroed | step 4 `1ffb3c16d8c6e182461876a30734af73320a45a9` |
| I | dry adiabat of the bottom interior cell | same step 4 sha. Step 6 reran I on the on-disk `21bc2a15` binary and matched this D and this θ_e′ max bit for bit |
| LP | local polytrope, deck only | `21bc2a15` |
| C | moist adiabat through the bottom interior cell | `21bc2a15` |
| M | per-cell reversible saturated adiabat; face density is the mean of cells i−1 and i | `dcb59eecd17942b558b8628765c73a56f6f46070` |
| S | smooth5, `reconstruct.vertical.scale` true | `21bc2a15` |
| N_s | none, vertical scale true | on-disk binary of `21bc2a15` (the library has no `WB_FACE_MEAN`). Source HEAD was `dcb59ee`; that source is not what ran |
| I_s | isentrope, vertical scale true | same binary as N_s. The deck changes only `reconstruct.vertical.scale` |

Step 6 logs do not contain `HydroOptions : vertical` or `* scale =`. Reconstruction is printed only when verbose is on, and these decks do not set that key.

The three Athena 50 m values in the table (none 0.346 K, smooth5 0.135 K, isentrope 0.051 K) are the figures named in the ask. They were not measured here. Athena D/D0 uses 0.135 K. Snapy D/D0 uses 0.2576904296875 K. No Athena θ_e′ max was given.

## One table

D/D0 is the quotient of the D column and that row's smooth5, rounded to 10 decimal places. Bar ratios already posted are repeated in the next section; this column does not replace them.

| arm | D (K) | D/D0 | θ_e′ max (K) |
| --- | --- | --- | --- |
| smooth5 | 0.2576904296875 | 1.0000000000 | 5.1402587890625 |
| N | 0.264190673828125 | 1.0252250118 | 4.642791748046875 |
| I | 0.136749267578125 | 0.5306726670 | 4.151153564453125 |
| LP | 0.330078125 | 1.2809095216 | 4.718353271484375 |
| C | 0.137115478515625 | 0.5320937944 | 4.15521240234375 |
| M | 0.3985595703125 | 1.5466603505 | 5.06146240234375 |
| S | 0.173370361328125 | 0.6727854098 | 4.87652587890625 |
| N_s | 0.33935546875 | 1.3169114164 | 4.62078857421875 |
| I_s | 0.09527587890625 | 0.3697299858 | 3.811065673828125 |
| Athena none | 0.346 | 2.5629629630 | — |
| Athena smooth5 | 0.135 | 1.0000000000 | — |
| Athena isentrope | 0.051 | 0.3777777778 | — |

Each posted snapy maximum is shared by a mirror pair. The cells are in the step 6 post.

## Settled

Reference is causal. Zeroing the density reference leaves D_N = 0.264190673828125 K (+2.52% against D0). The bottom-cell dry adiabat gives D_I = 0.136749267578125 K (−46.93% against D0). Step 4 recorded path consistency supported: D_I ≤ 0.180 and D_N ≥ 0.232. The step 6 unscaled rerun matched that I frame, including the maximum.

Moisture in the column reference has no effect. D_C / D_I = 1.00267797, which is ≥ 0.9, so the moist column and the dry column meet the posted bar.

Per-cell centred forms are the worst drops. D_LP = 0.330078125 K (posted 1.28090952 × D0) and D_M = 0.3985595703125 K are larger than smooth5, N, I, C, S, and I_s. D_M / D_LP = 1.20747041, outside ±10% of D_LP, and D_M is not ≤ 0.5 D_LP. Per-cell moisture is not a null. The null is the column result above.

Scale interacts with the reference. It does not shrink every arm. Posted ratios: D_S / D0 = 0.67278541, D_N_s / D_N = 1.2845096588134766, D_I_s / D_I = 0.6967194676399231. None + scale is larger than none. Isentrope + scale is smaller than isentrope. None of the three named step-6 bars fired.

## Not settled

No posted form meets a 0.02 K floor at 50 m, and 50 m is the only spacing these O3 posts contain. The smallest snapy D is I_s = 0.09527587890625 K. Athena's 50 m isentrope is 0.051 K. Both are above 0.02 K.

Why snapy's isentrope (0.136749267578125 K) is about 2.7× Athena's 50 m isentrope (0.051 K) was not measured. The quotient of those two figures is 2.681.
