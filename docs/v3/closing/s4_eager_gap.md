# S4: the eager-path launch gap and the ruled bound

Status: **complete for the landed head `4c023985`** (`v3/final`; 5090 in 5.5, H100 in 5.6); the
default-commit numbers at `8854ccd4` stay in 5.3 and 5.4. Chapters 11 and 16 of the technical
report cite this file; it carries the S4 table and the bound, nothing else.

## 1. The bound as ruled

Per S4 cell (c1, c1d, c3, S3; definitions in section 3):

1. **CUDA, default execution** (no `execution` key, which resolves to cuda-graph, non-strict, on a
   CUDA device): ms/step **<= main eager** on every cell. Segments the executor refuses run eager
   with the reason logged; the cell still counts under the default column.
2. **CPU eager** (the CPU default): ms/step within **5 %** of main eager on every cell.
3. **CUDA explicit eager** (`execution: eager`): the gap to main eager is **reported**, not bounded.

Main is chengcli/snapy `730b35007d16389c64116e4bf9d2fe93707e29e9` (v2.14.1, the pre-V3 head; it
has no `execution` option, so main is eager only). Both sides build on kintera 2.6.12.

## 2. Method

Three columns per cell, same build of each side, one session, before/after interleaved, 3 reps,
median of the per-run medians (`docs/v3/closing/tools/speed/bench.py --warmup 5 --steps 100
--threads 8`, deck and `--dt-mode` per cell):

| column | side | deck key |
|---|---|---|
| main eager | 730b3500 | none |
| head default | V3 head | none |
| head eager | V3 head | `execution: eager` |

Launches per step: `bench.py ... --profile 4` with `SPEED_PROF_TRACE=1` writes the torch profiler
chrome trace; the count is the number of `kernel`, `gpu_memcpy` and `gpu_memset` events divided by
the profiled steps. The captured path is checked from the run's `execution_report` (captures,
replays, eager segments) and `capability.eager_reasons`.

## 3. Cells

| cell | deck | notes |
|---|---|---|
| c1 | `straka_hllc` FP64 | 1 block |
| c1d | `examples/straka.yaml` (shipped) | 1 block |
| c3 | `cubed6_moist` | 6 blocks in one process on CUDA; 6 ranks on CPU |
| S3 | `cubed6_moist_implicit`, `--dt-mode varying` | 6 blocks in one process |

## 4. Launches per step (eager)

Counted with the trace method above. The V3 column is the eager-counters content
(`f6a6fdaa`, library unchanged through `92a38547`); the combined head `70f6e15e` adds the seam
rotation cache, which removes exactly 1008 kernel launches on the six-panel decks and nothing
else (memcpy equal, no memsets).

| cell | main 730b3500 | V3 eager-counters | V3 combined head 70f6e15e |
|---|---|---|---|
| c1 | 1840 | 2001 | 2001 |
| c1 FP32 | - | 2145 | - |
| c1d | 1260 | 1407 | 1407 |
| c3 | 30096 | 36336 | 35322 (= 36330 - 1008 on the counting card) |
| S3 | 35616 | 43230 | 42216 (= 43224 - 1008 on the counting card) |

The main and eager-counters columns were counted on an H100; the combined-head column on a
1080 Ti, where the eager-counters tip counts 6 below the H100 numbers on both decks (a host and
driver effect present before the merge).

## 5. ms/step

### 5.1 RTX 5090 (GPU idle, host load below 3; kintera 2.6.12)

Explicit eager vs main eager, ratio of medians, at `3a2283bc` (the S4 decks run no settling, so
the later eager-counters commits leave these paths unchanged):

| cell | head eager / main eager | head cuda-graph ms/step |
|---|---|---|
| c1 | 1.112 | 3.590 (3 captures) |
| c1 FP32 | 1.123 | - |
| c1d | 1.131 | - |
| c3 | 1.274 | - |
| S3 | 1.228 | 27.645 (18 captures, 1890 replays) |

Default column: **pending** (the three-column run on the default commit).

### 5.2 H100 (shared node)

The node ran another user's training on all eight GPUs at 85-99 % utilisation and a host load of
60-125 throughout; every number here carries that caveat and none is a headline. Ratios of the
median of 3 run medians, eager-counters content `f6a6fdaa`:

| cell | CUDA: head eager / main eager | CPU: head eager / main eager |
|---|---|---|
| c1 | 1.089 | 1.057 |
| c1d | 1.109 | 1.054 |
| c3 | 1.213 | 1.000 (6 ranks) |
| S3 | 1.282 | - |

An earlier pass on the same node at `6d658e21` gave CPU 1.016 / 1.010 / 1.038 for c1 / c1d / c3;
the CPU ratios move with the host load, so the 5 % bound on CPU should be read from a quiet
window (none was available during the closing work; the gate scripts wait for load below 40 and
an idle GPU).

### 5.3 H100, the default commit (shared node)

Head `8854ccd4` (the default-execution pair on the #316 head, `dfd5a4c9`, on `939f7928`) against
main `730b3500`, both on kintera 2.6.12, one session, interleaved, 3 reps of 100 steps; ms/step is
the median of the 3 run medians with the min and max of the run medians in brackets. The other
job held every GPU at 93-99 % and the host load was 70-130 throughout.

| cell | main eager | head, no key (default) | head, `execution: eager` | default / main | eager / main |
|---|---|---|---|---|---|
| c1 | 19.03 [18.86, 19.24] | 8.85 [8.80, 8.89] | 19.84 [19.53, 21.00] | 0.465 | 1.043 |
| c1d | 13.68 [13.65, 14.65] | 7.75 [7.66, 7.75] | 14.66 [14.63, 14.86] | 0.566 | 1.071 |
| c3 | 128.64 [123.36, 137.48] | 55.25 [54.92, 56.38] | 153.13 [138.25, 155.69] | 0.429 | 1.190 |
| S3 | 154.20 [144.60, 191.13] | 73.03 [72.73, 73.50] | 187.17 [162.70, 282.41] | 0.474 | 1.214 |

Bound 1 (default <= main eager): holds on every cell. Capture report of the no-key runs: c1 and
c1d 3 captures, 315 replays, 0 eager segments; c3 and S3 18 captures, 1890 replays, 126 cuts,
0 eager segments; `eager_reasons` empty. The explicit `execution: eager` runs report 0 captures
and 315 / 1890 eager segments, so the key is honoured (bound 3: the gap reported, not bounded).

Launches/step for the two head columns (trace method, the same card):

| cell | head, no key (default) | head, `execution: eager` |
|---|---|---|
| c1 | 2019 (2016 kernels, 3 memcpy) | 2001 (1833, 168) |
| c1d | 1425 (1422, 3) | 1407 (1309, 98) |
| c3 | 35400 (35256, 144) | 35328 (32268, 3060) |
| S3 | 42384 (42240, 144) | 42222 (38604, 3618) |

The explicit-eager column equals the pre-wallbatch count on this card. Under the default the
profiler still records the replayed kernels one by one, so the count is about the same; the
host-to-device copies drop to the per-step dt and status fills, and the time saving in the table
comes from the replays, not from fewer kernels.

### 5.4 RTX 5090, the default commit

Head `8854ccd4` against main `730b3500` on an idle RTX 5090 (host load 1.4-3.2; kintera 2.6.12,
torch 2.10.0+cu128; clean CUDA Release build of the head, main from its earlier build; 3 interleaved
reps of 100 steps after 5 warm-up steps; ms/step). Explicit-eager ms/step is main times the ratio
posted for that column.

| cell | main eager | head, no key (default) | head, `execution: eager` | default / main | eager / main |
|---|---|---|---|---|---|
| c1 | 7.415 | 3.589 | 8.16 | 0.484 | 1.101 |
| c1 FP32 | 7.975 | 3.697 | 8.89 | 0.464 | 1.115 |
| c1d | 5.206 | 2.574 | 5.91 | 0.494 | 1.135 |
| c3 | 36.827 | 19.433 | 45.5 | 0.528 | 1.236 |
| S3 | 46.670 | 26.927 | 56.9 | 0.577 | 1.220 |

Bound 1 holds on every cell. With no key the default captures every cell with no refusal
(c1 cells 3 captures, 315 replays, 0 eager; c3 and S3 18 captures, 1890 replays, 0 eager); the
default's end state is byte-identical to explicit eager in every tensor in all 3 reps of every
cell. Launches/step main / default / explicit eager: c1 1840 / 2019 / 2001, c3 30096 / 35400 /
35328; the default's count includes the kernels replayed inside the graphs, and what the host
issues per step drops to 15 on c1 and about 1670 on c3.

### 5.5 RTX 5090, the landed head

`v3/final` at `4c023985` (12 commits on `939f7928`: #316, the default, the wall batch and its MPS
operands, the EfStatus save list) against main `730b3500` on the idle RTX 5090 under the setup of
5.4 (clean CUDA Release builds of both heads; 3 interleaved reps of 100 steps after 5 warm-up
steps; ms/step).

| cell | main eager | head, no key (default) | head, `execution: eager` | default / main | eager / main |
|---|---|---|---|---|---|
| c1 | 7.374 | 3.628 | 8.239 | 0.492 | 1.117 |
| c1 FP32 | 7.957 | 3.764 | 8.985 | 0.473 | 1.129 |
| c1d | 5.138 | 2.607 | 5.862 | 0.507 | 1.141 |
| c3 | 36.709 | 19.524 | 45.763 | 0.532 | 1.247 |
| S3 | 46.183 | 27.042 | 55.549 | 0.586 | 1.203 |

Bound 1 holds on every cell. The default captures every cell with no refusal and its end state is
byte-identical to explicit eager in every rep. Against `8854ccd4` (5.4) the default is 1-2 % slower
with 23-156 more eager launches per step, the cost of the wall-batch and EfStatus save-list
entries. Same host: the serial ctest by name gives 259 passed / 3 failed / 3 skipped of 265 on the
head against 256 / 3 / 3 of 262 on `939f7928`, no name changed verdict, the three new tests pass;
#349 on CUDA is byte-identical to `939f7928` in FP64 and FP32 for the default and for explicit
eager.

### 5.6 H100, the landed head (shared node)

Same node and caveat as 5.2 (host load 68-80, every GPU busy with another user's job). CPU eager,
3 interleaved reps of 30 steps after 3 warm-up steps, ms/step as the median of the 3 run medians
with the run medians in brackets:

| cell | main `730b3500` eager | head `4c023985` eager | head / main |
|---|---|---|---|
| c1 (1 rank, 8 threads) | 58.96 [56.56, 58.96, 59.95] | 61.78 [60.39, 61.78, 63.78] | 1.048 |
| c3 (6 ranks, 4 threads) | 217.64 [201.46, 217.64, 220.15] | 226.33 [221.82, 226.33, 247.69] | 1.040 |

Both are within the 5 % bound; the spread of the run medians (up to 12 % on c3) is the host load,
as in 5.2. #349 (23 cases) is byte-identical to `939f7928` in FP64 and FP32 on CPU and on CUDA.
Serial ctest by name, both heads built clean on this node: CUDA 246 passed / 14 failed / 5 skipped
of 265 against 243 / 14 / 5 of 262, the 14 names identical on both sides and all environmental
(the torchrun tests' fixed port while a second ctest ran on the node, the two-GPU ucx test under a
one-GPU pin, the restart and netcdf tests' output lookup on this node); CPU 199 passed / 12
failed / 19 skipped of 230 against 199 / 12 / 16 of 227, the 12 names identical. Six multi-process
names on the `939f7928` side first failed on the torchrun port while the two runs overlapped and
were rerun serially: they pass on both heads. No name changed verdict; the three new tests pass on
both devices.

## 6. Reading

The eager gap is launch-bound: the V3 eager step issues 9-21 % more launches than main on every
cell (section 4), and the ms/step ratios on an idle 5090 (section 5.1) order the same way as those
counts. The cuda-graph path replays the captured stages; on the shared H100 it ran c1 at 8.9 ms/step
against 19-20 ms eager and S3 at 104 ms against 217 ms eager (same session as section 5.2). That is
why cuda-graph became the CUDA default rather than closing the eager gap kernel by kernel: with
the default commit the no-key path runs at 0.46-0.58 of main eager on the idle 5090 and 0.43-0.57
on the shared H100, on every S4 cell, and 0.47-0.59 on the idle 5090 at the landed head (5.5).
