# Model evaluation report

Generated: 2026-10-18. Script: `scripts/eval_model.py` (deterministic seed 42).
Env: Windows 11, RTX 3050 (CUDA, torch 2.6.0+cu124), python-chess 1.11.2,
Stockfish sf_19 (local `engine/stockfish.exe`).

Two checkpoints are reported:

- `models/chess_brain_ultra.pt` (75.4 MB) - the original, trained with the
  pre-fix pipeline that had the defects listed below.
- `models/chess_brain_20261018.pt` - retrained **after** the pipeline fixes
  with `brain/train.py --epochs 20 --num-games 2500 --seed 42`
  (2500 greedy-material self-play games, ~10 % random moves, value labels
  NaN-masked for unfinished games). The saved checkpoint is the best
  validation loss (epoch 2; further epochs overfit the synthetic set).

## Results

| Metric | chess_brain_ultra.pt | chess_brain_20261018.pt |
|---|---|---|
| Policy top-1 agreement vs SF d8 | **0.1467** (14.7 %) | **0.2000** (20.0 %) |
| Policy top-3 containment vs SF d8 | **0.2800** (28.0 %) | **0.3800** (38.0 %) |
| Mean abs value diff (model vs tanh(cp/400)) | **0.8070** | **0.6408** |
| Pearson correlation (value) | **-0.0000** | **0.4438** |

## Method

- 300 positions, sampled from short random-play games seeded by 4 fixed
  opening lines, plies >= 4, seed 42 (`collect_positions` in the script is
  reproducible bit-for-bit).
- **Policy metric**: top-1 and top-3 agreement against Stockfish depth 8,
  computed on the 4096 (from,to) index space that the network actually
  outputs; promotions are collapsed to the queen variant the same way the
  bot resolves them (`brain/encoder.py::policy_best_move`).
- **Value metric**: model value (tanh output) against `tanh(cp/400)` of
  Stockfish depth 8 (white-relative, mate scores saturated at ±10000 cp).
- Random/legal-only baseline for top-1 would be roughly 1 / (avg. number of
  legal moves), i.e. a few percent.

## Results (chess_brain_ultra.pt, pre-retrain snapshot)

| Metric | Value |
|---|---|
| Policy top-1 agreement vs SF d8 | **0.1467** (14.7 %) |
| Policy top-3 containment vs SF d8 | **0.2800** (28.0 %) |
| Mean abs value diff (model vs tanh(cp/400)) | **0.8070** |
| Pearson correlation (value) | **-0.0000** |

Console output of the retrained checkpoint run (`chess_brain_20261018.pt`):

```
positions=300 device=cuda
... 300/300: top1=0.200 top3=0.380
policy top-1 agreement vs Stockfish depth 8: 0.2000
policy top-3 containment vs Stockfish depth 8: 0.3800
signal (mean abs value diff): 0.6408
pearson(corr): 0.4438
```

## Interpretation

- The fixes in this branch produced a real improvement: top-1 went from
  ≈15 % to 20 %, top-3 from ≈28 % to 38 %, and the value head now carries
  actual signal (Pearson ≈ 0.44, mean abs diff 0.64). The previous
  checkpoint's value head was statistically indistinguishable from noise
  (correlation ≈ 0, mean absolute error ≈ 0.81).
- The fixed defects were: duplicated deterministic games, a wrong
  material-perspective sign, value labels recorded from the white side for
  every position, underpromotions mapped onto the queen slot, and truncated
  games labeled as draws.
- Both checkpoints are still far from usable chess strength: the training
  signal is greedy-material self-play against 4096 output classes, and
  validation loss degrades monotonically after ~2 epochs on this synthetic
  set (overfitting), so the checkpoint is capped by the data, not by the
  architecture.

## Recommendation

Do not use either checkpoint for standalone move selection. The repo's bot
plays with Stockfish, and the neural network is optional
(`USE_NEURAL_FALLBACK=0` by default). The next useful step for the neural
part would be better supervision than greedy-material self-play data
(e.g. Grandmaster PGNs - the code from quoting them is in `data/` - or an
AlphaZero-style tree search), re-measured with this same script so that
numbers remain comparable.
