# Model evaluation report (chess_brain_ultra.pt)

Generated: 2026-10-18. Script: `scripts/eval_model.py` (deterministic seed 42).
Env: Windows 11, RTX 3050 (CUDA, torch 2.6.0+cu124), python-chess 1.11.2,
Stockfish sf_19 (local `engine/stockfish.exe`), model `models/chess_brain_ultra.pt`
(75.4 MB, tracked in git, loaded with `torch.load(weights_only=True)`).

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

## Results

| Metric | Value |
|---|---|
| Policy top-1 agreement vs SF d8 | **0.1467** (14.7 %) |
| Policy top-3 containment vs SF d8 | **0.2800** (28.0 %) |
| Mean abs value diff (model vs tanh(cp/400)) | **0.8070** |
| Pearson correlation (value) | **-0.0000** |

Console output of the run:

```
positions=300 device=cuda
... 300/300: top1=0.147 top3=0.280
policy top-1 agreement vs Stockfish depth 8: 0.1467
policy top-3 containment vs Stockfish depth 8: 0.2800
signal (mean abs value diff): 0.8070
pearson(corr): -0.0000
```

## Interpretation

- Top-1 ≈ 15 % is well above the random baseline but far from usable chess
  strength; top-3 ≈ 28 % confirms the policy head carries very little
  positional signal.
- The value head is statistically indistinguishable from noise
  (correlation ≈ 0, mean absolute error ≈ 0.81 ≈ 2 × expected spread of
  ±random labels).
- Consistent with the training-pipeline defects that were fixed in this
  branch (see README and git history): the dataset had duplicated
  deterministic games, a wrong material-perspective sign, value labels
  recorded from the white side for every position, underpromotions mapped
  onto the queen slot, and truncated games labeled as draws.

## Recommendation

Do not use `chess_brain_ultra.pt` for move selection. The repo's bot plays
with Stockfish, and the neural network is optional (`USE_NEURAL_FALLBACK=0`
by default). Retraining is only meaningful after re-checking that a
rebalanced dataset (proper perspective labels, deduplication, promotion
handling) yields a stepping-stone improvement measured with this same
script.
