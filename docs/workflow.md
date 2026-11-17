# Team workflow

## Branches

- `main` always works: tests pass and the notebook runs.
- One branch per task: `video/frame-sampling`, `audio/aasist-eval`, `docs/lit-review`.
- Open a pull request into `main`; the other person reviews before merging.

## Ownership

| Area | Lead | Folder |
|---|---|---|
| Video module, Grad-CAM | Farhan | `models/video.py`, `data/video_preprocess.py` |
| Audio-visual module (FYP2) | Farhan | `models/av.py` (to add) |
| Audio module | Hazim | `models/audio.py`, `data/asvspoof.py` |
| Fusion, web app (FYP2) | Hazim | `fusion/`, `app/` (to add) |
| Metrics, evaluation, report | Both | `evaluation/` |

## Experiments

- Every run gets its own `output_dir` with `history.json`, `best.pt` and `*_metrics.json`.
- Log each finished run in `docs/experiments.md` (date, config, git commit, key numbers),
  because these results go straight into the FYP report.
- Fix the random seed (config `seed`) so results can be reproduced.

## Before pushing

```bash
pytest
ruff check src tests
```
