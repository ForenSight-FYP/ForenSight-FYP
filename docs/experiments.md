# Experiment log

Add one row per finished run. Numbers here go into the FYP report, so record the git
commit and config exactly.

| Date | Who | Module | Config | Commit | Train data | Test data | AUC | EER | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-17 | Hazim | Audio | configs/audio_aasist.yaml (official pretrained AASIST, no training) | ae9560c | ASVspoof 2019 LA train (by AASIST authors) | ASVspoof 2019 LA eval, all 71,237 clips (Hugging Face copy) | 0.9993 | 0.82% | Pipeline sanity check; paper reports ~0.83% |
| | | | | | | | | | |
