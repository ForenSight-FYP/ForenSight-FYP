# Experiment log

Add one row per finished run. Numbers here go into the FYP report, so record the git
commit and config exactly.

| Date | Who | Module | Config | Commit | Train data | Test data | AUC | EER | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-17 | Hazim | Audio | configs/audio_aasist.yaml (official pretrained AASIST, no training) | ae9560c | ASVspoof 2019 LA train (by AASIST authors) | ASVspoof 2019 LA eval, all 71,237 clips (Hugging Face copy) | 0.9993 | 0.82% | Pipeline sanity check; paper reports ~0.83% |
| 2026-11-20 | Farhan, Hazim | Video | configs/video_effnet.yaml (EfficientNet-B4, 380 px, 8 epochs, best.pt = epoch 4) | d1eb0ba | FF++ c23 train face crops (22,630 frames) | FF++ c23 test, 280 videos (10,980 frames, mean of frame scores) | 0.9754 | 3.57% | Best val AUC 0.982. Target AUC >= 0.90 met. Test acc/F1 use a test-set EER threshold, so not reported |
| | | | | | | | | | |
