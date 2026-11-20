# Experiment log

Add one row per finished run. Numbers here go into the FYP report, so record the git
commit and config exactly.

| Date | Who | Module | Config | Commit | Train data | Test data | AUC | EER | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-17 | Hazim | Audio | configs/audio_aasist.yaml (official pretrained AASIST, no training) | ae9560c | ASVspoof 2019 LA train (by AASIST authors) | ASVspoof 2019 LA eval, all 71,237 clips (Hugging Face copy) | 0.9993 | 0.82% | Pipeline sanity check; paper reports ~0.83% |
| 2026-11-20 | Farhan, Hazim | Video | configs/video_effnet.yaml (EfficientNet-B4, 380 px, 8 epochs, best.pt = epoch 4) | d1eb0ba | FF++ c23 train face crops (22,630 frames) | FF++ c23 test, 280 videos (10,980 frames, mean of frame scores) | 0.9754 | 3.57% | Best val AUC 0.982. Target AUC >= 0.90 met. Test acc/F1 use a test-set EER threshold, so not reported |
| 2026-11-20 | Hazim | Audio | configs/audio_aasist.yaml (official pretrained AASIST, no training), chunk_size=20000 | 48bb292 | ASVspoof 2019 LA train (by AASIST authors) | ASVspoof 2021 DF eval, part00 subset: 133,464 of 533,928 eval trials | 0.9163 | 17.04% | Matches the 17.04% published for this checkpoint on the full DF21 set (SpeechAntiSpoofingBenchmarks/AASIST), so part00 is representative |
| 2026-11-18 | Farhan | Audio | configs/audio_melcnn.yaml (MelCNN, 20 epochs, best.pt = epoch 17 by dev EER) | main before PR #8 (not recorded) | ASVspoof 2019 LA train (LA.zip, Zenodo 6906306) | ASVspoof 2019 LA eval, all 71,237 clips (Hugging Face copy) | 0.9287 | 14.35% | Dev EER 0.35%: large dev-to-eval gap, overfits the 6 training attacks |
| 2026-11-20 | Hazim | Audio | MelCNN above, scored with eval_audio, chunk_size=20000 | 48bb292 | ASVspoof 2019 LA train | ASVspoof 2021 DF eval, part00 subset (133,464 trials) | 0.8522 | 21.86% | Same clips as the AASIST DF21 row |
| | | | | | | | | | |
