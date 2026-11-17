# ForenSight

**An Explainable Multimodal Deepfake and Synthetic Media Detection System**
Final Year Project, Kulliyyah of ICT, IIUM (BICS 4321).

ForenSight estimates whether a recorded video or audio file is manipulated or AI-generated. It does this with three detection modules, a fusion step and an explainable report:

| Module | Model | Training data | Main metric |
|---|---|---|---|
| Video | EfficientNet-B4 on face crops | FaceForensics++ (cross-test: Celeb-DF) | AUC |
| Audio | AASIST (official weights) + a light mel-CNN baseline | ASVspoof 2019 LA → 2021 DF | EER |
| Audio-visual | SyncNet-style lip–speech features + classifier | FakeAVCeleb (backup: LAV-DF) | AUC |
| Fusion | Calibrated logistic regression on module scores | FakeAVCeleb | AUC |

## Repository layout

```
configs/            YAML configs per experiment
src/forensight/
  data/             preprocessing (frames, faces, audio) and dataset indexes
  models/           video, audio and (later) audio-visual models
  training/         training entry points
  evaluation/       metrics (EER, AUC, F1, FPR/FNR) and evaluation scripts
  fusion/           late fusion with missing-modality support
notebooks/          Colab notebooks (GPU training)
scripts/            setup helpers (third-party models)
docs/               dataset access, team workflow
tests/              pytest suite (runs on CPU, no datasets needed)
```

## Quick start

Local (CPU, for development and tests):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

GPU training: open `notebooks/colab_quickstart.ipynb` in Google Colab. It mounts Drive, installs the package and runs the training scripts.

```bash
# video module (FaceForensics++, after preprocessing)
python -m forensight.data.video_preprocess --config configs/video_effnet.yaml
python -m forensight.training.train_video --config configs/video_effnet.yaml

# audio module
python -m forensight.training.train_audio --config configs/audio_melcnn.yaml
bash scripts/setup_aasist.sh   # official AASIST code + pretrained weights
python -m forensight.evaluation.eval_audio --config configs/audio_aasist.yaml
```

## Status (FYP1)

- [x] Repo skeleton, metrics, fusion, test suite
- [x] Video preprocessing (frame sampling + face cropping) and EfficientNet training loop
- [x] ASVspoof protocol parsing, audio preprocessing, mel-CNN baseline and AASIST wrapper
- [ ] Dataset access approved (see `docs/datasets.md`)
- [ ] First baseline results: video AUC on FF++, audio EER on ASVspoof 2021 DF
- [ ] Audio-visual module (FYP2)
- [ ] Explainability (Grad-CAM, timeline, spectrogram saliency) (FYP2)
- [ ] Flask web app and PDF report (FYP2)

## Team

| Area | Lead |
|---|---|
| Video and audio-visual modules, Grad-CAM | Farhan |
| Audio module, fusion, web app | Hazim |
| Data preparation, explainability report, evaluation | Both |

See `docs/workflow.md` for branching and review rules.
