# Datasets: access and setup

Approval can take days to weeks, so **send the requests in week 1**. Use your IIUM email
and name the supervisor once one has agreed. Never commit dataset files to git (they are
licence-restricted; `.gitignore` already blocks common media formats).

| Dataset | Used for | How to get it | Status |
|---|---|---|---|
| FaceForensics++ (c23) | Video train/val/test | Fill in the Google Form linked from the FF++ GitHub README (github.com/ondyari/FaceForensics). You receive a download script. Download only `c23` videos. | Not requested |
| Celeb-DF v2 | Video cross-dataset test | Request form linked from github.com/yuezunli/celeb-deepfakeforensics | Not requested |
| ASVspoof 2019 LA | Audio train/dev/eval | Open download from the University of Edinburgh DataShare (search "ASVspoof 2019 database") | Not downloaded |
| ASVspoof 2021 DF | Audio cross-test | Open download (eval audio on Zenodo) + keys from asvspoof.org | Not downloaded |
| FakeAVCeleb | Audio-visual + fusion | Request form linked from github.com/DASH-Lab/FakeAVCeleb | Not requested |
| LAV-DF | Audio-visual backup | Accept terms on Hugging Face (ControlNet/LAV-DF) | Not requested |

## Storage plan

- Keep raw datasets in Google Drive: `MyDrive/forensight/datasets/<name>/`.
- For training, copy or unzip to the Colab local disk (`/content/...`) first. Reading
  thousands of small files from Drive is very slow.
- Preprocessed face crops for FF++ c23 at 32 frames per video are roughly
  tens of GB; if Drive space is tight, lower `num_frames` or `image_size` in the config.

## Request email template

> Subject: Dataset access request: [dataset name] for undergraduate research
>
> Dear [maintainers],
>
> We are final-year Computer Science students at the International Islamic University
> Malaysia (IIUM), working on a final year project on explainable deepfake detection,
> supervised by [supervisor name]. We would like to request access to [dataset name]
> for non-commercial academic research only, and we agree to the dataset's terms of use.
>
> Names: Farhan Aiman bin Ahmad Nadzri, Muhammad Hazim bin Rosnizam
> Institution: Kulliyyah of ICT, IIUM
>
> Thank you,
> [name]
