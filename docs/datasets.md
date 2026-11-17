# Datasets: access and setup

Approval can take days to weeks, so **send the requests in week 1**. Use your IIUM email
and name the supervisor once one has agreed. Never commit dataset files to git (they are
licence-restricted; `.gitignore` already blocks common media formats).

| Dataset | Used for | How to get it | Status |
|---|---|---|---|
| FaceForensics++ (c23) | Video train/val/test | Fill in the Google Form linked from the FF++ GitHub README (github.com/ondyari/FaceForensics). You receive a download script. Download only `c23` videos. | Not requested |
| Celeb-DF v2 | Video cross-dataset test | Request form linked from github.com/yuezunli/celeb-deepfakeforensics | Not requested |
| ASVspoof 2019 LA | Audio train/dev/eval | **Eval:** notebook 2a, from Hugging Face (SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA: official eval audio + CM protocol, ODC-By). **Train/dev:** notebook 2c, the full LA.zip (7.6 GB, MD5 30c98f11d8b2bc21f2c257bfd78bb5c5) from the organisers' Zenodo record 6906306 (DOI 10.7488/ds/2555, ODC-By). Edinburgh DataShare (handle 10283/3336) returns 403 to Colab. | Automated |
| ASVspoof 2021 DF | Audio cross-test | Automated in notebook section 2d: keys from asvspoof.org (DF-keys-full.tar.gz) + eval audio from Zenodo record 4835108 (4 parts, ~8.6 GB each, 34.5 GB total). By default only part00 is used; results are reported as a part00 subset. | Automated (part00) |
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
