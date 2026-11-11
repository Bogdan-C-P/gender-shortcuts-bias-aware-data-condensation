# Context-aware emotion recognition on CAER and EMOTIC

Code accompanying a manuscript on context-aware emotion recognition (CAER, CAER-S, EMOTIC), by Bogdan Popescu and
Corneliu Florea (LAPI, National University of Science and Technology Politehnica Bucharest).

This release contains the data preparation code and part of the analyses. The remaining code will be released when
the article is published.

The repository contains no images, videos, embeddings or model weights. CAER, CAER-S and EMOTIC must be obtained from
their authors.

## Contents

| folder | contents |
|---|---|
| `scripts/` | data preparation: frame extraction, face crops, image embeddings, VLM descriptions, transcripts |
| `experiments/` | analyses on frozen embeddings, fine-tuning and video models |
| `dcond/` | shared code: dataset loading, encoders, classifiers, metrics, plotting |
| `dcond/third_party/emotionclip/` | tokenizer and model code from EmotionCLIP (own licence) |
| `third_party/clip-caer/` | patch that adapts the official CLIP-CAER code to CAER |

## Setup

Python 3.12 on Linux, one 12 GB GPU.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The VLM descriptions (`requirements-vlm.txt`, vLLM) and the transcripts (`faster-whisper`) need separate environments.

| dataset | expected location |
|---|---|
| CAER videos | `$CAER_ROOT/{train,validation,test}/{Class}/NNNN.avi` (default `data/caer_raw/CAER`) |
| CAER-S images | `data/caers_raw/CAER-S/{train,test}/{Class}/NNNN.png` |
| EMOTIC | `$EMOTIC_ROOT/{train,val,test}.csv` + image folders (default `data/emotic_raw`) |

Derived data is written under `data/`, results under `experiments/results/`. Run scripts from the repository root.

## Licence

MIT (see `LICENSE`), except `dcond/third_party/emotionclip`, which keeps its own licence.
