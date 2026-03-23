# VoiceIntent 🎙️
### Real-Time Speech Command Intent Recognition

> CNN-based speech classifier trained on MFCC features from the Fluent Speech Commands dataset.
> 30,043 utterances · 97 speakers · 31 intent classes · ~97% test accuracy

---

## Demo

```
Upload a WAV file → MFCC extraction → CNN inference → Intent + confidence in <100ms
```

**Supported commands (sample):**
- *"Turn on the lights in the bedroom"* → `activate__lights__bedroom`
- *"Increase the volume"* → `increase__volume__none`
- *"What's the weather like?"* → `get__weather__none`
- *"Bring me my newspaper"* → `bring__newspaper__none`

---

## Architecture

```
Raw audio (WAV/FLAC/OGG)
    │
    ▼
Resample → 16 kHz mono
    │
    ▼
MFCC extraction
  • 40 coefficients
  • 25ms window / 10ms hop
  • Fixed to 128 time frames
  • Per-feature normalisation
    │
    ▼
CNN (input: 40 × 128 × 1)
  ├── Block 1: Conv2D(64) × 2 → BN → MaxPool → Dropout(0.25)
  ├── Block 2: Conv2D(128) × 2 → BN → MaxPool → Dropout(0.25)
  ├── Block 3: Conv2D(256) → BN → GlobalAvgPool → Dropout(0.4)
  └── Dense(256) → Dropout(0.5) → Softmax(31)
    │
    ▼
Intent label + confidence scores
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/voiceintent.git
cd voiceintent
pip install -r requirements.txt
```

### 2. Download the dataset

Request access at https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/

Place it at:
```
data/
└── fluent_speech_commands/
    ├── data/
    │   ├── train_data.csv
    │   ├── valid_data.csv
    │   └── test_data.csv
    └── wavs/
        └── speakers/
            └── ...
```

### 3. Train

```bash
python train.py --data data/fluent_speech_commands --epochs 30
```

For a quick debug run (500 samples per split):
```bash
python train.py --data data/fluent_speech_commands --epochs 5 --max-samples 500
```

Training produces:
- `models/voiceintent_cnn.keras` — best checkpoint
- `models/label_encoder.pkl` — fitted label encoder

### 4. Evaluate

```bash
python evaluate.py --data data/fluent_speech_commands
```

Produces `outputs/confusion_matrix.png` and per-class metrics.

### 5. Launch the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Project Structure

```
voiceintent/
├── app.py              ← Streamlit UI (inference + visualisation)
├── train.py            ← Training script (CLI + importable)
├── evaluate.py         ← Evaluation + confusion matrix
├── requirements.txt
├── README.md
│
├── models/
│   ├── model.py        ← CNN architecture definition
│   ├── voiceintent_cnn.keras   ← (generated after training)
│   └── label_encoder.pkl       ← (generated after training)
│
├── utils/
│   ├── __init__.py
│   ├── audio.py        ← Audio I/O, resampling, normalisation
│   ├── features.py     ← MFCC extraction
│   ├── dataset.py      ← FSC dataset loader
│   └── intents.py      ← Intent label definitions
│
├── data/               ← Place dataset here (not committed)
└── outputs/            ← Evaluation outputs
```

---

## Results

| Split | Accuracy |
|-------|----------|
| Train | ~99.1%   |
| Val   | ~97.8%   |
| Test  | ~97.2%   |

*Results from training on the full FSC dataset for 30 epochs on a single GPU.*

---

## Dataset License

The Fluent Speech Commands dataset is used under the **Fluent Speech Commands Public License**.
- Non-commercial and academic use only.
- Do **not** redistribute the dataset.
- See `LICENSE_DATASET.txt` for full terms.

The code in this repository is MIT licensed.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Deep learning | TensorFlow / Keras |
| Audio features | Librosa |
| Audio I/O | SoundFile |
| ML utilities | Scikit-learn |
| UI | Streamlit |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib |

---

## Citation

If you use the Fluent Speech Commands dataset, please cite:

```
Lugosch, L., Ravanelli, M., Ignoto, P., Tomar, V. S., & Bengio, Y. (2019).
Speech Model Pre-Training for End-to-End Spoken Language Understanding.
Interspeech 2019.
```
