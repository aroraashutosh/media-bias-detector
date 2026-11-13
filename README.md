# Geopolitical Media Framing & Bias Detector

Detects loaded (biased) wording in news sentences and explains which words trigger it.
RoBERTa-base fine-tuned on the BABE dataset; also compares against TF-IDF + Logistic Regression and a TextCNN.

## Results (735 held-out test sentences)
| Model | Accuracy |
|---|---|
| TF-IDF + Logistic Regression | 73.2% |
| TextCNN (best of 3 variants, 5 seeds) | 73.1% ± 0.8 |
| RoBERTa-base fine-tuned (3 seeds) | 79.0% ± 2.5 (76.3 / 79.6 / 81.2) |

## Files
- `cnn_babe.py` - logistic regression + TextCNN experiments (`python cnn_babe.py babe.xlsx results.json`)
- `roberta_babe.py` - RoBERTa fine-tuning and evaluation (set `SEED` at the top)
- `bias_demo_colab.py` - trains, saves the model, opens a simple demo app (Google Colab, T4 GPU)
- `bias_analyzer_colab.py` - full analyzer: word-level reasons, whole articles, live news (RSS), compare two sources
- `space/` - package for a permanent Hugging Face Space (set `MODEL_ID` in `app.py`)

## Run
1. Open Google Colab, choose a T4 GPU runtime, `pip install transformers`.
2. Paste and run `bias_demo_colab.py` (BABE downloads automatically), then `bias_analyzer_colab.py`.

## Limits
Measures wording, not truth. Trained on US political news. Misses favourable spin. Labels are subjective.
