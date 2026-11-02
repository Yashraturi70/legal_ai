# Legal Outcome Prediction — HLDC Bail Demo

Fast, reproducible Phase-1 implementation for the legal outcome prediction project.

**Input:** HLDC `facts-and-arguments` only.  
**Output:** Bail Granted (1) / Bail Denied (0).

Judge opinion/result must not be supplied to the predictive model because that causes outcome leakage.

## Run
1. Put HLDC/IL-TUR parquet files in `data/`.
2. Install: `pip install -r requirements.txt`
3. Train:
`python train.py --train data/train_all.parquet --valid data/dev_all.parquet --test data/test_all.parquet`
4. Start demo:
`python app.py`
5. Open `http://127.0.0.1:5000`.

The live demo deliberately uses a fast TF-IDF + Logistic Regression baseline so it can be trained quickly. It also provides similar-case retrieval and feature evidence. A transformer experiment can be added after the baseline is working.

Do not claim local accuracy/F1 until `train.py` has actually been run on the dataset.
