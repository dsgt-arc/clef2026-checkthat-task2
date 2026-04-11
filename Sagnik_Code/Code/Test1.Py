"""
CLEF 2026 CheckThat! Task 2 - Reasoning Trace Ranking + Verdict Prediction
Pipeline:
  1. Load train/test JSON data
  2. Extract features from (claim, evidence, reasoning_trace) tuples
  3. Train a verifier model to score/rank reasoning traces
  4. Aggregate top-k traces to predict final verdict
  5. Evaluate with Recall@k, MRR@k, macro F1

Dependencies:
    pip install sentence-transformers scikit-learn numpy torch

Conda location:
Run from: /storage/home/hcoda1/3/ssinha348/miniconda3
Run command: /storage/home/hcoda1/3/ssinha348/miniconda3/envs/env_checkthat/bin/python /storage/project/ps-clef2026_checkthat_t2-0/ssinha348/Code/Test1.Py
conda activate env_checkthat
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from sentence_transformers import SentenceTransformer, CrossEncoder
import pandas as pd
import os

# ─────────────────────────── CONFIG ───────────────────────────
DATA_DIR = Path("/storage/project/ps-clef2026_checkthat_t2-0/shared/Data/English")
TRAIN_FILE = DATA_DIR / "clef_2026_checkthat_english_train.json"
TEST_FILE  = DATA_DIR / "clef2026_gpt4_o_mini_val.json"

output_dir = Path("/storage/project/ps-clef2026_checkthat_t2-0/ssinha348/Results/")
os.makedirs(output_dir, exist_ok=True)

TOP_K = 5                          # k for Recall@k and MRR@k
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # fast & effective
BI_ENCODER_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"  # fallback features


# ─────────────────────────── DATA LOADING ────────────────────
def load_json(path: Path):
    with open(path) as f:
        data = json.load(f)
    # Accept both a list of records and a single record wrapped in a list
    if isinstance(data, dict):
        data = [data]
    return data


def build_samples(records):
    """
    Flatten each record into one sample per reasoning trace.
    Returns list of dicts and a list of record-level dicts for verdict eval.
    """
    trace_samples = []   # for ranking model
    record_level  = []   # for final verdict eval

    for rec in records:
        claim     = rec["claim"]
        evidences = " ".join(rec.get("evidences", []))
        traces    = rec["Reasoning_traces"]
        verdicts  = rec["Verdict_list"]
        label     = rec["label"]          # ground-truth label for the claim

        # A trace is "correct" if its verdict matches the record label
        correct_indices = [
            i for i, v in enumerate(verdicts)
            if v.strip().lower() == label.strip().lower()  # label can be True/False/Conflicting
        ]

        for i, (trace, verdict) in enumerate(zip(traces, verdicts)):
            trace_samples.append({
                "claim":           claim,
                "evidences":       evidences,
                "trace":           trace,
                "trace_verdict":   verdict,
                "label":           label,
                "is_correct":      int(i in correct_indices),
                "record_id":       id(rec),   # group traces back to their record
            })

        record_level.append({
            "claim":           claim,
            "evidences":       evidences,
            "traces":          traces,
            "verdicts":        verdicts,
            "label":           label,
            "correct_indices": correct_indices,
            "record_id":       id(rec),
        })

    return trace_samples, record_level


# ─────────────────────── FEATURE EXTRACTION ──────────────────
class FeatureExtractor:
    """
    Combines cross-encoder relevance score with simple lexical features.
    Falls back to bi-encoder cosine similarity if cross-encoder is slow.
    """
    def __init__(self, use_cross_encoder=True):
        self.bi_enc = SentenceTransformer(BI_ENCODER_MODEL)
        self.cross_enc = None
        if use_cross_encoder:
            try:
                self.cross_enc = CrossEncoder(CROSS_ENCODER_MODEL)
                print(f"[INFO] CrossEncoder loaded: {CROSS_ENCODER_MODEL}")
            except Exception as e:
                print(f"[WARN] CrossEncoder failed: {e}. Using bi-encoder only.")

    def _text_for_sample(self, s):
        return f"Claim: {s['claim']}\nEvidence: {s['evidences']}\nTrace: {s['trace']}"

    def extract(self, samples):
        """Return feature matrix (N, D)."""
        print(f"[INFO] Extracting features for {len(samples)} trace samples …")

        # ── Feature 1: BI-ENCODER similarity between (claim+evidence) and trace ──
        claim_evid_texts = [f"{s['claim']} {s['evidences']}" for s in samples]
        trace_texts      = [s["trace"] for s in samples]

        ce_emb  = self.bi_enc.encode(claim_evid_texts, batch_size=64,
                                      show_progress_bar=True, normalize_embeddings=True)
        tr_emb  = self.bi_enc.encode(trace_texts, batch_size=64,
                                      show_progress_bar=True, normalize_embeddings=True)
        cos_sim = np.sum(ce_emb * tr_emb, axis=1, keepdims=True)   # (N,1)

        # ── Feature 2: Cross-encoder score (query = claim+evid, doc = trace) ──
        if self.cross_enc is not None:
            pairs  = [(f"{s['claim']} {s['evidences']}", s["trace"]) for s in samples]
            ce_scores = self.cross_enc.predict(pairs, batch_size=32,
                                               show_progress_bar=True)
            ce_scores = ce_scores.reshape(-1, 1)
        else:
            ce_scores = cos_sim   # fallback

        # ── Feature 3: Trace length (longer ≈ more detailed reasoning) ──
        lengths = np.array([[len(s["trace"].split())] for s in samples], dtype=float)
        lengths = lengths / (lengths.max() + 1e-9)   # normalise

        # ── Feature 4: Does the trace contain "REFUTES"/"SUPPORTS"/"NEI"? ──
        label_flags = []
        for s in samples:
            t = s["trace"].upper()
            label_flags.append([
                int("SUPPORT" in t),
                int("REFUTE" in t),
                int("NEI" in t or "NOT ENOUGH" in t),
                int("CONFLICTING" in t),
                int("FALSE" in t),
                int("TRUE" in t),
            ])
        label_flags = np.array(label_flags, dtype=float)

        features = np.hstack([ce_scores, cos_sim, lengths, label_flags])
        print(f"[INFO] Feature matrix shape: {features.shape}")
        return features


# ──────────────────────── VERIFIER MODEL ─────────────────────
class TraceVerifier:
    """
    Binary classifier: correct trace (1) vs. incorrect trace (0).
    Scoring probability used for ranking.
    """
    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")

    def fit(self, X, y):
        self.clf.fit(X, y)
        print("[INFO] Verifier trained.")

    def score(self, X):
        """Returns probability of being a 'correct' trace."""
        return self.clf.predict_proba(X)[:, 1]


# ─────────────────────── RANKING EVALUATION ──────────────────
def recall_at_k(ranked_indices, correct_set, k):
    """Recall@k for a single query."""
    top_k = set(ranked_indices[:k])
    if not correct_set:
        return 0.0
    return len(top_k & correct_set) / len(correct_set)


def reciprocal_rank(ranked_indices, correct_set):
    """MRR contribution for a single query."""
    for rank, idx in enumerate(ranked_indices, start=1):
        if idx in correct_set:
            return 1.0 / rank
    return 0.0


def evaluate_ranking(record_level, scores_by_record):
    """
    scores_by_record: dict {record_id: np.array of scores per trace}
    Returns Recall@K and MRR@K averaged over all records.
    """
    recalls, mrrs = [], []
    for rec in record_level:
        rid      = rec["record_id"]
        scores   = scores_by_record[rid]
        ranked   = np.argsort(-scores).tolist()    # descending
        correct  = set(rec["correct_indices"])

        recalls.append(recall_at_k(ranked, correct, TOP_K))
        mrrs.append(reciprocal_rank(ranked, correct))

    return np.mean(recalls), np.mean(mrrs)


# ─────────────────── VERDICT FROM TOP-K TRACES ───────────────
def predict_verdict(record_level, scores_by_record):
    """
    For each record: pick top-K traces, majority-vote their verdicts.
    Returns list of predicted verdicts (one per record).
    """
    predictions = []
    for rec in record_level:
        rid    = rec["record_id"]
        scores = scores_by_record[rid]
        top_k  = np.argsort(-scores)[:TOP_K]
        top_verdicts = [rec["verdicts"][i] for i in top_k]
        pred = Counter(top_verdicts).most_common(1)[0][0]  # whatever verdict wins majority
        predictions.append(pred)
    return predictions


# ─────────────────────── MAIN PIPELINE ───────────────────────
def main():
    # 1. Load data
    print("[STEP 1] Loading data …")
    train_records = load_json(TRAIN_FILE)
    test_records  = load_json(TEST_FILE)
    print(f"  Train records: {len(train_records)}  |  Test records: {len(test_records)}")

    # 2. Build samples
    train_samples, train_record_level = build_samples(train_records)
    test_samples,  test_record_level  = build_samples(test_records)
    print(f"  Train traces: {len(train_samples)}  |  Test traces: {len(test_samples)}")

    # 3. Feature extraction
    print("[STEP 2] Feature extraction …")
    extractor   = FeatureExtractor(use_cross_encoder=True)
    X_train     = extractor.extract(train_samples)
    X_test      = extractor.extract(test_samples)
    y_train     = np.array([s["is_correct"] for s in train_samples])

    # 4. Train verifier
    print("[STEP 3] Training verifier …")
    verifier = TraceVerifier()
    verifier.fit(X_train, y_train)

    # 5. Score test traces & group by record
    print("[STEP 4] Scoring test traces …")
    test_scores = verifier.score(X_test)

    # Group scores back to records preserving order
    scores_by_record = {}
    pointer = 0
    for rec in test_record_level:
        n = len(rec["traces"])
        scores_by_record[rec["record_id"]] = test_scores[pointer:pointer + n]
        pointer += n

    # 6. Ranking evaluation
    print("[STEP 5] Ranking evaluation …")
    recall, mrr = evaluate_ranking(test_record_level, scores_by_record)
    print(f"  Recall@{TOP_K}: {recall:.4f}")
    print(f"  MRR@{TOP_K}:    {mrr:.4f}")

    # 7. Verdict prediction & classification evaluation
    print("[STEP 6] Verdict prediction …")
    pred_verdicts  = predict_verdict(test_record_level, scores_by_record)
    true_verdicts  = [rec["label"] for rec in test_record_level]

    # Encode labels
    le = LabelEncoder()
    le.fit(true_verdicts + pred_verdicts)
    y_true_enc = le.transform(true_verdicts)
    y_pred_enc = le.transform(pred_verdicts)

    macro_f1 = f1_score(y_true_enc, y_pred_enc, average="macro", zero_division=0)
    print(f"  Macro F1: {macro_f1:.4f}")
    print("\n" + classification_report(y_true_enc, y_pred_enc,
                                       target_names=le.classes_, zero_division=0))

    # 8. Final combined score (average rank of macro_f1 and Recall@k)
    print("─" * 50)
    print(f"  Recall@{TOP_K}: {recall:.4f}")
    print(f"  MRR@{TOP_K}:    {mrr:.4f}")
    print(f"  Macro F1:      {macro_f1:.4f}")
    avg_score = (recall + macro_f1) / 2
    print(f"  Avg Score (Recall@k + MacroF1)/2: {avg_score:.4f}")
    
    data = {
        'Approach': ['Baseline Logistic Regression'],
        "recall_at_k": recall,
        "mrr_at_k": mrr,
        "macro_f1": macro_f1,
        "avg_score": avg_score
    }
    Results = pd.DataFrame(data)
    Results.to_csv(output_dir / "Results1.csv",index=False)

    return {
        "recall_at_k": recall,
        "mrr_at_k": mrr,
        "macro_f1": macro_f1,
        "avg_score": avg_score,
    }


if __name__ == "__main__":
    results = main()