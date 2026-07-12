# DS@GT ARC at CheckThat! 2026: LLM-Based Trace Ranking and Grouped Reward Modeling for Multilingual Numerical Claim Verification

Code and experiments for our CLEF 2026 CheckThat! **Task 2** submission — ranking LLM-generated reasoning traces and predicting a final verdict (True / False / Conflicting) for numerical claims in **English** and **Arabic**.

**Authors:** Sagnik Sinha, Shreyas Shrestha — Georgia Institute of Technology
**Paper:** `docs/paper_333.pdf` (CLEF 2026 Working Notes)

## Task

Given a claim, its evidence, and multiple LLM-generated reasoning traces (each with a predicted verdict), a verifier model must:
1. Rank the reasoning traces by how useful they are for reaching the correct verdict.
2. Derive a final claim-level verdict from the top-ranked traces.

Evaluated with **Recall@5** / **MRR@5** (trace ranking) and **Macro-F1** (verdict prediction); final score = average of Macro-F1 and Recall@5.

## Our Approach

We built two complementary systems, both supervised at the **trace level**: each reasoning trace gets a binary "useful" label (1 if its verdict matches the gold claim label, 0 otherwise).

### Approach I — LLM-Based Verifier (LoRA)
Fine-tunes an LLM as a binary trace classifier and selects the final verdict via Best-of-N.
- **English:** Qwen2.5-Math-7B backbone (best of Qwen2.5-Math-7B, Llama-3.2-1B, ModernBERT-large)
- **Arabic:** AraBERT vs. multilingual BERT
- LoRA fine-tuning (r=8, α=16) on attention/FFN projections
- Also tested **sub-claim decomposition** (splitting claims into ≤2 sub-claims via Llama-3.2-3B-Instruct before verification) — this *hurt* performance, suggesting decomposition adds noise rather than signal.

### Approach II — Grouped Reward-Based Trace Ranking (TF-IDF)
A lightweight, interpretable alternative to a fine-tuned neural re-ranker:
- Word (1-2gram) + character (3-5gram) TF-IDF features, plus handcrafted numeric/temporal overlap features (shared numbers, years, months, percentages, currency)
- SGD classifier (logistic loss, L2) scores each trace's utility
- Trace scores are grouped by verdict label (True/False/Conflicting), aggregated with a top-k + count-bonus rule, and the verdict group with the highest score wins

## Results (Test Set)

**English**

| Approach | Score | Macro-F1 | Recall@5 | True F1 | Conflicting F1 | False F1 |
|---|---|---|---|---|---|---|
| Approach I | 41.88 | 58.98 | 24.77 | 72.42 | 12.74 | 88.77 |
| Approach I + Sub-Claims | 39.68 | 55.79 | 23.57 | 64.68 | 13.33 | 89.34 |
| Approach II | 40.50 | 58.78 | 22.21 | 70.44 | **18.02** | 87.88 |

**Arabic**

| Approach | Score | Macro-F1 | Recall@5 | True F1 | False F1 |
|---|---|---|---|---|---|
| AraBERT | 59.13 | 84.85 | 33.41 | 82.19 | 87.52 |
| BERT (multilingual) | 55.93 | 82.32 | 29.54 | 83.25 | 83.38 |

**Key findings:**
- The LLM-based verifier (Approach I) wins overall, especially on Recall@5.
- The TF-IDF reward model (Approach II) is notably stronger on the harder **Conflicting** class — evidence for it is often spread across multiple traces, which grouped aggregation captures well.
- Sub-claim decomposition did not help; it introduced noise.
- AraBERT clearly beats general multilingual BERT for Arabic.

## Repository Structure

```
.
├── README.md
├── docs/
│   └── paper.pdf                     # CLEF 2026 Working Notes paper
├── data_prep/
│   └── build_reasoning_traces.py         # builds trace-level training data for the LLM verifier (Approach I)
├── notebooks/
│   ├── approach1_english_qwen.ipynb      # Approach I, English (Qwen2.5-Math-7B, LoRA)
│   ├── approach1_arabic_arabert.ipynb    # Approach I, Arabic (AraBERT, LoRA)
│   └── approach2_tfidf_reward_model.ipynb # Approach II, TF-IDF grouped reward model
```

## Setup

```bash
git clone https://github.com/dsgt-arc/clef2026-checkthat-task2.git
cd clef2026-checkthat-task2
pip install -r requirements.txt
```

Set your Hugging Face token as an environment variable rather than hardcoding it:
```bash
export HF_TOKEN="your_token_here"
```

## Citation

If you use this work, please cite our CLEF 2026 Working Notes paper (see `docs/paper_333.pdf`).

## Acknowledgments

Data Science at Georgia Tech (DS@GT) CLEF competition group; PACE research computing resources at Georgia Tech.
