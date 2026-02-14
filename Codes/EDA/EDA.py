import json
from collections import Counter

def load_json(path):
    # with open(path, "r") as f:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_reasoning_distribution(data):
    return Counter([len(c.get("Reasoning_traces", [])) for c in data])

def compute_verdict_distribution(data):
    all_verdicts = []
    for c in data:
        all_verdicts.extend(c.get("Verdict_list", []))
    return Counter(all_verdicts)

data=load_json(r"C:\Users\sagni\Documents\Personal Files\DS@GT\clef2026-checkthat-task2\Data\English\clef_2026_checkthat_english_train.json")

print(type(data))
print(type(data[0]))
print(len(data))
print(compute_reasoning_distribution(data))
print(compute_verdict_distribution(data))
