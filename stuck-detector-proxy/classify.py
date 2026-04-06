#!/usr/bin/env python3
"""
Stuck classifier — reads text from stdin, outputs score to stdout.
Called by the Node.js proxy as a subprocess.

Usage: echo "thinking text..." | python3 classify.py
Output: {"score": 0.87, "label": "stuck", "reasons": ["self_sim", "circle_kw"]}
"""

import sys
import json
import re
import pickle
import os
import numpy as np

MODEL_PATH = os.environ.get(
    "STUCK_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "classifier-repos", "dataset", "stuck_classifier.pkl"),
)

# Also check relative to this script
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "/home/nicolas/source/classifier-repos/dataset/stuck_classifier.pkl"


def extract_features(text):
    feats = {}

    # Substring repetition
    seen = {}
    max_repeat = 0
    for i in range(0, len(text) - 20, 10):
        sub = text[i : i + 20]
        seen[sub] = seen.get(sub, 0) + 1
        max_repeat = max(max_repeat, seen[sub])
    feats["max_substr_repeat"] = max_repeat

    # Vocabulary diversity
    words = text.lower().split()
    feats["vocab_diversity"] = len(set(words)) / max(len(words), 1)

    # Circle keywords
    circle = re.findall(
        r"\b(try again|let me try|another approach|actually,|wait,|hmm|"
        r"let me reconsider|that didn.t work|same error|still failing|"
        r"let me re-read|let me look again|I was wrong|no that.s not right)\b",
        text,
        re.IGNORECASE,
    )
    feats["circle_kw"] = len(circle)

    # False starts
    fs = re.findall(r"(?:^|\n)\s*(?:Actually|Wait|Hmm|No,|Let me)", text)
    feats["false_starts"] = len(fs)

    # Self-similarity
    if len(text) > 200:
        half = len(text) // 2
        w1 = set(text[:half].lower().split())
        w2 = text[half:].lower().split()
        feats["self_sim"] = sum(1 for w in w2 if w in w1) / max(len(w2), 1)
    else:
        feats["self_sim"] = 0

    # Sentence stats
    sents = re.split(r"[.!?\n]+", text)
    slens = [len(s.split()) for s in sents if s.strip()]
    feats["avg_sent_len"] = float(np.mean(slens)) if slens else 0
    feats["sent_len_std"] = float(np.std(slens)) if slens else 0
    feats["question_marks"] = text.count("?")

    # Code ratio
    code_chars = len(re.findall(r"[{}\[\]();=<>]", text))
    feats["code_ratio"] = code_chars / max(len(text), 1)

    return feats


def main():
    text = sys.stdin.read()
    if len(text) < 100:
        print(json.dumps({"score": 0.0, "label": "productive", "reasons": []}))
        return

    try:
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
    except Exception as e:
        print(json.dumps({"score": 0.0, "label": "error", "reasons": [str(e)]}))
        return

    scaler = model_data["scaler"]
    clf = model_data["classifier"]
    feature_names = model_data["feature_names"]

    feats = extract_features(text)
    X = np.array([[feats[k] for k in feature_names]])
    X_scaled = scaler.transform(X)

    score = float(clf.predict_proba(X_scaled)[0, 1])
    label = "stuck" if score >= 0.5 else "productive"

    # Top contributing features
    reasons = []
    for name, coef in zip(feature_names, clf.coef_[0]):
        if coef > 0.5 and feats[name] > 0:
            reasons.append(name)

    print(json.dumps({"score": round(score, 3), "label": label, "reasons": reasons}))


if __name__ == "__main__":
    main()
