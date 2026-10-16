"""Baselines on the v1 synthetic data (README "Measured performance"). Run from repo root: python experiments/baseline_check.py"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from core.anomaly_detection import generate_synthetic_data
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
X, y = generate_synthetic_data()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
def rep(name, p):
    print(f"{name:38s} acc={accuracy_score(yte,p):.3f} P={precision_score(yte,p):.3f} R={recall_score(yte,p):.3f} F1={f1_score(yte,p):.3f}")
for c in [0.02, 0.05, 0.1, 0.2, 0.3]:
    m = IsolationForest(n_estimators=200, contamination=c, random_state=42).fit(Xtr_s)
    rep(f"IsolationForest contamination={c}", (m.predict(Xte_s) == -1).astype(int))
rep("LogisticRegression", LogisticRegression(max_iter=1000).fit(Xtr_s, ytr).predict(Xte_s))
rep("RandomForest", RandomForestClassifier(random_state=0).fit(Xtr_s, ytr).predict(Xte_s))
rep("Depth-1 stump (all features)", DecisionTreeClassifier(max_depth=1).fit(Xtr, ytr).predict(Xte))
rep("Rule: days_between_bookings < 20", (Xte[:,0] < 20).astype(int))
