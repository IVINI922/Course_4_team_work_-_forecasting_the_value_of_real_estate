import joblib
import numpy as np
import pandas as pd

model = joblib.load("artifacts/final_model_gb.joblib")
print("Число деревьев:", model.n_estimators)
print("Число признаков:", model.n_features_in_)

# Смотрим, на каких порогах реально сплитует дерево №0
tree = model.estimators_[0, 0].tree_
print("\nДерево №0, пороги по признакам:")
for node_id in range(tree.node_count):
    feat = tree.feature[node_id]
    thresh = tree.threshold[node_id]
    if feat >= 0:   # -2 = лист
        print(f"  node {node_id}: feature={feat}, threshold={thresh:.4f}")
input()