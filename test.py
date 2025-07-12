import pandas as pd
from catboost import CatBoostClassifier

df = pd.read_csv("balanced_output.csv").sample(100, random_state=42)
X = df.drop(columns=["Label"])
y = df["Label"]

model = CatBoostClassifier(task_type='GPU', devices='0', iterations=20)
model.fit(X, y)
