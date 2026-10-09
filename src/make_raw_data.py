"""
Builds data/raw_students.csv, a survey style file with the kind of mistakes
that show up when data is typed in by hand (mixed spellings, units written
into cells, missing answers, repeated rows, impossible values).

The values are simulated, so the file is a practice dataset and not a
record of real students. Run once; the seed makes it reproducible.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 520

gender = rng.choice(["Male", "Female"], n)
age = rng.choice([15, 16, 17, 18], n, p=[0.15, 0.35, 0.35, 0.15])
area = rng.choice(["Urban", "Rural"], n, p=[0.6, 0.4])
parent = rng.choice(["School", "Graduate", "Postgraduate"], n, p=[0.40, 0.42, 0.18])
internet = np.where(area == "Urban", rng.random(n) < 0.88, rng.random(n) < 0.62)
tuition = rng.random(n) < 0.35
study = np.clip(rng.normal(11, 4.5, n), 0.5, 40).round(1)
attend = np.clip(rng.normal(82, 11, n), 40, 100).round(0)
sleep = np.clip(rng.normal(7.0, 1.0, n), 4, 10).round(1)
prev = np.clip(rng.normal(62, 14, n) + 1.5 * (parent != "School"), 15, 98).round(0)

final = (
    14
    + 0.48 * prev
    + 1.05 * np.minimum(study, 25)
    + 0.16 * attend
    + 3.0 * tuition
    + 1.8 * internet
    + np.select([parent == "Graduate", parent == "Postgraduate"], [1.0, 2.0], 0)
    - 1.1 * (sleep - 7.5) ** 2
    + rng.normal(0, 5, n)
)
final = np.clip(final, 5, 100).round(0)

df = pd.DataFrame({
    "student_id": [f"S{i:04d}" for i in range(1, n + 1)],
    "gender": gender, "age": age.astype(object), "area": area,
    "parent_education": parent,
    "study_hours_week": study.astype(object),
    "attendance": attend.astype(object),
    "sleep_hours": sleep.astype(object),
    "internet_access": np.where(internet, "Yes", "No").astype(object),
    "private_tuition": np.where(tuition, "Yes", "No").astype(object),
    "previous_score": prev.astype(object),
    "final_score": final.astype(object),
})

def pick(p):
    return rng.random(n) < p

g_map = {"Male": ["Male", "male", "M", "MALE ", "m"], "Female": ["Female", "female", "F", "FEMALE", " f"]}
df["gender"] = [rng.choice(g_map[g]) for g in df["gender"]]
p_map = {"School": ["School", "school", "Schooling ", "SCHOOL"],
         "Graduate": ["Graduate", "graduate", "Graduate ", "UG"],
         "Postgraduate": ["Postgraduate", "post graduate", "PG", "postgraduate"]}
df["parent_education"] = [rng.choice(p_map[x]) for x in df["parent_education"]]
for col in ["internet_access", "private_tuition"]:
    yes = ["Yes", "yes", "Y", "1"]; no = ["No", "no", "N", "0"]
    df[col] = [rng.choice(yes if v == "Yes" else no) for v in df[col]]
df["area"] = [a if r > 0.1 else a.lower() for a, r in zip(df["area"], rng.random(n))]

m = pick(0.25); df.loc[m, "attendance"] = df.loc[m, "attendance"].map(lambda v: f"{int(v)}%")
m = pick(0.10); df.loc[m, "age"] = df.loc[m, "age"].map(lambda v: f"{v} yrs")
m = pick(0.06); df.loc[m, "study_hours_week"] = df.loc[m, "study_hours_week"].map(lambda v: f"{v} hrs")

for col, p in [("study_hours_week", .06), ("attendance", .05), ("sleep_hours", .05),
               ("parent_education", .04), ("age", .03), ("previous_score", .04),
               ("internet_access", .02), ("gender", .02)]:
    df.loc[pick(p), col] = np.nan
df.loc[pick(0.015), "attendance"] = "N/A"
df.loc[rng.choice(n, 6, replace=False), "final_score"] = np.nan

idx = rng.choice(n, 14, replace=False)
df.loc[idx[:4], "study_hours_week"] = [95, 120, -4, 168]
df.loc[idx[4:7], "sleep_hours"] = [24, 0, 18]
df.loc[idx[7:10], "attendance"] = [150, 940, -20]
df.loc[idx[10:12], "previous_score"] = [785, 880]
df.loc[idx[12:14], "final_score"] = [250, -10]

df = pd.concat([df, df.sample(14, random_state=1)], ignore_index=True)
df = df.sample(frac=1, random_state=7).reset_index(drop=True)
df.to_csv("data/raw_students.csv", index=False)
print(df.shape)
