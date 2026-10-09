"""
Student score analysis: cleaning, wrangling, exploration and modelling.
Run from the project folder:  python src/analysis.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25})
BLUE, GREY = "#2f5d8a", "#8a8a8a"
log = []          # every cleaning step is recorded here


def note(step, found, action):
    log.append({"step": step, "found": found, "action": action})
    print(f"[{step}] {found} -> {action}")


# 1. Load and inspect --------------------------------------------------------
raw = pd.read_csv("data/raw_students.csv")
n_raw = len(raw)
missing_raw = raw.isna().sum().sort_values(ascending=False)
print(raw.dtypes, "\n", missing_raw)

df = raw.copy()

# 2. Text columns: strip, lower-case, map variants to one label --------------
for c in ["gender", "area", "parent_education", "internet_access", "private_tuition"]:
    df[c] = df[c].astype("string").str.strip()

gmap = {"male": "Male", "m": "Male", "female": "Female", "f": "Female"}
pmap = {"school": "School", "schooling": "School", "graduate": "Graduate", "ug": "Graduate",
        "postgraduate": "Postgraduate", "post graduate": "Postgraduate", "pg": "Postgraduate"}
ymap = {"yes": "Yes", "y": "Yes", "1": "Yes", "no": "No", "n": "No", "0": "No"}

before = {c: df[c].nunique() for c in ["gender", "parent_education", "internet_access",
                                        "private_tuition", "area"]}
df["gender"] = df["gender"].str.lower().map(gmap)
df["parent_education"] = df["parent_education"].str.lower().map(pmap)
df["area"] = df["area"].str.title()
for c in ["internet_access", "private_tuition"]:
    df[c] = df[c].str.lower().map(ymap)
after = {c: df[c].nunique() for c in before}
show = lambda d: ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in d.items())
note("Category labels", f"labels in use per column: {show(before)}",
     f"mapped to consistent labels: {show(after)}")

# 3. Numbers stored as text --------------------------------------------------
for c in ["age", "study_hours_week", "attendance", "sleep_hours",
          "previous_score", "final_score"]:
    cleaned_txt = (df[c].astype("string").str.replace(r"[^0-9.\-]", "", regex=True)
                   .replace("", pd.NA))
    df[c] = pd.to_numeric(cleaned_txt, errors="coerce").astype(float)
txt_cells = int((raw[["age", "study_hours_week", "attendance"]].astype("string")
                 .apply(lambda s: s.str.contains(r"%|hrs|yrs", na=False)).sum()).sum())
note("Units inside cells", f"{txt_cells} cells such as '78%', '14.3 hrs', '16 yrs'",
     "removed the unit text and converted the columns to numbers")

# 4. Duplicates --------------------------------------------------------------
dups = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
note("Duplicate rows", f"{dups} fully repeated rows", "kept the first copy of each")
assert df["student_id"].is_unique

# 5. Target column ----------------------------------------------------------
bad_target = df["final_score"].isna() | ~df["final_score"].between(0, 100)
note("Final score", f"{int(bad_target.sum())} rows missing or outside 0 to 100",
     "dropped these rows (the target should not be guessed)")
df = df[~bad_target].reset_index(drop=True)

# 6. Impossible values become missing ---------------------------------------
rules = {"study_hours_week": (0, 60), "attendance": (0, 100),
         "sleep_hours": (3, 12), "previous_score": (0, 100), "age": (10, 25)}
for c, (lo, hi) in rules.items():
    bad = df[c].notna() & ~df[c].between(lo, hi)
    if bad.sum() > 0:
        note(f"Range check: {c}", f"{int(bad.sum())} values outside {lo} to {hi}",
             "set to missing, filled in the next step")
    df.loc[bad, c] = np.nan

# 7. Missing values ----------------------------------------------------------
miss_now = df.isna().sum()
miss_now = miss_now[miss_now > 0]
num_cols = ["age", "study_hours_week", "attendance", "sleep_hours", "previous_score"]
cat_cols = ["gender", "parent_education", "internet_access"]
medians = {c: float(df[c].median()) for c in num_cols}
medians["age"] = float(round(medians["age"]))
modes = {c: df[c].mode()[0] for c in cat_cols}
for c in num_cols:
    df[c] = df[c].fillna(medians[c])
for c in cat_cols:
    df[c] = df[c].fillna(modes[c])
note("Missing values", f"{int(miss_now.sum())} empty cells in {len(miss_now)} columns",
     "numeric columns filled with the median, text columns with the most common label")
assert df.isna().sum().sum() == 0
n_clean = len(df)

# 8. Outlier report with the IQR rule (after the range checks) ---------------
iqr_rows = []
for c in ["study_hours_week", "attendance", "sleep_hours", "previous_score", "final_score"]:
    q1, q3 = df[c].quantile([.25, .75]); i = q3 - q1
    k = int(((df[c] < q1 - 1.5 * i) | (df[c] > q3 + 1.5 * i)).sum())
    iqr_rows.append({"column": c, "flagged": k})
note("IQR outlier check", ", ".join(f"{r['column']}: {r['flagged']}" for r in iqr_rows),
     "kept; these are plausible values, not typing errors")

# 9. Wrangling: new columns --------------------------------------------------
df["study_band"] = pd.cut(df["study_hours_week"], [-0.1, 7, 14, 100],
                          labels=["Low (up to 7 h)", "Medium (7 to 14 h)", "High (over 14 h)"])
df["attendance_band"] = pd.cut(df["attendance"], [-0.1, 70, 85, 100],
                               labels=["Below 70%", "70 to 85%", "Above 85%"])
df["gain"] = df["final_score"] - df["previous_score"]
df["grade"] = pd.cut(df["final_score"], [-1, 39, 59, 74, 100],
                     labels=["Fail", "Pass", "Merit", "Distinction"])
df["parent_education"] = pd.Categorical(df["parent_education"],
                                        ["School", "Graduate", "Postgraduate"], ordered=True)
df.to_csv("data/clean_students.csv", index=False)

by_study = df.groupby("study_band", observed=True)["final_score"].agg(["count", "mean"]).round(1)
by_att = df.groupby("attendance_band", observed=True)["final_score"].agg(["count", "mean"]).round(1)
by_par = df.groupby("parent_education", observed=True)["final_score"].agg(["count", "mean"]).round(1)
by_tui = df.groupby("private_tuition")["final_score"].agg(["count", "mean"]).round(1)
by_net = df.groupby("internet_access")["final_score"].agg(["count", "mean"]).round(1)
by_gen = df.groupby("gender")["final_score"].agg(["count", "mean"]).round(1)
by_area = df.groupby("area")["final_score"].agg(["count", "mean"]).round(1)
pivot = df.pivot_table(index="study_band", columns="private_tuition", values="final_score",
                       aggfunc="mean", observed=True).round(1)
grade_counts = df["grade"].value_counts().reindex(["Fail", "Pass", "Merit", "Distinction"])
sleep_band = pd.cut(df["sleep_hours"], [0, 5.5, 6.5, 7.5, 8.5, 12])
by_sleep = df.groupby(sleep_band, observed=True)["final_score"].mean().round(1)
print(by_study, by_att, by_par, by_tui, by_net, by_gen, by_area, pivot, by_sleep, sep="\n\n")

# 10. Exploration ------------------------------------------------------------
num_all = ["age", "study_hours_week", "attendance", "sleep_hours", "previous_score", "final_score"]
corr = df[num_all].corr().round(2)
print(corr["final_score"])
desc = df[num_all].describe().round(2)

# model matrix
X = pd.DataFrame({
    "study_hours_week": df["study_hours_week"], "attendance": df["attendance"],
    "sleep_gap": (df["sleep_hours"] - 7.5).abs(), "previous_score": df["previous_score"],
    "private_tuition": (df["private_tuition"] == "Yes").astype(int),
    "internet_access": (df["internet_access"] == "Yes").astype(int),
    "parent_graduate": (df["parent_education"] == "Graduate").astype(int),
    "parent_postgraduate": (df["parent_education"] == "Postgraduate").astype(int),
})
y = df["final_score"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)


def score(model, name):
    model.fit(Xtr, ytr)
    p = model.predict(Xte)
    return {"model": name, "mae": round(mean_absolute_error(yte, p), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(yte, p))), 2),
            "r2": round(r2_score(yte, p), 3)}, p


base = {"model": "Mean baseline", "mae": round(mean_absolute_error(yte, np.full(len(yte), ytr.mean())), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(yte, np.full(len(yte), ytr.mean())))), 2), "r2": 0.0}
lr = LinearRegression(); rf = RandomForestRegressor(300, min_samples_leaf=5, random_state=42)
m_lr, p_lr = score(lr, "Linear regression")
m_rf, p_rf = score(rf, "Random forest")
cv = cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2")
print(base, m_lr, m_rf, cv)

coef = dict(zip(X.columns, lr.coef_.round(4)))
imp = dict(zip(X.columns, rf.feature_importances_.round(3)))
# standardised effect so features with different units can be compared
std_effect = {c: round(float(lr.coef_[i] * X[c].std()), 2) for i, c in enumerate(X.columns)}

# 11. Figures ----------------------------------------------------------------
def save(name):
    plt.tight_layout(); plt.savefig(f"figures/{name}.png"); plt.close()

mr = missing_raw[missing_raw > 0].sort_values()
plt.figure(figsize=(5.2, 3)); plt.barh(mr.index, mr.values, color=BLUE)
plt.xlabel("Empty cells in the raw file"); save("missing_raw")

fig, ax = plt.subplots(2, 2, figsize=(6.4, 4.4))
for a, c, t in zip(ax.ravel(), ["study_hours_week", "attendance", "sleep_hours", "final_score"],
                   ["Study hours per week", "Attendance (%)", "Sleep hours", "Final score"]):
    a.hist(df[c], bins=22, color=BLUE, edgecolor="white"); a.set_title(t, fontsize=9)
save("histograms")

plt.figure(figsize=(5.4, 4.4))
im = plt.imshow(corr, cmap="Blues", vmin=-0.2, vmax=1); plt.grid(False)
labs = ["age", "study", "attend", "sleep", "prev", "final"]
plt.xticks(range(6), labs); plt.yticks(range(6), labs)
for i in range(6):
    for j in range(6):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                 color="white" if corr.iloc[i, j] > 0.55 else "black")
plt.colorbar(im, fraction=0.046); save("correlation")

plt.figure(figsize=(5.2, 3.6))
plt.scatter(df["study_hours_week"], df["final_score"], s=9, alpha=.45, color=BLUE)
z = np.polyfit(df["study_hours_week"], df["final_score"], 1)
xs = np.linspace(0, df["study_hours_week"].max(), 50)
plt.plot(xs, np.polyval(z, xs), color="black", lw=1.2)
plt.xlabel("Study hours per week"); plt.ylabel("Final score"); save("study_vs_score")

fig, ax = plt.subplots(1, 3, figsize=(7, 3), sharey=True)
for a, s, t in zip(ax, [by_study, by_att, by_par], ["Study hours", "Attendance", "Parent education"]):
    a.bar(range(len(s)), s["mean"], color=BLUE)
    a.set_xticks(range(len(s))); a.set_xticklabels([str(i).split(" (")[0] for i in s.index], fontsize=7, rotation=15)
    a.set_title(t, fontsize=9); a.set_ylim(0, 100)
    for k, v in enumerate(s["mean"]):
        a.text(k, v + 1.5, f"{v:.1f}", ha="center", fontsize=8)
ax[0].set_ylabel("Mean final score"); save("group_means")

plt.figure(figsize=(4.4, 4.2))
plt.scatter(yte, p_lr, s=12, alpha=.55, color=BLUE)
plt.plot([20, 100], [20, 100], color="black", lw=1)
plt.xlabel("Actual score"); plt.ylabel("Predicted score"); save("actual_vs_pred")

e = pd.Series(std_effect).sort_values()
plt.figure(figsize=(5.4, 3.2)); plt.barh(e.index, e.values, color=BLUE)
plt.xlabel("Change in score for a one standard deviation rise"); save("effects")

# 12. Results for the dashboard and the report -------------------------------
def tab(s):
    return [{"label": str(i), "n": int(r["count"]), "mean": float(r["mean"])} for i, r in s.iterrows()]

out = {
    "rows_raw": n_raw, "rows_clean": n_clean, "cols_raw": raw.shape[1],
    "log": log, "missing_raw": {k: int(v) for k, v in missing_raw.items() if v > 0},
    "desc": desc.to_dict(), "iqr": iqr_rows,
    "corr_final": corr["final_score"].drop("final_score").to_dict(),
    "groups": {"study": tab(by_study), "attendance": tab(by_att), "parent": tab(by_par),
               "tuition": tab(by_tui), "internet": tab(by_net), "gender": tab(by_gen), "area": tab(by_area)},
    "pivot": {str(i): {str(k): float(v) for k, v in r.items()} for i, r in pivot.iterrows()},
    "sleep": {str(k): float(v) for k, v in by_sleep.items()},
    "grades": {str(k): int(v) for k, v in grade_counts.items()},
    "models": [base, m_lr, m_rf], "cv_r2": [round(float(v), 3) for v in cv],
    "cv_mean": round(float(cv.mean()), 3),
    "coef": {k: float(v) for k, v in coef.items()}, "intercept": float(lr.intercept_),
    "std_effect": std_effect, "rf_importance": imp,
    "ranges": {c: [float(df[c].min()), float(df[c].max()), float(df[c].median())]
               for c in ["study_hours_week", "attendance", "sleep_hours", "previous_score"]},
    "points": [[round(a, 1), int(b)] for a, b in zip(df["study_hours_week"], df["final_score"])],
    "train_n": len(Xtr), "test_n": len(Xte),
}
json.dump(out, open("data/results.json", "w"), indent=1)
print("done", n_raw, "->", n_clean)
