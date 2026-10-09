Student exam score analysis (Data Science mini project)

Run order (from this folder):
  python src/make_raw_data.py     creates data/raw_students.csv (simulated)
  python src/analysis.py          cleaning, wrangling, EDA, models, figures, data/results.json
  python src/build_dashboard.py   writes dashboard.html from data/results.json
Report: compile report.tex with pdflatex (twice). Edit the name fields at the top first.
Needs: pandas, numpy, matplotlib, scikit-learn
