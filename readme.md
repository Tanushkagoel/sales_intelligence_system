# SkyGeni Sales Intelligence – Solution

A data-driven insight system to help a CRO understand why win rate has dropped and what actions to take. Built for the SkyGeni Sales Intelligence challenge.

---

## Approach

- **Problem framing**: The real issue is *diagnostic blindness*—knowing *where* and *why* win rate is breaking down so the team can act. The solution frames key questions, metrics, and assumptions .
- **EDA & insights**: Python EDA on `skygeni_sales_data.csv` surfaces three business insights (win rate by region/quarter, lead source & product impact, sales cycle vs outcome) and two custom metrics: **Pipeline Quality Score (PQS)** and **Cycle Win Efficiency (CWE)**. See [eda_insights.py](eda_insights.py).
- **Decision engine**: **Win Rate Driver Analysis** (Option B). A simple logistic model plus segment lift tables identify which factors hurt or help win rate and produce an actionable report and CSVs. See [decision_engine.py](decision_engine.py).
- **Reflection**: Assumptions, production risks, next steps, and least-confident parts are in .

---

## How to run

### 1. Environment

```bash
cd cache/sales_intelligence   # or your repo path
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Use `python3` instead of `python` if your system does not alias `python` to Python 3.

### 2. Data

Place `skygeni_sales_data.csv` in this directory (columns: `deal_id`, `created_date`, `closed_date`, `sales_rep_id`, `industry`, `region`, `product_type`, `lead_source`, `deal_stage`, `deal_amount`, `sales_cycle_days`, `outcome`).

### 3. EDA and insights (Part 2)

```bash
python eda_insights.py
```

- Prints EDA summary, three business insights, and two custom metrics (PQS, CWE).
- If matplotlib/seaborn are installed, saves plots under `outputs/`.

### 4. Decision engine – Win Rate Driver Analysis (Part 3)

```bash
python decision_engine.py
```

- Trains a simple logistic model and builds segment lift tables.
- Prints an actionable summary and writes:
    - `outputs/win_rate_drivers_coefficients.csv`
    - `outputs/win_rate_segment_lift.csv`
    - `outputs/win_rate_driver_report.txt`

---

## Key decisions

| Area | Choice | Reason |
|------|--------|--------|
| Decision engine option | **B – Win Rate Driver Analysis** | Directly addresses “what is going wrong and where to focus.” |
| Custom metrics | **PQS**, **CWE** | PQS measures pipeline “quality” by historical segment win rate; CWE combines win rate and cycle length for efficiency. |
| Model | Logistic regression | Interpretable coefficients and segment lift; good enough for “which levers matter” without overfitting. |
| Outputs | Report + CSVs | Sales leaders can use the report in meetings and CSVs in BI or further analysis. |

---

## Repository layout

```
cache/sales_intelligence/
├── README.md                 # This file   
├── skygeni_sales_data.csv   # Input data
├── requirements.txt      # Part 1
├── eda_insights.py          # Part 2 – EDA, insights, PQS, CWE
├── decision_engine.py       # Part 3 – Win rate driver analysis
├── SYSTEM_DESIGN.md         # Part 4
└── outputs/                 # Created on run: CSVs, report, optional plots
```

---

## Evaluation alignment

- **Problem framing & business thinking**: PROBLEM_FRAMING.md + reflection.
- **Insight quality & metric design**: EDA script (3 insights, 2 custom metrics) and business-language explanations.
- **Decision engine**: Win rate driver model, segment lift, actionable report and usage notes.
- **Engineering & code quality**: Modular scripts, requirements, clear run instructions.
- **Communication & clarity**: README, structured docs, and plain-language “why it matters / what action” in insights and report.
