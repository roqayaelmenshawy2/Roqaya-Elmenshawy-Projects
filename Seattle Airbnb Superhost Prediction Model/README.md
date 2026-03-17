# 🏡 Seattle Airbnb Superhost Prediction Model

A product-oriented machine learning project that predicts whether an Airbnb host will achieve Superhost status — and translates those predictions into actionable product recommendations.

---

## Overview

Airbnb's Superhost designation is a powerful trust signal that drives bookings and host retention. This project frames the prediction problem not as a pure ML exercise, but as a **product decision-making tool**: identifying which host behaviors Airbnb can encourage through platform design to improve host quality and user experience.

The dataset contains ~1,500 Airbnb listings from Seattle, WA (collected January 2016), filtered to the four most popular neighborhoods and house/apartment property types.

---

## Key Results

| Metric | Value |
|---|---|
| Model | Logistic Regression |
| Final Features | `review_scores_rating`, `number_of_reviews`, `neighborhood` |
| Test AUC | **0.853** |
| Selected Threshold | 0.36 |
| TPR at Threshold | 0.619 |
| FPR at Threshold | 0.132 |
| Specificity | 0.868 |

---

## Project Structure

```
├── Seattle_Airbnb_Superhost_Prediction_Model.ipynb   # Main analysis notebook
├── seattle_airbnb_listings.csv                        # Dataset
└── README.md
```

---

## Methodology

### 1. Data Preparation
- Encoded the binary target variable (`host_is_superhost`) as 0/1
- Performed an 80/20 train-test split (`random_state=207`)
- Z-score scaled all numerical features using **training set statistics only** (to prevent data leakage on the test set)

### 2. Full Model
Fit a logistic regression with four features: `review_scores_rating`, `number_of_reviews`, `neighborhood`, and `property_type`. Evaluated performance using ROC/AUC on the held-out test set.

### 3. Backwards Elimination
Systematically dropped one feature at a time, keeping the change that most improved test AUC. `property_type` was dropped — it was overfitting the training data. No further single-variable drop improved AUC, so elimination stopped.

### 4. Final Model & Threshold Selection
The final model uses three features. A threshold of **0.36** was selected to satisfy a business constraint: at most 40% of true Superhosts should be misclassified as non-Superhosts (TPR ≥ 0.60), while minimizing false positives among non-Superhosts.

### 5. Interpretation
No multicollinearity was detected (correlation between numerical features < 0.7). The three strongest predictors by slope magnitude are `neighborhood[T.Broadway]`, `review_scores_rating`, and `number_of_reviews`.

---

## Product Recommendations

The model's outputs informed five product opportunities:

1. **Increase review volume** — Post-stay prompts, simplified review UX, and incentives to drive more reviews
2. **Improve review quality** — Real-time host dashboards, AI-powered listing suggestions, and declining-rating alerts
3. **Early Superhost identification** — An "Emerging Superhost" program with personalized onboarding for high-potential hosts
4. **Host engagement tools** — Response-time reminders, smart booking notifications, and gamified performance tracking
5. **Location-aware strategy** — Neighborhood-level benchmarks and dynamic pricing recommendations to help hosts optimize within their market

---

## Tech Stack

- **Python** — pandas, NumPy, scikit-learn, statsmodels, matplotlib, seaborn
- **Modeling** — Logistic Regression (statsmodels for interpretability, scikit-learn for metrics)
- **Evaluation** — ROC/AUC, confusion matrix, specificity/sensitivity analysis

---

## How to Run

1. Clone the repo and ensure `seattle_airbnb_listings.csv` is in the same directory as the notebook
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn statsmodels matplotlib seaborn
   ```
3. Open and run `Seattle_Airbnb_Superhost_Prediction_Model.ipynb` top to bottom

---

## Dataset Variables

**Listing features:** `price`, `review_scores_rating`, `number_of_reviews`, `security_deposit`, `cleaning_fee`, `neighborhood`, `property_type`, `room_type`, `accommodates`, `bathrooms`, `beds`

**Host features:** `host_is_superhost`, `host_response_rate`, `host_listings_count`, `host_since`
