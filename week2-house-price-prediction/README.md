# House Price Prediction — Multiple Linear Regression from Scratch

Week 2 portfolio project, built alongside Andrew Ng's Machine Learning Specialization (Course 1). Implements multiple linear regression entirely from scratch using NumPy — no scikit-learn model fitting — to demonstrate understanding of the underlying math, not just library usage.

## Dataset

Synthetically generated housing data (200 examples) with a known ground-truth relationship, so model correctness can be verified directly by comparing learned weights against the true weights used to generate the data — rather than relying on loss alone.

Features: `size_sqft`, `bedrooms`, `age_years`, `distance_to_city_km`.

## Concepts demonstrated

- **Multiple linear regression** with 4 input features
- **Vectorization** — cost and gradient computed via matrix operations (`X @ w`, `X.T @ errors`), no Python loops over examples
- **Feature scaling** — z-score normalization implemented from scratch, with before/after comparison showing why it's necessary (one feature ranged ~3400, another ranged 4)
- **Gradient descent convergence** — verified via cost-vs-iteration plots
- **Learning rate selection** — compared 5 learning rates (0.001 to 3.0) side by side; observed underfitting-by-impatience at low alpha and divergence at alpha=3.0
- **Feature engineering** — tested a derived feature (`size_sqft / bedrooms`); found it improved cost by only ~0.23%, correctly reflecting that it wasn't part of the true underlying relationship
- **Polynomial regression** — tested `age_years²`; found a similarly marginal ~0.44% improvement for the same reason

## Results

- Learned weights recovered the true generating weights closely (e.g., learned `size_sqft` weight of 149.6 vs. true 150)
- Final Mean Absolute Error: **$11,824.64** — consistent with the $15,000 standard deviation noise injected into the synthetic data, indicating the model is fitting close to the irreducible noise floor rather than underfitting

## Key learning

Feature engineering and polynomial terms only help when they capture real signal in the data. Both experiments here produced small, consistent improvements (~0.2-0.4%) precisely because neither derived feature was part of the true relationship used to generate the data — a deliberate test of the model's ability to *not* be fooled by irrelevant features.

## Stack

Python, NumPy, Matplotlib, Jupyter

## Setup

\`\`\`bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
\`\`\`
