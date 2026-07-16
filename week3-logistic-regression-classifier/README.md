# Logistic Regression Classifier from Scratch

A binary classifier built entirely from scratch using NumPy — no scikit-learn —
demonstrating core supervised machine learning concepts from Andrew Ng's
Machine Learning Specialization (Course 1, Week 3).

## What This Project Demonstrates

- **Model representation**: the sigmoid function and logistic regression's
  hypothesis function
- **Decision boundaries**: both linear and nonlinear (via polynomial features)
- **Logistic loss (cost function)**: derivation and implementation from scratch
- **Gradient descent**: manual implementation of the update rules, including
  simultaneous updates
- **The problem of overfitting**: demonstrated by training high-degree
  polynomial features on noisy data, producing a boundary that chases
  individual noisy points
- **L2 regularization**: implemented from scratch, shown to progressively
  smooth an overfit decision boundary — and shown to cause underfitting when
  taken too far
- **Learning rate behavior**: comparing convergence across a range of learning
  rates, with a discussion of why "too large" is data-dependent, not universal

## Project Structure

All work is in a single notebook: `week3_logistic_regression.ipynb`

1. Synthetic dataset generation (2 features, binary labels)
2. Sigmoid function and model representation
3. Decision boundary visualization
4. Cost function (logistic loss) implementation
5. Gradient descent implementation and training
6. Trained boundary + accuracy evaluation
7. Polynomial feature engineering
8. Overfitting demonstration on noisy data
9. L2-regularized cost function and gradient descent
10. Regularization strength (lambda) comparison across multiple values
11. Learning rate experimentation

## Key Results

- Baseline logistic regression on clean, separable data achieved 100% training accuracy
- Adding polynomial features (degree 6) and mislabeled/noisy points exposed
  clear overfitting behavior in the decision boundary
- Increasing lambda from 0 to 1000 traced the full bias-variance tradeoff: from
  overfitting, to a reasonable middle ground, to underfitting (collapsing to
  ~50% accuracy at lambda=1000)

## Tech Stack

- Python 3
- NumPy (all model math implemented manually — no ML libraries)
- Matplotlib (visualization)
- Jupyter Notebook

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
jupyter notebook week3_logistic_regression.ipynb
```

## Scope Note

This project is intentionally scoped to Week 3 concepts only. It does not use
train/validation/test splits, scikit-learn, or any techniques from later
weeks/courses — those will appear in future portfolio projects as the
curriculum progresses.

## Part of a Series

This is part of an ongoing series of ML portfolio projects built in lockstep
with Andrew Ng's Machine Learning Specialization.
