# SiC-AZ91-Alloy-Composite-UTS-Predictor
# Machine Learning Prediction of the Ultimate Tensile Strength of SiC Micro-Particle Reinforced AZ91 Composites

A machine learning project that predicts the **ultimate tensile strength (UTS)** of SiC micro-particle reinforced **AZ91 magnesium composites** for application in light-vehicle and aerospace industries. The trained model is served through an interactive Streamlit web app.



## Background

- **AZ91** is a lightweight magnesium alloy containing about 9% aluminium and 1% zinc, valued where low weight matters.
- **SiC (silicon carbide)** is a hard ceramic added to the alloy as micro-sized particles to reinforce it and raise its strength.
- **UTS** is the maximum stress a material can withstand while being stretched before it breaks.

## Dataset

83 experimental samples of SiC-reinforced AZ91 composites. The model uses four inputs:

| Input | Description |
|---|---|
| `SiC_vol_perc` | SiC content (volume %) |
| `SiC_size_um` | SiC particle size (µm) |
| `av_grain_um` | Average grain size of the matrix (µm) |
| Processing group | Stir casting, Deformed, Other casting, or Homogenized |

The target is `UTS_MPa`.

The original data contains 13 processing routes, several with only one to three samples. For a more stable model they were merged into four groups and one-hot encoded:

| Group | Routes included |
|---|---|
| Stir casting | Stir casting |
| Deformed | Hot extrusion, ECAP, ECAP+rolled, rolled |
| Other casting | Semi-solid stirring, centrifugal casting, simulated, CCDFed |
| Homogenized | Homogenized T6, T7, T8, T9 |

## Method

1. Exploratory data analysis and cleaning
2. 80/20 train/test split (66 training rows, 17 test rows, `random_state=42`)
3. Three models trained and compared: Linear Regression, Random Forest, XGBoost
4. Evaluation with R², RMSE and MAE on the test set, plus 5-fold cross-validation on the training set
5. Interpretation with feature importance and partial dependence plots
6. Deployment of the selected model as a Streamlit app

## Results

| Model | Test R² | Test RMSE (MPa) | Test MAE (MPa) | CV R² |
|---|---|---|---|---|
| Random Forest | 0.667 | 73.6 | 46.7 | 0.509 |
| XGBoost | 0.547 | 85.8 | 45.6 | 0.582 |
| Linear Regression | 0.231 | 111.8 | 87.0 | 0.171 |

Random Forest and XGBoost performed comparably. **Random Forest** was selected for the app because of its stronger test performance and more stable cross-validation scores.

**Feature importance (Random Forest):** SiC volume % (0.34) and grain size (0.27) were the most influential inputs, followed by the processing group (Deformed 0.16, Stir casting 0.09) and SiC particle size (0.12). The partial dependence plots show strength increasing with SiC content and decreasing with grain size, consistent with the expected strengthening effects of particle reinforcement and grain refinement (Hall-Petch).

## Scope

The model was trained on SiC-reinforced AZ91 only and is intended for that material system. Predictions are most reliable within the ranges most densely covered by the experimental data (SiC up to 20 vol%, grain size up to 15 µm, particle size up to 50 µm). Outside these ranges the estimates extend learned trends and are best confirmed experimentally.

## Project structure

```
SiC-AZ91-UTS-Prediction/
├── app.py                  # Streamlit application
├── requirements.txt        # Dependencies for the app
├── requirements-dev.txt    # Extra dependencies for the notebook
├── models/
│   ├── rf_uts_pipeline.joblib      # Trained Random Forest pipeline
│   └── feature_columns.joblib      # Feature names and order
├── notebooks/
│   └── 01_eda_and_modeling.ipynb   # EDA, modeling and evaluation
└── data/
    └── raw/SiC Data.xlsx
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To rerun the notebook, install the extra packages with `pip install -r requirements-dev.txt`.

The saved model was created with scikit-learn 1.9.1, so `requirements.txt` pins that version to ensure it loads correctly.

## Author

Developed and deployed by **Hamid Kofi Abdul**
