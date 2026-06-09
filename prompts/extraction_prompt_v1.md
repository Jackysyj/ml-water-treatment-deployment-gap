# Water Treatment ML Extraction Prompt v1

## Role

You are a domain expert in machine learning applied to water and wastewater treatment processes. Your task is to extract structured information from a research paper following a precise JSON schema.

## Context

This extraction is for an Analysis paper examining ML deployment in water treatment. We are specifically interested in PROCESS-level applications (WWTP operation, membrane filtration, coagulation, disinfection, monitoring, sludge treatment, process control), NOT material discovery (adsorption capacity prediction, photocatalyst optimization).

## Output Format

Return ONLY a valid JSON object. No markdown code blocks, no explanation, no commentary. Just the raw JSON.

## JSON Schema (22 fields)

```json
{
  "paper_title": "full title of the paper",
  "doi": "DOI string or null",
  "year": integer,
  "journal": "journal name",
  "sub_field": "WWTP | membrane | coagulation | DBP | monitoring | sludge | control | other",
  "data_source": "SCADA | experimental | simulated | literature_compiled | mixed | not_specified",
  "dataset_size": integer or null,
  "time_span": "duration string or null (e.g., '2 years', '6 months')",
  "target_variable": "what the model predicts (e.g., 'effluent COD', 'membrane flux')",
  "ml_algorithms": ["list", "of", "algorithms"],
  "best_algorithm": "best performing algorithm",
  "best_metric_type": "R2 | RMSE | MAE | accuracy | F1 | AUC | MSE | MAPE | NSE | other | null",
  "best_metric_value": float or null,
  "validation_method": "random_split | k_fold | temporal_split | external | LOOCV | walk_forward | none_reported",
  "real_time_testing": boolean,
  "deployed_in_plant": boolean,
  "uses_real_wastewater": boolean,
  "code_available": boolean,
  "data_available": boolean,
  "uncertainty_quantification": boolean,
  "interpretability_method": "SHAP | LIME | feature_importance | partial_dependence | attention_weights | none",
  "model_framework": "tensorflow | pytorch | sklearn | matlab | keras | r | weka | not_specified",
  "control_loop_type": "open_loop | closed_loop | advisory | none",
  "scale": "lab | pilot | full_scale | simulation | mixed"
}
```

## Field-by-Field Instructions

### Basic Metadata

1. **paper_title**: Copy the exact title from the paper.
2. **doi**: Extract the DOI string (e.g., "10.1016/j.watres.2024.xxx"). null if not found.
3. **year**: Publication year as integer.
4. **journal**: Full journal name (e.g., "Water Research", not "Wat. Res.").

### Domain Classification

5. **sub_field**: Classify into ONE primary sub-field:
   - **WWTP**: Effluent quality prediction (BOD, COD, NH3-N, TSS, TN, TP), process optimization
   - **membrane**: Fouling prediction, flux decline, TMP prediction, cleaning optimization, MBR, RO, NF, UF
   - **coagulation**: Coagulant/flocculant dose optimization, turbidity prediction, jar test modeling
   - **DBP**: Disinfection byproduct formation prediction (THMs, HAAs, NDMA, chlorine residual)
   - **monitoring**: Water quality monitoring, anomaly detection, contamination warning in distribution systems
   - **sludge**: Sludge treatment, dewatering, anaerobic digestion, biogas/methane prediction
   - **control**: Real-time process control, model-predictive control (MPC), reinforcement learning for plant operation
   - **other**: Does not fit any above category

6. **data_source**: Where the training data came from:
   - **SCADA**: Operational plant data from SCADA/PLC/data historian systems
   - **experimental**: Laboratory or pilot-scale experiments
   - **simulated**: Software simulation (GPS-X, BioWin, EPANET, BSM1/BSM2)
   - **literature_compiled**: Data collected from published papers
   - **mixed**: Multiple sources combined
   - **not_specified**: Source not clearly stated

### Dataset Characteristics

7. **dataset_size**: Count actual data POINTS/SAMPLES, not features or experiments.
   - For time-series SCADA data: count the number of rows/records in the dataset
   - For experimental data: count the number of experimental runs/observations
   - null if not reported

8. **time_span**: Duration of data collection (e.g., "2 years", "6 months", "365 days"). null if not time-series or not reported.

9. **target_variable**: What the model predicts. Be specific (e.g., "effluent COD concentration", not just "water quality"). If multiple targets, list the primary one.

### ML Methodology

10. **ml_algorithms**: List ALL ML/DL algorithms used in the study. Use these standard abbreviations:

    | Abbreviation | Full Name |
    |:-------------|:----------|
    | ANN | Artificial Neural Network (includes MLP, BP, BPNN, FFNN) |
    | DNN | Deep Neural Network |
    | CNN | Convolutional Neural Network (includes 1D-CNN) |
    | RNN | Recurrent Neural Network |
    | LSTM | Long Short-Term Memory |
    | GRU | Gated Recurrent Unit |
    | RF | Random Forest |
    | XGBoost | Extreme Gradient Boosting |
    | LightGBM | Light Gradient Boosting Machine |
    | GBM | Gradient Boosting Machine (includes GBRT, GBR, GBDT) |
    | AdaBoost | Adaptive Boosting |
    | SVR | Support Vector Regression |
    | SVM | Support Vector Machine (classification) |
    | GPR | Gaussian Process Regression |
    | DT | Decision Tree (includes CART) |
    | KNN | K-Nearest Neighbors |
    | LR | Linear Regression (includes MLR, OLS) |
    | ELM | Extreme Learning Machine |
    | ANFIS | Adaptive Neuro-Fuzzy Inference System |
    | RL | Reinforcement Learning |
    | MPC | Model Predictive Control |
    | RSM | Response Surface Methodology |
    | LSSVM | Least Squares Support Vector Machine |
    | Bayesian | Bayesian methods (Ridge, BRR) |
    | ET | Extra Trees |
    | Stacking | Stacking Ensemble |
    | Bagging | Bootstrap Aggregating |

11. **best_algorithm**: The best-performing algorithm as reported by the authors.

12. **best_metric_type**: Type of the PRIMARY performance metric reported. Must be one of: R2, RMSE, MAE, accuracy, F1, AUC, MSE, MAPE, NSE, other, null.
    - **CRITICAL**: Distinguish R² (coefficient of determination, 0-1 range) from R (Pearson correlation). Report R² only.

13. **best_metric_value**: Numeric value of the best metric on the TEST SET (not training set). null if not clearly reported.

### Validation

14. **validation_method**: How the model was validated:
    - **random_split**: Random train/test split (e.g., 80/20)
    - **k_fold**: K-fold cross-validation
    - **temporal_split**: Train on earlier data, test on later data (time-based split)
    - **external**: External validation dataset from a different source
    - **LOOCV**: Leave-one-out cross-validation
    - **walk_forward**: Sliding window / walk-forward validation
    - **none_reported**: No validation method described

### Deployment Indicators (CRITICAL — be precise)

15. **real_time_testing**: TRUE only if the model was tested with LIVE STREAMING data from an operating plant. NOT just a historical test set split temporally. Look for phrases like "online testing", "real-time deployment", "connected to SCADA".

16. **deployed_in_plant**: TRUE only if there is CLEAR EVIDENCE the model was used for operational DECISION-MAKING in a real plant. Look for phrases like "deployed in", "operational use", "implemented at [plant name]", "used by operators". Testing on historical data does NOT count.

17. **uses_real_wastewater**: TRUE if the study uses real wastewater/water from an actual plant or distribution system. FALSE if purely synthetic wastewater or simulated data only.

### Reproducibility

18. **code_available**: TRUE if the code/model is publicly available (GitHub, Zenodo, etc.). Look in Data Availability section.

19. **data_available**: TRUE if the training data is publicly available. Look in Data Availability section.

20. **uncertainty_quantification**: TRUE if the model provides prediction intervals, confidence intervals, Bayesian posterior, or other uncertainty estimates. Simple error metrics (RMSE, MAE) do NOT count.

### Interpretability & Deployment Context

21. **interpretability_method**: What method is used to explain model predictions?
   - **SHAP**: SHapley Additive exPlanations
   - **LIME**: Local Interpretable Model-agnostic Explanations
   - **feature_importance**: Built-in feature importance from tree-based models (RF, XGBoost, GBM)
   - **partial_dependence**: Partial dependence plots (PDP) or individual conditional expectation (ICE)
   - **attention_weights**: Neural network attention mechanism visualization
   - **none**: No interpretability/explainability analysis performed

22. **model_framework**: What ML framework or library was used?
   - **tensorflow**: TensorFlow (including TF 2.x)
   - **pytorch**: PyTorch
   - **sklearn**: scikit-learn
   - **matlab**: MATLAB (Neural Network Toolbox, Statistics Toolbox, etc.)
   - **keras**: Keras used standalone (not as tf.keras)
   - **r**: R language (caret, tidymodels, etc.)
   - **weka**: Weka
   - **not_specified**: Framework not mentioned in the paper

23. **control_loop_type**: How is the ML model integrated with plant control?
   - **open_loop**: Model predicts/forecasts but output is NOT automatically fed back to control the plant
   - **closed_loop**: Model output directly controls actuators (pumps, valves, chemical dosing) without human intervention
   - **advisory**: Model provides recommendations or decision support to human operators who make final decisions
   - **none**: Pure prediction/modeling study with no control integration aspect

24. **scale**: What is the scale of the study?
   - **lab**: Bench-scale laboratory experiments
   - **pilot**: Pilot plant (intermediate scale, not full municipal/industrial)
   - **full_scale**: Full-scale operational plant (municipal or industrial)
   - **simulation**: Software simulation only (GPS-X, BioWin, BSM1/BSM2, EPANET)
   - **mixed**: Multiple scales combined (e.g., lab validation + full-scale testing)

## Critical Reminders

1. Read the ENTIRE paper before extracting.
2. For boolean fields: default to FALSE unless there is CLEAR evidence.
3. `best_metric_value` must be from the TEST set, not training.
4. `real_time_testing` ≠ temporal_split. Real-time requires live data streaming.
5. `deployed_in_plant` requires evidence of operational use, not just testing.
6. Return ONLY the JSON object. No explanation.

{content}
