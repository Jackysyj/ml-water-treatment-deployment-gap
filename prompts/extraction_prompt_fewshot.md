# Water Treatment ML Extraction Prompt v2

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
   - **WWTP**: Effluent quality prediction (BOD, COD, NH3-N, TSS, TN, TP), energy consumption prediction, process optimization of activated sludge systems
   - **membrane**: Fouling prediction, flux decline, TMP prediction, cleaning optimization, MBR, RO, NF, UF
   - **coagulation**: Coagulant/flocculant dose optimization, turbidity prediction, jar test modeling
   - **DBP**: Disinfection byproduct formation prediction (THMs, HAAs, NDMA, chlorine residual)
   - **monitoring**: Water quality monitoring, anomaly detection, contamination warning in distribution systems
   - **sludge**: Sludge treatment, dewatering, anaerobic digestion, biogas/methane prediction
   - **control**: Real-time process control, model-predictive control (MPC), reinforcement learning for plant operation, DO/aeration control
   - **other**: Does not fit any above category

6. **data_source**: Where the training data came from:
   - **SCADA**: Operational plant data from SCADA/PLC/data historian systems, online sensors
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

8. **time_span**: Duration of data collection. Use natural language (e.g., "2 years", "6 months", "365 days"). null if not time-series or not reported.

9. **target_variable**: The PRIMARY prediction target. Be specific (e.g., "effluent COD" not just "water quality").

### ML Model Details

10. **ml_algorithms**: List ALL ML algorithms used in the study. Use standard abbreviations:
    - Neural networks: ANN, LSTM, GRU, CNN, RNN, Transformer, MLP, BPNN, ELM, RBFNN, Autoencoder
    - Tree-based: RF, XGBoost, GBM, GBDT, DT, AdaBoost, LightGBM, CatBoost, ET (Extra Trees)
    - Kernel/linear: SVR, SVM, GPR, LR, LASSO, Ridge, ElasticNet, KNN, LSSVM
    - Control: RL, DRL, DQN, PPO, A3C, MPC
    - Other: Bayesian, Fuzzy, ANFIS, PLS, PCR, Ensemble
    - Include ALL models compared, not just the best one

11. **best_algorithm**: The single best-performing algorithm as reported by the authors. Use the same abbreviation as in ml_algorithms.

### Performance Metrics (CRITICAL — read carefully)

12. **best_metric_type**: The PRIMARY metric used to identify the best model. Follow this decision tree:

    **Step 1 — Identify the metric the authors use to declare a "best" model.**
    Look for phrases like "achieved the highest R²", "lowest RMSE", "best accuracy".

    **Step 2 — Distinguish R² from R (correlation coefficient).**
    - R² (coefficient of determination) is always between -∞ and 1, typically 0.8–0.99 for good models
    - R (Pearson correlation) is between -1 and 1
    - If the paper says "R = 0.95" or "correlation coefficient = 0.95", that is R, NOT R²
    - If the paper says "R² = 0.95" or "coefficient of determination = 0.95", that is R2
    - If ambiguous, check: values very close to 1.0 (e.g., 0.999) are more likely R²

    **Step 3 — If multiple metrics are reported, use this priority:**
    R2 > RMSE > MAE > MSE > MAPE > NSE > accuracy > F1 > AUC > other

    **Step 4 — Map to allowed values:**
    - R2, RMSE, MAE, accuracy, F1, AUC, MSE, MAPE, NSE → use directly
    - NRMSE, nRMSE → "RMSE" (note: value will be normalized)
    - R (correlation) → "other"
    - IoA (Index of Agreement) → "other"
    - null → metric not reported or unclear

13. **best_metric_value**: The numeric value of the best metric on the TEST set.
    - Must be from test/validation set, NOT training set
    - If only training metrics reported, use null
    - For R²: typically 0.0–1.0
    - For RMSE/MAE/MSE: can be any positive number (units depend on target variable)
    - For MAPE: typically 0–100 (percentage)
    - For accuracy/F1/AUC: typically 0.0–1.0 or 0–100

### Validation Strategy (CRITICAL — read carefully)

14. **validation_method**: How the model was validated. Follow this decision tree:

    **Step 1 — Look for explicit statements about data splitting.**
    Search for: "train-test split", "cross-validation", "k-fold", "holdout", "temporal", "chronological".

    **Step 2 — Classify using these rules:**
    - **random_split**: Data randomly divided into train/test (e.g., "80/20 split", "70/15/15 split"). This is the DEFAULT if the paper says "split" without specifying temporal order.
    - **k_fold**: k-fold cross-validation (e.g., "5-fold CV", "10-fold cross-validation", "leave-one-out" with small k). Includes stratified k-fold.
    - **temporal_split**: Train on EARLIER data, test on LATER data. Look for: "first N months for training, last M months for testing", "2018-2020 training, 2021 testing", "chronological split". The key indicator is that time order is preserved.
    - **external**: Model tested on a completely different dataset (different plant, different location, different time period collected independently).
    - **LOOCV**: Leave-one-out cross-validation (each sample used as test once).
    - **walk_forward**: Sliding/expanding window validation. Look for: "rolling forecast", "walk-forward", "sliding window validation".
    - **none_reported**: No validation strategy described. The paper only reports training performance or does not mention how data was split.

    **Step 3 — Common pitfalls:**
    - "80/20 split" without mentioning time → random_split (NOT temporal_split)
    - "trained on 2019 data, tested on 2020 data" → temporal_split
    - "5-fold CV" → k_fold (even if also has a final test set)
    - "validated on data from Plant B" → external
    - Paper only shows training R² → none_reported

### Deployment & Real-World Testing

15. **real_time_testing**: TRUE only if the model was tested with LIVE, streaming data from an operating plant in real-time. NOT the same as:
    - Using historical SCADA data (that's just offline testing)
    - Temporal train/test split (that's validation_method)
    - Simulation testing
    Look for: "online testing", "real-time deployment", "live data feed", "connected to SCADA in real-time"

16. **deployed_in_plant**: TRUE only if there is CLEAR evidence the model was actually deployed for operational decision-making in a real plant. NOT:
    - Developing a GUI or dashboard (that's a prototype)
    - Testing on historical data (that's offline validation)
    - "Can be deployed" or "suitable for deployment" (that's a claim, not evidence)
    - Pilot testing in a lab (that's experimental)
    Look for: "deployed at [plant name]", "operational since [date]", "integrated into the plant's control system"

17. **uses_real_wastewater**: TRUE if the study uses real water/wastewater from an actual plant or distribution system. FALSE if purely synthetic, simulated, or lab-prepared solutions.

### Reproducibility

18. **code_available**: TRUE if the code/model is publicly available (GitHub, Zenodo, etc.). Look in Data Availability section, footnotes, or acknowledgments.

19. **data_available**: TRUE if the training data is publicly available. Look in Data Availability section. "Data available upon request" = FALSE.

20. **uncertainty_quantification**: TRUE if the model provides prediction intervals, confidence intervals, Bayesian posterior, or other uncertainty estimates. Simple error metrics (RMSE, MAE) do NOT count. Ensemble variance counts.

### Interpretability & Deployment Context

21. **interpretability_method**: What method is used to explain model predictions?
   - **SHAP**: SHapley Additive exPlanations (look for "SHAP values", "Shapley")
   - **LIME**: Local Interpretable Model-agnostic Explanations
   - **feature_importance**: Built-in feature importance from tree-based models (RF, XGBoost, GBM). Also includes "variable importance", "Gini importance", "permutation importance"
   - **partial_dependence**: Partial dependence plots (PDP) or individual conditional expectation (ICE)
   - **attention_weights**: Neural network attention mechanism visualization
   - **none**: No interpretability/explainability analysis performed
   - If MULTIPLE methods used, report the most sophisticated: SHAP > LIME > partial_dependence > feature_importance > attention_weights

22. **model_framework**: What ML framework or library was used?
   - **tensorflow**: TensorFlow (including TF 2.x, tf.keras)
   - **pytorch**: PyTorch
   - **sklearn**: scikit-learn
   - **matlab**: MATLAB (Neural Network Toolbox, Statistics Toolbox, nntool, fitnet, etc.)
   - **keras**: Keras used standalone (not as tf.keras). If paper says "Keras with TensorFlow backend", use "tensorflow"
   - **r**: R language (caret, tidymodels, randomForest package, etc.)
   - **weka**: Weka
   - **not_specified**: Framework not mentioned in the paper. Do NOT guess from the algorithm used.

23. **control_loop_type**: How is the ML model integrated with plant control?
   - **open_loop**: Model predicts/forecasts but output is NOT automatically fed back to control the plant
   - **closed_loop**: Model output directly controls actuators (pumps, valves, chemical dosing) without human intervention. Includes RL/DRL agents that directly output control actions.
   - **advisory**: Model provides recommendations or decision support to human operators who make final decisions. Includes "decision support system", "operator guidance"
   - **none**: Pure prediction/modeling study with no control integration aspect. This is the DEFAULT for most papers.

24. **scale**: What is the scale of the study?
   - **lab**: Bench-scale laboratory experiments
   - **pilot**: Pilot plant (intermediate scale, not full municipal/industrial)
   - **full_scale**: Full-scale operational plant (municipal or industrial). If data comes from a real WWTP via SCADA, this is typically full_scale.
   - **simulation**: Software simulation only (GPS-X, BioWin, BSM1/BSM2, EPANET)
   - **mixed**: Multiple scales combined (e.g., lab validation + full-scale testing, or simulation + real plant data)

## Critical Reminders

1. Read the ENTIRE paper before extracting.
2. For boolean fields: default to FALSE unless there is CLEAR evidence.
3. `best_metric_value` must be from the TEST set, not training.
4. `real_time_testing` ≠ temporal_split. Real-time requires live data streaming.
5. `deployed_in_plant` requires evidence of operational use, not just testing.
6. `best_metric_type`: R ≠ R². Check carefully. Use the decision tree above.
7. `validation_method`: "80/20 split" without time context = random_split, NOT temporal_split.
8. Return ONLY the JSON object. No explanation.

## Examples

Below are 5 annotated examples showing the expected extraction output for different sub-fields. Study these carefully before extracting from the target paper.

### Example 1: MEMBRANE paper

**Paper excerpt:**
```
# Predicting Membrane Fouling of Submerged Membrane Bioreactor Wastewater Treatment Plants Using Machine Learning

Yunyi Zhu, Yuan Wang, Elisabeth Zhu, Zeyu Ma, Hanchen Wang, Chunsheng Chen, Jing Guan, and T. David Waite\*

Cite This: https://doi.org/10.1021/acs.est.4c12835

# Read Online

# ACCESS I

LMetrics & More

# Article Recommendations

Supporting Information

ABSTRACT: Membrane fouling remains a significant challenge in the operation of membrane bioreactors (MBRs). Plant operators rely heavily on observations of fltration performance from noisy sensor data to assess membrane fouling conditions and lab-based protocols for plant maintenance, often leading to inaccurate estimations of future performance and delayed membrane cleaning. This challenge is further compounded by the difficulty in integrating existing complex mechanistic models with the Internet of Things (IoT) systems of wastewater treatment plants (WwTPs). By harnessing data obtained from WWTPs, along with innovative data denoising and model training strategies, we developed a machine learning application (MBR-Net) that is capable of forecasting membrane fouling, as indicated by permeability, for a

data collection data processing model training A A sensors > -> |OT denoising PEE-Huber -> " -> <- loT platform encoder →decoder   
submerged   
MBR plant selection tuning   
MBR-Net model deployment performance evaluation

fu irreversible fouling under different desired fluxes, cleaning conditions and feedwater
[... remainder of paper ...]
```

**Expected output:**
```json
{
  "paper_title": "Predicting Membrane Fouling of Submerged Membrane Bioreactor Wastewater Treatment Plants Using Machine Learning",
  "doi": "10.1021/acs.est.4c12835",
  "year": 2025,
  "journal": "Environmental Science & Technology",
  "sub_field": "membrane",
  "data_source": "SCADA",
  "dataset_size": null,
  "time_span": "~2 years",
  "target_variable": "membrane permeability",
  "ml_algorithms": [
    "LSTM"
  ],
  "best_algorithm": "LSTM",
  "best_metric_type": "R2",
  "best_metric_value": 0.87,
  "validation_method": "temporal_split",
  "real_time_testing": true,
  "deployed_in_plant": false,
  "uses_real_wastewater": true,
  "code_available": true,
  "data_available": false,
  "uncertainty_quantification": false,
  "interpretability_method": "SHAP",
  "model_framework": "pytorch",
  "control_loop_type": "none",
  "scale": "full_scale"
}
```


### Example 2: WWTP paper

**Paper excerpt:**
```
# Forecasting of Wastewater Treatment Plant Key Features Using Deep Learning-Based Models: A Case Study

TUOYUAN CHENG 1, FOUZI HARROU 2, (Member, IEEE), FARID KADRI3, YING SUN 2, AND TOROVE LEIKNES 1

1Division of Biological and Environmental Science and Engineering (BESE), Water Desalination and Reuse Center, King Abdullah University of Science and   
Technology (KAUST), Thuwal 23955-6900, Saudi Arabia   
2Statistics Program, Division of Computer, Electrical and Mathematical Sciences and Engineering (CEMSE), King Abdullah University of Science and   
Technology (KAUST), Thuwal 23955-6900, Saudi Arabia   
3Aeroline and Customer Services, Agence 1024, Sopra Steria Group, 31770 Colomiers, France

Corresponding author: Fouzi Harrou (fouzi.harrou@kaust.edu.sa)

This work was supported by the King Abdullah University of Science and Technology (KAUST), Office of Sponsored Research (OSR) under Award OSR-2019-CRG7-3800.

ABSTRACT The accurate forecast of wastewater treatment plant (WWTP) key features can comprehend and predict the plant behavior to support process design and controls, improve system reliability, reduce operational costs, and endorse optimization of overall performances. Deep learning technologies as proven data-driven soft-sensors should be developed for WWTP applications to tackle the process of non-linearity and the dynamic nature of environmental data. This study adopts deep learning-based models as soft-sensors to forecast WWTP key features, such as influent flo
[... remainder of paper ...]
```

**Expected output:**
```json
{
  "paper_title": "Forecasting of Wastewater Treatment Plant Key Features Using Deep Learning-Based Models: A Case Study",
  "doi": "10.1109/ACCESS.2021.3063784",
  "year": 2021,
  "journal": "IEEE Access",
  "sub_field": "WWTP",
  "data_source": "SCADA",
  "dataset_size": 2557,
  "time_span": "7 years (September 2010 to September 2017)",
  "target_variable": "influent flow, influent temperature, influent BOD, effluent chloride, effluent BOD, power consumption",
  "ml_algorithms": [
    "LSTM",
    "GRU"
  ],
  "best_algorithm": "GRU",
  "best_metric_type": "MAPE",
  "best_metric_value": 0.07,
  "validation_method": "temporal_split",
  "real_time_testing": false,
  "deployed_in_plant": false,
  "uses_real_wastewater": true,
  "code_available": false,
  "data_available": false,
  "uncertainty_quantification": false,
  "interpretability_method": "none",
  "model_framework": "tensorflow",
  "control_loop_type": "none",
  "scale": "full_scale"
}
```


### Example 3: CONTROL paper

**Paper excerpt:**
```
Article

# LSTM-Based Model-Predictive Control with Rationality Verification for Bioreactors in Wastewater Treatment

Yuting Liu 1, Wenchong Tian 1, Jun Xie 2, Weizhong Huang 2 and Kunlun Xin $\mathbf { 1 } , { * } \textcircled { | | }$

Academic Editor: Constantinos V. Chrysikopoulos

Received: 17 April 2023   
Revised: 28 April 2023   
Accepted: 29 April 2023   
Published: 5 May 2023

1 College of Environmental Science and Engineering, Tongji University, Shanghai 200092, China; lalyt0924@163.com (Y.L.); wenchong@tongji.edu.cn (W.T.)   
2 Shanghai Urban Construction Design Research Institute, Shanghai 200125, China   
\* Correspondence: xkl@tongji.edu.cn

Abstract: With the increasing demands for higher treatment efficiency, better effluent quality, and energy conservation in Urban Wastewater Treatment Plants (WWTPs), research has already been conducted to construct an optimized control system for Anaerobic-Anoxic-Oxic (AAO) process using a data-driven approach. However, existing data-driven optimization control systems for AAO mainly focus on improving effluent water quality and reducing energy consumption, therefore they lack consideration for the stability of bioreactors. Meanwhile, safety in the optimization control process is still missing, resulting in a lack of reliability in practical applications. In this study, long short-term memory based model-predictive control (LSTM-MPC) with safety verificationis developed for the real-time control of AAO. It is used to optimi
[... remainder of paper ...]
```

**Expected output:**
```json
{
  "paper_title": "LSTM-Based Model-Predictive Control with Rationality Verification for Bioreactors in Wastewater Treatment",
  "doi": "10.3390/w15091779",
  "year": 2023,
  "journal": "Water",
  "sub_field": "control",
  "data_source": "SCADA",
  "dataset_size": 30000,
  "time_span": "June 2019 to September 2019",
  "target_variable": "Bioreactor state variables (DO, MLSS, ORP, NO3)",
  "ml_algorithms": [
    "LSTM",
    "MPC"
  ],
  "best_algorithm": "LSTM",
  "best_metric_type": "NSE",
  "best_metric_value": 0.99,
  "validation_method": "random_split",
  "real_time_testing": true,
  "deployed_in_plant": false,
  "uses_real_wastewater": true,
  "code_available": true,
  "data_available": true,
  "uncertainty_quantification": false,
  "interpretability_method": "none",
  "model_framework": "pytorch",
  "control_loop_type": "closed_loop",
  "scale": "full_scale"
}
```


### Example 4: DBP paper

**Paper excerpt:**
```
Full length article

# Predicting regulated and emerging disinfection byproducts in small drinking water catchments using machine learning

Boris Droz ${ \mathrm { a } } , { \mathrm { b } } , { } ^ { \ast } , 1 _ { \emptyset }$ , Elena Fernandez-Pascual ´ a,b , Jean O’Dwyer $\mathrm { a , b , c } _ { \Phi }$ , Emma H. Goslan $\mathrm { d } _ { \oplus }$ Xie Quishi b , Connie O’Driscoll e , Simon Harrison a,b , John Weatherill a,b,c,\*

a School of Biological, Earth and Environmental Sciences, University College Cork, Cork T23 TK30, Ireland b Sustainability Institute, Ellen Hutchins Building, University College Cork, Cork T23 XE10, Ireland c iCRAG Research Ireland Centre for Applied Geosciences, University College Dublin, Dublin D04 V1W8, Ireland d Cranfield Water Science Institute, Cranfield University, Cranfield MK43 0AL, UK e Ryan Hanley Ltd., Castlebar F23 E400, Ireland

# A R T I C L E I N F O

# A B S T R A C T

Keywords:   
Disinfection byproduct formation   
machine learning   
fluorescence excitation-emission matrix spec  
troscopy   
dissolved organic matter   
drinking water   
UV–Vis spectroscopy

Potentially harmful concentrations of disinfection byproducts (DBPs) arise from unintended reactions between chemical disinfectants and dissolved organic matter (DOM) present in raw drinking water sources. We explore the application of machine learning tools trained on DOM spectroscopic variables and hydrochemical parameters to predict the formation of regulated and emerg
[... remainder of paper ...]
```

**Expected output:**
```json
{
  "paper_title": "Predicting regulated and emerging disinfection byproducts in small drinking water catchments using machine learning",
  "doi": "10.1016/j.envint.2025.109923",
  "year": 2025,
  "journal": "Environment International",
  "sub_field": "DBP",
  "data_source": "experimental",
  "dataset_size": 198,
  "time_span": null,
  "target_variable": "DBP formation potential (THMs, HAAs, HANs, HKs, TCNM)",
  "ml_algorithms": [
    "ANN",
    "Bagging",
    "GBM",
    "SVM"
  ],
  "best_algorithm": "ANN",
  "best_metric_type": "R2",
  "best_metric_value": 0.86,
  "validation_method": "k_fold",
  "real_time_testing": false,
  "deployed_in_plant": false,
  "uses_real_wastewater": true,
  "code_available": true,
  "data_available": false,
  "uncertainty_quantification": true,
  "interpretability_method": "SHAP",
  "model_framework": "sklearn",
  "control_loop_type": "none",
  "scale": "lab"
}
```


### Example 5: SLUDGE paper

**Paper excerpt:**
```
# Machine learning–based methane yield prediction using a structured anaerobic digestion dataset

Asit Chatterjee 1, \*, Mahim Mathur 1, Anil Pal 2 and Mukesh Kumar Gupta 3

1 Department of Civil Engineering, Suresh Gyan Vihar University, Jaipur, India.   
2 Department of Computer Application, Suresh Gyan Vihar University, Jaipur, India.   
3 Department of Electrical Engineering, Suresh Gyan Vihar University, Jaipur, India.

World Journal of Advanced Engineering Technology and Sciences, 2025, 17(03), 112–120

Publication history: Received 30 October 2025; revised on 06 December 2025; accepted on 09 December 2025

Article DOI: https://doi.org/10.30574/wjaets.2025.17.3.1546

# Abstract

Accurate prediction of methane yield is essential for optimizing anaerobic digestion (AD) systems and improving the efficiency of agricultural biomass-to-energy conversion. This study presents a machine learning–based predictive framework trained on a structured and experimentally derived dataset encompassing physicochemical feedstock properties, operational parameters, and biogas performance indicators. The dataset includes more than 500 labeled samples representing major agricultural residues, characterized by Total Solids (TS), Volatile Solids (VS), $\mathrm { C } / \mathrm { N }$ ratio, lignocellulosic composition, temperature, pH, and Organic Loading Rate. Six supervised learning algorithms like Gradient Boosting Regressor (GBR), Light GBM, Cat Boost, Extra Trees, K-Nearest Neighbors (KNN),
[... remainder of paper ...]
```

**Expected output:**
```json
{
  "paper_title": "Machine learning–based methane yield prediction using a structured anaerobic digestion dataset",
  "doi": "10.30574/wjaets.2025.17.3.1546",
  "year": 2025,
  "journal": "World Journal of Advanced Engineering Technology and Sciences",
  "sub_field": "sludge",
  "data_source": "experimental",
  "dataset_size": 500,
  "time_span": null,
  "target_variable": "Methane yield (mL CH4/g VS)",
  "ml_algorithms": [
    "GBM",
    "LightGBM",
    "XGBoost",
    "ET",
    "KNN",
    "LR"
  ],
  "best_algorithm": "LightGBM",
  "best_metric_type": "R2",
  "best_metric_value": 0.95,
  "validation_method": "k_fold",
  "real_time_testing": false,
  "deployed_in_plant": false,
  "uses_real_wastewater": false,
  "code_available": false,
  "data_available": false,
  "uncertainty_quantification": false,
  "interpretability_method": "feature_importance",
  "model_framework": "sklearn",
  "control_loop_type": "none",
  "scale": "lab"
}
```


## Now extract from the following paper:

{content}
