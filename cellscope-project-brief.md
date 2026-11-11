# CellScope: Explainable Biomedical Image Classification and Review Workbench

## Project brief

**CellScope** is a portfolio project that combines biomedical image analysis, machine learning, software engineering, and user-focused application development.

The application will accept a microscopy image, predict its cell type or phenotype, show the model's confidence and visual explanation, and allow a user to review the prediction. The project should be built as a clean-room implementation using public data only. It must be presented as an educational research and software project, **not as a clinical diagnostic system**.

The project is intentionally designed to support applications to:

- Biopharma and pharmaceutical companies
- Life-science software companies
- Data analyst and data scientist roles
- Machine-learning and computer-vision roles
- Python/backend or full-stack software roles
- Scientific computing and bioinformatics-adjacent roles

It adds skills that are not already the central focus of your current resume: computer vision, reproducible ML experimentation, model evaluation, explainability, model-serving, testing, and product-style documentation.

## Why this is a good project for your profile

Your resume already demonstrates experience with web applications, bioprocess data, statistical analysis, reporting, control charts, PCA, cloud services, testing, and application deployment. CellScope should therefore avoid becoming another bioreactor-monitoring application. Instead, it should use a different technical problem while preserving a credible life-science connection.

The project gives you two stories to tell:

### Life-science story

> I understand that biomedical ML is not only about training a model. I can work with image data, labels, class imbalance, experimental variation, dataset provenance, validation design, and the limits of biological conclusions.

### IT and software story

> I can take a model beyond a notebook and turn it into a maintainable application with a clear interface, input validation, automated tests, reproducible experiments, explainability, and documented limitations.

## Recommended project scope

Do not begin with a large image archive or a complex cloud deployment. Build the project in three levels:

1. **Core ML study:** train and evaluate several classifiers on BloodMNIST.
2. **Interactive application:** create a local Streamlit review interface.
3. **Engineering and scientific extensions:** add tests, explainability, experiment tracking, data-provenance documentation, and one optional second dataset or task.

Docker is **not required**. A Python virtual environment and a `requirements.txt` file are sufficient for the first complete version. Docker can remain an optional stretch goal.

## Project outcome

At the end, a reviewer should be able to:

1. Read the README and understand the scientific question.
2. Install the project locally without needing an API key.
3. Run a reproducible training command.
4. Inspect the evaluation results and confusion matrix.
5. Launch the Streamlit application.
6. Upload or select an image and see a prediction.
7. See the confidence distribution and model explanation.
8. Understand which cases the model gets wrong.
9. Review the data source, license, model limitations, and intended use.

## Project question

Use this question as the central research question:

> How reliably can lightweight image-classification models distinguish blood-cell categories from microscopy images, and how can a user understand when a prediction should be treated cautiously?

This question is more credible than simply asking, “Can a neural network classify images?” It gives you room to discuss baselines, errors, uncertainty, class imbalance, interpretability, and practical software design.

## Primary data source: MedMNIST BloodMNIST

### Why BloodMNIST is the recommended starting point

BloodMNIST is small enough to work with on an ordinary computer and is distributed through the official MedMNIST project. The BloodMNIST subset contains 17,092 RGB images organized into eight blood-cell classes, with standard training, validation, and test splits. The official MedMNIST project provides standardized biomedical datasets and multiple image-size options, including the lightweight 28 × 28 format and larger MedMNIST+ formats.

Official sources:

- MedMNIST website: <https://medmnist.com/>
- Official MedMNIST code and download instructions: <https://github.com/MedMNIST/MedMNIST>
- MedMNIST v2 paper: <https://doi.org/10.1038/s41597-022-01721-8>
- Official project license information: <https://medmnist.com/#license>

The MedMNIST collection is intended for research and educational use and explicitly states that it is not intended for clinical use. Keep that limitation visible in the application and README.

### BloodMNIST classes

The eight classes are:

- Basophil
- Eosinophil
- Erythroblast
- Immature granulocytes
- Lymphocyte
- Monocyte
- Neutrophil
- Platelet

Use the class names in the interface rather than displaying only numeric labels.

### License and attribution

MedMNIST is listed under Creative Commons Attribution 4.0 International, with dataset-specific exceptions noted by the official project. Check the official license page before publishing a derivative dataset or redistributing image files. Include the MedMNIST paper and the source-data citations in your README.

A safe portfolio practice is:

- Do not commit the complete dataset to GitHub.
- Download it through the official package or source link.
- Commit only a small set of legally permitted demonstration assets, if needed.
- Record the dataset version, access date, source URL, license, and checksum or file metadata.
- Include attribution in the README and application About section.

## Optional second data source: Broad Bioimage Benchmark Collection

After the BloodMNIST version is complete, add one optional extension from the Broad Bioimage Benchmark Collection, commonly called BBBC. It contains freely downloadable microscopy image sets with descriptions and different kinds of ground truth.

Official source:

- BBBC image-set catalogue: <https://bbbc.broadinstitute.org/image_sets>

Do not begin with the largest datasets. Select one focused extension according to your available storage and computing power.

### Option A: BBBC041 malaria blood-smear images

BBBC041 contains bright-field blood-smear images with infected and uninfected cell categories, class labels, and bounding-box annotations. The official page states that the archive is approximately 2.26 GB and that the data are licensed under CC BY-NC-SA 3.0.

Source:

- BBBC041: <https://bbbc.broadinstitute.org/BBBC041>

Possible extension:

- Use the cell annotations to build a small object-detection or cell-cropping pipeline.
- Compare a classifier trained on isolated crops with one trained on a different image source.
- Study how performance changes across contributing sources.

Important limitation: the non-commercial license makes BBBC041 less suitable for presenting the project as a commercially reusable dataset. Use it as an optional research extension and preserve its license notice.

### Option B: BBBC021 compound-profiling images

BBBC021 contains human MCF-7 cells treated with small molecules and imaged in three channels: DNA, F-actin, and beta-tubulin. It includes metadata and mechanism-of-action labels for a subset of compound-concentration combinations. The official page lists 39,600 TIFF image files and provides metadata and ground-truth files.

Source:

- BBBC021: <https://bbbc.broadinstitute.org/BBBC021>

Possible extension:

- Build a multi-channel image loader.
- Compare single-channel and three-channel models.
- Predict broad mechanism-of-action categories.
- Use leave-one-compound-out validation, rather than randomly splitting individual images.

Important limitation: the dataset page identifies AstraZeneca Pharmaceuticals as the copyright holder. Read and comply with the dataset terms before redistributing images or using the data for any purpose beyond the permitted use.

### Option C: BBBC022 Cell Painting images

BBBC022 contains U2OS-cell Cell Painting images across five channels and includes metadata for compounds, plates, wells, sites, concentrations, and image paths. The official page describes a large dataset with 345,600 16-bit TIFF files.

Source:

- BBBC022: <https://bbbc.broadinstitute.org/BBBC022>

Possible extension:

- Build a metadata-first sample selector.
- Compare channel combinations.
- Compute image embeddings and visualize them with PCA or UMAP.
- Search for morphological similarity between compound treatments.

This is an advanced extension. Do not download the complete archive until the core application works. Use a small, documented subset selected by plate, well, channel, or compound.

## Recommended technical stack

### Core language

- **Python 3.10 or newer**
- Use a virtual environment with `venv`.
- Record the exact Python version in the README.

### Data handling and scientific computing

- **NumPy** for numerical arrays
- **pandas** for metadata and tabular analysis
- **Pillow** for image loading and preprocessing
- **scikit-image** for image-processing utilities when needed
- **Matplotlib** and **Seaborn** for publication-quality analysis figures

### Machine learning

- **scikit-learn** for classical baselines, preprocessing, metrics, confusion matrices, and cross-validation
- **PyTorch** for the neural-network model
- **torchvision** for image transforms and transfer-learning models
- **joblib** for saving classical models and preprocessing objects

### Application layer

- **Streamlit** for the first user interface
- Optional later alternative: **FastAPI** for a prediction API
- Optional later front end: plain HTML/JavaScript or a small React interface, only if you want to emphasize full-stack development

### Explainability

Use one of the following:

- Grad-CAM or a comparable class-activation-map approach for CNN explanations
- Integrated gradients or saliency maps
- A simple probability chart and nearest-example comparison as a baseline explanation

Start with the probability chart and correctly classified/incorrectly classified examples. Add Grad-CAM only after the prediction pipeline is stable.

### Testing and code quality

- **pytest** for unit and integration tests
- **ruff** for linting and formatting
- Optional: **mypy** for gradual type checking
- Optional: **pre-commit** for local quality checks

### Reproducibility and experiment tracking

Start with simple, transparent files:

- YAML or JSON configuration files
- Fixed random seeds
- CSV or JSON metric exports
- Saved confusion matrices and training curves

Optional later tools:

- **MLflow** for experiment tracking
- **Weights & Biases** if you are comfortable using an external service

Do not make experiment-tracking software a prerequisite for the first version. Clear CSV results and a reproducible command are enough.

### Packaging and deployment

Required for the first version:

- `requirements.txt`
- `README.md`
- Local Streamlit command
- Clear data-download instructions

Optional:

- Streamlit Community Cloud or another permitted hosting option
- GitHub Actions for tests
- Docker

## Suggested architecture

```text
                 +----------------------+
                 | Public dataset      |
                 | BloodMNIST           |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Data preparation     |
                 | validation, splits,  |
                 | transforms, metadata|
                 +----------+-----------+
                            |
                            v
          +-----------------+-----------------+
          |                                   |
          v                                   v
+---------------------+             +----------------------+
| Baseline models     |             | PyTorch model        |
| HOG + SVM /        |             | small CNN or         |
| logistic regression|             | transfer learning   |
+----------+----------+             +----------+-----------+
           |                                   |
           +-----------------+-----------------+
                             v
                 +----------------------+
                 | Evaluation and      |
                 | error analysis       |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Model artifact and   |
                 | class mapping        |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Streamlit review     |
                 | interface            |
                 +----------------------+
```

## Recommended repository structure

```text
cellscope/
├── README.md
├── LICENSE
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── baseline.yaml
│   └── cnn.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_baseline_models.ipynb
│   └── 03_error_analysis.ipynb
├── src/
│   └── cellscope/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── preprocessing.py
│       ├── models.py
│       ├── train.py
│       ├── evaluate.py
│       ├── predict.py
│       ├── explain.py
│       └── provenance.py
├── app/
│   └── streamlit_app.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_data_validation.py
│   ├── test_prediction.py
│   └── test_metrics.py
├── reports/
│   ├── figures/
│   ├── metrics/
│   └── model_card.md
└── scripts/
    ├── download_data.py
    ├── train_baseline.py
    └── evaluate_model.py
```

Keep notebooks for exploration, but move reusable logic into `src/`. This is an important distinction between a portfolio project that demonstrates software engineering and a collection of disconnected notebooks.

## Functional requirements

### Data ingestion

The project must:

- Download or load BloodMNIST through the official MedMNIST package.
- Identify the selected split explicitly.
- Convert labels into readable class names.
- Validate that images have the expected number of channels and dimensions.
- Record the data source and license.
- Avoid silently mixing training, validation, and test data.

### Data exploration

Create a dataset-audit report containing:

- Number of samples per split
- Number of samples per class
- Example image grid for every class
- Pixel-value distribution
- Image dimensions and channel count
- Class imbalance assessment
- Duplicate or near-duplicate checks where practical
- Any preprocessing performed before training

### Baseline model

Start with a classical baseline before training a deep-learning model. Suitable choices are:

1. Logistic regression on flattened normalized pixels
2. Random forest on flattened or engineered features
3. HOG features with a support-vector machine

The baseline is valuable because it gives you a reference point. A neural network should not be described as successful merely because it produces a high score; it should improve on a meaningful baseline under the same evaluation conditions.

### Neural-network model

Implement a small CNN first:

- Input normalization
- Two or three convolution blocks
- ReLU or another simple activation
- Pooling
- Dropout where justified
- Fully connected classification head
- Cross-entropy loss
- Early stopping or a fixed training budget

Only after the baseline CNN works should you add transfer learning. For larger image sizes, compare a lightweight pretrained model such as ResNet-18 or EfficientNet-B0.

### Prediction behavior

For every prediction, retain:

- Predicted class
- Probability for every class
- Top-1 confidence
- Top-3 predicted classes
- Model version
- Preprocessing version
- Inference time
- Optional explanation image

Do not show confidence as if it were a probability of clinical truth. Label it as model confidence or predicted probability and explain that model confidence can be poorly calibrated.

### Review interface

The Streamlit interface should contain these views:

#### Home

- Project purpose
- Dataset citation
- Intended use and limitations
- Link to the model card

#### Single-image prediction

- Image preview
- Predicted class
- Top-3 predictions
- Probability bar chart
- Explanation image if available
- Inference time
- Warning if the input does not match expected image properties

#### Dataset explorer

- Class selector
- Random sample gallery
- Image metadata where available
- Correct versus incorrect sample examples

#### Model evaluation

- Overall accuracy and balanced accuracy
- Macro and weighted precision, recall, and F1
- Confusion matrix
- Per-class metrics
- Confidence distribution
- Example errors

#### Model comparison

- Baseline versus CNN metrics
- Training time
- Inference time
- Number of parameters, if available
- Model-size comparison

## Evaluation plan

### Metrics

Report more than accuracy, especially if class frequencies are not equal:

- Accuracy
- Balanced accuracy
- Macro-averaged precision
- Macro-averaged recall
- Macro-averaged F1 score
- Weighted F1 score
- Per-class precision and recall
- Confusion matrix
- One-vs-rest ROC-AUC where appropriate
- Calibration curve or reliability diagram
- Inference time per image

### Validation discipline

Use the official train, validation, and test splits where possible. Do not tune hyperparameters on the test set.

The workflow should be:

1. Use the training set to fit models.
2. Use the validation set to select preprocessing, architecture, and hyperparameters.
3. Lock the final configuration.
4. Evaluate once on the test set.
5. Record the final result in a dated experiment report.

### Error analysis

Do not stop after generating a score. Inspect:

- The most confused class pairs
- High-confidence incorrect predictions
- Low-confidence correct predictions
- Images with unusual brightness or background
- Performance by class
- Whether errors appear related to morphology, image quality, or acquisition differences

Create a table with columns such as:

| Image identifier | True class | Predicted class | Confidence | Error category | Notes |
|---|---|---|---:|---|---|
| Example identifier | True label | Predicted label | 0.00 | Visual ambiguity, low quality, or possible label issue | Short observation |

Replace the example row with real observations once the analysis is complete; do not publish fabricated results.

### Calibration and uncertainty

Add a practical uncertainty feature without claiming that it solves uncertainty completely:

- Define a low-confidence threshold using the validation set.
- Mark predictions below that threshold as “Review recommended.”
- Compare error rates above and below the threshold.
- Plot confidence histograms for correct and incorrect predictions.
- Explain that softmax confidence is not automatically calibrated.

An optional advanced extension is temperature scaling on the validation set.

## Data leakage risks to discuss

Include a dedicated README section explaining potential leakage:

- Duplicate or near-duplicate images across splits
- Augmented copies accidentally appearing in both training and test sets
- Metadata-derived labels being used as model features
- Selecting images after inspecting the test results
- Tuning the decision threshold on the test set
- Reporting only the best experiment

For any optional BBBC extension, use the experimental unit correctly. For example, if multiple images come from the same well, plate, compound, or specimen, do not randomly distribute related images across train and test sets without discussing the consequence.

## Explainability plan

### Minimum version

The first version should show:

- Top-3 class probabilities
- A confidence warning
- Similar examples from the training set
- Confusion-matrix-based explanation of common mistakes

### Advanced version

Add Grad-CAM or another saliency method to produce a heatmap over the image. The interface should state that a heatmap is an indication of image regions associated with the model output, not proof that the highlighted region is biologically meaningful.

Evaluate explanations qualitatively:

- Are heatmaps concentrated on the cell rather than image borders?
- Do they change substantially under small image transformations?
- Do they appear different for correct and incorrect predictions?
- Are there signs that the model is using staining artifacts or background patterns?

## Data provenance and governance

Create `data/README.md` with:

- Dataset name
- Official URL
- Version or release identifier
- Date accessed
- License
- Attribution requirements
- Download method
- Expected files or package behavior
- Local storage location
- Whether image files are excluded from version control
- Any transformations applied
- Checksum or file-size information where practical

Create `reports/model_card.md` with:

- Model purpose
- Intended users
- Intended use
- Out-of-scope uses
- Training data
- Evaluation data
- Preprocessing
- Model architecture
- Metrics
- Known limitations
- Ethical and safety considerations
- Reproducibility instructions
- Dataset and paper citations

## Step-by-step implementation plan

### Phase 0: Define the project contract

**Goal:** establish the scope before writing model code.

Tasks:

- Create the repository.
- Write the project question.
- Select BloodMNIST as the primary dataset.
- Record the dataset source and license.
- Define the first success criteria.
- Create a minimal `README.md`.

Success criteria:

- Another person can understand what the project does.
- The README says clearly that the project is not for clinical diagnosis.
- The repository contains no employer code, internal information, or confidential data.

### Phase 1: Environment and data loading

**Goal:** load the official data reproducibly.

Tasks:

- Create a virtual environment.
- Install the initial dependencies.
- Load BloodMNIST using the official package.
- Print shapes and labels.
- Create a class-name mapping.
- Save a small data-audit report.

Suggested initial installation:

```bash
python -m venv .venv
```

Windows activation:

```bash
.venv\Scripts\activate
```

macOS/Linux activation:

```bash
source .venv/bin/activate
```

Initial packages:

```bash
python -m pip install --upgrade pip
python -m pip install medmnist numpy pandas pillow matplotlib seaborn scikit-learn torch torchvision streamlit pytest ruff
python -m pip freeze > requirements.txt
```

If a package installation differs by operating system, document the working command in the README instead of hiding the difference.

Success criteria:

- The dataset loads from a clean environment.
- The class distribution is printed.
- An image grid is generated.
- The data source is documented.

### Phase 2: Dataset audit

**Goal:** understand the data before training.

Tasks:

- Count samples per class and split.
- Plot the class distribution.
- Display representative images.
- Inspect pixel ranges.
- Check missing or malformed records.
- Check for duplicate identifiers.
- Write a short interpretation of the dataset limitations.

Success criteria:

- The audit is saved under `reports/`.
- Every plot has a title, axis labels, and a clear legend where appropriate.
- The report distinguishes observations from assumptions.

### Phase 3: Classical baselines

**Goal:** create trustworthy reference points.

Tasks:

- Normalize inputs.
- Train logistic regression.
- Train HOG plus SVM if practical.
- Record training time.
- Evaluate on validation data.
- Save metrics and confusion matrices.

Success criteria:

- Results are reproducible with a fixed seed.
- The baseline is evaluated using the same test protocol later used for the CNN.
- The README explains why a baseline is included.

### Phase 4: Small CNN

**Goal:** learn the complete deep-learning workflow.

Tasks:

- Create a PyTorch dataset wrapper.
- Define the model.
- Add training and validation loops.
- Save the best validation checkpoint.
- Plot training and validation loss.
- Add early stopping or a fixed training budget.
- Evaluate once on the test set after finalizing the model.

Success criteria:

- Training can be run from a command or script, not only from a notebook.
- The model checkpoint includes the class mapping and preprocessing configuration.
- Training and evaluation metrics are saved.

### Phase 5: Error analysis and uncertainty

**Goal:** demonstrate scientific judgment.

Tasks:

- Generate a confusion matrix.
- Identify the most common confusion pairs.
- Save examples of high-confidence errors.
- Add confidence histograms.
- Define a validation-based “Review recommended” rule.
- Write a short error-analysis report.

Success criteria:

- You can explain at least three failure patterns using evidence from the results.
- The application does not imply that confidence equals correctness.
- The report includes limitations, not only successes.

### Phase 6: Streamlit application

**Goal:** turn the analysis into a usable product.

Tasks:

- Build a home page.
- Add single-image inference.
- Add probability charts.
- Add a dataset explorer.
- Add model evaluation visuals.
- Add a model-information page.
- Add input validation and readable error messages.

Run locally with:

```bash
streamlit run app/streamlit_app.py
```

Success criteria:

- A user can launch the application using one documented command.
- A malformed input produces an actionable message.
- The interface shows dataset attribution and limitations.
- The model output is understandable without opening the source code.

### Phase 7: Testing and quality

**Goal:** make the repository look like a real software project.

Add tests for:

- Image normalization
- Label-to-class-name conversion
- Expected image shape
- Invalid file handling
- Probability values summing approximately to one
- Prediction output schema
- Empty dataset behavior
- Metric calculation on a small known example
- Model loading and inference

Run:

```bash
pytest
ruff check .
ruff format --check .
```

Success criteria:

- Tests run from a clean environment.
- No test depends on an unavailable employer system.
- The application logic can be tested without launching the full UI.

### Phase 8: Reproducibility and documentation

**Goal:** make the project easy to review.

The README should include:

- Project overview
- Screenshot or short screen recording
- Architecture diagram
- Setup instructions
- Dataset download instructions
- Training instructions
- Evaluation methodology
- Results table
- Error-analysis summary
- Limitations
- License and attribution
- Future work

Create a small results table with measured values only:

| Model | Input | Balanced accuracy | Macro F1 | Test accuracy | Inference time | Notes |
|---|---|---:|---:|---:|---:|---|
| Logistic regression | 28 × 28 RGB | Measured result | Measured result | Measured result | Measured result | Baseline |
| HOG + SVM | HOG features | Measured result | Measured result | Measured result | Measured result | Classical CV baseline |
| Small CNN | 28 × 28 RGB | Measured result | Measured result | Measured result | Measured result | Final lightweight model |

Do not fill this table with invented numbers. The value of the table is that it reports your actual experimental results.

## Optional extensions by career target

### For biopharma and pharma applications

Add:

- Dataset provenance and version tracking
- Experimental-unit-aware splitting
- Model card
- Class-imbalance discussion
- Quality-control checks
- Human-review workflow
- Reproducible analysis report
- Clear distinction between research use and clinical use
- Optional multi-channel microscopy extension using BBBC data

The strongest message is scientific responsibility, not simply a large accuracy value.

### For data-analytics applications

Add:

- Dataset-audit dashboard
- Data-quality metrics
- Distribution-shift visualizations
- Per-class trend and error analysis
- Calibration plots
- A concise executive-style findings page
- Exportable CSV of prediction results

### For machine-learning applications

Add:

- Baseline comparison
- Hyperparameter experiment table
- Ablation study
- Calibration
- Explainability
- Error taxonomy
- Reproducibility command
- Model-size and latency comparison

### For Python/backend applications

Add:

- A clean service layer separate from Streamlit UI
- Typed prediction request and response objects
- Unit and integration tests
- Structured application logging
- Model loading only once at application start
- Input-size and file-type validation
- Optional FastAPI endpoint

### For general IT and full-stack applications

Add:

- User-friendly empty states
- Clear error messages
- Responsive layout
- Search and filtering for the image explorer
- Downloadable prediction report
- Version information in the UI
- Optional database for storing local review records

Do not add every extension. Choose the two or three that support the role you are targeting.

## Stretch goal: optional FastAPI service

Only attempt this after the Streamlit version works.

Create an endpoint such as:

```text
POST /predict
```

The endpoint can accept an image and return:

```json
{
  "predicted_class": "lymphocyte",
  "top_predictions": [
    {"class": "lymphocyte", "probability": 0.72},
    {"class": "monocyte", "probability": 0.14},
    {"class": "neutrophil", "probability": 0.08}
  ],
  "review_recommended": false,
  "model_version": "documented-version",
  "inference_time_ms": 0
}
```

Replace illustrative values with real values in the implementation. Add request-validation tests and keep the service local unless you specifically want to learn deployment.

## Stretch goal: model monitoring simulation

Because production ML systems need monitoring, add a local simulation rather than connecting to confidential production data.

Generate or select an evaluation batch and calculate:

- Input image-size distribution
- Pixel-intensity distribution
- Class-prediction distribution
- Confidence distribution
- Percentage of low-confidence predictions
- Percentage of invalid inputs

Compare these values between two batches and display a warning when a distribution changes substantially. Clearly label this as a demonstration of monitoring concepts, not proof of real-world drift detection.

## What not to do

Avoid these common portfolio weaknesses:

- Do not present a single accuracy value without a baseline.
- Do not randomly split related images when the dataset has repeated experimental units.
- Do not call the application a diagnostic tool.
- Do not use employer data, code, screenshots, internal labels, or confidential documentation.
- Do not commit large raw datasets to the repository.
- Do not spend weeks learning Docker before completing the core project.
- Do not build a complex React front end before the model and evaluation are trustworthy.
- Do not include generated or fabricated metrics.
- Do not hide incorrect predictions.
- Do not claim explainability when the system only displays a confidence score.

## Portfolio presentation package

Publish these items:

1. GitHub repository
2. README with setup and results
3. Two or three screenshots
4. A short screen recording of the application
5. Dataset and model card
6. One concise technical write-up
7. Optional hosted demo, if it can be done safely and within available resources

Recommended screenshots:

- Home page with project purpose and limitations
- Single-image prediction with probability chart
- Confusion matrix and error-analysis view
- Optional heatmap explanation

## Suggested README opening

> CellScope is an educational biomedical computer-vision project for classifying microscopy images and presenting model predictions through an explainable review interface. It compares classical machine-learning baselines with a lightweight neural network, evaluates performance using class-aware metrics, and highlights uncertain and incorrect predictions. The project is not intended for clinical diagnosis or treatment decisions.

## Suggested interview explanation

A strong two-minute explanation would be:

> I wanted a project that connected my life-science background with software and machine learning without repeating the bioprocess-monitoring work from my job. I built CellScope using public microscopy data. I started with a dataset audit and classical baselines, then trained a small CNN and compared the models using balanced metrics, confusion matrices, calibration, and error analysis. I packaged the final workflow in a local Streamlit application that shows predictions, probabilities, and review warnings. The most important part was treating validation and data provenance seriously: I documented the dataset license, separated test evaluation from model selection, and made the system explicit about its educational—not clinical—scope.

## Recommended first milestone

Complete only the following before adding any advanced feature:

- BloodMNIST loads from a clean virtual environment.
- Dataset audit is complete.
- Logistic-regression baseline is evaluated.
- Small CNN is trained and evaluated.
- Confusion matrix and error examples are saved.
- Streamlit can display one prediction.
- README explains setup, data source, metrics, and limitations.

Once that milestone works, choose exactly two extensions:

1. Explainability and uncertainty
2. Automated tests
3. FastAPI service
4. Optional BBBC dataset
5. Model monitoring simulation
6. GitHub Actions

For your goals, the best combination is:

- **Explainability and uncertainty** for biopharma credibility
- **Automated tests** for general IT credibility

## Definition of done

CellScope is complete when:

- The repository can be cloned and run without Docker.
- The dataset source, license, and attribution are documented.
- The data-audit report exists.
- At least one classical baseline and one neural model are compared.
- Test-set evaluation is performed only after model selection.
- Metrics include balanced and per-class results.
- The main error patterns are documented.
- The application shows predictions, probabilities, and limitations.
- Automated tests cover important non-UI behavior.
- No confidential employer information is used.
- The README contains screenshots, setup instructions, and a short technical summary.

## Source links

- MedMNIST official site: <https://medmnist.com/>
- MedMNIST official repository: <https://github.com/MedMNIST/MedMNIST>
- MedMNIST v2 publication: <https://doi.org/10.1038/s41597-022-01721-8>
- Broad Bioimage Benchmark Collection catalogue: <https://bbbc.broadinstitute.org/image_sets>
- BBBC041 malaria blood-smear dataset: <https://bbbc.broadinstitute.org/BBBC041>
- BBBC021 MCF-7 compound-profiling dataset: <https://bbbc.broadinstitute.org/BBBC021>
- BBBC022 Cell Painting dataset: <https://bbbc.broadinstitute.org/BBBC022>

Always revisit the official source pages before publication because dataset access instructions, metadata, and licensing details can change.
