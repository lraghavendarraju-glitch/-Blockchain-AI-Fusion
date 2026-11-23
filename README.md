[![DOI](https://zenodo.org/badge/1468498632.svg)](https://doi.org/10.5281/zenodo.24460254)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/lraghavendarraju-glitch/-Blockchain-AI-Fusion)

# Blockchain-AI Fusion for Secure and Transparent Data Science Workflows

## Overview

Blockchain-AI Fusion is an advanced framework that integrates Artificial Intelligence, Deep Learning, Blockchain Technology, and Cryptographic Provenance to improve the security, integrity, transparency, and reliability of data science workflows.

The framework combines workflow analysis, graph-based learning, anomaly detection, trust evaluation, cryptographic verification, and blockchain-based governance to identify suspicious activities and maintain secure workflow execution.

The proposed ProvSeqGAT architecture integrates Bidirectional Gated Recurrent Units (BiGRU) and Graph Attention Networks (GATv2) to analyze sequential workflow activities and dependency relationships.

The system also supports Hyperledger Fabric integration for permissioned blockchain-based workflow governance.

## Key Features

- Secure and transparent data science workflow management.
- AI-based workflow anomaly detection.
- Deep learning-based sequential and graph analysis.
- ProvSeqGAT architecture using BiGRU and GATv2.
- Cryptographic provenance verification.
- Digital signatures and integrity commitments.
- Workflow dependency graph analysis.
- Multiple workflow attack detection mechanisms.
- Actor-based trust evaluation.
- Adaptive validation and access governance.
- Automated audit reconstruction.
- Workflow-level and step-level anomaly detection.
- Baseline model comparisons.
- Statistical performance evaluation.
- Detailed visualization and reporting.
- Hyperledger Fabric integration.
- Secure artifact storage using AES-GCM encryption.

## System Architecture

The framework consists of the following major components:

1. **Workflow Management:** Processes workflow activities, execution sequences, and dependency relationships.

2. **Cryptographic Provenance:** Maintains integrity commitments and verifies workflow artifacts using cryptographic mechanisms.

3. **Blockchain Governance:** Supports permissioned workflow validation and governance through Hyperledger Fabric.

4. **ProvSeqGAT Model:** Combines BiGRU and GATv2 to learn temporal and structural workflow representations.

5. **Anomaly Detection:** Identifies suspicious workflow behavior through workflow-level and step-level prediction.

6. **Trust Evaluation:** Analyzes actor history and integrates trust information into workflow validation.

7. **Audit and Verification:** Reconstructs workflow execution records and supports integrity verification.

8. **Performance Evaluation:** Compares multiple detection approaches and produces evaluation reports and visualizations.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python 3.10+ |
| Deep Learning | PyTorch |
| Sequence Learning | BiGRU |
| Graph Learning | GATv2 |
| Machine Learning | Scikit-learn, XGBoost |
| Graph Processing | NetworkX |
| Cryptography | Python Cryptography |
| Blockchain | Hyperledger Fabric |
| Backend Integration | Python, Go |
| Optimization | Optuna |
| Data Processing | NumPy |
| Visualization | Matplotlib |
| Testing | Pytest |

## Model Architecture: ProvSeqGAT

ProvSeqGAT combines temporal sequence learning and graph attention mechanisms for workflow security analysis.

### Main Components

- **Feature Representation:** Extracts 40 numerical workflow-step features and incorporates step-type and certified-role embeddings.
- **Feature Projection:** Uses a linear transformation, Layer Normalization, and GELU activation.
- **BiGRU:** Captures forward and backward temporal dependencies in workflow execution.
- **GATv2:** Learns relationships between connected workflow nodes using graph attention.
- **Graph Pooling:** Aggregates node representations into workflow-level features.
- **Workflow Prediction Head:** Estimates the probability of suspicious workflow activity.
- **Step Prediction Head:** Identifies potentially compromised workflow steps.
- **Trust Fusion:** Incorporates actor-history information into workflow evaluation.

### Learning and Optimization

- AdamW optimizer.
- Cosine learning-rate scheduling with warm-up.
- Dropout regularization.
- Gradient clipping.
- Validation-based early stopping.
- Workflow-level and step-level binary classification losses.

## Baseline Models

The framework supports comparisons with the following approaches:

| Model | Description |
|---|---|
| RuleOnly | Rule-based workflow violation detection |
| Isolation Forest | Unsupervised anomaly detection |
| One-Class SVM | Normal-behavior boundary detection |
| XGBoost | Supervised workflow classification |
| LSTM Autoencoder | Sequential reconstruction-based detection |
| DeepLog | Sequential event prediction |
| ProvenanceGraph IF Proxy | Graph-structure-based anomaly detection |

## Project Structure

```text
Blockchain-AI-Fusion/
│
├── configs/
│   └── experiment.yaml
│
├── docs/
│   ├── METHODOLOGY_MAP.md
│   └── VALIDATION.md
│
├── examples/
│   └── Sample files and Colab quickstart
│
├── fabric/
│   ├── chaincode/
│   │   └── wfgov/
│   ├── gateway/
│   ├── network/
│   └── README.md
│
├── src/
│   └── blockai/
│       ├── Core application modules
│       └── Command-line interface
│
├── tests/
│   └── Automated tests
│
├── .gitignore
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```

## System Requirements

- Python 3.10 or later.
- PyTorch 2.1 or later.
- NumPy.
- Scikit-learn.
- NetworkX.
- Cryptography.
- Git.

Additional dependencies may be required for optional experiments and blockchain integration.

Supported environments:

- Windows
- Linux
- macOS
- Google Colab

A CUDA-compatible GPU can accelerate model training, but CPU execution is also supported.

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/lraghavendarraju-glitch/-Blockchain-AI-Fusion.git
```

### Step 2: Navigate to the Project Directory

```bash
cd -Blockchain-AI-Fusion
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### Step 4: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 5: Install Dependencies

For the complete installation:

```bash
python -m pip install -e ".[experiments,tests]"
```

For the basic installation:

```bash
python -m pip install -e .
```

### Step 6: Verify Installation

```bash
python -m blockai --help
```

## Running the Application

### 1. Run the Initial System Test

```bash
python -m blockai smoke --output results/smoke
```

This executes the main workflow processing, model training, evaluation, trust scoring, ledger operations, and visualization pipeline.

### 2. Run a Quick Test

```bash
python -m blockai smoke --n 72 --epochs 2 --skip-baselines --output results/quick
```

### 3. Execute Workflow Processing

```bash
python -m blockai generate --n 10000 --seed 42 --output datasets/workflows.jsonl.gz
```

### 4. Execute the Main Experiment

```bash
python -m blockai experiment --input datasets/workflows.jsonl.gz --config configs/experiment.yaml --seed 42 --protocol random --baselines --output results/main
```

### 5. Run Model Ablation

```bash
python -m blockai ablation --input datasets/workflows.jsonl.gz --config configs/experiment.yaml --output results/ablations
```

### 6. Perform Hyperparameter Optimization

```bash
python -m blockai tune --input datasets/workflows.jsonl.gz --config configs/experiment.yaml --trials 40 --output results/optuna
```

### 7. Perform Repeated Evaluation

```bash
python -m blockai repeats --config configs/experiment.yaml --baselines --output results/repetitions
```

## Blockchain Integration

The framework supports two ledger backends:

### Local Ledger

The local ledger supports signed provenance records, integrity commitments, and append-only workflow transaction logging.

Run:

```bash
python -m blockai ledger --backend simulated --limit 50 --input datasets/workflows.jsonl.gz --output results/ledger
```

### Hyperledger Fabric

The framework provides:

- Go-based Fabric chaincode.
- Fabric Gateway REST bridge.
- Identity and certificate-based authentication.
- Permissioned workflow governance.
- Private data collection support.
- Secure transaction handling.

To start the provided Fabric test network:

```bash
./fabric/network/start-test-network.sh
```

This integration requires Docker, Hyperledger Fabric tools, network configuration, and valid certificates.


## Provenance and Audit Management

The framework provides commands for managing workflow provenance and auditing.

### Workflow Audit

```bash
python -m blockai audit --limit 10 --input datasets/workflows.jsonl.gz --model results/main/model --output results/audit.json
```

### Secure Artifact Storage

```bash
python -m blockai artifact-store --input datasets/workflows.jsonl.gz --limit 10 --output results/artifacts --key-file results/secrets/artifact.key
```

### Governance Validation

```bash
python -m blockai governance-demo --input datasets/workflows.jsonl.gz --model results/main/model --auto-audit --output results/governance.json
```

## External Dataset Integration

The framework provides adapters for external workflow sources.

### WfCommons

Source: https://github.com/wfcommons/WfInstances

Download workflow data:

```bash
python -m blockai download --dataset wfcommons --limit 20 --output datasets/wfcommons
```

Process the dataset:

```bash
python -m blockai external --source wfcommons --input datasets/wfcommons --model results/main/model --inject-attacks --output results/external_wfcommons
```

### OpenML

Source: https://www.openml.org

Download dataset information:

```bash
python -m blockai download --dataset openml --limit 50 --output datasets/openml
```

Process the dataset:

```bash
python -m blockai external --source openml --input datasets/openml --model results/main/model --inject-attacks --output results/external_openml
```

## Model Evaluation

The framework evaluates workflow detection performance using multiple machine learning and deep learning approaches.

Evaluation outputs include:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC curves.
- Precision-Recall curves.
- Confusion matrices.
- Per-attack recall.
- Workflow-length analysis.
- Comparative model performance.
- Statistical evaluation reports.

The evaluation pipeline uses separate training, validation, and testing data to support reliable performance assessment.

## Output and Results

The application stores execution outputs in the results directory.

Typical outputs include:

- Trained model checkpoints.
- Workflow prediction results.
- Anomaly detection reports.
- Trust evaluation results.
- Audit records.
- Provenance information.
- Model evaluation metrics.
- Performance comparison charts.
- Statistical analysis reports.

Results are saved according to the selected experiment configuration and output directory.

## Testing

Run the automated test suite using:

```bash
pytest -q
```

To verify the command-line interface:

```bash
python -m blockai --help
```

## Security Considerations

- Cryptographic keys must be stored securely.
- Private certificates must not be committed to GitHub.
- Authentication tokens must be protected.
- Blockchain transactions require proper identity and authorization configuration.
- External datasets must retain their original licenses and attribution.
- Local ledger operations and Hyperledger Fabric transactions are separate implementations.


## Acknowledgements

The project uses open-source technologies and frameworks, including PyTorch, Scikit-learn, NetworkX, Python Cryptography, and Hyperledger Fabric.


## Repository

**Project Name:** Blockchain-AI Fusion for Secure and Transparent Data Science Workflows

**GitHub Repository:**

https://github.com/lraghavendarraju-glitch/-Blockchain-AI-Fusion

---
