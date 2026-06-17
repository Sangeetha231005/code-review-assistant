# 🛡️ AI-Driven Automated Code Review Assistant

An **AI-powered automated code review system** that detects **security vulnerabilities, unsafe logic flows, and code quality issues** in source code during development and pull requests.

This project combines **static analysis**, **AUG-PDG (Augmented Program Dependence Graphs)**, and a **fine-tuned CodeBERT model**, and integrates seamlessly with **GitHub Actions** to enforce security and quality checks before code is merged.

---

## 🚀 Key Features

### 🔍 Multi-Language Support (6 Languages)
- Python
- Java
- JavaScript
- PHP
- Ruby
- Go

### 🧠 AI-Based Vulnerability Detection
- Fine-tuned **CodeBERT** (`microsoft/codebert-base`)
- Encoder **frozen** for stable logic learning
- Learns **concrete vulnerability flow patterns**

### 🧩 AUG-PDG (Augmented Program Dependence Graph)
- Tracks **SOURCE → SINK → SANITIZATION**
- Enables semantic reasoning beyond regex matching

### 🔐 Security Vulnerabilities Detected
- SQL Injection
- Command Injection
- Cross-Site Scripting (XSS)
- Unsafe system calls
- Missing input sanitization

### 📐 Static Code Analysis
- Pylint
- Flake8
- Bandit
- Language-specific linters

### 🔁 CI/CD Integration
- GitHub Actions workflow
- Runs automatically on `push` and `pull_request`

### 📊 Explainable Decisions
- Clear reasoning for **APPROVE / REVIEW / REJECT**

---

## 📂 Dataset

The vulnerability detection model was trained using the **Devign Dataset**, a widely used benchmark dataset for software vulnerability detection research.

### 🔗 Dataset Source

- **Devign Dataset:** https://huggingface.co/datasets/DetectVul/devign

### 📊 Dataset Overview

| Property | Value |
|-----------|---------|
| Dataset | Devign |
| Domain | Software Vulnerability Detection |
| Source | Real-World Open-Source Projects |
| Labels | Vulnerable / Non-Vulnerable |
| Primary Language | C/C++ |
| Task | Function-Level Vulnerability Classification |
| Total Samples | 27,318 | | Dataset Size | 80.1 MB |

---

## 🏗️ System Architecture
<img width="1688" height="811" alt="image" src="https://github.com/user-attachments/assets/b35daed9-a915-4257-a280-fc46d7f27820" />

```text
Source Code
   ↓
Language Detection
(Python | Java | JavaScript | PHP | Ruby | Go)
   ↓
Style & Code Quality Analysis
(Formatting • Readability • Standards)
   ↓
Rule-Based Security Scanning
(Known Vulnerability Patterns)
   ↓
AST Parsing
(Tree-sitter)
   ↓
AUG-PDG Construction
(Control • Data • Call Dependencies)
   ↓
Taint-Aware Data-Flow Extraction
(SOURCE → SINK → SANITIZATION)
   ↓
Flow Normalization
(Standardized Vulnerability Logic)
   ↓
AI Vulnerability Analysis
(Fine-Tuned CodeBERT)
   ↓
Decision Engine
(APPROVE • REVIEW RECOMMENDED • REVIEW REQUIRED • REJECT)
   ↓
GitHub Actions CI Enforcement

```

---

## 🤖 Machine Learning Model

- **Base Model:** microsoft/codebert-base  
- **Training:** Encoder Frozen  
- **Epochs:** 2  
- **Execution:** CPU compatible (GPU optional)

### Model Input Format
```text
[VULNERABILITY_FLOW]
SOURCE: <source>
SINK: <sink>
SANITIZATION: <sanitization>
```

---

## 📁 Project Structure

```text
code-review-assistant/
├── src/
│   ├── __init__.py
│   ├── aug_pdg.py
│   ├── input_processor.py
│   ├── integration.py
│   ├── language_detector.py
│   ├── linter_runner.py
│   ├── security_scanner.py
│   ├── style_checker.py
│   └── vulnerability_training.py
│
├── models/
│   ├── vulnerability_logic_model/
│   └── vulnerability_logic_production/
│
├── test_integration.py
├── requirements.txt
├── .gitignore
├── .github/
│   └── workflows/
│       └── code-review.yml
│
└── README.md

```

---

## ⚙️ Installation

```bash
git clone https://github.com/Sangeetha231005/code-review-assistant.git
cd code-review-assistant
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python test_integration.py
```

---

## 🔄 GitHub Actions

- Automatically runs on **push** and **pull_request**
- Blocks merge on **critical vulnerabilities**

---

## 📊 Performance Metrics

The vulnerability detection model was evaluated on a held-out test set
containing concrete AUG-PDG patterns.

| Metric       | Value |
|--------------|-------|
| Accuracy     | 100%  |
| Precision    | 100%  |
| Recall       | 100%  |
| F1-Score     | 100%  |

> Note: These results reflect performance on **concrete vulnerability flow
patterns** that closely match AUG-PDG extraction output.  
Real-world performance may vary depending on code complexity and language usage.


