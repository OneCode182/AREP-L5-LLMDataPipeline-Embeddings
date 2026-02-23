# Embeddings and Text Processing Architecture

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

> A rigorous deep-dive into the foundational data pipelines of Large Language Models: From raw text ingestion and BPE Tokenization to Autoregressive Sliding Windows and Continuous Embedding Projections. All built from scratch in PyTorch.

---

## Table of Contents

- [Embeddings and Text Processing Architecture](#embeddings-and-text-processing-architecture)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Core Objectives](#core-objectives)
  - [Project Structure](#project-structure)
  - [Architecture \& Data Pipeline](#architecture--data-pipeline)
  - [Setup \& Installation](#setup--installation)
  - [Notebook Walkthrough](#notebook-walkthrough)
  - [Practical Experiment Results](#practical-experiment-results)
      - [Base Case](#base-case)
      - [Experiment](#experiment)
    - [Why is overlap useful?](#why-is-overlap-useful)
    - [1. Context Density](#1-context-density)
    - [2. Semantic Continuity](#2-semantic-continuity)
  - [Theoretical Reflection](#theoretical-reflection)
    - [*Why do embeddings encode meaning, and how are they related to Neural Network concepts?*](#why-do-embeddings-encode-meaning-and-how-are-they-related-to-neural-network-concepts)
  - [References](#references)
  - [Author](#author)

---

## Overview

This repository executes a fundamental sequence in LLM design: converting discrete human language into continuous numerical tensors capable of gradient-based optimization.

Built upon the principles from *Build a Large Language Model (From Scratch)* by Sebastian Raschka, this lab explores why representations like **Byte Pair Encoding (BPE)** and **Dense Embeddings** dominate modern transformer topology over naive one-hot encoding.

### Core Objectives

1. **Lexical Analysis:** Implement state-of-the-art tokenization (`tiktoken`).
2. **Dataset Modeling:** Construct sliding-window generators for autoregressive target prediction.
3. **Semantic Spaces:** Architect dual-embedding layers (Token + Positional) directly mapping to the self-attention input bottleneck.

---

## Project Structure

```text
/
├── README.md
├── .gitignore
└── src/
    ├── embeddings.ipynb    # The main Jupyter Notebook
    ├── the-verdict.txt     # Training Corpus (Edith Wharton)
    └── requirements.txt
```

---

## Architecture & Data Pipeline

The data flows systematically from human-readable text into dense mathematical formats:

```text
                    ┌─────────────────────────────────┐
                    │      LEXICAL INGESTION           │
                    │                                  │
                    │  Raw Corpus (the-verdict.txt)    │
                    │    │                             │
                    │    ▼                             │
                    │  BPE Tokenizer (tiktoken gpt2)   │
                    │    │  (Compression & OOV bypass) │
                    │    ▼                             │
                    │  List of Discrete Token IDs      │
                    └─────────────────────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │      DATA STRUCTURING            │
                    │                                  │
                    │  Sliding Window Dataset          │
                    │    (max_length=N, stride=S)      │
                    │    │                             │
                    │    ▼                             │
                    │  (Input_Tensor, Target_Tensor)   │
                    └─────────────────────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │      VECTOR PROJECTION           │
                    │                                  │
                    │  Input Tensor (Batch, Context)   │
                    │    │                             │
                    │    ▼                             │
                    │  torch.nn.Embedding (Tokens)     │
                    │   +                              │
                    │  torch.nn.Embedding (Positional) │
                    │    │                             │
                    │    ▼                             │
                    │  Dense Semantic Tensor           │
                    │  (Batch, Context, Embed_Dim)     │
                    └─────────────────────────────────┘
```

---

## Setup & Installation

To ensure an absolutely pristine and isolated environment, execution via a Python Virtual Environment (`venv`) is strongly enforced.

<details>
<summary> <b>Linux (Arch / Ubuntu)</b></summary>

```bash
# 1. Update and install Python/pip (if not installed)
sudo pacman -S python python-pip  # For Arch Linux
# sudo apt install python3 python3-pip python3-venv  # For Ubuntu/Debian

# 2. Navigate to the project directory
cd AREP-L5/src

# 3. Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch the notebook
jupyter notebook embeddings.ipynb
```

</details>

<details>
<summary> <b>macOS</b></summary>

```bash
# 1. Install Python via Homebrew (if not installed)
brew install python

# 2. Navigate to the project directory
cd AREP-L5/src

# 3. Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch the notebook
jupyter notebook embeddings.ipynb
```

</details>

<details>
<summary> <b>Windows (PowerShell)</b></summary>

```powershell
# 1. Navigate to the project directory
cd AREP-L5\src

# 2. Create the virtual environment
python -m venv venv

# 3. Activate the virtual environment
.\venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch the notebook
jupyter notebook embeddings.ipynb
```

</details>

<details>
<summary> <b>requirements.txt</b> contents</summary>

```text
torch>=2.0.0
tiktoken>=0.5.0
jupyter>=1.0.0
nbformat>=5.0.0
```

</details>

---

## Notebook Walkthrough

The notebook `embeddings.ipynb` executes with **zero errors**. Each technical step concisely explains its core purpose for LLM pipelines.

| Section | Description | Analytical Focus |
| :------ | :---------- | :--------------- |
| **1. Raw Text Ingestion** | Loads the text corpus (`the-verdict.txt`). | Basic ingestion of the string. |
| **2. Tokenization with BPE** | Compresses text via `tiktoken`. | BPE limits vocabulary explosion and eliminates OOV errors. |
| **3. Autoregressive Datasets** | PyTorch `Dataset` with window/stride. | Structuring sequence-to-target pairs for generative training. |
| **4. Experiment: Overlap and Stride** | Runs overlap baseline and experimental setups. | Evaluates how parameters affect datasets. |
| **5. Experiment Results** | Compares `stride=4` base vs `stride=1` overlap. | Overlap ensures unbroken semantic continuity across boundaries. |
| **6. Embedding Layers** | Token + Positional `nn.Embedding`s. | Capturing relationships in dense continuous space and injecting sequence order. |
| **7. Reflection** | Meaning as geometric distance. | Analyzes how Neural Networks use backpropagation to learn meanings contextually. |

---

## Practical Experiment Results

#### Base Case

- The window size (`max_length`) was set to `4` and the stride to `4`.
- Because the window size is exactly the same as the stride, the text blocks do not overlap.
- As a result, 1286 samples were generated from the test text.

#### Experiment

- The window size (`max_length`) was kept at `4`, but the stride was changed to `1`.
- This caused the window to advance by consuming only one token at a time, reusing part of the previous context.
- As a result, 5154 samples were generated from the same text.

> [!NOTE]
> In summary, by making the stride smaller, the number of samples increased from 1200 to more than 5100 using the exact same text.

### Why is overlap useful?

There are two main reasons why setting a `stride` shorter than `max_length` (creating overlap) is crucial for LLM training:

### 1. Context Density

Training a language model requires massive amounts of data (input -> token). When using a stride of `1`, the text is being squeezed to the maximum; the model evaluates the entire transition of all possible tokens.

### 2. Semantic Continuity

An example to explain this is _The dog jumped the fence and started running quickly_. If this sentence is cut into pieces (losing quality and cohesion of its words and meaning), you might have something like ["The", "dog", "jumped", "the"] and another ["fence", "and", "started", "to"]. 

This creates the problem that the model doesn't see the words "jumped" and "fence" together (it processes them with different contexts). This is where overlap comes in, to solve this. By overlapping the windows, it is ensured that the words at the edge of the first block appear at the beginning of the next. This forces the attention mechanism to learn uninterrupted grammar and smooth word transitions, preventing the model from becoming confused when generating new text.

---

## Theoretical Reflection

### *Why do embeddings encode meaning, and how are they related to Neural Network concepts?*

Embeddings give meaning to words not through definitions, but through context. They are based on the idea that words used in similar sentences tend to mean the same thing.

Mathematically, an embedding is simply a matrix of weights (coordinates) within the first layer of the neural network. Initially, everything is random, but as the model trains by trying to predict the next word in a text, it uses backpropagation (error optimization) to push and move these coordinates. Thus, if 'computador' and 'ordenador' are used in the same context, the model's mathematics pushes their coordinates so that they end up very close in multidimensional space. Therefore, for a neural network, 'meaning' is literally a geometric distance created by the need to reduce its prediction errors.

---

## References

1. Raschka, S. *Build a Large Language Model (From Scratch)*. Chapter 2.
2. OpenAI. [Tiktoken Source Code](https://github.com/openai/tiktoken).
3. PyTorch Documentation. [`torch.nn.Embedding`](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html).

---

## Author

**Sergio Andrey Silva Rodriguez**  
*Systems Engineering Student*  
Escuela Colombiana de Ingeniería Julio Garavito