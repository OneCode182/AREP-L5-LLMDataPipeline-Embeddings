# Project Guide: Embeddings and Text Processing

Follow the excellent guide found in Chapter 2 of *Build a Large Language Model (From Scratch)* by Sebastian Raschka.

## 1. Preparation and Resources
Download only these two files:

* **Notebook:** [ch02.ipynb](https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/ch02.ipynb)
* **Text:** [the-verdict.txt](https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt)

> **Initial Instructions:** Run the notebook end-to-end in your environment (install `torch` and `tiktoken` if needed).Deep dive into the LLM data pipeline: implementing manual tokenization, sliding window sampling for training datasets, and developing embedding layers from scratch following architectural principles of modern transformers.

---

## 2. Task: Create `embeddings.ipynb`
In your repo create an `embeddings.ipynb` notebook that contains:

### Technical Content
* **Core code:** The core code from the book (you can copy or re-run sections).
* **Explanations:** At least 4 Markdown cells with *your own explanations* on why each major step matters for LLMs / agentic systems.
* **Reflection:** Include an answer to *"Why do embeddings encode meaning, and how are they related to NN concepts?"*

### Practical Experiment
One small experiment: change the `max_length` and `stride` parameters:
1. Report how many samples you get.
2. Explain why overlap is useful.

---

## 3. Grading Part 1

| Criteria | Points |
| :--- | :--- |
| Notebook runs cleanly with outputs | 10 pts |
| Clear personal explanations | 25 pts |
| Experiment & understanding shown | 15 pts |
| **Total** | **50 pts** |

---

## 4. Submission
Submit the GitHub repo URL via the platform.

**[Add submission]**