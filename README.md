# Quantifying-Predictive-Uncertainty-in-Transformer-Architectures-
Quantifying Predictive Uncertainty in Transformer Architectures via Monte Carlo Dropout for Trustworthy NLP

Abstract
Standard Transformer-based models often exhibit overconfidence in
their predictions, even when faced with out-of-distribution (OOD) or noisy
inputs. This lack of reliable uncertainty quantification poses a significant
challenge for Trustworthy AI. In this project, I propose to implement and
evaluate Monte Carlo (MC) Dropout within a pre-trained Transformer
architecture (BERT). By maintaining stochastic dropout during inference,
we can treat the model as an approximate Bayesian Neural Network. This
allows for the extraction of predictive variance, providing a mathematical
signal for model uncertainty. I expect to demonstrate that this variance
serves as a robust indicator for detecting distribution shifts and adversarial
perturbations, ultimately enhancing the reliability of NLP systems in
high-stakes environments.

1 Introduction and Motivation
Modern Natural Language Processing (NLP) is dominated by the Transformer
architecture, yet these models remain "black boxes" regarding their confidence.
A truly trustworthy AI must not only be accurate but also "self-aware"; it should
signal when it is likely to be wrong. This is particularly critical in the presence of
Distribution Shift, where the data seen at test-time differs from the training
set.
The motivation for this project is to bridge the gap between deterministic
point-estimation and probabilistic inference. By implementing a Bayesian approximation,
we aim to provide a "Confidence Score" alongside every classification.
This allows users to set a threshold for manual intervention, making the system
safer for real-world deployment.

2 Task and Datasets
The primary task for this project is **Sentiment Classification** and **Reliability
Benchmarking**.
• In-Distribution (ID): The SST-2 (Stanford Sentiment Treebank)
dataset.
• Out-of-Distribution (OOD): The Wikitext-103 dataset (to test "I
don’t know" responses to non-sentiment text) and a manually corrupted
version of SST-2 containing character-level adversarial noise (typos, swaps).
Example Inputs & Outputs:
• ID Input: "A visually stunning masterpiece." → Output: Positive (Var:
0.02)
• OOD Input (Noise): "A v1su@lly stunn1ng m@sterp1ece." → Output:
Positive (Var: 0.45)

3 Literature Review
The Transformer architecture has revolutionized NLP through the multi-head
attention mechanism. However, standard training yields deterministic softmax
probabilities that are often poorly calibrated. Gal & Ghahramani demonstrated
that Dropout can be used as a mathematically grounded Bayesian approximation.
While this has been applied to Vision Transformers, its utility in detecting
linguistic distribution shifts in fine-tuned BERT models remains a fruitful area
for experimental validation in the context of Trustworthy AI.

4 Novel Idea: Stochastic Transformer Wrapper
My approach differs from existing work by implementing a **Stochastic Inference
Wrapper** around the standard BERT architecture. Unlike typical models that
disable dropout during evaluation (‘model.eval()‘), my modification keeps specific
dropout masks active.
For an input x, we perform T forward passes. For each pass t, we obtain a
probability ˆyt. We then compute:
Predictive Mean: μ =
1
T
XT
t=1
ˆyt (1)
Predictive Variance: σ2 =
1
T
XT
t=1
(ˆyt − μ)2 (2)
This project will specifically test if the variance σ2 is a better predictor of
model failure than the standard Softmax "confidence."

5 Expected Outcomes
I plan to achieve the following:
1. Reliability Diagrams: I expect to show that the MC-Dropout model is
better calibrated than a standard BERT baseline.
2. OOD Detection Benchmark: I aim to prove that the predictive variance
σ2 increases by a factor of at least 3× when the model is presented with
non-sentiment text or adversarial noise.
3. Trust-Score Visualization: A demonstration showing how high-variance
tokens identify the source of model confusion.
## How to run

1. **Environment:** `pip install -r requirements.txt`
2. **Data check:** `python scripts/run_data_demo.py` (downloads SST-2, Wikitext-103; prints samples).
3. **Train BERT on SST-2:** `python scripts/train.py --out checkpoints/bert_sst2`
4. **MC Dropout inference:** `python scripts/eval_mc_dropout.py --checkpoint checkpoints/bert_sst2 --out results/mc_predictions.npz`
5. **Calibration:** `python scripts/eval_calibration.py --predictions results/mc_predictions.npz --out results/calibration.npz`
6. **Plots:** Open `notebooks/02_results.ipynb` and run (reliability diagram, OOD variance).

References
[1] Vaswani, A., et al. (2017) Attention is all you need. Advances in Neural Information
Processing Systems.
[2] Gal, Y., & Ghahramani, Z. (2016) Dropout as a Bayesian approximation:
Representing model uncertainty in deep learning. ICML.
[3] Devlin, J., et al. (2018) BERT: Pre-training of deep bidirectional transformers
for language understanding. arXiv:1810.04805.
