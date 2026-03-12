# TRAK Data Attribution on ResNet-18 (CIFAR-10)

## Overview

This project explores **training data attribution** in deep neural networks using **TRAK (Training Data Attribution)**.  
The goal is to understand **which training samples most strongly influence a model’s prediction**.

The experiment is performed on a **ResNet-18 model trained on the CIFAR-10 dataset**. By computing attribution scores for training samples, we can identify the examples that most contributed to the prediction of a given test image.

Understanding influential samples helps reveal:

- how models make decisions  
- what patterns the model relies on  
- potential dataset biases or shortcuts learned during training  

---

## Method

The workflow of the experiment is:

1. Train a **ResNet-18 model** on the **CIFAR-10 dataset**.
2. Select a **test sample** from the dataset.
3. Apply **TRAK attribution** to compute the influence of each training example on the prediction.
4. Rank training samples based on their **influence scores**.
5. Visualize the **top influential training samples**.

TRAK estimates the contribution of each training data point to a model’s prediction using gradient-based influence approximations.

---

## Influence Visualization

<img width="3396" height="2367" alt="top_influential_samples" src="https://github.com/user-attachments/assets/e6160a2b-13b0-4c5a-8d70-bf30dc1f00b2" />

**Figure Description**

- The **test image** is shown along with the **top influential training samples**.
- Each training sample is associated with an **influence score**, indicating how strongly it contributed to the model’s prediction.
- Higher scores indicate **greater influence on the model output**.

This visualization allows us to inspect whether the influential samples:

- belong to the **same class**
- share **visual similarities**
- reveal **biases in the learned representation**

---

## Key Insights

The influential samples provide insight into how the model interprets visual patterns. In many cases, the most influential examples may:

- share **similar textures or colors**
- contain **similar backgrounds**
- resemble **structural features** of the test image

This demonstrates that deep learning models often rely on **statistical visual patterns learned during training**.

---

## Applications

Training data attribution techniques like TRAK are useful for:

- **Model interpretability**
- **Debugging dataset biases**
- **Identifying mislabeled training samples**
- **Understanding representation learning**
- **Improving dataset quality**

---

## Technologies Used

- **PyTorch**
- **ResNet-18**
- **CIFAR-10**
- **TRAK (Training Data Attribution)**

---

## Future Work

Potential extensions of this project include:

- Applying attribution analysis on **larger datasets (e.g., ImageNet)**
- Comparing TRAK with other attribution techniques
- Investigating **dataset bias through influential examples**
- Combining attribution with **visual explanation methods**

---
