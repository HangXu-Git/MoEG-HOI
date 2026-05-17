<div align="center">

# MoEG-HOI: Mixture of Expert Groups for One-Stage Hand-Object Interaction Motion Generation with Hand-Finger-Joint Semantic Guidance

**Hang Xu**<sup>1</sup>, **Yang Xiao**<sup>1*</sup>, **Changlong Jiang**<sup>1</sup>

<sup>1</sup>Your Institution Name &nbsp;&nbsp;&nbsp;&nbsp; <sup>2</sup>Collaborator Institution Name <br>
<sup>*</sup>Equal Contribution

<br>

[![Paper](https://img.shields.io/badge/Paper-AAAI_2026-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/38102)

**[AAAI 2026]** Official PyTorch implementation of MoEG-HOI.

</div>

## 📖 Abstract
This repository contains the official code and pre-trained models for **MoEG-HOI**, a novel one-stage framework that introduces the Mixture of Experts (MoE) architecture to the 3D Hand-Object Interaction (HOI) motion generation task for the first time. 

## 🌟 Key Features
- **One-stage MoE-based diffusion framework:** Replaces coarse-to-fine multi-stage pipelines with end-to-end trainable HOI motion generation.
- **Hierarchical Semantics-guided Expert Groups (Hand–Finger–Joint):** Explicitly model the articulated structure of hands with specialized experts under global-to-local semantic guidance
- **Action & Noise-Aware Routing:** A dynamic routing mechanism conditioned jointly on the semantic action label and the current noise timestep of the diffusion model.

---

## 🚀 News & Updates
- **[2025/11]** 🎉 MoEG-HOI has been accepted by **AAAI 2026**!
- **[2026/xx]** Code and pre-trained weights will be released soon.

---
