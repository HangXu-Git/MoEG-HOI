# MoEG-HOI: Mixture of Expert Groups for One-Stage Hand-Object Interaction Motion Generation with Hand-Finger-Joint Semantic Guidance

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-Arxiv%20Link-blue)](#)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](#)
[![Video](https://img.shields.io/badge/Video-YouTube-red)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[AAAI 2026]** Official PyTorch implementation of MoEG-HOI.

</div>

## 📖 Abstract
This repository contains the official code and pre-trained models for **MoEG-HOI**, a novel one-stage framework that introduces the Mixture of Experts (MoE) architecture to the 3D Hand-Object Interaction (HOI) motion generation task for the first time. 

Given textual descriptions and object trajectories, MoEG-HOI synthesizes physically consistent and fine-grained 3D hand motion sequences $M$. By leveraging a hierarchical "Hand-Finger-Joint" expert design within a diffusion-based framework, our method efficiently captures complex manipulation semantics. Furthermore, we introduce a novel joint routing mechanism that is both **action-aware** and **noise-aware**, utilizing action labels and noise timesteps to dynamically route tokens to the most specialized expert groups during the generative process.

## 🌟 Key Features
- **One-Stage Framework:** Simplifies the traditional multi-stage HOI generation pipeline into an elegant, end-to-end diffusion process.
- **Hierarchical Expert Groups:** Specialized network branches focusing on different anatomical levels (Hand, Finger, Joint) for fine-grained pose generation.
- **Action & Noise-Aware Routing:** A dynamic routing mechanism conditioned jointly on the semantic action label and the current noise timestep of the diffusion model.
- **Physical Consistency:** Generates highly realistic interactions with minimal penetration and precise contact alignment.

---

## 🚀 News & Updates
- **[2026/03]** 🎉 MoEG-HOI has been accepted by **AAAI 2026**!
- **[2026/xx]** Code and pre-trained weights will be released soon.

---

## 🛠️ Installation

**1. Clone the repository:**
```bash
git clone [https://github.com/YourUsername/MoEG-HOI.git](https://github.com/YourUsername/MoEG-HOI.git)
cd MoEG-HOI
