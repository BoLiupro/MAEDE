# MAEDE: A Multi-Granularity Self-Guiding Graph Diffusion Model for Predicting Regional Private Car Activity

## 📘 Overview
**MAEDE (Multi-grAnularity sElf-guiding Diffusion graph modEl)** is a unified framework for **probabilistic forecasting** of regional private car activity. MAEDE is designed to address the inherent stochasticity, multi-granularity temporal patterns, and complex environmental influences on private car movements within urban spaces.

By leveraging a **conditional diffusion backbone**, **LLM-enhanced visual-textual contrastive learning**, and a **multi-granularity self-guiding mechanism**, MAEDE captures both fine-grained variations and high-level temporal trends, outperforming state-of-the-art baselines.

<p align="center">
  <img src="src\framework.pdf" width="600px" alt="MAEDE Framework" />
</p>

### 📄 Abstract
Transportation networks have become a vital component of city infrastructure, with private cars accounting for the largest share of urban transportation. Private car activity reflects individualized travel behaviors, such as Arrive-Stay-Leave (ASL) sequences, critical for effective traffic management, parking optimization, and urban planning. In this study, we propose MAEDE for probabilistic forecasting of regional private car activity. MAEDE employs a conditional diffusion model as its backbone, leveraging the strengths of diffusion models in probabilistic modeling and their ability to capture complex joint distributions. Additionally, MAEDE incorporates satellite imagery to capture environmental contexts using an LLM-enhanced contrastive learning module. In this module, the LLM generates textual descriptions related to private car activity for each image, and the contrastive learning component aligns these textual and visual representations, effectively encoding critical urban features. To address both long-term trends and short-term fluctuations in private car activity, MAEDE uses a multi-granularity self-guiding mechanism. This mechanism utilizes coarse-grained data at various granularity levels as targets to guide the denoising process, enabling the simultaneous capture of high-level trends and fine-grained details. Extensive experiments on real-world private car trip datasets demonstrate that MAEDE outperforms state-of-the-art baselines, achieving improvements of over 8% in Mean Squared Error (MSE) and 14% in Continuous Ranked Probability Score (CRPS). Cross-city evaluations further validate MAEDE’s robustness and generalization across diverse urban environments.

### 🧠 Model Highlights
- Diffusion-based probabilistic prediction  
- Multi-granularity supervision (hourly, daily)  
- Self-guiding Gated WaveNet encoder  
- LLM-guided image-text contrastive urban representation  

---

## 📊 Dataset Information
We collected a large-scale private car trip dataset from three provincial capital cities in China: **Changsha**, **Guangzhou**, and **Shenzhen**. The data spans **March 1 to August 27, 2024**, and contains **1,871,959 trip records**.

The dataset reflects Arrive-Stay-Leave (ASL) behavior aggregated at the regional level and is used to train and evaluate regional activity prediction.

Due to privacy concerns, only partial (anonymized) data samples are included in this repository. For detailed dataset statistics and processing logic, refer to **Appendix A** of the paper.

### 🔐 Data Privacy & Ethics
We are acutely aware of the privacy implications of using mobility data for behavioral modeling. The following measures were strictly followed:

- ✅ Full authorization was obtained from both users and the data provider.  
- ✅ Explicit user consent was collected at data generation.  
- ✅ Researchers had no access to raw individual-level trajectories.  
- ✅ Only aggregated, anonymized region-level data was used.  
- ✅ No personally identifiable information (PII) was accessible.  
- ✅ All sensitive processing (e.g., de-identification, secure storage) was done by the operator.  
- ✅ The protocol was approved by all involved institutions and providers.  

---

## ⚙️ Usage Pipeline

### 1. Encode Images using LLM-enhanced Encoder
```bash
python encode.py
