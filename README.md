# Multimodal Diagnostic Assistant with Debate-Based Multi-Agent Verification
### 11-Week Final Year Project Plan

**Team size:** 3
**Core idea:** Three CNNs (X-ray, skin, ECG) → probable disease list → multi-agent debate pipeline (Advocate → Skeptic/Challenger → Evidence-check → Moderator, repeated over N rounds) → hallucination-reduced final output for a doctor.

---

## 1. 11-Week Timeline

| Week | Phase | Goals | Suggested split (3 people) |
|---|---|---|---|
| **1** | Setup & Literature | Finalize scope, datasets (ChestX-ray14/CheXpert, HAM10000/ISIC, MIT-BIH/PTB-XL), tech stack (confirm LangGraph + LLM choice as in CORTEX-X), repo structure, literature review write-up, doctor/domain-expert contact if possible | 1 person per modality starts dataset scouting; 1 person drafts the debate-architecture spec |
| **2** | Data pipeline | Preprocessing + augmentation for all 3 modalities, train/val/test splits, class-imbalance handling (all 3 domains are heavily imbalanced), EDA report | Split by modality (1 each) |
| **3** | CNN v1 | Baseline CNNs (transfer learning: DenseNet121/EfficientNet for X-ray & skin, CNN-LSTM/1D-CNN for ECG) | Split by modality |
| **4** | CNN v2 + explainability | Hyperparameter tuning, class-imbalance-aware loss (focal/asymmetric loss), add Grad-CAM/saliency maps for X-ray & skin, attention maps for ECG | Split by modality; 1 person starts building the "confidence + evidence packet" schema each CNN will emit |
| **5** | Model finalization | Freeze v1 of all 3 models, document metrics (AUC, F1, per-class), export as servable artifacts (ONNX/pickle/FastAPI-ready) | Converge as a team, write model cards |
| **6** | Debate architecture design | Define agent roles precisely (Advocate, Skeptic, Evidence/Literature-checker, Moderator), prompt templates, state schema in LangGraph, stopping criteria (fixed rounds vs. convergence), scoring/confidence aggregation logic | 1 person owns LangGraph orchestration; 1 owns prompt engineering; 1 owns evidence/RAG layer |
| **7** | Debate implementation v1 | Build the graph node-by-node: hypothesis generation → advocacy → challenge → rebuttal/evidence request → next round. Get it working on synthetic/text-only cases first (no CNN yet) | Same split, integrate |
| **8** | Integration | Wire CNN outputs (label + confidence + saliency map) into the debate graph as the seed evidence for each agent. Add RAG grounding against a curated medical knowledge base (guidelines, drug/condition reference) so the "evidence" node isn't hallucinating either | Full team |
| **9** | End-to-end system + UI | FastAPI backend, simple doctor-facing dashboard (upload image/ECG → CNN outputs → live debate transcript → final verdict with confidence + cited evidence + saliency overlays) | 1 backend, 1 frontend, 1 on debate-logic bugs |
| **10** | Evaluation | Quantitative: accuracy/F1 of final debated output vs. raw CNN-only baseline vs. single-LLM-judge baseline. Ablations: vary number of rounds/agents (per Du et al.'s findings, more rounds/agents generally help but with diminishing returns and cost). Qualitative: hallucination rate, doctor/domain-expert feedback if available | Full team, divide metrics vs. ablations vs. qualitative |
| **11** | Report, paper, demo | Final report/thesis, poster, demo video, buffer for last bug fixes, prep for viva/defense | Full team |

**Buffer note:** weeks 4, 8, and 10 are the highest-risk weeks (CNN convergence issues, integration bugs, evaluation surprises) — don't schedule anything else critical those weeks.

---

## 2. What you can add to strengthen the project

**To make the debate loop actually trustworthy (not just longer):**
- **Ground the "evidence" node in retrieval, not just LLM memory.** A pure LLM-vs-LLM debate can still have both sides hallucinate confidently. Add a RAG layer over real clinical references (WHO guidelines, drug/condition databases, or a curated PubMed subset) so the Skeptic/Evidence agent can cite something verifiable rather than generate plausible-sounding objections from its own memory. This is the difference between "debate reduces hallucination" and "debate reduces hallucination *and stays grounded*."
- **Confidence calibration, not just a label.** Have each CNN emit calibrated probabilities (temperature scaling / Platt scaling), and pass that uncertainty into the debate as a prior — a 51% confidence finding should be argued differently than a 97% one.
- **Explainability tied to the debate, not just the CNN.** Grad-CAM on the X-ray/skin models is good, but also log *why* each agent's argument was accepted or rejected each round — this becomes your audit trail, which matters a lot for a medical-facing system and for your viva.
- **Ablation on rounds/agents.** The original multi-agent debate literature explicitly shows accuracy improves with more agents and more rounds but at real compute cost — running this ablation yourself is an easy, evaluable contribution for your report.
- **Bias/fairness check.** Skin-tone bias in dermoscopy datasets (HAM10000/ISIC skew toward lighter skin) and demographic imbalance in ECG/X-ray datasets are well-documented problems — even a small fairness analysis section adds real credibility.
- **Human-in-the-loop escape hatch.** Let the doctor interject mid-debate or flag a verdict as wrong — feed that back as a correction. Even a simple version of this is a strong "future work → we implemented it" story.
- **Multimodal fusion (stretch goal).** If a patient has both an X-ray and ECG, let the debate agents reference findings across modalities rather than treating each as fully independent — this is closer to how a real multidisciplinary team (MDT) works and matches recent literature (see MedChat, GI-oncology MDT paper below).
- **Uncertainty/abstention.** If after N rounds no consensus is reached, the system should say "insufficient evidence / refer to specialist" rather than force an answer — this is both safer and a good research point (aligns with "don't hallucinate, abstain" literature).

---

## 3. Relevant research papers by component

### A. Multi-agent debate & hallucination reduction (core of your novelty)
- **Du et al., "Improving Factuality and Reasoning in Language Models through Multiagent Debate"** (arXiv:2305.14325, ICML 2024) — the foundational "society of minds" debate paper: multiple LLM instances independently answer, then critique each other's reasoning over several rounds, converging on more factual answers. This is almost exactly your architecture's ancestor. Explicitly reports accuracy improving with more agents and more debate rounds.
- **Liang et al., "Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate"** — extends debate to reduce "degeneration-of-thought" (agents converging too early on the wrong answer), relevant to how you design your advocate/skeptic prompts so they don't just agree with each other.
- **"Removal of Hallucination on Hallucination: Debate-Augmented RAG"** (arXiv:2505.18581) — directly relevant: combines RAG with multi-agent debate to fix the "second-order hallucination" problem (debate agents hallucinating about retrieved evidence itself). Good citation for your RAG-grounding addition above.
- **"Minimizing Hallucinations and Communication Costs: Adversarial Debate and Voting Mechanisms in LLM-Based Multi-Agents"** (MDPI Applied Sciences, 2025) — adversarial debate + voting + entropy compression to cut token cost, and explicitly discusses medical consultation as a target use case.
- **"The Confident Liar: Diagnosing Multi-Agent Debate with Log-Probabilities and LLM-as-Judge"** (arXiv:2606.10296) — useful for your evaluation section: proposes ways to actually measure debate quality beyond final-answer accuracy, which most prior work skips.
- **"Towards Detecting LLMs Hallucination via Markov Chain-based Multi-agent Debate Framework"** — proposes a scoring mechanism for response consistency across debate rounds; relevant to your confidence-aggregation design.

### B. Multi-agent systems specifically in medical diagnosis
- **"MedChat: A Multi-Agent Framework for Multimodal Diagnosis with Large Language Models"** (arXiv:2506.07400) — very close to your project: combines vision models with role-specific LLM agents coordinated by a director agent, explicitly to reduce hallucination and mirror multidisciplinary team reasoning.
- **"Multi-Agent Intelligence for Multidisciplinary Decision-Making in Gastrointestinal Oncology"** (arXiv:2512.08674) — models real hospital tumor-board style consensus among expert agents; good reference for justifying your advocate/skeptic/moderator role design as mirroring real clinical MDTs.
- **"Council Mode: A Heterogeneous Multi-Agent Consensus Framework for Reducing LLM Hallucination and Bias"** (arXiv:2604.02923) — parallel generation across multiple models + a synthesis/consensus model, with concrete hallucination-reduction numbers; useful comparison baseline.
- **Review: agentic AI for reducing hallucination in radiology workflows (2024–2025 evidence review)** — surveys multi-agent role-based systems, RAG, and uncertainty quantification specifically for radiology LLM hallucination, good for your literature review's radiology-specific framing.

### C. X-ray CNN component
- **Rajpurkar et al., CheXNet** (the foundational 121-layer DenseNet chest X-ray model, benchmarked against radiologists) — cite as your baseline architecture reference.
- **"CheX-DS: Improving Chest X-ray Image Classification with Ensemble Learning Based on DenseNet and Swin Transformer"** (arXiv:2505.11168) — current SOTA-adjacent approach on ChestX-ray14, addresses long-tail/class-imbalance, directly useful for your training methodology.
- **"Comparison of EfficientNet CNN models for multi-label chest X-ray disease diagnosis"** (PeerJ CS, 2025) — practical comparison across EfficientNetB0–B7 with attention, useful for architecture selection/ablation.
- **"Parallel CNN-ELM" for 17 lung diseases including COVID-19"** (ScienceDirect) — good related-work citation, also reviews prior CXR models (BCCNN, ResNet-50, DenseNet+LSTM).

### D. Skin lesion CNN component
- **"A Comprehensive Systematic Review: Advancements in Skin Cancer Classification and Segmentation Using the ISIC Dataset"** (ScienceDirect, 2024) — good systematic review to cite for your literature survey and to flag dataset/augmentation issues.
- **"Enhancing skin lesion classification: a CNN approach with human baseline comparison"** (PeerJ CS, 2025) — EfficientNetB3 on ISIC-2019/2020 benchmarked directly against 69 medical professionals; strong reference for how to frame your own human-comparison evaluation.
- **"LesionNet: an automated approach for skin lesion classification using SIFT features with customized CNN"** (PMC, 2024) — good architecture-comparison reference across ISBI/ISIC/PH2/HAM10000.

### E. ECG CNN component
- **"A hybrid deep learning network for automatic diagnosis of cardiac arrhythmia based on 12-lead ECG" (CBGM: CNN-BiGRU + multi-head attention)** (Nature Sci Reports, 2024) — strong architecture reference (99%+ accuracy on MIT-BIH), directly applicable to your ECG model.
- **"An explainable deep learning framework for trustworthy arrhythmia detection from ECG signals"** (Nature Sci Reports, 2025) — combines MITDB/PTBDB with explainability, useful both for your ECG model design and your Grad-CAM-equivalent explainability addition.
- **"Interpretable Deep Learning Models for Arrhythmia Classification Based on ECG Signals Using PTB-XL Dataset"** (MDPI, 2025) — CNN vs. VGG16 across binary/multiclass/subclass tasks on PTB-XL, with lead-specific interpretability; very relevant given PTB-XL is a common, well-documented dataset choice.
- **"Fusion-Based Deep Learning Ensemble on MIT-BIH and PTB-XL ECG Databases for Enhanced Cardiac Diagnosis"** (medRxiv, 2025) — fuses both major ECG datasets with an ensemble; relevant if you want to combine MIT-BIH + PTB-XL rather than pick one.

---

## 4. Suggested report/paper structure (maps to your evaluation week)
1. Introduction & motivation (clinical need for hallucination-resistant AI assistance)
2. Related work (Sections A–E above)
3. System architecture (3 CNNs + debate graph diagram — you already have a base for this from CORTEX-X)
4. Methodology (per-modality CNN training + debate protocol design)
5. Evaluation (accuracy vs. baselines, ablation on rounds/agents, hallucination-rate metric, optional expert feedback)
6. Limitations & ethics (bias, dataset consent, "this is a decision-support tool, not a diagnostic authority" disclaimer — important to state explicitly)
7. Conclusion & future work (multimodal fusion, human-in-the-loop, larger clinical validation)

If it'd help, I can also generate an architecture diagram of the full pipeline (CNNs → debate graph → doctor dashboard) to drop straight into your report.
