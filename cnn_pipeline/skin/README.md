# Diagnostic Pipeline — Output Schema Specification

Specification document for the standardized `DiagnosticState` JSON output emitted by the CNN specialist inference pipelines (`ham10000`, `ecg`). This schema serves as the single source of truth for the downstream multi-agent debate engine and clinical UI layer.

---

## 1. Full Schema Reference Example

```json
{
  "modality": "dermoscopy",
  "input_type": "image",
  "prediction": {
    "primary_diagnosis": "Melanocytic Nevus",
    "raw_class": "nv",
    "probability": 0.9343984723091125,
    "calibrated_probability": 0.9343984723091125,
    "findings": [
      {
        "label": "Melanocytic Nevus",
        "raw_class": "nv",
        "probability": 0.9343984723091125,
        "calibrated_probability": 0.9343984723091125
      },
      {
        "label": "Benign Keratosis-like Lesion",
        "raw_class": "bkl",
        "probability": 0.021221067756414413,
        "calibrated_probability": 0.021221067756414413
      },
      {
        "label": "Dermatofibroma",
        "raw_class": "df",
        "probability": 0.020126955583691597,
        "calibrated_probability": 0.020126955583691597
      },
      {
        "label": "Vascular Lesion",
        "raw_class": "vasc",
        "probability": 0.010494813323020935,
        "calibrated_probability": 0.010494813323020935
      },
      {
        "label": "Actinic Keratosis / Intraepithelial Carcinoma",
        "raw_class": "akiec",
        "probability": 0.00677603529766202,
        "calibrated_probability": 0.00677603529766202
      },
      {
        "label": "Melanoma",
        "raw_class": "mel",
        "probability": 0.004234527237713337,
        "calibrated_probability": 0.004234527237713337
      },
      {
        "label": "Basal Cell Carcinoma",
        "raw_class": "bcc",
        "probability": 0.0027481403667479753,
        "calibrated_probability": 0.0027481403667479753
      }
    ]
  },
  "differential": [
    {
      "label": "Melanocytic Nevus",
      "raw_class": "nv",
      "probability": 0.9343984723091125,
      "calibrated_probability": 0.9343984723091125
    },
    {
      "label": "Benign Keratosis-like Lesion",
      "raw_class": "bkl",
      "probability": 0.021221067756414413,
      "calibrated_probability": 0.021221067756414413
    },
    {
      "label": "Dermatofibroma",
      "raw_class": "df",
      "probability": 0.020126955583691597,
      "calibrated_probability": 0.020126955583691597
    },
    {
      "label": "Vascular Lesion",
      "raw_class": "vasc",
      "probability": 0.010494813323020935,
      "calibrated_probability": 0.010494813323020935
    },
    {
      "label": "Actinic Keratosis / Intraepithelial Carcinoma",
      "raw_class": "akiec",
      "probability": 0.00677603529766202,
      "calibrated_probability": 0.00677603529766202
    },
    {
      "label": "Melanoma",
      "raw_class": "mel",
      "probability": 0.004234527237713337,
      "calibrated_probability": 0.004234527237713337
    },
    {
      "label": "Basal Cell Carcinoma",
      "raw_class": "bcc",
      "probability": 0.0027481403667479753,
      "calibrated_probability": 0.0027481403667479753
    }
  ],
  "evidence": {
    "positive_findings": [
      "Symmetric lesion with uniform pigment distribution across central and peripheral zones.",
      "Regular reticular pigment network and uniform peripheral globules consistent with benign melanocytic nevus."
    ],
    "negative_findings": [
      "Absence of atypical streaks, blue-white veil, marked asymmetry, or regression structures."
    ]
  },
  "explainability": {
    "method": "Grad-CAM",
    "target_layer": "resnet18.layer4",
    "region_description": "Spatial attention concentrated on lesion center, irregular borders, and peripheral pigment distribution."
  }
}
```

---

## 2. TypeScript Type Definitions

```typescript
export type ModalityType = "dermoscopy" | "ecg";
export type InputType = "image" | "waveform" | "tabular";

export interface DifferentialItem {
  label: string;                  // Human-readable clinical label
  raw_class: string;              // Short internal code ('nv', 'mel', etc.)
  probability: number;            // Softmax probability [0.0, 1.0]
  calibrated_probability: number; // Platt/Temperature-scaled confidence [0.0, 1.0]
}

export interface PredictionPayload {
  primary_diagnosis: string;      // Top-ranked clinical label
  raw_class: string;              // Top-ranked raw class code
  probability: number;            // Top class softmax confidence
  calibrated_probability: number; // Top class calibrated confidence
  findings: DifferentialItem[];   // Full ranked differential (all classes)
}

export interface EvidencePayload {
  positive_findings: string[];    // Findings that support primary_diagnosis
  negative_findings: string[];    // Pathologies ruled out / absent
}

export interface ExplainabilityPayload {
  method: "Grad-CAM" | "Score-CAM" | "Integrated-Gradients";
  target_layer: string;           // CNN layer where heatmaps are hooked
  region_description: string;     // Textual explanation of spatial focus
}

export interface DiagnosticState {
  modality: ModalityType;
  input_type: InputType;
  prediction: PredictionPayload;
  differential: DifferentialItem[];
  evidence: EvidencePayload;
  explainability: ExplainabilityPayload;
}
```

---

## 3. Root-Level Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `modality` | `string` | Yes | Identifies diagnostic source modality (`"dermoscopy"` for HAM10000, `"ecg"` for ECG). |
| `input_type` | `string` | Yes | Data representation fed to CNN (`"image"`). |
| `prediction` | `object` | Yes | Primary diagnostic outcome and top-line probabilities. |
| `differential` | `array` | Yes | Complete ordered list of candidate diagnoses (top level for fast UI consumption). |
| `evidence` | `object` | Yes | Clinical arguments (positive + negative) passed to debate agents. |
| `explainability` | `object` | Yes | Visual attribution and saliency metadata. |

---

## 4. `prediction` Object Breakdown

```json
{
  "primary_diagnosis": "Melanocytic Nevus",
  "raw_class": "nv",
  "probability": 0.9343984723091125,
  "calibrated_probability": 0.9343984723091125,
  "findings": [ ... ]
}
```

- **`primary_diagnosis`** (`string`): The clinical display name of the class with highest probability (`argmax(probabilities)`).
- **`raw_class`** (`string`): The short machine-readable key (e.g. `'nv'`, `'mel'`, `'bcc'`). Matches `CLASS_INDEX_TO_NAME`.
- **`probability`** (`float`): Raw softmax output `exp(z_i) / sum(exp(z))` where `z` is the final layer logits vector. Range: `[0.0, 1.0]`.
- **`calibrated_probability`** (`float`): Post-processed probability via temperature scaling (`T`) or Platt scaling. Calibrated to represent empirical true risk. Equals `probability` when calibration temperature $T = 1.0$.
- **`findings`** (`array<DifferentialItem>`): Exact duplicate of `differential` embedded directly in the prediction envelope.

---

## 5. `differential` Array Breakdown

An array of 7 objects (for HAM10000) strictly sorted in **descending order of probability**.

### Item Structure

| Property | Type | Description |
|---|---|---|
| `label` | `string` | Full clinical title suitable for doctor-facing UI |
| `raw_class` | `string` | Internal category key matching dataset ground truth |
| `probability` | `float` | Class-specific softmax posterior |
| `calibrated_probability` | `float` | Class-specific calibrated posterior |

### Invariant Constraints

1. **Ordering**: `differential[i].probability >= differential[i+1].probability` $\forall i$.
2. **Completeness**: Contains exactly all 7 classes defined in the vocabulary.
3. **Unit Sum**: $\sum_{i=0}^{6} \text{differential}[i]\text{.probability} = 1.0 \pm 10^{-6}$.
4. **Primary Match**: `prediction.primary_diagnosis == differential[0].label` and `prediction.raw_class == differential[0].raw_class`.

---

## 6. HAM10000 Class Labels Catalog

The 7 diagnostic categories recognized by the classifier:

| Code (`raw_class`) | Clinical Label (`label`) | Pathological Nature | Histological / Dermoscopic Markers |
|---|---|---|---|
| `nv` | **Melanocytic Nevus** | Benign mole | Symmetric reticular network, uniform peripheral globules, homogeneous pigment center |
| `mel` | **Melanoma** | Malignant cancer | Asymmetry, atypical pigment network, irregular dots/globules, peripheral streaks, blue-white veil |
| `bkl` | **Benign Keratosis-like Lesion** | Benign (SK / solar lentigo) | Milky-orange/brown pseudonetwork, moth-eaten borders, comedo-like openings, horn pseudocysts |
| `bcc` | **Basal Cell Carcinoma** | Malignant cancer | Arborizing (tree-like) telangiectasias, blue-gray ovoid nests, shiny white structures, ulceration |
| `akiec` | **Actinic Keratosis / Bowen's Disease** | Pre-cancerous / In-situ | Erythematous background, keratotic scale/crust, strawberry pattern with follicular white halos |
| `df` | **Dermatofibroma** | Benign fibrous nodule | Central white scar-like patch, delicate fine peripheral pigment network |
| `vasc` | **Vascular Lesion** | Benign vascular | Red-to-violaceous or blue-black lacunae (vascular lagoons), absence of pigment network |

---

## 7. `evidence` Object Breakdown

The evidence block extracts diagnostic justifications to serve as arguments in multi-agent medical consensus debates.

```json
{
  "positive_findings": [
    "Symmetric lesion with uniform pigment distribution across central and peripheral zones.",
    "Regular reticular pigment network and uniform peripheral globules consistent with benign melanocytic nevus."
  ],
  "negative_findings": [
    "Absence of atypical streaks, blue-white veil, marked asymmetry, or regression structures."
  ]
}
```

### Complete Class-by-Class Evidence Templates

#### `mel` (Melanoma)
- **`positive_findings`**:
  - *"Asymmetric multicomponent pigment pattern with architectural disorganization."*
  - *"Presence of atypical pigment network, irregular dots/globules, and peripheral streaks."*
  - *"Blue-white veil and abnormal vascular structures indicative of malignant melanoma."*
- **`negative_findings`**:
  - *"Absence of regular benign reticular network or uniform homogeneous architecture."*

#### `nv` (Melanocytic Nevus)
- **`positive_findings`**:
  - *"Symmetric lesion with uniform pigment distribution across central and peripheral zones."*
  - *"Regular reticular pigment network and uniform peripheral globules consistent with benign melanocytic nevus."*
- **`negative_findings`**:
  - *"Absence of atypical streaks, blue-white veil, marked asymmetry, or regression structures."*

#### `bcc` (Basal Cell Carcinoma)
- **`positive_findings`**:
  - *"Prominent arborizing (tree-like) telangiectasias across the surface of the lesion."*
  - *"Blue-gray ovoid nests, multiple blue-gray globules, and focal ulceration typical of basal cell carcinoma."*
- **`negative_findings`**:
  - *"Absence of pigmented melanocytic network or uniform reticular grid pattern."*

#### `akiec` (Actinic Keratosis / Bowen's Disease)
- **`positive_findings`**:
  - *"Prominent erythematous background with white-to-yellow keratotic surface scale and crust."*
  - *"Strawberry pattern with targetoid hair follicles surrounded by a white halo, typical of actinic keratosis / Bowen's disease."*
- **`negative_findings`**:
  - *"Absence of discrete deep dermal pigment nests or prominent arborizing telangiectasia."*

#### `bkl` (Benign Keratosis-like Lesion)
- **`positive_findings`**:
  - *"Milky-orange or brownish structureless areas with sharp, moth-eaten borders."*
  - *"Characteristic comedo-like openings, horn pseudocysts, and fingerprint-like structures consistent with benign keratosis."*
- **`negative_findings`**:
  - *"Absence of atypical melanocytic pigment network or deep invasive architectural signs."*

#### `df` (Dermatofibroma)
- **`positive_findings`**:
  - *"Distinct central white patch or scar-like structureless area."*
  - *"Delicate, fine peripheral pigment network characteristic of dermatofibroma."*
- **`negative_findings`**:
  - *"Absence of marked structural asymmetry, irregular streaks, or blue-white veil."*

#### `vasc` (Vascular Lesion)
- **`positive_findings`**:
  - *"Well-demarcated red, violaceous, or blue-black lacunae (vascular lagoons)."*
  - *"Distinct vascular spaces with reddish-purple coloration typical of benign vascular lesions (hemangioma / angioma)."*
- **`negative_findings`**:
  - *"Absence of melanocytic pigment network, pigment globules, or arborizing telangiectasia."*

---

## 8. `explainability` Object Breakdown

Provides interpretability audit trails for clinical safety and XAI visualization.

```json
{
  "method": "Grad-CAM",
  "target_layer": "resnet18.layer4",
  "region_description": "Spatial attention concentrated on lesion center, irregular borders, and peripheral pigment distribution."
}
```

| Field | Type | Description |
|---|---|---|
| `method` | `string` | XAI algorithm used (`"Grad-CAM"`). Computes $\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$ |
| `target_layer` | `string` | PyTorch module layer hook (`"resnet18.layer4"` — the final 512-channel $7 \times 7$ feature map before Global Average Pooling) |
| `region_description` | `string` | Clinical localization summary indicating where visual attention was concentrated on the dermatoscopy frame |

---

## 9. Downstream Consumption Guide

### In Multi-Agent Medical Debate

```python
from cnn_pipeline.ham10000 import predict_ham10000

result = predict_ham10000("patient_lesion.jpg")

# 1. Primary claim proposed by Specialist Agent:
claim = f"Proposed diagnosis: {result['prediction']['primary_diagnosis']} ({result['prediction']['probability']:.1%})"

# 2. Evidence passed to Critic Agent:
arguments_for = result['evidence']['positive_findings']
counter_checks = result['evidence']['negative_findings']

# 3. Differential passed to Arbiter Agent:
competing_differentials = [
    item for item in result['differential'] if item['probability'] > 0.05
]
```

### In Frontend / Doctor UI

- **Top Card**: Displays `prediction.primary_diagnosis` with a colored confidence badge (`prediction.probability`).
- **Differential Bar Chart**: Horizontal bar chart driven by `differential` array (`item.label` vs `item.probability`).
- **Checklist**: Green checkmarks for `evidence.positive_findings`, red exclusion marks for `evidence.negative_findings`.
- **Grad-CAM Overlay**: Rendered on top of source image based on `explainability.target_layer`.
