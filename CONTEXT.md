# RF Drone Spectrogram Classification

This context defines the shared language for turning RF IQ recordings into spectrogram evidence and classifying whether a signal source is drone or non-drone. It exists to keep dataset, label, and evaluation terminology consistent across preprocessing, training, and detection workflows.

## Language

**Context**:
A single bounded context covering RF drone detection from IQ-derived spectrogram data.
_Avoid_: Multi-context split, separate subdomains (for now)

**Drone Signal**:
The positive class in this context: RF evidence whose source is treated as a drone emitter in the dataset.
_Avoid_: drone, UAV image, aircraft

**Non-Drone Signal**:
The negative class in this context: RF evidence treated as not originating from a drone emitter.
_Avoid_: background only, noise only, normal class

**Spectrogram Sample**:
The atomic unit for training and inference: one spectrogram image derived from a bounded RF time segment and assigned exactly one class label.
_Avoid_: frame (video sense), raw signal, waveform image

**Label Assignment**:
Each **Spectrogram Sample** maps to exactly one of two labels: **Drone Signal** or **Non-Drone Signal**.
_Avoid_: multi-label tagging, soft class membership

**RF Recording**:
A time-continuous raw IQ capture that serves as a source from which multiple **Spectrogram Sample** objects can be derived.
_Avoid_: dataset sample, spectrogram image, training item

**Dataset Split**:
A partition of labeled **Spectrogram Sample** data into three disjoint subsets: **Train Split**, **Validation Split**, and **Test Split**.
_Avoid_: random pool, mixed evaluation set

**Train Split**:
The subset used to fit model parameters.
_Avoid_: tuning set, final benchmark set

**Validation Split**:
The subset used for model selection and early stopping decisions during development.
_Avoid_: final test, production set

**Test Split**:
The held-out subset used only for final performance reporting.
_Avoid_: training monitor set, validation clone

**Binary Classification**:
A supervised task that learns a decision boundary between **Drone Signal** and **Non-Drone Signal** from labeled **Spectrogram Sample** data.
_Avoid_: anomaly-only detection, one-class modeling

**One-Class Detection**:
A detection task that models only **Drone Signal** profile and flags samples by deviation from that profile using a threshold rule.
_Avoid_: two-class discriminative training, balanced binary learning

**Linear Probe**:
A shallow classifier trained on fixed embeddings from a frozen feature extractor to separate **Drone Signal** and **Non-Drone Signal**.
_Avoid_: full end-to-end fine-tuning, one-class threshold detector

**Detection Result**:
The inference output for one **Spectrogram Sample**, consisting of a predicted class and its associated confidence score.
_Avoid_: raw logit dump, report row, batch summary

**Macro F1**:
The primary model-comparison metric in this context, defined as the unweighted mean of per-class F1 scores across **Drone Signal** and **Non-Drone Signal**.
_Avoid_: accuracy-only ranking, positive-class-only score

## Example Dialogue

Dev: "This **RF Recording** produced 240 **Spectrogram Sample** items. How do we split them?"
Domain Expert: "Place each sample into exactly one **Dataset Split**: **Train Split**, **Validation Split**, or **Test Split**."
Dev: "For this experiment, should we run **Binary Classification** or **One-Class Detection**?"
Domain Expert: "Use **Binary Classification** when both classes are labeled; use **One-Class Detection** when we model only **Drone Signal**."
Dev: "We also trained a **Linear Probe** on frozen embeddings. How do we compare it?"
Domain Expert: "Compare all approaches by **Macro F1** and inspect each **Detection Result** for confidence behavior."
