# TFMixer for ELS Knock-In Prediction
This repository contains the Pytorch implementation code for the paper "TFMixer: A Hybrid Attention Model with
Macroeconomic Data for ELS Knock-In Prediction"

Knock-in events can profoundly impact ELS returns, yet contract terms alone often fall short for precise risk estimation. TFMixer addresses this gap by integrating macroeconomic data from FRED-MD and FRED-QD through a time–feature attention mechanism. It demonstrates robust knock-in detection (F1=0.8957, AUROC=0.9076), offering proactive insights for ELS risk management.

## 1. Architectures
Below is a conceptual diagram of how the model processes data in parallel along two axes:

<img src=https://github.com/dxlabskku/TFMixer/blob/main/model.png/>

1) Time-Dimension Path : 
Embeds the sequence of combined features across multiple time steps.
Uses a TimesBlock to detect different periodicities in the input, performing FFT-based frequency analysis and a set of Inception-like convolution layers.
Passes the time-embedded representation through multiple layers of multi-head self-attention and feedforward layers.

2) Feature-Dimension Path : 
Transposes the data to treat each feature as a “token,” applying multi-head self-attention over the feature axis.
Captures inter-feature relationships at every time step.

3) Final Decision : 
Concatenates the final states from both paths.
Produces a single logit (or multiple logits) for classification or regression tasks.

## Dependencies
- CUDA 12.2
- python 3.10.12
- pytorch 2.5.1
- numpy 1.26.4

## Data
Three core data sources were utilized. First, 61,711 ELS products (issued November 6, 2009–November 6, 2022) were compiled, each mapped to six major underlying indices and labeled for knock-in events. Second, weekly market data for those six indices was collected via yfinance (November 6, 2007–November 6, 2022), ensuring 1–3-year lookback spans. Finally, 275 macroeconomic features from FRED-MD and FRED-QD were merged and deduplicated for the same period. Together, these sources offer a comprehensive view of both ELS contract conditions and broader economic signals.

| Data Collection     | Feature |
|---------------------|---------|
| ELS Contract Terms  | 29      |
| Underlying Assets   | 6       |
| FRED data           | 275     |


## Experiment & Result
Experiments used a 13-year ELS dataset (2009.11.06–2022.11.06), split by issuance date into 9 years of training (41,670 items), 1 year of validation (7,468 items), and 3 years of testing (12,573 items) to prevent data leakage. Training employed mini-batches of 128 with the Adam optimizer, a cosine-annealing learning rate (starting at 7×10−5), and mixed-precision (AMP) for faster GPU performance. All experiments ran in Python/PyTorch on an AMD EPYC 7282 server equipped with an NVIDIA RTX 4090 and 64 GB RAM.

TFMixer is introduced for predicting knock-in events in ELS by incorporating macroeconomic data (FRED-MD, FRED-QD), asset prices, and contract terms. Unlike traditional methods that rely on short-term indicators, TFMixer analyzes both time (long-range price patterns) and feature dimensions (macroeconomic interactions) in parallel. Over a 128-week horizon, it achieves an F1 of 0.8957 and an AUROC of 0.9076, surpassing popular attention-based and recurrent models in balancing sensitivity and precision. This approach also reduces computational overhead, giving investors timely risk signals for better ELS management.

