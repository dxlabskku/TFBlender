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
Testing accuracy of Node Classification without node features are summarized below.

<table>
  <tr>
    <td><b>Method</b></td>
    <td><b>Feature</b></td>
    <td><b>Cora</b></td>
    <td><b>CiteSeer</b></td>
    <td><b>PubMed</b></td>
    <td><b>Flickr</b></td>
  </tr>
  <tr>
    <td rowspan="2">GCN</td>
    <td>One-hot</td>
    <td align="right">69.96</td>
    <td align="right">45.00</td>
    <td align="right">63.04</td>
    <td align="right">51.36</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right">71.06</td>
    <td align="right">46.82</td>
    <td align="right">74.40</td>
    <td align="right">54.75</td>
  </tr>
  <tr>
    <td rowspan="2">GAT</td>
    <td>One-hot</td>
    <td align="right">67.80</td>
    <td align="right">45.26</td>
    <td align="right">66.50</td>
    <td align="right">52.68</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right">69.54</td>
    <td align="right">45.16</td>
    <td align="right">75.22</td>
    <td align="right">54.02</td>
  </tr>
  <tr>
    <td rowspan="2">GraphSAGE</td>
    <td>One-hot</td>
    <td align="right">61.38</td>
    <td align="right">36.12</td>
    <td align="right">55.18</td>
    <td align="right">-</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right">70.96</td>
    <td align="right">45.66</td>
    <td align="right">73.52</td>
    <td align="right">-</td>
  </tr>
  <tr>
    <td rowspan="2">GIN</td>
    <td>One-hot</td>
    <td align="right">42.52</td>
    <td align="right">28.64</td>
    <td align="right">41.86</td>
    <td align="right">43.24</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right">60.82</td>
    <td align="right">40.80</td>
    <td align="right">57.12</td>
    <td align="right"><b>55.71</b></td>
  </tr>
  <tr>
    <td rowspan="2">APPNP</td>
    <td>One-hot</td>
    <td align="right">58.28</td>
    <td align="right">33.12</td>
    <td align="right">45.26</td>
    <td align="right">42.34</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right">70.60</td>
    <td align="right">39.86</td>
    <td align="right">63.04</td>
    <td align="right">51.54</td>
  </tr>
  <tr>
    <td rowspan="2">JKNet</td>
    <td>One-hot</td>
    <td align="right">63.54</td>
    <td align="right">39.62</td>
    <td align="right">59.22</td>
    <td align="right">44.98</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right">70.76</td>
    <td align="right">42.26</td>
    <td align="right">73.10</td>
    <td align="right">53.98</td>
  </tr>
  <tr>
    <td rowspan="2">GCNII</td>
    <td>One-hot</td>
    <td align="right">62.78</td>
    <td align="right">44.44</td>
    <td align="right">66.48</td>
    <td align="right">42.34</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right">67.22</td>
    <td align="right">41.86</td>
    <td align="right">73.46</td>
    <td align="right">54.05</td>
  </tr>
  <tr>
    <td rowspan="2">GatedGCN</td>
    <td>One-hot</td>
    <td align="right">65.52</td>
    <td align="right">42.32</td>
    <td align="right">68.82</td>
    <td align="right">55.61</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right">57.58</td>
    <td align="right">42.72</td>
    <td align="right">75.02</td>
    <td align="right">53.96</td>
  </tr>
  <tr>
    <td rowspan="2">GGCU</td>
    <td>One-hot</td>
    <td align="right"><b>73.92</b></td>
    <td align="right"><b>52.10</b></td>
    <td align="right">72.42</td>
    <td align="right">55.04</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right">72.98</td>
    <td align="right">46.34</td>
    <td align="right"><b>76.30</b></td>
    <td align="right">54.65</td>
  </tr>
</table>


TFMixer is introduced for predicting knock-in events in ELS by incorporating macroeconomic data (FRED-MD, FRED-QD), asset prices, and contract terms. Unlike traditional methods that rely on short-term indicators, TFMixer analyzes both time (long-range price patterns) and feature dimensions (macroeconomic interactions) in parallel. Over a 128-week horizon, it achieves an F1 of 0.8957 and an AUROC of 0.9076, surpassing popular attention-based and recurrent models in balancing sensitivity and precision. This approach also reduces computational overhead, giving investors timely risk signals for better ELS management.

