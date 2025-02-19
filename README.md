# TFMixer for ELS Knock-In Prediction
This repository contains the Pytorch implementation code for the paper "TFMixer: A Hybrid Attention Model with
Macroeconomic Data for ELS Knock-In Prediction"

Knock-in events can profoundly impact ELS returns, yet contract terms alone often fall short for precise risk estimation. TFMixer addresses this gap by integrating macroeconomic data from FRED-MD and FRED-QD through a time–feature attention mechanism. It demonstrates robust knock-in detection (F1=0.8957, AUROC=0.9076), offering proactive insights for ELS risk management.

## 1. Architectures
Below is a conceptual diagram of how the model processes data in parallel along two axes:

<img src=https://github.com/dxlabskku/TFMixer/blob/main/model.png/>


## 1) Time-Dimension Path

Embeds the sequence of combined features across multiple time steps.
Uses a TimesBlock to detect different periodicities in the input, performing FFT-based frequency analysis and a set of Inception-like convolution layers.
Passes the time-embedded representation through multiple layers of multi-head self-attention and feedforward layers.

## 2) Feature-Dimension Path

Transposes the data to treat each feature as a “token,” applying multi-head self-attention over the feature axis.
Captures inter-feature relationships at every time step.

## 3) Final Decision

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


## Results
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

Testing accuracy of Link Prediction without node features are summarized below.

<table>
  <tr>
    <td><b>Method</b></td>
    <td><b>Feature</b></td>
    <td><b>Cora</b></td>
    <td><b>CiteSeer</b></td>
    <td><b>PubMed</b></td>
  </tr>
  <tr>
    <td rowspan="2">GCN</td>
    <td>One-hot</td>
    <td align="right">62.29</td>
    <td align="right">62.86</td>
    <td align="right">64.17</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right">76.06</td>
    <td align="right">79.49</td>
    <td align="right">79.66</td>
  </tr>
  <tr>
    <td rowspan="2">GAT</td>
    <td>One-hot</td>
    <td align="right">64.05</td>
    <td align="right">63.30</td>
    <td align="right">64.77</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right">77.80</td>
    <td align="right">80.11</td>
    <td align="right">80.17</td>
  </tr>
  <tr>
    <td rowspan="2">GraphSAGE</td>
    <td>One-hot</td>
    <td align="right">67.84</td>
    <td align="right">62.24</td>
    <td align="right">64.02</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right">75.59</td>
    <td align="right">78.15</td>
    <td align="right">76.58</td>
  </tr>
  <tr>
    <td rowspan="2">GIN</td>
    <td>One-hot</td>
    <td align="right">65.00</td>
    <td align="right">64.57</td>
    <td align="right">70.95</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right">76.00</td>
    <td align="right">75.80</td>
    <td align="right">76.90</td>
  </tr>
  <tr>
    <td rowspan="2">APPNP</td>
    <td>One-hot</td>
    <td align="right">50.00</td>
    <td align="right">50.00</td>
    <td align="right">50.00</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right">76.48</td>
    <td align="right">73.30</td>
    <td align="right">81.60</td>
  </tr>
  <tr>
    <td rowspan="2">JKNet</td>
    <td>One-hot</td>
    <td align="right">57.79</td>
    <td align="right">60.84</td>
    <td align="right">59.44</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right">76.29</td>
    <td align="right">74.25</td>
    <td align="right">76.84</td>
  </tr>
  <tr>
    <td rowspan="2">GCNII</td>
    <td>One-hot</td>
    <td align="right">55.87</td>
    <td align="right">58.35</td>
    <td align="right">59.55</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right">76.42</td>
    <td align="right">76.29</td>
    <td align="right">80.79</td>
  </tr>
  <tr>
    <td rowspan="2">GatedGCN</td>
    <td>One-hot</td>
    <td align="right">66.21</td>
    <td align="right">67.84</td>
    <td align="right">64.58</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right">77.95</td>
    <td align="right">73.85</td>
    <td align="right">77.01</td>
  </tr>
  <tr>
    <td rowspan="2">GGCU</td>
    <td>One-hot</td>
    <td align="right">70.02</td>
    <td align="right">68.00</td>
    <td align="right">69.73</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right">77.20</td>
    <td align="right">79.49</td>
    <td align="right">77.78</td>
  </tr>
  <tr>
    <td rowspan="2">GGCU $_{trends}$</td>
    <td>One-hot</td>
    <td align="right">70.68</td>
    <td align="right">68.64</td>
    <td align="right">69.97</td>
  </tr>
  <tr>
    <td>Deepwalk</td>
    <td align="right"><b>81.40</b></td>
    <td align="right"><b>81.96</b></td>
    <td align="right"><b>81.84</b></td>
  </tr>
</table>

## Usage
You can run node classification or link prediction with the one-hot representation through the following commands.

```
python train_node_classification.py
python train_link_prediction.py
```

You can use the following commands if you want to run with GPUs.

```
python train_node_classification.py device=cuda
python train_link_prediction.py device=cuda
```

You can replace the one-hot representation with the Deepwalk representation through the following commands.

```
python train_node_classification.py feature=deepwalk
python train_link_prediction.py feature=deepwalk
```

You can also run with GGCU $_{trends}$ in link prediction through the following commands.

```
python train_link_prediction.py method=trends
```

## Hyperparameters
The following hyperparameters are tuned with grid search.

Hyperparameters used in node classification with the one-hot representation of the Cora dataset, set as default values, are as follows:

<table>
  <tr>
    <td><b>Name</b></td>
    <td><b>Type</b></td>
    <td><b>Value</b></td>
  </tr>
  <tr>
    <td>alpha</td>
    <td>float</td>
    <td>0.4</td>
  </tr>
  <tr>
    <td>n_layer</td>
    <td>int</td>
    <td>10</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>float</td>
    <td>1e-3</td>
  </tr>
  <tr>
    <td>weight_decay</td>
    <td>float</td>
    <td>5e-4</td>
  </tr>
</table>

Hyperparameters used in link prediction with the one-hot representation of the Cora dataset are as follows:

<table>
  <tr>
    <td><b>Name</b></td>
    <td><b>Type</b></td>
    <td><b>Value</b></td>
  </tr>
  <tr>
    <td>alpha</td>
    <td>float</td>
    <td>0.1</td>
  </tr>
  <tr>
    <td>n_layer</td>
    <td>int</td>
    <td>10</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>float</td>
    <td>1e-3</td>
  </tr>
  <tr>
    <td>weight_decay</td>
    <td>float</td>
    <td>5e-4</td>
  </tr>
</table>

Hyperparameters of GGCU $_{trends}$ used in link prediction with the one-hot representation of the Cora dataset are as follows:

<table>
  <tr>
    <td><b>Name</b></td>
    <td><b>Type</b></td>
    <td><b>Value</b></td>
  </tr>
  <tr>
    <td>alpha</td>
    <td>float</td>
    <td>1.0</td>
  </tr>
  <tr>
    <td>n_layer</td>
    <td>int</td>
    <td>2</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>float</td>
    <td>1e-2</td>
  </tr>
  <tr>
    <td>weight_decay</td>
    <td>float</td>
    <td>5e-4</td>
  </tr>
</table>

You can change hyperparameters through the additional command "{name}={value}".

For example:

```
python train_node_classification.py alpha=0.2
```
