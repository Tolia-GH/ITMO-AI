## 5. Physics-informed NNs, Generative Physics and Astronomy

Ilya Makarov
Team Lead
tg: https://t.me/iamakarov

Andrei Zakharov
Senior researcher
tg: https://t.me/and_rei_z
Daniil Sukhorukov
Researcher
tg: https://t.me/dsuh0i
Dmitry Zhevnenko
Senior researcher
tg: https://t.me/

### Description

**Physics-informed Neural Networks**

Aim: Generative AI with built-in physical consistency

Tasks:
- Develop and train models with new modalities for physical fields (e.g. symbolic and hybrid representations)
- Improve SOTA architectures Neural Operators, NeuralODE, PINNs, MPP
- Design algorithms for advanced data assimilation in PINNs, training stability (using RL), and better domain knowledge integration

Datasets: [The WELL](https://polymathic-ai.org/the_well/), [PDEBench](https://github.com/pdebench/PDEBench)

Links: Multiple Physics Pretraining (Polymathic, 24ʼ),
Geo-FNO (23ʼ), OmniArch (24ʼ), review (22ʼ)

Large multiphysics datasets
(simulations+experiments)

Physical Laws
(PDEs, conservation, constraints)

GenAI:
NNs (surrogate models),
Hybrid (complex) Loss,
Training with GenAI priors,
Multi-fidelity learning,
Inverse Design, ...

Physically consistent predictions

## 7. Foundation models for Satellite Imaging and Climate

Ilya Makarov
Team Lead
tg: https://t.me/iamakarov

Daniil Sukhorukov
Researcher
tg: https://t.me/dsuh0i

Daniil Storonkin
Researcher
tg: https://t.me/Daniil_Andreevich_S

### Seismic Foundation Models: From Reflection Volumes to Event Waveforms

**Description**
Seismic foundation models learn transferable representations from large-scale unlabeled geophysical data. SFM, NCS-Models, and ThinkOnward GFM cover reflection seismics through 2D sections, multi-view 2.5D slices, and 3D seismic volumes, while SeisCLIP targets passive seismology using three-component waveforms, time–frequency spectra, and event metadata. Their common core is Transformer-based self-supervision: masked reconstruction for spatial seismic data and contrastive alignment for waveform–metadata pairs. The resulting modular backbone can support seismic processing, geological interpretation, inversion, similarity search, and earthquake-analysis tasks.

**Tasks**
- Unify & tokenize seismic data. Build standardized loaders for 2D inline/crossline sections, 2.5D multi-view slices, 3D seismic cubes, three-component waveforms, spectrograms, and physical metadata. Harmonize amplitude normalization, spatial geometry, sampling rates, acquisition parameters, masks, and missing-data patterns.
- Pretrain transferable representations. Compare 2D, 2.5D, and 3D ViT-MAE backbones; combine conventional patch masking with physically meaningful trace and volume masking. Introduce SeisCLIP-style contrastive learning to align seismic signals with phase, source, survey, and geological metadata.
- Adapt & evaluate across tasks. Use frozen probing, lightweight adapters, and full fine-tuning for trace interpolation, denoising, seismic facies and geobody segmentation, fault and horizon tracking, reflectivity inversion, similarity search, event classification, localization, and focal-mechanism estimation. Evaluate reconstruction quality, segmentation accuracy, retrieval performance, and cross-survey or cross-region generalization.

**Links**
- SFM: [Paper](https://arxiv.org/abs/2309.02791) · [GitHub, weights and datasets](https://github.com/shenghanlin/SeismicFoundationModel)
- NCS-Models: [Paper](https://arxiv.org/abs/2603.23211) · [2D/2.5D/3D models](https://huggingface.co/collections/NorskRegnesentralSTI/ncs-models?utm_source=chatgpt.com) · [GitHub](https://github.com/NorskRegnesentral/NCS_models)
- ThinkOnward GFM: [Model](https://huggingface.co/thinkonward/geophysical-foundation-model) · [Patch the Planet dataset](https://huggingface.co/datasets/thinkonward/patch-the-planet) · [GitHub](https://github.com/thinkonward/geophysical-foundation-model)
- SeisCLIP: [Paper](https://arxiv.org/abs/2309.02320) · [GitHub and weights](https://github.com/sixu0/SeisCLIP) · [STEAD dataset](https://github.com/smousavi05/STEAD)

### Interpretable Weather Forecasting Models

**Description**
Interpretable weather forecasting aims to explain why a deep learning model produces a particular prediction of a cyclone, heatwave, strong precipitation, or other extreme event. We focus on transformer-based weather models and neural foundation models, analysing which input variables, pressure levels, geographical regions, and previous timesteps contribute to the forecast. Attribution methods, latent-space analysis, and physically meaningful concepts are used to improve model transparency, identify forecast errors, and evaluate whether neural models learn realistic atmospheric relationships.

**Tasks**
- Benchmarks & case studies: prepare ERA5-based test cases for cyclones, heatwaves, atmospheric rivers, and precipitation extremes; select target variables, forecast horizons, and geographical regions.
- Attribution methods: implement saliency maps, Integrated Gradients, occlusion/RISE, Layer-wise Relevance Propagation, and attention-based explanations for transformer weather models.
- Variable and spatial importance: estimate the contribution of atmospheric variables, pressure levels, input timesteps, and geographical regions to individual forecasts.
- Concept & latent analysis: identify representations associated with cyclones, fronts, atmospheric circulation regimes, moisture transport, seasonality, and other physically meaningful structures.
- Validation of explanations: perform perturbation, feature-removal, and model-randomization tests; evaluate faithfulness, robustness, localization, and consistency across forecast lead times.

**Links**
- [Corrformer (Nature Machine Intelligence, 2023)](https://www.nature.com/articles/s42256-023-00667-9)
- [Aurora Latent Regime Analysis and Attribution (arXiv, 2026)](https://arxiv.org/abs/2606.26361)
- [Interpretable ML for Weather and Climate: A Survey (arXiv, 2024)](https://arxiv.org/abs/2403.18864)
- [Concept Bottleneck Time-Series Transformers (arXiv / ICLR 2025 submission)](https://arxiv.org/abs/2410.06070)

### Description

Adapt the state-of-the-art Visual Language Models VLMs to the task of geospatial imagery analysis. This project focuses on pretraining and finetuning general-purpose image VLMs to remote-sensing applications that require spatial reasoning and interpretability. Representative downstream tasks include visual question answering, scene classification, and visual grounding.

### Tasks:

1. Collect datasets and train baseline models to match the current SOTA. (Done)
2. Improve the baseline GeoVLM (e.g. architecture tweaks, dataset curation).
3. Design and implement the robust pipeline for model evaluation.

### Links:
1. [GeoChat: Grounded Large Vision-Language Model for Remote Sensing](https://github.com/mbzuai-oryx/GeoChat)
2. [RSEvalKit](https://github.com/fitzpchao/RSEvalKit)
3. [LHRS-Bot: Empowering Remote Sensing with VGI-Enhanced Large Multimodal Language Model](https://github.com/NJU-LHRS/LHRS-Bot)

## 8. Predictive maintenance

Ilya Makarov
Team Lead
tg: https://t.me/iamakarov

Aleksandr Kovalenko
Junior researcher
tg: https://t.me/aekovalenko
Dmitry Zhevnenko
Senior researcher
tg: https://t.me/DmitryZhev
Andrei Zakharov
Senior researcher
tg: https://t.me/and_rei_z

### Description

Predictive maintenance is an advanced approach in industrial data analytics that uses sensor data to forecast equipment conditions.

This data typically consists of multivariate time series, capturing parameters such as vibration, temperature, pressure, and other operational metrics.

Machine learning and deep learning techniques enable predictive maintenance to address key challenges: diagnosing equipment health, detecting anomalies and early failure signs, and estimating components' remaining useful life (RUL).

By analyzing these patterns, predictive maintenance facilitates proactive maintenance—reducing unplanned downtime, optimizing repair schedules, and preventing costly breakdowns.

### Directions

**Graph Neural Networks**
GNNs take into account information about the relationships between equipment components

**Self-supervised learning**
SSL methods enable learning from unlabeled and sparsely labeled data

**Remaining Useful Life**
RUL techniques enable the forecasting of the remaining useful life of equipment units

### AI for advanced battery science

Aim: Smarter longer-lasting batteries using data-driven models

Tasks:
- Bild DL models from real-world battery data (in collaboration with Hong Kong University of Science and Technology)
- Develop surrogate AI-tools for battery behavior integrating physics-based and data-driven methods
- Design and implement anomaly detection pipelines
- Explore cross-domain learning for new chemistries
- Present results in reports and academic writing

Useful links:
- [BatteryLife: A Comprehensive Dataset and Benchmark](https://arxiv.org/pdf/2502.18807)
- [Battery safety: Machine learning-based prognostics](https://www.sciencedirect.com/science/article/pii/S0360128523000722)