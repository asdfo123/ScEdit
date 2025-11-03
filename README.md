# ScEdit: Script-based Assessment of Knowledge Editing
**✨ [ACL 2025 Findings] ✨**

[![arXiv](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2505.23291)
![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)



## Overview

Knowledge Editing (KE) aims to efficiently update information in Large Language Models (LLMs) without full retraining. Current KE evaluations often use simple, fact-based ("What"-type) questions, which may not reflect real-world performance.

ScEdit (Script-based Assessment of Knowledge Editing) introduces a novel framework that evaluates KE using action-based ("How"-type) questions. It assesses how well LLMs integrate updated knowledge into procedural, multi-step tasks (scripts). ScEdit covers both counterfactual and temporal edits, offering a more challenging and realistic assessment.

**Key Contributions:**
* A script-based assessment framework evaluating KE in complex reasoning and generation tasks. 
* The ScEdit benchmark, with datasets for counterfactual (ScEdit-CF) and temporal (ScEdit-T) edits. 
* Comprehensive evaluation using token-level and text-level metrics. 

## The ScEdit Framework

ScEdit evaluates KE by assessing a model's ability to generate or modify a **Script** (a step-by-step guide) in response to a **Script Question** (a "How"-type query) after a **Fact** (an $(s,r,o)$ triple) has been edited ($o^c \rightarrow o$).

**Evaluation Levels:**
* **Token-level:** Uses cloze-format prompts. Metrics include Efficacy Success (ES), Script-based ES (S-ES), Script-based Neighborhood Success (S-NS), and Script-based Bleedover (S-BO).
* **Text-level:** Assesses full script quality via automatic (GPT-4) and human evaluations on a 7-point Likert scale. Metrics: Executability (Exec.), Coherence (Coh.), Consistency (Cons.), and Completeness (Comp.).


## Repository Structure


The repository is organized as follows:
```
.
├── data/
│   ├── ScEdit-CF.json
│   └── ScEdit-T.json
└── src/
└── (Source code coming soon)
```

### `data/` Directory

This directory contains the core datasets for the ScEdit benchmark.

* **`ScEdit-CF.json`**: The CounterFactual dataset.
    * **Cases**: 1830
    * **Description**: Evaluates edits where facts are changed to counterfactual alternatives.
    * **Key Fields per Case**:
        * `case_id`: Unique identifier.
        * `prompt`: Original factual query.
        * `ground_truth` ($o^c$): Correct original answer.
        * `target_new` ($o$): New counterfactual answer.
        * `subject` ($s$): Main entity.
        * `rephrase_prompts` Script-based prompt.
        * `neighborhood_prompts`: Script-based neighborhood prompt.
        * `interrupt_step`: Step in `rephrase_prompts` where the fact is truncated.
        * `property`: Type of query.
        * `generation_prompts`: Scipt Qusetions.

* **`ScEdit-T.json`**: The Temporal dataset.
    * **Cases**: 1762
    * **Description**: Assesses adaptation to temporal edits.
    * **Key Fields per Case**:
        * `case_id`: Unique identifier.
        * `subject` ($s$): Main entity.
        * `relation` ($r$): Attribute being updated.
        * `prompt`: Original query.
        * `old_update` ($o^c$): Outdated value.
        * `new_update` ($o$): New, correct value.
        * `question` : Script-based prompt.
        * `neighborhood`: Neighbor facts (Each object represents a "neighboring" fact, typically involving a *similar subject* ($s'$) and the *same relation* ($r$), to test for unintended bleedover effects (S-BO). Contains sub-fields like `object`, `prompt`, and `question` for the neighbor).



### `src/` Directory

This directory will contain the source code for:
* Knowledge editing method implementations.

* *(Source code coming soon)*


## Editing Methods Evaluated

* Fine-tune (FT)
* Constrained Fine-Tuning (FT+L)
* ROME (Rank-One Model Editing)
* MEMIT (Mass-Editing Memory in a Transformer)
* MEND (Meta-learning for Fast, Localized Parameter Adjustments)
* PROMPT (In-context learning via prefixing)

## How to Use

Source code is coming soon. Our datasets are marginally based on the data structures of [`ROME`](https://github.com/kmeng01/rome) and 
[`MEMIT`](https://github.com/kmeng01/memit); you could refer to their implementations as a preliminary guide.

## Citation

If you use ScEdit in your research, please cite our paper:

```bibtex
@inproceedings{li2025ScEdit,
  title={ScEdit: Script-based Assessment of Knowledge Editing},
  author={Li, Xinye and Zheng, Zunwen and Zhang, Qian and Zhuang, Dekai and Kang, Jiabao and Xu, Liyan and Liu, Qingbin and Chen, Xi and Tu, Zhiying and Chu, Dianhui and Sui, Dianbo},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2025},
  year={2025}
}
```
## Thanks
Our work builds upon and is inspired by several excellent projects of Knowledge Editing. We extend our gratitude to the creators and maintainers of:

[`ROME`](https://github.com/kmeng01/rome)
[`MEMIT`](https://github.com/kmeng01/memit)
[`AlphaEdit`](https://github.com/jianghoucheng/AlphaEdit)
[`EasyEdit`](https://github.com/zjunlp/EasyEdit)
[`WikiFactDiff`](https://github.com/Orange-OpenSource/WikiFactDiff)
