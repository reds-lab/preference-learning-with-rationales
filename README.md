# Data-Centric Human Preference Optimization with Rationales




This repository contains the code and released models for our paper [Data-Centric Human Preference Optimization with Rationales](https://arxiv.org/abs/2407.14477). We propose to enhance existing preference learning frameworks with rationales.


![Alt Text](https://github.com/REDSgnaoh/preference_rationale/blob/main/preference_learning_w_rats.gif)


## Dataset

We have generated the dataset based on the prompts in our paper.

We currently provide the rationale-enhanced datasets for the [Intel-ORCA-DPO-pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs):

+ with general rationales: https://huggingface.co/datasets/redsgnaoh/orcaratgen
+ with detailed rationales: https://huggingface.co/datasets/redsgnaoh/orcaratspec

## Running the Preference Learning with Rationales Code

To run any model with any dataset, please edit the configuration files and run the python:

```{python}
python train.py loss=rdpo ++loss.gamma=0.001 lr=5e-7 model=mistral7bv2 datasets=[orcaratspec] exp_name=rdpo_g0-001_lr5e-7_orcaratspec_mistral7b2 mode=train
```

Use ``rorpo-simple`` for the RORPO loss.




## Sampling for AlpacaEval Evaluation


To sample the trained model to evaluate on the AlpacaEval 2.0 benchmark, please run the following prompt:

```{python}
python eval.py --config-path=data/rdpo_g0-001_lr5e-7_orcaratspec_mistral7b2 --config-name=config ++mode=alpacaeval ++n_samples=805 ++model.eval_batch_size=35 ++samples_dir=folder_for_alpaca_samples ++exp_name=your_exp_name
```

Feel free to sample for other prompts as well, following the code repository below.


## Codebase


The code is based on this repository: [https://github.com/ContextualAI/HALOs/](https://github.com/ContextualAI/HALOs/).
We appreciate the authors for providing the user-friendly code.

For more details to edit the code, please check out this repository.

## Bugs/Questions

Please feel free to contact us for any questions, suggestions, and comments. Thank you for your help!

## Citation
Please cite our paper if you find the repo helpful in your work:

```bibtex
@misc{just2024datacentrichumanpreferenceoptimization,
      title={Data-Centric Human Preference Optimization with Rationales}, 
      author={Hoang Anh Just and Ming Jin and Anit Sahu and Huy Phan and Ruoxi Jia},
      year={2024},
      eprint={2407.14477},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.14477}, 
}
```
