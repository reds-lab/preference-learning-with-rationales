# Data-Centric Human Preference Optimization with Rationales



<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://www.justhoanganh.com/" target="_blank" style="text-decoration: none;">Hoang Anh Just<sup>1</sup></a>&nbsp;,&nbsp;
    <a href="https://jinming.tech/" target="_blank" style="text-decoration: none;">Ming Jin<sup>1</sup></a>&nbsp;,&nbsp;
    <a href="https://anitksahu.github.io/" target="_blank" style="text-decoration: none;">Anit Sahu<sup>2</sup></a>&nbsp;,&nbsp;
    <a href="https://pquochuy.github.io/" target="_blank" style="text-decoration: none;">Huy Phan<sup>2</sup></a>&nbsp;,&nbsp;
    <a href="https://ruoxijia.info/" target="_blank" style="text-decoration: none;">Ruoxi Jia<sup>1</sup></a>&nbsp;&nbsp;
    <br/> 
    <sup>1</sup>Virginia Tech&nbsp;&nbsp;&nbsp;<sup>2</sup>Amazon&nbsp;&nbsp;&nbsp;
      <br/>
      Paper: <a href="https://arxiv.org/abs/2407.14477" target="_blank" style="text-decoration: none;">https://arxiv.org/abs/2407.14477</a>
</p>



This repository contains the code for our paper [Data-Centric Human Preference Optimization with Rationales](https://arxiv.org/abs/2407.14477). 

We propose to enhance existing preference learning frameworks with rationales to explain the reasons.


![preference_rationale/blob/main/wrationales.gif](https://github.com/reds-lab/preference-learning-with-rationales/blob/main/wrationales.gif)


## Dataset

We have generated the dataset based on the prompts in our paper.

We currently provide the rationale-enhanced datasets for the [Intel-ORCA-DPO-pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs):

+ with general rationales: https://huggingface.co/datasets/redsgnaoh/orcaratgen
+ with detailed rationales: https://huggingface.co/datasets/redsgnaoh/orcaratspec

## Running the Preference Learning with Rationales Code

To run any model with any dataset, please edit the configuration files and run the python file, e.g. to train Mistral-7B-Instruct-v0.2 with RDPO (DPO with Rationales) loss, we can run the following line:

```{python}
python train.py loss=rdpo ++loss.gamma=0.001 lr=5e-7 model=mistral7bv2 datasets=[orcaratspec] exp_name=rdpo_g0-001_lr5e-7_orcaratspec_mistral7b2 mode=train
```

Use ``rorpo-simple`` for the RORPO loss and ``rdpo`` for the RDPO loss.




## Sampling for AlpacaEval Evaluation


To sample the trained model to evaluate on the AlpacaEval 2.0 benchmark, please run the following prompt (with the above-trained model):

```{python}
python eval.py --config-path=data/rdpo_g0-001_lr5e-7_orcaratspec_mistral7b2 --config-name=config ++mode=alpacaeval ++n_samples=805 ++model.eval_batch_size=35 ++samples_dir=folder_for_alpaca_samples ++exp_name=your_exp_name
```

Feel free to evaluate on other prompts as well, following the code repository below.


## Codebase


The code is based on this repository: [https://github.com/ContextualAI/HALOs/](https://github.com/ContextualAI/HALOs/).
We greatly appreciate the authors for providing the user-friendly code.

For more details to edit the code, please check out their repository.

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
