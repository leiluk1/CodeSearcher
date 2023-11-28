# CodeSearcher

## Problem statement and overview

This project addresses the task of finding source code given its description. Such a tool could be useful for both experienced and novice programmers for development and learning augmentation. Moreover, this tool could help gather specifically themed code data sets. 

Furthermore, our project addresses the issue highlighted in [Scaling Down to Scale Up](https://arxiv.org/pdf/2303.15647.pdf): the lack of sufficient bench-marking for Parameter-Efficient Fine-Tuning (PEFT) methods. More specifically, we have developed a framework for [CodeT5+](https://arxiv.org/pdf/2305.07922.pdf) efficient fine-tuning and evaluation on various programming language data sets, providing the checkpoints for them.

## Installation 

1. Clone this repo
2. Run `pip install -r src/requirements.txt`
3. Run `streamlit run app.py`


## Links
1. [Colab with model serving](https://colab.research.google.com/drive/146d-8ngKj4Ox7fuXGXjoCf__v3wmeEAr?authuser=1#scrollTo=1ruac9acOd11)

## Credits

This project is intended for the project part in the Practical Machine Learning and Deep Learning course at Innopolis University.

Project developed and done by:
* Leila Khaertdinova, l.khaertdinova@innopolis.university
* Karim Galliamov, k.galliamov@innopolis.university
* Karina Denisova, k.denisova@innopolis.university
