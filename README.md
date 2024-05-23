# CodeSearcher

## Paper description

This work addresses the task of finding source code given its description. Such a tool could be useful for both experienced and novice programmers for development and learning augmentation. Moreover, this tool could help gather specifically themed code data sets. 

Furthermore, our paper addresses the issue highlighted in [Scaling Down to Scale Up](https://arxiv.org/pdf/2303.15647.pdf): the lack of sufficient bench-marking for Parameter-Efficient Fine-Tuning (PEFT) methods. More specifically, we have developed a framework for [CodeT5+](https://arxiv.org/pdf/2305.07922.pdf) efficient fine-tuning and evaluation on various programming language data sets, providing the checkpoints for them.

## Installation 

1. Clone this repo
2. Run `pip install -r requirements.txt`
3. Run `streamlit run app.py`


## Links and news

1. [Paper preprint](http://arxiv.org/abs/2405.04126)
2. Accepted for AINL-2024, recieved best paper award

## Credits

We would like to express our thanks to Innopolis University for providing part of resources and facilities that were essential for conducting the experiments in this work. We extend our sincere gratitude to Professor V. Ivanov from Innopolis University for his invaluable guidance and support throughout our research. 

Project developed and done by:
* Leila Khaertdinova, l.khaertdinova@innopolis.university
* Karim Galliamov, k.galliamov@innopolis.university
* Karina Denisova, k.denisova@innopolis.university
