#Overview

Code and datasets of our paper [Aspect-oriented Opinion Alignment Network for Aspect-Based Sentiment Classification](https://ebooks.iospress.nl/volumearticle/64368) accepted by ECAI 2023.

## Requirements

numpy>=1.13.3
torch>=0.4.0
transformers>=3.5.1,<4.0.0
sklearn

To install requirements, run `pip install -r requirements.txt`.

## Training

To train the AOAN model, run:

`python train.py`

## Credits

The code and datasets in this repository are based on [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch) .

## Citation
Xueyi Liu, Rui Hou, Yanglei Gan, Da Luo, Changlin Li, Xiaojun Shi, Qiao Liu. "Aspect-oriented Opinion Alignment Network for Aspect-Based Sentiment Classification." In Proceedings of the 26th European Conference on Artificial Intelligence (ECAI), pp. 1552-1559. 2023. 


@inproceedings{liu2023Aspect,
  title={Aspect-oriented Opinion Alignment Network for Aspect-Based Sentiment Classification},
  author={Liu, Xueyi and Hou, Rui and Gan, Yanglei and Luo, Da and Li, Changlin and Shi, Xiaojun and Liu, Qiao},
  booktitle={Proceedings of the 26th European Conference on Artificial Intelligence (ECAI)},
  pages={1552--1559},
  year={2023}
}
