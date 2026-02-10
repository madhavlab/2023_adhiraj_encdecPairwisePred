## About

This repository contains the implementation of the paper "Enc-Dec RNN Acoustic Word Embeddings learned via Pairwise Prediction" [Adhiraj Banerjee, Vipul Arora].



> Paper: [Enc-Dec RNN Acoustic Word Embeddings learned via Pairwise Prediction](https://www.isca-archive.org/interspeech_2023/banerjee23_interspeech.pdf)


## Setup

#### Environment Setup (using uv)

This project uses **uv** for fast and reproducible Python environment management.

1. Install uv

**macOS/Linux**
```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

**Using pip**

```bash
pip install uv
```


#### Clone the Repository
```sh
git clone https://github.com/madhavlab/2023_adhiraj_encdecPairwisePred 
cd 2023_adhiraj_encdecPairwisePred
```




## Usage


### Training




Run :

```bash
uv run main.py --ckpt_dir /path/to/savedir
```

---





## Datasets 

- **Dataset**: [LibriSpeech Word Alignments](https://github.com/CorentinJ/librispeech-alignments)

## Citation

If you find our work useful, please cite:
```sh
@inproceedings{banerjee2023enc,
  title={Enc-Dec RNN Acoustic Word Embeddings learned via Pairwise Prediction},
  author={Banerjee, Adhiraj and Arora, Vipul},
  booktitle={Proc. Interspeech 2023},
  pages={1478--1482},
  year={2023}
}

```



