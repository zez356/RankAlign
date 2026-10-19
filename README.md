# RankAlign: Vision-Language Representation Alignment via Rank Transformation

[**Enzhe Zhao**](https://openreview.net/profile?id=~Enzhe_Zhao1)<sup>1,2*</sup> &nbsp;
[**Marco Fiorucci**](https://openreview.net/profile?id=~Marco_Fiorucci1)<sup>1*</sup> &nbsp;
[**Lamberto Ballan**](https://openreview.net/profile?id=~Lamberto_Ballan3)<sup>1</sup>

<sup>1</sup>University of Padova, Italy &nbsp; <sup>2</sup>Harbin Institute of Technology, China &nbsp; <sup>*</sup>Equal contribution


[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg)](https://openreview.net/forum?id=ngwv68jt7S)
[![OpenReview](https://img.shields.io/badge/OpenReview-Paper-8c1b13.svg)](https://openreview.net/forum?id=ngwv68jt7S)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) -->
<!-- [![Project Page](https://img.shields.io/badge/Project-Page-grey)](https://PROJECT_PAGE_URL) -->
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

## Abstract

While the Platonic Representation Hypothesis posits that vision and language models converge toward compatible latent geometries, recovering class-level correspondences between unpaired modalities without cross-modal supervision remains remains challenging.
Existing relational matching methods compare within-modality similarity kernels directly, although the same semantic structure may be represented through different numericalparameterizations across modalities. We propose RankAlign, a rank-based normalization strategy for class-level unpaired vision-language matching.
RankAlign transforms each within-modality similarity kernel into an ordinal relational representation before permutation matching, while leaving the correspondence space and matching solver unchanged. The resulting representation is invariant to strictly increasing reparameterizations of similarity values while preserving the ordinal structure used for matching. We provide exact-recovery and stability guarantees that characterize correspondence recovery under both exact and partial ordinal agreement. Experiments across diverse frozen vision and language encoders and multiple datasets demonstrate improved and more stable class-level correspondence recovery, including under label corruption and domain shift.

## Environment setup

```bash
git clone https://github.com/zez356/RankAlign.git
cd RankAlign
pip install numpy scipy torch torchvision sentence-transformers transformers timm POT
pip install git+https://github.com/openai/CLIP.git
```

## Usage

All experiments are run with `rankalign/run.py`. 
A minimal example on CIFAR-100 with DiNOv2:

```bash
python -u rankalign/run.py --dataset cifar100 --vision dinov2  --sizes 10 --data_root /path/to/data --out results/demo.csv
```


The main options can be combined freely, and several values can be given at once:

- `--dataset`: `cifar100`, `imagenet100`
- `--vision`: `clip`, `siglip`, `siglip2`, `deit`, `dinov2`
- `--language`: `bge`, `e5`
- `--solver`: `faq`, `ot`, `fhg` (adapted from [itsamatch](https://github.com/dominik-schnaus/itsamatch))
- `--sizes`: numbers of classes N to match (`10 20 ... 100`)
- `--seeds`: number of random repetitions



### Reproducing the results

Matching accuracy versus problem size on CIFAR-100 and ImageNet-100 (Fig.2):

```bash
python -u rankalign/run.py \
  --dataset cifar100 imagenet100 \
  --vision clip siglip siglip2 deit dinov2 \
  --solver fhg --timelimit 300 \
  --sizes 10 20 30 40 50 60 70 80 90 100 \
  --seeds 5 \
  --data_root /path/to/data \
  --out results/scalability.csv
```



## LAION

LAION pseudo-classes are built from the [LAION-Aesthetic square-cleaned](https://huggingface.co/datasets/opendiffusionai/laion2b-en-aesthetic-square-cleaned)
split (20 classes); 

##  CheXpert
CheXpert from the public [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) release
(12 single-label pathologies). Subset construction is in `data/extra.py`.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{zhao2026rankalign,
  title     = {RankAlign: Vision-Language Representation Alignment via Rank Transformation},
  author    = {Zhao, Enzhe and Fiorucci, Marco and Ballan, Lamberto},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=ngwv68jt7S}
}
```

## License

This project is released under the MIT License (see `LICENSE`).
