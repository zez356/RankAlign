# RankAlign: Vision-Language Representation Alignment via Rank Transformation

## Abstract

The Platonic Representation Hypothesis suggests that vision and language models may converge toward compatible latent
structures, yet recovering class-level correspondences between unpaired modalities without cross-modal supervision
remains challenging. Existing relational matching methods compare within-modality similarity kernels directly, although
the same semantic structure may be represented through different numerical parameterizations across modalities. We
propose RankAlign, a rank-based normalization strategy for class-level unpaired vision-language matching. RankAlign
transforms each within-modality similarity kernel into an ordinal relational representation before permutation
matching, while leaving the correspondence space and matching solver unchanged. The resulting representation is
invariant to strictly increasing reparameterizations of similarity values while preserving the ordinal structure used
for matching. We provide exact-recovery and stability guarantees that characterize correspondence recovery under both
exact and partial ordinal agreement. Experiments across diverse frozen vision and language encoders and multiple
datasets demonstrate improved and more stable class-level correspondence recovery, including under label corruption
and domain shift.

## Installation

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



### Reproducing the paper

Matching accuracy versus problem size on CIFAR-100 and ImageNet-100:

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
