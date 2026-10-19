from __future__ import annotations
import os
import torch
from torch.utils.data import DataLoader, default_collate
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor
from data.benchmarks import load_images
from data.texts import class_texts
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_ROOT = os.environ.get('RANKALIGN_EMB_ROOT', os.path.join(ROOT, 'embeddings'))
DATASET_DIR = {'cifar100': 'CIFAR-100', 'imagenet100': 'ImageNet-100'}
VISION_FILES = {'clip': 'clip.pt', 'siglip': 'siglip.pt', 'siglip2': 'siglip2.pt', 'deit': 'deit.pt', 'dinov2': 'dinov2.pt'}
LANGUAGE_FILES = {'bge': 'bge.pt', 'e5': 'e5.pt'}

def dataset_dir(emb_root: str, dataset: str) -> str:
    return os.path.join(emb_root, DATASET_DIR[dataset])

def _require(path: str, kind: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Missing precomputed {kind}: {path}\nRun rankalign/run.py with --data_root / --imagenet_val to generate it. See README.md for the expected layout.')
    return path

def vision_path(emb_root: str, dataset: str, name: str) -> str:
    if name not in VISION_FILES:
        raise ValueError(f"Unknown vision representation '{name}'. Choose from: {', '.join(VISION_FILES)}")
    return _require(os.path.join(dataset_dir(emb_root, dataset), 'vision', VISION_FILES[name]), f"vision representation '{name}'")

def language_path(emb_root: str, dataset: str, name: str) -> str:
    if name not in LANGUAGE_FILES:
        raise ValueError(f"Unknown language representation '{name}'. Choose from: {', '.join(LANGUAGE_FILES)}")
    return _require(os.path.join(dataset_dir(emb_root, dataset), 'language', LANGUAGE_FILES[name]), f"language representation '{name}'")

def labels_path(emb_root: str, dataset: str) -> str:
    return _require(os.path.join(dataset_dir(emb_root, dataset), 'labels.pt'), 'labels')
(IMAGENET_MEAN, IMAGENET_STD) = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

class Backbone:

    def build(self, device):
        raise NotImplementedError

    def embed(self, model, x):
        return model(x)

class _HFImageTransform:

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, img):
        return self.processor(images=img.convert('RGB'), return_tensors='pt')['pixel_values'][0]

class CLIPBackbone(Backbone):
    name = 'ViT-L/14@336px'

    def build(self, device):
        import clip
        (model, preprocess) = clip.load(self.name, device='cpu')
        return (model.float(), preprocess)

    def embed(self, model, x):
        return model.encode_image(x)

class SigLIPBackbone(Backbone):

    def __init__(self, hf_name):
        self.hf_name = hf_name

    def build(self, device):
        from transformers import AutoImageProcessor, AutoModel
        model = AutoModel.from_pretrained(self.hf_name, torch_dtype=torch.float32)
        processor = AutoImageProcessor.from_pretrained(self.hf_name)
        return (model, _HFImageTransform(processor))

    def embed(self, model, x):
        out = model.get_image_features(pixel_values=x)
        return out if torch.is_tensor(out) else out.pooler_output

class DeiTBackbone(Backbone):
    hub_name = 'deit_base_distilled_patch16_384'
    resolution = 384

    def build(self, device):
        model = torch.hub.load('facebookresearch/deit:main', self.hub_name, pretrained=True)
        transform = Compose([Resize((self.resolution, self.resolution), interpolation=InterpolationMode.BICUBIC, antialias=True), ToTensor(), Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        return (model, transform)

class DINOv2Backbone(Backbone):
    hub_name = 'dinov2_vitg14'
    resolution = 224

    def build(self, device):
        model = torch.hub.load('facebookresearch/dinov2', self.hub_name, pretrained=True)
        transform = Compose([Resize((self.resolution, self.resolution), antialias=True), ToTensor(), Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        return (model, transform)

    def embed(self, model, x):
        return model.forward_features(x)['x_norm_patchtokens'].mean(dim=1)
BACKBONES = {'clip': CLIPBackbone, 'siglip': lambda : SigLIPBackbone('google/siglip-base-patch16-224'), 'siglip2': lambda : SigLIPBackbone('google/siglip2-base-patch16-224'), 'deit': DeiTBackbone, 'dinov2': DINOv2Backbone}

def _images_only(batch):
    return default_collate([x for (x, _) in batch])

@torch.inference_mode()
def extract(backbone, dataset, device, batch_size, num_workers):
    (model, transform) = backbone.build(device)
    model = model.to(device).eval()
    dataset.transform = transform
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=device.startswith('cuda'), collate_fn=_images_only)
    feats = []
    for (i, x) in enumerate(loader):
        feats.append(backbone.embed(model, x.to(device)).float().cpu().double())
        if i % 50 == 0:
            print(f'    batch {i + 1}/{len(loader)}', flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(feats, dim=0)

def generate_vision(datasets, models, emb_root, data_root=None, imagenet_val=None, device=None, batch_size=16, num_workers=4):
    device = device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    for ds_name in datasets:
        ds = load_images(ds_name, data_root, imagenet_val)
        out_dir = os.path.join(emb_root, DATASET_DIR[ds_name])
        os.makedirs(os.path.join(out_dir, 'vision'), exist_ok=True)
        labels = torch.as_tensor(ds.targets, dtype=torch.long)
        lpath = os.path.join(out_dir, 'labels.pt')
        if os.path.exists(lpath):
            if not torch.equal(torch.load(lpath, map_location='cpu').long().reshape(-1), labels):
                raise RuntimeError(f'existing {lpath} does not match the image order of {ds_name}')
        else:
            torch.save(labels, lpath)
        print(f'[vision] {ds_name}: {len(ds)} images, {len(torch.unique(labels))} classes', flush=True)
        for m in models:
            out = os.path.join(out_dir, 'vision', f'{m}.pt')
            print(f'  {m} on {device}', flush=True)
            V = extract(BACKBONES[m](), ds, device, batch_size, num_workers)
            assert V.shape[0] == len(labels)
            torch.save(V.contiguous(), out)
            print(f'  saved {out} {tuple(V.shape)}', flush=True)
ENCODERS = {'bge': 'BAAI/bge-large-en-v1.5', 'e5': 'intfloat/e5-large-v2'}
QUERY_PREFIX = {'e5': 'query: '}

def class_prototypes(text_emb: torch.Tensor) -> torch.Tensor:
    E = text_emb.double()
    E = E / E.norm(dim=-1, keepdim=True)
    E = E.nanmean(dim=1)
    return E / E.norm(dim=-1, keepdim=True)

@torch.inference_mode()
def encode(model, texts: list[list[str]], prefix: str, batch_size: int) -> torch.Tensor:
    (C, P) = (len(texts), len(texts[0]))
    flat = [prefix + t for row in texts for t in row]
    E = model.encode(flat, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
    return E.cpu().double().view(C, P, -1)

def generate_language(datasets, models, emb_root, device=None, batch_size=64):
    from sentence_transformers import SentenceTransformer
    device = device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    for name in models:
        model = SentenceTransformer(ENCODERS[name], device=device)
        model.eval()
        print(f'[language] {name}: {ENCODERS[name]} on {device}', flush=True)
        for ds in datasets:
            out = os.path.join(emb_root, DATASET_DIR[ds], 'language', f'{name}.pt')
            L = class_prototypes(encode(model, class_texts(ds), QUERY_PREFIX.get(name, ''), batch_size))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            torch.save(L.contiguous(), out)
            print(f'  {ds}: saved {out} {tuple(L.shape)}', flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def ensure_representations(datasets, vision_names, language, emb_root, data_root=None, imagenet_val=None):
    for ds in datasets:
        d = dataset_dir(emb_root, ds)
        missing_vision = [v for v in vision_names if not os.path.isfile(os.path.join(d, 'vision', VISION_FILES[v]))]
        missing_labels = not os.path.isfile(os.path.join(d, 'labels.pt'))
        missing_language = not os.path.isfile(os.path.join(d, 'language', LANGUAGE_FILES[language]))
        if missing_vision or missing_labels:
            print(f"{ds}: generating vision representations {missing_vision or ['labels']}", flush=True)
            generate_vision([ds], missing_vision, emb_root, data_root=data_root, imagenet_val=imagenet_val)
        if missing_language:
            print(f'{ds}: generating language representation {language}', flush=True)
            generate_language([ds], [language], emb_root)
