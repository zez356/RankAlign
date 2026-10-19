import hashlib
import os
import random
import re
import pandas as pd

LAION_CLASSES = ['flower', 'dress', 'table', 'hat', 'cake', 'tree', 'shoe', 'car', 'bag', 'bread', 'chair', 'bed', 'beach', 'motorcycle', 'bicycle', 'bird', 'dog', 'book', 'camera', 'horse']
MIN_SIDE = 256
MIN_SIMILARITY = 0.2
MAX_PUNSAFE = 0.25
MAX_PWATERMARK = 0.6
MIN_AESTHETIC = 6.0
LAION_RULES = {'flower': {'include': ['\\bflower\\b', '\\bflowers\\b', '\\bbouquet\\b', '\\brose\\b', '\\btulip\\b'], 'exclude': ['\\bflower pattern\\b', '\\bflower print\\b', '\\bflower wallpaper\\b', '\\bflower logo\\b', '\\bflower drawing\\b']}, 'dress': {'include': ['\\bdress\\b', '\\bgown\\b', '\\bevening dress\\b', '\\bwedding dress\\b'], 'exclude': ['\\bdress pattern\\b', '\\bdress logo\\b', '\\bdress drawing\\b']}, 'table': {'include': ['\\btable\\b', '\\bdining table\\b', '\\bwooden table\\b', '\\bdesk table\\b'], 'exclude': ['\\bperiodic table\\b', '\\btable chart\\b', '\\btable template\\b', '\\btable logo\\b', '\\btable drawing\\b']}, 'hat': {'include': ['\\bhat\\b', '\\bcap\\b', '\\bbaseball cap\\b', '\\bsun hat\\b'], 'exclude': ['\\bhat logo\\b', '\\bhat drawing\\b']}, 'cake': {'include': ['\\bcake\\b', '\\bbirthday cake\\b', '\\bwedding cake\\b', '\\bchocolate cake\\b'], 'exclude': ['\\bcake topper\\b', '\\bcake stand\\b', '\\bcake box\\b', '\\bcake logo\\b', '\\bcake drawing\\b']}, 'tree': {'include': ['\\btree\\b', '\\btrees\\b'], 'exclude': ['\\btree diagram\\b', '\\btree logo\\b', '\\btree drawing\\b']}, 'shoe': {'include': ['\\bshoe\\b', '\\bshoes\\b', '\\bsneaker\\b', '\\bsneakers\\b', '\\bboot\\b', '\\bboots\\b', '\\bsandal\\b', '\\bsandals\\b'], 'exclude': ['\\bshoe rack\\b', '\\bshoe box\\b', '\\bshoe pattern\\b', '\\bshoe logo\\b', '\\bshoe drawing\\b']}, 'car': {'include': ['\\bcar\\b', '\\bautomobile\\b', '\\bsedan\\b', '\\bcoupe\\b', '\\bsuv\\b'], 'exclude': ['\\btoy car\\b', '\\brail car\\b', '\\bcar logo\\b', '\\bcar drawing\\b', '\\bcar icon\\b', '\\bcarpet\\b', '\\bcard\\b']}, 'bag': {'include': ['\\bbag\\b', '\\bhandbag\\b', '\\bpurse\\b', '\\btote bag\\b', '\\bshoulder bag\\b', '\\bcrossbody bag\\b', '\\bbackpack\\b'], 'exclude': ['\\bgift bag\\b', '\\bbag pattern\\b', '\\bbag logo\\b', '\\bbag drawing\\b', '\\bbag icon\\b']}, 'bread': {'include': ['\\bbread\\b', '\\bloaf\\b', '\\bloaf of bread\\b', '\\bbaguette\\b'], 'exclude': ['\\bbreadboard\\b', '\\bbread logo\\b', '\\bbread drawing\\b']}, 'chair': {'include': ['\\bchair\\b', '\\barmchair\\b', '\\boffice chair\\b', '\\bdining chair\\b'], 'exclude': ['\\bchair cover\\b', '\\bchair cushion\\b', '\\bchair pattern\\b', '\\bchair logo\\b', '\\bchair drawing\\b']}, 'bed': {'include': ['\\bbed\\b', '\\bbed frame\\b'], 'exclude': ['\\bbedroom\\b', '\\bbed sheet\\b', '\\bbed logo\\b', '\\bbed drawing\\b']}, 'beach': {'include': ['\\bbeach\\b', '\\bseaside\\b', '\\bbeach scene\\b'], 'exclude': ['\\bbeach towel\\b', '\\bbeach logo\\b', '\\bbeach drawing\\b']}, 'motorcycle': {'include': ['\\bmotorcycle\\b', '\\bmotorbike\\b'], 'exclude': ['\\bmotorcycle logo\\b', '\\bmotorcycle drawing\\b']}, 'bicycle': {'include': ['\\bbicycle\\b', '\\bbike\\b', '\\broad bike\\b', '\\bmountain bike\\b'], 'exclude': ['\\bbike rack\\b', '\\bbike logo\\b', '\\bbike drawing\\b']}, 'bird': {'include': ['\\bbird\\b', '\\bparrot\\b', '\\bsparrow\\b'], 'exclude': ['\\bbird logo\\b', '\\bbird drawing\\b']}, 'dog': {'include': ['\\bdog\\b', '\\bpuppy\\b', '\\bcanine\\b'], 'exclude': ['\\bdog bed\\b', '\\bdog toy\\b', '\\bdog pattern\\b', '\\bdog logo\\b', '\\bdog drawing\\b', '\\bdog tag\\b']}, 'book': {'include': ['\\bbook\\b', '\\bnovel\\b', '\\bopen book\\b'], 'exclude': ['\\bbook cover\\b', '\\bbook mockup\\b', '\\bbook logo\\b', '\\bbook drawing\\b']}, 'camera': {'include': ['\\bcamera\\b', '\\bdslr\\b', '\\bmirrorless camera\\b'], 'exclude': ['\\bcamera icon\\b', '\\bcamera drawing\\b', '\\bcamera logo\\b']}, 'horse': {'include': ['\\bhorse\\b'], 'exclude': ['\\bhorse logo\\b', '\\bhorse drawing\\b']}}
LAION_EXCLUDE = ['[ぁ-んァ-ン一-龯]', '[가-힣]', '[а-яА-Я]', '\\blogo\\b', '\\bicon\\b', '\\bclipart\\b', '\\billustration\\b', '\\bdrawing\\b', '\\bsketch\\b', '\\bsticker\\b', '\\bposter\\b', '\\bwallpaper\\b', '\\btemplate\\b', '\\bmockup\\b']

def clean(text):
    return re.sub('\\s+', ' ', str(text or '')).strip()

def looks_english(text, min_ratio=0.65):
    letters = [ch for ch in text if ch.isalpha()]
    return bool(letters) and sum(('a' <= ch.lower() <= 'z' for ch in letters)) / len(letters) >= min_ratio

def any_match(patterns, text):
    return any((re.search(p, text, flags=re.IGNORECASE) for p in patterns))

def class_score(text, cls):
    text = clean(text).lower()
    if not text or not looks_english(text) or len(text.split()) > 48 or any_match(LAION_EXCLUDE, text):
        return 0
    if any_match(LAION_RULES[cls]['exclude'], text):
        return 0
    return sum((bool(re.search(p, text, flags=re.IGNORECASE)) for p in LAION_RULES[cls]['include']))

def assign_class(caption, moondream):
    scores = {c: class_score(caption, c) + 2 * class_score(moondream, c) for c in LAION_CLASSES}
    ranked = sorted(((s, c) for (c, s) in scores.items() if s > 0), reverse=True)
    if not ranked or (len(ranked) > 1 and ranked[1][0] == ranked[0][0]):
        return None
    return ranked[0][1]

def passes_filters(row):

    def val(key):
        try:
            return float(row[key]) if row.get(key) is not None else None
        except (TypeError, ValueError):
            return None
    (w, h, sim) = (val('WIDTH'), val('HEIGHT'), val('similarity'))
    if sim is not None and sim < MIN_SIMILARITY:
        return False
    if w is not None and w < MIN_SIDE or (h is not None and h < MIN_SIDE):
        return False
    if (val('punsafe') or 0) > MAX_PUNSAFE or (val('pwatermark') or 0) > MAX_PWATERMARK:
        return False
    aesthetic = val('aesthetic')
    return aesthetic is None or aesthetic >= MIN_AESTHETIC

def laion_subset(records, min_per_class=200, max_per_class=500, seed=42):
    rng = random.Random(seed)
    target = {c: rng.randint(min_per_class, max_per_class) for c in LAION_CLASSES}
    count = {c: 0 for c in LAION_CLASSES}
    (seen_url, seen_caption, out) = (set(), set(), [])
    for r in records:
        if all((count[c] >= target[c] for c in LAION_CLASSES)):
            break
        url = clean(r.get('url'))
        (caption, moondream) = (clean(r.get('TEXT')), clean(r.get('moondream')))
        if not url or url in seen_url or (not passes_filters(r)):
            continue
        cls = assign_class(caption, moondream)
        if cls is None or count[cls] >= target[cls]:
            continue
        key = hashlib.sha1((moondream or caption).lower().encode()).hexdigest()
        if key in seen_caption:
            continue
        count[cls] += 1
        seen_url.add(url)
        seen_caption.add(key)
        out.append({'class_name': cls, 'label': LAION_CLASSES.index(cls), 'TEXT': caption, 'moondream': moondream, 'url': url})
    return pd.DataFrame(out)
CHEXPERT_CLASSES = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture']

def chexpert_single_label(root):
    df = pd.concat([pd.read_csv(os.path.join(root, f)) for f in ('train.csv', 'valid.csv')], ignore_index=True)
    positive = df[CHEXPERT_CLASSES] == 1
    df = df[positive.sum(axis=1) == 1].copy()
    df['label'] = positive.loc[df.index].values.argmax(axis=1)
    return df
