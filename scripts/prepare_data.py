import json, zipfile, shutil, random
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "archive.zip"
FILTERED_JSON = ROOT / "data" / "dataset_filtered.json"
IMG_DIR = ROOT / "data" / "images"
OUT = ROOT / "data" / "yolo"
OUT_IMAGES = OUT / "images"
OUT_LABELS = OUT / "labels"

def coco_to_yolo(bbox, W=1920, H=1080):
    x,y,w,h = bbox
    return (x+w/2)/W, (y+h/2)/H, w/W, h/H

def main():
    data = json.loads(FILTERED_JSON.read_text())
    id2name = {c['id']:c['name'] for c in data['categories']}
    names = [id2name[i] for i in range(len(id2name))]
    print(f"Classes: {names}  Images: {len(data['images'])}  Anns: {len(data['annotations'])}")

    z = zipfile.ZipFile(ZIP)
    fname_to_folder = {}
    for n in z.namelist():
        if n.endswith('.jpg'):
            parts = n.split('/')
            folder = '/'.join(parts[1:3]) if len(parts)>=3 else 'unk'
            fname_to_folder[parts[-1]] = folder

    for im in data['images']:
        if im['file_name'] not in fname_to_folder:
            raise ValueError(f"missing folder for {im['file_name']}")

    anns_by_img = defaultdict(list)
    for a in data['annotations']:
        anns_by_img[a['image_id']].append(a)

    dominant = {}
    for im in data['images']:
        anns = anns_by_img[im['id']]
        if not anns:
            dominant[im['id']] = -1
        else:
            c = Counter(a['category_id'] for a in anns)
            dominant[im['id']] = c.most_common(1)[0][0]

    groups = [fname_to_folder[im['file_name']] for im in data['images']]
    uniq_groups = sorted(set(groups))
    print(f"Groups: {Counter(groups)}")

    from sklearn.model_selection import StratifiedShuffleSplit
    y = [dominant[im['id']] for im in data['images']]
    total = len(data['images'])
    # Two-stage stratified shuffle: first 70% train, 30% temp, then split temp into 15/15 val/test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(sss1.split(data['images'], y))
    y_temp = [y[i] for i in temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_rel, test_rel = next(sss2.split([data['images'][i] for i in temp_idx], y_temp))
    val_idx = set(temp_idx[val_rel]); test_idx = set(temp_idx[test_rel]); train_idx = list(train_idx)
    print(f"Stratified sizes: train {len(train_idx)} val {len(val_idx)} test {len(test_idx)}")
    for s_name, s_idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        cnt = Counter(y[i] for i in s_idx)
        print(f"  {s_name} cnts {dict(cnt)}")
    # Group distribution per split for audit
    for s_name, s_idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        gc = Counter(groups[i] for i in s_idx)
        print(f"  {s_name} groups {dict(gc)}")
    note = "NOTE: Group leakage accepted — group-strict split leaves net_plastic absent from train (clustered in Bistrina). Stratified image-level ensures all 6 classes present per split for trainability; mitigated by augmentation. Justification logged for memo."
    print(note)

    def idx_to_split(idx):
        if idx in test_idx: return 'test'
        if idx in val_idx: return 'val'
        return 'train'

    # Clean out
    if OUT.exists():
        shutil.rmtree(OUT)
    for s in ['train','val','test']:
        (OUT_IMAGES/s).mkdir(parents=True, exist_ok=True)
        (OUT_LABELS/s).mkdir(parents=True, exist_ok=True)

    # Need image id -> group for logging
    split_groups = Counter()
    for idx, im in enumerate(data['images']):
        split_groups[(idx_to_split(idx), fname_to_folder[im['file_name']])] += 1

    # Write YOLO
    id_to_split_indices = defaultdict(list)
    for idx, im in enumerate(data['images']):
        s = idx_to_split(idx)
        id_to_split_indices[s].append(idx)

    for s in ['train','val','test']:
        for idx in id_to_split_indices[s]:
            im = data['images'][idx]
            src = IMG_DIR / im['file_name']
            dst_img = OUT_IMAGES / s / im['file_name']
            dst_lbl = OUT_LABELS / s / (Path(im['file_name']).stem + '.txt')
            # copy image (or symlink on mac, copy on win)
            shutil.copy2(src, dst_img)
            anns = anns_by_img[im['id']]
            lines = []
            for a in anns:
                cx, cy, w, h = coco_to_yolo(a['bbox'])
                # clip
                cx = min(max(cx,0),1); cy = min(max(cy,0),1); w = min(max(w,0),1); h = min(max(h,0),1)
                if w<=0 or h<=0:
                    continue
                lines.append(f"{a['category_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            dst_lbl.write_text("\n".join(lines))

    # data.yaml
    yaml = f"""path: {OUT.as_posix()}
train: images/train
val: images/val
test: images/test
nc: {len(names)}
names: {names}
"""
    (OUT / "data.yaml").write_text(yaml)
    # also root data.yaml for convenience
    (ROOT / "data" / "data.yaml").write_text(yaml)

    # splits.json for audit
    splits = {}
    for s in ['train','val','test']:
        splits[s] = [data['images'][i]['file_name'] for i in id_to_split_indices[s]]
    (OUT / "splits.json").write_text(json.dumps({k: len(v) for k,v in splits.items()}, indent=2))
    # detailed
    (OUT / "splits_files.json").write_text(json.dumps(splits, indent=2))

    print(yaml)
    for s in ['train','val','test']:
        print(f"{s}: {len(id_to_split_indices[s])} images  labels {len(list((OUT_LABELS/s).glob('*.txt')))}")
        # per class counts in split
        cnt = Counter()
        for idx in id_to_split_indices[s]:
            for a in anns_by_img[data['images'][idx]['id']]:
                cnt[id2name[a['category_id']]] += 1
        print(f"  {dict(cnt)}")

    print("Groups per split:")
    for (s,g),c in sorted(split_groups.items()):
        print(f"  {s:5s} {g:30s} {c}")

if __name__ == "__main__":
    main()
