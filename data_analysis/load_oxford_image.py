import os, shutil


DATASET_DIR = "oxford_pet"
# oxford_pet というフォルダがあったら削除する？
if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)
    
categories = []
for fname in os.listdir("tmp/images"):
    if not fname.endswith(".jpg"):
        continue
    elements = os.path.basename(fname).split('_')
    category = '_'.join(elements[:-1])

    if category not in categories:
        os.makedirs(os.path.join(DATASET_DIR, "Train", category), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, "Validation", category), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, "Test", category), exist_ok=True)

        categories.append(category)

# Validationセット内に存在する誤った画像を除外
NG_LIST =  ["shiba_inu_156.jpg", "shiba_inu_157.jpg", "saint_bernard_147.jpg", "saint_bernard_148.jpg"]

for category in categories:
    fnames = []
    for fname in os.listdir("tmp/images"):
        if not fname.endswith(".jpg") or not fname.startswith(category):
            continue
        fnames.append(fname)
    fnames = sorted(fnames)

    for fname in fnames[:50]:
        shutil.copy(f"tmp/images/{fname}", os.path.join(DATASET_DIR, "Train", category, fname))
    for fname in fnames[50:100]:
        if fname in NG_LIST:
            continue
        shutil.copy(f"tmp/images/{fname}", os.path.join(DATASET_DIR, "Validation", category, fname))
    for fname in fnames[100:]:
        shutil.copy(f"tmp/images/{fname}", os.path.join(DATASET_DIR, "Test", category, fname))

