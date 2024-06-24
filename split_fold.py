import os
import shutil
from sklearn.model_selection import KFold

# 데이터셋 디렉토리 설정
lwir_image_dir = 'datasets/kaist-rgbt/train/images/lwir'
visible_image_dir = 'datasets/kaist-rgbt/train/images/visible'
label_dir = 'datasets/kaist-rgbt/train/labels'
folds_dir = 'datasets/kaist-rgbt/folds'

# lwir와 visible 이미지 파일 리스트 생성
lwir_image_files = [os.path.join(lwir_image_dir, f) for f in os.listdir(lwir_image_dir) if f.endswith('.jpg')]
visible_image_files = [os.path.join(visible_image_dir, f) for f in os.listdir(visible_image_dir) if f.endswith('.jpg')]

# KFold 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(kf.split(lwir_image_files)):
    fold_dir = os.path.join(folds_dir, f'fold{fold+1}')
    os.makedirs(os.path.join(fold_dir, 'train/images/lwir'), exist_ok=True)
    os.makedirs(os.path.join(fold_dir, 'train/images/visible'), exist_ok=True)
    os.makedirs(os.path.join(fold_dir, 'train/labels'), exist_ok=True)
    os.makedirs(os.path.join(fold_dir, 'val/images/lwir'), exist_ok=True)
    os.makedirs(os.path.join(fold_dir, 'val/images/visible'), exist_ok=True)
    os.makedirs(os.path.join(fold_dir, 'val/labels'), exist_ok=True)

    # train 데이터 분할
    for idx in train_index:
        lwir_image = lwir_image_files[idx]
        visible_image = visible_image_files[idx]
        label_file = os.path.join(label_dir, os.path.basename(lwir_image).replace('.jpg', '.txt'))

        shutil.copy(lwir_image, os.path.join(fold_dir, 'train/images/lwir'))
        shutil.copy(visible_image, os.path.join(fold_dir, 'train/images/visible'))
        if os.path.exists(label_file):
            shutil.copy(label_file, os.path.join(fold_dir, 'train/labels'))

    # validation 데이터 분할
    for idx in val_index:
        lwir_image = lwir_image_files[idx]
        visible_image = visible_image_files[idx]
        label_file = os.path.join(label_dir, os.path.basename(lwir_image).replace('.jpg', '.txt'))

        shutil.copy(lwir_image, os.path.join(fold_dir, 'val/images/lwir'))
        shutil.copy(visible_image, os.path.join(fold_dir, 'val/images/visible'))
        if os.path.exists(label_file):
            shutil.copy(label_file, os.path.join(fold_dir, 'val/labels'))
