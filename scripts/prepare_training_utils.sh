export UTILS_DIR="${DATA_DIR}/training_utils"

python -m datagen_imagination.export_mp3d_text_feats \
    --save_dir="${UTILS_DIR}"

python -m datagen_imagination.get_category_freq \
    --data_dir=/scratch/hshah/ForeSightDataset/Structured3D/ \
    --output_dir="${UTILS_DIR}" \
    --text_feat_path="${UTILS_DIR}/mp3d_text_feats.npy"
