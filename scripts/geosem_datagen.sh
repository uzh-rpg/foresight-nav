python -m datagen_imagination.geosem_map_generation.gen_geosem_map \
    --config='configs/geosem_map_gen.yaml'

python -m datagen_imagination.geosem_map_generation.combine_gen_lists \
    --shard_dir=$SCENE_DIR \
    --num_shards=0 \
    --output_dir=$SCENE_DIR

python -m datagen_imagination.prepare_geosem_maps \
    --data_dir=$SCENE_DIR \
    --output_dir="${DATA_DIR}/imagination_training_data"