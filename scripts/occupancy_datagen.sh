export SCENE_DIR="${DATA_DIR}/Structured3D"

# Generate pointclouds for each scene
python -m datagen_imagination.occupancy_generation.generate_point_cloud_stru3d \
    --data_root=$SCENE_DIR

# Generate occupancy maps for each scene
python -m datagen_imagination.occupancy_generation.generate_occupancy \
    --data_dir=$SCENE_DIR

# Generate occupancy masks for imagination training by simulation of 
# an agent in the scene
python -m datagen_imagination.occupancy_generation.simulate_planner \
    --data_dir=$SCENE_DIR

# Generate valid .txt files containing training and validation scenes
python -m datagen_imagination.occupancy_generation.gen_scene_split \
    --data_root=$SCENE_DIR
