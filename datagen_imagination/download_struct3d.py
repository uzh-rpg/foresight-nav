import os
import argparse
from tqdm import tqdm
import requests

base_url = '<Structured3D-Download-URL>/Structured3D_panorama_{:02d}.zip'
available_shards = 18

def download_struct3d(data_dir: str, num_shards: int):
    zip_dir = os.path.join(data_dir, 'zips')
    if os.path.exists(zip_dir):
        raise ValueError(f'{zip_dir} already exists. Please remove it and try again.')
    os.makedirs(zip_dir)

    for i in tqdm(range(num_shards)):
        zip_file = os.path.join(zip_dir, f'Structured3D_panorama_{i:02d}.zip')
        r = requests.get(url=f'{base_url.format(i)}', stream=True)
        with open(zip_file, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=512):
                if chunk:
                    fd.write(chunk)
        os.system(f'unzip {zip_file} -d {data_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Struct3D dataset')
    parser.add_argument('--data_dir', default='data', required=True, help='Directory to save the dataset')
    parser.add_argument('--num_shards', default=available_shards, type=int, help='Number of dataset shards to download')
    args = parser.parse_args()
    download_struct3d(args.data_dir, args.num_shards)