from pprint import pprint

import torch
from huggingface_hub import HfApi

import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

repo_id = "/media/forceyqj/3ac51dbb-cd9f-41ab-a94c-a15affbbbe36/flexiv_teleop_dataset/lerobot/flexiv_peel_cucumber_inputForce"
# We can have a look and fetch its metadata to know more about it:
ds_meta = LeRobotDatasetMetadata(repo_id)

# By instantiating just this class, you can quickly access useful information about the content and the
# structure of the dataset without downloading the actual data yet (only metadata files — which are
# lightweight).
print(f"Total number of episodes: {ds_meta.total_episodes}")
print(f"Average number of frames per episode: {ds_meta.total_frames / ds_meta.total_episodes:.3f}")
print(f"Frames per second used during data collection: {ds_meta.fps}")
print(f"Robot type: {ds_meta.robot_type}")
print(f"keys to access images from cameras: {ds_meta.camera_keys=}\n")

dataset = LeRobotDataset(repo_id)

dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=0,
    batch_size=32,
    shuffle=True,
)
camera_keys = dataset.meta.camera_keys

for batch in dataloader:
    for camera_key in camera_keys:
        print(f"{batch[camera_key].shape=}")  # (32, 4, c, h, w)
    print(f"{batch['observation.state'].shape=}")  # (32, 6, c)
    print(f"{batch['action'].shape=}")  # (32, 64, c)
    break
