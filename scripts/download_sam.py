import argparse
from pathlib import Path
import urllib.request

CHECKPOINTS = {
    "sam2.1_hiera_small": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "checkpoints/sam2.1_hiera_small.pt",
        "configs/sam2.1/sam2.1_hiera_s.yaml",
    ),
    "sam2.1_hiera_base_plus": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "checkpoints/sam2.1_hiera_base_plus.pt",
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
    ),
    "sam2.1_hiera_large": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "checkpoints/sam2.1_hiera_large.pt",
        "configs/sam2.1/sam2.1_hiera_l.yaml",
    ),
}

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=CHECKPOINTS, default="sam2.1_hiera_small")
args = parser.parse_args()

url, dest, _ = CHECKPOINTS[args.variant]
Path(dest).parent.mkdir(parents=True, exist_ok=True)
if not Path(dest).exists():
    print("Downloading:", url)
    urllib.request.urlretrieve(url, dest)
else:
    print("Already exists:", dest)
print("Checkpoint:", dest)
