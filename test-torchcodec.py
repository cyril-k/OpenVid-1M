from typing import Optional
import torch
import requests


# Video source: https://www.pexels.com/video/dog-eating-854132/
# License: CC0. Author: Coverr.
url = "https://videos.pexels.com/video-files/854132/854132-sd_640_360_25fps.mp4"
response = requests.get(url, headers={"User-Agent": ""})
if response.status_code != 200:
    raise RuntimeError(f"Failed to download video. {response.status_code = }.")

raw_video_bytes = response.content

from torchcodec.decoders import VideoDecoder

# You can also pass a path to a local file!
decoder = VideoDecoder(raw_video_bytes, device="cuda")


print(decoder.metadata)