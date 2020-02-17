import json
from pathlib import Path
from shutil import copyfile

p = Path('out/obstacle_noise3/robust-epc')
for stat in p.glob('**/openaigym.episode_batch.*.stats*'):
    with open(stat) as f:
        results = json.load(f)
    try:
        seed = results['episode_seeds'][0]
        length = results['episode_lengths'][0]
        for video in stat.parent.glob("*.mp4"):
            print("Copying", video)
            copyfile(video, video.parent.parent / "{}_{}.mp4".format(seed, p.name))
    except IndexError:
        continue
