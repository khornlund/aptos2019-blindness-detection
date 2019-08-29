import subprocess
import shutil
import json
from pathlib import Path


def kaggle_upload(run_dir, epochs):
    # create the directory to contain our new dataset
    run_dir = Path(run_dir)
    upload_dir = Path('./upload')
    target_dir = upload_dir / run_dir.parent.stem / run_dir.stem
    target_dir.mkdir(parents=True, exist_ok=True)

    # copy in the config file
    config = run_dir / 'checkpoints/config.yaml'
    shutil.copy(config, target_dir)

    # copy in the saved weights for the selected epochs
    for e in epochs:
        checkpoint = run_dir / f'checkpoints/checkpoint-epoch{e}.pth'
        shutil.copy(checkpoint, target_dir)

    # create metadata file
    meta = {
        'title': str(run_dir.stem),
        'subtitle': str(run_dir.parent.stem),
        'id': f'khornlund/{run_dir.stem}',
        'licenses': [{'name': 'CC0-1.0'}]
    }

    with open(target_dir.parent / 'dataset-metadata.json', 'w') as fh:
        json.dump(meta, fh, indent=4)

    # upload
    subprocess.run(
        ['kaggle', 'datasets', 'create', '-p', str(target_dir.parent), '-r', 'zip'],
        stdout=subprocess.PIPE
    )
