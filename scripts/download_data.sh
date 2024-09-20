pip install gdown
mkdir data
mkdir data/body_models
# dataset
gdown --folder "https://drive.google.com/drive/folders/1DM7oIJwxwoljVxAfhfktocTptwVX5sqR?usp=sharing"
mv motionfix-dataset data/ 
# tmr folder
gdown --folder "https://drive.google.com/drive/folders/15LHeriOCjmh4Cp5H9M94xoFGBN0amxI8?usp=sharing"
mv tmr-evaluator eval-deps
# smpl models
gdown --folder "https://drive.google.com/drive/folders/1s3re2I1OzBimQIpudUEFB1hClFWOPJjC?usp=drive_link"
mv smplh data/body_models
# tmed checkpoints
gdown --folder "https://drive.google.com/drive/folders/1M_i_zUSlktdEKf-xBF9g6y7N-lfDtuPD?usp=sharing"
mkdir experiments
mv tmed experiments/
mkdir experiments/tmed/checkpoints
mv experiments/tmed/last.ckpt experiments/tmed/checkpoints/
