pip install gdown
mkdir data-test
mkdir data-test/body_models
# dataset
gdown --folder "https://drive.google.com/drive/folders/1DM7oIJwxwoljVxAfhfktocTptwVX5sqR?usp=sharing"
mv motionfix-dataset data/ 
# tmr folder
gdown --folder "https://drive.google.com/drive/folders/15LHeriOCjmh4Cp5H9M94xoFGBN0amxI8?usp=sharing"
mv tmr-evaluator eval-deps
# smpl models
gdown --folder "https://drive.google.com/drive/folders/1s3re2I1OzBimQIpudUEFB1hClFWOPJjC?usp=drive_link"
mv smplh data-test/body_models

