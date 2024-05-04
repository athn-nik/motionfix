
 

<p align="center">

  <h1 align="center">SINC: Spatial Composition of 3D Human Motions for Simultaneous Action Generation
    <br>
    <a href='https://arxiv.org/abs/2304.10417'>
    <img src='https://img.shields.io/badge/arxiv-report-red' alt='ArXiv PDF'>
    </a>
    <a href='https://sinc.is.tue.mpg.de/' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
  </h1>
  <p align="center">
    <a href="https://ps.is.mpg.de/person/nathanasiou"><strong>Nikos Athanasiou</strong></a>
    |
    <a href="https://ps.is.tuebingen.mpg.de/person/black"><strong>Michael J. Black</strong></a>
    |
    <a href="https://imagine.enpc.fr/~varolg"><strong>G&#252;l Varol</strong></a>
  </p>
  <h2 align="center">ICCV 2023</h2>
 <div align="center">Official PyTorch implementation of the paper "SINC: Spatial Composition of 3D Human Motions for Simultaneous Action Generation" </div>
 <div align="center">
  </div>
</p>
<!-- <p float="center">
  <div align="center">
  <img src="assets/sinc_tsr.gif" />
  </div> -->
</p>

<!-- | Paper Video                                                                                                | Qualitative Results                                                                                                |
|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| [![PaperVideo](https://img.youtube.com/vi/vidid/0.jpg)](https://www.youtube.com/) | -->

## Extracting Visuals

``` bash
python visual_pkls.py --path path/to/ --ds bodilex --mode s2t
```


``` bash
python render_motions_bodilex.py --path_to_json /path/a.json 
--start 0 --upto 4 --batch_size 2  
--outdir debug-colors2 
--mode sequence --ca red --cb green
```
* start: start point of the list
* upto: endpoint
* outdir: where to put the output file
* mode: sequence (image) /  video
* ca: color of source motion
* cb color of generated/target motion

<h2 align="center">Environment & Basic Setup</h2>

<details>
  <summary>Details</summary>
SINC has been implemented and tested on Ubuntu 20.04 with python >= 3.10.

Clone the repo:
```bash
git clone https://github.com/athn-nik/sinc.git
```

After it do this to install DistillBERT:

```shell
cd deps/
git lfs install
git clone https://huggingface.co/distilbert-base-uncased
cd ..
```

Install the requirements using `virtualenv` :
```bash
# pip
source scripts/install.sh
```
You can do something equivalent with `conda` as well.
</details>



[comment]: <> (## Running the Demo)

[comment]: <> (We have prepared a nice demo code to run SINC on arbitrary videos. )



<h2 align="center">Data & Training</h2>

 <details>
  <summary>Details</summary>

<div align="center"><em>There is no need to do this step if you have followed the instructions and have done it for TEACH. Just use the ones from TEACH.</em></div>

<div align="center"><h3>Step 1: Data Setup</h3></center></div>

Download the data from [AMASS website](https://amass.is.tue.mpg.de). Then, run this command to extract the amass sequences that are annotated in babel:

```shell
python scripts/process_amass.py --input-path /path/to/data --output-path path/of/choice/default_is_/babel/babel-smplh-30fps-male --use-betas --gender male
```

Download the data from [TEACH website](https://teach.is.tue.mpg.de), after signing in. The data SINC was trained was a processed version of BABEL. Hence, we provide them directly to your via our website, where you will also find more relevant details. 
Finally, download the male SMPLH male body model from the [SMPLX website](https://smpl-x.is.tue.mpg.de/). Specifically the AMASS version of the SMPLH model. Then, follow the instructions [here](https://github.com/vchoutas/smplx/blob/main/tools/README.md#smpl-h-version-used-in-amass) to extract the smplh model in pickle format.

The run this script and change your paths accordingly inside it extract the different babel splits from amass:

```shell
python scripts/amass_splits_babel.py
```

Then create a directory named `data` and put the babel data and the processed amass data in.
You should end up with a data folder with the structure like this:

```
data
|-- amass
|  `-- your-processed-amass-data 
|
|-- babel
|   `-- babel-teach
|       `...
|   `-- babel-smplh-30fps-male 
|       `...
|
|-- smpl_models
|   `-- smplh
|       `--SMPLH_MALE.pkl
```

Be careful not to push any data! 
Then you should softlink inside this repo. To softlink your data, do:

`ln -s /path/to/data`

You can do the same for your experiments:

`ln -s /path/to/logs experiments`

Then you can use this directory for your experiments.

<div align="center"><h3>Step 2 (a): Training</h3></center></div>

To start training after activating your environment. Do:

```shell
python train.py experiment=baseline logger=none
```

Explore `configs/train.yaml` to change some basic things like where you want
your output stored, which data you want to choose if you want to do a small
experiment on a subset of the data etc.
You can disable the text augmentations and using `single_text_desc: false` in the
model configuration file. You can check the `train.yaml` for the main configuration
and this file will point you to the rest of the configs (eg. `model` refers to a config found in
the folder `configs/model` etc.).

<div align="center"><h3>Step 2 (b): Training MLD</h3></center></div>

Prior to running this code for MLD please create and activate an environment according to their [repo](https://github.com/ChenFengYe/motion-latent-diffusion). Please do the `1. Conda Environment` and `2. Dependencies` out of the steps in their repo.

```shell
python train.py experiment=some_name run_id=mld-synth0.5-4gpu model=mld data.synthetic=true data.proportion_synthetic=0.5 data.dtype=seg+seq+spatial_pairs machine.batch_size=16 model.optim.lr=1e-4 logger=wandb sampler.max_len=150
```

</details>
<h2 align="center"> AMASS Compositions </h2>

<details>
  <summary>Details</summary>
  Given that you have downloaded and processed the data, you can create spatial compositions
  from gropundtruth motions of BABEL subset from AMASS using a standalone script:

  ```shell
  python compose_motions.py
  ```
</details>

<h2 align="center"> Evaluation</h2>

<details>
  <summary>Details</summary>

After training, to sample and evaluate a model which has been stored in a folder `/path/to/experiment`
``` bash
python sample.py folder=/path/to/experiment/ ckpt_name=699 set=small

python eval.py folder=/path/to/experiment/ ckpt_name=699 set=small
```

- You can change the `jointstype` for the sampling script to output and save rotations and translation by setting `joinstype=rots`.
- By setting the `set=full` you will obtain the results on the full BABEL validation set.

You can calculate the TEMOS score using: 

``` bash
python sample_eval_latent.py folder=/is/cluster/fast/nathanasiou/logs/space/single-text-baselines/rs_only/babel-amass/ ckpt_name=699 set=small
```

or for model trained using MLD:
```
python mld_temos.py folder=/is/cluster/fast/nathanasiou/logs/sinc/sinc-arxiv/mld-wo-synth/babel-amass ckpt_name=399 set=small
```
</details>

 
## Citation

```bibtex
@inproceedings{SINC:ICCV:2022,
  title={{SINC}: Spatial Composition of {3D} Human Motions for Simultaneous Action Generation},
  author={Athanasiou, Nikos and Petrovich, Mathis and Black, Michael J. and Varol, G\"{u}l },
  booktitle = {ICCV},
  year = {2023}
}

```
## License
This code is available for **non-commercial scientific research purposes** as defined in the [LICENSE file](LICENSE). By downloading and using this code you agree to the terms in the [LICENSE](LICENSE). Third-party datasets and software are subject to their respective licenses.

## References
Many part of this code were based on the official implementation of [TEMOS](https://github.com/Mathux/TEMOS).

## Contact

This code repository was implemented by [Nikos Athanasiou](https://is.mpg.de/~nathanasiou) and [Mathis Petrovich](https://mathis.petrovich.fr/).

Give a ⭐ if you like.

For commercial licensing (and all related questions for business applications), please contact ps-licensing@tue.mpg.de.