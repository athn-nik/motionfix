
 

<p align="center">

  <h1 align="center">MotionFix: Text-Driven 3D Human Motion Editing
    <br>
    <a href='https://www.arxiv.org/abs/2408.00712'>
    <img src='https://img.shields.io/badge/arxiv-report-red' alt='ArXiv PDF'>
    </a>
    <a href='https://motionfix.is.tue.mpg.de/' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
  </h1>
  <p align="center">
    <a href="https://ps.is.mpg.de/person/nathanasiou"><strong>Nikos Athanasiou</strong></a>
    |
    <a href="https://ps.is.mpg.de/person/acseke"><strong>Alpár Cseke</strong></a>
    |
    <a href="https://ps.is.mpg.de/person/mdiomataris"><strong>Markos Diomataris</strong></a>
    |
    <a href="https://ps.is.tuebingen.mpg.de/person/black"><strong>Michael J. Black</strong></a>
    |
    <a href="https://imagine.enpc.fr/~varolg"><strong>G&#252;l Varol</strong></a>
  </p>
  <h2 align="center">Siggraph Asia 2024, Tokyo, Japan</h2>
 <div align="center">
  <img src="assets/raise_arm_higher.gif" width="55%" />
  </div>
 <div align="center">Official PyTorch implementation of the paper "MotionFix: Text-Driven 3D Human Motion Editing". This repository includes instructions for the released dataset and implementation of TMED, the model used in the paper along with the main experiments.</div>
</p>

<!-- | Paper Video                                                                                                | Qualitative Results                                                                                                |
|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| [![PaperVideo](https://img.youtube.com/vi/vidid/0.jpg)](https://www.youtube.com/) | -->

## Evaluation of Model

### Step 1: Extract the samples

```bash
python parallel_motionfix_eval.py --mode eval --runs experiments/clean-motionfix/bodilex_hml3d/50-50_bs128_300ts_clip77_with_zeros_source/ --ds bodilex --inpaint
```

- inpaint: if you should use inpaint or not
- ds: bodilex / sinc_synth
- runs: path to the foler `exp_name`
- mode: eval (don't ask why).

### Step 2: Compute the metrics

```bash
python parallel_evaluation.py --ds bodilex --extras 'samples_path=experiments/kinedit/bodilex/lr1-4_300ts_bs128_wo_sched/steps_300_bodilex_noise_last' --set val
```

_Quotes_ : `'` quotes are needed!

- set: val/test/all --> defaults is `test`
- ds: bodilex/sinc_synth
- extras: the path to the experiment you are evaluating for.

## Extracting Visuals

``` bash
python visual_pkls.py --path path/to/ --ds bodilex --mode s2t
```
* start: path to samples should look like
    - `exp_name/steps_XX_DS_noise_last/ld_txt-YY_ld_mot-WW`
* ds: bodilex / sinc_synth
* mode: source2target or target2target (based on T2T score what to extract for renders)

``` bash
python render_motions_bodilex.py --path_to_json /path/a.json 
--start 0 --upto 4 --batch_size 2  
--outdir debug-colors2 
--mode sequence --ca red --cb green
```
* path_to_json: path returned from previous script
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

## Citation

```bibtex
@inproceedings{athanasiou2024motionfix,
  title = {{MotionFix}: Text-Driven 3D Human Motion Editing},
  author = {Athanasiou, Nikos and Ceske, Alpar and Diomataris, Markos and Black, Michael J. and Varol, G{\"u}l},
  booktitle = {SIGGRAPH Asia 2024 Conference Papers},
  year = {2024}
}
```

## License
This code is available for **non-commercial scientific research purposes** as defined in the [LICENSE file](LICENSE). By downloading and using this code you agree to the terms in the [LICENSE](LICENSE). Third-party datasets and software are subject to their respective licenses.


## Contact

This code repository was implemented by [Nikos Athanasiou](https://is.mpg.de/~nathanasiou).

Give a ⭐ if you like it.

For commercial licensing (and all related questions for business applications), please contact ps-licensing@tue.mpg.de.
