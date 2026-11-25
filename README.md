<h1 align="center">RecoBundles</h1>

<p align="center">
  Extracts white matter bundles from a whole-brain .trk tractogram with DIPY RecoBundles using the HCP842 bundle atlas, and outputs a WMC classification plus recognised bundles and labels.
</p>

<p align="center">
  <a href="https://github.com/brainlife/abcd-spec"><img src="https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg" alt="ABCD Spec v1.1"></a>
  <a href="https://doi.org/10.25663/brainlife.app.113"><img src="https://img.shields.io/badge/Brainlife-brainlife.app.113-blue.svg" alt="Run on brainlife.io"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-see_LICENSE-lightgrey.svg" alt="License"></a>
</p>

## At a glance

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Extracts white matter bundles from a whole-brain .trk tractogram with DIPY RecoBundles using the HCP842 bundle atlas, and outputs a WMC classification plus recognised bundles and labels.**

Registers the tractogram to the HCP842 atlas with streamline-based linear registration (SLR) and recognises each atlas model bundle with RecoBundles, optionally refining the result. Use it for automatic bundle segmentation of a whole-brain tractogram in subject space. The atlas is downloaded at run time by DIPY.

| | |
|---|---|
| **Inputs** | `neuro/track/trk` |
| **Outputs** | `neuro/wmc` · `raw` |
| **Container** | `brainlife/dipy:1.4.0` |
| **Resources** | 8 CPUs |
| **Run it** | [doi.org/10.25663/brainlife.app.113](https://doi.org/10.25663/brainlife.app.113) |

#### Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `track` | `neuro/track/trk` | yes | Whole-brain tractogram (.trk). |

#### Outputs

| Output | Datatype | Description |
|---|---|---|
| `wmc` | `neuro/wmc` | White matter classification of the input streamlines into the recognised bundles, with per-tract JSON for visualisation. |
| `output` | `raw` | Recognised bundle streamlines and their labels. |

<details>
<summary><b>Parameters</b> (16)</summary>

| Parameter | Type | Default | Description |
|---|---|---|---|
| `greater_than` | number | `50` | Keep streamlines that have a length greater than this value. |
| `less_than` | number | `1000000` | Keep streamlines have a length lesser than this value. |
| `no_slr` | boolean | `false` | Disable streamline-based linear registration of the tractogram to the atlas. |
| `clust_thr` | number | `15` | MDF distance threshold for all streamlines. |
| `reduction_thr` | number | `15` | Reduce search space by (mm). |
| `reduction_distance` | enum | `mdf` | Reduction distance type. Options: `mam`, `mdf`. |
| `model_clust_thr` | number | `2.5` | MDF distance threshold for the model bundles. |
| `pruning_thr` | number | `8` | Pruning after matching. |
| `pruning_distance` | enum | `mdf` | Pruning distance type. Options: `mam`, `mdf`. |
| `slr_metric` | enum | `symmetric` | Metric used for the SLR registration. Options: `asymmetric`, `diagonal`, `symmetric`. |
| `slr_transform` | enum | `similarity` | Transform type for SLR (translation, rigid, similarity or scaling). Options: `rigid`, `scaling`, `similarity`, `translation`. |
| `slr_matrix` | enum | `small` | Size of the streamline subset used for SLR (nano 100 to huge 1200 streamlines). Options: `nano`, `tiny`, `small`, `medium`, `large`, `huge`. |
| `refine` | boolean | `false` | Enable refine of recognized bundle. |
| `r_reduction_thr` | number | `12` | Refine reduce search space by (mm). |
| `r_pruning_thr` | number | `6` | Refine pruning after matching. |
| `no_r_slr` | boolean | `false` | Don't enable Refine local Streamline-based Linear Registration. |

</details>

<!-- brainlife-app:end -->

## Running the App

### On brainlife.io

Open the App at [doi.org/10.25663/brainlife.app.113](https://doi.org/10.25663/brainlife.app.113), choose `neuro/track/trk` from your project, set the parameters and press **Execute**. The results are saved to your project as `neuro/wmc`, `raw`.

### Locally

You need [Singularity/Apptainer](https://apptainer.org/) (the App runs in `brainlife/dipy:1.4.0`), [`jq`](https://jqlang.github.io/jq/).

1. Clone this repository.
2. Create `config.json` with the paths to your inputs and the parameters:

   ```json
   {
       "streamline_files": "/path/to/track/streamline_files",
       "greater_than": 50,
       "less_than": 1000000,
       "no_slr": false,
       "clust_thr": 15,
       "reduction_thr": 15,
       "reduction_distance": "mdf",
       "model_clust_thr": 2.5,
       "pruning_thr": 8,
       "pruning_distance": "mdf",
       "slr_metric": "symmetric",
       "slr_transform": "similarity",
       "slr_matrix": "small",
       "refine": false,
       "r_reduction_thr": 12,
       "r_pruning_thr": 6,
       "no_r_slr": false
   }
   ```

3. Run `./main`.

## DIPY

- This is a Brainlife wrapper App stemming from the [`dipy_recobundles`](https://dipy.org/documentation/latest/reference_cmd/dipy_recobundles/) workflow.
- This single wrapper is exposed through an apps registered on [Brainlife.io](https://brainlife.io).
- More information about DIPY : [https://dipy.org/](https://dipy.org/).
- More information about the command line `dipy_recobundles`: [Command line Reference](https://dipy.org/documentation/latest/reference_cmd/dipy_recobundles/).

## Citing

If you use this App, please cite it, the software it builds on and brainlife.io:

- RecoBundles. brainlife.io App. [doi:10.25663/brainlife.app.113](https://doi.org/10.25663/brainlife.app.113)
- Garyfallidis E, et al. Dipy, a library for the analysis of diffusion MRI data. Front Neuroinform 8:8 (2014). [doi:10.3389/fninf.2014.00008](https://doi.org/10.3389/fninf.2014.00008)
- Hayashi S, Caron BA, Heinsfeld AS, et al. brainlife.io: a decentralized and open-source cloud platform to support neuroscience research. *Nature Methods* 21:809–813 (2024). [doi:10.1038/s41592-024-02237-2](https://doi.org/10.1038/s41592-024-02237-2)
- Avesani P, McPherson B, Hayashi S, et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. *Scientific Data* 6:69 (2019). [doi:10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

## Authors

- Serge Koudoro
- Javier Ricardo Guaje Guerra

**Contributors**

- Soichi Hayashi

## Funding

[![Wellcome-226486/Z/22/Z](https://img.shields.io/badge/Wellcome-226486%2FZ%2F22%2FZ-purple.svg)](https://wellcome.org/)
[![NIH-NINDS-U24NS140384](https://img.shields.io/badge/NIH_NINDS-U24NS140384-green.svg)](https://reporter.nih.gov/project-details/11261037)
[![NIH-NINDS-UM1NS132207](https://img.shields.io/badge/NIH_NINDS-UM1NS132207-green.svg)](https://reporter.nih.gov/project-details/11411328)
[![NIH-NIBIB-R01EB029272](https://img.shields.io/badge/NIH_NIBIB-R01EB029272-green.svg)](https://reporter.nih.gov/project-details/10262925)
[![NIH-NIBIB-R01EB030896](https://img.shields.io/badge/NIH_NIBIB-R01EB030896-green.svg)](https://reporter.nih.gov/project-details/10428625)
[![NIH-NIMH-R01MH126699](https://img.shields.io/badge/NIH_NIMH-R01MH126699-green.svg)](https://reporter.nih.gov/project-details/10460628)
[![NIH-NIDCD-R01DC020088](https://img.shields.io/badge/NIH_NIDCD-R01DC020088-green.svg)](https://reporter.nih.gov/project-details/11140284)
[![NSF-BCS-1734853](https://img.shields.io/badge/NSF_BCS-1734853-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1734853)
[![NSF-BCS-2203524](https://img.shields.io/badge/NSF_BCS-2203524-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2203524)
[![NSF-IIS-1636893](https://img.shields.io/badge/NSF_IIS-1636893-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1636893)
[![NSF-OAC-2148729](https://img.shields.io/badge/NSF_OAC-2148729-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2148729)
[![NSF-IIS-1912270](https://img.shields.io/badge/NSF_IIS-1912270-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1912270)
[![NSF-IIS-2148700](https://img.shields.io/badge/NSF_IIS-2148700-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2148700)
[![NSF-OAC-1916518](https://img.shields.io/badge/NSF_OAC-1916518-blue.svg)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1916518)

<sub>brainlife.io, which runs this App, is publicly funded by Wellcome, the U.S. National Institutes of Health and the U.S. National Science Foundation; acknowledging that support helps keep it running.</sub>

## License

See [LICENSE](LICENSE).
