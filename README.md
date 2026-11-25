[![Abcdspec-compliant](https://img.shields.io/badge/ABCD_Spec-v1.1-green.svg)](https://github.com/brain-life/abcd-spec)
[![Run on Brainlife.io](https://img.shields.io/badge/Brainlife-bl.app.113-blue.svg)](https://doi.org/10.25663/brainlife.app.113)

# Dipy Recobundles

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Extracts white matter bundles from a whole-brain .trk tractogram with DIPY RecoBundles using the HCP842 bundle atlas, and outputs a WMC classification plus recognised bundles and labels.**

Registers the tractogram to the HCP842 atlas with streamline-based linear registration (SLR) and recognises each atlas model bundle with RecoBundles, optionally refining the result. Use it for automatic bundle segmentation of a whole-brain tractogram in subject space. The atlas is downloaded at run time by DIPY.

| | |
|---|---|
| 📥 **Inputs** | `neuro/track/trk` |
| 📤 **Outputs** | `neuro/wmc` · `raw` |
| 🐳 **Container** | `brainlife/dipy:1.4.0` |
| ⚙️ **Resources** | 8 CPUs |
| 🔗 **Run it** | [doi.org/10.25663/brainlife.app.113](https://doi.org/10.25663/brainlife.app.113) |

#### 📥 Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `track` | `neuro/track/trk` | yes | Whole-brain tractogram (.trk). |

#### 📤 Outputs

| Output | Datatype | Description |
|---|---|---|
| `wmc` | `neuro/wmc` | White matter classification of the input streamlines into the recognised bundles, with per-tract JSON for visualisation. |
| `output` | `raw` | Recognised bundle streamlines and their labels. |

<details>
<summary><b>⚙️ Parameters</b> (16)</summary>

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

📚 **References used by this App:**
- Garyfallidis E, et al. Dipy, a library for the analysis of diffusion MRI data. Front Neuroinform 8:8 (2014). [doi:10.3389/fninf.2014.00008](https://doi.org/10.3389/fninf.2014.00008)
- Avesani P, et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. Sci Data 6:69 (2019). [doi:10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

<!-- brainlife-app:end -->

The App uses the Recobundles method to do an automatic extraction of brain pathways.

### Authors
- [Serge Koudoro](skoudoro@iu.edu)
- [Javier Ricardo Guaje Guerra](jrguajeg@iu.edu)

### Contributors
- [Soichi Hayashi](soichih@gmail.com)

### Funding Acknowledgement
brainlife.io is publicly funded and for the sustainability of the project it is helpful to Acknowledge the use of the platform. We kindly ask that you acknowledge the funding below in your code and publications. Copy and past the following lines into your repository when using this code.

[![NSF-BCS-1734853](https://img.shields.io/badge/NSF_BCS-1734853-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1734853)
[![NSF-BCS-1636893](https://img.shields.io/badge/NSF_BCS-1636893-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1636893)
[![NSF-ACI-1916518](https://img.shields.io/badge/NSF_ACI-1916518-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1916518)
[![NSF-IIS-1912270](https://img.shields.io/badge/NSF_IIS-1912270-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1912270)
[![NIH-NIBIB-R01EB029272](https://img.shields.io/badge/NIH_NIBIB-R01EB029272-green.svg)](https://grantome.com/grant/NIH/R01-EB029272-01)

### Citations
We ask that you the following articles when publishing papers that used data, code or other resources created by the brainlife.io community.

1. Garyfallidis, E., Brett, M., Amirbekian, B., Rokem, A., van der Walt, S., Descoteaux, M., Nimmo-Smith, I., & Dipy Contributors (2014). Dipy, a library for the analysis of diffusion MRI data. Frontiers in neuroinformatics, 8, 8. [https://doi.org/10.3389/fninf.2014.00008](https://doi.org/10.3389/fninf.2014.00008)

2. Avesani, P., McPherson, B., Hayashi, S. et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. Sci Data 6, 69 (2019). [https://doi.org/10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

## Running the App

#### 1. On Brainlife.io

You can see a list of [Dipy Apps currently regsitered on Brainlife](https://brainlife.io/apps#dipy). Find the App that you'd like to run and click "Execute" tab to specify dataset that you'd like to run the App on.

#### 2. On  your machine (Running Locally)

To run this command, you can simply type:

`singularity exec -e docker://brainlife/dipy:1.1.1 dipy_recobundles [your_args]`

To see the documentation of all arguments, [go to the following page](https://dipy.org/documentation/1.1.1./reference_cmd/dipy_recobundles/)

## Input

To see the documentation of all the arguments, follow this [link](https://dipy.org/documentation/1.1.1./reference_cmd/dipy_recobundles/).

## Output

All output files will be generated according to the passed arguments, as explained [here](https://dipy.org/documentation/1.1.1./reference_cmd/dipy_recobundles/).

### Dependencies

This app runs on [singularity](https://www.sylabs.io/singularity/).

### DIPY
- This is a Brainlife wrapper App stemming from the [`dipy_recobundles`](https://dipy.org/documentation/latest/reference_cmd/dipy_recobundles/) workflow.
- This single wrapper is exposed through an apps registered on [Brainlife.io](https://brainlife.io).
- More information about DIPY : [https://dipy.org/](https://dipy.org/).
- More information about the command line `dipy_recobundles`: [Command line Reference](https://dipy.org/documentation/latest/reference_cmd/dipy_recobundles/).
