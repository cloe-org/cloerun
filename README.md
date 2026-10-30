<p align="center">
  <img src="docs/cloerun_banner.png" alt="cloerun banner" width="600">
</p>

# cloerun

Run Euclid inference campaigns with
[`cloelike`](https://github.com/cloe-org/cloelike) likelihoods: generate
synthetic data vectors, sample posteriors and analyse the chains, from YAML
campaign files such as those of
[`dr1-inference-configs`](https://github.com/cloe-org/dr1-inference-configs).

The interface is four modules, one per step, each also a `cloerun` command:

| Module | Command | Does |
|---|---|---|
| `cloerun.campaign` | `cloerun campaign FILE` | Resolves the runs of a campaign and runs each one in its `run.mode` |
| `cloerun.sdv` | `cloerun sdv FILE` | Writes the theory at the fiducial point as a synthetic data vector |
| `cloerun.chain` | `cloerun chain FILE` | Samples the posterior (`--test`: one loglike at the fiducial point) |
| `cloerun.analyse` | `cloerun analyse FILE` | Writes a summary table and a triangle plot of finished chains |

Everything else (config parsing, probes, priors, samplers) lives in the
private `cloerun._core`.

Currently supported:

- **Photometric probes** (angular power spectra,
  `cloelike.EuclidLikelihood_photo_Cls`): `WL` (cosmic shear), `GCph`, `GGL`,
  `2x2pt` (GGL + GCph) and `3x2pt`.
- **Spectroscopic probes**: `GCspectro_Pls` (power spectrum multipoles),
  `GCspectro_BAO` (BAO alphas) and `GCspectro_Pls_BAO` (both).
- **Sampler**: [`nautilus`](https://nautilus-sampler.readthedocs.io/).

## Installation

`cloerun` works with [`cloelib`](https://github.com/cloe-org/cloelib) v0.11
and [`cloelike`](https://github.com/cloe-org/cloelike) v0.3, together with
[`cosmolib`](https://github.com/astro-ph/cosmolib),
[`euclidlib`](https://github.com/euclidlib/euclidlib) and the external
libraries of the theory model you pick (`CAMB`, `HMcode2020Emu`, `FAST-PT`
for TATT and the nonlinear galaxy bias, `COMET` or `PBJ` for the spectroscopic
multipoles, ...). The steps below give this layout:

```
cloe-org/
├── cloelib/
├── cloelike/
├── cloerun/          this repository
└── dr1-inference/    the campaign configs
```

1. Create the `cloe-org` environment of
   [`cloe-org-environments`](https://github.com/cloe-org/cloe-org-environments),
   which brings the external libraries:

   ```bash
   git clone https://github.com/cloe-org/cloe-org-environments.git
   mamba env create -f cloe-org-environments/environments/cloe-org-env-macOS-v2026.1.yaml  # or -linux-
   mamba activate cloe-org-env-v2026.1  # cloe-org-env with the linux files
   ```

2. Install `cloelib` v0.11 and `cloelike` v0.3:

   ```bash
   mkdir cloe-org && cd cloe-org
   git clone --branch v0.11 https://github.com/cloe-org/cloelib.git
   git clone --branch v0.3 https://github.com/cloe-org/cloelike.git
   pip install -e cloelib -e cloelike
   ```

3. Install `cloerun`, which also installs the `cloerun` command:

   ```bash
   git clone https://github.com/cloe-org/cloerun.git
   pip install -e "cloerun[analyse]"
   ```

   Without local clones of `cloelib` and `cloelike`,
   `pip install -e "cloerun[cloe,analyse]"` installs both releases from
   GitHub instead of step 2.

4. Get the campaign configs, e.g. those of the Euclid DR1 analysis:

   ```bash
   git clone https://github.com/cloe-org/dr1-inference-configs.git dr1-inference
   ```

## Usage

From the `cloe-org/` folder, with the configs of `dr1-inference/`:

```bash
C=dr1-inference/cosmic-shear
cloerun sdv      $C/config.yaml                             # synthetic data vector of the config
cloerun chain    $C/config.yaml --test                      # one loglike at the fiducial point
cloerun chain    $C/config.yaml                             # sample
cloerun campaign $C/campaign.yaml                           # every run, in its run.mode
cloerun campaign $C/campaign.yaml --dry-run                 # print the resolved configs
cloerun chain    $C/campaign.yaml --run chain_WL_baseline   # one run of the campaign
cloerun analyse  $C/campaign.yaml --run chain_WL_baseline --params omch2 H0 S8
cloerun chain    cloerun/examples/gcspectro_pls_nautilus.yaml   # a spectroscopic chain
```

Every command takes a campaign file, a single pipeline config (`config.yaml`)
or a cloerun config (see [cloerun configs](#cloerun-configs)), and:

- `--run NAME` (repeat for several) selects runs of a campaign.
- `--output-dir DIR` is where the outputs go (`${output_dir}`). By default
  it is the campaign's `output_dir`, or `./chains` for a single config, so
  that nothing is written into the configs repository.
- `--set PATH=VALUE` (repeat for several) changes one setting of every run
  without editing the shared configs, as a component would: a dotted
  `PATH`, and a yaml `VALUE`. Use it for what only holds on your machine,
  e.g. the data paths or a smaller sampler:

  ```bash
  cloerun chain $C/config.yaml \
      --set run.root_dir=/path/to/dr1/data \
      --set inference.sampler.settings.n_live=400 \
      --set inference.sampler.settings.pool=8 \
      --set inference.sampler.run_settings.timeout=3600
  ```

`--test` evaluates the log-likelihood at the fiducial values of the config,
and at the prior median of the sampled parameters without one.
`cloerun <command> --help` lists the options; `python -m cloerun` works as
well.

From Python:

```python
from cloerun import load_runs, run_chain, write_sdv
from cloerun.analyse import analyse_run

for name, config, mode in load_runs("campaign.yaml", ["chain_WL_baseline"]):
    chain = run_chain(config)
    analyse_run(config, parameters=["omch2", "H0", "S8"])
```

## Campaigns

A campaign file lists runs, each built from a config (`base`) or from
another run (`parent`), and modified by named `components`. A `base` is a
[pipeline config](#pipeline-configs) (the photometric DR1 runs) or a
[cloerun config](#cloerun-configs) (any probe, including the spectroscopic
ones):

```yaml
output_dir: /path/to/output/directory
output_name: campaign_scalecuts
runs:
  - name: chain_WL_baseline
    base: config.yaml
  - name: chain_WL_baseline_w0wa
    parent: chain_WL_baseline
    components: [priors_w0wa]
components:
  priors_w0wa:
    w0: -3.0 -1.0 -0.33
    wa: -3.0 0.0 3.0
```

Each key of a component is merged where that key is in the config: at the
top level (`run`, `modelling`, `inference`; `probe`, `data`, `sampler`, ...
of a cloerun config), else at the one place it appears (`scale_cuts`,
`data_vector`, a parameter such as `w0`, ...). A dotted key
(`modelling.coupling.mode`) gives the place explicitly, which is needed for
keys that appear in several places (`mode`, `file`, ...).

A spectroscopic campaign, from the examples:

```yaml
output_dir: chains
output_name: spectro_campaign
runs:
  - name: bao_baseline
    base: examples/gcspectro_bao_nautilus.yaml
    components: [checkpoints]
  - name: bao_tight_bbn          # same run, tighter BBN prior
    parent: bao_baseline
    components: [tight_bbn]
  - name: pls_eft
    base: examples/gcspectro_pls_nautilus.yaml
components:
  checkpoints:
    sampler.settings:
      filepath: ${output_dir}/${name}.hdf5
  tight_bbn:
    ombh2: {dist: norm, loc: 0.0219961, scale: 0.0001}
```

The outputs of a run built from a cloerun config go to the campaign's
`${output_dir}`, named after the run; its other settings are kept. Such runs
are chains (`cloerun chain --test` evaluates one loglike); `cloerun sdv` only
supports the photometric probes.

Strings may use `${name}` (the run), `${output_dir}`
(`<output_dir>/<output_name>`, where the outputs of the runs are written) and
`${root_dir}` (`run.root_dir`). Relative paths are resolved against the
campaign file for `base` and `output_dir`, and against the pipeline config
otherwise.

### Pipeline configs

A pipeline config has three blocks:

- **`run`**: `mode` -- `chain`, `sdv` (write the theory at the fiducial point
  to `modelling.output.file`, to read back as `data_vector`) or `test` -- and
  `root_dir`.
- **`modelling`**: `probe` (`WL`, `GCph`, `GGL`, `2x2pt`, `3x2pt`),
  `parameters` (the parameter file), `nz` (`file`, or `shear` and
  `positions`: FITS, or a text table of z and one column per bin such as the
  Spaceborne `nz_shear.txt`; `zs_min`, `zs_max`, `zs_num`), `theory`,
  `systematics` (passed to `cloelike` with `settings`), `coupling` (`mode`,
  `mixing_matrices`), `binning` (`kind`: `log`, `linear`, `sqrt`, or
  `fromfile` with a text `file` whose first column is ell, used when there is
  no data vector to read the multipoles from) and `output.file`.
- **`inference`**: `likelihood` (`data_vector`, `covariance`,
  `covariance_key`, `noise_free_mock`), `scale_cuts` (`per_block`, with
  `WL`, `GGL` and `GC`, or `from_file`, a yaml file of one range per
  spectrum), `derived_outputs` (`sigma8_0`, `S8`, stored as chain columns)
  and `sampler`. The `sdv` mode does not read it, so it may be left out.

In the parameter file, `value` is fixed, `min start max` is a flat prior and
`gauss mean sigma` a Gaussian one, with `start` and `mean` the fiducial
values. `logAs` and `dz_she_{i}` are read as `logA` and `dz_shear_{i}`;
`Omega_k0`, `alpha_s`, `CIA` and the n(z) widths default to `0`, `0`, `0.0134`
and `1`; and the parameters the likelihood does not read (e.g. the galaxy
bias of a WL run) are dropped, so one parameter file serves every probe.

`sampler.settings.pool` evaluates the likelihood in that many processes; the
likelihood is rebuilt once in each of them, so this works on macOS too.

## cloerun configs

The pipeline configs are translated into cloerun configs, which can also be
written directly: see the annotated `examples/3x2pt_nautilus.yaml` for the
photometric options and `examples/gcspectro_pls_nautilus.yaml` for the
spectroscopic ones. `cloerun campaign --dry-run` prints the cloerun config of
each run. File paths in `data` are relative to `data.base_directory`, itself
relative to the directory of the config file.

The examples read the synthetic data of the
[cloe-org Zenodo community](https://zenodo.org/communities/cloe-org/records),
the same files as the [playground](https://github.com/cloe-org/playground)
notebooks and scripts. By default they expect them in `../data`, next to this
repository:

| Examples | Files | Zenodo record |
|---|---|---|
| `3x2pt`, `wl` | `nz_example.fits` | [15092862](https://doi.org/10.5281/zenodo.15092862) |
| `3x2pt`, `wl` | `synthetic_coupled_cells.fits`, `mixmats_example.fits`, `covmat_2D.npz` | [19071608](https://doi.org/10.5281/zenodo.19071608) |
| `gcspectro_pls`, `gcspectro_pls_bao` | `mps_pk_*`, `cov_pk_Gauss_GCspectro_comet_*`, `mixmat_pk_*` | [18711304](https://doi.org/10.5281/zenodo.18711304) |
| `gcspectro_bao`, `gcspectro_pls_bao` | `EUC_LE3_BAO_*` | [19729182](https://doi.org/10.5281/zenodo.19729182) |
| `gcspectro_pls_bao` | `cov_pk_Gauss_GCspectro_FS+BAO_*` | [21219969](https://doi.org/10.5281/zenodo.21219969) |

For the photometric examples:

```bash
mkdir -p ../data && cd ../data
curl -LO https://zenodo.org/records/15092862/files/nz_example.fits
curl -LO https://zenodo.org/records/19071608/files/synthetic_coupled_cells.fits
curl -LO https://zenodo.org/records/19071608/files/mixmats_example.fits
curl -LO https://zenodo.org/records/19071608/files/covmat_2D.npz
cd - && cloerun chain examples/wl_nautilus.yaml
```

For the spectroscopic examples (about 40 MB):

```bash
mkdir -p ../data && cd ../data
for record in 18711304 19729182 21219969; do
  curl -s https://zenodo.org/api/records/$record \
    | python -c "import json, sys; [print(f['key'], f['links']['self']) for f in json.load(sys.stdin)['files']]" \
    | while read -r name url; do curl -sL -o "$name" "$url"; done
done
cd - && cloerun chain examples/gcspectro_bao_nautilus.yaml
```

The spectroscopic examples need `COMET` (`comet-emu`) for the multipoles, or
`PBJ` (`pbjcosmo`) with `spectro_power.code: PBJ`.

In the photometric configs:

- **`probe`** picks the theory model: `background`, `linear_perturbations`
  and `nonlinear_perturbations` (`CAMB`, `HMcode2020Emu`, `EE2`), an optional
  `baryon_boost` (`HMcode2020`, `FLAMINGO`) and the `mode` (`coupled` with
  the mixing matrices, or `decoupled`). `probe.settings` is passed as it is
  to the `cloelike` likelihood: `ia_model` (`NLA`, `TATT`, `null`),
  `galaxy_bias_model` (`poly`, `per_bin`, `per_bin_int`, `nonlinear`),
  `include_rsd`, `nonlinear_param_keys`, ...
- **`data`** gives the n(z) (`nz`, or `nz_positions` and `nz_shear`), the
  redshift grid they are resampled on (`z_grid`), the angular power spectra
  (`cells`), the mixing matrices (`mixing`) and the covariance (`covariance`,
  `.npz` or `.npy`, with `covariance_key` for a `.npz`). The covariance can
  cover the full 3x2pt data vector (blocks ordered WL, GGL, GCph) for any
  probe: the blocks of the probe are extracted from it.
- **`scale_cuts`** gives one multipole range per block (`WL`, `GGL`, `GCph`),
  and optionally per-spectrum ranges under `overrides`, e.g.
  `POS-SHE-1-3: [10, 300]`.
- **Per-bin parameters** are one scalar per tomographic bin in `cloelike`
  (`dz_shear_1`, ..., `b1_photo_bin0`, ...). Write them once with the `{i}`
  placeholder, in `fixed_parameters` or `sampled_parameters`:
  `dz_shear_{i}: {dist: norm, loc: 0.0, scale: 0.01}` expands to every bin,
  with the indexing `cloelike` uses for that parameter. An explicit name,
  e.g. `dz_shear_3`, overrides the template for that bin.
- **`data.noise_free_mock: true`** replaces the data vector with the theory
  at the fiducial point: `fixed_parameters` plus `fiducial_parameters`. The
  synthetic data was generated with a single model, so this is the way to
  get a posterior centred on the fiducial with any other model.

Sampling `omch2`, `ombh2`, `logA` or `h` enables their conversion to the native
parameters automatically (see `cloerun._core.derived_parameters`).

Probes, samplers and theory backends are behind small lookup tables in
`cloerun._core.loader` (`PROBES`, `SAMPLERS`, `BACKGROUNDS`, `PERTURBATIONS`,
`NONLINEAR_PERTURBATIONS`, `BARYON_BOOSTS`, `SPECTRO_POWERS`): adding one is a
new module plus one entry there.

## Output

A chain writes two files to `output.directory` (`${output_dir}` for a
campaign), both named after the run:

- **`<name>.npz`**: the chain, in the same format whatever the sampler
  (see `cloerun.chain.Chain`): a 2D `samples` array and a `columns` list,
  always `[logweight, loglikelihood, <parameter_1>, ..., <parameter_N>]`,
  followed by the `derived_outputs` if any. Load it with `Chain.load(path)`,
  or directly with `numpy.load`.
- **`<name>.config.yaml`**: the cloerun config of the run, with a
  `software_versions` block recording the installed versions of `cloerun`,
  `cloelike`, `cloelib`, `euclidlib`, `numpy`, `scipy` and of the packages
  the run depends on.

`cloerun analyse` adds, next to them:

- **`<name>.summary.md`**: weighted mean, standard deviation, 16/50/84%
  quantiles of every column, with the fiducial value and the shift of the
  mean from it in posterior sigmas.
- **`<name>.triangle.pdf`**: a [`getdist`](https://getdist.readthedocs.io/)
  triangle plot of the `--params` (default: all but the per-bin nuisance
  parameters), with the fiducial values marked.

## Tests

```bash
pip install -e ".[test,analyse]"
pytest
```

## Example output

![Example posterior](docs/example_posterior.png)
