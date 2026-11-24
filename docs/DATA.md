# Data

All datasets are public (Zenodo). `nett data download` fetches only the files the paper uses into
`$NETT_DATA/raw/<name>/` (MD5 of every file is verified against the checksum Zenodo reports for the record), and
`nett data prepare` builds the layout read by `nett/data/registry.py`.

| key | Zenodo record | file | notes |
|---|---|---|---|
| `nielsen-m`, `-t`, `-mammoth` | 7673985 | `{M,T,mammoth}_sim_{1.0e+01,3.2e+01,1.0e+02}.h5` | simulated, ground truth in the `model` group; larger Poisson rate = less noise |
| `bone-main/-remount/-combined` | 10995088 | `data_set_1.h5`, `data_set_2.h5`, `full_geometry.geo` (inside the 1.9 GB `investigating_missing_wedge.zip`) | two mounts of one sample; both padded to 65×66 detector pixels; combined = both mounts |
| `steel-wire` | 10889439 | `gamma_220.h5` | WAXS (Carlsen et al. 2024), **hot pixel masked** |
| `aragonite` | 14800063 | `aragonite_221.h5` | WAXS, **rebinned 96 → 48 azimuthal bins** |
| `beaver-tooth` | 15826720 | `apatite_002.h5` | WAXS |
| `narwhal-tusk` | 15089410 | `SASTT_ID108_150um_mineral_projections.h5` (inside the 631 MB `SAXS_tensor_tomography-projections.zip`) | SAXS (check the licence stated on the record) |

## The two modifications of raw data (`nett/data/prepare.py`)

* **aragonite**: the Zenodo file has 96 azimuthal bins. The experiments use 48 (mean of adjacent bin pairs; data and
  bin-centre angles), like the steel-wire and beaver-tooth files. With 96 bins the SH baseline exceeded 256 GB of memory.
* **steel wire**: in `gamma_220.h5`, projection 166, pixel (46, 58) has the value 6.9e8 (all other valid pixels are
  below 5e3) with valid weight 1. Its weight is set to 0. Nothing else is changed.

`nett data download` + `nett data prepare` were run from scratch on 2026-11-24 (about 2.9 GB, 2.5 min): every file matched the
Zenodo checksum, the extracted bone and narwhal files are byte-identical (MD5) to the copies used for the paper, and the Nielsen,
steel-wire, aragonite and beaver-tooth downloads are MD5-identical to the paper's raw files. `nett data prepare` was verified to reproduce, bit-for-bit in content (SHA-256 over all datasets), the two prepared
files that were used to produce the paper's numbers.

## Derived data created on the fly

Pruned subsets (sparsity experiment: Max-Min / random removal orders, nested; level = % of projections kept) and
hold-out splits are computed in memory from the loaded data (`nett/data/splits.py`); no files are written. The random
orders and splits were verified to be identical to the files used for the paper (random pruning, levels 20/50/90,
seeds 0/1/2: identical data and angles).

Hold-out conventions: phantoms and bone train on all but one projection (`holdout="none"`; the held-out projection
only monitors training); experimental datasets use a random 15 % hold-out (seed 42) or the *high-tilt* hold-out (all
projections of the highest tilt level).

## Layout

```
$NETT_DATA/raw/<name>/...          downloads
$NETT_DATA/nielsen/ bone/ steel_wire/ aragonite_b48/ beaver_tooth/ narwhal_tusk/     prepared (links or derived files)
```
