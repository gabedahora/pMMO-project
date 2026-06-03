# pMMO-project
This will contain files and script for the pMMO project


# Quinol Mapper (replace_quinol_exactmap_v2.py)

Membrane-aware quinol replacement for GROMACS systems using ITP/PDB-guided exact mapping, tail-aware alignment, and automatic topology updates.

## Overview

This repository contains tools to replace quinols in prebuilt GROMACS membrane systems while preserving the physical orientation of the molecules in the bilayer.

The main use case is replacing an existing quinol species already embedded in a membrane, such as `UQOL8`, with a different quinol-like molecule such as:

- `MUQO8`
- `MUQOL8`
- `DPLA`
- `DURO`

The core idea is simple:

- keep the **headgroup aligned to the original headgroup**
- keep the **tail aligned to the original tail**
- reuse as many existing coordinates as possible
- place only the unmatched atoms more loosely
- preserve membrane-facing orientation as much as possible

This is especially useful when the new ligand is chemically similar to the old one, but not identical in atom count or side-chain structure.

---l

## Why this repository exists

Naive ligand replacement in membrane systems often causes serious problems:

- tails pointing toward water
- headgroups buried in the bilayer core
- molecules lying flat on the protein surface
- broken local packing around quinols
- excessive manual correction in VMD

This toolkit was built to reduce those issues by using:

1. **topology-aware atom matching**
2. **template-guided coordinate transfer**
3. **tail-aware map expansion**
4. **membrane-aware orientation checks**
5. **automatic `.top` updates**

---

## Main features

- Replace quinols directly in a `.gro + .top` GROMACS system
- Read atom naming and bonding from `.itp`
- Read coordinates from ligand template `.pdb`
- Use a seed atom map and automatically expand it
- Prefer **head-to-head** and **tail-to-tail** matching
- Preserve shared tail carbon backbone whenever possible
- Allow unmatched atoms to be placed more loosely
- Update topology include statements and molecule names
- Write expanded maps for inspection and reuse
- Support membrane midplane-aware placement control

---

## Repository contents

Typical files in this repository may include:

- `replace_quinol_exactmap_v2.py`  
  Main replacement script using expanded exact mapping

- `UQOL8_to_*.map`  
  Seed maps between old and new quinols

- `UQOL8_to_*.expanded.map`  
  Expanded maps generated automatically by the script

- helper scripts for:
  - orientation checking
  - map generation
  - format conversion
  - PACKMOL fallback workflows

---

## Requirements

- Python 3.9+
- `numpy`

Optional but useful:

- `scipy`
- `MDAnalysis`
- `packmol`
- `VMD`
- `GROMACS`

A conda or micromamba environment is recommended.

Example:

```bash
micromamba create -n quinol-tools python=3.11 numpy scipy mdanalysis -c conda-forge
micromamba activate quinol-tools

python ../../replace_quinol_exactmap_v2.py \
  --in-gro step5_input.gro \
  --top topol.top \
  --out-gro step5_input_MUQO8_exact_v2.gro \
  --out-top topol_MUQO8_exact_v2.top \
  --old-resname UQOL8 \
  --new-resname MUQO8 \
  --new-itp toppar/MUQO8.itp \
  --template-pdb ligandrm.pdb \
  --template-units A \
  --atom-map ../../UQOL8_to_MUQO8.map \
  --write-expanded-map UQOL8_to_MUQO8.expanded.v2.map \
  --seed 1 \
  --tail-rotate-trials 256 \
  --min-heavy-dist 0.07 \
  --midplane-z 8.5265
