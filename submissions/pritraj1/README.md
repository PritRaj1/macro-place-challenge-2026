This is not a competition submission, (fork was created after closing). I'm just curious.

# Plan

Two parts to the problem:

|  | Characteristics | Cost |
|:---|:---:|---:|
| Legalization | discrete, combinatorial, local, dynamic (moving) hard constraints | displacement from illegal, residual wirelength |
| Relative placement | discrete (can be relaxed), combinatorial, global, soft constraints | wirelength, density |

## 1. Legalization

Approach: constraint graph + lp

We map out the hard separation constraints to a pairwise graph and use existing linear programming packages to satisfy them while minimizing displacement to a target. Aims to preserve approximate position:

<div align="center">

<img src="../../gifs/legalize_ibm1.png" width="400">

<sub><b>Legalization on IBM01, overlaps=0, positions preserved</b></sub>

</div>

<div align="center">

<img src="../../gifs/legalize_ibm1_centered.png" width="400">

<sub><b>Legalization when all overlapping (centred)</b></sub>

</div>

## 2. Global placer in front of legalizer

Approach: smooth energy-based langevin dynamics

We relax relative placement into a continuous optimization problem using Langevin diffusion. Drives hard macros toward low-wirelength, uniform-density states prior to LP legalization:

<div align="center">

<img src="../../gifs/pritraj_ibm01.gif" width="400">

<sub><b>IBM01 optimization</b></sub>

</div>

## 3. Co-optimization

Approach: force-directed soft-macro alignment

Hard macro moves disrupt net connections to standard cell clusters. After legalization, run force-directed placement on soft macros (`plc.optimize_stdcells`) to re-align clusters around fixed hard macros, minimizing residual HPWL and density degradation.

<div align="center">

<img src="../../gifs/pritraj_ibm01.png" width="400">

<sub><b>IBM01 final</b></sub>

</div>
