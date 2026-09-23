# FluxNoise
This repository has code and simulations used for modeling the dynamics and behavior of a TLSs spins bath coupled to a superconducting qubit loop. The effect of both qubit magnetic field and external magnetic field are taken into account for solving the TLSs system.

## 1. Installing the superconducting version of FastHenry

First, download the **superconducting version of FastHenry**. It is available from the following link:

http://www.wrcad.com/freestuff.html

This version works on **Linux** as well as Linux environments running on Windows, such as **MinGW**.

Once downloaded, copy the files and folders from this repository into the FastHenry installation directory.

---

## 2. Generating the FastHenry input file

To generate the geometry and mesh of the rectangular superconducting loop, run:

```bash
python3 gen_rectangle_loop.py
```

The **y-length is automatically set to 5 μm**. To change this value, modify it directly in `gen_rectangle_loop.py`.

When prompted, choose the **x-length** of the superconducting loop.

The script generates a FastHenry input file with the following naming convention:

```text
rectangular_loop_rounded_(x length)x(y length).inp
```

For example:

```text
rectangular_loop_rounded_50x5.inp
```

This `.inp` file is the input file used by FastHenry.

### Optional: visualize the mesh

To visualize the generated mesh, run:

```bash
python3 plot_mesh.py
```

This can be used to check the geometry and adjust the mesh resolution. The mesh settings can be modified directly in `gen_rectangle_loop.py`.

---

## 3. Running FastHenry

Once the `.inp` file has been generated, compile it using FastHenry:

```bash
./bin/fasthenry filename.inp -d grids
```

FastHenry does not provide an option to customize the output filenames. Each compilation generates:

```text
Jmag1_0.mat
Jimag1_0.mat
Jreal1_0.mat
```

Since we work in the **DC limit**, corresponding to a current frequency close to zero, the relevant file is:

```text
Jmag1_0.mat
```

To prevent this file from being overwritten by subsequent simulations, rename it according to the following convention:

```bash
mv Jmag1_0.mat (x length)x(y length).mat
```

For example:

```bash
mv Jmag1_0.mat 50x5.mat
```

---

## 4. Processing the FastHenry data

The generated `.mat` files can be moved to the `loop_simulation` folder, which contains several tools for processing and visualizing the results.

### `interpolate.py`

The current density is continuous and smooth. This script interpolates the FastHenry data onto a finer grid, producing `.mat` files with higher resolution without introducing or removing physical information from the original data.

### `current_plot.py`

This script allows the current density to be plotted and provides current-density cuts through the loop. It can be used to visualize and inspect the FastHenry results.

---

## 5. Spin-bath and flux-noise calculations

Once the current-density data have been processed, they can be used for the spin-bath calculations.

The repository contains two main Jupyter notebooks:

### `FluxNoise`

The `FluxNoise` notebook performs the **numerical calculations** required for the spin-bath analysis, using the current-density distributions obtained from FastHenry. Further information is in the notebook and the main solvers are in the `diffusion2d_complete.py` file

### `analytical`

The `analytical` notebook obtains the corresponding **analytical results**. It contains the solution of the diffusion problem using Fourier transforms, together with the equations for an infinite superconducting strip.

The two notebooks can therefore be used to compare the numerical results obtained from the FastHenry current distributions with the analytical predictions for an infinite strip.
