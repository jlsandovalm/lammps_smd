/* ----------------------------------------------------------------------
 *
 *                    *** Smooth Mach Dynamics ***
 *
 * This file is part of the USER-SMD package for LAMMPS.
 * Copyright (2014) Georg C. Ganzenmueller, georg.ganzenmueller@emi.fhg.de
 * Fraunhofer Ernst-Mach Institute for High-Speed Dynamics, EMI,
 * Eckerstrasse 4, D-79104 Freiburg i.Br, Germany.
 *
 * ----------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 http://lammps.sandia.gov, Sandia National Laboratories
 Steve Plimpton, sjplimp@sandia.gov

 Copyright (2003) Sandia Corporation.  Under the terms of Contract
 DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
 certain rights in this software.  This software is distributed under
 the GNU General Public License.

 See the README file in the top-level LAMMPS directory.
 ------------------------------------------------------------------------- */

#include "pair_smd_mpm.h"

#include "math.h"
#include "float.h"
#include "stdlib.h"
#include "string.h"
#include "atom.h"
#include "domain.h"
#include "force.h"
#include "update.h"
#include "modify.h"
#include "fix.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"
#include <stdio.h>
#include <iostream>
#include "smd_material_models.h"
#include "smd_math.h"
#include "smd_kernels.h"

using namespace SMD_Kernels;
using namespace std;
using namespace LAMMPS_NS;
using namespace SMD_Math;

#include <Eigen/SVD>
#include <Eigen/Eigen>
using namespace Eigen;

#define FORMAT1 "%60s : %g\n"
#define FORMAT2 "\n.............................. %s \n"
#define BIG 1.0e22
#define MASS_CUTOFF 1.0e-8
#define STENCIL_LOW 1
#define STENCIL_HIGH 3
#define GRID_OFFSET 2

PairSmdMpm::PairSmdMpm(LAMMPS *lmp) :
		Pair(lmp) {

	// per-type arrays
	Q1 = NULL;
	eos = viscosity = strength = NULL;
	c0_type = NULL;
	c0 = NULL;
	heat_conduction_coeff = NULL;
	Lookup = NULL;

	nmax = 0; // make sure no atom on this proc such that initial memory allocation is correct
	stressTensor = L = F = NULL;
	numNeighs = NULL;
	particleVelocities = particleAccelerations = NULL;
	heat_gradient = NULL;
	particleHeat = particleHeatRate = NULL;

	comm_forward = 20; // this pair style communicates 8 doubles to ghost atoms

	Bp_exists = false;
	APIC = false;
}

/* ---------------------------------------------------------------------- */

PairSmdMpm::~PairSmdMpm() {
	if (allocated) {
		//printf("... deallocating\n");
		memory->destroy(Q1);
		memory->destroy(rho0);
		memory->destroy(eos);
		memory->destroy(viscosity);
		memory->destroy(strength);
		memory->destroy(c0_type);
		memory->destroy(heat_conduction_coeff);
		memory->destroy(Lookup);

		delete[] c0;
		delete[] stressTensor;
		delete[] L;
		delete[] F;
		delete[] numNeighs;
		delete[] particleVelocities;
		delete[] particleAccelerations;
		delete[] heat_gradient;
		delete[] particleHeatRate;
		delete[] particleHeat;

	}
}

/* ---------------------------------------------------------------------- */

void PairSmdMpm::CreateGrid() {
	double **x = atom->x;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	int nall = nlocal + atom->nghost;
	int i, itype;
	double minx, miny, minz, maxx, maxy, maxz;
	icellsize = 1.0 / cellsize; // inverse of cell size

	// get bounds of this processor's simulation box
	//printf("bounds min: %f %f %f\n", domain->sublo[0], domain->sublo[1], domain->sublo[2]);
	//printf("bounds max: %f %f %f\n", domain->subhi[0], domain->subhi[1], domain->subhi[2]);

	// get min / max position of all particles
	minx = miny = minz = BIG;
	maxx = maxy = maxz = -BIG;

	for (i = 0; i < nall; i++) {
		itype = type[i];
		if (setflag[itype][itype]) {
			minx = MIN(minx, x[i][0]);
			maxx = MAX(maxx, x[i][0]);
			miny = MIN(miny, x[i][1]);
			maxy = MAX(maxy, x[i][1]);
			minz = MIN(minz, x[i][2]);
			maxz = MAX(maxz, x[i][2]);
		}
	}

	// we want the leftmost index to be 0, i.e. index(minx - kernel bandwidth > 0
	// to this end, we assume that the kernel does not cover more than three cells to either side

	min_ix = icellsize * minx;
	max_ix = icellsize * maxx;
	//printf("minx=%f, min_ix=%d\n", minx, min_ix);
	min_iy = icellsize * miny;
	max_iy = icellsize * maxy;
	min_iz = icellsize * minz;
	max_iz = icellsize * maxz;

	grid_nx = (max_ix - min_ix) + 5;
	grid_ny = (max_iy - min_iy) + 5;
	grid_nz = (max_iz - min_iz) + 5;

	// allocate grid storage
	// we need a triple of indices (i, j, k)

	memory->create(gridnodes, grid_nx, grid_ny, grid_nz, "pair:gridnodes");

}

/*
 * Steps 1 to 3, i.e.,
 * 1) compute node mass
 * 2) compute node momentum
 * 3) compute node velocity
 */

void PairSmdMpm::PointsToGrid() {
	double **smd_data_9 = atom->smd_data_9;
	double **x = atom->x;
	double **v = atom->v;
	double **vest = atom->vest;
	double *rmass = atom->rmass;
	double *heat = atom->heat;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	int nall = nlocal + atom->nghost;
	int i, itype;
	int ix, iy, iz, jx, jy, jz;
	double delx_scaled, delx_scaled_abs, dely_scaled, dely_scaled_abs, wf, wfx, wfy;
	double delz_scaled, delz_scaled_abs, wfz;
	double px_shifted, py_shifted, pz_shifted; // shifted coords of particles
	Vector3d vel_APIC, vel_particle, vel_particle_est, dx;
	Matrix3d eye, Dp, Dp_inv, Cp, Bp;

	// clear the grid
	for (ix = 0; ix < grid_nx; ix++) {
		for (iy = 0; iy < grid_ny; iy++) {
			for (iz = 0; iz < grid_nz; iz++) {
				gridnodes[ix][iy][iz].mass = 0.0;
				gridnodes[ix][iy][iz].heat = 0.0;
				gridnodes[ix][iy][iz].dheat_dt = 0.0;
				gridnodes[ix][iy][iz].f.setZero();
				gridnodes[ix][iy][iz].v.setZero();
				gridnodes[ix][iy][iz].vest.setZero();
				gridnodes[ix][iy][iz].isVelocityBC = false;
			}
		}
	}

	// set up Dp matrix for APIC correction
	eye.setIdentity();
	Dp = (1. / 3.) * cellsize * cellsize * eye;
	if (domain->dimension == 2) {
		Dp(2, 2) = 1.0;
	}
	Dp_inv = Dp.inverse();
	Cp.setZero();

	for (i = 0; i < nall; i++) {

		itype = type[i];
		if (setflag[itype][itype]) {

			px_shifted = x[i][0] - min_ix * cellsize + GRID_OFFSET * cellsize;
			py_shifted = x[i][1] - min_iy * cellsize + GRID_OFFSET * cellsize;
			pz_shifted = x[i][2] - min_iz * cellsize + GRID_OFFSET * cellsize;

			ix = icellsize * px_shifted;
			iy = icellsize * py_shifted;
			iz = icellsize * pz_shifted;

			vel_particle << v[i][0], v[i][1], v[i][2];
			vel_particle_est << vest[i][0], vest[i][1], vest[i][2];

			if (APIC) {
				if (Bp_exists) {
					/*
					 * get the APIC correction matrix B from atom data structure
					 */
					Bp(0, 0) = smd_data_9[i][0];
					Bp(0, 1) = smd_data_9[i][1];
					Bp(0, 2) = smd_data_9[i][2];
					Bp(1, 0) = smd_data_9[i][3];
					Bp(1, 1) = smd_data_9[i][4];
					Bp(1, 2) = smd_data_9[i][5];
					Bp(2, 0) = smd_data_9[i][6];
					Bp(2, 1) = smd_data_9[i][7];
					Bp(2, 2) = smd_data_9[i][8];

					Cp = Bp * Dp_inv; // use the stored Bp from last timestep
				} else {
					Cp.setZero();
				}
			}

			for (jx = ix - STENCIL_LOW; jx < ix + STENCIL_HIGH; jx++) {

				// check that cell indices are within bounds
				if ((jx < 0) || (jx >= grid_nx)) {
					printf("x cell index %d is outside range 0 .. %d\n", jx, grid_nx);
					error->one(FLERR, "");
				}

				delx_scaled = px_shifted * icellsize - 1.0 * jx;
				dx(0) = delx_scaled * cellsize;
				delx_scaled_abs = fabs(delx_scaled);
				wfx = DisneyKernel(delx_scaled_abs);
				if (wfx > 0.0) {

					for (jy = iy - STENCIL_LOW; jy < iy + STENCIL_HIGH; jy++) {

						if ((jy < 0) || (jy >= grid_ny)) {
							printf("y cell indey %d is outside range 0 .. %d\n", jy, grid_ny);
							error->one(FLERR, "");
						}

						dely_scaled = py_shifted * icellsize - 1.0 * jy;
						dx(1) = dely_scaled * cellsize;
						dely_scaled_abs = fabs(dely_scaled);
						wfy = DisneyKernel(dely_scaled_abs);
						if (wfy > 0.0) {

							for (jz = iz - STENCIL_LOW; jz < iz + STENCIL_HIGH; jz++) {

								if ((jz < 0) || (jz >= grid_nz)) {
									printf("z cell index %d is outside range 0 .. %d\n", jz, grid_nz);
									error->one(FLERR, "");
								}

								delz_scaled = pz_shifted * icellsize - 1.0 * jz;
								dx(2) = delz_scaled * cellsize;
								delz_scaled_abs = fabs(delz_scaled);
								wfz = DisneyKernel(delz_scaled_abs);
								if (wfz > 0.0) {

									wf = wfx * wfy * wfz; // this is the total weight function -- a dyadic product of the cartesian weight functions

									if (APIC) {
										vel_APIC = Cp * dx + vel_particle; // this is the APIC corrected velocity
										gridnodes[jx][jy][jz].v += wf * rmass[i] * vel_APIC;
									} else {
										gridnodes[jx][jy][jz].v += wf * rmass[i] * vel_particle;
									}

									gridnodes[jx][jy][jz].vest += wf * rmass[i] * vel_particle_est;

									gridnodes[jx][jy][jz].mass += wf * rmass[i];
									gridnodes[jx][jy][jz].heat += wf * rmass[i] * heat[i];
								}
							}
						}

					}

				}
			}
		} // end if setflag[itype][itype]
	}

	// normalize grid data
	for (ix = 0; ix < grid_nx; ix++) {
		for (iy = 0; iy < grid_ny; iy++) {
			for (iz = 0; iz < grid_nz; iz++) {
				if (gridnodes[ix][iy][iz].mass > MASS_CUTOFF) {
					gridnodes[ix][iy][iz].vest /= gridnodes[ix][iy][iz].mass;
					gridnodes[ix][iy][iz].v /= gridnodes[ix][iy][iz].mass;
					gridnodes[ix][iy][iz].heat /= gridnodes[ix][iy][iz].mass;
				}
			}
		}
	}
}

void PairSmdMpm::ApplyVelocityBC() {
	double **x = atom->x;
	double **v = atom->v;
	int *type = atom->type;
	tagint *mol = atom->molecule;
	int nlocal = atom->nlocal;
	int nall = nlocal + atom->nghost;
	int i, itype;
	int ix, iy, iz, jx, jy, jz;
	double delx_scaled, delx_scaled_abs, dely_scaled, dely_scaled_abs, wf, wfx, wfy;
	double delz_scaled, delz_scaled_abs, wfz;
	double px_shifted, py_shifted, pz_shifted; // shifted coords of particles
	Vector3d vel_particle;

	for (i = 0; i < nall; i++) {

		itype = type[i];
		if (setflag[itype][itype]) {

			/*
			 * only do this for boundary particles which have mol id = 1000
			 */

			if (mol[i] == 1000) {

				//if (APIC) {
				//	error->one(FLERR, "Cannot use APIC with velocity boundary condtions\n");
				//}

				px_shifted = x[i][0] - min_ix * cellsize + GRID_OFFSET * cellsize;
				py_shifted = x[i][1] - min_iy * cellsize + GRID_OFFSET * cellsize;
				pz_shifted = x[i][2] - min_iz * cellsize + GRID_OFFSET * cellsize;

				ix = icellsize * px_shifted;
				iy = icellsize * py_shifted;
				iz = icellsize * pz_shifted;

				vel_particle << v[i][0], v[i][1], v[i][2];

				for (jx = ix - STENCIL_LOW; jx < ix + STENCIL_HIGH; jx++) {
					delx_scaled = px_shifted * icellsize - 1.0 * jx;
					delx_scaled_abs = fabs(delx_scaled);
					wfx = DisneyKernel(delx_scaled_abs);

					for (jy = iy - STENCIL_LOW; jy < iy + STENCIL_HIGH; jy++) {
						dely_scaled = py_shifted * icellsize - 1.0 * jy;
						dely_scaled_abs = fabs(dely_scaled);
						wfy = DisneyKernel(dely_scaled_abs);

						for (jz = iz - STENCIL_LOW; jz < iz + STENCIL_HIGH; jz++) {
							delz_scaled = pz_shifted * icellsize - 1.0 * jz;
							delz_scaled_abs = fabs(delz_scaled);
							wfz = DisneyKernel(delz_scaled_abs);

							wf = wfx * wfy * wfz; // this is the total weight function -- a dyadic product of the cartesian weight functions

							if (wf > 0.0) {
								gridnodes[jx][jy][jz].v = vel_particle;
								gridnodes[jx][jy][jz].vest = vel_particle;
								gridnodes[jx][jy][jz].isVelocityBC = true;
							}

						}

					}
				}
			} // end if mol = 1000
		} // end if setflag[itype][itype]
	}

}

/*
 * Solve the momentum balance on the grid
 * 4) compute particle deformation gradient
 * 5) update particle strain and compute updated particle stress
 * 6) compute internal grid forces from particle stresses
 */

void PairSmdMpm::ComputeVelocityGradient() {
	int *type = atom->type;
	double **x = atom->x;
	int nlocal = atom->nlocal;
	int i, itype;

	int ix, iy, iz, jx, jy, jz;
	double px_shifted_scaled, py_shifted_scaled, pz_shifted_scaled; // shifted coords of particles
	Vector3d g, vel_grid;
	Matrix3d velocity_gradient;
	double delx_scaled, dely_scaled, wfx, wfy, wf, wfdx, wfdy;
	double delz_scaled, wfz, wfdz;

	for (i = 0; i < nlocal; i++) {

		itype = type[i];
		if (setflag[itype][itype]) {

			heat_gradient[i].setZero();
			velocity_gradient.setZero();
			px_shifted_scaled = x[i][0] * icellsize - min_ix + GRID_OFFSET; // these are the particle's coords in units of the underlying grid,
			py_shifted_scaled = x[i][1] * icellsize - min_iy + GRID_OFFSET; // shifted in space to align with the grid
			pz_shifted_scaled = x[i][2] * icellsize - min_iz + GRID_OFFSET;

			ix = (int) px_shifted_scaled;
			iy = (int) py_shifted_scaled;
			iz = (int) pz_shifted_scaled;

			for (jx = ix - STENCIL_LOW; jx < ix + STENCIL_HIGH; jx++) {
				delx_scaled = px_shifted_scaled - 1.0 * jx;
				DisneyKernelAndDerivative(icellsize, delx_scaled, wfx, wfdx);
				if (wfx > 0.0) {

					for (jy = iy - STENCIL_LOW; jy < iy + STENCIL_HIGH; jy++) {
						dely_scaled = py_shifted_scaled - 1.0 * jy;
						DisneyKernelAndDerivative(icellsize, dely_scaled, wfy, wfdy);
						if (wfy > 0.0) {

							for (jz = iz - STENCIL_LOW; jz < iz + STENCIL_HIGH; jz++) {
								delz_scaled = pz_shifted_scaled - 1.0 * jz;
								DisneyKernelAndDerivative(icellsize, delz_scaled, wfz, wfdz);
								if (wfz > 0.0) {

									wf = wfx * wfy * wfz; // this is the total weight function -- a dyadic product of the cartesian weight functions

									g(0) = wfdx * wfy * wfz; // this is the kernel gradient
									g(1) = wfdy * wfx * wfz;
									g(2) = wfdz * wfx * wfy;

									velocity_gradient += gridnodes[jx][jy][jz].vest * g.transpose();
									heat_gradient[i] += gridnodes[jx][jy][jz].heat * g; // units: energy / distance
								}
							}
						}
					}
				}
			}
			L[i] = velocity_gradient;
		} // end if (setflag[itype][itype])
	}
}
// ---- end velocity gradients ----

void PairSmdMpm::ComputeGridForces() {
	double **x = atom->x;
	double **f = atom->f;
	double *vfrac = atom->vfrac;
	double *rmass = atom->rmass;
	int *type = atom->type;
	int nall = atom->nlocal + atom->nghost;
	int i, itype;
	int ix, iy, iz, jx, jy, jz;
	double px_shifted_scaled, py_shifted_scaled, pz_shifted_scaled;
	Vector3d g, force, scaled_temperature_gradient;
	double delx_scaled, dely_scaled, delz_scaled, wfx, wfy, wfz, wf, wfdx, wfdy, wfdz;
	Matrix3d scaledStress;

// ---- compute internal forces ---
	for (i = 0; i < nall; i++) {

		itype = type[i];
		if (setflag[itype][itype]) {

			px_shifted_scaled = x[i][0] * icellsize - min_ix + GRID_OFFSET; // these are the particle's coords in units of the underlying grid,
			py_shifted_scaled = x[i][1] * icellsize - min_iy + GRID_OFFSET; // shifted in space to align with the grid
			pz_shifted_scaled = x[i][2] * icellsize - min_iz + GRID_OFFSET;

			ix = (int) px_shifted_scaled;
			iy = (int) py_shifted_scaled;
			iz = (int) pz_shifted_scaled;

			scaledStress = -vfrac[i] * stressTensor[i];
			scaled_temperature_gradient = -vfrac[i] * heat_conduction_coeff[itype] * heat_gradient[i]
					/ (Lookup[HEAT_CAPACITY][itype] * rmass[i]); // units: volume * Temperature  / distance

			for (jx = ix - STENCIL_LOW; jx < ix + STENCIL_HIGH; jx++) {
				delx_scaled = px_shifted_scaled - 1.0 * jx;
				DisneyKernelAndDerivative(icellsize, delx_scaled, wfx, wfdx);
				if (wfx > 0.0) {

					for (jy = iy - STENCIL_LOW; jy < iy + STENCIL_HIGH; jy++) {
						dely_scaled = py_shifted_scaled - 1.0 * jy;
						DisneyKernelAndDerivative(icellsize, dely_scaled, wfy, wfdy);
						if (wfy > 0.0) {

							for (jz = iz - STENCIL_LOW; jz < iz + STENCIL_HIGH; jz++) {
								delz_scaled = pz_shifted_scaled - 1.0 * jz;
								DisneyKernelAndDerivative(icellsize, delz_scaled, wfz, wfdz);
								if (wfz > 0.0) {

									wf = wfx * wfy * wfz; // this is the total weight function -- a dyadic product of the cartesian weight functions

									g(0) = wfdx * wfy * wfz; // this is the kernel gradient
									g(1) = wfdy * wfx * wfz;
									g(2) = wfdz * wfx * wfy;

									force = scaledStress * g; // this is the force from the divergence of the stress field

									force(0) += wf * f[i][0]; // these are body force from other force fields, e.g. contact
									force(1) += wf * f[i][1];
									force(2) += wf * f[i][2];

									gridnodes[jx][jy][jz].f += force;
									gridnodes[jx][jy][jz].dheat_dt += scaled_temperature_gradient.dot(g);
								}
							}
						}
					}
				}
			}
		} // end if (setflag[itype][itype])
	}
}

/*
 * update grid velocities using grid forces
 */

void PairSmdMpm::UpdateGridVelocities() {

	int ix, iy, iz;
	double dtm;

	for (ix = 0; ix < grid_nx; ix++) {
		for (iy = 0; iy < grid_ny; iy++) {
			for (iz = 0; iz < grid_nz; iz++) {
				if (gridnodes[ix][iy][iz].mass > MASS_CUTOFF) {
					if (gridnodes[ix][iy][iz].isVelocityBC == false) {
						dtm = update->dt / gridnodes[ix][iy][iz].mass;
						gridnodes[ix][iy][iz].v += dtm * gridnodes[ix][iy][iz].f;
						gridnodes[ix][iy][iz].heat += dtm * gridnodes[ix][iy][iz].dheat_dt;
					} else {
						//gridnodes[ix][iy][iz].fx = 0.0;
						//gridnodes[ix][iy][iz].fy = 0.0;
						//gridnodes[ix][iy][iz].fz = 0.0;
					}
				}
			}
		}
	}
}

void PairSmdMpm::GridToPoints() {
	double **smd_data_9 = atom->smd_data_9;
	double **x = atom->x;
	double **v = atom->v;
	double **f = atom->f;
	double *rmass = atom->rmass;
	int *type = atom->type;
	tagint *mol = atom->molecule;
	int nlocal = atom->nlocal;
	int i, itype;
	int ix, iy, iz, jx, jy, jz;
	double px_shifted, py_shifted, pz_shifted; // shifted coords of particles
	double delx_scaled, delx_scaled_abs, dely_scaled, dely_scaled_abs, wfx, wfy, wf;
	double delz_scaled, delz_scaled_abs, wfz;
	Vector3d vel_grid, dx, vel_particle;
	Matrix3d Bp;

	for (i = 0; i < nlocal; i++) {

		particleVelocities[i].setZero();
		particleAccelerations[i].setZero();

		itype = type[i];
		if (setflag[itype][itype]) {

			px_shifted = x[i][0] - min_ix * cellsize + GRID_OFFSET * cellsize;
			py_shifted = x[i][1] - min_iy * cellsize + GRID_OFFSET * cellsize;
			pz_shifted = x[i][2] - min_iz * cellsize + GRID_OFFSET * cellsize;

			ix = icellsize * px_shifted;
			iy = icellsize * py_shifted;
			iz = icellsize * pz_shifted;

			particleVelocities[i].setZero();
			particleAccelerations[i].setZero();
			particleHeat[i] = 0.0;
			particleHeatRate[i] = 0.0;
			vel_particle << v[i][0], v[i][1], v[i][2];
			Bp.setZero();

			for (jx = ix - STENCIL_LOW; jx < ix + STENCIL_HIGH; jx++) {

				delx_scaled = px_shifted * icellsize - 1.0 * jx;
				dx(0) = delx_scaled * cellsize;
				delx_scaled_abs = fabs(delx_scaled);
				wfx = DisneyKernel(delx_scaled_abs);
				if (wfx > 0.0) {

					for (jy = iy - STENCIL_LOW; jy < iy + STENCIL_HIGH; jy++) {

						dely_scaled = py_shifted * icellsize - 1.0 * jy;
						dx(1) = dely_scaled * cellsize;
						dely_scaled_abs = fabs(dely_scaled);
						wfy = DisneyKernel(dely_scaled_abs);
						if (wfy > 0.0) {

							for (jz = iz - STENCIL_LOW; jz < iz + STENCIL_HIGH; jz++) {

								delz_scaled = pz_shifted * icellsize - 1.0 * jz;
								dx(2) = delz_scaled * cellsize;
								delz_scaled_abs = fabs(delz_scaled);
								wfz = DisneyKernel(delz_scaled_abs);
								if (wfz > 0.0) {

									wf = wfx * wfy * wfz; // this is the total weight function -- a dyadic product of the cartesian weight functions

									particleVelocities[i] += wf * gridnodes[jx][jy][jz].v;

									// NEED TO COMPUTE BP_n+1 here using the updated grid velocities
									if (gridnodes[jx][jy][jz].isVelocityBC == false) {
										if (APIC) {
											Bp += wf * gridnodes[jx][jy][jz].v * dx.transpose();
										}
									} else {
										if (APIC) {
											Bp += wf * vel_particle * dx.transpose();
										}
									}

									if (gridnodes[jx][jy][jz].mass > MASS_CUTOFF) {
										particleAccelerations[i] += wf * gridnodes[jx][jy][jz].f / gridnodes[jx][jy][jz].mass;
										particleHeatRate[i] += wf * gridnodes[jx][jy][jz].dheat_dt;
										particleHeat[i] += wf * gridnodes[jx][jy][jz].heat;
									}

									if (wf > 0.0) {
										if (mol[i] == 1000) {
											printf("check\n");
											if ((vel_grid - vel_particle).norm() > 1.0e-6) {
												cout << "mol = " << mol[i] << ",  grid BC status is "
														<< gridnodes[jx][jy][jz].isVelocityBC << endl;
												cout << "vel error " << (vel_grid - vel_particle).norm() << endl;
												cout << "grid vel is " << vel_grid.transpose() << endl;
												cout << "particle vel is " << vel_particle.transpose() << endl << endl;

											}
										}
									}

								}
							}
						}
					}
				}
			}

			/*
			 * store the APIC correction matrix in atom data structure so it moves
			 * with the particle if it is exchanged to another proc.
			 */

			if (APIC) {
				smd_data_9[i][0] = Bp(0, 0);
				smd_data_9[i][1] = Bp(0, 1);
				smd_data_9[i][2] = Bp(0, 2);

				smd_data_9[i][3] = Bp(1, 0);
				smd_data_9[i][4] = Bp(1, 1);
				smd_data_9[i][5] = Bp(1, 2);

				smd_data_9[i][6] = Bp(2, 0);
				smd_data_9[i][7] = Bp(2, 1);
				smd_data_9[i][8] = Bp(2, 2);
			}

			if (domain->dimension == 2) {
				particleAccelerations[i](2) = 0.0;
				particleVelocities[i](2) = 0.0;
			}

			f[i][0] = rmass[i] * particleAccelerations[i](0);
			f[i][1] = rmass[i] * particleAccelerations[i](1);
			f[i][2] = rmass[i] * particleAccelerations[i](2);

			if (mol[i] == 1000) {

				Vector3d oldvel;
				oldvel << v[i][0], v[i][1], v[i][2];

				if ((oldvel - particleVelocities[i]).norm() > 1.0e-8) {
					cout << " this is old vel " << oldvel.transpose() << endl;
					cout << " this is new vel " << particleVelocities[i].transpose() << endl << endl;
				}

				particleVelocities[i] << v[i][0], v[i][1], v[i][2];
				particleAccelerations[i].setZero();
			}

		} // end if (setflag[itype][itype])
	}

	Bp_exists = true;

}

/* ---------------------------------------------------------------------- */
void PairSmdMpm::UpdateDeformationGradient() {

	/*
	 * this is currently deactivated because the smd_data_9 array is used for storing
	 * the APIC correction matrix Bp
	 */
	error->one(FLERR, "should not be here");

// given the velocity gradient, update the deformation gradient
	double **smd_data_9 = atom->smd_data_9;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	int i, itype;
	Matrix3d F, Fincr, eye;
	eye.setIdentity();

// transfer particle velocities to grid nodes
	for (i = 0; i < nlocal; i++) {

		itype = type[i];
		if (setflag[itype][itype]) {

			F(0, 0) = smd_data_9[i][0];
			F(0, 1) = smd_data_9[i][1];
			F(0, 2) = smd_data_9[i][2];

			F(1, 0) = smd_data_9[i][3];
			F(1, 1) = smd_data_9[i][4];
			F(1, 2) = smd_data_9[i][5];

			F(2, 0) = smd_data_9[i][6];
			F(2, 1) = smd_data_9[i][7];
			F(2, 2) = smd_data_9[i][8];

			//cout << "this is F befor update" << endl << F << endl;

			Fincr = eye + 1.0 * update->dt * L[i];
			F = Fincr * F;

			//cout << "this is F" << endl << F << endl;

			smd_data_9[i][0] = F(0, 0);
			smd_data_9[i][1] = F(0, 1);
			smd_data_9[i][2] = F(0, 2);

			smd_data_9[i][3] = F(1, 0);
			smd_data_9[i][4] = F(1, 1);
			smd_data_9[i][5] = F(1, 2);

			smd_data_9[i][6] = F(2, 0);
			smd_data_9[i][7] = F(2, 1);
			smd_data_9[i][8] = F(2, 2);
		}

	}

}

/* ---------------------------------------------------------------------- */

void PairSmdMpm::DestroyGrid() {

	memory->destroy(gridnodes);

}

/* ---------------------------------------------------------------------- */

void PairSmdMpm::compute(int eflag, int vflag) {

	if (eflag || vflag)
		ev_setup(eflag, vflag);
	else
		evflag = vflag_fdotr = 0;

	if (atom->nmax > nmax) {
//printf("... allocating in compute with nmax = %d\n", atom->nmax);
		nmax = atom->nmax;
		delete[] c0;
		c0 = new double[nmax];
		delete[] stressTensor;
		stressTensor = new Matrix3d[nmax];
		delete[] L;
		L = new Matrix3d[nmax];
		delete[] F;
		F = new Matrix3d[nmax];
		delete[] numNeighs;
		numNeighs = new int[nmax];
		delete[] particleVelocities;
		particleVelocities = new Vector3d[nmax];
		delete[] particleAccelerations;
		particleAccelerations = new Vector3d[nmax];
		Bp_exists = false;
		delete[] heat_gradient;
		heat_gradient = new Vector3d[nmax];
		delete[] particleHeat;
		particleHeat = new double[nmax];
		delete[] particleHeatRate;
		particleHeatRate = new double[nmax];
	}

	CreateGrid();

	if (APIC) {
		comm->forward_comm_pair(this); // need to do one forward comm here to have APIC Bp on ghosts
	}
	PointsToGrid();

	ComputeVelocityGradient(); // using current velocities
	//SolveHeatEquation();

	AssembleStressTensor();

	ApplyVelocityBC();

	comm->forward_comm_pair(this);
	ComputeGridForces();
	UpdateGridVelocities();

	GridToPoints();

	DestroyGrid();

}

void PairSmdMpm::UpdateStress() {
	double **tlsph_stress = atom->smd_stress;
	double *de = atom->de;
	double *vfrac = atom->vfrac;
	int *type = atom->type;
	Matrix3d D, eye, d_dev, stressRate, oldStress, newStress;
	double d_iso;
	int i, itype;
	int nlocal = atom->nlocal;
	eye.setIdentity();

	dtCFL = BIG;
	double FACTOR = 1.0;

	for (i = 0; i < nlocal; i++) {
		itype = type[i];
		if (setflag[itype][itype] == 1) {

			oldStress(0, 0) = tlsph_stress[i][0];
			oldStress(0, 1) = tlsph_stress[i][1];
			oldStress(0, 2) = tlsph_stress[i][2];
			oldStress(1, 1) = tlsph_stress[i][3];
			oldStress(1, 2) = tlsph_stress[i][4];
			oldStress(2, 2) = tlsph_stress[i][5];
			oldStress(1, 0) = oldStress(0, 1);
			oldStress(2, 0) = oldStress(0, 2);
			oldStress(2, 1) = oldStress(1, 2);

			D = 0.5 * (L[i] + L[i].transpose());
			d_dev = Deviator(D);
			d_iso = D.trace();
			vfrac[i] += FACTOR * update->dt * vfrac[i] * d_iso; // update the volume

			stressRate = Lookup[BULK_MODULUS][itype] * d_iso * eye + 2.0 * Lookup[SHEAR_MODULUS][itype] * d_dev;
			newStress = oldStress + FACTOR * update->dt * stressRate;

			tlsph_stress[i][0] = newStress(0, 0);
			tlsph_stress[i][1] = newStress(0, 1);
			tlsph_stress[i][2] = newStress(0, 2);
			tlsph_stress[i][3] = newStress(1, 1);
			tlsph_stress[i][4] = newStress(1, 2);
			tlsph_stress[i][5] = newStress(2, 2);

			c0[i] = Lookup[REFERENCE_SOUNDSPEED][itype];

			/*
			 * stable timestep based on speed-of-sound
			 */

			dtCFL = MIN(cellsize / c0[i], dtCFL);

			/*
			 * potential energy
			 */

			de[i] += 0.5 * vfrac[i] * (newStress.cwiseProduct(D)).sum();
		}
	}

}

void PairSmdMpm::GetStress() {
	double **tlsph_stress = atom->smd_stress;
	int *type = atom->type;
	int i, itype;
	int nlocal = atom->nlocal;

	for (i = 0; i < nlocal; i++) {
		itype = type[i];
		if (setflag[itype][itype] == 1) {
			stressTensor[i](0, 0) = tlsph_stress[i][0];
			stressTensor[i](0, 1) = tlsph_stress[i][1];
			stressTensor[i](0, 2) = tlsph_stress[i][2];
			stressTensor[i](1, 1) = tlsph_stress[i][3];
			stressTensor[i](1, 2) = tlsph_stress[i][4];
			stressTensor[i](2, 2) = tlsph_stress[i][5];
			stressTensor[i](1, 0) = stressTensor[i](0, 1);
			stressTensor[i](2, 0) = stressTensor[i](0, 2);
			stressTensor[i](2, 1) = stressTensor[i](1, 2);
		}
	}

}

/* ----------------------------------------------------------------------
 Assemble total stress tensor with pressure, material sterength, and
 viscosity contributions.
 ------------------------------------------------------------------------- */
void PairSmdMpm::AssembleStressTensor() {
	double *vfrac = atom->vfrac;
	double *rmass = atom->rmass;
	double *eff_plastic_strain = atom->eff_plastic_strain;
	double **tlsph_stress = atom->smd_stress;
	double *e = atom->e;
	double *de = atom->de;
//double **x = atom->x;
	int *type = atom->type;
	double pFinal;
	int i, itype;
	int nlocal = atom->nlocal;
	Matrix3d E, D, Ddev, W, V, sigma_diag;
	Matrix3d eye, stressRate, StressRateDevJaumann;
	Matrix3d sigmaInitial_dev, d_dev, sigmaFinal_dev, stressRateDev, oldStressDeviator, newStressDeviator;
	double plastic_strain_increment, yieldStress;
	double dt = update->dt;
	double vol, newPressure;
	double G_eff = 0.0; // effective shear modulus
	double K_eff; // effective bulk modulus
	double M, p_wave_speed;
	double rho, effectiveViscosity, d_iso;
	Matrix3d deltaStressDev;

	dtCFL = 1.0e22;
	eye.setIdentity();

	for (i = 0; i < nlocal; i++) {
		itype = type[i];
		if (setflag[itype][itype] == 1) {
			newStressDeviator.setZero();
			newPressure = 0.0;
			stressTensor[i].setZero();

			effectiveViscosity = 0.0;
			K_eff = 0.0;
			G_eff = 0.0;
			D = 0.5 * (L[i] + L[i].transpose());
			//E = 0.5 * (F[i].transpose() * F[i] - eye);
			//E /= update->dt;

			d_iso = D.trace();
			vfrac[i] += update->dt * vfrac[i] * d_iso; // update the volume

			vol = vfrac[i];
			rho = rmass[i] / vol;

			switch (eos[itype]) {
			default:
				error->one(FLERR, "unknown EOS.");
				break;
			case NONE:
				pFinal = 0.0;
				c0[i] = 1.0;
				break;
			case EOS_TAIT:
				TaitEOS_density(Lookup[EOS_TAIT_EXPONENT][itype], Lookup[REFERENCE_SOUNDSPEED][itype],
						Lookup[REFERENCE_DENSITY][itype], rho, newPressure, c0[i]);
				//printf("new pressure =%f\n", newPressure);

				break;
			case EOS_PERFECT_GAS:
				PerfectGasEOS(Lookup[EOS_PERFECT_GAS_GAMMA][itype], vol, rmass[i], e[i], newPressure, c0[i]);
				break;
			case EOS_LINEAR:
				newPressure = Lookup[BULK_MODULUS][itype] * (rho / Lookup[REFERENCE_DENSITY][itype] - 1.0);
				//printf("p=%f, rho0=%f, rho=%f\n", newPressure, Lookup[REFERENCE_DENSITY][itype], rho);
				c0[i] = Lookup[REFERENCE_SOUNDSPEED][itype];
				break;
			}

			K_eff = c0[i] * c0[i] * rho; // effective bulk modulus

			/*
			 * ******************************* STRENGTH MODELS ************************************************
			 */

			if (strength[itype] != NONE) {
				/*
				 * initial stress state: given by the unrotateted Cauchy stress.
				 * Assemble Eigen 3d matrix from stored stress state
				 */
				oldStressDeviator(0, 0) = tlsph_stress[i][0];
				oldStressDeviator(0, 1) = tlsph_stress[i][1];
				oldStressDeviator(0, 2) = tlsph_stress[i][2];
				oldStressDeviator(1, 1) = tlsph_stress[i][3];
				oldStressDeviator(1, 2) = tlsph_stress[i][4];
				oldStressDeviator(2, 2) = tlsph_stress[i][5];
				oldStressDeviator(1, 0) = oldStressDeviator(0, 1);
				oldStressDeviator(2, 0) = oldStressDeviator(0, 2);
				oldStressDeviator(2, 1) = oldStressDeviator(1, 2);

				W = 0.5 * (L[i] - L[i].transpose()); // spin tensor:: need this for Jaumann rate
				d_dev = Deviator(D);

				switch (strength[itype]) {
				default:
					error->one(FLERR, "unknown strength model.");
					break;
				case STRENGTH_LINEAR:

					// here in a version with pressure part
//					stressRateDev = Lookup[BULK_MODULUS][itype] * d_iso * eye + 2.0 * Lookup[SHEAR_MODULUS][itype] * d_dev;
//					c0[i] = Lookup[REFERENCE_SOUNDSPEED][itype];
//					newPressure = 0.0;

					// here only stress deviator
					stressRateDev = 2.0 * Lookup[SHEAR_MODULUS][itype] * d_dev;
					//cout << "stress rate deviator is " << endl << stressRateDev << endl;
					break;

				case STRENGTH_LINEAR_PLASTIC:
					yieldStress = Lookup[YIELD_STRENGTH][itype] + Lookup[HARDENING_PARAMETER][itype] * eff_plastic_strain[i];

					LinearPlasticStrength(Lookup[SHEAR_MODULUS][itype], yieldStress, oldStressDeviator, d_dev, dt,
							newStressDeviator, stressRateDev, plastic_strain_increment);

//					if (x[i][0] > -65.0) {
//						newStressDeviator.setZero();
//						stressRateDev.setZero();
//						plastic_strain_increment = 0.0;
//					}

					eff_plastic_strain[i] += plastic_strain_increment;

					break;
				}

				StressRateDevJaumann = stressRateDev; // - W * oldStressDeviator + oldStressDeviator * W;
				newStressDeviator = oldStressDeviator + dt * StressRateDevJaumann;

				tlsph_stress[i][0] = newStressDeviator(0, 0);
				tlsph_stress[i][1] = newStressDeviator(0, 1);
				tlsph_stress[i][2] = newStressDeviator(0, 2);
				tlsph_stress[i][3] = newStressDeviator(1, 1);
				tlsph_stress[i][4] = newStressDeviator(1, 2);
				tlsph_stress[i][5] = newStressDeviator(2, 2);

				// estimate effective shear modulus for time step stability
				deltaStressDev = oldStressDeviator - newStressDeviator;
				G_eff = effective_shear_modulus(d_dev, deltaStressDev, dt, itype);

			} // end if (strength[itype] != NONE)

			if (viscosity[itype] != NONE) {
				d_dev = Deviator(D);

				switch (viscosity[itype]) {
				default:
					error->one(FLERR, "unknown viscosity model.");
					break;
				case VISCOSITY_NEWTON:
					effectiveViscosity = Lookup[VISCOSITY_MU][itype];
					newStressDeviator = 2.0 * effectiveViscosity * d_dev; // newton original
					break;
				}
			} // end if (viscosity[itype] != NONE)

			/*
			 * assemble stress Tensor from pressure and deviatoric parts
			 */

			stressTensor[i] = -newPressure * eye + newStressDeviator;

			/*
			 * stable timestep based on speed-of-sound
			 */

			M = K_eff + 4.0 * G_eff / 3.0;
			p_wave_speed = sqrt(M / rho);
			dtCFL = MIN(cellsize / p_wave_speed, dtCFL);

			/*
			 * stable timestep based on viscosity
			 */
			if (viscosity[itype] != NONE) {
				dtCFL = MIN(cellsize * cellsize * rho / (effectiveViscosity), dtCFL);
			}

			/*
			 * elastic energy rate
			 */

			de[i] += vol * (stressTensor[i].cwiseProduct(D)).sum();

		}
		// end if (setflag[itype][itype] == 1)
	} // end loop over nlocal

	// fallback if no atoms are present:
	int check_flag = 0;
	for (itype = 1; itype < atom->ntypes; itype++) {
		if (setflag[itype][itype] == 1) {
			p_wave_speed = sqrt(Lookup[BULK_MODULUS][itype] / Lookup[REFERENCE_DENSITY][itype]);
			dtCFL = MIN(cellsize / p_wave_speed, dtCFL);
			check_flag = 1;
		}
	}
	if (check_flag == 0) {
		error->one(FLERR, "pair smd/mpm could not compute a valid stable time increment");
	}

//printf("stable timestep = %g\n", 0.1 * hMin * MaxBulkVelocity);
}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairSmdMpm::allocate() {

	allocated = 1;
	int n = atom->ntypes;

	memory->create(setflag, n + 1, n + 1, "pair:setflag");

	memory->create(Q1, n + 1, "pair:Q1");
	memory->create(rho0, n + 1, "pair:Q2");
	memory->create(c0_type, n + 1, "pair:c0_type");
	memory->create(heat_conduction_coeff, n + 1, "pair:heat_conduction_coeff");
	memory->create(eos, n + 1, "pair:eosmodel");
	memory->create(viscosity, n + 1, "pair:viscositymodel");
	memory->create(strength, n + 1, "pair:strengthmodel");

	memory->create(Lookup, MAX_KEY_VALUE, n + 1, "pair:LookupTable");

	memory->create(cutsq, n + 1, n + 1, "pair:cutsq"); // always needs to be allocated, even with granular neighborlist

	/*
	 * initialize arrays to default values
	 */

	for (int i = 1; i <= n; i++) {
		heat_conduction_coeff[i] = 0.0;
		for (int j = i; j <= n; j++) {
			setflag[i][j] = 0;
		}
	}

}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairSmdMpm::settings(int narg, char **arg) {
	if (narg < 1) {
		printf("narg = %d\n", narg);
		error->all(FLERR, "Illegal number of arguments for pair_style mpm");
	}

	cellsize = force->numeric(FLERR, arg[0]);

	if (comm->me == 0) {
		printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n");
		printf("... SMD / MPM PROPERTIES\n\n");
		printf("... cell size is %f \n", cellsize);
	}

	int iarg = 0;
	while (true) {

		iarg++;

		if (iarg >= narg) {
			break;
		}

		if (strcmp(arg[iarg], "APIC") == 0) {
			APIC = true;
			if (comm->me == 0) {
				printf("... will use APIC corrected velocities to conserve momentum\n");
			}
		} else {
			char msg[128];
			sprintf(msg, "Illegal keyword for pair smd/mpm: %s\n", arg[iarg]);
			error->all(FLERR, msg);
		}

	}

// error check
//if ((gradient_correction_flag == true) && (density_summation)) {
//	error->all(FLERR, "Cannot use *DENSITY_SUMMATION in combination with *YES_GRADIENT_CORRECTION");
//}

	if (comm->me == 0)
		printf(">>========>>========>>========>>========>>========>>========>>========>>========\n");

}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairSmdMpm::coeff(int narg, char **arg) {
	int ioffset, iarg, iNextKwd, itype, jtype;
	char str[128];
	std::string s, t;

	if (narg < 3) {
		sprintf(str, "number of arguments for pair mpm is too small!");
		error->all(FLERR, str);
	}
	if (!allocated)
		allocate();

	/*
	 * if parameters are give in i,i form, i.e., no a cross interaction, set material parameters
	 */

	if (force->inumeric(FLERR, arg[0]) == force->inumeric(FLERR, arg[1])) {

		itype = force->inumeric(FLERR, arg[0]);
		eos[itype] = viscosity[itype] = strength[itype] = NONE;

		if (comm->me == 0) {
			printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n");
			printf("...SMD / MPM PROPERTIES OF PARTICLE TYPE %d\n\n", itype);
		}

		/*
		 * read parameters which are common -- regardless of material / eos model
		 */

		ioffset = 2;
		if (strcmp(arg[ioffset], "*COMMON") != 0) {
			sprintf(str, "common keyword missing!");
			error->all(FLERR, str);
		} else {
		}

		t = string("*");
		iNextKwd = -1;
		for (iarg = ioffset + 1; iarg < narg; iarg++) {
			s = string(arg[iarg]);
			if (s.compare(0, t.length(), t) == 0) {
				iNextKwd = iarg;
				break;
			}
		}

		//printf("keyword following *COMMON is %s\n", arg[iNextKwd]);

		if (iNextKwd < 0) {
			sprintf(str, "no *KEYWORD terminates *COMMON");
			error->all(FLERR, str);
		}

		if (iNextKwd - ioffset != 5 + 1) {
			sprintf(str, "expected 5 arguments following *COMMON but got %d\n", iNextKwd - ioffset - 1);
			error->all(FLERR, str);
		}

		Lookup[REFERENCE_DENSITY][itype] = force->numeric(FLERR, arg[ioffset + 1]);
		Lookup[REFERENCE_SOUNDSPEED][itype] = force->numeric(FLERR, arg[ioffset + 2]);
		Q1[itype] = force->numeric(FLERR, arg[ioffset + 3]);
		Lookup[HEAT_CAPACITY][itype] = force->numeric(FLERR, arg[ioffset + 4]);
		Lookup[HOURGLASS_CONTROL_AMPLITUDE][itype] = force->numeric(FLERR, arg[ioffset + 5]);

		Lookup[BULK_MODULUS][itype] = Lookup[REFERENCE_SOUNDSPEED][itype] * Lookup[REFERENCE_SOUNDSPEED][itype]
				* Lookup[REFERENCE_DENSITY][itype];

		if (comm->me == 0) {
			printf("material unspecific properties for SMD/MPM definition of particle type %d:\n", itype);
			printf(FORMAT1, "reference density", Lookup[REFERENCE_DENSITY][itype]);
			printf(FORMAT1, "reference speed of sound", Lookup[REFERENCE_SOUNDSPEED][itype]);
			printf(FORMAT1, "linear viscosity coefficient", Q1[itype]);
			printf(FORMAT1, "heat capacity [energy / (mass * temperature)]", Lookup[HEAT_CAPACITY][itype]);
			printf(FORMAT1, "bulk modulus", Lookup[BULK_MODULUS][itype]);
			printf(FORMAT1, "hourglass control amplitude", Lookup[HOURGLASS_CONTROL_AMPLITUDE][itype]);
		}

		/*
		 * read following material cards
		 */

//		if (comm->me == 0) {
//			printf("next kwd is %s\n", arg[iNextKwd]);
//		}
		while (true) {
			if (strcmp(arg[iNextKwd], "*END") == 0) {
//				if (comm->me == 0) {
//					sprintf(str, "found *END");
//					error->message(FLERR, str);
//				}
				break;
			}

			ioffset = iNextKwd;
			if (strcmp(arg[ioffset], "*EOS_TAIT") == 0) {

				/*
				 * Tait EOS
				 */

				eos[itype] = EOS_TAIT;
				//printf("reading *EOS_TAIT\n");

				t = string("*");
				iNextKwd = -1;
				for (iarg = ioffset + 1; iarg < narg; iarg++) {
					s = string(arg[iarg]);
					if (s.compare(0, t.length(), t) == 0) {
						iNextKwd = iarg;
						break;
					}
				}

				if (iNextKwd < 0) {
					sprintf(str, "no *KEYWORD terminates *EOS_TAIT");
					error->all(FLERR, str);
				}

				if (iNextKwd - ioffset != 1 + 1) {
					sprintf(str, "expected 1 arguments following *EOS_TAIT but got %d\n", iNextKwd - ioffset - 1);
					error->all(FLERR, str);
				}

				Lookup[EOS_TAIT_EXPONENT][itype] = force->numeric(FLERR, arg[ioffset + 1]);

				if (comm->me == 0) {
					printf(FORMAT2, "Tait EOS");
					printf(FORMAT1, "Exponent", Lookup[EOS_TAIT_EXPONENT][itype]);
				}
			} // end Tait EOS

			else if (strcmp(arg[ioffset], "*EOS_PERFECT_GAS") == 0) {

				/*
				 * Perfect Gas EOS
				 */

				eos[itype] = EOS_PERFECT_GAS;
				//printf("reading *EOS_PERFECT_GAS\n");

				t = string("*");
				iNextKwd = -1;
				for (iarg = ioffset + 1; iarg < narg; iarg++) {
					s = string(arg[iarg]);
					if (s.compare(0, t.length(), t) == 0) {
						iNextKwd = iarg;
						break;
					}
				}

				if (iNextKwd < 0) {
					sprintf(str, "no *KEYWORD terminates *EOS_PERFECT_GAS");
					error->all(FLERR, str);
				}

				if (iNextKwd - ioffset != 1 + 1) {
					sprintf(str, "expected 1 arguments following *EOS_PERFECT_GAS but got %d\n", iNextKwd - ioffset - 1);
					error->all(FLERR, str);
				}

				Lookup[EOS_PERFECT_GAS_GAMMA][itype] = force->numeric(FLERR, arg[ioffset + 1]);

				if (comm->me == 0) {
					printf(FORMAT2, "Perfect Gas EOS");
					printf(FORMAT1, "Heat Capacity Ratio Gamma", Lookup[EOS_PERFECT_GAS_GAMMA][itype]);
				}
			} // end Perfect Gas EOS
			else if (strcmp(arg[ioffset], "*EOS_LINEAR") == 0) {

				/*
				 * Linear EOS
				 */

				eos[itype] = EOS_LINEAR;

				t = string("*");
				iNextKwd = -1;
				for (iarg = ioffset + 1; iarg < narg; iarg++) {
					s = string(arg[iarg]);
					if (s.compare(0, t.length(), t) == 0) {
						iNextKwd = iarg;
						break;
					}
				}

				if (iNextKwd < 0) {
					sprintf(str, "no *KEYWORD terminates *EOS_LINEAR");
					error->all(FLERR, str);
				}

				if (iNextKwd - ioffset != 0 + 1) {
					sprintf(str, "expected 0 arguments following *EOS_LINEAR but got %d\n", iNextKwd - ioffset - 1);
					error->all(FLERR, str);
				}

				if (comm->me == 0) {
					printf(FORMAT2, "Linear EOS");
					printf(FORMAT1, "Bulk modulus", Lookup[BULK_MODULUS][itype]);
				}
			} // end Linear EOS
			else if (strcmp(arg[ioffset], "*STRENGTH_LINEAR_PLASTIC") == 0) {

				/*
				 * linear elastic / ideal plastic material model with strength
				 */

				strength[itype] = STRENGTH_LINEAR_PLASTIC;

				t = string("*");
				iNextKwd = -1;
				for (iarg = ioffset + 1; iarg < narg; iarg++) {
					s = string(arg[iarg]);
					if (s.compare(0, t.length(), t) == 0) {
						iNextKwd = iarg;
						break;
					}
				}

				if (iNextKwd < 0) {
					sprintf(str, "no *KEYWORD terminates *STRENGTH_LINEAR_PLASTIC");
					error->all(FLERR, str);
				}

				if (iNextKwd - ioffset != 3 + 1) {
					sprintf(str, "expected 3 arguments following *STRENGTH_LINEAR_PLASTIC but got %d\n", iNextKwd - ioffset - 1);
					error->all(FLERR, str);
				}

				Lookup[SHEAR_MODULUS][itype] = force->numeric(FLERR, arg[ioffset + 1]);
				Lookup[YIELD_STRENGTH][itype] = force->numeric(FLERR, arg[ioffset + 2]);
				Lookup[HARDENING_PARAMETER][itype] = force->numeric(FLERR, arg[ioffset + 3]);

				if (comm->me == 0) {
					printf(FORMAT2, "linear elastic / ideal plastic material mode");
					printf(FORMAT1, "yield_strength", Lookup[YIELD_STRENGTH][itype]);
					printf(FORMAT1, "constant hardening parameter", Lookup[HARDENING_PARAMETER][itype]);
					printf(FORMAT1, "shear modulus", Lookup[SHEAR_MODULUS][itype]);
				}
			} // end *STRENGTH_LINEAR_PLASTIC
			else if (strcmp(arg[ioffset], "*STRENGTH_LINEAR") == 0) {

				/*
				 * linear elastic / ideal plastic material model with strength
				 */

				strength[itype] = STRENGTH_LINEAR;
				t = string("*");
				iNextKwd = -1;
				for (iarg = ioffset + 1; iarg < narg; iarg++) {
					s = string(arg[iarg]);
					if (s.compare(0, t.length(), t) == 0) {
						iNextKwd = iarg;
						break;
					}
				}

				if (iNextKwd < 0) {
					sprintf(str, "no *KEYWORD terminates *STRENGTH_LINEAR");
					error->all(FLERR, str);
				}

				if (iNextKwd - ioffset != 1 + 1) {
					sprintf(str, "expected 1 arguments following *STRENGTH_LINEAR but got %d\n", iNextKwd - ioffset - 1);
					error->all(FLERR, str);
				}

				Lookup[SHEAR_MODULUS][itype] = force->numeric(FLERR, arg[ioffset + 1]);

				if (comm->me == 0) {
					printf(FORMAT2, "linear elastic strength model");
					printf(FORMAT1, "shear modulus", Lookup[SHEAR_MODULUS][itype]);
				}
			} // end *STRENGTH_LINEAR
			else if (strcmp(arg[ioffset], "*VISCOSITY_NEWTON") == 0) {

				/*
				 * linear elastic / ideal plastic material model with strength
				 */

				viscosity[itype] = VISCOSITY_NEWTON;
				t = string("*");
				iNextKwd = -1;
				for (iarg = ioffset + 1; iarg < narg; iarg++) {
					s = string(arg[iarg]);
					if (s.compare(0, t.length(), t) == 0) {
						iNextKwd = iarg;
						break;
					}
				}

				if (iNextKwd < 0) {
					sprintf(str, "no *KEYWORD terminates *VISCOSITY_NEWTON");
					error->all(FLERR, str);
				}

				if (iNextKwd - ioffset != 1 + 1) {
					sprintf(str, "expected 1 arguments following *VISCOSITY_NEWTON but got %d\n", iNextKwd - ioffset - 1);
					error->all(FLERR, str);
				}

				Lookup[VISCOSITY_MU][itype] = force->numeric(FLERR, arg[ioffset + 1]);

				if (comm->me == 0) {
					printf(FORMAT2, "Newton viscosity model");
					printf(FORMAT1, "viscosity mu", Lookup[VISCOSITY_MU][itype]);
				}
			} // end *STRENGTH_VISCOSITY_NEWTON
			else if (strcmp(arg[ioffset], "*HEAT_CONDUCTION") == 0) {

				t = string("*");
				iNextKwd = -1;
				for (iarg = ioffset + 1; iarg < narg; iarg++) {
					s = string(arg[iarg]);
					if (s.compare(0, t.length(), t) == 0) {
						iNextKwd = iarg;
						break;
					}
				}

				if (iNextKwd < 0) {
					sprintf(str, "no *KEYWORD terminates *HEAT_CONDUCTION");
					error->all(FLERR, str);
				}

				if (iNextKwd - ioffset != 1 + 1) {
					sprintf(str, "expected 1 arguments following *HEAT_CONDUCTION but got %d\n", iNextKwd - ioffset - 1);
					error->all(FLERR, str);
				}

				heat_conduction_coeff[itype] = force->numeric(FLERR, arg[ioffset + 1]);

				if (comm->me == 0) {
					printf(FORMAT2, "Fourier type heat conduction");
					printf(FORMAT1, "kappa", heat_conduction_coeff[itype]);
				}
			} // end *HEAT_CONDUCTION

			else {
				sprintf(str, "unknown *KEYWORD: %s", arg[ioffset]);
				error->all(FLERR, str);
			}

		}

		/*
		 * copy data which is looked up in inner pairwise loops from slow maps to fast arrays
		 */

		rho0[itype] = Lookup[REFERENCE_DENSITY][itype];
		c0_type[itype] = Lookup[REFERENCE_SOUNDSPEED][itype];
		setflag[itype][itype] = 1;

		/*
		 * error checks
		 */

		if ((viscosity[itype] != NONE) && (strength[itype] != NONE)) {
			sprintf(str, "cannot have both a strength and viscosity model for particle type %d", itype);
			error->all(FLERR, str);
		}

		if (eos[itype] == NONE) {
			sprintf(str, "must specify an EOS for particle type %d", itype);
			error->all(FLERR, str);
		}

	} else {
		/*
		 * we are reading a cross-interaction line for particle types i, j
		 */

		itype = force->inumeric(FLERR, arg[0]);
		jtype = force->inumeric(FLERR, arg[1]);

		if (strcmp(arg[2], "*CROSS") != 0) {
			sprintf(str, "mpm cross interaction between particle type %d and %d requested, however, *CROSS keyword is missing",
					itype, jtype);
			error->all(FLERR, str);
		}

		if (setflag[itype][itype] != 1) {
			sprintf(str,
					"mpm cross interaction between particle type %d and %d requested, however, properties of type %d  have not yet been specified",
					itype, jtype, itype);
			error->all(FLERR, str);
		}

		if (setflag[jtype][jtype] != 1) {
			sprintf(str,
					"mpm cross interaction between particle type %d and %d requested, however, properties of type %d  have not yet been specified",
					itype, jtype, jtype);
			error->all(FLERR, str);
		}

		setflag[itype][jtype] = 1;
		setflag[jtype][itype] = 1;

		if (comm->me == 0) {
			printf(">>========>>========>>========>>========>>========>>========>>========>>========\n");
		}

	}
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairSmdMpm::init_one(int i, int j) {

	if (!allocated)
		allocate();

	if (setflag[i][j] == 0)
		error->all(FLERR, "All pair coeffs are not set");

	return 2.0 * cellsize;
}

/* ----------------------------------------------------------------------
 init specific to this pair style
 ------------------------------------------------------------------------- */

void PairSmdMpm::init_style() {
// request a granular neighbor list
	//int irequest = neighbor->request(this);
	//neighbor->requests[irequest]->half = 0;
	//neighbor->requests[irequest]->gran = 1;
}

/* ----------------------------------------------------------------------
 neighbor callback to inform pair style of neighbor list to use
 optional granular history list
 ------------------------------------------------------------------------- */

void PairSmdMpm::init_list(int id, NeighList *ptr) {
	if (id == 0)
		list = ptr;
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based arrays
 ------------------------------------------------------------------------- */

double PairSmdMpm::memory_usage() {

//printf("in memory usage\n");

	return 11 * nmax * sizeof(double) + grid_nx * grid_ny * grid_nz * sizeof(Gridnode);

}

/* ---------------------------------------------------------------------- */

int PairSmdMpm::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc) {
	double *vfrac = atom->vfrac;
	double **smd_data_9 = atom->smd_data_9;
	double **f = atom->f;
	int i, j, m;

//printf("packing comm\n");
	m = 0;
	for (i = 0; i < n; i++) {
		j = list[i];
		buf[m++] = vfrac[j];
		buf[m++] = c0[j]; //2

		buf[m++] = stressTensor[j](0, 0); // pack symmetric stress tensor
		buf[m++] = stressTensor[j](1, 1);
		buf[m++] = stressTensor[j](2, 2);
		buf[m++] = stressTensor[j](0, 1);
		buf[m++] = stressTensor[j](0, 2);
		buf[m++] = stressTensor[j](1, 2); // 2 + 6 = 8

		buf[m++] = smd_data_9[j][0];
		buf[m++] = smd_data_9[j][1];
		buf[m++] = smd_data_9[j][2];
		buf[m++] = smd_data_9[j][3];
		buf[m++] = smd_data_9[j][4];
		buf[m++] = smd_data_9[j][5];
		buf[m++] = smd_data_9[j][6];
		buf[m++] = smd_data_9[j][7];
		buf[m++] = smd_data_9[j][8];

		buf[m++] = f[j][0];
		buf[m++] = f[j][1];
		buf[m++] = f[j][2];

	}
	return m;
}

/* ---------------------------------------------------------------------- */

void PairSmdMpm::unpack_forward_comm(int n, int first, double *buf) {
	double *vfrac = atom->vfrac;
	double **smd_data_9 = atom->smd_data_9;
	double **f = atom->f;
	int i, m, last;

	m = 0;
	last = first + n;
	for (i = first; i < last; i++) {
		vfrac[i] = buf[m++];
		c0[i] = buf[m++]; // 2

		stressTensor[i](0, 0) = buf[m++];
		stressTensor[i](1, 1) = buf[m++];
		stressTensor[i](2, 2) = buf[m++];
		stressTensor[i](0, 1) = buf[m++];
		stressTensor[i](0, 2) = buf[m++];
		stressTensor[i](1, 2) = buf[m++]; // 2 + 6 = 8
		stressTensor[i](1, 0) = stressTensor[i](0, 1);
		stressTensor[i](2, 0) = stressTensor[i](0, 2);
		stressTensor[i](2, 1) = stressTensor[i](1, 2);

		smd_data_9[i][0] = buf[m++];
		smd_data_9[i][1] = buf[m++];
		smd_data_9[i][2] = buf[m++];
		smd_data_9[i][3] = buf[m++];
		smd_data_9[i][4] = buf[m++];
		smd_data_9[i][5] = buf[m++];
		smd_data_9[i][6] = buf[m++];
		smd_data_9[i][7] = buf[m++];
		smd_data_9[i][8] = buf[m++];

		f[i][0] = buf[m++];
		f[i][1] = buf[m++];
		f[i][2] = buf[m++];

	}
}

/*
 * EXTRACT
 */

void *PairSmdMpm::extract(const char *str, int &i) {
	if (strcmp(str, "smd/ulsph/stressTensor_ptr") == 0) {
		return (void *) stressTensor;
	} else if (strcmp(str, "smd/ulsph/velocityGradient_ptr") == 0) {
		return (void *) L;
	} else if (strcmp(str, "smd/ulsph/numNeighs_ptr") == 0) {
		return (void *) numNeighs;
	} else if (strcmp(str, "smd/ulsph/dtCFL_ptr") == 0) {
		return (void *) &dtCFL;
	} else if (strcmp(str, "smd/mpm/particleVelocities_ptr") == 0) {
		return (void *) particleVelocities;
	} else if (strcmp(str, "smd/mpm/particleAccelerations_ptr") == 0) {
		return (void *) particleAccelerations;
	} else if (strcmp(str, "smd/mpm/particleHeat_ptr") == 0) {
		return (void *) particleHeat;
	} else if (strcmp(str, "smd/mpm/particleHeatRate_ptr") == 0) {
		return (void *) particleHeatRate;
	}

	return NULL;
}

/* ----------------------------------------------------------------------
 compute effective shear modulus by dividing rate of deviatoric stress with rate of shear deformation
 ------------------------------------------------------------------------- */

double PairSmdMpm::effective_shear_modulus(const Matrix3d d_dev, const Matrix3d deltaStressDev, const double dt, const int itype) {
	double G_eff; // effective shear modulus, see Pronto 2d eq. 3.4.7
	double deltaStressDevSum, shearRateSq, strain_increment;

	if (domain->dimension == 3) {
		deltaStressDevSum = deltaStressDev(0, 1) * deltaStressDev(0, 1) + deltaStressDev(0, 2) * deltaStressDev(0, 2)
				+ deltaStressDev(1, 2) * deltaStressDev(1, 2);
		shearRateSq = d_dev(0, 1) * d_dev(0, 1) + d_dev(0, 2) * d_dev(0, 2) + d_dev(1, 2) * d_dev(1, 2);
	} else {
		deltaStressDevSum = deltaStressDev(0, 1) * deltaStressDev(0, 1);
		shearRateSq = d_dev(0, 1) * d_dev(0, 1);
	}

	strain_increment = dt * dt * shearRateSq;

	if (strain_increment > 1.0e-12) {
		G_eff = 0.5 * sqrt(deltaStressDevSum / strain_increment);
	} else {
		if (strength[itype] != NONE) {
			G_eff = Lookup[SHEAR_MODULUS][itype];
		} else {
			G_eff = 0.0;
		}
	}

	return G_eff;

}

void PairSmdMpm::SolveHeatEquation() {
	int ix, iy, iz;
	double d2heat_dx2, d2heat_dy2, d2heat_dz2;
	for (ix = 1; ix < grid_nx - 1; ix++) {
		for (iy = 1; iy < grid_ny - 1; iy++) {
			for (iz = 1; iz < grid_nz - 1; iz++) {

				if (gridnodes[ix][iy][iz].mass > MASS_CUTOFF) {
					// second derivative of heat in x direction
					d2heat_dx2 = gridnodes[ix + 1][iy][iz].heat - 2.0 * gridnodes[ix][iy][iz].heat + gridnodes[ix - 1][iy][iz].heat;
					d2heat_dy2 = gridnodes[ix][iy + 1][iz].heat - 2.0 * gridnodes[ix][iy][iz].heat + gridnodes[ix][iy - 1][iz].heat;
					d2heat_dz2 = gridnodes[ix][iy][iz + 1].heat - 2.0 * gridnodes[ix][iy][iz].heat + gridnodes[ix][iy][iz - 1].heat;

					gridnodes[ix][iy][iz].dheat_dt = d2heat_dx2 + d2heat_dy2 + d2heat_dz2;

				}

			}
		}
	}

}

