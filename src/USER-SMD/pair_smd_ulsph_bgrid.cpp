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

#include "math.h"
#include "float.h"
#include "stdlib.h"
#include "string.h"
#include "pair_smd_ulsph_bgrid.h"
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

#define ARTIFICIAL_STRESS false
#define FORMAT1 "%60s : %g\n"
#define FORMAT2 "\n.............................. %s \n"
#define BIG 1.0e22;

PairULSPHBG::PairULSPHBG(LAMMPS *lmp) :
		Pair(lmp) {

	// per-type arrays
	Q1 = NULL;
	eos = viscosity = strength = NULL;
	c0_type = NULL;
	c0 = NULL;
	Lookup = NULL;
	artificial_stress = NULL;
	artificial_pressure = NULL;

	nmax = 0; // make sure no atom on this proc such that initial memory allocation is correct
	stressTensor = L = K = NULL;
	shepardWeight = NULL;
	numNeighs = NULL;
	rho = NULL;
	neighborhoodRho = NULL;

	velocity_gradient_required = false; // turn off computation of velocity gradient by default
	density_summation = velocity_gradient = false;

	comm_forward = 8; // this pair style communicates 8 doubles to ghost atoms
	updateFlag = 0;
}

/* ---------------------------------------------------------------------- */

PairULSPHBG::~PairULSPHBG() {
	if (allocated) {
		//printf("... deallocating\n");
		memory->destroy(Q1);
		memory->destroy(rho0);
		memory->destroy(eos);
		memory->destroy(viscosity);
		memory->destroy(strength);
		memory->destroy(c0_type);
		memory->destroy(Lookup);
		memory->destroy(artificial_pressure);
		memory->destroy(artificial_stress);

		delete[] onerad_dynamic;
		delete[] onerad_frozen;
		delete[] maxrad_dynamic;
		delete[] maxrad_frozen;

		delete[] K;
		delete[] shepardWeight;
		delete[] c0;
		delete[] stressTensor;
		delete[] L;
		delete[] numNeighs;
		delete[] rho;
		delete[] neighborhoodRho;

	}
}

/* ----------------------------------------------------------------------
 *
 * Re-compute mass density from scratch.
 * Only used for plain fluid SPH with no physical viscosity models.
 *
 ---------------------------------------------------------------------- */

void PairULSPHBG::PreCompute_DensitySummation() {
	double *radius = atom->radius;
	double **x = atom->x;
	double *rmass = atom->rmass;
	int *type = atom->type;
	int *ilist, *jlist, *numneigh;
	int **firstneigh;
	int nlocal = atom->nlocal;
	int inum, jnum, ii, jj, i, itype, jtype, j;
	double h, irad, hsq, rSq, wf;
	Vector3d dx, xi, xj;

	// set up neighbor list variables
	inum = list->inum;
	ilist = list->ilist;
	numneigh = list->numneigh;
	firstneigh = list->firstneigh;

	// zero accumulators
	for (i = 0; i < nlocal; i++) {
		rho[i] = 0.0;
		//shepardWeight[i] = 0.0;
	}

	/*
	 * only recompute mass density if density summation is used.
	 * otherwise, change in mass density is time-integrated
	 */
	for (i = 0; i < nlocal; i++) {
		itype = type[i];
		if (setflag[itype][itype] == 1) {
			// initialize particle density with self-contribution.
			h = 2.0 * radius[i];
			hsq = h * h;
			Poly6Kernel(hsq, h, 0.0, domain->dimension, wf);
			rho[i] = wf * rmass[i]; // / shepardWeight[i];
			//printf("SIC to rho is %f\n", rho[i]);
		}
	}

	for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];
		itype = type[i];
		jlist = firstneigh[i];
		jnum = numneigh[i];
		irad = radius[i];

		xi << x[i][0], x[i][1], x[i][2];

		for (jj = 0; jj < jnum; jj++) {
			j = jlist[jj];
			j &= NEIGHMASK;

			xj << x[j][0], x[j][1], x[j][2];
			dx = xj - xi;
			rSq = dx.squaredNorm();
			h = irad + radius[j];
			hsq = h * h;
			if (rSq < hsq) {

				jtype = type[j];
				Poly6Kernel(hsq, h, rSq, domain->dimension, wf);

				if (setflag[itype][itype] == 1) {
					rho[i] += wf * rmass[j]; // / shepardWeight[i];
				}

				if (j < nlocal) {
					if (setflag[jtype][jtype] == 1) {
						rho[j] += wf * rmass[i]; // / shepardWeight[j];
					}
				}
			} // end if check distance
		} // end loop over j
	} // end loop over i
}

/* ----------------------------------------------------------------------
 *
 * Compute shape matrix for kernel gradient correction and velocity gradient.
 * This is used if material strength or viscosity models are employed.
 *
 ---------------------------------------------------------------------- */

void PairULSPHBG::PreCompute() {
	double **atom_data9 = atom->smd_data_9;
	double *radius = atom->radius;
	double **x = atom->x;
	double **x0 = atom->x0;
	double **v = atom->vest;
	double *vfrac = atom->vfrac;
	int *type = atom->type;
	int *ilist, *jlist, *numneigh;
	int **firstneigh;
	int nlocal = atom->nlocal;
	int inum, jnum, ii, jj, i, itype, jtype, j, idim;
	double wfd, h, irad, r, rSq, wf, ivol, jvol;
	Vector3d dx, dv, g, du;
	Matrix3d Ktmp, Ltmp, Ftmp, K3di, D;
	Vector3d xi, xj, vi, vj, x0i, x0j, dx0;
	Matrix2d K2di, K2d;

	// zero accumulators
	for (i = 0; i < nlocal; i++) {
		itype = type[i];
		if (setflag[itype][itype]) {
			if (gradient_correction_flag) {
				K[i].setZero();
			} else {
				K[i].setIdentity();
			}
			L[i].setZero();
		}
	}

	// set up neighbor list variables
	inum = list->inum;
	ilist = list->ilist;
	numneigh = list->numneigh;
	firstneigh = list->firstneigh;

	for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];
		itype = type[i];
		jlist = firstneigh[i];
		jnum = numneigh[i];
		irad = radius[i];
		ivol = vfrac[i];

		// initialize Eigen data structures from LAMMPS data structures
		for (idim = 0; idim < 3; idim++) {
			x0i(idim) = x0[i][idim];
			xi(idim) = x[i][idim];
			vi(idim) = v[i][idim];
		}

		for (jj = 0; jj < jnum; jj++) {
			j = jlist[jj];
			j &= NEIGHMASK;

			for (idim = 0; idim < 3; idim++) {
				x0j(idim) = x0[j][idim];
				xj(idim) = x[j][idim];
				vj(idim) = v[j][idim];
			}

			dx = xj - xi;

			rSq = dx.squaredNorm();
			h = irad + radius[j];
			if (rSq < h * h) {

				r = sqrt(rSq);
				jtype = type[j];
				jvol = vfrac[j];

				// distance vectors in current and reference configuration, velocity difference
				dv = vj - vi;
				dx0 = x0j - x0i;

				// kernel and derivative
				spiky_kernel_and_derivative(h, r, domain->dimension, wf, wfd);
				//barbara_kernel_and_derivative(h, r, domain->dimension, wf, wfd);

				// uncorrected kernel gradient
				g = (wfd / r) * dx;

				/* build correction matrix for kernel derivatives */
				if (gradient_correction_flag) {
					Ktmp = -g * dx.transpose();
					K[i] += jvol * Ktmp;
				}

				// velocity gradient L
				Ltmp = -dv * g.transpose();
				L[i] += jvol * Ltmp;

				if (j < nlocal) {

					if (gradient_correction_flag) {
						K[j] += ivol * Ktmp;
					}

					L[j] += ivol * Ltmp;
				}
			} // end if check distance
		} // end loop over j

	} // end loop over i

	/*
	 * invert shape matrix and compute corrected quantities
	 */

	for (i = 0; i < nlocal; i++) {
		itype = type[i];
		if (setflag[itype][itype]) {
			if (gradient_correction_flag) {
				pseudo_inverse_SVD_limit_eigenvalue(K[i], 1000.0);
				L[i] *= K[i];
			} // end if (gradient_correction[itype]) {

			/*
			 * accumulate strain increments
			 * we abuse the atom array "atom_data_9" for this purpose, which was originally designed to hold the deformation gradient.
			 */
			D = update->dt * 0.5 * (L[i] + L[i].transpose());
			atom_data9[i][0] += D(0, 0); // xx
			atom_data9[i][1] += D(1, 1); // yy
			atom_data9[i][2] += D(2, 2); // zz
			atom_data9[i][3] += D(0, 1); // xy
			atom_data9[i][4] += D(0, 2); // xz
			atom_data9[i][5] += D(1, 2); // yz

		} // end if (setflag[itype][itype])
	} // end loop over i = 0 to nlocal

}

/* ---------------------------------------------------------------------- */

void PairULSPHBG::CreateGrid() {
	double **x = atom->x;
	int nlocal = atom->nlocal;
	int nall = nlocal + atom->nghost;
	int i;
	int ix, iy, iz;
	double minx, miny, minz, maxx, maxy, maxz;
	cellsize = 2.4 * 0.2; // cell size
	icellsize = 1.0 / cellsize; // inverse of cell size

	// get bounds of this processor's simulation box
	//printf("bounds min: %f %f %f\n", domain->sublo[0], domain->sublo[1], domain->sublo[2]);
	//printf("bounds max: %f %f %f\n", domain->subhi[0], domain->subhi[1], domain->subhi[2]);

	// get min / max position of all particles
	minx = miny = minz = BIG
	;
	maxx = maxy = maxz = -BIG
	;
	for (i = 0; i < nall; i++) {
		minx = MIN(minx, x[i][0]);
		maxx = MAX(maxx, x[i][0]);
		miny = MIN(miny, x[i][1]);
		maxy = MAX(maxy, x[i][1]);
		minz = MIN(minz, x[i][2]);
		maxz = MAX(maxz, x[i][2]);
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

	grid_nx = (max_ix - min_ix) + 7;
	grid_ny = (max_iy - min_iy) + 7;
	grid_nz = (max_iz - min_iz) + 7;

	// allocate grid storage
	// we need a triple of indices (i, j, k)

	memory->create(gridnodes, grid_nx, grid_ny, grid_nz, "pair:gridnodes");

	for (ix = 0; ix < grid_nx; ix++) {
		for (iy = 0; iy < grid_ny; iy++) {
			for (iz = 0; iz < grid_nz; iz++) {
				gridnodes[ix][iy][iz].mass = 0.0;
				gridnodes[ix][iy][iz].vx = 0.0;
				gridnodes[ix][iy][iz].vy = 0.0;
				gridnodes[ix][iy][iz].vz = 0.0;
			}
		}
	}

}

/* ---------------------------------------------------------------------- */

void PairULSPHBG::ScatterToGrid() {
	double **x = atom->x;
	double **v = atom->v;
	double *rmass = atom->rmass;
	int nlocal = atom->nlocal;
	int nall = nlocal + atom->nghost;
	int i;
	int ix, iy, iz, jx, jy, jz;
	double delx, dely, delz, wf;
	double px_shifted, py_shifted, pz_shifted; // shifted coords of particles

	// transfer particle velocities to grid nodes
	for (i = 0; i < nall; i++) {

		px_shifted = x[i][0] - min_ix * cellsize + 3 * cellsize;
		py_shifted = x[i][1] - min_iy * cellsize + 3 * cellsize;
		pz_shifted = x[i][2] - min_iz * cellsize + 3 * cellsize;

		ix = icellsize * px_shifted;
		iy = icellsize * py_shifted;
		iz = icellsize * pz_shifted;

		for (jx = ix - 1; jx < ix + 3; jx++) {
			//jy = iy;
			jz = iz;
			for (jy = iy - 1; jy < iy + 3; jy++) {
				//	for (jz = iz - 1; jz < iz + 3; jz++) {

				//printf("cell indices: %d %d %d\n", jx, jy, jz);

				// check that cell indices are within bounds
				if ((jx < 0) || (jx >= grid_nx)) {
					printf("x cell index %d is outside range 0 .. %d\n", jx, grid_nx);
					error->one(FLERR, "");
				}
				if ((jy < 0) || (jy >= grid_ny)) {
					printf("y cell indey %d is outside range 0 .. %d\n", jy, grid_ny);
					error->one(FLERR, "");
				}
				if ((jz < 0) || (jz >= grid_nz)) {
					printf("z cell indez %d is outside range 0 .. %d\n", jz, grid_nz);
					error->one(FLERR, "");
				}

				// scaled distance between particle and grid node
				delx = fabs(px_shifted * icellsize - jx);
				dely = fabs(py_shifted * icellsize - jy);
				delz = 0.0;
				//fabs(pz_shifted * icellsize - jz); // already scaled

				wf = DisneyKernel(delx) * DisneyKernel(dely); // * DisneyKernel(delz);

				//printf("particle x=%f, shifted px =%f, cell x=%f, rscaled=%f, wf=%f\n", x[i][0], px_shifted, jx * cellsize, r, wf);

				//printf("x=%f, pos grid x=%f, dx=%f\n", x[i][0], (jx - shift_ix) * cellsize, delx);

				gridnodes[jx][jy][jz].mass += wf * rmass[i];
				gridnodes[jx][jy][jz].vx += wf * rmass[i] * v[i][0];
				gridnodes[jx][jy][jz].vy += wf * rmass[i] * v[i][1];
				gridnodes[jx][jy][jz].vz += wf * rmass[i] * v[i][2];

				//if (fabs(wf) > 1.0e-12)
				numNeighs[i] += 1;
			}
		}
		//}
	}

	// normalize grid data
	for (ix = 0; ix < grid_nx; ix++) {
		for (iy = 0; iy < grid_ny; iy++) {
			for (iz = 0; iz < grid_nz; iz++) {
				//printf("grid node mass = %f\n", gridnodes[ix][iy][iz].mass);
				//printf("grid node y velocity = %f\n", gridnodes[ix][iy][iz].vy);
				if (gridnodes[ix][iy][iz].mass > 1.0e-8) {
					gridnodes[ix][iy][iz].vx /= gridnodes[ix][iy][iz].mass;
					gridnodes[ix][iy][iz].vy /= gridnodes[ix][iy][iz].mass;
					gridnodes[ix][iy][iz].vz /= gridnodes[ix][iy][iz].mass;
					//printf("grid node y velocity = %f\n", gridnodes[ix][iy][iz].vy);
				}
			}
		}
	}

}

/* ---------------------------------------------------------------------- */

void PairULSPHBG::GatherFromGrid() {
	double **x = atom->x;
	double **v = atom->v;
	int nlocal = atom->nlocal;
	int i;
	int ix, iy, iz, jx, jy, jz;
	double px_shifted, py_shifted, pz_shifted; // shifted coords of particles
	Vector3d g, dx, vel_grid, vel_particle, vel_diff;
	Matrix3d velocity_gradient;
	double delx, dely, delz, wfx, wfy, wfz, wfdx, wfdy, wfdz;

	// transfer particle velocities to grid nodes
	for (i = 0; i < nlocal; i++) {

		velocity_gradient.setZero();

		px_shifted = x[i][0] - min_ix * cellsize + 3 * cellsize;
		py_shifted = x[i][1] - min_iy * cellsize + 3 * cellsize;
		pz_shifted = x[i][2] - min_iz * cellsize + 3 * cellsize;

		ix = icellsize * px_shifted;
		iy = icellsize * py_shifted;
		iz = icellsize * pz_shifted;

		vel_particle << v[i][0], v[i][1], v[i][2];

		for (jx = ix - 1; jx < ix + 3; jx++) {
			jz = iz;
			for (jy = iy - 1; jy < iy + 3; jy++) {
				//for (jz = iz - 1; jz < iz + 3; jz++) {

				// check that cell indices are within bounds
				if ((jx < 0) || (jx >= grid_nx)) {
					printf("x cell index %d is outside range 0 .. %d\n", jx, grid_nx);
					error->one(FLERR, "");
				}
				if ((jy < 0) || (jy >= grid_ny)) {
					printf("y cell indey %d is outside range 0 .. %d\n", jy, grid_ny);
					error->one(FLERR, "");
				}
				if ((jz < 0) || (jz >= grid_nz)) {
					printf("z cell indez %d is outside range 0 .. %d\n", jz, grid_nz);
					error->one(FLERR, "");
				}

				// scaled distance between particle and grid node
				delx = fabs(px_shifted * icellsize - jx);
				dely = fabs(py_shifted * icellsize - jy);
				delz = fabs(pz_shifted * icellsize - jz); // already scaled

				wfx = DisneyKernel(delx);
				wfy = DisneyKernel(dely);
				wfz = DisneyKernel(delz);

				wfdx = DisneyKernelDerivative(delx) * icellsize;
				wfdy = DisneyKernelDerivative(dely) * icellsize;
				wfdz = DisneyKernelDerivative(delz);

				dx(0) = px_shifted - jx * cellsize;
				dx(1) = py_shifted - jy * cellsize;
				dx(2) = 0.0; //pz_shifted - jz * cellsize;

				g(0) = wfdx * wfy;
				g(1) = wfdy * wfx;
				g(2) = 0.0;

				if ((dx(0)) < 0.0) g(0) *= -1.0;
				if ((dx(1)) < 0.0) g(1) *= -1.0;

				vel_grid << gridnodes[jx][jy][jz].vx, gridnodes[jx][jy][jz].vy , gridnodes[jx][jy][jz].vz;



				vel_diff = vel_grid - vel_particle;
//				if (vel_diff.norm() > 1.0e-8) {
//					printf("vel grid=%f, vel particle=%f\n", vel_grid(0), vel_particle(0));
//				}

				velocity_gradient += vel_diff * g.transpose();
			}
			//}
		}

		L[i] = velocity_gradient;
	}
}

/* ---------------------------------------------------------------------- */

void PairULSPHBG::DestroyGrid() {

	memory->destroy(gridnodes);

}

/* ---------------------------------------------------------------------- */

void PairULSPHBG::compute(int eflag, int vflag) {
	double **x = atom->x;
	double **v = atom->v;
	double **f = atom->f;
	double *vfrac = atom->vfrac;
	double *de = atom->de;
	double *rmass = atom->rmass;
	double *radius = atom->radius;
	double *contact_radius = atom->contact_radius;
	double **atom_data9 = atom->smd_data_9;

	int *type = atom->type;
	int nlocal = atom->nlocal;
	int i, j, ii, jj, jnum, itype, jtype, iDim, inum;
	double r, wf, wfd, h, rSq, ivol, jvol, irho, jrho;
	double mu_ij, c_ij, rho_ij;
	double delVdotDelR, visc_magnitude, deltaE;
	int *ilist, *jlist, *numneigh;
	int **firstneigh;
	Vector3d fi, fj, dx, dv, f_stress, g;
	Vector3d xi, xj, vi, vj, f_visc, sumForces, f_stress_new;
	Vector3d gamma, dx0, du_est, du;
	double r_ref, weight, p;

	double ini_dist;
	Matrix3d S, D, V, eye;
	eye.setIdentity();
	int k;
	SelfAdjointEigenSolver<Matrix3d> es;

	if (eflag || vflag)
		ev_setup(eflag, vflag);
	else
		evflag = vflag_fdotr = 0;

	if (atom->nmax > nmax) {
//printf("... allocating in compute with nmax = %d\n", atom->nmax);
		nmax = atom->nmax;
		delete[] K;
		K = new Matrix3d[nmax];
		delete[] shepardWeight;
		shepardWeight = new double[nmax];
		delete[] c0;
		c0 = new double[nmax];
		delete[] stressTensor;
		stressTensor = new Matrix3d[nmax];
		delete[] L;
		L = new Matrix3d[nmax];
		delete[] numNeighs;
		numNeighs = new int[nmax];
		delete[] rho;
		rho = new double[nmax];
		delete[] neighborhoodRho;
		neighborhoodRho = new double[nmax];

	}

// zero accumulators
	for (i = 0; i < nlocal; i++) {
		shepardWeight[i] = 0.0;
		numNeighs[i] = 0;
		neighborhoodRho[i] = 0.0;
	}

	/*
	 * if this is the very first step, zero the array which holds the accumulated strain
	 */
	if (update->ntimestep == 0) {
		for (i = 0; i < nlocal; i++) {
			itype = type[i];
			if (setflag[itype][itype]) {
				for (j = 0; j < 9; j++) {
					atom_data9[i][j] = 0.0;
				}
			}
		}
	}

	if (density_summation) {
		//printf("dens summ\n");
		PreCompute_DensitySummation();

		for (i = 0; i < nlocal; i++) { //compute volumes from rho
			itype = type[i];
			if (setflag[itype][itype]) {
				vfrac[i] = rmass[i] / rho[i];
			}
		}

	}

	if (velocity_gradient) {
		PairULSPHBG::PreCompute(); // get velocity gradient and kernel gradient correction
	}

	CreateGrid();
	ScatterToGrid();
	GatherFromGrid();
	DestroyGrid();

	PairULSPHBG::AssembleStressTensor();

	/*
	 * QUANTITIES ABOVE HAVE ONLY BEEN CALCULATED FOR NLOCAL PARTICLES.
	 * NEED TO DO A FORWARD COMMUNICATION TO GHOST ATOMS NOW
	 */
	comm->forward_comm_pair(this);

	updateFlag = 0;

	/*
	 * iterate over pairs of particles i, j and assign forces using pre-computed pressure
	 */

// set up neighbor list variables
	inum = list->inum;
	ilist = list->ilist;
	numneigh = list->numneigh;
	firstneigh = list->firstneigh;

	for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];
		itype = type[i];
		jlist = firstneigh[i];
		jnum = numneigh[i];
		ivol = vfrac[i];
		irho = rmass[i] / ivol;

		// initialize Eigen data structures from LAMMPS data structures
		for (iDim = 0; iDim < 3; iDim++) {
			xi(iDim) = x[i][iDim];
			vi(iDim) = v[i][iDim];
		}

		for (jj = 0; jj < jnum; jj++) {
			j = jlist[jj];
			j &= NEIGHMASK;

			xj(0) = x[j][0];
			xj(1) = x[j][1];
			xj(2) = x[j][2];

			dx = xj - xi;
			rSq = dx.squaredNorm();
			h = radius[i] + radius[j];
			if (rSq < h * h) {

				// initialize Eigen data structures from LAMMPS data structures
				for (iDim = 0; iDim < 3; iDim++) {
					vj(iDim) = v[j][iDim];
				}

				r = sqrt(rSq);
				jtype = type[j];
				jvol = vfrac[j];
				jrho = rmass[j] / jvol;

				// distance vectors in current and reference configuration, velocity difference
				dv = vj - vi;

				// kernel and derivative
				spiky_kernel_and_derivative(h, r, domain->dimension, wf, wfd);
				//barbara_kernel_and_derivative(h, r, domain->dimension, wf, wfd);

				// uncorrected kernel gradient
				g = (wfd / r) * dx;

				delVdotDelR = dx.dot(dv) / (r + 0.1 * h); // project relative velocity onto unit particle distance vector [m/s]

				S = stressTensor[i] + stressTensor[j];

				if (artificial_pressure[itype][jtype] > 0.0) {
					p = S.trace();
					if (p > 0.0) { // we are in tension
						r_ref = contact_radius[i] + contact_radius[j];
						weight = Kernel_Cubic_Spline(r, h) / Kernel_Cubic_Spline(r_ref, h);
						weight = pow(weight, 4.0);
						S -= artificial_pressure[itype][jtype] * weight * p * eye;
					}
				}

				/*
				 * artificial stress to control tensile instability
				 * Only works if particles are uniformly spaced initially.
				 */
				if (artificial_stress[itype][jtype] > 0.0) {
					ini_dist = contact_radius[i] + contact_radius[j];
					weight = Kernel_Cubic_Spline(r, h) / Kernel_Cubic_Spline(ini_dist, h);
					weight = pow(weight, 4.0);

					es.compute(S);
					D = es.eigenvalues().asDiagonal();
					for (k = 0; k < 3; k++) {
						if (D(k, k) > 0.0) {
							D(k, k) -= weight * artificial_stress[itype][jtype] * D(k, k);
						}
					}
					V = es.eigenvectors();
					S = V * D * V.inverse();
				}

				// compute forces
				f_stress = -ivol * jvol * S * g; // DO NOT TOUCH SIGN

				/*
				 * artificial viscosity -- alpha is dimensionless
				 * MonaghanBalsara form of the artificial viscosity
				 */

				c_ij = 0.5 * (c0[i] + c0[j]);
				LimitDoubleMagnitude(delVdotDelR, 1.1 * c_ij);

				mu_ij = h * delVdotDelR / (r + 0.1 * h); // units: [m * m/s / m = m/s]
				rho_ij = 0.5 * (irho + jrho);
				visc_magnitude = 0.5 * (Q1[itype] + Q1[jtype]) * c_ij * mu_ij / rho_ij;
				f_visc = -rmass[i] * rmass[j] * visc_magnitude * g;

				// sum stress and viscous forces
				sumForces = f_stress + f_visc;

				// energy rate -- project velocity onto force vector
				deltaE = sumForces.dot(dv);

				// apply forces to pair of particles
				f[i][0] += sumForces(0);
				f[i][1] += sumForces(1);
				f[i][2] += sumForces(2);
				de[i] += deltaE;

				// accumulate smooth velocities
				shepardWeight[i] += jvol * wf;
				//numNeighs[i] += 1;
				neighborhoodRho[i] += wf * rmass[j];

				if (j < nlocal) {
					f[j][0] -= sumForces(0);
					f[j][1] -= sumForces(1);
					f[j][2] -= sumForces(2);
					de[j] += deltaE;

					shepardWeight[j] += wf * ivol;
					//numNeighs[j] += 1;
					neighborhoodRho[j] += wf * rmass[i];
				}

				// tally atomistic stress tensor
				if (evflag) {
					ev_tally_xyz(i, j, nlocal, 0, 0.0, 0.0, sumForces(0), sumForces(1), sumForces(2), dx(0), dx(1), dx(2));
				}
			}

		}
	}

	for (i = 0; i < nlocal; i++) {
		itype = type[i];
		if (setflag[itype][itype] == 1) {
			if (shepardWeight[i] != 0.0) {
				neighborhoodRho[i] /= shepardWeight[i];
			} else {
				neighborhoodRho[i] = 0.0;
			}
		} // end check if particle is SPH-type
	} // end loop over i = 0 to nlocal

	if (vflag_fdotr)
		virial_fdotr_compute();

}

/* ----------------------------------------------------------------------
 Assemble total stress tensor with pressure, material sterength, and
 viscosity contributions.
 ------------------------------------------------------------------------- */
void PairULSPHBG::AssembleStressTensor() {
	double *radius = atom->radius;
	double *vfrac = atom->vfrac;
	double *rmass = atom->rmass;
	double *eff_plastic_strain = atom->eff_plastic_strain;
	double **tlsph_stress = atom->smd_stress;
	double *e = atom->e;
	int *type = atom->type;
	double pFinal;
	int i, itype;
	int nlocal = atom->nlocal;
	Matrix3d D, Ddev, W, V, sigma_diag;
	Matrix3d eye, stressRate, StressRateDevJaumann;
	Matrix3d sigmaInitial_dev, d_dev, sigmaFinal_dev, stressRateDev, oldStressDeviator, newStressDeviator;
	double plastic_strain_increment, yieldStress;
	double dt = update->dt;
	double vol, newPressure;
	double G_eff = 0.0; // effective shear modulus
	double K_eff; // effective bulk modulus
	double M, p_wave_speed;
	double rho, effectiveViscosity;
	Matrix3d deltaStressDev;

	dtCFL = 1.0e22;
	eye.setIdentity();

	for (i = 0; i < nlocal; i++) {
		itype = type[i];
		if (setflag[itype][itype] == 1) {
			newStressDeviator.setZero();
			newPressure = 0.0;
			stressTensor[i].setZero();
			vol = vfrac[i];
			rho = rmass[i] / vfrac[i];
			effectiveViscosity = 0.0;
			K_eff = 0.0;
			G_eff = 0.0;
			D = 0.5 * (L[i] + L[i].transpose());

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
					eff_plastic_strain[i] += plastic_strain_increment;

					break;
				}

				StressRateDevJaumann = stressRateDev - W * oldStressDeviator + oldStressDeviator * W;
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
//					double shear_rate = 2.0
//							* sqrt(d_dev(0, 1) * d_dev(0, 1) + d_dev(0, 2) * d_dev(0, 2) + d_dev(1, 2) * d_dev(1, 2)); // 3d
					//cout << "shear rate: " << shear_rate << endl;
					//effectiveViscosity = PA6_270C(shear_rate);
					//if (effectiveViscosity > 178.062e-6) {
					//	printf("effective visc is %f\n", effectiveViscosity);
					//}
					newStressDeviator = 2.0 * effectiveViscosity * d_dev; // newton original
					//cout << "this is Ddev " << endl << d_dev << endl << endl;
					break;
				}
			} // end if (viscosity[itype] != NONE)

			/*
			 * assemble stress Tensor from pressure and deviatoric parts
			 */

			stressTensor[i] = -newPressure * eye + newStressDeviator;

			/*
			 * kernel gradient correction
			 */
			double scale = 1.0;
			if (gradient_correction_flag) {
				SelfAdjointEigenSolver<Matrix3d> es;
				es.compute(K[i]);
				scale = es.eigenvalues().maxCoeff();
				stressTensor[i] *= K[i];
			}

			/*
			 * stable timestep based on speed-of-sound
			 */

			M = scale * K_eff + 4.0 * G_eff / 3.0;
			p_wave_speed = sqrt(M / rho);
			dtCFL = MIN(2 * radius[i] / p_wave_speed, dtCFL);

			/*
			 * stable timestep based on viscosity
			 */
			if (viscosity[itype] != NONE) {
				dtCFL = MIN(4 * radius[i] * radius[i] * rho / (scale * effectiveViscosity), dtCFL);
			}

		}
		// end if (setflag[itype][itype] == 1)
	} // end loop over nlocal

//printf("stable timestep = %g\n", 0.1 * hMin * MaxBulkVelocity);
}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairULSPHBG::allocate() {

	allocated = 1;
	int n = atom->ntypes;

	memory->create(setflag, n + 1, n + 1, "pair:setflag");

	memory->create(Q1, n + 1, "pair:Q1");
	memory->create(rho0, n + 1, "pair:Q2");
	memory->create(c0_type, n + 1, "pair:c0_type");
	memory->create(eos, n + 1, "pair:eosmodel");
	memory->create(viscosity, n + 1, "pair:viscositymodel");
	memory->create(strength, n + 1, "pair:strengthmodel");

	memory->create(Lookup, MAX_KEY_VALUE, n + 1, "pair:LookupTable");

	memory->create(artificial_pressure, n + 1, n + 1, "pair:artificial_pressure");
	memory->create(artificial_stress, n + 1, n + 1, "pair:artificial_stress");
	memory->create(cutsq, n + 1, n + 1, "pair:cutsq");		// always needs to be allocated, even with granular neighborlist

	/*
	 * initialize arrays to default values
	 */

	for (int i = 1; i <= n; i++) {
		for (int j = i; j <= n; j++) {
			artificial_pressure[i][j] = 0.0;
			artificial_stress[i][j] = 0.0;
			setflag[i][j] = 0;
		}
	}

	onerad_dynamic = new double[n + 1];
	onerad_frozen = new double[n + 1];
	maxrad_dynamic = new double[n + 1];
	maxrad_frozen = new double[n + 1];

}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairULSPHBG::settings(int narg, char **arg) {
	if (narg != 3) {
		printf("narg = %d\n", narg);
		error->all(FLERR, "Illegal number of arguments for pair_style ulsph");
	}

	if (comm->me == 0) {
		printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n");
		printf("... SMD / ULSPH PROPERTIES\n\n");
	}

	if (strcmp(arg[0], "*DENSITY_SUMMATION") == 0) {
		density_summation = true;
		density_continuity = false;
		if (comm->me == 0)
			printf("... density summation active\n");
	} else if (strcmp(arg[0], "*DENSITY_CONTINUITY") == 0) {
		density_continuity = true;
		density_summation = false;
		if (comm->me == 0)
			printf("... density continuity active\n");
	} else {
		error->all(FLERR,
				"Illegal settings keyword for first keyword of pair style ulsph. Must be either *DENSITY_SUMMATION or *DENSITY_CONTINUITY");
	}

	if (strcmp(arg[1], "*VELOCITY_GRADIENT") == 0) {
		velocity_gradient = true;
		if (comm->me == 0)
			printf("... computation of velocity gradients active\n");
	} else if (strcmp(arg[1], "*NO_VELOCITY_GRADIENT") == 0) {
		velocity_gradient = false;
		if (comm->me == 0)
			printf("... computation of velocity gradients NOT active\n");
	} else {
		error->all(FLERR,
				"Illegal settings keyword for first keyword of pair style ulsph. Must be either *VELOCITY_GRADIENT or *NO_VELOCITY_GRADIENT");
	}

	if (strcmp(arg[2], "*GRADIENT_CORRECTION") == 0) {
		gradient_correction_flag = true;
		if (comm->me == 0)
			printf("... first order correction of kernel gradients is active\n");
	} else if (strcmp(arg[2], "*NO_GRADIENT_CORRECTION") == 0) {
		gradient_correction_flag = false;
		if (comm->me == 0)
			printf("... first order correction of kernel gradients is NOT active\n");
	} else {
		error->all(FLERR, "Illegal settings keyword for pair style ulsph");
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

void PairULSPHBG::coeff(int narg, char **arg) {
	int ioffset, iarg, iNextKwd, itype, jtype;
	char str[128];
	std::string s, t;

	if (narg < 3) {
		sprintf(str, "number of arguments for pair ulsph is too small!");
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
			printf("...SMD / ULSPH PROPERTIES OF PARTICLE TYPE %d\n\n", itype);
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
			printf("material unspecific properties for SMD/ULSPH definition of particle type %d:\n", itype);
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

				if (velocity_gradient != true) {
					error->all(FLERR, "A strength model was requested but *VELOCITY_GRADIENT is not set");
				}

				/*
				 * linear elastic / ideal plastic material model with strength
				 */

				strength[itype] = STRENGTH_LINEAR_PLASTIC;
				velocity_gradient_required = true;
				//printf("reading *LINEAR_PLASTIC\n");

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

				if (velocity_gradient != true) {
					error->all(FLERR, "A strength model was requested but *VELOCITY_GRADIENT is not set");
				}

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

				if (velocity_gradient != true) {
					error->all(FLERR, "A viscosity model was requested but *VELOCITY_GRADIENT is not set");
				}

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

			else if (strcmp(arg[ioffset], "*ARTIFICIAL_PRESSURE") == 0) {

				/*
				 * use Monaghan's artificial pressure to prevent particle clumping
				 */

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
					sprintf(str, "no *KEYWORD terminates *ARTIFICIAL_PRESSURE");
					error->all(FLERR, str);
				}

				if (iNextKwd - ioffset != 1 + 1) {
					sprintf(str, "expected 1 arguments following *ARTIFICIAL_PRESSURE but got %d\n", iNextKwd - ioffset - 1);
					error->all(FLERR, str);
				}

				artificial_pressure[itype][itype] = force->numeric(FLERR, arg[ioffset + 1]);

				if (comm->me == 0) {
					printf(FORMAT2, "Artificial Pressure is enabled.");
					printf(FORMAT1, "Artificial Pressure amplitude", artificial_pressure[itype][itype]);
				}
			} // end *ARTIFICIAL_PRESSURE

			else if (strcmp(arg[ioffset], "*ARTIFICIAL_STRESS") == 0) {

				/*
				 * use Monaghan's artificial stress to prevent particle clumping
				 */

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
					sprintf(str, "no *KEYWORD terminates *ARTIFICIAL_STRESS");
					error->all(FLERR, str);
				}

				if (iNextKwd - ioffset != 1 + 1) {
					sprintf(str, "expected 1 arguments following *ARTIFICIAL_STRESS but got %d\n", iNextKwd - ioffset - 1);
					error->all(FLERR, str);
				}

				artificial_stress[itype][itype] = force->numeric(FLERR, arg[ioffset + 1]);

				if (comm->me == 0) {
					printf(FORMAT2, "Artificial Stress is enabled.");
					printf(FORMAT1, "Artificial Stress amplitude", artificial_stress[itype][itype]);
				}
			} // end *ARTIFICIAL_STRESS

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
			sprintf(str, "ulsph cross interaction between particle type %d and %d requested, however, *CROSS keyword is missing",
					itype, jtype);
			error->all(FLERR, str);
		}

		if (setflag[itype][itype] != 1) {
			sprintf(str,
					"ulsph cross interaction between particle type %d and %d requested, however, properties of type %d  have not yet been specified",
					itype, jtype, itype);
			error->all(FLERR, str);
		}

		if (setflag[jtype][jtype] != 1) {
			sprintf(str,
					"ulsph cross interaction between particle type %d and %d requested, however, properties of type %d  have not yet been specified",
					itype, jtype, jtype);
			error->all(FLERR, str);
		}

		setflag[itype][jtype] = 1;
		setflag[jtype][itype] = 1;

		if ((artificial_pressure[itype][itype] > 0.0) && (artificial_pressure[jtype][jtype] > 0.0)) {
			artificial_pressure[itype][jtype] = 0.5 * (artificial_pressure[itype][itype] + artificial_pressure[jtype][jtype]);
			artificial_pressure[jtype][itype] = artificial_pressure[itype][jtype];
		} else {
			artificial_pressure[itype][jtype] = artificial_pressure[jtype][itype] = 0.0;
		}

		if ((artificial_stress[itype][itype] > 0.0) && (artificial_stress[jtype][jtype] > 0.0)) {
			artificial_stress[itype][jtype] = 0.5 * (artificial_stress[itype][itype] + artificial_stress[jtype][jtype]);
			artificial_stress[jtype][itype] = artificial_stress[itype][jtype];
		} else {
			artificial_stress[itype][jtype] = artificial_stress[jtype][itype] = 0.0;
		}

		if (comm->me == 0) {
			printf(">>========>>========>>========>>========>>========>>========>>========>>========\n");
		}

	}
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairULSPHBG::init_one(int i, int j) {

	if (!allocated)
		allocate();

	if (setflag[i][j] == 0)
		error->all(FLERR, "All pair coeffs are not set");

// cutoff = sum of max I,J radii for
// dynamic/dynamic & dynamic/frozen interactions, but not frozen/frozen

	double cutoff = maxrad_dynamic[i] + maxrad_dynamic[j];
	cutoff = MAX(cutoff, maxrad_frozen[i] + maxrad_dynamic[j]);
	cutoff = MAX(cutoff, maxrad_dynamic[i] + maxrad_frozen[j]);
//printf("cutoff for pair sph/fluid = %f\n", cutoff);
	return cutoff;
}

/* ----------------------------------------------------------------------
 init specific to this pair style
 ------------------------------------------------------------------------- */

void PairULSPHBG::init_style() {
	int i;

//printf(" in init style\n");
// request a granular neighbor list
	int irequest = neighbor->request(this);
	neighbor->requests[irequest]->half = 0;
	neighbor->requests[irequest]->gran = 1;

// set maxrad_dynamic and maxrad_frozen for each type
// include future Fix pour particles as dynamic

	for (i = 1; i <= atom->ntypes; i++)
		onerad_dynamic[i] = onerad_frozen[i] = 0.0;

	double *radius = atom->radius;
	int *type = atom->type;
	int nlocal = atom->nlocal;

	for (i = 0; i < nlocal; i++)
		onerad_dynamic[type[i]] = MAX(onerad_dynamic[type[i]], radius[i]);

	MPI_Allreduce(&onerad_dynamic[1], &maxrad_dynamic[1], atom->ntypes, MPI_DOUBLE, MPI_MAX, world);
	MPI_Allreduce(&onerad_frozen[1], &maxrad_frozen[1], atom->ntypes, MPI_DOUBLE, MPI_MAX, world);

}

/* ----------------------------------------------------------------------
 neighbor callback to inform pair style of neighbor list to use
 optional granular history list
 ------------------------------------------------------------------------- */

void PairULSPHBG::init_list(int id, NeighList *ptr) {
	if (id == 0)
		list = ptr;
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based arrays
 ------------------------------------------------------------------------- */

double PairULSPHBG::memory_usage() {

//printf("in memory usage\n");

	return 11 * nmax * sizeof(double);

}

/* ---------------------------------------------------------------------- */

int PairULSPHBG::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc) {
	double *vfrac = atom->vfrac;
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
	}
	return m;
}

/* ---------------------------------------------------------------------- */

void PairULSPHBG::unpack_forward_comm(int n, int first, double *buf) {
	double *vfrac = atom->vfrac;
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
	}
}

/*
 * EXTRACT
 */

void *PairULSPHBG::extract(const char *str, int &i) {
	if (strcmp(str, "smd/ulsph/stressTensor_ptr") == 0) {
		return (void *) stressTensor;
	} else if (strcmp(str, "smd/ulsph/velocityGradient_ptr") == 0) {
		return (void *) L;
	} else if (strcmp(str, "smd/ulsph/numNeighs_ptr") == 0) {
		return (void *) numNeighs;
	} else if (strcmp(str, "smd/ulsph/dtCFL_ptr") == 0) {
		return (void *) &dtCFL;
	} else if (strcmp(str, "smd/ulsph/updateFlag_ptr") == 0) {
		return (void *) &updateFlag;
	} else if (strcmp(str, "smd/ulsph/neighborhoodRho_ptr") == 0) {
		return (void *) neighborhoodRho;
	} else if (strcmp(str, "smd/ulsph/shape_matrix_ptr") == 0) {
		return (void *) K;
	}

	return NULL;
}

/* ----------------------------------------------------------------------
 compute effective shear modulus by dividing rate of deviatoric stress with rate of shear deformation
 ------------------------------------------------------------------------- */

double PairULSPHBG::effective_shear_modulus(const Matrix3d d_dev, const Matrix3d deltaStressDev, const double dt, const int itype) {
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

