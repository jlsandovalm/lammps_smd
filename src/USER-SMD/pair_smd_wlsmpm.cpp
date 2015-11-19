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

#include "pair_smd_wlsmpm.h"

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
#define MIN_MATRIX_DETERMINANT 1.0e-8
#define MASS_CUTOFF 1.0e-8
#define STENCIL_LOW 1
#define STENCIL_HIGH 3
#define GRID_OFFSET 4

PairSmdWlsMpm::PairSmdWlsMpm(LAMMPS *lmp) :
		Pair(lmp) {

	// per-type arrays
	Q1 = NULL;
	eos = viscosity = strength = NULL;
	c0_type = NULL;
	c0 = NULL;
	Lookup = NULL;

	nmax = 0; // make sure no atom on this proc such that initial memory allocation is correct
	stressTensor = L = F = NULL;
	M = NULL;
	numNeighs = NULL;
	particleVelocities = particleAccelerations = NULL;

	comm_forward = 20; // this pair style communicates 8 doubles to ghost atoms
}

/* ---------------------------------------------------------------------- */

PairSmdWlsMpm::~PairSmdWlsMpm() {
	if (allocated) {
		//printf("... deallocating\n");
		memory->destroy(Q1);
		memory->destroy(rho0);
		memory->destroy(eos);
		memory->destroy(viscosity);
		memory->destroy(strength);
		memory->destroy(c0_type);
		memory->destroy(Lookup);

		delete[] c0;
		delete[] stressTensor;
		delete[] L;
		delete[] F;
		delete[] M;
		delete[] numNeighs;
		delete[] particleVelocities;
		delete[] particleAccelerations;

	}
}

/* ---------------------------------------------------------------------- */

void PairSmdWlsMpm::CreateGrid() {
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
	minx = miny = minz = BIG
	;
	maxx = maxy = maxz = -BIG
	;

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

	grid_nx = (max_ix - min_ix) + 10;
	grid_ny = (max_iy - min_iy) + 10;
	grid_nz = (max_iz - min_iz) + 10;

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

void PairSmdWlsMpm::PointsToGrid() {
	double **x = atom->x;
	double **v = atom->v;
	double **vest = atom->vest;
	double *rmass = atom->rmass;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	int nall = nlocal + atom->nghost;
	int i, itype;
	int ix, iy, iz, jx, jy, jz;
	double delx_scaled, delx_scaled_abs, dely_scaled, dely_scaled_abs, wf, wfx, wfy;
	double delz_scaled, delz_scaled_abs, wfz;
	double px_shifted, py_shifted, pz_shifted; // shifted coords of particles
	Vector3d vel_particle, vel_particle_est;
	Vector4d l;
	Matrix4d Minv;

	// clear the grid
	for (ix = 0; ix < grid_nx; ix++) {
		for (iy = 0; iy < grid_ny; iy++) {
			for (iz = 0; iz < grid_nz; iz++) {
				gridnodes[ix][iy][iz].mass = 0.0;
				gridnodes[ix][iy][iz].fx = 0.0;
				gridnodes[ix][iy][iz].fy = 0.0;
				gridnodes[ix][iy][iz].fz = 0.0;
				gridnodes[ix][iy][iz].isVelocityBC = false;
				gridnodes[ix][iy][iz].M.setZero();
				gridnodes[ix][iy][iz].l_vx.setZero();
				gridnodes[ix][iy][iz].l_vy.setZero();
				gridnodes[ix][iy][iz].l_vz.setZero();
				gridnodes[ix][iy][iz].l_vestx.setZero();
				gridnodes[ix][iy][iz].l_vesty.setZero();
				gridnodes[ix][iy][iz].l_vestz.setZero();
				gridnodes[ix][iy][iz].isAccurate = false;
			}
		}
	}

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
			if (update->ntimestep == 0) {
				vel_particle_est = vel_particle;
			} else {
				vel_particle_est << vest[i][0], vest[i][1], vest[i][2];
			}

			for (jx = ix - STENCIL_LOW; jx < ix + STENCIL_HIGH; jx++) {

				// check that cell indices are within bounds
				if ((jx < 0) || (jx >= grid_nx)) {
					printf("x cell index %d is outside range 0 .. %d\n", jx, grid_nx);
					error->one(FLERR, "");
				}

				delx_scaled = px_shifted * icellsize - 1.0 * jx;
				delx_scaled_abs = fabs(delx_scaled);
				wfx = DisneyKernel(delx_scaled_abs);
				if (wfx > 0.0) {

					for (jy = iy - STENCIL_LOW; jy < iy + STENCIL_HIGH; jy++) {

						if ((jy < 0) || (jy >= grid_ny)) {
							printf("y cell indey %d is outside range 0 .. %d\n", jy, grid_ny);
							error->one(FLERR, "");
						}

						dely_scaled = py_shifted * icellsize - 1.0 * jy;
						dely_scaled_abs = fabs(dely_scaled);
						wfy = DisneyKernel(dely_scaled_abs);
						if (wfy > 0.0) {

							for (jz = iz - STENCIL_LOW; jz < iz + STENCIL_HIGH; jz++) {

								if ((jz < 0) || (jz >= grid_nz)) {
									printf("z cell index %d is outside range 0 .. %d\n", jz, grid_nz);
									error->one(FLERR, "");
								}

								delz_scaled = pz_shifted * icellsize - 1.0 * jz;
								delz_scaled_abs = fabs(delz_scaled);
								wfz = DisneyKernel(delz_scaled_abs);
								if (wfz > 0.0) {

									wf = wfx * wfy * wfz; // this is the total weight function -- a dyadic product of the cartesian weight functions

									// build moment matrix on grid nodes
									l << 1.0, delx_scaled, dely_scaled, delz_scaled;

									gridnodes[jx][jy][jz].l_vx += wf * vel_particle(0) * l;
									gridnodes[jx][jy][jz].l_vy += wf * vel_particle(1) * l;
									gridnodes[jx][jy][jz].l_vz += wf * vel_particle(2) * l;

									gridnodes[jx][jy][jz].l_vestx += wf * vel_particle_est(0) * l;
									gridnodes[jx][jy][jz].l_vesty += wf * vel_particle_est(1) * l;
									gridnodes[jx][jy][jz].l_vestz += wf * vel_particle_est(2) * l;

									gridnodes[jx][jy][jz].mass += wf * rmass[i];
									gridnodes[jx][jy][jz].M += wf * l * l.transpose();
								}
							}
						}
					}
				}
			} // end loop over x cell index
		} // end if setflag[itype][itype]
	}

	// normalize grid data
	for (ix = 0; ix < grid_nx; ix++) {
		for (iy = 0; iy < grid_ny; iy++) {
			for (iz = 0; iz < grid_nz; iz++) {
				if (gridnodes[ix][iy][iz].mass > MASS_CUTOFF) {
					if (domain->dimension == 2) {
						/*
						 * WLS moment matrix will have undefined z components.
						 * Take care of this by augmenting the (1 x y) space with diagonal entry for z so
						 * matrix inversion below will be successful.
						 */
						gridnodes[ix][iy][iz].M.row(3).setZero();
						gridnodes[ix][iy][iz].M.col(3).setZero();
						gridnodes[ix][iy][iz].M(3, 3) = 1.0;
					}

					//pinv4(gridnodes[ix][iy][iz].M);

					if (fabs(gridnodes[ix][iy][iz].M.determinant()) > MIN_MATRIX_DETERMINANT) {

						/*
						 * invert moment matrix and store it
						 */
						Minv = gridnodes[ix][iy][iz].M.inverse();
						gridnodes[ix][iy][iz].M = Minv;
						//Minv = gridnodes[ix][iy][iz].M; // = Minv;

						/*
						 * this is the intermediate velocity at t + 0.5*deltat
						 * it is used to compute the velocity gradient
						 */
						gridnodes[ix][iy][iz].l_vestx = (Minv * gridnodes[ix][iy][iz].l_vestx).eval();
						gridnodes[ix][iy][iz].l_vesty = (Minv * gridnodes[ix][iy][iz].l_vesty).eval();
						gridnodes[ix][iy][iz].l_vestz = (Minv * gridnodes[ix][iy][iz].l_vestz).eval();

						/*
						 * this is velocity at beginning of step.
						 */
						gridnodes[ix][iy][iz].l_vx = (Minv * gridnodes[ix][iy][iz].l_vx).eval();
						gridnodes[ix][iy][iz].l_vy = (Minv * gridnodes[ix][iy][iz].l_vy).eval();
						gridnodes[ix][iy][iz].l_vz = (Minv * gridnodes[ix][iy][iz].l_vz).eval();

						/*
						 * mark this grid node as having accurate, i.e., WLS interpolated data on it.
						 */
						gridnodes[ix][iy][iz].isAccurate = true;

//						cout << "==================================================" << endl;
//						cout << "this is vx: " << gridnodes[ix][iy][iz].vx << endl;
//						cout << "this is nodal M" << endl << gridnodes[ix][iy][iz].M << endl << endl;
						//cout << "this is nodal M inverted " << endl << gridnodes[ix][iy][iz].M.inverse() << endl << endl;
						//cout << "this is l_vx                " << gridnodes[ix][iy][iz].l_vx.transpose() << endl;
						// << "this is the solution vector " << l.transpose() << endl << endl;
						//cout << "this is the difference between C0 and C1 interpolation of vx: " << gridnodes[ix][iy][iz].vx - l(0) << endl;
					} else {
						gridnodes[ix][iy][iz].M.setZero();
					}
				}
			}
		}
	}
}

/*
 * Solve the momentum balance on the grid
 * 4) compute particle deformation gradient
 * 5) update particle strain and compute updated particle stress
 * 6) compute internal grid forces from particle stresses
 */

void PairSmdWlsMpm::ComputeVelocityGradient() {
	int *type = atom->type;
	double **x = atom->x;
	double **v = atom->v;
	int nlocal = atom->nlocal;
	int i, itype;

	int ix, iy, iz, jx, jy, jz;
	double px_shifted, py_shifted, pz_shifted; // shifted coords of particles
	Vector3d g, vel_grid, dx, vel_particle;
	Matrix3d velocity_gradient;
	double delx_scaled, delx_scaled_abs, dely_scaled, dely_scaled_abs, wfx, wfy, wf;
	double delz_scaled, delz_scaled_abs, wfz;
	Vector4d l, l_vx, l_vy, l_vz;

	// compute deformation gradient
	for (i = 0; i < nlocal; i++) {

		itype = type[i];
		if (setflag[itype][itype]) {

			velocity_gradient.setZero();
			vel_particle << v[i][0], v[i][1], v[i][2];
			px_shifted = x[i][0] - min_ix * cellsize + GRID_OFFSET * cellsize;
			py_shifted = x[i][1] - min_iy * cellsize + GRID_OFFSET * cellsize;
			pz_shifted = x[i][2] - min_iz * cellsize + GRID_OFFSET * cellsize;
			l_vx.setZero();
			l_vy.setZero();
			l_vz.setZero();

			ix = icellsize * px_shifted;
			iy = icellsize * py_shifted;
			iz = icellsize * pz_shifted;

			double norm = 0.0;

			for (jx = ix - STENCIL_LOW; jx < ix + STENCIL_HIGH; jx++) {

				delx_scaled = px_shifted * icellsize - 1.0 * jx;
				delx_scaled_abs = fabs(delx_scaled);
				wfx = DisneyKernel(delx_scaled_abs);

				for (jy = iy - STENCIL_LOW; jy < iy + STENCIL_HIGH; jy++) {

					dely_scaled = py_shifted * icellsize - 1.0 * jy;
					dely_scaled_abs = fabs(dely_scaled);
					wfy = DisneyKernel(dely_scaled_abs);

					for (jz = iz - STENCIL_LOW; jz < iz + STENCIL_HIGH; jz++) {

						// only use information on grid nodes which is accurate
						if (gridnodes[jx][jy][jz].isAccurate) {

							//if (fabs(gridnodes[jx][jy][jz].dvxdx - 1.0) > 1.0e-3) {
							//	printf("grid dvxdx = %f\n", gridnodes[jx][jy][jz].dvxdx);
							//}

							delz_scaled = pz_shifted * icellsize - 1.0 * jz;
							delz_scaled_abs = fabs(delz_scaled);
							wfz = DisneyKernel(delz_scaled_abs);

							wf = wfx * wfy * wfz; // this is the total weight function -- a dyadic product of the cartesian weight functions

							velocity_gradient(0, 0) += wf * gridnodes[jx][jy][jz].l_vestx(1);
							velocity_gradient(0, 1) += wf * gridnodes[jx][jy][jz].l_vestx(2);
							velocity_gradient(0, 2) += wf * gridnodes[jx][jy][jz].l_vestx(3);

							velocity_gradient(1, 0) += wf * gridnodes[jx][jy][jz].l_vesty(1);
							velocity_gradient(1, 1) += wf * gridnodes[jx][jy][jz].l_vesty(2);
							velocity_gradient(1, 2) += wf * gridnodes[jx][jy][jz].l_vesty(3);

							velocity_gradient(1, 0) += wf * gridnodes[jx][jy][jz].l_vestz(1);
							velocity_gradient(1, 1) += wf * gridnodes[jx][jy][jz].l_vestz(2);
							velocity_gradient(1, 2) += wf * gridnodes[jx][jy][jz].l_vestz(3);

//							velocity_gradient(0, 0) += wf * gridnodes[jx][jy][jz].l_vx(1);
//							velocity_gradient(0, 1) += wf * gridnodes[jx][jy][jz].l_vx(2);
//							velocity_gradient(0, 2) += wf * gridnodes[jx][jy][jz].l_vx(3);
//
//							velocity_gradient(1, 0) += wf * gridnodes[jx][jy][jz].l_vy(1);
//							velocity_gradient(1, 1) += wf * gridnodes[jx][jy][jz].l_vy(2);
//							velocity_gradient(1, 2) += wf * gridnodes[jx][jy][jz].l_vy(3);
//
//							velocity_gradient(1, 0) += wf * gridnodes[jx][jy][jz].l_vz(1);
//							velocity_gradient(1, 1) += wf * gridnodes[jx][jy][jz].l_vz(2);
//							velocity_gradient(1, 2) += wf * gridnodes[jx][jy][jz].l_vz(3);

							norm += wf;
						}
					}
				}
			}

			L[i] = icellsize * velocity_gradient / norm;
		}

	} // end if (setflag[itype][itype])
}
// ---- end velocity gradients ----

void PairSmdWlsMpm::ComputeGridForces() {
	double **x = atom->x;
	//double **f = atom->f;
	double *vfrac = atom->vfrac;
	int *type = atom->type;
	int nall = atom->nlocal + atom->nghost;
	int i, itype;
	int ix, iy, iz, jx, jy, jz;
	double px_shifted, py_shifted, pz_shifted; // shifted coords of particles
	Vector3d g, force, dx;
	double delx_scaled, delx_scaled_abs, dely_scaled, dely_scaled_abs, wfx, wfy, wf, wfdx, wfdy;
	double delz_scaled, delz_scaled_abs, wfz, wfdz, vol;
	Matrix3d scaledStress;
	Matrix4d g4, l;

// ---- compute internal forces ---
	for (i = 0; i < nall; i++) {

		itype = type[i];
		if (setflag[itype][itype]) {

			px_shifted = x[i][0] - min_ix * cellsize + GRID_OFFSET * cellsize;
			py_shifted = x[i][1] - min_iy * cellsize + GRID_OFFSET * cellsize;
			pz_shifted = x[i][2] - min_iz * cellsize + GRID_OFFSET * cellsize;

			ix = icellsize * px_shifted;
			iy = icellsize * py_shifted;
			iz = icellsize * pz_shifted;

			vol = vfrac[i];
			scaledStress = -vol * stressTensor[i];

			for (jx = ix - STENCIL_LOW; jx < ix + STENCIL_HIGH; jx++) {

				delx_scaled = px_shifted * icellsize - 1.0 * jx;
				delx_scaled_abs = fabs(delx_scaled);
				dx(0) = delx_scaled * cellsize;
				wfx = DisneyKernel(delx_scaled_abs);
				wfdx = DisneyKernelDerivative(delx_scaled_abs) * icellsize;
				if (delx_scaled < 0.0)
					wfdx = -wfdx;

				for (jy = iy - STENCIL_LOW; jy < iy + STENCIL_HIGH; jy++) {

					dely_scaled = py_shifted * icellsize - 1.0 * jy;
					dely_scaled_abs = fabs(dely_scaled);
					dx(1) = dely_scaled * cellsize;
					wfy = DisneyKernel(dely_scaled_abs);
					wfdy = DisneyKernelDerivative(dely_scaled_abs) * icellsize;
					if (dely_scaled < 0.0)
						wfdy = -wfdy;

					for (jz = iz - STENCIL_LOW; jz < iz + STENCIL_HIGH; jz++) {

						delz_scaled = pz_shifted * icellsize - 1.0 * jz;
						delz_scaled_abs = fabs(delz_scaled);
						dx(2) = delz_scaled * cellsize;
						wfz = DisneyKernel(delz_scaled_abs);
						wfdz = DisneyKernelDerivative(delz_scaled_abs) * icellsize;
						if (delz_scaled < 0.0)
							wfdz = -wfdz;

						wf = wfx * wfy * wfz; // this is the total weight function -- a dyadic product of the cartesian weight functions

						g(0) = wfdx * wfy * wfz; // this is the kernel gradient
						g(1) = wfdy * wfx * wfz;
						g(2) = wfdz * wfx * wfy;

						force = scaledStress * g;

						//force(0) += wf * f[i][0]; // these are body force from other force fields, e.g. contact
						//force(1) += wf * f[i][1];
						//force(2) += wf * f[i][2];

						gridnodes[jx][jy][jz].fx += force(0);
						gridnodes[jx][jy][jz].fy += force(1);
						gridnodes[jx][jy][jz].fz += force(2);
					}
				}
			}
		} // end if (setflag[itype][itype])
	}
}

/*
 * update grid velocities using grid forces
 */

void PairSmdWlsMpm::UpdateGridVelocities() {

	int ix, iy, iz;
	double dtm;

	for (ix = 0; ix < grid_nx; ix++) {
		for (iy = 0; iy < grid_ny; iy++) {
			for (iz = 0; iz < grid_nz; iz++) {
				if (gridnodes[ix][iy][iz].mass > MASS_CUTOFF) {
					if (gridnodes[ix][iy][iz].isVelocityBC == false) {
						dtm = update->dt / gridnodes[ix][iy][iz].mass;
						gridnodes[ix][iy][iz].l_vx(0) += dtm * gridnodes[ix][iy][iz].fx;
						gridnodes[ix][iy][iz].l_vy(0) += dtm * gridnodes[ix][iy][iz].fy;
						gridnodes[ix][iy][iz].l_vz(0) += dtm * gridnodes[ix][iy][iz].fz;
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

void PairSmdWlsMpm::GridToPoints() {
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
	double norm;
	Vector4d l;

	for (i = 0; i < nlocal; i++) {

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
			vel_particle << v[i][0], v[i][1], v[i][2];

			norm = 0.0;

			for (jx = ix - STENCIL_LOW; jx < ix + STENCIL_HIGH; jx++) {

				delx_scaled = px_shifted * icellsize - 1.0 * jx;
				delx_scaled_abs = fabs(delx_scaled);
				wfx = DisneyKernel(delx_scaled_abs);

				for (jy = iy - STENCIL_LOW; jy < iy + STENCIL_HIGH; jy++) {

					dely_scaled = py_shifted * icellsize - 1.0 * jy;
					dely_scaled_abs = fabs(dely_scaled);
					wfy = DisneyKernel(dely_scaled_abs);

					for (jz = iz - STENCIL_LOW; jz < iz + STENCIL_HIGH; jz++) {

						if (gridnodes[jx][jy][jz].isAccurate) {

							delz_scaled = pz_shifted * icellsize - 1.0 * jz;
							delz_scaled_abs = fabs(delz_scaled);
							wfz = DisneyKernel(delz_scaled_abs);

							wf = wfx * wfy * wfz;

							if (gridnodes[jx][jy][jz].mass > MASS_CUTOFF) {
								particleAccelerations[i](0) += wf * gridnodes[jx][jy][jz].fx / gridnodes[jx][jy][jz].mass;
								particleAccelerations[i](1) += wf * gridnodes[jx][jy][jz].fy / gridnodes[jx][jy][jz].mass;
								//particleAccelerations[i](2) += wf * gridnodes[jx][jy][jz].fz / gridnodes[jx][jy][jz].mass;
							}

							norm += wf;

							if (domain->dimension == 2) {
								delz_scaled = 0.0;
							}

							l << 1.0, delx_scaled, dely_scaled, delz_scaled;
							particleVelocities[i](0) += wf * l.dot(gridnodes[jx][jy][jz].l_vx);
							particleVelocities[i](1) += wf * l.dot(gridnodes[jx][jy][jz].l_vy);
							particleVelocities[i](2) += wf * l.dot(gridnodes[jx][jy][jz].l_vz);
						}

					}
				}
			}

			particleVelocities[i] /= norm;
			particleAccelerations[i] /= norm;

			f[i][0] = rmass[i] * particleAccelerations[i](0);
			f[i][1] = rmass[i] * particleAccelerations[i](1);
			f[i][2] = rmass[i] * particleAccelerations[i](2);

			if (mol[i] < 0) {

				Vector3d oldvel;
				oldvel << v[i][0], v[i][1], v[i][2];

				if ((oldvel - particleVelocities[i]).norm() > 1.0e-3) {
					cout << " this is old vel " << oldvel.transpose() << endl;
					cout << " this is new vel " << particleVelocities[i].transpose() << endl << endl;
					error->one(FLERR, "--");
				}

				//particleVelocities[i] << v[i][0], v[i][1], v[i][2];
				//particleAccelerations[i].setZero();
			}

		} // end if (setflag[itype][itype])
	}
}

/* ---------------------------------------------------------------------- */
void PairSmdWlsMpm::UpdateDeformationGradient() {

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

void PairSmdWlsMpm::DestroyGrid() {

	memory->destroy(gridnodes);

}

/* ---------------------------------------------------------------------- */

void PairSmdWlsMpm::compute(int eflag, int vflag) {

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
		delete[] M;
		M = new Matrix4d[nmax];
	}

	CreateGrid();
	PointsToGrid();

	ComputeVelocityGradient(); // this only interpolates the velocity gradient to the particle

	AssembleStressTensor();

	comm->forward_comm_pair(this);
	ComputeGridForces();
	UpdateGridVelocities();

	GridToPoints();

	DumpGrid();

	DestroyGrid();

}

void PairSmdWlsMpm::UpdateStress() {
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

void PairSmdWlsMpm::GetStress() {
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
void PairSmdWlsMpm::AssembleStressTensor() {
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
			vfrac[i] += update->dt * vfrac[i] * d_iso;			// update the volume

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

//printf("stable timestep = %g\n", 0.1 * hMin * MaxBulkVelocity);
}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairSmdWlsMpm::allocate() {

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

	memory->create(cutsq, n + 1, n + 1, "pair:cutsq"); // always needs to be allocated, even with granular neighborlist

	/*
	 * initialize arrays to default values
	 */

	for (int i = 1; i <= n; i++) {
		for (int j = i; j <= n; j++) {
			setflag[i][j] = 0;
		}
	}

}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairSmdWlsMpm::settings(int narg, char **arg) {
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

void PairSmdWlsMpm::coeff(int narg, char **arg) {
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

double PairSmdWlsMpm::init_one(int i, int j) {

	if (!allocated)
		allocate();

	if (setflag[i][j] == 0)
		error->all(FLERR, "All pair coeffs are not set");

	return 3.0 * cellsize;
}

/* ----------------------------------------------------------------------
 init specific to this pair style
 ------------------------------------------------------------------------- */

void PairSmdWlsMpm::init_style() {
// request a granular neighbor list
	int irequest = neighbor->request(this);
	neighbor->requests[irequest]->half = 0;
	neighbor->requests[irequest]->gran = 1;
}

/* ----------------------------------------------------------------------
 neighbor callback to inform pair style of neighbor list to use
 optional granular history list
 ------------------------------------------------------------------------- */

void PairSmdWlsMpm::init_list(int id, NeighList *ptr) {
	if (id == 0)
		list = ptr;
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based arrays
 ------------------------------------------------------------------------- */

double PairSmdWlsMpm::memory_usage() {

//printf("in memory usage\n");

	return 11 * nmax * sizeof(double) + grid_nx * grid_ny * grid_nz * sizeof(Gridnode);

}

/* ---------------------------------------------------------------------- */

int PairSmdWlsMpm::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc) {
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

void PairSmdWlsMpm::unpack_forward_comm(int n, int first, double *buf) {
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

void *PairSmdWlsMpm::extract(const char *str, int &i) {
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
	}

	return NULL;
}

/* ----------------------------------------------------------------------
 compute effective shear modulus by dividing rate of deviatoric stress with rate of shear deformation
 ------------------------------------------------------------------------- */

double PairSmdWlsMpm::effective_shear_modulus(const Matrix3d d_dev, const Matrix3d deltaStressDev, const double dt,
		const int itype) {
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

/*
 * build WLS moment matrix
 * only the upper symmetric half is built.
 */

void PairSmdWlsMpm::AccumulateMomentMatrix(double x, double y, double z, double W, Matrix4d &M) {

	M(0, 0) += W;
	M(0, 0) += W * x * x;
	M(1, 1) += W * x * x;
	M(2, 2) += W * y * y;
	M(3, 3) += W * z * z;

	M(0, 1) += W * x;
	M(0, 2) += W * y;
	M(0, 3) += W * z;

	M(1, 2) += W * x * y;
	M(1, 3) += W * x * z;

	M(2, 3) += W * y * z;
}

void PairSmdWlsMpm::DumpGrid() {
	int ix, iy, iz;

	FILE * f;
	f = fopen("grid.dump", "w");

	fprintf(f, "%d\n\n", grid_nx * grid_ny * grid_nz);

	for (ix = 0; ix < grid_nx; ix++) {
		for (iy = 0; iy < grid_ny; iy++) {
			for (iz = 0; iz < grid_nz; iz++) {
				if (gridnodes[ix][iy][iz].isAccurate) {
					fprintf(f, "X %f %f %f %f\n", ix * cellsize, iy * cellsize, iz * cellsize, gridnodes[ix][iy][iz].l_vestx(0));
				} else {
					fprintf(f, "Y %f %f %f %f\n", ix * cellsize, iy * cellsize, iz * cellsize, -999.0);
				}

			}
		}
	}

	fclose(f);

}

