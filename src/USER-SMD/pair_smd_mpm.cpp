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
#include <unsupported/Eigen/MatrixFunctions>

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
#define FACTOR 1

enum {
	COPY_VELOCITIES, COPY_FORCES, COPY_NEW_VELOCITIES
};

PairSmdMpm::PairSmdMpm(LAMMPS *lmp) :
		Pair(lmp) {

	// per-type arrays
	Q1 = NULL;
	eos = viscosity = strength = NULL;
	c0_type = NULL;
	c0 = NULL;
	J = NULL;
	vol = NULL;
	heat_conduction_coeff = NULL;
	Lookup = NULL;

	nmax = 0; // make sure no atom on this proc such that initial memory allocation is correct
	stressTensor = L = F = NULL;
	numNeighs = NULL;
	particleVelocities = particleAccelerations = NULL;
	heat_gradient = NULL;
	particleHeat = particleHeatRate = NULL;

	comm_forward = 10; // this pair style communicates 8 doubles to ghost atoms

	Bp_exists = false;
	APIC = false;

	timeone_PointstoGrid = timeone_Gradients = timeone_MaterialModel = timeone_GridForces = timeone_UpdateGrid =
			timeone_GridToPoints = 0.0;
	timeone_SymmetryBC = 0.0;

	symmetry_plane_y_plus_exists = symmetry_plane_y_minus_exists = false;
	symmetry_plane_x_plus_exists = symmetry_plane_x_minus_exists = false;
	symmetry_plane_z_plus_exists = symmetry_plane_z_minus_exists = false;
	noslip_symmetry_plane_y_plus_exists = noslip_symmetry_plane_y_minus_exists = false;
}

/* ---------------------------------------------------------------------- */

PairSmdMpm::~PairSmdMpm() {

	double time_PointstoGrid, time_Gradients, time_MaterialModel, time_GridForces, time_UpdateGrid, time_GridToPoints,
			time_SymmetryBC;

	MPI_Allreduce(&timeone_PointstoGrid, &time_PointstoGrid, 1, MPI_DOUBLE, MPI_SUM, world);
	MPI_Allreduce(&timeone_Gradients, &time_Gradients, 1, MPI_DOUBLE, MPI_SUM, world);
	MPI_Allreduce(&timeone_MaterialModel, &time_MaterialModel, 1, MPI_DOUBLE, MPI_SUM, world);
	MPI_Allreduce(&timeone_GridForces, &time_GridForces, 1, MPI_DOUBLE, MPI_SUM, world);
	MPI_Allreduce(&timeone_UpdateGrid, &time_UpdateGrid, 1, MPI_DOUBLE, MPI_SUM, world);
	MPI_Allreduce(&timeone_GridToPoints, &time_GridToPoints, 1, MPI_DOUBLE, MPI_SUM, world);
	MPI_Allreduce(&timeone_SymmetryBC, &time_SymmetryBC, 1, MPI_DOUBLE, MPI_SUM, world);

	double time_sum = time_PointstoGrid + time_Gradients + time_MaterialModel + time_GridForces + time_UpdateGrid
			+ time_GridToPoints + time_SymmetryBC;
	if (comm->me == 0) {
		printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n");
		printf("... SMD / MPM CPU (NOT WALL CLOCK) TIMING STATISTICS\n\n");
		printf("%20s is %10.2f seconds, %3.2f percent of pair time\n", "points to grid", time_PointstoGrid,
				100 * time_PointstoGrid / time_sum);
		printf("%20s is %10.2f seconds, %3.2f percent of pair time\n", "gradients", time_Gradients,
				100 * time_Gradients / time_sum);
		printf("%20s is %10.2f seconds, %3.2f percent of pair time\n", "material model", time_MaterialModel,
				100 * time_MaterialModel / time_sum);
		printf("%20s is %10.2f seconds, %3.2f percent of pair time\n", "grid forces", time_GridForces,
				100 * time_GridForces / time_sum);
		printf("%20s is %10.2f seconds, %3.2f percent of pair time\n", "grid update", time_UpdateGrid,
				100 * time_UpdateGrid / time_sum);
		printf("%20s is %10.2f seconds, %3.2f percent of pair time\n", "grid to points", time_GridToPoints,
				100 * time_GridToPoints / time_sum);
		printf("%20s is %10.2f seconds, %3.2f percent of pair time\n", "symmetry BC", time_SymmetryBC,
				100 * time_SymmetryBC / time_sum);
		printf(">>========>>========>>========>>========>>========>>========>>========>>========\n\n");
	}

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
		delete[] J;
		delete[] vol;

	}
}

/* ---------------------------------------------------------------------- */

void PairSmdMpm::CreateGrid() {
	double **x = atom->x;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	int nall = nlocal + atom->nghost;
	int i, itype;
	icellsize = 1.0 / cellsize; // inverse of cell size

	// get bounds of this processor's simulation box
	//printf("bounds min: %f %f %f\n", domain->sublo[0], domain->sublo[1], domain->sublo[2]);
	//printf("bounds max: %f %f %f\n", domain->subhi[0], domain->subhi[1], domain->subhi[2]);

	// get min / max position of all particles
	minx = domain->sublo[0];
	miny = domain->sublo[1];
	minz = domain->sublo[2];
	maxx = domain->subhi[0];
	maxy = domain->subhi[1];
	maxz = domain->subhi[2];
	//minx = miny = minz = BIG;
	//maxx = maxy = maxz = -BIG;

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

	if (symmetry_plane_y_plus_exists) {
		if (miny < symmetry_plane_y_plus_location - 0.1 * cellsize) {
			error->one(FLERR, "Cannot have particle or box boundary below y+ symmetry plane.");
		}
		miny = symmetry_plane_y_plus_location;
	}

	if (symmetry_plane_y_minus_exists) {
		if (maxy > symmetry_plane_y_minus_location + 0.1 * cellsize) {
			printf("maxy is %f, symm plane is at %f\n", maxy, symmetry_plane_y_minus_location + 0.1 * cellsize);
			error->one(FLERR, "Cannot have particle or box boundary above y- symmetry plane.");
		}
		maxy = symmetry_plane_y_minus_location;
	}

	if (symmetry_plane_x_plus_exists) {
		if (minx < symmetry_plane_x_plus_location - 0.1 * cellsize) {
			error->one(FLERR, "Cannot have particle or box boundary below x+ symmetry plane.");
		}
		minx = symmetry_plane_x_plus_location;
	}

	if (symmetry_plane_x_minus_exists) {
		if (maxx > symmetry_plane_x_minus_location + 0.1 * cellsize) {
			error->one(FLERR, "Cannot have particle or box boundary above x- symmetry plane.");
		}
		maxx = symmetry_plane_x_minus_location;
	}

	if (symmetry_plane_z_plus_exists) {
		if (minz < symmetry_plane_z_plus_location - 0.1 * cellsize) {
			error->warning(FLERR, "Cannot have particle or box boundary below z+ symmetry plane.");
		}
		minz = symmetry_plane_z_plus_location;
	}

	if (symmetry_plane_z_minus_exists) {
		if (maxz > symmetry_plane_z_minus_location + 0.1 * cellsize) {
			error->one(FLERR, "Cannot have particle or box boundary above z- symmetry plane.");
		}
		maxz = symmetry_plane_z_minus_location;
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

	minx -= cellsize; // this ensures we have only positive cell indices
	miny -= cellsize; // needed for linear gridnodes
	minz -= cellsize;
	maxx -= cellsize;
	maxy -= cellsize;
	maxz -= cellsize;

	// allocate grid storage
	// we need a triple of indices (i, j, k)

//	printf("proc %d: nx=%f, ny=%f, nz=%f\n", comm->me, minx, miny, minz);
//	printf("proc %d: nx=%f, ny=%f, nz=%f\n", comm->me, maxx, maxy, maxz);
//	printf("proc %d: nx=%d, ny=%d, nz=%d\n", comm->me, grid_nx, grid_ny, grid_nz);

	memory->create(gridnodes, grid_nx, grid_ny, grid_nz, "pair:gridnodes");

	Ncells = grid_nx * grid_ny * grid_nz;
	lgridnodes = NULL;
	lgridnodes = new Gridnode[Ncells];

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
	double delz_scaled, delz_scaled_abs, wfz, particle_mass, particle_heat;
	double px_shifted, py_shifted, pz_shifted; // shifted coords of particles
	Vector3d vel_APIC, vel_particle, vel_particle_est, dx;
	Matrix3d eye, Dp, Dp_inv, Cp, Bp;

	// clear the grid
	for (ix = 0; ix < grid_nx; ix++) {
		for (iy = 0; iy < grid_ny; iy++) {
			for (iz = 0; iz < grid_nz; iz++) {
				//memset(&gridnodes[ix][iy][iz], 0, sizeof(Gridnode));
//
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

			particle_mass = rmass[i]; // pre-compute all possible quantities
			vel_particle << v[i][0], v[i][1], v[i][2];
			vel_particle_est << vest[i][0], vest[i][1], vest[i][2];
			vel_particle *= particle_mass;
			vel_particle_est *= particle_mass;
			particle_heat = particle_mass * heat[i];

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

									wf = wfx * wfy * wfz; // this is the total weight function -- a dyadic product of the cartesian weight functions

									//if (APIC) {
									//	vel_APIC = Cp * dx + vel_particle; // this is the APIC corrected velocity
									//	gridnodes[jx][jy][jz].v += wf * rmass[i] * vel_APIC;
									//} else {
									gridnodes[jx][jy][jz].v += wf * vel_particle;
									//}

									gridnodes[jx][jy][jz].vest += wf * vel_particle_est;
									gridnodes[jx][jy][jz].mass += wf * particle_mass;
									gridnodes[jx][jy][jz].heat += wf * particle_heat;
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

// use a linear array for grid nodes

void PairSmdMpm::PointsToGridLinear() {
	double **v = atom->v;
	double **vest = atom->vest;
	double *rmass = atom->rmass;
	double *heat = atom->heat;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	int nall = nlocal + atom->nghost;
	int ref_node;
	double wfx[4], wfy[4], wfz[4];
	Vector3d vel_particle, vel_particle_est;

	for (int icell = 0; icell < Ncells; icell++) {
		lgridnodes[icell].mass = 0.0;
		lgridnodes[icell].heat = 0.0;
		lgridnodes[icell].dheat_dt = 0.0;
		lgridnodes[icell].f.setZero();
		lgridnodes[icell].v.setZero();
		lgridnodes[icell].vest.setZero();
		lgridnodes[icell].isVelocityBC = false;
	}

	for (int i = 0; i < nall; i++) {

		int itype = type[i];
		if (setflag[itype][itype]) {

			// pre-compute all possible quantities
			double particle_mass = rmass[i];
			double particle_heat = particle_mass * heat[i];
			vel_particle << v[i][0], v[i][1], v[i][2];
			vel_particle_est << vest[i][0], vest[i][1], vest[i][2];
			vel_particle *= particle_mass;
			vel_particle_est *= particle_mass;

			PreComputeGridWeights(i, ref_node, wfx, wfy, wfz);

			// loop over all cell neighbors for this particle
			for (int ix = 0; ix < 4; ix++) {
				for (int iy = 0; iy < 4; iy++) {
					for (int iz = 0; iz < 4; iz++) {

						double wf = wfx[ix] * wfy[iy] * wfz[iz]; // total weight function
						int node_index = ref_node + ix + iy * grid_nx + iz * grid_nx * grid_ny;

						if ((node_index < 0) || (node_index >= Ncells)) { // memory access error check
							printf("node index %d outside allowed range %d to %d\n", node_index, 0, Ncells);
							exit(1);
						}

						lgridnodes[node_index].v += wf * vel_particle;
						lgridnodes[node_index].vest += wf * vel_particle_est;
						lgridnodes[node_index].mass += wf * particle_mass;
						lgridnodes[node_index].heat += wf * particle_heat;
					}
				}
			}

		} // end if setflag
	} // end loop over nall

	// normalize all grid cells
	for (int icell = 0; icell < Ncells; icell++) {
		if (lgridnodes[icell].mass > MASS_CUTOFF) {
			lgridnodes[icell].v /= lgridnodes[icell].mass;
			lgridnodes[icell].vest /= lgridnodes[icell].mass;
			lgridnodes[icell].heat /= lgridnodes[icell].mass;
		}
	}

}

void PairSmdMpm::PreComputeGridWeights(const int i, int &ref_node, double *wfx, double *wfy, double *wfz) {
	double **x = atom->x;

	double ssx = icellsize * (x[i][0] - minx); // shifted position in grid coords
	double ssy = icellsize * (x[i][1] - miny); // shifted position in grid coords
	double ssz = icellsize * (x[i][2] - minz); // shifted position in grid coords

	int ref_ix = (int) ssx - 1; // this is the x location of the reference node in cell space
	int ref_iy = (int) ssy - 1; // this is the y location of the reference node in cell space
	int ref_iz = (int) ssz - 1; // this is the z location of the reference node in cell space
	ref_node = ref_ix + ref_iy * grid_nx + ref_iz * grid_nx * grid_ny; // this is the index of the reference node

	if ((ref_node < 0) || (ref_node > Ncells - 1)) { // memory access error check
		printf("ref node %d outside allowed range %d to %d\n", ref_node, 0, Ncells);
		printf("ref_ix=%d, ref_iy=%d, ref_iz=%d\n", ref_ix, ref_iy, ref_iz);
		printf("ssx = %f, ssy=%f, ssz=%f\n", ssx, ssy, ssz);
		printf("x-minx=%f, y-miny=%f, z-minz=%f\n", (x[i][0] - minx), (x[i][1] - miny), (x[i][2] - minz));
		exit(1);
	}

	// pre-compute kernel weights
	for (int ioff = 0; ioff < 4; ioff++) { // this is the x stencil visible to the current particle
		double dist = fabs(ref_ix + ioff - ssx); // this is the x distance in world coords
		wfx[ioff] = DisneyKernel(dist);
		dist = fabs(ref_iy + ioff - ssy); // this is the x distance in world coords
		wfy[ioff] = DisneyKernel(dist);
		dist = fabs(ref_iz + ioff - ssz); // this is the x distance in world coords
		wfz[ioff] = DisneyKernel(dist);
	}

}

void PairSmdMpm::PreComputeGridWeightsAndDerivatives(const int i, int &ref_node, double *wfx, double *wfy, double *wfz,
		double *wfdx, double *wfdy, double *wfdz) {
	double **x = atom->x;

	double ssx = icellsize * (x[i][0] - minx); // shifted position in grid coords
	double ssy = icellsize * (x[i][1] - miny); // shifted position in grid coords
	double ssz = icellsize * (x[i][2] - minz); // shifted position in grid coords

	int ref_ix = (int) ssx - 1; // this is the x location of the reference node in cell space
	int ref_iy = (int) ssy - 1; // this is the y location of the reference node in cell space
	int ref_iz = (int) ssz - 1; // this is the z location of the reference node in cell space
	ref_node = ref_ix + ref_iy * grid_nx + ref_iz * grid_nx * grid_ny; // this is the index of the reference node

	if ((ref_node < 0) || (ref_node > Ncells - 1)) { // memory access error check
		printf("ref node %d outside allowed range %d to %d\n", ref_node, 0, Ncells);
		printf("ref_ix=%d, ref_iy=%d, ref_iz=%d\n", ref_ix, ref_iy, ref_iz);
		printf("ssx = %f, ssy=%f, ssz=%f\n", ssx, ssy, ssz);
		printf("x-minx=%f, y-miny=%f, z-minz=%f\n", (x[i][0] - minx), (x[i][1] - miny), (x[i][2] - minz));
		exit(1);
	}

	// pre-compute kernel weights
	for (int ioff = 0; ioff < 4; ioff++) { // this is the x stencil visible to the current particle
		double dist = ref_ix + ioff - ssx; // this is the x distance in world coords
		DisneyKernelAndDerivative(icellsize, dist, wfx[ioff], wfdx[ioff]);
		dist = ref_iy + ioff - ssy; // this is the y distance in world coords
		DisneyKernelAndDerivative(icellsize, dist, wfy[ioff], wfdy[ioff]);
		dist = ref_iz + ioff - ssz; // this is the z distance in world coords
		DisneyKernelAndDerivative(icellsize, dist, wfz[ioff], wfdz[ioff]);
	}

}

/*
 * Extrapolate only particle velocities to grid.
 * Used in MUSL.
 */

void PairSmdMpm::ScatterVelocities() {
	double **x = atom->x;
	double **v = atom->v;
	double *rmass = atom->rmass;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	int nall = nlocal + atom->nghost;
	int i, itype;
	int ix, iy, iz, jx, jy, jz;
	double delx_scaled, delx_scaled_abs, dely_scaled, dely_scaled_abs, wf, wfx, wfy;
	double delz_scaled, delz_scaled_abs, wfz, particle_mass;
	double px_shifted, py_shifted, pz_shifted; // shifted coords of particles
	Vector3d vel_particle;

	// clear the grid
	for (ix = 0; ix < grid_nx; ix++) {
		for (iy = 0; iy < grid_ny; iy++) {
			for (iz = 0; iz < grid_nz; iz++) {
				gridnodes[ix][iy][iz].v.setZero();
				gridnodes[ix][iy][iz].mass = 0.0;
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

			//vel_particle = rmass[i] * particleVelocities[i];
			vel_particle << v[i][0], v[i][1], v[i][2];
			vel_particle *= rmass[i];
			particle_mass = rmass[i];

			for (jz = iz - STENCIL_LOW; jz < iz + STENCIL_HIGH; jz++) {

				if ((jz < 0) || (jz >= grid_nz)) {
					printf("z cell index %d is outside range 0 .. %d\n", jz, grid_nz);
					error->one(FLERR, "");
				}

				delz_scaled = pz_shifted * icellsize - 1.0 * jz;
				delz_scaled_abs = fabs(delz_scaled);
				wfz = DisneyKernel(delz_scaled_abs);
				if (wfz > 0.0) {
					for (jy = iy - STENCIL_LOW; jy < iy + STENCIL_HIGH; jy++) {

						if ((jy < 0) || (jy >= grid_ny)) {
							printf("y cell indey %d is outside range 0 .. %d\n", jy, grid_ny);
							error->one(FLERR, "");
						}

						dely_scaled = py_shifted * icellsize - 1.0 * jy;
						dely_scaled_abs = fabs(dely_scaled);
						wfy = DisneyKernel(dely_scaled_abs);
						if (wfy > 0.0) {
							for (jx = ix - STENCIL_LOW; jx < ix + STENCIL_HIGH; jx++) {

								if ((jx < 0) || (jx >= grid_nx)) {
									printf("x cell index %d is outside range 0 .. %d\n", jx, grid_nx);
									error->one(FLERR, "");
								}

								delx_scaled = px_shifted * icellsize - 1.0 * jx;
								delx_scaled_abs = fabs(delx_scaled);
								wfx = DisneyKernel(delx_scaled_abs);
								if (wfx > 0.0) {
									wf = wfx * wfy * wfz; // this is the total weight function -- a dyadic product of the cartesian weight functions
									gridnodes[jx][jy][jz].v += wf * vel_particle;
									gridnodes[jx][jy][jz].mass += wf * particle_mass;
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
					gridnodes[ix][iy][iz].v /= gridnodes[ix][iy][iz].mass;
				}
			}
		}
	}
}

void PairSmdMpm::ApplyVelocityBC() {
	double **x = atom->x;
	double **v = atom->v;
	double *heat = atom->heat;
	int *type = atom->type;
	tagint *mol = atom->molecule;
	int nlocal = atom->nlocal;
	int nall = nlocal + atom->nghost;
	int i, itype;
	int ix, iy, iz, jx, jy, jz;
	double delx_scaled, delx_scaled_abs, dely_scaled, dely_scaled_abs, wf, wfx, wfy;
	double delz_scaled, delz_scaled_abs, wfz, heat_particle;
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
				heat_particle = heat[i];

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
								//gridnodes[jx][jy][jz].heat = heat_particle;
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
	Vector3d g, vel_grid, particle_heat_gradient;
	Matrix3d velocity_gradient;
	double delx_scaled, dely_scaled, wfx, wfy, wf, wfdx, wfdy;
	double delz_scaled, wfz, wfdz;

	for (i = 0; i < nlocal; i++) {

		itype = type[i];
		if (setflag[itype][itype]) {

			particle_heat_gradient.setZero();
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

									velocity_gradient += gridnodes[jx][jy][jz].vest * g.transpose(); // this is for USF
											//velocity_gradient += gridnodes[jx][jy][jz].v * g.transpose(); // this is for USL
									particle_heat_gradient += gridnodes[jx][jy][jz].heat * g; // units: energy / distance
								}
							}
						}
					}
				}
			}
			L[i] = velocity_gradient;
			heat_gradient[i] = particle_heat_gradient;
		} // end if (setflag[itype][itype])
	}
}

void PairSmdMpm::VelocityGradientLinear() {
	int *type = atom->type;
	int nlocal = atom->nlocal;
	int nall = nlocal + atom->nghost;
	int ref_node;
	double wfx[4], wfy[4], wfz[4], wfdx[4], wfdy[4], wfdz[4];
	Vector3d particle_heat_gradient, g;
	Matrix3d velocity_gradient;

	for (int i = 0; i < nall; i++) {

		int itype = type[i];
		if (setflag[itype][itype]) {

			velocity_gradient.setZero();
			particle_heat_gradient.setZero();
			PreComputeGridWeightsAndDerivatives(i, ref_node, wfx, wfy, wfz, wfdx, wfdy, wfdz);

			// loop over all cell neighbors for this particle
			for (int ix = 0; ix < 4; ix++) {
				for (int iy = 0; iy < 4; iy++) {
					for (int iz = 0; iz < 4; iz++) {

						g(0) = wfdx[ix] * wfy[iy] * wfz[iz]; // this is the kernel gradient
						g(1) = wfdy[iy] * wfx[ix] * wfz[iz];
						g(2) = wfdz[iz] * wfx[ix] * wfy[iy];

						int node_index = ref_node + ix + iy * grid_nx + iz * grid_nx * grid_ny;
						velocity_gradient += lgridnodes[node_index].vest * g.transpose(); // this is for USF
						particle_heat_gradient += lgridnodes[node_index].heat * g; // units: energy / distance
					}
				}
			}

			L[i] = -velocity_gradient;
			heat_gradient[i] = -particle_heat_gradient;
		} // end if setflag
	} // end loop over nall
}

// ---- end velocity gradients ----

void PairSmdMpm::ComputeGridForces() {
	double **x = atom->x;
	double **f = atom->f;
	double *rmass = atom->rmass;
	int *type = atom->type;
	int nall = atom->nlocal + atom->nghost;
	int i, itype;
	int ix, iy, iz, jx, jy, jz;
	double px_shifted_scaled, py_shifted_scaled, pz_shifted_scaled;
	Vector3d g, force, scaled_temperature_gradient, otherForces;
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

			scaledStress = -vol[i] * stressTensor[i];
			scaled_temperature_gradient = -vol[i] * heat_conduction_coeff[itype] * heat_gradient[i]
					/ (Lookup[HEAT_CAPACITY][itype] * rmass[i]); // units: volume * Temperature  / distance
			otherForces << f[i][0], f[i][1], f[i][2];

			for (jz = iz - STENCIL_LOW; jz < iz + STENCIL_HIGH; jz++) {
				delz_scaled = pz_shifted_scaled - 1.0 * jz;
				DisneyKernelAndDerivative(icellsize, delz_scaled, wfz, wfdz);
				if (wfz > 0.0) {

					for (jy = iy - STENCIL_LOW; jy < iy + STENCIL_HIGH; jy++) {
						dely_scaled = py_shifted_scaled - 1.0 * jy;
						DisneyKernelAndDerivative(icellsize, dely_scaled, wfy, wfdy);
						if (wfy > 0.0) {

							for (jx = ix - STENCIL_LOW; jx < ix + STENCIL_HIGH; jx++) {
								delx_scaled = px_shifted_scaled - 1.0 * jx;
								DisneyKernelAndDerivative(icellsize, delx_scaled, wfx, wfdx);
								if (wfx > 0.0) {

									wf = wfx * wfy * wfz; // this is the total weight function -- a dyadic product of the cartesian weight functions

									g(0) = wfdx * wfy * wfz; // this is the kernel gradient
									g(1) = wfdy * wfx * wfz;
									g(2) = wfdz * wfx * wfy;

									/*
									 * this is the force from the divergence of the stress field plus
									 *forces from other force fields, e.g. contact
									 */
									gridnodes[jx][jy][jz].f += scaledStress * g + wf * otherForces;
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
	double dt = FACTOR * update->dt;

	for (ix = 0; ix < grid_nx; ix++) {
		for (iy = 0; iy < grid_ny; iy++) {
			for (iz = 0; iz < grid_nz; iz++) {
				if (gridnodes[ix][iy][iz].mass > MASS_CUTOFF) {
					if (gridnodes[ix][iy][iz].isVelocityBC == false) {
						dtm = dt / gridnodes[ix][iy][iz].mass;
						gridnodes[ix][iy][iz].v += dtm * gridnodes[ix][iy][iz].f;
						gridnodes[ix][iy][iz].heat += dtm * gridnodes[ix][iy][iz].dheat_dt;
					} else {
						// Buzzi 2008 says that momentum on grid boundary nodes needs to be zeroed.
						/*
						 * But: wee need the force information.
						 * Rather than zeroing the force, use BC node force selectively when interpolating back to particles
						 */
						//gridnodes[ix][iy][iz].f.setZero();
					}
				}
			}
		}
	}
}

void PairSmdMpm::ApplyNoSlipSymmetryBC() {

//	int iy, ix, iz, source, target;
//	double px_shifted, py_shifted, pz_shifted;
//
//	if (noslip_symmetry_plane_y_plus_exists) {
//
//		// find y grid index corresponding location of y plus symmetry plane
//		py_shifted = noslip_symmetry_plane_y_plus_location - min_iy * cellsize + GRID_OFFSET * cellsize;
//		iy = icellsize * py_shifted;
//
//		if ((iy < 0) || (iy >= grid_ny)) {
//			printf("y cell index %d is outside range 0 .. %d\n", iy, grid_ny);
//			error->one(FLERR, "");
//		}
//
//		for (ix = 0; ix < grid_nx; ix++) {
//			for (iz = 0; iz < grid_nz; iz++) {
//
//				if (ix * cellsize > 40.0) {
//
//					// set all velocities to zero in the symmetry plane -- no slip condition
//					gridnodes[ix][iy][iz].v.setZero();
//					gridnodes[ix][iy][iz].vest.setZero();
//					gridnodes[ix][iy][iz].f.setZero();
//
//					// mirror velocity of nodes on the +-side to the -side
//
//					source = iy + 1;
//					target = iy - 1;
//
//					// check that cell indices are within bounds
//					if ((source < 0) || (source >= grid_ny)) {
//						printf("map from y cell index %d is outside range 0 .. %d\n", source, grid_ny);
//						error->one(FLERR, "");
//					}
//
//					if ((target < 0) || (target >= grid_ny)) {
//						printf("map to y cell index %d is outside range 0 .. %d\n", target, grid_ny);
//						error->one(FLERR, "");
//					}
//
//					// we duplicate: (jy = iy + 1 -> ky = iy - 1)
//					gridnodes[ix][target][iz].v(1) = -gridnodes[ix][source][iz].v(1);
//					gridnodes[ix][target][iz].vest(1) = -gridnodes[ix][source][iz].vest(1);
//					gridnodes[ix][target][iz].f(1) = -gridnodes[ix][source][iz].f(1);
//					gridnodes[ix][target][iz].mass = gridnodes[ix][source][iz].mass;
//				}
//
//			}
//		}
//	}
//
//	if (noslip_symmetry_plane_y_minus_exists) {
//
//		// find y grid index corresponding location of y plus symmetry plane
//		py_shifted = noslip_symmetry_plane_y_minus_location - min_iy * cellsize + GRID_OFFSET * cellsize;
//		iy = icellsize * py_shifted;
//
//		if ((iy < 0) || (iy >= grid_ny)) {
//			printf("y cell index %d is outside range 0 .. %d\n", iy, grid_ny);
//			error->one(FLERR, "");
//		}
//
//		for (ix = 0; ix < grid_nx; ix++) {
//			for (iz = 0; iz < grid_nz; iz++) {
//
//				//if (ix * cellsize > 11.0) {
//				// set all velocities to zero in the symmetry plane -- no slip condition
//				gridnodes[ix][iy][iz].v.setZero();
//				gridnodes[ix][iy][iz].vest.setZero();
//				//gridnodes[ix][iy][iz].f.setZero();
//
//				// mirror velocity of nodes on the +-side to the -side
//				source = iy - 1;
//				target = iy + 1;
//
//				// check that cell indices are within bounds
//				if ((source < 0) || (source >= grid_ny)) {
//					printf("map from y cell index %d is outside range 0 .. %d\n", source, grid_ny);
//					error->one(FLERR, "");
//				}
//
//				if ((target < 0) || (target >= grid_ny)) {
//					printf("map to y cell index %d is outside range 0 .. %d\n", target, grid_ny);
//					error->one(FLERR, "");
//				}
//
//				// we duplicate: (jy = iy + 1 -> ky = iy - 1)
//				gridnodes[ix][target][iz].v(1) = -gridnodes[ix][source][iz].v(1);
//				gridnodes[ix][target][iz].vest(1) = -gridnodes[ix][source][iz].vest(1);
//				gridnodes[ix][target][iz].f(1) = -gridnodes[ix][source][iz].f(1);
//				gridnodes[ix][target][iz].mass = gridnodes[ix][source][iz].mass;
//				//}
//
//			}
//		}
//	}
}

void PairSmdMpm::ApplySymmetryBC(int mode) {

	int iy, ix, iz, source, target;
	double px_shifted, py_shifted, pz_shifted;

	if (symmetry_plane_y_plus_exists) {

		// find y grid index corresponding location of y plus symmetry plane
		py_shifted = symmetry_plane_y_plus_location - min_iy * cellsize + GRID_OFFSET * cellsize;
		iy = icellsize * py_shifted;

		if ((iy < 0) || (iy >= grid_ny)) {
			printf("y cell index %d is outside range 0 .. %d\n", iy, grid_ny);
			error->one(FLERR, "");
		}

		for (ix = 0; ix < grid_nx; ix++) {
			for (iz = 0; iz < grid_nz; iz++) {

				// set y velocity to zero in the symmetry plane
				gridnodes[ix][iy][iz].v(1) = 0.0;
				gridnodes[ix][iy][iz].vest(1) = 0.0;
				gridnodes[ix][iy][iz].f(1) = 0.0;

				// mirror velocity of nodes on the +-side to the -side

				source = iy + 1;
				target = iy - 1;

				// check that cell indices are within bounds
				if ((source < 0) || (source >= grid_ny)) {
					printf("map from y cell index %d is outside range 0 .. %d\n", source, grid_ny);
					error->one(FLERR, "");
				}

				if ((target < 0) || (target >= grid_ny)) {
					printf("map to y cell index %d is outside range 0 .. %d\n", target, grid_ny);
					error->one(FLERR, "");
				}

				// we duplicate: (jy = iy + 1 -> ky = iy - 1)
				gridnodes[ix][target][iz].v(1) = -gridnodes[ix][source][iz].v(1);
				gridnodes[ix][target][iz].vest(1) = -gridnodes[ix][source][iz].vest(1);
				gridnodes[ix][target][iz].f(1) = -gridnodes[ix][source][iz].f(1);
				gridnodes[ix][target][iz].mass = gridnodes[ix][source][iz].mass;

			}
		}
	}

	if (symmetry_plane_y_minus_exists) {

		// find y grid index corresponding location of y plus symmetry plane
		py_shifted = symmetry_plane_y_minus_location - min_iy * cellsize + GRID_OFFSET * cellsize;
		iy = icellsize * py_shifted;

		if ((iy < 0) || (iy >= grid_ny)) {
			printf("y cell index %d is outside range 0 .. %d\n", iy, grid_ny);
			error->one(FLERR, "");
		}

		for (ix = 0; ix < grid_nx; ix++) {
			for (iz = 0; iz < grid_nz; iz++) {

				// set y velocity to zero in the symmetry plane
				gridnodes[ix][iy][iz].v(1) = 0.0;
				gridnodes[ix][iy][iz].vest(1) = 0.0;
				gridnodes[ix][iy][iz].f(1) = 0.0;

				// mirror velocity of nodes on the +-side to the -side

				source = iy - 1;
				target = iy + 1;

				// check that cell indices are within bounds
				if ((source < 0) || (source >= grid_ny)) {
					printf("map from y cell index %d is outside range 0 .. %d\n", source, grid_ny);
					error->one(FLERR, "");
				}

				if ((target < 0) || (target >= grid_ny)) {
					printf("map to y cell index %d is outside range 0 .. %d\n", target, grid_ny);
					error->one(FLERR, "");
				}

				// we duplicate: (jy = iy + 1 -> ky = iy - 1)
				gridnodes[ix][target][iz].v(1) = -gridnodes[ix][source][iz].v(1);
				gridnodes[ix][target][iz].vest(1) = -gridnodes[ix][source][iz].vest(1);
				gridnodes[ix][target][iz].f(1) = -gridnodes[ix][source][iz].f(1);
				gridnodes[ix][target][iz].mass = gridnodes[ix][source][iz].mass;

			}
		}
	}

	if (symmetry_plane_x_plus_exists) {

		// find x grid index corresponding location of x plus symmetry plane
		px_shifted = symmetry_plane_x_plus_location - min_ix * cellsize + GRID_OFFSET * cellsize;
		ix = icellsize * px_shifted;

		if ((ix < 0) || (ix >= grid_nx)) {
			printf("x cell index %d is outside range 0 .. %d\n", ix, grid_nx);
			error->one(FLERR, "");
		}

		for (iy = 0; iy < grid_ny; iy++) {
			for (iz = 0; iz < grid_nz; iz++) {

				// set y velocity to zero in the symmetry plane
				gridnodes[ix][iy][iz].v(0) = 0.0;
				gridnodes[ix][iy][iz].vest(0) = 0.0;
				gridnodes[ix][iy][iz].f(0) = 0.0;

				// mirror velocity of nodes on the +-side to the -side

				source = ix + 1;
				target = ix - 1;

				// check that cell indices are within bounds
				if ((source < 0) || (source >= grid_nx)) {
					printf("map from x cell index %d is outside range 0 .. %d\n", source, grid_nx);
					error->one(FLERR, "");
				}

				if ((target < 0) || (target >= grid_nx)) {
					printf("map to x cell index %d is outside range 0 .. %d\n", target, grid_nx);
					error->one(FLERR, "");
				}

				// we duplicate: (jy = iy + 1 -> ky = iy - 1)
				gridnodes[target][iy][iz].v(0) = -gridnodes[source][iy][iz].v(0);
				gridnodes[target][iy][iz].vest(0) = -gridnodes[source][iy][iz].vest(0);
				gridnodes[target][iy][iz].f(0) = -gridnodes[source][iy][iz].f(0);
				gridnodes[target][iy][iz].mass = gridnodes[source][iy][iz].mass;
			}
		}
	}

	if (symmetry_plane_x_minus_exists) {

		// find x grid index corresponding location of x plus symmetry plane
		px_shifted = symmetry_plane_x_minus_location - min_ix * cellsize + GRID_OFFSET * cellsize;
		ix = icellsize * px_shifted;

		if ((ix < 0) || (ix >= grid_nx)) {
			printf("x cell index %d is outside range 0 .. %d\n", ix, grid_nx);
			error->one(FLERR, "");
		}

		for (iy = 0; iy < grid_ny; iy++) {
			for (iz = 0; iz < grid_nz; iz++) {

				// set y velocity to zero in the symmetry plane
				gridnodes[ix][iy][iz].v(0) = 0.0;
				gridnodes[ix][iy][iz].vest(0) = 0.0;
				gridnodes[ix][iy][iz].f(0) = 0.0;

				// mirror velocity of nodes on the +-side to the -side

				source = ix - 1;
				target = ix + 1;

				// check that cell indices are within bounds
				if ((source < 0) || (source >= grid_nx)) {
					printf("map from x cell index %d is outside range 0 .. %d\n", source, grid_nx);
					error->one(FLERR, "");
				}

				if ((target < 0) || (target >= grid_nx)) {
					printf("map to x cell index %d is outside range 0 .. %d\n", target, grid_nx);
					error->one(FLERR, "");
				}

				// we duplicate: (jy = iy + 1 -> ky = iy - 1)
				gridnodes[target][iy][iz].v(0) = -gridnodes[source][iy][iz].v(0);
				gridnodes[target][iy][iz].vest(0) = -gridnodes[source][iy][iz].vest(0);
				gridnodes[target][iy][iz].f(0) = -gridnodes[source][iy][iz].f(0);
				gridnodes[target][iy][iz].mass = gridnodes[source][iy][iz].mass;
			}
		}
	}

	if (symmetry_plane_z_plus_exists) {
		// find z grid index corresponding location of z plus symmetry plane
		pz_shifted = symmetry_plane_z_plus_location - min_iz * cellsize + GRID_OFFSET * cellsize;
		iz = icellsize * pz_shifted;

		if ((iz < 0) || (iz >= grid_nz)) {
			printf("z cell index %d is outside range 0 .. %d\n", iz, grid_nz);
			error->one(FLERR, "");
		}

		for (iy = 0; iy < grid_ny; iy++) {
			for (ix = 0; ix < grid_nx; ix++) {

				// set y velocity to zero in the symmetry plane
				gridnodes[ix][iy][iz].v(2) = 0.0;
				gridnodes[ix][iy][iz].vest(2) = 0.0;
				gridnodes[ix][iy][iz].f(2) = 0.0;

				// mirror velocity of nodes on the +-side to the -side

				source = iz + 1;
				target = iz - 1;

				// check that cell indices are within bounds
				if ((source < 0) || (source >= grid_nz)) {
					printf("map from z cell index %d is outside range 0 .. %d\n", source, grid_nz);
					error->one(FLERR, "");
				}

				if ((target < 0) || (target >= grid_nz)) {
					printf("map to z cell index %d is outside range 0 .. %d\n", target, grid_nz);
					error->one(FLERR, "");
				}

				// we duplicate: (jy = iy + 1 -> ky = iy - 1)
				gridnodes[ix][iy][target].v(2) = -gridnodes[ix][iy][source].v(2);
				gridnodes[ix][iy][target].vest(2) = -gridnodes[ix][iy][source].vest(2);
				gridnodes[ix][iy][target].f(2) = -gridnodes[ix][iy][source].f(2);
				gridnodes[ix][iy][target].mass = gridnodes[ix][iy][source].mass;
			}
		}
	}

	if (symmetry_plane_z_minus_exists) {
		// find z grid index corresponding location of z minus symmetry plane
		pz_shifted = symmetry_plane_z_minus_location - min_iz * cellsize + GRID_OFFSET * cellsize;
		iz = icellsize * pz_shifted;

		if ((iz < 0) || (iz >= grid_nz)) {
			printf("z cell index %d is outside range 0 .. %d\n", iz, grid_nz);
			error->one(FLERR, "");
		}

		for (iy = 0; iy < grid_ny; iy++) {
			for (ix = 0; ix < grid_nx; ix++) {

				// set y velocity to zero in the symmetry plane
				gridnodes[ix][iy][iz].v(2) = 0.0;
				gridnodes[ix][iy][iz].vest(2) = 0.0;
				gridnodes[ix][iy][iz].f(2) = 0.0;

				// mirror velocity of nodes on the +-side to the -side

				source = iz - 1;
				target = iz + 1;

				// check that cell indices are within bounds
				if ((source < 0) || (source >= grid_nz)) {
					printf("map from z cell index %d is outside range 0 .. %d\n", source, grid_nz);
					error->one(FLERR, "");
				}

				if ((target < 0) || (target >= grid_nz)) {
					printf("map to z cell index %d is outside range 0 .. %d\n", target, grid_nz);
					error->one(FLERR, "");
				}

				// we duplicate: (jy = iy + 1 -> ky = iy - 1)
				gridnodes[ix][iy][target].v(2) = -gridnodes[ix][iy][source].v(2);
				gridnodes[ix][iy][target].vest(2) = -gridnodes[ix][iy][source].vest(2);
				gridnodes[ix][iy][target].f(2) = -gridnodes[ix][iy][source].f(2);
				gridnodes[ix][iy][target].mass = gridnodes[ix][iy][source].mass;
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
	Vector3d dx, vel_particle;
	Matrix3d Bp;

	for (i = 0; i < nlocal; i++) {

		particleVelocities[i].setZero();
		particleAccelerations[i].setZero();
		particleHeat[i] = 0.0;
		particleHeatRate[i] = 0.0;

		itype = type[i];
		if (setflag[itype][itype]) {

			px_shifted = x[i][0] - min_ix * cellsize + GRID_OFFSET * cellsize;
			py_shifted = x[i][1] - min_iy * cellsize + GRID_OFFSET * cellsize;
			pz_shifted = x[i][2] - min_iz * cellsize + GRID_OFFSET * cellsize;

			ix = icellsize * px_shifted;
			iy = icellsize * py_shifted;
			iz = icellsize * pz_shifted;

			vel_particle << v[i][0], v[i][1], v[i][2];
			Bp.setZero();

			for (jz = iz - STENCIL_LOW; jz < iz + STENCIL_HIGH; jz++) {

				delz_scaled = pz_shifted * icellsize - 1.0 * jz;
				dx(2) = delz_scaled * cellsize;
				delz_scaled_abs = fabs(delz_scaled);
				wfz = DisneyKernel(delz_scaled_abs);
				if (wfz > 0.0) {

					for (jy = iy - STENCIL_LOW; jy < iy + STENCIL_HIGH; jy++) {

						dely_scaled = py_shifted * icellsize - 1.0 * jy;
						dx(1) = dely_scaled * cellsize;
						dely_scaled_abs = fabs(dely_scaled);
						wfy = DisneyKernel(dely_scaled_abs);
						if (wfy > 0.0) {

							for (jx = ix - STENCIL_LOW; jx < ix + STENCIL_HIGH; jx++) {

								delx_scaled = px_shifted * icellsize - 1.0 * jx;
								dx(0) = delx_scaled * cellsize;
								delx_scaled_abs = fabs(delx_scaled);
								wfx = DisneyKernel(delx_scaled_abs);
								if (wfx > 0.0) {

									wf = wfx * wfy * wfz; // this is the total weight function -- a dyadic product of the cartesian weight functions

									particleVelocities[i] += wf * gridnodes[jx][jy][jz].v;

									// NEED TO COMPUTE BP_n+1 here using the updated grid velocities
//									if (gridnodes[jx][jy][jz].isVelocityBC == false) {
//										if (APIC) {
//											Bp += wf * gridnodes[jx][jy][jz].v * dx.transpose();
//										}
//									} else {
//										if (APIC) {
//											Bp += wf * vel_particle * dx.transpose();
//										}
//									}

									if (gridnodes[jx][jy][jz].mass > MASS_CUTOFF) {

										/*
										 * If a grid node acts as constant velocity boundary condition, its acceleration is actually zero,
										 * even though non-zero forces are available on such nodes.
										 * Therefore, normal particles get no acceleration contribution from this particle.
										 * Velocity BC particles, however, accumulate the nodal accelerations for output purposes.
										 * Note that velocity BC particles are time-integrated in a special way such that no use
										 * is made of the particle acceleration data.
										 *
										 */
										if (gridnodes[jx][jy][jz].isVelocityBC) {
											if (mol[i] == 1000) {
												particleAccelerations[i] += wf * gridnodes[jx][jy][jz].f
														/ gridnodes[jx][jy][jz].mass;
											}
										} else {
											particleAccelerations[i] += wf * gridnodes[jx][jy][jz].f / gridnodes[jx][jy][jz].mass;
										}

										particleHeatRate[i] += wf * gridnodes[jx][jy][jz].dheat_dt;
										particleHeat[i] += wf * gridnodes[jx][jy][jz].heat;
									}

									if (wf > 0.0) {
										if (mol[i] == 1000) {
											if ((gridnodes[jx][jy][jz].v - vel_particle).norm() > 1.0e-6) {
												cout << "mol = " << mol[i] << ",  grid BC status is "
														<< gridnodes[jx][jy][jz].isVelocityBC << endl;
												cout << "vel error " << (gridnodes[jx][jy][jz].v - vel_particle).norm() << endl;
												cout << "grid vel is " << gridnodes[jx][jy][jz].v.transpose() << endl;
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
	//error->one(FLERR, "should not be here");
// given the velocity gradient, update the deformation gradient
	double **smd_data_9 = atom->smd_data_9;
	double *vol0 = atom->vfrac;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	int i, itype;
	Matrix3d F, Fincr, eye;
	eye.setIdentity();

	for (i = 0; i < nlocal; i++) {

		itype = type[i];
		if (setflag[itype][itype]) {

			/*
			 * compute deformation gradient using full exponential propagation
			 */

			Fincr = (update->dt * L[i]).exp();
			F = Fincr * Map<Matrix3d>(smd_data_9[i]);
			J[i] = F.determinant();
			vol[i] = vol0[i] * J[i];

			//cout << "this is F after update" << endl << F << endl;

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
	delete[] lgridnodes;

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
		delete[] J;
		J = new double[nmax];
		delete[] vol;
		vol = new double[nmax];
	}

	USF();
}

void PairSmdMpm::USF() {

	CreateGrid();


	timeone_PointstoGrid -= MPI_Wtime();
	PointsToGrid();
	//PointsToGridLinear();
	ApplyVelocityBC();
	timeone_PointstoGrid += MPI_Wtime();
	//DumpGrid();

	timeone_SymmetryBC -= MPI_Wtime();
	ApplySymmetryBC(COPY_VELOCITIES);
	timeone_SymmetryBC += MPI_Wtime();

	timeone_Gradients -= MPI_Wtime();
	ComputeVelocityGradient(); // using current velocities
	//VelocityGradientLinear();
	timeone_Gradients += MPI_Wtime();
	UpdateDeformationGradient();

	timeone_MaterialModel -= MPI_Wtime();
	GetStress();
	AssembleStressTensor();
	comm->forward_comm_pair(this);
	timeone_MaterialModel += MPI_Wtime();

	timeone_GridForces -= MPI_Wtime();
	ComputeGridForces();
	timeone_GridForces += MPI_Wtime();

	timeone_UpdateGrid -= MPI_Wtime();
	UpdateGridVelocities();
	timeone_UpdateGrid += MPI_Wtime();

	timeone_SymmetryBC -= MPI_Wtime();
	ApplySymmetryBC(COPY_VELOCITIES);
	timeone_SymmetryBC += MPI_Wtime();

	timeone_GridToPoints -= MPI_Wtime();
	GridToPoints();
	timeone_GridToPoints += MPI_Wtime();

	DestroyGrid();
}

void PairSmdMpm::USL() {
	CreateGrid();
	PointsToGrid();
	GetStress();
	ComputeGridForces();
	ApplySymmetryBC(COPY_VELOCITIES);
	UpdateGridVelocities();
	GridToPoints();

	AdvanceParticles();
	// -- compute new grid velocities from updated particle positions
	ScatterVelocities();
	ApplySymmetryBC(COPY_VELOCITIES);
	ComputeVelocityGradient();
	AssembleStressTensor();
	AdvanceParticlesEnergy();
	DestroyGrid();
}

void PairSmdMpm::UpdateStress() {
	double **tlsph_stress = atom->smd_stress;
	double *de = atom->de;
	int *type = atom->type;
	Matrix3d D, eye, d_dev, stressRate, oldStress, newStress;
	double d_iso;
	int i, itype;
	int nlocal = atom->nlocal;
	eye.setIdentity();

	dtCFL = BIG;

	for (i = 0; i < nlocal; i++) {
		itype = type[i];
		if (setflag[itype][itype] == 1) {

			D = 0.5 * (L[i] + L[i].transpose());
			d_dev = Deviator(D);
			d_iso = D.trace();

			stressRate = Lookup[BULK_MODULUS][itype] * d_iso * eye + 2.0 * Lookup[SHEAR_MODULUS][itype] * d_dev;
			stressTensor[i] += update->dt * stressRate;

			// store updated stress
			tlsph_stress[i][0] = stressTensor[i](0, 0);
			tlsph_stress[i][1] = stressTensor[i](0, 1);
			tlsph_stress[i][2] = stressTensor[i](0, 2);
			tlsph_stress[i][3] = stressTensor[i](1, 1);
			tlsph_stress[i][4] = stressTensor[i](1, 2);
			tlsph_stress[i][5] = stressTensor[i](2, 2);

			c0[i] = Lookup[REFERENCE_SOUNDSPEED][itype];

			/*
			 * stable timestep based on speed-of-sound
			 */

			dtCFL = MIN(cellsize / c0[i], dtCFL);

			/*
			 * potential energy
			 */

			de[i] += vol[i] * (stressTensor[i].cwiseProduct(D)).sum();
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
	double *rmass = atom->rmass;
	double *eff_plastic_strain = atom->eff_plastic_strain;
	double **tlsph_stress = atom->smd_stress;
	double *heat = atom->heat;
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
	double dt = FACTOR * update->dt;
	double newPressure;
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

			effectiveViscosity = 0.0;
			K_eff = 0.0;
			G_eff = 0.0;
			D = 0.5 * (L[i] + L[i].transpose());

			d_iso = D.trace();

			rho = rmass[i] / vol[i];

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
				PerfectGasEOS(Lookup[EOS_PERFECT_GAS_GAMMA][itype], vol[i], rmass[i], e[i], newPressure, c0[i]);
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
				 * initial stress state has already been assigned before in function GetStress()
				 * we need the deviator here.
				 */
				oldStressDeviator = Deviator(stressTensor[i]);

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

					if (heat[i] > 0.05) {

						yieldStress = Lookup[YIELD_STRENGTH][itype] + Lookup[HARDENING_PARAMETER][itype] * eff_plastic_strain[i];

						LinearPlasticStrength(Lookup[SHEAR_MODULUS][itype], yieldStress, oldStressDeviator, d_dev, dt,
								newStressDeviator, stressRateDev, plastic_strain_increment);
					} else {
						stressRateDev = 2.0 * Lookup[SHEAR_MODULUS][itype] * d_dev;
						plastic_strain_increment = 0.0;
					}

					// this is for MPM extrusion

//					if (x[i][0] > -65.0) {
//						newStressDeviator.setZero();
//						stressRateDev.setZero();
//						plastic_strain_increment = 0.0;
//					}

					eff_plastic_strain[i] += plastic_strain_increment;

					break;
				}

				StressRateDevJaumann = stressRateDev - W * oldStressDeviator + oldStressDeviator * W;
				newStressDeviator = oldStressDeviator + dt * StressRateDevJaumann;

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
			 * assemble updated stress Tensor from pressure and deviatoric parts
			 */
			stressTensor[i] = -newPressure * eye + newStressDeviator;
			//cout << "this is the new stress deviator: " << newStressDeviator << endl;

			/*
			 * store new stress Tensor
			 */
			tlsph_stress[i][0] = stressTensor[i](0, 0);
			tlsph_stress[i][1] = stressTensor[i](0, 1);
			tlsph_stress[i][2] = stressTensor[i](0, 2);
			tlsph_stress[i][3] = stressTensor[i](1, 1);
			tlsph_stress[i][4] = stressTensor[i](1, 2);
			tlsph_stress[i][5] = stressTensor[i](2, 2);

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

			de[i] += FACTOR * vol[i] * (stressTensor[i].cwiseProduct(D)).sum();

		}
		// end if (setflag[itype][itype] == 1)
	} // end loop over nlocal

// fallback if no atoms are present:
	int check_flag = 0;
	for (itype = 1; itype <= atom->ntypes; itype++) {
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
		} else if (strcmp(arg[iarg], "sym_y_+") == 0) {
			symmetry_plane_y_plus_exists = true;

			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected float following sym_y_+ keyword");
			}
			symmetry_plane_y_plus_location = force->numeric(FLERR, arg[iarg]);

			if (comm->me == 0) {
				printf("... +y symmetry plane at y = %f\n", symmetry_plane_y_plus_location);
			}
		} else if (strcmp(arg[iarg], "sym_y_-") == 0) {
			symmetry_plane_y_minus_exists = true;

			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected float following sym_y_- keyword");
			}
			symmetry_plane_y_minus_location = force->numeric(FLERR, arg[iarg]);

			if (comm->me == 0) {
				printf("... -y symmetry plane at y = %f\n", symmetry_plane_y_minus_location);
			}
		} else if (strcmp(arg[iarg], "sym_x_+") == 0) {
			symmetry_plane_x_plus_exists = true;

			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected float following sym_x_+ keyword");
			}
			symmetry_plane_x_plus_location = force->numeric(FLERR, arg[iarg]);

			if (comm->me == 0) {
				printf("... +x symmetry plane at x = %f\n", symmetry_plane_x_plus_location);
			}
		} else if (strcmp(arg[iarg], "sym_x_-") == 0) {
			symmetry_plane_x_minus_exists = true;

			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected float following sym_x_- keyword");
			}
			symmetry_plane_x_minus_location = force->numeric(FLERR, arg[iarg]);

			if (comm->me == 0) {
				printf("... -x symmetry plane at x = %f\n", symmetry_plane_x_minus_location);
			}
		} else if (strcmp(arg[iarg], "sym_z_+") == 0) {
			symmetry_plane_z_plus_exists = true;

			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected float following sym_z_+ keyword");
			}
			symmetry_plane_z_plus_location = force->numeric(FLERR, arg[iarg]);

			if (comm->me == 0) {
				printf("... +z symmetry plane at z = %f\n", symmetry_plane_z_plus_location);
			}
		} else if (strcmp(arg[iarg], "sym_z_-") == 0) {
			symmetry_plane_z_minus_exists = true;

			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected float following sym_z_- keyword");
			}
			symmetry_plane_z_minus_location = force->numeric(FLERR, arg[iarg]);

			if (comm->me == 0) {
				printf("... -z symmetry plane at z = %f\n", symmetry_plane_z_minus_location);
			}
		} else if (strcmp(arg[iarg], "noslip_sym_y_+") == 0) {
			noslip_symmetry_plane_y_plus_exists = true;

			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected float following noslip_sym_y_+ keyword");
			}
			noslip_symmetry_plane_y_plus_location = force->numeric(FLERR, arg[iarg]);

			if (comm->me == 0) {
				printf("... NOSLIP +y symmetry plane at y = %f\n", noslip_symmetry_plane_y_plus_location);
			}
		} else if (strcmp(arg[iarg], "noslip_sym_y_-") == 0) {
			noslip_symmetry_plane_y_minus_exists = true;

			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected float following noslip_sym_y_- keyword");
			}
			noslip_symmetry_plane_y_minus_location = force->numeric(FLERR, arg[iarg]);

			if (comm->me == 0) {
				printf("... NOSLIP_-y symmetry plane at y = %f\n", noslip_symmetry_plane_y_minus_location);
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
	double **f = atom->f;
	int i, j, m;

//printf("packing comm\n");
	m = 0;
	for (i = 0; i < n; i++) {
		j = list[i];
		buf[m++] = vol[j];

		buf[m++] = stressTensor[j](0, 0); // pack symmetric stress tensor
		buf[m++] = stressTensor[j](1, 1);
		buf[m++] = stressTensor[j](2, 2);
		buf[m++] = stressTensor[j](0, 1);
		buf[m++] = stressTensor[j](0, 2);
		buf[m++] = stressTensor[j](1, 2); // 2 + 6 = 8

		buf[m++] = f[j][0];
		buf[m++] = f[j][1];
		buf[m++] = f[j][2];

	}
	return m;
}

/* ---------------------------------------------------------------------- */

void PairSmdMpm::unpack_forward_comm(int n, int first, double *buf) {
	double **f = atom->f;
	int i, m, last;

	m = 0;
	last = first + n;
	for (i = first; i < last; i++) {
		vol[i] = buf[m++];

		stressTensor[i](0, 0) = buf[m++];
		stressTensor[i](1, 1) = buf[m++];
		stressTensor[i](2, 2) = buf[m++];
		stressTensor[i](0, 1) = buf[m++];
		stressTensor[i](0, 2) = buf[m++];
		stressTensor[i](1, 2) = buf[m++]; // 2 + 6 = 8
		stressTensor[i](1, 0) = stressTensor[i](0, 1);
		stressTensor[i](2, 0) = stressTensor[i](0, 2);
		stressTensor[i](2, 1) = stressTensor[i](1, 2);

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

void PairSmdMpm::ComputeGridGradients() {
	int ix, iy, iz;
	for (ix = 1; ix < grid_nx - 1; ix++) {
		for (iy = 1; iy < grid_ny - 1; iy++) {
			for (iz = 1; iz < grid_nz - 1; iz++) {

				//if (gridnodes[ix][iy][iz].mass > MASS_CUTOFF) {
				// central differences

//				if (gridnodes[ix][iy][iz].mass > MASS_CUTOFF) {
//
//					if ((gridnodes[ix + 1][iy][iz].mass > MASS_CUTOFF) && (gridnodes[ix - 1][iy][iz].mass > MASS_CUTOFF)) {
//						gridnodes[ix][iy][iz].Lxx = 0.5 * icellsize
//								* (gridnodes[ix + 1][iy][iz].v(0) - gridnodes[ix - 1][iy][iz].v(0));
//					}
//
//					gridnodes[ix][iy][iz].Lyy = icellsize
//							* (gridnodes[ix][iy + 1][iz].v(1) - 2.0 * gridnodes[ix][iy][iz].v(1) + gridnodes[ix][iy - 1][iz].v(1));
//
//					gridnodes[ix][iy][iz].Lxy = icellsize
//							* (gridnodes[ix + 1][iy][iz].v(1) - 2.0 * gridnodes[ix][iy][iz].v(1) + gridnodes[ix - 1][iy][iz].v(1));
//					gridnodes[ix][iy][iz].Lyx = icellsize
//							* (gridnodes[ix][iy + 1][iz].v(0) - 2.0 * gridnodes[ix][iy][iz].v(0) + gridnodes[ix][iy - 1][iz].v(0));
//
//				}

			}
		}
	}
}

void PairSmdMpm::GridGradientsToParticles() {
	double **x = atom->x;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	int i, itype;
	int ix, iy, iz, jx, jy, jz;
	double px_shifted, py_shifted, pz_shifted; // shifted coords of particles
	double delx_scaled, delx_scaled_abs, dely_scaled, dely_scaled_abs, wfx, wfy, wf;
	double delz_scaled, delz_scaled_abs, wfz;

	for (i = 0; i < nlocal; i++) {

		L[i].setZero();

		itype = type[i];
		if (setflag[itype][itype]) {

			px_shifted = x[i][0] - min_ix * cellsize + GRID_OFFSET * cellsize;
			py_shifted = x[i][1] - min_iy * cellsize + GRID_OFFSET * cellsize;
			pz_shifted = x[i][2] - min_iz * cellsize + GRID_OFFSET * cellsize;

			ix = icellsize * px_shifted;
			iy = icellsize * py_shifted;
			iz = icellsize * pz_shifted;

			for (jz = iz - STENCIL_LOW; jz < iz + STENCIL_HIGH; jz++) {

				delz_scaled = pz_shifted * icellsize - 1.0 * jz;
				delz_scaled_abs = fabs(delz_scaled);
				wfz = DisneyKernel(delz_scaled_abs);
				if (wfz > 0.0) {

					for (jy = iy - STENCIL_LOW; jy < iy + STENCIL_HIGH; jy++) {

						dely_scaled = py_shifted * icellsize - 1.0 * jy;
						dely_scaled_abs = fabs(dely_scaled);
						wfy = DisneyKernel(dely_scaled_abs);
						if (wfy > 0.0) {

							for (jx = ix - STENCIL_LOW; jx < ix + STENCIL_HIGH; jx++) {

								delx_scaled = px_shifted * icellsize - 1.0 * jx;
								delx_scaled_abs = fabs(delx_scaled);
								wfx = DisneyKernel(delx_scaled_abs);
								if (wfx > 0.0) {

									wf = wfx * wfy * wfz; // this is the total weight function -- a dyadic product of the cartesian weight functions

//									L[i](0, 0) += wf * gridnodes[jx][jy][jz].Lxx;
//									L[i](1, 1) += wf * gridnodes[jx][jy][jz].Lyy;
//									L[i](0, 1) += wf * gridnodes[jx][jy][jz].Lxy;
//									L[i](1, 0) += wf * gridnodes[jx][jy][jz].Lyx;
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
 * do time integration of particles
 */

void PairSmdMpm::AdvanceParticles() {

	int nlocal = atom->nlocal;
	int i;
	double **x = atom->x;
	double **v = atom->v;

	for (i = 0; i < nlocal; i++) {
		v[i][0] += update->dt * particleAccelerations[i](0);
		v[i][1] += update->dt * particleAccelerations[i](1);
		v[i][2] += update->dt * particleAccelerations[i](2);

		x[i][0] += update->dt * particleVelocities[i](0);
		x[i][1] += update->dt * particleVelocities[i](1);
		x[i][2] += update->dt * particleVelocities[i](2);

	}

}

void PairSmdMpm::AdvanceParticlesEnergy() {

	int nlocal = atom->nlocal;
	int i;
	double *e = atom->e;
	double *de = atom->de;

	for (i = 0; i < nlocal; i++) {
		e[i] += 0.5 * update->dt * de[i];
	}

}

void PairSmdMpm::DumpGrid() {

	printf("... dumping grid\n");

	int count = 0;
	for (int ix = 0; ix < grid_nx; ix++) {
		for (int iy = 0; iy < grid_ny; iy++) {
			for (int iz = 0; iz < grid_nz; iz++) {
				int icell = ix + iy * grid_nx + iz * grid_nx * grid_ny;
				if (lgridnodes[icell].mass > MASS_CUTOFF) {
					count++;
				}
			}

		}
	}

	FILE * f;
	f = fopen("grid.dump", "w");

	fprintf(f, "%d\n\n", count);

	for (int ix = 0; ix < grid_nx; ix++) {
		for (int iy = 0; iy < grid_ny; iy++) {
			for (int iz = 0; iz < grid_nz; iz++) {
				int icell = ix + iy * grid_nx + iz * grid_nx * grid_ny;
				if (lgridnodes[icell].mass > MASS_CUTOFF) {
					fprintf(f, "X %f %f %f %g\n", ix * cellsize, iy * cellsize, iz * cellsize, lgridnodes[icell].mass);
				}
			}

		}
	}

	fclose(f);

}
