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

#include "pair_smd_mpm_linear.h"

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
#include "region.h"
#include <stdio.h>
#include <iostream>
#include "smd_material_models.h"
#include "smd_math.h"
#include "smd_kernels.h"
#include <unsupported/Eigen/MatrixFunctions>
#include "SmdMatDB.h"

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
#define MASS_CUTOFF 1.0e-12
#define FACTOR 1

enum {
	Xplus, Yplus, Zplus, Xminus, Yminus, Zminus, STAGE1, STAGE2
};

PairSmdMpmLin::PairSmdMpmLin(LAMMPS *lmp) :
		Pair(lmp) {

	iproc = comm->me;
	J = NULL;
	vol = NULL;

	nmax = 0; // make sure no atom on this proc such that initial memory allocation is correct
	stressTensor = L = F = R = NULL;
	particleVelocities = particleAccelerations = NULL;
	heat_gradient = NULL;
	particleHeat = particleHeatRate = NULL;
	neighs = NULL;

	comm_forward = 10; // this pair style communicates 16 doubles to ghost atoms

	timeone_PointstoGrid = timeone_Gradients = timeone_MaterialModel = timeone_GridForces = timeone_UpdateGrid =
			timeone_GridToPoints = 0.0;
	timeone_SymmetryBC = timeone_Comm = timeone_UpdateParticles = 0.0;

	symmetry_plane_y_plus_exists = symmetry_plane_y_minus_exists = false;
	symmetry_plane_x_plus_exists = symmetry_plane_x_minus_exists = false;
	symmetry_plane_z_plus_exists = symmetry_plane_z_minus_exists = false;
	noslip_symmetry_plane_y_plus_exists = noslip_symmetry_plane_y_minus_exists = false;

	int retcode = matDB.ReadMaterials(atom->ntypes);
	if (retcode < 0) {
		error->one(FLERR, "failed to read material database");
	}
	if (iproc == 0) {
		matDB.PrintData();
	}
	corotated = false;
	true_deformation = false;

}

/* ---------------------------------------------------------------------- */

PairSmdMpmLin::~PairSmdMpmLin() {

	double time_PointstoGrid, time_Gradients, time_MaterialModel, time_GridForces, time_UpdateGrid, time_GridToPoints,
			time_SymmetryBC, time_UpdateParticles, time_Comm;

	MPI_Allreduce(&timeone_PointstoGrid, &time_PointstoGrid, 1, MPI_DOUBLE, MPI_SUM, world);
	MPI_Allreduce(&timeone_Gradients, &time_Gradients, 1, MPI_DOUBLE, MPI_SUM, world);
	MPI_Allreduce(&timeone_MaterialModel, &time_MaterialModel, 1, MPI_DOUBLE, MPI_SUM, world);
	MPI_Allreduce(&timeone_GridForces, &time_GridForces, 1, MPI_DOUBLE, MPI_SUM, world);
	MPI_Allreduce(&timeone_UpdateGrid, &time_UpdateGrid, 1, MPI_DOUBLE, MPI_SUM, world);
	MPI_Allreduce(&timeone_GridToPoints, &time_GridToPoints, 1, MPI_DOUBLE, MPI_SUM, world);
	MPI_Allreduce(&timeone_SymmetryBC, &time_SymmetryBC, 1, MPI_DOUBLE, MPI_SUM, world);
	MPI_Allreduce(&timeone_UpdateParticles, &time_UpdateParticles, 1, MPI_DOUBLE, MPI_SUM, world);
	MPI_Allreduce(&timeone_Comm, &time_Comm, 1, MPI_DOUBLE, MPI_SUM, world);

	double time_sum = time_PointstoGrid + time_Gradients + time_MaterialModel + time_GridForces + time_UpdateGrid
			+ time_GridToPoints + time_SymmetryBC + time_UpdateParticles + time_Comm;

	if (iproc == 0) {
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
		printf("%20s is %10.2f seconds, %3.2f percent of pair time\n", "particle update", time_UpdateParticles,
				100 * time_UpdateParticles / time_sum);
		printf("%20s is %10.2f seconds, %3.2f percent of pair time\n", "grid to points", time_GridToPoints,
				100 * time_GridToPoints / time_sum);
		printf("%20s is %10.2f seconds, %3.2f percent of pair time\n", "symmetry BC", time_SymmetryBC,
				100 * time_SymmetryBC / time_sum);
		printf("%20s is %10.2f seconds, %3.2f percent of pair time\n", "communication", time_Comm, 100 * time_Comm / time_sum);
		printf(">>========>>========>>========>>========>>========>>========>>========>>========\n\n");
	}

	if (allocated) {
		//printf("... deallocating\n");
		delete[] stressTensor;
		delete[] L;
		delete[] F;
		delete[] particleVelocities;
		delete[] particleAccelerations;
		delete[] heat_gradient;
		delete[] particleHeatRate;
		delete[] particleHeat;
		delete[] J;
		delete[] vol;
		delete[] neighs;

	}

}

/* ---------------------------------------------------------------------- */

void PairSmdMpmLin::CreateGrid() {
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
//	minx = domain->sublo[0];
//	miny = domain->sublo[1];
//	minz = domain->sublo[2];
//	maxx = domain->subhi[0];
//	maxy = domain->subhi[1];
//	maxz = domain->subhi[2];
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

	if (symmetry_plane_y_plus_exists) {
		//if (miny < symmetry_plane_y_plus_location - 0.1 * cellsize) {
//			error->one(FLERR, "Cannot have particle or box boundary below y+ symmetry plane.");
		//}
		//miny = symmetry_plane_y_plus_location;
	}

	if (symmetry_plane_y_minus_exists) {
		if (maxy > symmetry_plane_y_minus_location + 0.1 * cellsize) {
			printf("maxy is %f, symm plane is at %f\n", maxy, symmetry_plane_y_minus_location + 0.1 * cellsize);
			error->one(FLERR, "Cannot have particle or box boundary above y- symmetry plane.");
		}
		//maxy = symmetry_plane_y_minus_location;
	}

	if (symmetry_plane_x_plus_exists) {
		if (minx < symmetry_plane_x_plus_location - 0.1 * cellsize) {
			error->one(FLERR, "Cannot have particle or box boundary below x+ symmetry plane.");
		}
		//minx = symmetry_plane_x_plus_location;
	}

	if (symmetry_plane_x_minus_exists) {
		if (maxx > symmetry_plane_x_minus_location + 0.1 * cellsize) {
			printf("maxx=%f\n", maxx);
			error->one(FLERR, "Cannot have particle or box boundary above x- symmetry plane.");
		}
		//maxx = symmetry_plane_x_minus_location;
	}

	if (symmetry_plane_z_plus_exists) {
		if (minz < symmetry_plane_z_plus_location - 0.1 * cellsize) {
			error->warning(FLERR, "Cannot have particle or box boundary below z+ symmetry plane.");
		}
		//minz = symmetry_plane_z_plus_location;
	}

	if (symmetry_plane_z_minus_exists) {
		if (maxz > symmetry_plane_z_minus_location + 0.1 * cellsize) {
			error->one(FLERR, "Cannot have particle or box boundary above z- symmetry plane.");
		}
		//maxz = symmetry_plane_z_minus_location;
	}

	// we want the leftmost index to be 0, i.e. index(minx - kernel bandwidth > 0
	// to this end, we assume that the kernel does not cover more than three cells to either side

	minix = static_cast<int>(icellsize * minx) - 3;
	miniy = static_cast<int>(icellsize * miny) - 3;
	miniz = static_cast<int>(icellsize * minz) - 3;
	//printf("proc %d: ih*minx=%f, ih*miny=%f, ih*minz=%f\n", comm->me, icellsize * minx, icellsize * miny, icellsize * minz);
	//printf("proc %d: minix=%d, miniy=%d, miniz=%d\n", comm->me, minix, miniy, miniz);

	maxix = static_cast<int>(icellsize * maxx);
	maxiy = static_cast<int>(icellsize * maxy);
	maxiz = static_cast<int>(icellsize * maxz);
	//printf("proc %d: ih*maxx=%f, ih*maxy=%f, ih*maxz=%f\n", comm->me, icellsize * maxx, icellsize * maxy, icellsize * maxz);
	//printf("proc %d: maxix=%d, maxiy=%d, maxiz=%d\n", comm->me, maxix, maxiy, maxiz);

	grid_nx = (maxix - minix) + 4;
	grid_ny = (maxiy - miniy) + 4;
	grid_nz = (maxiz - miniz) + 4;

	grid_nz = MAX(grid_nz, 4);

	// allocate grid storage

	//printf("proc %d: minx=%f, miny=%f, minz=%f\n", comm->me, minx, miny, minz);
//	printf("proc %d: nx=%f, ny=%f, nz=%f\n", comm->me, maxx, maxy, maxz);
//	printf("proc %d: nx=%d, ny=%d, nz=%d\n", comm->me, grid_nx, grid_ny, grid_nz);

	Ncells = grid_nx * grid_ny * grid_nz;
	lgridnodes = NULL;
	lgridnodes = new Gridnode[Ncells];

}

// use a linear array for grid nodes

void PairSmdMpmLin::PointsToGrid() {
	double **v = atom->v;
	double **f = atom->f;
	double *rmass = atom->rmass;
	double *heat = atom->heat;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	int nall = nlocal + atom->nghost;
	int ref_node;
	double wfx[4], wfy[4], wfz[4];
	Vector3d vel_particle, otherForces;

	for (int icell = 0; icell < Ncells; icell++) {
		lgridnodes[icell].mass = 0.0;
		lgridnodes[icell].heat = 0.0;
		lgridnodes[icell].dheat_dt = 0.0;
		lgridnodes[icell].f.setZero();
		lgridnodes[icell].v.setZero();
		lgridnodes[icell].fbody.setZero();
		lgridnodes[icell].isVelocityBC = false;
	}

	for (int i = 0; i < nall; i++) {

		int itype = type[i];
		if (setflag[itype][itype]) {

			// pre-compute all possible quantities
			double particle_mass = rmass[i];
			double particle_heat = particle_mass * heat[i];
			vel_particle << v[i][0], v[i][1], v[i][2];
			vel_particle *= particle_mass;
			otherForces << f[i][0], f[i][1], f[i][2];

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
						lgridnodes[node_index].fbody += wf * otherForces;
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
			lgridnodes[icell].imass = 1.0 / lgridnodes[icell].mass;
			lgridnodes[icell].v *= lgridnodes[icell].imass;
			lgridnodes[icell].heat *= lgridnodes[icell].imass;
		}
	}

}

void PairSmdMpmLin::VelocitiesToGrid() {
	double **v = atom->v;
	double *rmass = atom->rmass;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	int nall = nlocal + atom->nghost;
	int ref_node;
	double wfx[4], wfy[4], wfz[4];
	Vector3d vel_particle;

	for (int icell = 0; icell < Ncells; icell++) {
		lgridnodes[icell].count = 0;
		lgridnodes[icell].mass = 0.0;
		lgridnodes[icell].v.setZero();
	}

	for (int i = 0; i < nall; i++) {

		int itype = type[i];
		if (setflag[itype][itype]) {

			// pre-compute all possible quantities
			double particle_mass = rmass[i];
			if (true_deformation) {
				// particles move with particleVelocities, so it should be more true to use this
				// velocitz fore computing acceleration. However, this leads to particle disorder
				// for elastic deformations and should only be used for pure fluid simulations
				vel_particle = particleVelocities[i];
			} else {
				// this is better for elastic deformations
				vel_particle << v[i][0], v[i][1], v[i][2];
			}
			vel_particle *= particle_mass;

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
						lgridnodes[node_index].mass += wf * particle_mass;
						lgridnodes[node_index].count++;
					}
				}
			}

		} // end if setflag
	} // end loop over nall

	// normalize all grid cells
	for (int icell = 0; icell < Ncells; icell++) {
		if (lgridnodes[icell].mass > MASS_CUTOFF) {
			lgridnodes[icell].imass = 1.0 / lgridnodes[icell].mass;
			lgridnodes[icell].v *= lgridnodes[icell].imass;
		}
	}

}

void PairSmdMpmLin::PreComputeGridWeights(const int i, int &ref_node, double *wfx, double *wfy, double *wfz) {
	double **x = atom->x;

	double ssx = icellsize * x[i][0] - static_cast<double>(minix); // shifted position in grid coords
	double ssy = icellsize * x[i][1] - static_cast<double>(miniy); // shifted position in grid coords
	double ssz = icellsize * x[i][2] - static_cast<double>(miniz); // shifted position in grid coords

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

void PairSmdMpmLin::PreComputeGridWeightsAndDerivatives(const int i, int &ref_node, double *wfx, double *wfy, double *wfz,
		double *wfdx, double *wfdy, double *wfdz) {
	double **x = atom->x;

	double ssx = icellsize * x[i][0] - static_cast<double>(minix); // shifted position in grid coords
	double ssy = icellsize * x[i][1] - static_cast<double>(miniy); // shifted position in grid coords
	double ssz = icellsize * x[i][2] - static_cast<double>(miniz); // shifted position in grid coords

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

void PairSmdMpmLin::ApplyVelocityBC() {
//	double **x = atom->x;
//	double **v = atom->v;
//	double *heat = atom->heat;
//	int *type = atom->type;
//	tagint *mol = atom->molecule;
//	int nlocal = atom->nlocal;
//	int nall = nlocal + atom->nghost;
//	int i, itype;
//	int ix, iy, iz, jx, jy, jz;
//	double delx_scaled, delx_scaled_abs, dely_scaled, dely_scaled_abs, wf, wfx, wfy;
//	double delz_scaled, delz_scaled_abs, wfz, heat_particle;
//	double px_shifted, py_shifted, pz_shifted; // shifted coords of particles
//	Vector3d vel_particle;
//
//	for (i = 0; i < nall; i++) {
//
//		itype = type[i];
//		if (setflag[itype][itype]) {
//
//			/*
//			 * only do this for boundary particles which have mol id = 1000
//			 */
//
//			if (mol[i] == 1000) {
//
//				//if (APIC) {
//				//	error->one(FLERR, "Cannot use APIC with velocity boundary condtions\n");
//				//}
//
//				px_shifted = x[i][0] - min_ix * cellsize + GRID_OFFSET * cellsize;
//				py_shifted = x[i][1] - min_iy * cellsize + GRID_OFFSET * cellsize;
//				pz_shifted = x[i][2] - min_iz * cellsize + GRID_OFFSET * cellsize;
//
//				ix = icellsize * px_shifted;
//				iy = icellsize * py_shifted;
//				iz = icellsize * pz_shifted;
//
//				vel_particle << v[i][0], v[i][1], v[i][2];
//				heat_particle = heat[i];
//
//				for (jx = ix - STENCIL_LOW; jx < ix + STENCIL_HIGH; jx++) {
//					delx_scaled = px_shifted * icellsize - 1.0 * jx;
//					delx_scaled_abs = fabs(delx_scaled);
//					wfx = DisneyKernel(delx_scaled_abs);
//
//					for (jy = iy - STENCIL_LOW; jy < iy + STENCIL_HIGH; jy++) {
//						dely_scaled = py_shifted * icellsize - 1.0 * jy;
//						dely_scaled_abs = fabs(dely_scaled);
//						wfy = DisneyKernel(dely_scaled_abs);
//
//						for (jz = iz - STENCIL_LOW; jz < iz + STENCIL_HIGH; jz++) {
//							delz_scaled = pz_shifted * icellsize - 1.0 * jz;
//							delz_scaled_abs = fabs(delz_scaled);
//							wfz = DisneyKernel(delz_scaled_abs);
//
//							wf = wfx * wfy * wfz; // this is the total weight function -- a dyadic product of the cartesian weight functions
//
//							if (wf > 0.0) {
//								//gridnodes[jx][jy][jz].heat = heat_particle;
//								gridnodes[jx][jy][jz].v = vel_particle;
//								gridnodes[jx][jy][jz].vest = vel_particle;
//								gridnodes[jx][jy][jz].isVelocityBC = true;
//							}
//
//						}
//
//					}
//				}
//			} // end if mol = 1000
//		} // end if setflag[itype][itype]
//	}

}

void PairSmdMpmLin::ComputeVelocityGradient() {
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

			int this_count = 0;
			velocity_gradient.setZero();
			PreComputeGridWeightsAndDerivatives(i, ref_node, wfx, wfy, wfz, wfdx, wfdy, wfdz);

			// loop over all cell neighbors for this particle
			for (int ix = 0; ix < 4; ix++) {
				for (int iy = 0; iy < 4; iy++) {
					for (int iz = 0; iz < 4; iz++) {

						g(0) = wfdx[ix] * wfy[iy] * wfz[iz]; // this is the kernel gradient
						g(1) = wfdy[iy] * wfx[ix] * wfz[iz];
						g(2) = wfdz[iz] * wfx[ix] * wfy[iy];

						int node_index = ref_node + ix + iy * grid_nx + iz * grid_nx * grid_ny;
						velocity_gradient += lgridnodes[node_index].v * g.transpose();
						this_count = MAX(this_count, lgridnodes[node_index].count);
					}
				}
			}

			L[i] = -velocity_gradient;
			neighs[i] = this_count;

			if (neighs[i] == 1) {
				/*
				 * self-contribution due to expansion
				 */
				double pressure = -stressTensor[i].trace() / 3.0;
				double c0 = pressure / matDB.gProps[itype].rho0;
				double expansionrate = 1.0e-1 * c0 * icellsize;
				//printf("self expansion rate is %g\n", expansionrate);
				L[i](0, 0) = expansionrate;
				L[i](1, 1) = expansionrate;
				if (flag3d) {
					L[i](0, 0) = expansionrate;
				}
			}
		} // end if setflag
	} // end loop over nall
}

// ---- end velocity gradients ----

void PairSmdMpmLin::ComputeGridForces() {
	int *type = atom->type;
	int nall = atom->nlocal + atom->nghost;
	int i, itype, ref_node;
	double wfx[4], wfy[4], wfz[4], wfdx[4], wfdy[4], wfdz[4];
	Vector3d g;
	Matrix3d scaledStress;

// ---- compute internal forces ---
	for (i = 0; i < nall; i++) {

		itype = type[i];
		if (setflag[itype][itype]) {

			scaledStress = vol[i] * stressTensor[i];
			PreComputeGridWeightsAndDerivatives(i, ref_node, wfx, wfy, wfz, wfdx, wfdy, wfdz);

			// loop over all cell neighbors for this particle
			for (int iz = 0; iz < 4; iz++) {
				for (int iy = 0; iy < 4; iy++) {
					for (int ix = 0; ix < 4; ix++) {

						g(0) = wfdx[ix] * wfy[iy] * wfz[iz]; // this is the kernel gradient
						g(1) = wfdy[iy] * wfx[ix] * wfz[iz];
						g(2) = wfdz[iz] * wfx[ix] * wfy[iy];

						int node_index = ref_node + ix + iy * grid_nx + iz * grid_nx * grid_ny;
						lgridnodes[node_index].f += scaledStress * g;
					}
				}
			}
		} // end if (setflag[itype][itype])
	} // end loop over nall particles
}

/*
 * update grid velocities using grid forces
 */

void PairSmdMpmLin::UpdateGridVelocities() {
	double dtm;
	double dt = FACTOR * update->dt;

	for (int icell = 0; icell < Ncells; icell++) {
		if (lgridnodes[icell].mass > MASS_CUTOFF) {
			dtm = dt * lgridnodes[icell].imass;
			lgridnodes[icell].v += dtm * (lgridnodes[icell].f + lgridnodes[icell].fbody);
			lgridnodes[icell].heat += dt * lgridnodes[icell].dheat_dt;
		}
	}

}

void PairSmdMpmLin::GridToPoints() {
	double **f = atom->f;
	double *rmass = atom->rmass;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	int ref_node, node_index;
	double wfx[4], wfy[4], wfz[4];
	double wf;
	Vector3d pv, pa;

	for (int i = 0; i < nlocal; i++) {

		int itype = type[i];
		if (setflag[itype][itype]) {

			pv.setZero();
			pa.setZero();
			double phr = 0.0;
			double ph = 0.0;
			PreComputeGridWeights(i, ref_node, wfx, wfy, wfz);

			// loop over all cell neighbors for this particle
			for (int ix = 0; ix < 4; ix++) {
				for (int iy = 0; iy < 4; iy++) {
					for (int iz = 0; iz < 4; iz++) {

						node_index = ref_node + ix + iy * grid_nx + iz * grid_nx * grid_ny;
						if (lgridnodes[node_index].mass > MASS_CUTOFF) {

							wf = wfx[ix] * wfy[iy] * wfz[iz]; // total weight function
							pv += wf * lgridnodes[node_index].v;
							pa += wf * lgridnodes[node_index].imass * (lgridnodes[node_index].f + lgridnodes[node_index].fbody);
							phr += wf * lgridnodes[node_index].dheat_dt;
							ph += wf * lgridnodes[node_index].heat;
						}

					}
				}
			}

			particleVelocities[i] = pv;
			particleAccelerations[i] = pa;
			particleHeatRate[i] = phr;
			particleHeat[i] = ph;

			f[i][0] = rmass[i] * particleAccelerations[i](0);
			f[i][1] = rmass[i] * particleAccelerations[i](1);
			f[i][2] = rmass[i] * particleAccelerations[i](2);
		} // end if setflag

	} // end loop over nlocal

}

/* ---------------------------------------------------------------------- */
void PairSmdMpmLin::UpdateDeformationGradient() {
	double **smd_data_9 = atom->smd_data_9;
	double *vol0 = atom->vfrac;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	int i, itype;
	Matrix3d F, Fincr, eye, U;
	bool status;
	eye.setIdentity();
	Affine3d T;

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

			if (corotated) {
				/*
				 * perform polar decomposition
				 */
//				T = F;
//				R[i] = T.rotation();
				status = PolDec(F, R[i], U, false); // polar decomposition of the deformation gradient, F = R * U
				if (!status) {
					error->message(FLERR, "Polar decomposition of deformation gradient failed.\n");
				}

			} else {
				R[i].setIdentity();
			}

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

void PairSmdMpmLin::DestroyGrid() {

	delete[] lgridnodes;

}

/* ---------------------------------------------------------------------- */

void PairSmdMpmLin::compute(int eflag, int vflag) {

	if (eflag || vflag)
		ev_setup(eflag, vflag);
	else
		evflag = vflag_fdotr = 0;

	if (atom->nmax > nmax) {
//printf("... allocating in compute with nmax = %d\n", atom->nmax);
		nmax = atom->nmax;
		delete[] stressTensor;
		stressTensor = new Matrix3d[nmax];
		delete[] L;
		L = new Matrix3d[nmax];
		delete[] F;
		F = new Matrix3d[nmax];
		delete[] R;
		R = new Matrix3d[nmax];
		delete[] particleVelocities;
		particleVelocities = new Vector3d[nmax];
		delete[] particleAccelerations;
		particleAccelerations = new Vector3d[nmax];
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
		delete[] neighs;
		neighs = new int[nmax];
	}

	MUSL();
}

void PairSmdMpmLin::USF() {
	CreateGrid();

	timeone_PointstoGrid -= MPI_Wtime();
	PointsToGrid();
	ComputeHeatGradientOnGrid();
	timeone_PointstoGrid += MPI_Wtime();

	timeone_SymmetryBC -= MPI_Wtime();
	CheckSymmetryBC();
	timeone_SymmetryBC += MPI_Wtime();

	timeone_Gradients -= MPI_Wtime();
	ComputeVelocityGradient();
	timeone_Gradients += MPI_Wtime();

	timeone_MaterialModel -= MPI_Wtime();
	UpdateDeformationGradient();
	UpdateStress();
	timeone_MaterialModel += MPI_Wtime();

	timeone_Comm -= MPI_Wtime();
	comm->forward_comm_pair(this); // need to have stress tensor on ghosts
	timeone_Comm += MPI_Wtime();

	timeone_GridForces -= MPI_Wtime();
	ComputeGridForces();
	timeone_GridForces += MPI_Wtime();

	timeone_SymmetryBC -= MPI_Wtime();
	CheckSymmetryBC();
	timeone_SymmetryBC += MPI_Wtime();

	timeone_UpdateGrid -= MPI_Wtime();
	UpdateGridVelocities(); // full step
	timeone_UpdateGrid += MPI_Wtime();

	timeone_SymmetryBC -= MPI_Wtime();
	CheckSymmetryBC();
	timeone_SymmetryBC += MPI_Wtime();

	timeone_GridToPoints -= MPI_Wtime();
	GridToPoints();
	timeone_GridToPoints += MPI_Wtime();

	timeone_UpdateParticles -= MPI_Wtime();
	AdvanceParticles(); // full step
	AdvanceParticlesEnergy();
	timeone_UpdateParticles += MPI_Wtime();

	DestroyGrid();
}

void PairSmdMpmLin::MUSL() {
	CreateGrid();

	timeone_PointstoGrid -= MPI_Wtime();
	PointsToGrid();
	ComputeHeatGradientOnGrid();
	timeone_PointstoGrid += MPI_Wtime();

	timeone_SymmetryBC -= MPI_Wtime();
	CheckSymmetryBC();
	timeone_SymmetryBC += MPI_Wtime();

	timeone_Comm -= MPI_Wtime();
	GetStress();
	comm_mode = STAGE1;
	comm->forward_comm_pair(this); // need to have stress tensor and other forces on ghosts
	timeone_Comm += MPI_Wtime();

	timeone_GridForces -= MPI_Wtime();
	ComputeGridForces();
	timeone_GridForces += MPI_Wtime();

	timeone_SymmetryBC -= MPI_Wtime();
	CheckSymmetryBC();
	timeone_SymmetryBC += MPI_Wtime();

	timeone_UpdateGrid -= MPI_Wtime();
	UpdateGridVelocities(); // full step
	timeone_UpdateGrid += MPI_Wtime();

	timeone_SymmetryBC -= MPI_Wtime();
	CheckSymmetryBC();
	timeone_SymmetryBC += MPI_Wtime();

	timeone_GridToPoints -= MPI_Wtime();
	GridToPoints();
	timeone_GridToPoints += MPI_Wtime();

	timeone_UpdateParticles -= MPI_Wtime();
	AdvanceParticles(); // full step
	timeone_UpdateParticles += MPI_Wtime();

	timeone_Comm -= MPI_Wtime();
	comm_mode = STAGE2;
	comm->forward_comm_pair(this); // need to have updated velocities and positions on ghosts
	timeone_Comm += MPI_Wtime();

	timeone_GridToPoints -= MPI_Wtime();
	VelocitiesToGrid(); // This is the M in MUSL -- scatter updated velocities to grid
	timeone_GridToPoints += MPI_Wtime();

	timeone_SymmetryBC -= MPI_Wtime();
	CheckSymmetryBC();
	timeone_SymmetryBC += MPI_Wtime();

	timeone_Gradients -= MPI_Wtime();
	ComputeVelocityGradient();
	timeone_Gradients += MPI_Wtime();

	timeone_MaterialModel -= MPI_Wtime();
	UpdateDeformationGradient();
	UpdateStress();
	timeone_MaterialModel += MPI_Wtime();

	timeone_UpdateParticles -= MPI_Wtime();
	AdvanceParticlesEnergy();
	timeone_UpdateParticles += MPI_Wtime();

	//DumpGrid();

	DestroyGrid();
}

void PairSmdMpmLin::GetStress() {
	double **elastic_stress = atom->smd_stress;
	double **smd_visc_stress = atom->smd_visc_stress;
	double **smd_data_9 = atom->smd_data_9;
	double *vol0 = atom->vfrac;
	int *type = atom->type;
	int i, itype;
	int nlocal = atom->nlocal;
	Matrix3d F, U;
	Affine3d T;
	bool status;

	for (i = 0; i < nlocal; i++) {
		itype = type[i];
		if (setflag[itype][itype] == 1) {
			stressTensor[i](0, 0) = elastic_stress[i][0] + smd_visc_stress[i][0];
			stressTensor[i](0, 1) = elastic_stress[i][1] + smd_visc_stress[i][1];
			stressTensor[i](0, 2) = elastic_stress[i][2] + smd_visc_stress[i][2];
			stressTensor[i](1, 1) = elastic_stress[i][3] + smd_visc_stress[i][3];
			stressTensor[i](1, 2) = elastic_stress[i][4] + smd_visc_stress[i][4];
			stressTensor[i](2, 2) = elastic_stress[i][5] + smd_visc_stress[i][5];

			stressTensor[i](1, 0) = stressTensor[i](0, 1);
			stressTensor[i](2, 0) = stressTensor[i](0, 2);
			stressTensor[i](2, 1) = stressTensor[i](1, 2);

			F = Map<Matrix3d>(smd_data_9[i]);
			J[i] = F.determinant();
			vol[i] = vol0[i] * J[i];

			if (corotated) {
				/*
				 * perform polar decomposition
				 */
//				T = F;
//				R[i] = T.rotation();
				status = PolDec(F, R[i], U, false); // polar decomposition of the deformation gradient, F = R * U
				if (!status) {
					error->message(FLERR, "Polar decomposition of deformation gradient failed.\n");
				}

				stressTensor[i] = (R[i] * stressTensor[i] * R[i].transpose()).eval();

			} else {
				R[i].setIdentity();
			}
		}
	}
}

/* ----------------------------------------------------------------------
 Assemble total stress tensor with pressure, material strength, and
 viscosity contributions.
 ------------------------------------------------------------------------- */
void PairSmdMpmLin::UpdateStress() {
	double *rmass = atom->rmass;
	double *eff_plastic_strain = atom->eff_plastic_strain;
	double **tlsph_stress = atom->smd_stress;
	double **smd_visc_stress = atom->smd_visc_stress;
	double *heat = atom->heat;
	double *de = atom->de;
//double **x = atom->x;
	int *type = atom->type;
	int i, itype;
	int nlocal = atom->nlocal;
	Matrix3d E, D, Ddev, W, V, sigma_diag;
	Matrix3d eye, stressRate, StressRateDevJaumann;
	Matrix3d stressIncrement, d_dev, devStrainIncrement, sigmaFinal_dev, stressRateDev, oldStressDeviator, newStressDeviator;
	double plasticStrainIncrement;
	double dt = FACTOR * update->dt;
	double newPressure;
	double G_eff = 0.0; // effective shear modulus
	double K_eff; // effective bulk modulus
	double M, p_wave_speed;
	double rho, effectiveViscosity, d_iso;
	double plastic_work; // dissipated plastic heat per unit volume
	Matrix3d deltaStressDev, oldStress;

	dtCFL = 1.0e22;
	eye.setIdentity();

	for (i = 0; i < nlocal; i++) {
		itype = type[i];
		if (setflag[itype][itype] == 1) {

			newPressure = 0.0;
			plastic_work = 0.0;

			effectiveViscosity = 0.0;
			K_eff = 0.0;
			G_eff = 0.0;
			D = 0.5 * (L[i] + L[i].transpose());
			if (corotated) {
				D = (R[i].transpose() * D * R[i]).eval(); // remove rotation
			}

			d_iso = D.trace();
			d_dev = Deviator(D);

			rho = rmass[i] / vol[i];

			double mu = 1.0 - J[i];
			double temperature = 1.0;

			/*
			 * Retrieve elastic stresses from the beginning of this time step
			 */
			oldStress(0, 0) = tlsph_stress[i][0];
			oldStress(0, 1) = tlsph_stress[i][1];
			oldStress(0, 2) = tlsph_stress[i][2];
			oldStress(1, 1) = tlsph_stress[i][3];
			oldStress(1, 2) = tlsph_stress[i][4];
			oldStress(2, 2) = tlsph_stress[i][5];
			oldStress(1, 0) = oldStress(0, 1);
			oldStress(2, 0) = oldStress(0, 2);
			oldStress(2, 1) = oldStress(1, 2);

			/*
			 * compute pressure
			 */

			matDB.ComputePressure(mu, temperature, itype, newPressure, K_eff);
			//if (fabs(mu) > 1.0e-3) printf("j=%g, , mu=%g, p=%g\n", J[i], mu, newPressure);

			/*
			 * ******************************* STRENGTH MODELS ************************************************
			 */
			newStressDeviator.setZero();
			if (matDB.gProps[itype].strengthType != 0) {

				oldStressDeviator = Deviator(oldStress);

				devStrainIncrement = dt * d_dev;

				matDB.ComputeDevStressIncrement(devStrainIncrement, itype, oldStressDeviator, plasticStrainIncrement,
						stressIncrement, plastic_work);
				eff_plastic_strain[i] += plasticStrainIncrement;

				newStressDeviator = oldStressDeviator + stressIncrement;

				// estimate effective shear modulus for time step stability
				deltaStressDev = oldStressDeviator - newStressDeviator;
				G_eff = effective_shear_modulus(d_dev / dt, deltaStressDev, dt, itype);

			} // end if (strength[itype] != NONE)

			/*
			 * assemble updated elastic stress Tensor from pressure and deviatoric parts
			 */
			stressTensor[i] = -newPressure * eye + newStressDeviator;

			//cout << "this is the new stress deviator: " << newStressDeviator << endl;

			/*
			 * store new stress Tensor (only elastic contributions)
			 */
			tlsph_stress[i][0] = stressTensor[i](0, 0);
			tlsph_stress[i][1] = stressTensor[i](0, 1);
			tlsph_stress[i][2] = stressTensor[i](0, 2);
			tlsph_stress[i][3] = stressTensor[i](1, 1);
			tlsph_stress[i][4] = stressTensor[i](1, 2);
			tlsph_stress[i][5] = stressTensor[i](2, 2);

			/*
			 * For keeping track of the stored elastic energy in the system,
			 * we need an intermediate stress tensor, evaluated at half-timestep
			 * This is simply sigma_old + 1/2 * sigma_increment
			 */
			Matrix3d sigma_one_half = oldStress + 0.5 * (stressTensor[i] - oldStress);

			/*
			 * add viscous stress
			 */

			if (matDB.gProps[itype].viscType != 0) {

				Matrix3d viscousStress;
				matDB.ComputeViscousStress(d_dev, itype, viscousStress);

				smd_visc_stress[i][0] = viscousStress(0, 0);
				smd_visc_stress[i][1] = viscousStress(0, 1);
				smd_visc_stress[i][2] = viscousStress(0, 2);
				smd_visc_stress[i][3] = viscousStress(1, 1);
				smd_visc_stress[i][4] = viscousStress(1, 2);
				smd_visc_stress[i][5] = viscousStress(2, 2);

				stressTensor[i] += viscousStress;

			}

			/*
			 * stable timestep based on speed-of-sound
			 */

			if (neighs[i] > 1) {

				M = K_eff + 4.0 * G_eff / 3.0;
				p_wave_speed = sqrt(M / rho);
				dtCFL = MIN(cellsize / p_wave_speed, dtCFL);

				/*
				 * stable timestep based on viscosity
				 */
				if (matDB.gProps[itype].viscType != 0) {
					dtCFL = MIN(cellsize * cellsize * rho / (effectiveViscosity), dtCFL);
				}
			} else {
				p_wave_speed = matDB.gProps[itype].c0;
				dtCFL = MIN(cellsize / p_wave_speed, dtCFL);
			}

			/*
			 * elastic energy rate -- without plastic heating
			 */

			//de[i] += FACTOR * vol[i] * ((stressTensor[i].cwiseProduct(D)).sum() - plastic_work/dt);
			de[i] += FACTOR * vol[i] * ((sigma_one_half.cwiseProduct(D)).sum() - plastic_work / dt);
			heat[i] += vol[i] * plastic_work;

			/*
			 * finally, rotate stress tensor forward to current configuration
			 */
			if (corotated) {
				stressTensor[i] = (R[i] * stressTensor[i] * R[i].transpose()).eval();
			}

		} // end if (setflag[itype][itype] == 1)
	} // end loop over nlocal

	// fallback if no atoms are present:
	int check_flag = 0;
	for (itype = 1; itype <= atom->ntypes; itype++) {
		if (setflag[itype][itype] == 1) {
			p_wave_speed = matDB.gProps[itype].c0;
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

void PairSmdMpmLin::allocate() {

	allocated = 1;
	int n = atom->ntypes;

	memory->create(setflag, n + 1, n + 1, "pair:setflag");
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

void PairSmdMpmLin::settings(int narg, char **arg) {

// defaults
	vlimit = -1.0;
	FLIP_contribution = 0.99;
	region_flag = 0;
	flag3d = true;

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

		if (strcmp(arg[iarg], "sym_y_+") == 0) {
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

		} else if (strcmp(arg[iarg], "limit_velocity") == 0) {
			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected number following limit_velocity");
			}
			vlimit = force->numeric(FLERR, arg[iarg]);

			if (comm->me == 0) {
				printf("... will limit velocities to <= %g\n", vlimit);
			}

		} else if (strcmp(arg[iarg], "FLIP") == 0) {
			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected number following FLIP");
			}
			FLIP_contribution = force->numeric(FLERR, arg[iarg]);
		} else if (strcmp(arg[iarg], "2d") == 0) {
			flag3d = false;
			if (comm->me == 0) {
				printf("...  2d integration with zero z component\n");
			}
		} else if (strcmp(arg[iarg], "exclude_region") == 0) {
			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected string following exclude_region");
			}
			nregion = domain->find_region(arg[iarg]);
			if (nregion == -1)
				error->all(FLERR, "exclude_region region ID does not exist");
			int n = strlen(arg[iarg]) + 1;
			idregion = new char[n];
			strcpy(idregion, arg[iarg]);
			region_flag = 1;

			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected exclude_region region_name vx vy vz");
			}
			const_vx = force->numeric(FLERR, arg[iarg]);
			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected exclude_region region_name vx vy vz");
			}
			const_vy = force->numeric(FLERR, arg[iarg]);
			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected exclude_region region_name vx vy vz");
			}
			const_vz = force->numeric(FLERR, arg[iarg]);
			if (comm->me == 0) {
				printf("... region %s with constant velocity %f %f %f\n", idregion, const_vx, const_vy, const_vz);
			}
		} else if (strcmp(arg[iarg], "corotated") == 0) {
			if (comm->me == 0) {
				printf("... will use corotated formulation for stress rates\n");
				corotated = true;
			}
		} else if (strcmp(arg[iarg], "true_deformation") == 0) {
			if (comm->me == 0) {
				printf("... will use true deformation for velocity gradient\n");
			}
			true_deformation = true;

		} else {
			char msg[128];
			sprintf(msg, "Illegal keyword for pair smd/mpm: %s\n", arg[iarg]);
			error->all(FLERR, msg);
		}

	}

	PIC_contribution = 1.0 - FLIP_contribution;

	if (comm->me == 0) {
		printf("... will use %3.2f FLIP and %3.2f PIC update for velocities\n", FLIP_contribution, PIC_contribution);
		printf(">>========>>========>>========>>========>>========>>========>>========>>========\n");
	}

}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairSmdMpmLin::coeff(int narg, char **arg) {
	int itype, jtype;
	char str[128];

	if (narg < 2) {
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

		if (comm->me == 0) {
			printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n");
			printf("...SMD / MPM PROPERTIES OF PARTICLE TYPE %d\n\n", itype);
		}

		/*
		 * read parameters which are common -- regardless of material / eos model
		 */

		setflag[itype][itype] = 1;

		/*
		 * error checks
		 */

	} else {
		/*
		 * we are reading a cross-interaction line for particle types i, j
		 */

		itype = force->inumeric(FLERR, arg[0]);
		jtype = force->inumeric(FLERR, arg[1]);

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

double PairSmdMpmLin::init_one(int i, int j) {

	if (!allocated)
		allocate();

	if (setflag[i][j] == 0)
		error->all(FLERR, "All pair coeffs are not set");

	return 2.0 * cellsize;
}

/* ----------------------------------------------------------------------
 init specific to this pair style
 ------------------------------------------------------------------------- */

void PairSmdMpmLin::init_style() {
// request a granular neighbor list
//int irequest = neighbor->request(this);
//neighbor->requests[irequest]->half = 0;
//neighbor->requests[irequest]->gran = 1;
}

/* ----------------------------------------------------------------------
 neighbor callback to inform pair style of neighbor list to use
 optional granular history list
 ------------------------------------------------------------------------- */

void PairSmdMpmLin::init_list(int id, NeighList *ptr) {
	if (id == 0)
		list = ptr;
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based arrays
 ------------------------------------------------------------------------- */

double PairSmdMpmLin::memory_usage() {

//printf("in memory usage\n");

	return 11 * nmax * sizeof(double) + Ncells * sizeof(Gridnode);

}

/* ---------------------------------------------------------------------- */

int PairSmdMpmLin::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc) {
	double **f = atom->f;
	double **v = atom->v;
	double **x = atom->x;
	int i, j, m;

//printf("packing comm\n");

	m = 0;
	if (comm_mode == STAGE1) {
		// first comm in MUSL: need volume, stress, forces on ghosts
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
	} else if (comm_mode == STAGE2) {
		// second comm in MUSL: need updated velocities and positions on ghosts
		for (i = 0; i < n; i++) {
			j = list[i];
			buf[m++] = v[j][0];
			buf[m++] = v[j][1];
			buf[m++] = v[j][2];

			buf[m++] = x[j][0];
			buf[m++] = x[j][1];
			buf[m++] = x[j][2];

			buf[m++] = particleVelocities[j](0);
			buf[m++] = particleVelocities[j](1);
			buf[m++] = particleVelocities[j](2);

			m++;
		}
	}
	return m;
}

/* ---------------------------------------------------------------------- */

void PairSmdMpmLin::unpack_forward_comm(int n, int first, double *buf) {
	double **f = atom->f;
	double **v = atom->v;
	double **x = atom->x;

	int i, m, last;

	m = 0;
	if (comm_mode == STAGE1) {
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
	} else if (comm_mode == STAGE2) {
		last = first + n;
		for (i = first; i < last; i++) {
			v[i][0] = buf[m++];
			v[i][1] = buf[m++];
			v[i][2] = buf[m++];

			x[i][0] = buf[m++];
			x[i][1] = buf[m++];
			x[i][2] = buf[m++];

			particleVelocities[i](0) = buf[m++];
			particleVelocities[i](1) = buf[m++];
			particleVelocities[i](2) = buf[m++];

			m++;
		}
	}
}

/*
 * EXTRACT
 */

void *PairSmdMpmLin::extract(const char *str, int &i) {
	if (strcmp(str, "smd/ulsph/stressTensor_ptr") == 0) {
		return (void *) stressTensor;
	} else if (strcmp(str, "smd/ulsph/velocityGradient_ptr") == 0) {
		return (void *) L;
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
	} else if (strcmp(str, "smd/ulsph/numNeighs_ptr") == 0) {
		return (void *) neighs;
	}

	return NULL;
}

/* ----------------------------------------------------------------------
 compute effective shear modulus by dividing rate of deviatoric stress with rate of shear deformation
 ------------------------------------------------------------------------- */

double PairSmdMpmLin::effective_shear_modulus(const Matrix3d d_dev, const Matrix3d deltaStressDev, const double dt,
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
		if (matDB.gProps[itype].strengthType != 0) {
			G_eff = matDB.gProps[itype].G0;
		} else {
			G_eff = 0.0;
		}
	}

	return G_eff;

}

/*
 * do time integration of particles
 */

void PairSmdMpmLin::AdvanceParticles() {

	int nlocal = atom->nlocal;
	int i, mode;
	int *type = atom->type;
	tagint *mol = atom->molecule;
	double **x = atom->x;
	double **v = atom->v;
	double *heat = atom->heat;
	double scale, vsq;

	if (region_flag == 1) {
		//domain->regions[nregion]->init();
		domain->regions[nregion]->prematch();
	}

	dtv = update->dt;

	for (i = 0; i < nlocal; i++) {

		int itype = type[i];
		if (setflag[itype][itype]) {

			mode = DEFAULT_INTEGRATION;
			if (mol[i] == 1000) {
				mode = CONSTANT_VELOCITY; // move particle with its own velocity, do not change velocity
			}
			if (region_flag == 1) {
				if (domain->regions[nregion]->match(x[i][0], x[i][1], x[i][2])) {
					mode = PRESCRIBED_VELOCITY;
				}
			}

			if (mode == DEFAULT_INTEGRATION) {

				//printf("doing default integration for particle %d, picis%f, flip %f\n", i, PIC_contribution, FLIP_contribution);

				//double a = particleAccelerations[i].norm();
				//if (a > 0.0)
//					printf("acceleration of particle is %f\n", a);

				// mixed FLIP-PIC update of velocities
				v[i][0] = PIC_contribution * particleVelocities[i](0)
						+ FLIP_contribution * (v[i][0] + dtv * particleAccelerations[i](0));
				v[i][1] = PIC_contribution * particleVelocities[i](1)
						+ FLIP_contribution * (v[i][1] + dtv * particleAccelerations[i](1));
				if (flag3d)
					v[i][2] = PIC_contribution * particleVelocities[i](2)
							+ FLIP_contribution * (v[i][2] + dtv * particleAccelerations[i](2));

				// particles are moved according to interpolated grid velocities
				x[i][0] += dtv * particleVelocities[i](0);
				x[i][1] += dtv * particleVelocities[i](1);
				if (flag3d)
					x[i][2] += dtv * particleVelocities[i](2);

				if (vlimit > 0.0) {
					vsq = v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2];
					if (vsq > vlimitsq) {
						scale = sqrt(vlimitsq / vsq);
						v[i][0] *= scale;
						v[i][1] *= scale;
						if (flag3d)
							v[i][2] *= scale;
					}
				}

//				if (symmetry_plane_x_plus_exists) {
//
//					if (fabs(x[i][0] - symmetry_plane_x_plus_location) < 0.1 * cellsize) {
//						x[i][0] = symmetry_plane_x_plus_location;
//						//particleVelocities[i](0) = 0.0;
//						//v[i][0] = 0.0;
//					}

//					if (x[i][0] < symmetry_plane_x_plus_location) {
//						x[i][0] = symmetry_plane_x_plus_location;
//						particleVelocities[i](0) = 0.0;
//						v[i][0] = 0.0;
//					}
//				}
//				if (symmetry_plane_x_minus_exists) {
//					if (x[i][0] > symmetry_plane_x_minus_location) {
//						x[i][0] = symmetry_plane_x_minus_location;
//						//particleVelocities[i](0) = 0.0;
//						//v[i][0] = 0.0;
//					}
//				}
//				if (symmetry_plane_y_plus_exists) {
//					if (x[i][1] < symmetry_plane_y_plus_location) {
//						printf("particle y=%f below y+ symmetry plane. resetting to %f\n", x[i][1], symmetry_plane_y_plus_location);
//						x[i][1] = symmetry_plane_y_plus_location;
//						//particleVelocities[i](1) = 0.0;
//						//v[i][1] = 0.0;
//					}
//				}
//				if (symmetry_plane_y_minus_exists) {
//					if (x[i][1] > symmetry_plane_y_minus_location) {
//						x[i][1] = symmetry_plane_y_minus_location;
//						//v[i][1] = 0.0;
//					}
//				}
//				if (symmetry_plane_z_plus_exists) {
//					if (x[i][2] < symmetry_plane_z_plus_location) {
//						x[i][2] = symmetry_plane_z_plus_location;
//						//v[i][2] = 0.0;
//					}
//				}
//				if (symmetry_plane_z_minus_exists) {
//					if (x[i][2] > symmetry_plane_z_minus_location) {
//						x[i][2] = symmetry_plane_z_minus_location;
//						//v[i][2] = 0.0;
//					}
//				}

				//heat[i] = PIC_contribution * particleHeat[i] + FLIP_contribution * (heat[i] + dtv * particleHeatRate[i]);
				heat[i] += dtv * particleHeatRate[i];

			} else if (mode == PRESCRIBED_VELOCITY) {

				// this is zero-acceleration (constant velocity) time integration for boundary particles
				x[i][0] += dtv * const_vx;
				x[i][1] += dtv * const_vy;
				if (flag3d)
					x[i][2] += dtv * const_vz;

				v[i][0] = const_vx;
				v[i][1] = const_vy;
				if (flag3d)
					v[i][2] = const_vz;

			}

			else if (mode == CONSTANT_VELOCITY) {

				// this is zero-acceleration (constant velocity) time integration for boundary particles
				x[i][0] += dtv * v[i][0];
				x[i][1] += dtv * v[i][1];
				if (flag3d)
					x[i][2] += dtv * v[i][2];

			}

		}

	}

}

void PairSmdMpmLin::AdvanceParticlesEnergy() {

	int nlocal = atom->nlocal;
	int i;
	int *type = atom->type;
	double *e = atom->e;
	double *de = atom->de;

	for (i = 0; i < nlocal; i++) {
		int itype = type[i];
		if (setflag[itype][itype]) {
			e[i] += update->dt * de[i];
		}
	}

}

void PairSmdMpmLin::ComputeHeatGradientOnGrid() {

// todo: introduce heat conduction coeff

	double factor = icellsize * icellsize;
	double totalflux = 0.0;
	double dxx, dyy, dzz;
	for (int ix = 1; ix < grid_nx - 1; ix++) {
		for (int iy = 1; iy < grid_ny - 1; iy++) {
			for (int iz = 1; iz < grid_nz - 1; iz++) {

				int icell = ix + iy * grid_nx + iz * grid_nx * grid_ny;

				if (lgridnodes[icell].mass > MASS_CUTOFF) {

					dxx = dyy = dzz = 0.0;

					// derivative in x direction
					int plus_cell = ix + 1 + iy * grid_nx + iz * grid_nx * grid_ny;
					int minus_cell = ix - 1 + iy * grid_nx + iz * grid_nx * grid_ny;

					if ((lgridnodes[plus_cell].mass > MASS_CUTOFF) && (lgridnodes[minus_cell].mass > MASS_CUTOFF)) {
						// can do central 2nd deriv
						dxx = factor * (lgridnodes[plus_cell].heat - 2.0 * lgridnodes[icell].heat + lgridnodes[minus_cell].heat);
					}

//					if (fabs(dxx) > 1.0e-5) {
//						printf("heat distribution: %f %f %f\n", lgridnodes[plus_cell].heat, lgridnodes[icell].heat,
//								lgridnodes[minus_cell].heat);
//						printf("xx 2nd deriv. : %f\n", dxx);
//					}

					// derivative in y direction
					plus_cell = ix + (iy + 1) * grid_nx + iz * grid_nx * grid_ny;
					minus_cell = ix + (iy - 1) * grid_nx + iz * grid_nx * grid_ny;
					if ((lgridnodes[plus_cell].mass > MASS_CUTOFF) && (lgridnodes[minus_cell].mass > MASS_CUTOFF)) {
						// can do central 2nd deriv
						dyy = factor * (lgridnodes[plus_cell].heat - 2.0 * lgridnodes[icell].heat + lgridnodes[minus_cell].heat);
					}

//					dyy = factor * (lgridnodes[plus_cell].heat - 2.0 * lgridnodes[icell].heat + lgridnodes[minus_cell].heat);

//					if (fabs(dyy) > 1.0e-5) {
//						printf("yy heat distribution: %f %f %f\n", lgridnodes[plus_cell].heat, lgridnodes[icell].heat,
//								lgridnodes[minus_cell].heat);
//						printf("yy 2nd deriv. : %f\n", dyy);
//					}

//					// derivative in z direction
					if (flag3d) {
						plus_cell = ix + iy * grid_nx + (iz + 1) * grid_nx * grid_ny;
						minus_cell = ix + iy * grid_nx + (iz - 1) * grid_nx * grid_ny;

						if ((lgridnodes[plus_cell].mass > MASS_CUTOFF) && (lgridnodes[minus_cell].mass > MASS_CUTOFF)) {
							// can do central 2nd deriv
							dzz = factor
									* (lgridnodes[plus_cell].heat - 2.0 * lgridnodes[icell].heat + lgridnodes[minus_cell].heat);
						}
					}
//
//					if (fabs(dzz) > 1.0e-5) {
//						printf("zz heat distribution: %f %f %f\n", lgridnodes[plus_cell].heat, lgridnodes[icell].heat,
//								lgridnodes[minus_cell].heat);
//						printf("zz 2nd deriv. : %f\n", dzz);
//					}

					totalflux += dxx + dyy + dzz;

					lgridnodes[icell].dheat_dt = dxx + dyy + dzz;
				}
			}
		}
	}

//printf("total heat flux is %f\n", totalflux);
}

void PairSmdMpmLin::DumpGrid() {

	printf("... dumping grid\n");

	int iz = 3;
	int count = 0;
	for (int ix = 0; ix < grid_nx; ix++) {
		for (int iy = 0; iy < grid_ny; iy++) {
			//for (int iz = 0; iz < grid_nz; iz++) {
			//int icell = ix + iy * grid_nx + iz * grid_nx * grid_ny;
			//if (lgridnodes[icell].mass > MASS_CUTOFF) {
			count++;
			//}
			//}

		}
	}

	FILE * f;
	f = fopen("grid.dump", "w");

	fprintf(f, "%d\n\n", count);

	for (int ix = 0; ix < grid_nx; ix++) {
		for (int iy = 0; iy < grid_ny; iy++) {
			//for (int iz = 0; iz < grid_nz; iz++) {
			int icell = ix + iy * grid_nx + iz * grid_nx * grid_ny;
			//if (lgridnodes[icell].mass > MASS_CUTOFF) {
			fprintf(f, "X %f %f %f %g\n", ix * cellsize, iy * cellsize, iz * cellsize, lgridnodes[icell].mass);
			//}
			//}

		}
	}

	fclose(f);

}

void PairSmdMpmLin::MirrorCellVelocity(int source, int target, int component) {

	if ((source < 0) || (source >= Ncells)) { // memory access error check
		printf("source  %d outside allowed range %d to %d\n", source, 0, Ncells);
		exit(1);
	}
	if ((target < 0) || (target >= Ncells)) { // memory access error check
		printf("target  %d outside allowed range %d to %d\n", target, 0, Ncells);
		exit(1);
	}

	lgridnodes[target].v = lgridnodes[source].v;
	lgridnodes[target].fbody = lgridnodes[source].fbody;
	lgridnodes[target].f = lgridnodes[source].f;
	lgridnodes[target].mass = lgridnodes[source].mass;
	lgridnodes[target].imass = lgridnodes[source].imass;
	lgridnodes[target].heat = lgridnodes[source].heat;

	lgridnodes[target].v(component) *= -1.0;
	lgridnodes[target].f(component) *= -1.0;
	lgridnodes[target].fbody(component) *= -1.0;
}

void PairSmdMpmLin::ApplySymmetryBC(int icell, int ix, int iy, int iz, int direction) {

	if ((icell < 0) || (icell >= Ncells)) { // memory access error check
		printf("icell  %d outside allowed range %d to %d\n", icell, 0, Ncells);
		printf("ix=%d, iy=%d, iz=%d\n", ix, iy, iz);
		printf("direction is %d\n", direction);
		exit(1);
	}

	if (direction == Xplus) {
		lgridnodes[icell].v(0) = 0.0;
		lgridnodes[icell].f(0) = 0.0;
		// mirror velocity of nodes on the +-side to the -side
		int source = ix + 1;
		int target = ix - 1;
		int sourcecell = source + iy * grid_nx + iz * grid_nx * grid_ny;
		int targetcell = target + iy * grid_nx + iz * grid_nx * grid_ny;
		MirrorCellVelocity(sourcecell, targetcell, 0);

	} else if (direction == Xminus) {
		lgridnodes[icell].v(0) = 0.0;
		lgridnodes[icell].f(0) = 0.0;
		// mirror velocity of nodes on the +-side to the -side
		int source = ix - 1;
		int target = ix + 1;
		int sourcecell = source + iy * grid_nx + iz * grid_nx * grid_ny;
		int targetcell = target + iy * grid_nx + iz * grid_nx * grid_ny;
		MirrorCellVelocity(sourcecell, targetcell, 0);

	} else if (direction == Yplus) {
		lgridnodes[icell].v(1) = 0.0;
		lgridnodes[icell].f(1) = 0.0;
		// mirror velocity of nodes on the +-side to the -side
		int source = iy + 1;
		int target = iy - 1;
		int sourcecell = ix + source * grid_nx + iz * grid_nx * grid_ny;
		int targetcell = ix + target * grid_nx + iz * grid_nx * grid_ny;
		MirrorCellVelocity(sourcecell, targetcell, 1);

	} else if (direction == Yminus) {
		lgridnodes[icell].v(1) = 0.0;
		lgridnodes[icell].f(1) = 0.0;
		// mirror velocity of nodes on the +-side to the -side
		int source = iy - 1;
		int target = iy + 1;
		int sourcecell = ix + source * grid_nx + iz * grid_nx * grid_ny;
		int targetcell = ix + target * grid_nx + iz * grid_nx * grid_ny;
		MirrorCellVelocity(sourcecell, targetcell, 1);

	} else if (direction == Zplus) {
		lgridnodes[icell].v(2) = 0.0;
		lgridnodes[icell].f(2) = 0.0;
		// mirror velocity of nodes on the +-side to the -side
		int source = iz + 1;
		int target = iz - 1;
		int sourcecell = ix + iy * grid_nx + source * grid_nx * grid_ny;
		int targetcell = ix + iy * grid_nx + target * grid_nx * grid_ny;
		MirrorCellVelocity(sourcecell, targetcell, 2);

	} else if (direction == Zminus) {
		lgridnodes[icell].v(2) = 0.0;
		lgridnodes[icell].f(2) = 0.0;
		// mirror velocity of nodes on the +-side to the -side
		int source = iz - 1;
		int target = iz + 1;
		int sourcecell = ix + iy * grid_nx + source * grid_nx * grid_ny;
		int targetcell = ix + iy * grid_nx + target * grid_nx * grid_ny;
		MirrorCellVelocity(sourcecell, targetcell, 2);
	}

}

void PairSmdMpmLin::CheckSymmetryBC() {

	if (symmetry_plane_x_plus_exists) {
		double ssx = icellsize * symmetry_plane_x_plus_location - static_cast<double>(minix); // shifted position in grid coords
		int ssx_index = static_cast<int>(ssx);

		if ((ssx_index > 0) && (ssx_index < grid_nx - 1)) { // this checks if symmetry plane is contained witin the current grid
			for (int iy = 0; iy < grid_ny; iy++) {
				for (int iz = 0; iz < grid_nz; iz++) {
					int icell = ssx_index + iy * grid_nx + iz * grid_nx * grid_ny;
					ApplySymmetryBC(icell, ssx_index, iy, iz, Xplus);
				}
			}
		}
	}

	if (symmetry_plane_x_minus_exists) {
		double ssx = icellsize * symmetry_plane_x_minus_location - static_cast<double>(minix); // shifted position in grid coords
		int ssx_index = static_cast<int>(ssx);
		if ((ssx_index > 0) && (ssx_index < grid_nx - 1)) { // this checks if symmetry plane is contained witin the current grid
			for (int iy = 0; iy < grid_ny; iy++) {
				for (int iz = 0; iz < grid_nz; iz++) {
					int icell = ssx_index + iy * grid_nx + iz * grid_nx * grid_ny;
					ApplySymmetryBC(icell, ssx_index, iy, iz, Xminus);
				}
			}
		}
	}

	if (symmetry_plane_y_plus_exists) {
		double ssy = icellsize * symmetry_plane_y_plus_location - static_cast<double>(miniy); // shifted position in grid coords
		int ssy_index = static_cast<int>(ssy);
		if ((ssy_index > 0) && (ssy_index < grid_ny - 1)) { // this checks if symmetry plane is contained witin the current grid
			for (int ix = 0; ix < grid_nx; ix++) {
				for (int iz = 0; iz < grid_nz; iz++) {
					int icell = ix + ssy_index * grid_nx + iz * grid_nx * grid_ny;
					ApplySymmetryBC(icell, ix, ssy_index, iz, Yplus);
				}
			}
		}
	}

	if (symmetry_plane_y_minus_exists) {
		double ssy = icellsize * symmetry_plane_y_minus_location - static_cast<double>(miniy); // shifted position in grid coords
		int ssy_index = static_cast<int>(ssy);
		if ((ssy_index > 0) && (ssy_index < grid_ny - 1)) { // this checks if symmetry plane is contained witin the current grid
			for (int ix = 0; ix < grid_nx; ix++) {
				for (int iz = 0; iz < grid_nz; iz++) {
					int icell = ix + ssy_index * grid_nx + iz * grid_nx * grid_ny;
					ApplySymmetryBC(icell, ix, ssy_index, iz, Yminus);
				}
			}
		}
	}

	if (symmetry_plane_z_plus_exists) {
		double ssz = icellsize * symmetry_plane_z_plus_location - static_cast<double>(miniz); // shifted position in grid coords
		int ssz_index = static_cast<int>(ssz);
		if ((ssz_index > 0) && (ssz_index < grid_nz - 1)) { // this checks if symmetry plane is contained witin the current grid
			for (int ix = 0; ix < grid_nx; ix++) {
				for (int iy = 0; iy < grid_ny; iy++) {
					int icell = ix + iy * grid_nx + ssz_index * grid_nx * grid_ny;
					ApplySymmetryBC(icell, ix, iy, ssz_index, Zplus);
				}
			}
		}
	}

	if (symmetry_plane_z_minus_exists) {
		double ssz = icellsize * symmetry_plane_z_minus_location - static_cast<double>(miniz); // shifted position in grid coords
		int ssz_index = static_cast<int>(ssz);
		if ((ssz_index > 0) && (ssz_index < grid_nz - 1)) { // this checks if symmetry plane is contained witin the current grid
			for (int ix = 0; ix < grid_nx; ix++) {
				for (int iy = 0; iy < grid_ny; iy++) {
					int icell = ix + iy * grid_nx + ssz_index * grid_nx * grid_ny;
					ApplySymmetryBC(icell, ix, iy, ssz_index, Zminus);
				}
			}
		}
	}
}
