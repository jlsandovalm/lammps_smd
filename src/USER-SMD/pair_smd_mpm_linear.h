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

#ifdef PAIR_CLASS

PairStyle(smd/mpm_linear,PairSmdMpmLin)

#else

#ifndef LMP_SMD_MPM_LINEAR_H
#define LMP_SMD_MPM_LINEAR_H

#include "pair.h"
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/SVD>
//#include "SmdMatDB.h"

using namespace Eigen;
using namespace std;
namespace LAMMPS_NS {

class PairSmdMpmLin: public Pair {
public:
	PairSmdMpmLin(class LAMMPS *);
	virtual ~PairSmdMpmLin();
	virtual void compute(int, int);
	void settings(int, char **);
	void coeff(int, char **);
	double init_one(int, int);
	void init_style();
	void init_list(int, class NeighList *);
	virtual double memory_usage();
	int pack_forward_comm(int, int *, double *, int, int *);
	void unpack_forward_comm(int, int, double *);
	void *extract(const char *, int &);
	void PreCompute();
	void PreCompute_DensitySummation();
	double effective_shear_modulus(const Matrix3d d_dev, const Matrix3d deltaStressDev, const int itype);

	void CreateGrid();
	void UpdateGridVelocities();
	void UpdateDeformationGradient();
	void DestroyGrid();

	void PointsToGrid();
	void ComputeVelocityGradient();
	void ComputeGridForces();
	void GridToPoints();
	void GetStress();
	void ApplyVelocityBC();
	void SolveHeatEquation();
	void MUSL();
	void USF();

	void AdvanceParticles();
	void AdvanceParticlesEnergy();
	void ComputeGridGradients();
	void DumpGrid();
	void GridGradientsToParticles();
	void PreComputeGridWeights(const int i, int &ref_node, double *wfx, double *wfy, double *wfz);
	void PreComputeGridWeightsAndDerivatives(const int i, int &ref_node, double *wfx, double *wfy, double *wfz, double *wfdx,
			double *wfdy, double *wfdz);
	void VelocitiesToGrid();
	void ComputeHeatGradientOnGrid();
	void ApplySymmetryBC(int icell, int ix, int iy, int iz, int direction);
	void CheckSymmetryBC();
	void UpdateStress();
	void MirrorCellVelocity(int source, int target, int component);

protected:
	void allocate();
	int nmax; // max number of atoms on this proc
	int *neighs;
	double *c0;
	double *particleHeat, *particleHeatRate;
	double *J; // determinant of deformation gradient
	double *vol; // current volume
	Matrix3d *stressTensor, *L, *F, *R;
	Vector3d *heat_gradient;

	double dtCFL;
	Vector3d *particleVelocities, *particleAccelerations; // per-particle angular momentum

private:

	// stuff for time integration
	enum {
		DEFAULT_INTEGRATION = 4000, CONSTANT_VELOCITY = 4001, PRESCRIBED_VELOCITY = 4002
	};
	double dtv, vlimit, vlimitsq;
	int mass_require;
	double FLIP_contribution, PIC_contribution;
	double const_vx, const_vy, const_vz;
	bool flag3d; // integrate z degree of freedom?
	int nregion, region_flag;
	char *idregion;
	int comm_mode;

	// enumerate some quantitities and associate these with integer values such that they can be used for lookup in an array structure

	struct Gridnode {
		double mass, heat, dheat_dt, imass;
		Vector3d v, fbody, f;
		bool isVelocityBC;
		int count;
	};

	double cellsize, icellsize;
	int minix, miniy, miniz, maxix, maxiy, maxiz;
	int grid_nx, grid_ny, grid_nz, Ncells;
	double minx, maxx, miny, maxy, minz, maxz;

	Gridnode *lgridnodes; // linear array of gridnodes

	double timeone_PointstoGrid, timeone_Gradients, timeone_MaterialModel, timeone_GridForces, timeone_UpdateGrid,
			timeone_GridToPoints, timeone_SymmetryBC, timeone_Comm, timeone_UpdateParticles;

	// symmetry planes
	double symmetry_plane_y_plus_location, symmetry_plane_y_minus_location, symmetry_plane_x_plus_location,
			symmetry_plane_x_minus_location, symmetry_plane_z_plus_location, symmetry_plane_z_minus_location;
	bool symmetry_plane_y_plus_exists, symmetry_plane_y_minus_exists, symmetry_plane_x_plus_exists, symmetry_plane_x_minus_exists,
			symmetry_plane_z_plus_exists, symmetry_plane_z_minus_exists;

	// NO SLIP symmetry planes
	bool noslip_symmetry_plane_y_plus_exists, noslip_symmetry_plane_y_minus_exists;
	double noslip_symmetry_plane_y_plus_location, noslip_symmetry_plane_y_minus_location;

	//SmdMatDB matDB;
	bool corotated;
	bool true_deformation;
	int iproc;
	int gimp_offset;

};

}

#endif
#endif

