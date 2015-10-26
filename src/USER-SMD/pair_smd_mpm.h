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

PairStyle(smd/mpm,PairSmdMpm)

#else

#ifndef LMP_ULSPH_BGRID_H
#define LMP_ULSPH_BGRID_H

#include "pair.h"
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/SVD>

using namespace Eigen;
using namespace std;
namespace LAMMPS_NS {

class PairSmdMpm: public Pair {
public:
	PairSmdMpm(class LAMMPS *);
	virtual ~PairSmdMpm();
	virtual void compute(int, int);
	void settings(int, char **);
	void coeff(int, char **);
	double init_one(int, int);
	void init_style();
	void init_list(int, class NeighList *);
	virtual double memory_usage();
	int pack_forward_comm(int, int *, double *, int, int *);
	void unpack_forward_comm(int, int, double *);
	void AssembleStressTensor();
	void *extract(const char *, int &);
	void PreCompute();
	void PreCompute_DensitySummation();
	double effective_shear_modulus(const Matrix3d d_dev, const Matrix3d deltaStressDev, const double dt, const int itype);

	void CreateGrid();
	void UpdateGridVelocities();
	void UpdateDeformationGradient();
	void DestroyGrid();

	void PointsToGrid();
	void ComputeVelocityGradient();
	void ComputeGridForces();
	void GridToPoints();
	void UpdateStress();
	void GetStress();
	void ApplyVelocityBC();

protected:

	bool APIC;

	double *c0_type; // reference speed of sound defined per particle type
	double *rho0; // reference mass density per type
	double *Q1; // linear artificial viscosity coeff
	int *eos, *viscosity, *strength; // eos and strength material models

	void allocate();

	int nmax; // max number of atoms on this proc
	int *numNeighs;
	double *c0;
	Matrix3d *stressTensor, *L, *F;

	double dtCFL;
	bool Bp_exists;

	Vector3d *particleVelocities, *particleAccelerations; // per-particle angular momentum

private:

	// enumerate EOSs. MUST BE IN THE RANGE [1000, 2000)
	enum {
		EOS_LINEAR = 1000, EOS_PERFECT_GAS = 1001, EOS_TAIT = 1002,
	};

	// enumerate physical viscosity models. MUST BE IN THE RANGE [2000, 3000)
	enum {
		VISCOSITY_NEWTON = 2000
	};

	// enumerate strength models. MUST BE IN THE RANGE [3000, 4000)
	enum {
		STRENGTH_LINEAR = 3000, STRENGTH_LINEAR_PLASTIC = 3001
	};

	// enumerate some quantitities and associate these with integer values such that they can be used for lookup in an array structure
	enum {
		NONE = 0,
		BULK_MODULUS = 1,
		HOURGLASS_CONTROL_AMPLITUDE = 2,
		EOS_TAIT_EXPONENT = 3,
		REFERENCE_SOUNDSPEED = 4,
		REFERENCE_DENSITY = 5,
		EOS_PERFECT_GAS_GAMMA = 6,
		SHEAR_MODULUS = 7,
		YIELD_STRENGTH = 8,
		YOUNGS_MODULUS = 9,
		POISSON_RATIO = 10,
		LAME_LAMBDA = 11,
		HEAT_CAPACITY = 12,
		M_MODULUS = 13,
		HARDENING_PARAMETER = 14,
		VISCOSITY_MU = 15,
		MAX_KEY_VALUE = 16
	};
	double **Lookup; // holds per-type material parameters for the quantities defined in enum statement above.

	struct Gridnode {
		double mass;
		double vx, vy, vz;
		double vestx, vesty, vestz;
		double fx, fy, fz;
		bool isVelocityBC;
		Vector3d u;
	};

	Gridnode ***gridnodes;
	double cellsize, icellsize;
	int grid_nx, grid_ny, grid_nz;
	int min_ix, min_iy, min_iz;
	int max_ix, max_iy, max_iz;

};

}

#endif
#endif

