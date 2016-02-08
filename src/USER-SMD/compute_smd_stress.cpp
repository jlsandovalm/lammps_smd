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

#include "string.h"
#include "compute_smd_stress.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "pair.h"
#include <Eigen/Eigen>
using namespace Eigen;
using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeSmdStress::ComputeSmdStress(LAMMPS *lmp, int narg, char **arg) :
		Compute(lmp, narg, arg) {
	if (narg != 3)
		error->all(FLERR, "Illegal compute smd/stress command");

	peratom_flag = 1;
	size_peratom_cols = 7;

	nmax = 0;
	stress_array = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSmdStress::~ComputeSmdStress() {
	memory->sfree(stress_array);
}

/* ---------------------------------------------------------------------- */

void ComputeSmdStress::init() {

	int count = 0;
	for (int i = 0; i < modify->ncompute; i++)
		if (strcmp(modify->compute[i]->style, "smd/stress") == 0)
			count++;
	if (count > 1 && comm->me == 0)
		error->warning(FLERR, "More than one compute smd/stress");
}

/* ---------------------------------------------------------------------- */

void ComputeSmdStress::compute_peratom() {
	invoked_peratom = update->ntimestep;
	Matrix3d stress, stress_deviator;
	double von_mises_stress;

	// grow vector array if necessary

	if (atom->nlocal > nmax) {
		memory->destroy(stress_array);
		nmax = atom->nmax;
		memory->create(stress_array, nmax, size_peratom_cols, "stresstensorVector");
		array_atom = stress_array;
	}

	// access all stress arrays from the various SMD methods.
	// output stress = sum of stresses

	bool tlsph, ulsph, mpm;
	tlsph = ulsph = mpm = false;
	int itmp = 0;

	Matrix3d *T_TLSPH = (Matrix3d *) force->pair->extract("smd/tlsph/stressTensor_ptr", itmp);
	if (T_TLSPH != NULL) {
		tlsph = true;
	}

	Matrix3d *T_ULSPH = (Matrix3d *) force->pair->extract("smd/ulsph/stressTensor_ptr", itmp);
	if (T_ULSPH != NULL) {
		ulsph = true;
	}

	Matrix3d *T_MPM = (Matrix3d *) force->pair->extract("smd/mpm/stressTensor_ptr", itmp);
	if (T_MPM != NULL) {
		mpm = true;
	}

	//std::cout << tlsph << ulsph << mpm;
	if (!((tlsph) || (ulsph) || (mpm))) {
		error->all(FLERR,
				"compute smd/stress could not access any stress tensors. Are the matching pair styles (ulsph, tlsph, mpm) present?");
	}

	int nlocal = atom->nlocal;
	int *mask = atom->mask;

	for (int i = 0; i < nlocal; i++) {
		if (mask[i] & groupbit) {

			stress.setZero();

			if (tlsph) {
				stress += T_TLSPH[i];
			}

			if (ulsph) {
				stress += T_ULSPH[i];
			}

			if (mpm) {
				stress += T_MPM[i];
			}

			// compute von mises stress
			stress_deviator = Deviator(stress);
			von_mises_stress = sqrt(3. / 2.) * stress_deviator.norm();

			stress_array[i][0] = stress(0, 0);
			stress_array[i][1] = stress(1, 1);
			stress_array[i][2] = stress(2, 2);
			stress_array[i][3] = stress(0, 1);
			stress_array[i][4] = stress(0, 2);
			stress_array[i][5] = stress(1, 2);
			stress_array[i][6] = von_mises_stress;

		} else {
			for (int j = 0; j < size_peratom_cols; j++) {
				stress_array[i][j] = 0.0;
			}
		}
	}
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeSmdStress::memory_usage() {
	double bytes = size_peratom_cols * nmax * sizeof(double);
	return bytes;
}

/*
 * deviator of a tensor
 */
Matrix3d ComputeSmdStress::Deviator(Matrix3d M) {
	Matrix3d eye;
	eye.setIdentity();
	eye *= M.trace() / 3.0;
	return M - eye;
}
