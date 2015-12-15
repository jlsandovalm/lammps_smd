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

#include "stdio.h"
#include "string.h"
#include "fix_smd_integrate_mpm.h"
#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "integrate.h"
#include "respa.h"
#include "memory.h"
#include "error.h"
#include "pair.h"
#include "domain.h"
#include "region.h"
#include <Eigen/Eigen>
#include <iostream>

using namespace Eigen;
using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixSMDIntegrateMpm::FixSMDIntegrateMpm(LAMMPS *lmp, int narg, char **arg) :
		Fix(lmp, narg, arg) {

	if ((atom->e_flag != 1) || (atom->vfrac_flag != 1))
		error->all(FLERR, "fix smd/integrate_mpm command requires atom_style with both energy and volume");

	if (narg < 3)
		error->all(FLERR, "Illegal number of arguments for fix smd/integrate_mpm command");

	adjust_radius_flag = false;
	vlimit = -1.0;
	flip_contribution = 0.99;
	region_flag = 0;
	int iarg = 3;

	if (comm->me == 0) {
		printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n");
		printf("fix smd/integrate_mpm is active for group: %s \n", arg[1]);
	}
	while (true) {

		if (iarg >= narg) {
			break;
		}

		if (strcmp(arg[iarg], "limit_velocity") == 0) {
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
			flip_contribution = force->numeric(FLERR, arg[iarg]);

			if (comm->me == 0) {
				printf("... will use %3.2f FLIP and %3.2f PIC update for velocities\n", flip_contribution, 1.0 - flip_contribution);
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
			domain->regions[nregion]->init();
			domain->regions[nregion]->prematch();
			region_flag = 1;
		} else {
			char msg[128];
			sprintf(msg, "Illegal keyword for smd/integrate_mpm: %s\n", arg[iarg]);
			error->all(FLERR, msg);
		}

		iarg++;

	}

	if (comm->me == 0) {
		printf("... will use %3.2f FLIP and %3.2f PIC update for velocities\n", flip_contribution, 1.0 - flip_contribution);
		printf(">>========>>========>>========>>========>>========>>========>>========>>========\n\n");
	}

	// set comm sizes needed by this fix
	atom->add_callback(0);

	time_integrate = 1;
}

/* ---------------------------------------------------------------------- */

int FixSMDIntegrateMpm::setmask() {
	int mask = 0;
	mask |= INITIAL_INTEGRATE;
	mask |= FINAL_INTEGRATE;
	return mask;
}

/* ---------------------------------------------------------------------- */

void FixSMDIntegrateMpm::init() {
	dtv = update->dt;
	dtf = 0.5 * update->dt * force->ftm2v;
	vlimitsq = vlimit * vlimit;
}

/* ----------------------------------------------------------------------
 allow for both per-type and per-atom mass
 ------------------------------------------------------------------------- */

void FixSMDIntegrateMpm::initial_integrate(int vflag) {


	// there is nothing to do here -- everything is handled in final-integrate
	return;

//	double **x = atom->x;
//	double **vest = atom->vest;
//	int *mask = atom->mask;
//	int nlocal = atom->nlocal;
//	tagint *mol = atom->molecule;
//
//	int i;
//
//	if (igroup == atom->firstgroup)
//		nlocal = atom->nfirst;
//
//	for (i = 0; i < nlocal; i++) {
//		if (mask[i] & groupbit) {
//
//			//if (mol[i] != 1000) {
////			x[i][0] += dtv * vest[i][0];
////			x[i][1] += dtv * vest[i][1];
////			x[i][2] += dtv * vest[i][2];
//			//}
//
//		}
//	}

}

/* ---------------------------------------------------------------------- */

void FixSMDIntegrateMpm::final_integrate() {

	double **x = atom->x;
	double **v = atom->v;
	double **vest = atom->vest;
	double *e = atom->e;
	double *heat = atom->heat;
	double *de = atom->de;
	tagint *mol = atom->molecule;

	int *mask = atom->mask;
	int nlocal = atom->nlocal;
	double vsq, scale;
	int i, itmp;

	Vector3d *particleVelocities = (Vector3d *) force->pair->extract("smd/mpm/particleVelocities_ptr", itmp);
	if (particleVelocities == NULL) {
		error->one(FLERR, "fix smd/integrate_mpm failed to accesss particleVelocities array");
	}

	Vector3d *particleAccelerations = (Vector3d *) force->pair->extract("smd/mpm/particleAccelerations_ptr", itmp);
	if (particleAccelerations == NULL) {
		error->one(FLERR, "fix smd/integrate_mpm failed to accesss particleAccelerations array");
	}

	double *particleHeat = (double *) force->pair->extract("smd/mpm/particleHeat_ptr", itmp);
	if (particleHeat == NULL) {
		error->one(FLERR, "fix smd/integrate_mpm failed to accesss particleHeat array");
	}

	double *particleHeatRate = (double *) force->pair->extract("smd/mpm/particleHeatRate_ptr", itmp);
	if (particleHeatRate == NULL) {
		error->one(FLERR, "fix smd/integrate_mpm failed to accesss particleHeatRate array");
	}

	if (igroup == atom->firstgroup)
		nlocal = atom->nfirst;

	for (i = 0; i < nlocal; i++) {
		if (mask[i] & groupbit) {

			if (mol[i] == 1000) {
				// this is zero-acceleration (constant velocity) time integration for boundary particles
				x[i][0] += dtv * v[i][0];
				x[i][1] += dtv * v[i][1];
				x[i][2] += dtv * v[i][2];

				vest[i][0] = v[i][0];
				vest[i][1] = v[i][1];
				vest[i][2] = v[i][2];

			} else {

				if (vlimit > 0.0) {
					vsq = v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2];
					if (vsq > vlimitsq) {
						scale = sqrt(vlimitsq / vsq);
						v[i][0] *= scale;
						v[i][1] *= scale;
						v[i][2] *= scale;
					}
				}

				if (region_flag == 0) {
					// mixed FLIP-PIC
					v[i][0] = (1. - flip_contribution) * particleVelocities[i](0)
							+ flip_contribution * (v[i][0] + dtv * particleAccelerations[i](0));
					v[i][1] = (1. - flip_contribution) * particleVelocities[i](1)
							+ flip_contribution * (v[i][1] + dtv * particleAccelerations[i](1));
					v[i][2] = (1. - flip_contribution) * particleVelocities[i](2)
							+ flip_contribution * (v[i][2] + dtv * particleAccelerations[i](2));
//
					x[i][0] += dtv * particleVelocities[i](0);
					x[i][1] += dtv * particleVelocities[i](1);
					x[i][2] += dtv * particleVelocities[i](2);

					//std::cout << "particle vel in integration is " << particleVelocities[i].transpose() << std::endl;

					vest[i][0] = particleVelocities[i](0) + 0.5*dtv * particleAccelerations[i](0); // these are the velocities used for
					vest[i][1] = particleVelocities[i](1) + 0.5*dtv * particleAccelerations[i](1); // computing the deformation gradient
					vest[i][2] = particleVelocities[i](2) + 0.5*dtv * particleAccelerations[i](2);

				} else {
					if (!domain->regions[nregion]->match(x[i][0], x[i][1], x[i][2])) {
						// mixed FLIP-PIC
						v[i][0] = (1. - flip_contribution) * particleVelocities[i](0)
								+ flip_contribution * (v[i][0] + dtv * particleAccelerations[i](0));
						v[i][1] = (1. - flip_contribution) * particleVelocities[i](1)
								+ flip_contribution * (v[i][1] + dtv * particleAccelerations[i](1));
						v[i][2] = (1. - flip_contribution) * particleVelocities[i](2)
								+ flip_contribution * (v[i][2] + dtv * particleAccelerations[i](2));

						x[i][0] += dtv * particleVelocities[i](0);
						x[i][1] += dtv * particleVelocities[i](1);
						x[i][2] += dtv * particleVelocities[i](2);

						vest[i][0] = particleVelocities[i](0) + dtv * particleAccelerations[i](0);
						vest[i][1] = particleVelocities[i](1) + dtv * particleAccelerations[i](1);
						vest[i][2] = particleVelocities[i](2) + dtv * particleAccelerations[i](2);

					} else {
						// this is zero-acceleration (constant velocity) time integration for boundary
						// for particles in a specified region, used e.g. with fix  smd/inflow
						x[i][0] += dtv * v[i][0];
						x[i][1] += dtv * v[i][1];
						x[i][2] += dtv * v[i][2];

						vest[i][0] = v[i][0];
						vest[i][1] = v[i][1];
						vest[i][2] = v[i][2];
					}
				}
			}

			e[i] += dtv * de[i];

			heat[i] = (1. - flip_contribution) * particleHeat[i] + flip_contribution * (heat[i] + dtv * particleHeatRate[i]);

		}
	}

}

/* ---------------------------------------------------------------------- */

void FixSMDIntegrateMpm::reset_dt() {
	dtv = update->dt;
	dtf = 0.5 * dtv;
}

