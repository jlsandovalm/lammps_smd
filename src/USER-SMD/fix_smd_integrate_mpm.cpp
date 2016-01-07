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

	vlimit = -1.0;
	FLIP_contribution = 0.99;
	region_flag = 0;
	flag3d = true;
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

		} else {
			char msg[128];
			sprintf(msg, "Illegal keyword for smd/integrate_mpm: %s\n", arg[iarg]);
			error->all(FLERR, msg);
		}

		iarg++;

	}

	PIC_contribution = 1.0 - FLIP_contribution;
	if (comm->me == 0) {
		printf("... will use %3.2f FLIP and %3.2f PIC update for velocities\n", FLIP_contribution, PIC_contribution);
		printf(">>========>>========>>========>>========>>========>>========>>========>>========\n\n");
	}

// set comm sizes needed by this fix
	atom->add_callback(0);

	time_integrate = 1;
}

/* ---------------------------------------------------------------------- */

int FixSMDIntegrateMpm::setmask() {
	int mask = 0;
	mask |= FINAL_INTEGRATE;
	return mask;
}

/* ---------------------------------------------------------------------- */

void FixSMDIntegrateMpm::init() {
	dtv = update->dt;
	vlimitsq = vlimit * vlimit;
}

/* ---------------------------------------------------------------------- */

void FixSMDIntegrateMpm::final_integrate() {

//return; // do nothing for now because everything is handeled in MUSL in pair style

	double **x = atom->x;
	double **v = atom->v;
	double **vest = atom->vest;
	double *e = atom->e;
	double *heat = atom->heat;
	double *de = atom->de;
	tagint *mol = atom->molecule;

	int *mask = atom->mask;
	int nlocal = atom->nlocal;
	double vsq, scale, pic_vel_correction_x, pic_vel_correction_y, pic_vel_correction_z;
	int i, itmp, mode;

	Vector3d *GridToParticleVelocities = (Vector3d *) force->pair->extract("smd/mpm/particleVelocities_ptr", itmp);
	if (GridToParticleVelocities == NULL) {
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

	if (region_flag == 1) {
		//domain->regions[nregion]->init();
		domain->regions[nregion]->prematch();
	}

	if (igroup == atom->firstgroup)
		nlocal = atom->nfirst;

	for (i = 0; i < nlocal; i++) {
		if (mask[i] & groupbit) {

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

				// PIC position update correction
				//pic_vel_correction_x = -0.5 * PIC_contribution * (v[i][0] - GridToParticleVelocities[i](0));
				//pic_vel_correction_y = -0.5 * PIC_contribution * (v[i][1] - GridToParticleVelocities[i](1));
				//pic_vel_correction_z = -0.5 * PIC_contribution * (v[i][2] - GridToParticleVelocities[i](2));
				pic_vel_correction_x = pic_vel_correction_y = pic_vel_correction_z = 0.0; // deactivated for now.

				// mixed FLIP-PIC update of velocities
				v[i][0] = PIC_contribution * GridToParticleVelocities[i](0)
						+ FLIP_contribution * (v[i][0] + dtv * particleAccelerations[i](0));
				v[i][1] = PIC_contribution * GridToParticleVelocities[i](1)
						+ FLIP_contribution * (v[i][1] + dtv * particleAccelerations[i](1));
				if (flag3d) v[i][2] = PIC_contribution * GridToParticleVelocities[i](2)
						+ FLIP_contribution * (v[i][2] + dtv * particleAccelerations[i](2));

				// particles are moved according to interpolated grid velocities
				x[i][0] += dtv * (GridToParticleVelocities[i](0) + pic_vel_correction_x);
				x[i][1] += dtv * (GridToParticleVelocities[i](1) + pic_vel_correction_y);
				if (flag3d) x[i][2] += dtv * (GridToParticleVelocities[i](2) + pic_vel_correction_z);

				// extrapolate velocities to next timestep
				vest[i][0] = GridToParticleVelocities[i](0) + pic_vel_correction_x + dtv * particleAccelerations[i](0);
				vest[i][1] = GridToParticleVelocities[i](1) + pic_vel_correction_y + dtv * particleAccelerations[i](1);
				if (flag3d) vest[i][2] = GridToParticleVelocities[i](2) + pic_vel_correction_z + dtv * particleAccelerations[i](2);

				if (vlimit > 0.0) {
					vsq = v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2];
					if (vsq > vlimitsq) {
						scale = sqrt(vlimitsq / vsq);
						v[i][0] *= scale;
						v[i][1] *= scale;
						if (flag3d) v[i][2] *= scale;
					}
				}

				heat[i] = PIC_contribution * particleHeat[i] + FLIP_contribution * (heat[i] + dtv * particleHeatRate[i]);
				//heat[i] = heat[i] + dtv * particleHeatRate[i];

			} else if (mode == PRESCRIBED_VELOCITY) {

//				int code = domain->regions[nregion]->match(x[i][0], x[i][1], x[i][2]);
//				printf("mol=%d, particle at x=%f moves at const vel, code is %d\n", mol[i], x[i][0], code);
//				printf("REGION BBOX: %f %f\n", domain->regions[nregion]->extent_xlo, domain->regions[nregion]->extent_xhi);

				// this is zero-acceleration (constant velocity) time integration for boundary particles
				x[i][0] += dtv * const_vx;
				x[i][1] += dtv * const_vy;
				if (flag3d) x[i][2] += dtv * const_vz;

				v[i][0] = const_vx;
				v[i][1] = const_vy;
				if (flag3d) v[i][2] = const_vz;

				vest[i][0] = const_vx;
				vest[i][1] = const_vy;
				if (flag3d) vest[i][2] = const_vz;

			}

			else if (mode == CONSTANT_VELOCITY) {

				// this is zero-acceleration (constant velocity) time integration for boundary particles
				x[i][0] += dtv * v[i][0];
				x[i][1] += dtv * v[i][1];
				if (flag3d) x[i][2] += dtv * v[i][2];

				vest[i][0] = v[i][0];
				vest[i][1] = v[i][1];
				if (flag3d) vest[i][2] = v[i][0];

			}

		}

		e[i] += dtv * de[i];
	}
}

/* ---------------------------------------------------------------------- */

void FixSMDIntegrateMpm::reset_dt() {
	dtv = update->dt;
}

