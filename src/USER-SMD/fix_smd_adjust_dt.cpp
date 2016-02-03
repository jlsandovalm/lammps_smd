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
#include "stdlib.h"
#include "string.h"
#include "fix_smd_adjust_dt.h"
#include "atom.h"
#include "update.h"
#include "integrate.h"
#include "domain.h"
#include "lattice.h"
#include "force.h"
#include "pair.h"
#include "modify.h"
#include "fix.h"
#include "output.h"
#include "dump.h"
#include "comm.h"
#include "error.h"
#include "run.h"
using namespace LAMMPS_NS;
using namespace FixConst;

#define BIG 1.0e20

/* ---------------------------------------------------------------------- */

FixSMDTlsphDtReset::FixSMDTlsphDtReset(LAMMPS *lmp, int narg, char **arg) :
		Fix(lmp, narg, arg) {
	if (narg != 4)
		error->all(FLERR, "Illegal fix smd/adjust_dt command. Synatx: fix _name_ smd/adjust_dt all CFL_factor");

	// set time_depend, else elapsed time accumulation can be messed up

	time_depend = 1;
	scalar_flag = 1;
	vector_flag = 1;
	size_vector = NUM; // there are NUM different methods
	global_freq = 1;
	extscalar = 0;
	extvector = 0;
	restart_global = 1; // this fix stores global (i.e., not per-atom) info: elaspsed time

	safety_factor = force->numeric(FLERR, arg[3]);

	// initializations
	t_elapsed = 0.0;
}

/* ---------------------------------------------------------------------- */

int FixSMDTlsphDtReset::setmask() {
	int mask = 0;
	mask |= INITIAL_INTEGRATE;
	mask |= END_OF_STEP;
	return mask;
}

/* ---------------------------------------------------------------------- */

void FixSMDTlsphDtReset::init() {
	dt = update->dt;
}

/* ---------------------------------------------------------------------- */

void FixSMDTlsphDtReset::setup(int vflag) {
	end_of_step();
}

/* ---------------------------------------------------------------------- */

void FixSMDTlsphDtReset::initial_integrate(int vflag) {

	//printf("in adjust_dt: dt = %20.10f\n", update->dt);

	//t_elapsed += update->dt;
}

/* ---------------------------------------------------------------------- */

void FixSMDTlsphDtReset::end_of_step() {
	double dtmin = BIG;
	int itmp = 0;

	/*
	 * extract minimum CFL timestep from TLSPH and ULSPH pair styles
	 */

	double *dtCFL_TLSPH = (double *) force->pair->extract("smd/tlsph/dtCFL_ptr", itmp);
	double *dtCFL_ULSPH = (double *) force->pair->extract("smd/ulsph/dtCFL_ptr", itmp);
	double *dt_TRI = (double *) force->pair->extract("smd/tri_surface/stable_time_increment_ptr", itmp);
	double *dt_HERTZ = (double *) force->pair->extract("smd/hertz/stable_time_increment_ptr", itmp);
	double *dt_PERI_IPMB = (double *) force->pair->extract("smd/peri_ipmb/stable_time_increment_ptr", itmp);
	double *dt_MPM = (double *) force->pair->extract("smd/mpm/dtCFL_ptr", itmp);

	if ((dtCFL_TLSPH == NULL) && (dtCFL_ULSPH == NULL) && (dt_TRI == NULL) && (dt_HERTZ == NULL) && (dt_PERI_IPMB == NULL)
			&& (dt_MPM == NULL)) {
		error->all(FLERR, "fix smd/adjust_dt failed to access a valid dtCFL");
	}

	double dt_array[NUM];
	if (dtCFL_TLSPH != NULL) {
		dtmin = MIN(dtmin, *dtCFL_TLSPH);
		dt_array[TLSPH] = *dtCFL_TLSPH;
	}

	if (dtCFL_ULSPH != NULL) {
		dtmin = MIN(dtmin, *dtCFL_ULSPH);
		dt_array[ULSPH] = *dtCFL_ULSPH;
	}

	if (dt_TRI != NULL) {
		dtmin = MIN(dtmin, *dt_TRI);
		dt_array[TRI] = *dt_TRI;
	}

	if (dt_HERTZ != NULL) {
		dtmin = MIN(dtmin, *dt_HERTZ);
		dt_array[HERTZ] = *dt_HERTZ;
	}

	if (dt_PERI_IPMB != NULL) {
		dtmin = MIN(dtmin, *dt_PERI_IPMB);
		dt_array[PERI_IPMB] = *dt_PERI_IPMB;
	}

	if (dt_MPM != NULL) {
		dtmin = MIN(dtmin, *dt_MPM);
		dt_array[MPM] = *dt_MPM;
		//printf("MPM timestep is %f, index is %d\n", dt_array[MPM], MPM);
	}

	MPI_Allreduce(&dt_array, &reduced_dt_array, NUM, MPI_DOUBLE, MPI_MIN, world);

//	printf("reduced, local\n");
//	for (int i = 1; i < NUM; i++) {
//		if (i == PERI_IPMB) {
//			printf("PERI_IPMB: %f %f\n", reduced_dt_array[i], dt_array[i]);
//		}
//	}

//	double **f = atom->f;
//	double *rmass = atom->rmass;
//	double *radius = atom->radius;
//	int *mask = atom->mask;
//	int nlocal = atom->nlocal;
//	double dtf, fsq, massinv;
//
//	for (int i = 0; i < nlocal; i++) {
//		if (mask[i] & groupbit) {
//			massinv = 1.0 / rmass[i];
//			fsq = f[i][0] * f[i][0] + f[i][1] * f[i][1] + f[i][2] * f[i][2];
//			if (fsq > 0.0) {
//				dtf = sqrt(2.0 * radius[i] / (sqrt(fsq) * massinv));
//			} else {
//				dtf = BIG;
//			}
//			dtmin = MIN(dtmin, dtf);
//		}
//	}

	dtmin *= safety_factor; // apply safety factor
	MPI_Allreduce(&dtmin, &dt, 1, MPI_DOUBLE, MPI_MIN, world);

	if (update->ntimestep == 0) {
		dt = 1.0e-16;
	}

	//printf("dtmin is now: %f, dt is now%f\n", dtmin, dt);

	if (update->run_duration > 0.0) {
		if (update->elapsed_time_in_run + dt >= update->run_duration) {
			dt = update->run_duration - update->elapsed_time_in_run;
			dt = MAX(dt, 1.0e-16);
		}
	}

	update->dt = dt;
	update->elapsed_time_in_run += update->dt;
	t_elapsed += dt;
	if (force->pair)
		force->pair->reset_dt();
	for (int i = 0; i < modify->nfix; i++)
		modify->fix[i]->reset_dt();
	update->atime = t_elapsed;
	update->atimestep = update->ntimestep;

}

/* ---------------------------------------------------------------------- */

double FixSMDTlsphDtReset::compute_scalar() {
	return t_elapsed;
}
/* ---------------------------------------------------------------------- */

double FixSMDTlsphDtReset::compute_vector(int n) {
	// return minimum timesteps for each method in this order:
	// enum {TLSPH = 0, ULSPH = 1, TRI = 2, HERTZ = 3, PERI_IPMB = 4, MPM = 5, NUM = 6};

	return safety_factor * reduced_dt_array[n];
}

/* ----------------------------------------------------------------------
 pack entire state of Fix into one write
 ------------------------------------------------------------------------- */

void FixSMDTlsphDtReset::write_restart(FILE *fp) {
	int n = 0;
	double list[1];
	list[n++] = t_elapsed;

	if (comm->me == 0) {
		int size = n * sizeof(double);
		fwrite(&size, sizeof(int), 1, fp);
		fwrite(list, sizeof(double), n, fp);
	}
}

/* ----------------------------------------------------------------------
 use state info from restart file to restart the Fix
 ------------------------------------------------------------------------- */

void FixSMDTlsphDtReset::restart(char *buf) {
	int n = 0;
	double *list = (double *) buf;
	t_elapsed = list[n++];
}

