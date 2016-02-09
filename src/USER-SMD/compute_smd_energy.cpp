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
#include "compute_smd_energy.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "SmdMatDB.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeSmdEnergy::ComputeSmdEnergy(LAMMPS *lmp, int narg, char **arg) :
		Compute(lmp, narg, arg) {
	if (narg != 3)
		error->all(FLERR, "Illegal compute smd/internal_energy command");
	if (atom->e_flag != 1)
		error->all(FLERR, "compute smd/internal_energy command requires atom_style with internal_energy (e.g. smd)");

	peratom_flag = 1;
	size_peratom_cols = 3;

	nmax = 0;
	energy_array = NULL;

	int retcode = SmdMatDB::instance().ReadMaterials(atom->ntypes);
	if (retcode < 0) {
		error->one(FLERR, "failed to read material database");
	}
}

/* ---------------------------------------------------------------------- */

ComputeSmdEnergy::~ComputeSmdEnergy() {
	memory->sfree(energy_array);
}

/* ---------------------------------------------------------------------- */

void ComputeSmdEnergy::init() {

	int count = 0;
	for (int i = 0; i < modify->ncompute; i++)
		if (strcmp(modify->compute[i]->style, "smd/internal_energy") == 0)
			count++;
	if (count > 1 && comm->me == 0)
		error->warning(FLERR, "More than one compute smd/internal_energy");
}

/* ---------------------------------------------------------------------- */

void ComputeSmdEnergy::compute_peratom() {
	invoked_peratom = update->ntimestep;

	// grow rhoVector array if necessary

	if (atom->nlocal > nmax) {
		memory->destroy(energy_array);
		nmax = atom->nmax;
		memory->create(energy_array, nmax, size_peratom_cols, "compute:smd_energies");
		array_atom = energy_array;
	}

	double *e = atom->e;
	double *heat = atom->heat;
	double *rmass = atom->rmass;
	int *mask = atom->mask;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	int itype;

	//printf("init? %d\n", SmdMatDB::instance().initialized);

	for (int i = 0; i < nlocal; i++) {
		if (mask[i] & groupbit) {
			itype = type[i];
			energy_array[i][0] = e[i];
			energy_array[i][1] = heat[i];
			//printf("itype = %d\n", itype);
			if (SmdMatDB::instance().initialized) {
				energy_array[i][2] = heat[i] / (rmass[i] * SmdMatDB::instance().gProps[itype].cp);
			} else {
				energy_array[i][2] = 0.0;
			}
		} else {
			energy_array[i][0] = 0.0;
			energy_array[i][1] = 0.0;
			energy_array[i][2] = 0.0;
		}
	}
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeSmdEnergy::memory_usage() {
	double bytes = nmax * sizeof(double);
	return bytes;
}
