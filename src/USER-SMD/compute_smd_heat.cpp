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
#include "compute_smd_heat.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeSMDHeat::ComputeSMDHeat(LAMMPS *lmp, int narg, char **arg) :
		Compute(lmp, narg, arg) {
	if (narg != 3)
		error->all(FLERR, "Illegal compute smd/heat command");
	if (atom->vfrac_flag != 1)
		error->all(FLERR, "compute smd/heat command requires atom_style with heat (e.g. smd)");

	peratom_flag = 1;
	size_peratom_cols = 0;

	nmax = 0;
	heatVector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSMDHeat::~ComputeSMDHeat() {
	memory->sfree(heatVector);
}

/* ---------------------------------------------------------------------- */

void ComputeSMDHeat::init() {

	int count = 0;
	for (int i = 0; i < modify->ncompute; i++)
		if (strcmp(modify->compute[i]->style, "smd/heat") == 0)
			count++;
	if (count > 1 && comm->me == 0)
		error->warning(FLERR, "More than one compute smd/heat");
}

/* ---------------------------------------------------------------------- */

void ComputeSMDHeat::compute_peratom() {
	invoked_peratom = update->ntimestep;

	// grow heatVector array if necessary

	if (atom->nlocal > nmax) {
		memory->sfree(heatVector);
		nmax = atom->nmax;
		heatVector = (double *) memory->smalloc(nmax * sizeof(double), "atom:heatVector");
		vector_atom = heatVector;
	}

	double *heat = atom->heat;
	int *mask = atom->mask;
	int nlocal = atom->nlocal;

	for (int i = 0; i < nlocal; i++) {
		if (mask[i] & groupbit) {
			heatVector[i] = heat[i];
		} else {
			heatVector[i] = 0.0;
		}
	}

}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeSMDHeat::memory_usage() {
	double bytes = nmax * sizeof(double);
	return bytes;
}
