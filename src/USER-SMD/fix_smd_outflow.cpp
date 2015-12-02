/* ----------------------------------------------------------------------
 *
 *                    *** Smooth Mach Dynamics ***
 *
 * This file is part of the USER-SMD package for LAMMPS.
 * Copyright (2014) Georg C. Ganzenmueller, georg.ganzenmueller@emi.fhg.de
 * Fraunhofer Ernst-Mach Institute for High-Speed Dynamics, EMI,
 * Eckerstrasse 4, D-79104 Freiburg i.Br, Germany.
 *
 * This file is based on fix evaporate from the MISC package.
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
#include "fix_smd_outflow.h"
#include "atom.h"
#include "atom_vec.h"
#include "molecule.h"
#include "update.h"
#include "domain.h"
#include "region.h"
#include "comm.h"
#include "force.h"
#include "group.h"
#include "random_park.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixSmdOutflow::FixSmdOutflow(LAMMPS *lmp, int narg, char **arg) :
		Fix(lmp, narg, arg) {
	if (narg != 5)
		error->all(FLERR, "Illegal number of arguments for fix smd/outflow");

	scalar_flag = 1;
	global_freq = 1;
	extscalar = 0;

	nevery = force->inumeric(FLERR, arg[3]);
	iregion = domain->find_region(arg[4]);
	int n = strlen(arg[4]) + 1;
	idregion = new char[n];
	strcpy(idregion, arg[4]);

	if (nevery <= 0)
		error->all(FLERR, "Illegal fix smd/outflow command");
	if (iregion == -1)
		error->all(FLERR, "Region ID for fix smd/outflow does not exist");

	// set up reneighboring

	force_reneighbor = 1;
	next_reneighbor = (update->ntimestep / nevery) * nevery + nevery;
	ndeleted = 0;

	nmax = 0;
	list = NULL;
	mark = NULL;

	if (comm->me == 0) {
		printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n");
		printf("fix smd/outflow is active for group: %s \n", arg[1]);
		printf("fix smd/outflow is active for region: %s \n", arg[4]);
		printf("... will remove particles every %d steps. \n", nevery);
		printf(">>========>>========>>========>>========>>========>>========>>========>>========\n");
	}

}

/* ---------------------------------------------------------------------- */

FixSmdOutflow::~FixSmdOutflow() {
	delete[] idregion;
	memory->destroy(list);
	memory->destroy(mark);
}

/* ---------------------------------------------------------------------- */

int FixSmdOutflow::setmask() {
	int mask = 0;
	mask |= PRE_EXCHANGE;
	return mask;
}

/* ---------------------------------------------------------------------- */

void FixSmdOutflow::init() {
	// set index and check validity of region

	iregion = domain->find_region(idregion);
	if (iregion == -1)
		error->all(FLERR, "Region ID for fix smd/outflow does not exist");

	// check that no deletable atoms are in atom->firstgroup
	// deleting such an atom would not leave firstgroup atoms first

	if (atom->firstgroup >= 0) {
		int *mask = atom->mask;
		int nlocal = atom->nlocal;
		int firstgroupbit = group->bitmask[atom->firstgroup];

		int flag = 0;
		for (int i = 0; i < nlocal; i++)
			if ((mask[i] & groupbit) && (mask[i] && firstgroupbit))
				flag = 1;

		int flagall;
		MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_SUM, world);

		if (flagall)
			error->all(FLERR, "Cannot evaporate atoms in atom_modify first group");
	}

}

/* ----------------------------------------------------------------------
 perform particle deletion
 done before exchange, borders, reneighbor
 so that ghost atoms and neighbor lists will be correct
 ------------------------------------------------------------------------- */

void FixSmdOutflow::pre_exchange() {
	int i, ndel;

	if (update->ntimestep != next_reneighbor)
		return;

	// grow list and mark arrays if necessary

	if (atom->nlocal > nmax) {
		memory->destroy(list);
		memory->destroy(mark);
		nmax = atom->nmax;
		memory->create(list, nmax, "evaporate:list");
		memory->create(mark, nmax, "evaporate:mark");
	}

	// ncount = # of deletable atoms in region that I own
	// nall = # on all procs
	// nbefore = # on procs before me
	// list[ncount] = list of local indices of atoms I can delete

	Region *region = domain->regions[iregion];
	region->prematch();

	double **x = atom->x;
	int *mask = atom->mask;
	int nlocal = atom->nlocal;

	int ncount = 0;
	for (i = 0; i < nlocal; i++) {
		if (mask[i] & groupbit) {
			if (region->match(x[i][0], x[i][1], x[i][2])) {
				list[ncount++] = i;
				mark[i] = 1;
			} else {
				mark[i] = 0;
			}
		}
	}

	int nall, nbefore;
	MPI_Allreduce(&ncount, &nall, 1, MPI_INT, MPI_SUM, world);
	MPI_Scan(&ncount, &nbefore, 1, MPI_INT, MPI_SUM, world);
	nbefore -= ncount;

	// ndel = total # of atom deletions, in or out of region
	// ndeltopo[1,2,3,4] = ditto for bonds, angles, dihedrals, impropers
	// mark[] = 1 if deleted

	ndel = 0;

	// delete my marked atoms
	// loop in reverse order to avoid copying marked atoms

	AtomVec *avec = atom->avec;

	for (i = nlocal - 1; i >= 0; i--) {
		if (mark[i]) {
			avec->copy(atom->nlocal - 1, i, 1);
			atom->nlocal--;
			ndel++;
		}
	}

	// reset global natoms and bonds, angles, etc
	// if global map exists, reset it now instead of waiting for comm
	// since deleting atoms messes up ghosts

	atom->natoms -= ndel;

	if (ndel && atom->map_style) {
		atom->nghost = 0;
		atom->map_init();
		atom->map_set();
	}

	// statistics

	ndeleted += ndel;
	next_reneighbor = update->ntimestep + nevery;
}

/* ----------------------------------------------------------------------
 return number of deleted particles
 ------------------------------------------------------------------------- */

double FixSmdOutflow::compute_scalar() {
	return 1.0 * ndeleted;
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based arrays
 ------------------------------------------------------------------------- */

double FixSmdOutflow::memory_usage() {
	double bytes = 2 * nmax * sizeof(int);
	return bytes;
}
