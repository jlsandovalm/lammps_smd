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

//#include "mpi.h"
#include "stdlib.h"
#include "string.h"
#include "smd_run_duration.h"
#include "update.h"
//#include "atom.h"
//#include "modify.h"
//#include "domain.h"
//#include "lattice.h"
//#include "comm.h"
//#include "irregular.h"
//#include "group.h"
#include "error.h"
#include "force.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

SmdRunDuration::SmdRunDuration(LAMMPS *lmp) :
		Pointers(lmp) {
}

/* ---------------------------------------------------------------------- */

void SmdRunDuration::command(int narg, char **arg) {

	if (narg != 1)
		error->all(FLERR, "Illegal smd_run_duration command");
	update->run_duration = force->numeric(FLERR, arg[0]);

}
