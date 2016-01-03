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
#include "string.h"
#include "stdlib.h"
#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "modify.h"
#include "domain.h"
#include "lattice.h"
#include "update.h"
#include "random_mars.h"
#include "error.h"
#include "fix_smd_inflow.h"
#include "region.h"
#include "force.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum {
	LAYOUT_UNIFORM, LAYOUT_NONUNIFORM, LAYOUT_TILED
};
// several files

#define BIG      1.0e30
#define EPSILON  1.0e-6
#define DEBUG false

/* ---------------------------------------------------------------------- */

FixSmdInflow::FixSmdInflow(LAMMPS *lmp, int narg, char **arg) :
		Fix(lmp, narg, arg) {
	force_reneighbor = 1;
	next_reneighbor = -1;
	box_change_size = 1;
	time_depend = 1;

	if (narg < 4)
		error->all(FLERR, "Illegal fix append/atoms command");

	xflag = yflag = zflag = 0;

	if (domain->lattice->nbasis != 1)
		error->all(FLERR, "Fix append/atoms requires a lattice with one basis atom to be defined");

	if ((domain->lattice->basis[0][0] != 0.0) || (domain->lattice->basis[0][1] != 0.0) || (domain->lattice->basis[0][2] != 0.0)) {
		error->all(FLERR, "Fix append/atoms requires a simple cubic lattice with a zero basis in 3d");
	}

	nbasis = domain->lattice->nbasis;
	basistype = new int[nbasis];
	for (int i = 0; i < nbasis; i++)
		basistype[i] = 1;

	particle_spacing = domain->lattice->xlattice;
	last_time = update->atime;
	velocity = 0.0;

	contact_radius_one = radius_one = particle_spacing; // default value
	heat_one = 1.0;
	rho_flag = velocity_flag = region_flag = type_flag = 0;
	first = true;

	int iarg = 0;
	iarg = 3;
	while (iarg < narg) {
		if (strcmp(arg[iarg], "rho") == 0) {
			if (iarg + 2 > narg)
				error->all(FLERR, "Illegal fix append/atoms command. Expected float following rho");
			rho = force->numeric(FLERR, arg[iarg + 1]);
			rho_flag = 1;
			iarg += 2;
		} else if (strcmp(arg[iarg], "heat") == 0) {
			if (iarg + 2 > narg)
				error->all(FLERR, "Illegal fix append/atoms command. Expected float following radius keyword");
			heat_one = force->numeric(FLERR, arg[iarg + 1]);
			iarg += 2;
		} else if (strcmp(arg[iarg], "type") == 0) {
			if (iarg + 2 > narg)
				error->all(FLERR, "Illegal fix append/atoms command. Expected float following radius keyword");
			type_one = force->inumeric(FLERR, arg[iarg + 1]);
			type_flag = 1;
			iarg += 2;
		} else if (strcmp(arg[iarg], "radius") == 0) {
			if (iarg + 2 > narg)
				error->all(FLERR, "Illegal fix append/atoms command. Expected float following radius keyword");
			radius_one = force->numeric(FLERR, arg[iarg + 1]);
			iarg += 2;
		} else if (strcmp(arg[iarg], "contact_radius") == 0) {
			if (iarg + 2 > narg)
				error->all(FLERR, "Illegal fix append/atoms command. Expected float following contact_radius keyword");
			contact_radius_one = force->numeric(FLERR, arg[iarg + 1]);
			iarg += 2;
		} else if (strcmp(arg[iarg], "freq") == 0) {
			if (iarg + 2 > narg)
				error->all(FLERR, "Illegal fix append/atoms command. Expected int following freq keyword");
			freq = force->inumeric(FLERR, arg[iarg + 1]);
			freq_flag = 1;
			iarg += 2;
		} else if (strcmp(arg[iarg], "region") == 0) {
			nregion = domain->find_region(arg[iarg + 1]);
			if (nregion == -1)
				error->all(FLERR, "Create_atoms region ID does not exist");
			int n = strlen(arg[iarg + 1]) + 1;
			idregion = new char[n];
			strcpy(idregion, arg[iarg + 1]);
			domain->regions[nregion]->init();
			domain->regions[nregion]->prematch();
			region_flag = 1;
			iarg += 2;
		} else if (strcmp(arg[iarg], "velocity") == 0) {
			if (iarg + 2 > narg)
				error->all(FLERR, "Illegal fix append/atoms command. Expected float following velocity keyword");
			velocity = force->numeric(FLERR, arg[iarg + 1]);
			velocity_flag = 1;
			iarg += 2;
		} else if (strcmp(arg[iarg], "direction") == 0) {
			if (iarg + 2 > narg)
				error->all(FLERR, "Illegal fix append/atoms command. Expected x or y or z following direction keyword");
			if (strcmp(arg[iarg + 1], "x") == 0) {
				xflag = 1;
			} else if (strcmp(arg[iarg + 1], "y") == 0) {
				yflag = 1;
			} else if (strcmp(arg[iarg + 1], "z") == 0) {
				zflag = 1;
			} else {
				error->all(FLERR, "Illegal fix append/atoms command. Expected x or y or z following direction keyword");
			}
			iarg += 2;
		} else
			error->all(FLERR, "Illegal fix append/atoms command");
	}

	// check if insertion direction has been correctly defined
	if (xflag + yflag + zflag != 1) {
		error->all(FLERR, "Illegal fix append/atoms command. check direction.");
	}

	if (type_flag != 1) {
		error->all(FLERR, "Fix smd/inflow requires type keyword.");
	}

	if (rho_flag != 1) {
		error->all(FLERR, "Fix smd/inflow requires rho keyword.");
	}

	if (freq_flag != 1) {
		error->all(FLERR, "Fix smd/inflow requires freq keyword.");
	}

	if (region_flag != 1) {
		error->all(FLERR, "Fix smd/inflow requires region keyword.");
	}

	if (velocity_flag != 1) {
		error->all(FLERR, "Fix smd/inflow requires velocity keyword.");
	}

	if (type_one > atom->ntypes || (type_one < 1)) {
		error->one(FLERR, "simulation box cannot accomodate supplied type for fix smd/inflow");
	}

	if (domain->triclinic == 1)
		error->all(FLERR, "Cannot append atoms to a triclinic box");

	// check insertion region for validity

	if (domain->regions[nregion]->bboxflag != 1) {
		error->all(FLERR,
				"insertion region for fix append atoms is of wrong type, use one for which a bounding box can be computed, e.g. block");
	}

	if (domain->dimension == 2) {
		volume_one = pow(particle_spacing, 2);
		if (zflag) {
			error->one(FLERR, "cannot insert in z direction for 2d simuation");
		}
	} else {
		volume_one = pow(particle_spacing, 3);
	}
	mass_one = rho * volume_one;

	extent_xlo = domain->regions[nregion]->extent_xlo;
	extent_xhi = domain->regions[nregion]->extent_xhi;
	extent_ylo = domain->regions[nregion]->extent_ylo;
	extent_yhi = domain->regions[nregion]->extent_yhi;
	extent_zlo = domain->regions[nregion]->extent_zlo;
	extent_zhi = domain->regions[nregion]->extent_zhi;

	double delx = extent_xhi - extent_xlo;
	double dely = extent_yhi - extent_ylo;
	double delz = extent_zhi - extent_zlo;

	if (xflag) {
		insertion_dimension = 0;
		if (delx > 0.999 * domain->lattice->xlattice) {
			error->all(FLERR,
					"insertion region is too deep in x direction. Reduce size so it is smaller than one lattice spacing.");
		}
	} else if (yflag) {
		insertion_dimension = 1;
		if (dely > 0.999 * domain->lattice->ylattice) {
			error->all(FLERR,
					"insertion region is too deep in y direction. Reduce size so it is smaller than one lattice spacing.");
		}
	} else if (zflag) {
		insertion_dimension = 2;
		if (delz > 0.999 * domain->lattice->zlattice) {
			printf("delz = %f\n", delz);
			error->all(FLERR,
					"insertion region is too deep in z direction. Reduce size so it is smaller than one lattice spacing.");
		}
	}

	/*
	 * determine insertion position
	 */

	double xmin, ymin, zmin, xmax, ymax, zmax;
	xmin = ymin = zmin = BIG;
	xmax = ymax = zmax = -BIG;

	domain->lattice->bbox(1, extent_xlo, extent_ylo, extent_zlo, xmin, ymin, zmin, xmax, ymax, zmax);
	domain->lattice->bbox(1, extent_xhi, extent_ylo, extent_zlo, xmin, ymin, zmin, xmax, ymax, zmax);
	domain->lattice->bbox(1, extent_xlo, extent_yhi, extent_zlo, xmin, ymin, zmin, xmax, ymax, zmax);
	domain->lattice->bbox(1, extent_xhi, extent_yhi, extent_zlo, xmin, ymin, zmin, xmax, ymax, zmax);
	domain->lattice->bbox(1, extent_xlo, extent_ylo, extent_zhi, xmin, ymin, zmin, xmax, ymax, zmax);
	domain->lattice->bbox(1, extent_xhi, extent_ylo, extent_zhi, xmin, ymin, zmin, xmax, ymax, zmax);
	domain->lattice->bbox(1, extent_xlo, extent_yhi, extent_zhi, xmin, ymin, zmin, xmax, ymax, zmax);
	domain->lattice->bbox(1, extent_xhi, extent_yhi, extent_zhi, xmin, ymin, zmin, xmax, ymax, zmax);

	int ilo, ihi, jlo, jhi, klo, khi;
	ilo = static_cast<int>(xmin);
	jlo = static_cast<int>(ymin);
	klo = static_cast<int>(zmin);
	ihi = static_cast<int>(xmax);
	jhi = static_cast<int>(ymax);
	khi = static_cast<int>(zmax);

	if (xflag) {
		insertion_height = last_insertion_height = ilo * domain->lattice->xlattice;
	} else if (zflag) {
		insertion_height = last_insertion_height = khi * domain->lattice->zlattice;
	}

	//printf("insertion height is at %f\n", insertion_height);

	//printf("z max index is %d -- position is %f\n", khi, khi * domain->lattice->zlattice);
	//printf("jhi = %d, khi = %d\n", klo, khi);

	if (comm->me == 0) {
		printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n");
		printf("fix smd/inflow is active for region: %s \n", idregion);
		printf("... will add particles every %d steps. \n", nevery);
		printf("... particles have velocity %f, volume %f, mass density %f\n", velocity, volume_one, rho);
		printf("... particles have mass %f, radius %f, contact_radius %f\n", mass_one, radius_one, contact_radius_one);
		printf("... particles have heat %f\n", heat_one);
		printf(">>========>>========>>========>>========>>========>>========>>========>>========\n");
	}

}

/* ---------------------------------------------------------------------- */

FixSmdInflow::~FixSmdInflow() {
	delete[] basistype;
}

/* ---------------------------------------------------------------------- */

int FixSmdInflow::setmask() {
	int mask = 0;
	mask |= PRE_EXCHANGE;
	mask |= INITIAL_INTEGRATE;
//mask |= POST_FORCE;
	return mask;
}

/* ---------------------------------------------------------------------- */

void FixSmdInflow::initial_integrate(int vflag) {
	if (update->ntimestep % freq == 0)
		next_reneighbor = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

void FixSmdInflow::setup(int vflag) {
	/*** CALL TO CREATE GROUP?  SEE POST_FORCE ***/
	post_force(vflag);
}

/* ---------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

void FixSmdInflow::post_force(int vflag) {

	return;
}

/* ---------------------------------------------------------------------- */

void FixSmdInflow::pre_exchange() {

	if (comm->layout == LAYOUT_TILED) {
		error->all(FLERR, "cannot use comm_style tiled with fix smd/inflow -- use comm_style brick\n");
	}

	int ntimestep = update->ntimestep;
	int addnode = 0;
	double *sublo = domain->sublo;
	double *subhi = domain->subhi;

	if (ntimestep % freq == 0) {
		//printf("proc %d enters pre-exchange\n", comm->me);

		double current_time = update->atime;
		double time_difference = current_time - last_time;
		double travel_distance = last_insertion_height + time_difference * velocity - insertion_height;
		double displacement_excess;
		if (first) {
			displacement_excess = 0.0;
		} else {
			displacement_excess = fabs(travel_distance) - particle_spacing;
		}

//printf("current time ")
		if (fabs(travel_distance) < particle_spacing) {
			// cannot insert now
//			printf(
//					"--- cant insert at timestep %d: time difference since last insertion is %f, particle have travelled distance %f, last insertion height is %f\n",
//					update->ntimestep, time_difference, travel_distance, last_insertion_height);
			//return;

		} else {
//			printf(
//					"++++ want to insert at timestep %d: time difference since last insertion is %f, particle have travelled distance %f, last insertion height is %f\n",
//					update->ntimestep, time_difference, travel_distance, last_insertion_height);
//			printf("velocity travelled distance is %f\n", time_difference * velocity);
//			printf("displacement excess is %f\n", displacement_excess);
			last_time = current_time;
			last_insertion_height = insertion_height - displacement_excess;
			//printf("proc %d at N=%ld sets last_insertion_height to %f, travel is %f, spacing is %f \n", comm->me, update->ntimestep, last_insertion_height, travel_distance, particle_spacing);

			/*
			 * the code below is for using the entire box for checking if an atom can be inserted
			 * we do not use this as we can instead use only the insertion volume to generate lattice indices
			 */

//				extent_xlo = domain->sublo[0];
//				extent_xhi = domain->subhi[0];
//				extent_ylo = domain->sublo[1];
//				extent_yhi = domain->subhi[1];
//				extent_zlo = domain->subhi[2];
//				extent_zhi = domain->subhi[2];
			double xmin, ymin, zmin, xmax, ymax, zmax;
			xmin = ymin = zmin = BIG;
			xmax = ymax = zmax = -BIG;

			domain->lattice->bbox(1, extent_xlo, extent_ylo, extent_zlo, xmin, ymin, zmin, xmax, ymax, zmax);
			domain->lattice->bbox(1, extent_xhi, extent_ylo, extent_zlo, xmin, ymin, zmin, xmax, ymax, zmax);
			domain->lattice->bbox(1, extent_xlo, extent_yhi, extent_zlo, xmin, ymin, zmin, xmax, ymax, zmax);
			domain->lattice->bbox(1, extent_xhi, extent_yhi, extent_zlo, xmin, ymin, zmin, xmax, ymax, zmax);
			domain->lattice->bbox(1, extent_xlo, extent_ylo, extent_zhi, xmin, ymin, zmin, xmax, ymax, zmax);
			domain->lattice->bbox(1, extent_xhi, extent_ylo, extent_zhi, xmin, ymin, zmin, xmax, ymax, zmax);
			domain->lattice->bbox(1, extent_xlo, extent_yhi, extent_zhi, xmin, ymin, zmin, xmax, ymax, zmax);
			domain->lattice->bbox(1, extent_xhi, extent_yhi, extent_zhi, xmin, ymin, zmin, xmax, ymax, zmax);

			int ilo, ihi, jlo, jhi, klo, khi;
			ilo = static_cast<int>(xmin);
			jlo = static_cast<int>(ymin);
			klo = static_cast<int>(zmin);
			ihi = static_cast<int>(xmax);
			jhi = static_cast<int>(ymax);
			khi = static_cast<int>(zmax);

			if (xmin < 0.0)
				ilo--;
			if (ymin < 0.0)
				jlo--;
			if (zmin < 0.0)
				klo--;

			double x[3];
			int i, j, k;

			for (k = klo; k <= khi; k++) {

//				printf("k = %d\n", k);

				for (j = jlo; j <= jhi; j++) {
					for (i = ilo; i <= ihi; i++) {
						x[0] = i;
						x[1] = j;
						x[2] = k;

						int flag = 0;

						// convert from lattice coords to box coords
						domain->lattice->lattice2box(x[0], x[1], x[2]);

						if (x[0] >= sublo[0]) {
							if (x[0] < subhi[0]) {
								if (x[1] >= sublo[1]) {
									if (x[1] < subhi[1]) {
										if (x[2] >= sublo[2]) {
											if (x[2] < subhi[2]) {
												flag = 1;
											} else {
												if (DEBUG)
													printf("z=%f > box hi =%f\n", x[2], subhi[2]);
											}
										} else {
											if (DEBUG)
												printf("z=%f > box hi =%f\n", x[2], sublo[2]);
										}
									} else {
										if (DEBUG)
											printf("y=%f > box hi =%f\n", x[1], subhi[1]);
									}
								} else {
									if (DEBUG)
										printf("y=%f > box hi =%f\n", x[1], sublo[1]);
								}
							} else {
								if (DEBUG)
									printf("x=%f > box hi =%f\n", x[0], subhi[0]);
							}
						} else {
							if (DEBUG)
								printf("x=%f < box x=%f\n", x[0], sublo[0]);
						}

						if ((flag == 0) && (DEBUG)) {
							printf("cannot insert pos = %f %f %f\n", x[0], x[1], x[2]);
							printf("box lo %f %f %f\n", sublo[0], sublo[1], sublo[2]);
							printf("box hi %f %f %f\n", subhi[0], subhi[1], subhi[2]);
						}

						if (!domain->regions[nregion]->match(x[0], x[1], x[2])) {
							//printf("no match for pos = %f %f %f\n", x[0], x[1], x[2]);
							flag = 0;
						}

						x[insertion_dimension] -= displacement_excess;

						//printf("**** STILL WANT TO INSERT 1\n");

						if (flag) {

							//printf("**** STILL WANT TO INSERT 2\n");

							addnode++;
							atom->avec->create_atom(basistype[0], x);
							//printf("inserting at position %f %f %f\n", x[0], x[1], x[2]);

							int nlocal = atom->nlocal;
							//printf("nlocal = %d, vol=%f, mass=%f\n", nlocal, volume_one, mass_one);
							int idx = nlocal - 1;
							double *vfrac = atom->vfrac;
							double *rmass = atom->rmass;
							double *heat = atom->heat;
							double *contact_radius = atom->contact_radius;
							double *radius = atom->radius;
							double **v = atom->v;
							double **vest = atom->vest;
							vfrac[idx] = volume_one;
							rmass[idx] = mass_one;
							radius[idx] = radius_one;
							contact_radius[idx] = contact_radius_one;
							heat[idx] = heat_one;

							v[idx][0] = vest[idx][0] = 0.0;
							v[idx][1] = vest[idx][1] = 0.0;
							v[idx][2] = vest[idx][2] = 0.0;
							v[idx][insertion_dimension] = vest[idx][insertion_dimension] = velocity;

						}
					}
				}
			}
			int addtotal = 0;
			//printf("proc %d is before barrier\n", comm->me);
			MPI_Barrier(world);
			MPI_Allreduce(&addnode, &addtotal, 1, MPI_INT, MPI_SUM, world);
			//printf("proc %d is after barrier\n", comm->me);

			if (addtotal) {
				first = false; // set first flag so insertion position is computed correctly upon next insert
				domain->reset_box();
				atom->natoms += addtotal;
				if (atom->natoms < 0 || atom->natoms > MAXBIGINT)
					error->all(FLERR, "Too many total atoms");
				if (atom->tag_enable)
					atom->tag_extend();
				if (atom->map_style) {
					atom->nghost = 0;
					atom->map_init();
					atom->map_set();
				}
			}
		}
	}
}
