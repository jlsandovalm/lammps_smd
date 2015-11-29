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

/* ---------------------------------------------------------------------- */

FixSmdInflow::FixSmdInflow(LAMMPS *lmp, int narg, char **arg) :
		Fix(lmp, narg, arg) {
	force_reneighbor = 1;
	next_reneighbor = -1;
	box_change_size = 1;
	time_depend = 1;

	if (narg < 4)
		error->all(FLERR, "Illegal fix append/atoms command");

	// default settings

	scaleflag = 1;
	spatflag = 0;
	xloflag = xhiflag = yloflag = yhiflag = zloflag = zhiflag = 0;

	tempflag = 0;

	ranflag = 0;
	ranx = 0.0;
	rany = 0.0;
	ranz = 0.0;

	randomx = NULL;
	randomt = NULL;

	if (domain->lattice->nbasis == 0)
		error->all(FLERR, "Fix append/atoms requires a lattice be defined");

	nbasis = domain->lattice->nbasis;
	basistype = new int[nbasis];
	for (int i = 0; i < nbasis; i++)
		basistype[i] = 1;

	particle_spacing = domain->lattice->xlattice;
	last_time = update->atime;
	velocity = 0.0;
	insertion_height = last_insertion_height = 10.0;
	first = true;

	int iarg = 0;
	iarg = 3;
	while (iarg < narg) {
		if (strcmp(arg[iarg], "rho") == 0) {
			if (iarg + 2 > narg)
				error->all(FLERR, "Illegal fix append/atoms command. Expected float following rho");
			rho = force->numeric(FLERR, arg[iarg + 1]);
			iarg += 2;
		} else if (strcmp(arg[iarg], "radius") == 0) {
			if (iarg + 2 > narg)
				error->all(FLERR, "Illegal fix append/atoms command. Expected float following radius keyword");
			radius_one = force->numeric(FLERR, arg[iarg + 1]);
			iarg += 2;
		} else if (strcmp(arg[iarg], "freq") == 0) {
			if (iarg + 2 > narg)
				error->all(FLERR, "Illegal fix append/atoms command. Expected int following freq keyword");
			freq = force->inumeric(FLERR, arg[iarg + 1]);
			iarg += 2;
		} else if (strcmp(arg[iarg], "region") == 0) {
			nregion = domain->find_region(arg[iarg + 1]);
			if (nregion == -1)
				error->all(FLERR, "Create_atoms region ID does not exist");
			domain->regions[nregion]->init();
			domain->regions[nregion]->prematch();
			iarg += 2;
		} else if (strcmp(arg[iarg], "velocity") == 0) {
			if (iarg + 2 > narg)
				error->all(FLERR, "Illegal fix append/atoms command. Expected float following velocity keyword");
			velocity = force->numeric(FLERR, arg[iarg + 1]);
			iarg += 2;
		} else if (strcmp(arg[iarg], "units") == 0) {
			if (iarg + 2 > narg)
				error->all(FLERR, "Illegal fix append/atoms command");
			if (strcmp(arg[iarg + 1], "box") == 0)
				scaleflag = 0;
			else if (strcmp(arg[iarg + 1], "lattice") == 0)
				scaleflag = 1;
			else
				error->all(FLERR, "Illegal fix append/atoms command");
			iarg += 2;
		} else
			error->all(FLERR, "Illegal fix append/atoms command");
	}

	if ((xloflag || xhiflag) && domain->xperiodic)
		error->all(FLERR, "Cannot use append/atoms in periodic dimension");
	if ((yloflag || yhiflag) && domain->yperiodic)
		error->all(FLERR, "Cannot use append/atoms in periodic dimension");
	if ((zloflag || zhiflag) && domain->zperiodic)
		error->all(FLERR, "Cannot use append/atoms in periodic dimension");

	if (domain->triclinic == 1)
		error->all(FLERR, "Cannot append atoms to a triclinic box");

	// setup scaling

	double xscale, yscale, zscale;
	if (scaleflag) {
		xscale = domain->lattice->xlattice;
		yscale = domain->lattice->ylattice;
		zscale = domain->lattice->zlattice;
	} else
		xscale = yscale = zscale = 1.0;

	if (xloflag || xhiflag)
		size *= xscale;
	if (yloflag || yhiflag)
		size *= yscale;
	if (zloflag || zhiflag)
		size *= zscale;

	if (ranflag) {
		ranx *= xscale;
		rany *= yscale;
		ranz *= zscale;
	}
}

/* ---------------------------------------------------------------------- */

FixSmdInflow::~FixSmdInflow() {
	delete[] basistype;

	if (ranflag)
		delete randomx;
	if (tempflag) {
		delete randomt;
		delete[] gfactor1;
		delete[] gfactor2;
	}
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

int FixSmdInflow::get_spatial() {

	error->one(FLERR, "get_spatial");

	if (update->ntimestep % freq == 0) {
		int ifix = modify->find_fix(spatialid);
		if (ifix < 0)
			error->all(FLERR, "Fix ID for fix ave/spatial does not exist");
		Fix *fix = modify->fix[ifix];

		int failed = 0;
		int count = 0;
		while (failed < 2) {
			double tmp = fix->compute_vector(2 * count);
			if (tmp == 0.0)
				failed++;
			else
				failed = 0;
			count++;
		}
		double *pos = new double[count - 2];
		double *val = new double[count - 2];
		for (int loop = 0; loop < count - 2; loop++) {
			pos[loop] = fix->compute_vector(2 * loop);
			val[loop] = fix->compute_vector(2 * loop + 1);
		}

		// always ignore the first and last

		double binsize = 2.0;
		double min_energy = 0.0;
		double max_energy = 0.0;
		int header = static_cast<int>(size / binsize);
		advance = 0;

		for (int loop = 1; loop <= header; loop++) {
			max_energy += val[loop];
		}
		for (int loop = count - 2 - 2 * header; loop <= count - 3 - header; loop++) {
			min_energy += val[loop];
		}
		max_energy /= header;
		min_energy /= header;

		double shockfront_min = 0.0;
		double shockfront_max = 0.0;
		double shockfront_loc = 0.0;
		int front_found1 = 0;
		for (int loop = count - 3 - header; loop > header; loop--) {
			if (front_found1 == 1)
				continue;
			if (val[loop] > min_energy + 0.1 * (max_energy - min_energy)) {
				shockfront_max = pos[loop];
				front_found1 = 1;
			}
		}
		int front_found2 = 0;
		for (int loop = header + 1; loop <= count - 3 - header; loop++) {
			if (val[loop] > min_energy + 0.6 * (max_energy - min_energy)) {
				shockfront_min = pos[loop];
				front_found2 = 1;
			}
		}
		if (front_found1 + front_found2 == 0)
			shockfront_loc = 0.0;
		else if (front_found1 + front_found2 == 1)
			shockfront_loc = shockfront_max + shockfront_min;
		else if (front_found1 == 1 && front_found2 == 1 && shockfront_max - shockfront_min > spatlead / 2.0)
			shockfront_loc = shockfront_max;
		else
			shockfront_loc = (shockfront_max + shockfront_min) / 2.0;
		if (comm->me == 0)
			printf("SHOCK: %g %g %g %g %g\n", shockfront_loc, shockfront_min, shockfront_max, domain->boxlo[2], domain->boxhi[2]);

		if (domain->boxhi[2] - shockfront_loc < spatlead)
			advance = 1;

		delete[] pos;
		delete[] val;
	}

	advance_sum = 0;
	MPI_Allreduce(&advance, &advance_sum, 1, MPI_INT, MPI_SUM, world);

	if (advance_sum > 0)
		return 1;
	else
		return 0;
}

/* ---------------------------------------------------------------------- */

void FixSmdInflow::post_force(int vflag) {

	return;

	double **f = atom->f;
	double **v = atom->v;
	double **x = atom->x;
	int *type = atom->type;
	int nlocal = atom->nlocal;

	double gamma1, gamma2;
	double tsqrt = sqrt(t_target);

	if (atom->mass) {
		if (tempflag) {
			for (int i = 1; i <= atom->ntypes; i++) {
				gfactor1[i] = -atom->mass[i] / t_period / force->ftm2v;
				gfactor2[i] = sqrt(atom->mass[i]) * sqrt(24.0 * force->boltz / t_period / update->dt / force->mvv2e) / force->ftm2v;
			}
		}
		for (int i = 0; i < nlocal; i++) {
			// SET TEMP AHEAD OF SHOCK
			if (tempflag && x[i][2] >= domain->boxhi[2] - t_extent) {
				gamma1 = gfactor1[type[i]];
				gamma2 = gfactor2[type[i]] * tsqrt;
				f[i][0] += gamma1 * v[i][0] + gamma2 * (randomt->uniform() - 0.5);
				f[i][1] += gamma1 * v[i][1] + gamma2 * (randomt->uniform() - 0.5);
				f[i][2] += gamma1 * v[i][2] + gamma2 * (randomt->uniform() - 0.5);
			}
			// FREEZE ATOMS AT BOUNDARY
			if (x[i][2] >= domain->boxhi[2] - size) {
				f[i][0] = 0.0;
				f[i][1] = 0.0;
				f[i][2] = 0.0;
				v[i][0] = 0.0;
				v[i][1] = 0.0;
				v[i][2] = 0.0;
			}
		}
	} else {
		double *rmass = atom->rmass;
		double boltz = force->boltz;
		double dt = update->dt;
		double mvv2e = force->mvv2e;
		double ftm2v = force->ftm2v;

		for (int i = 0; i < nlocal; i++) {

			// set temp ahead of shock

			if (tempflag && x[i][2] >= domain->boxhi[2] - t_extent) {
				gamma1 = -rmass[i] / t_period / ftm2v;
				gamma2 = sqrt(rmass[i]) * sqrt(24.0 * boltz / t_period / dt / mvv2e) / ftm2v;
				gamma2 *= tsqrt;
				f[i][0] += gamma1 * v[i][0] + gamma2 * (randomt->uniform() - 0.5);
				f[i][1] += gamma1 * v[i][1] + gamma2 * (randomt->uniform() - 0.5);
				f[i][2] += gamma1 * v[i][2] + gamma2 * (randomt->uniform() - 0.5);
			}

			// freeze atoms at boundary

			if (x[i][2] >= domain->boxhi[2] - size) {
				f[i][0] = 0.0;
				f[i][1] = 0.0;
				f[i][2] = 0.0;
				v[i][0] = 0.0;
				v[i][1] = 0.0;
				v[i][2] = 0.0;
			}
		}
	}
}

/* ---------------------------------------------------------------------- */

void FixSmdInflow::pre_exchange() {
	printf("proc %d enerts pre-exchange\n", comm->me);
	int ntimestep = update->ntimestep;
	int addnode = 0;

	if (ntimestep % freq == 0) {

		double current_time = update->atime;
		double time_difference = current_time - last_time;
		double travel_distance = last_insertion_height + time_difference * velocity - insertion_height;
		double displacement_excess;
		if (first) {
			displacement_excess = 0.0;
		} else {
			displacement_excess = fabs(travel_distance) - particle_spacing;
		}

		/*
		 * need to change code below so all procs enter into the barrier.
		 * cannot have simple per-proc return statements
		 */

		//printf("current time ")
		if (fabs(travel_distance) < particle_spacing) {
			// cannot insert now
			printf("cant insert at timestep %d: time difference since last insertion is %f, particle have travelled distance %f, last insertion height is %f\n",
					update->ntimestep, time_difference, travel_distance, last_insertion_height);
			//return;

		} else {
			printf("want to insert at timestep %d: time difference since last insertion is %f, particle have travelled distance %f, last insertion height is %f\n",
								update->ntimestep, time_difference, travel_distance, last_insertion_height);
			printf("velocity travelled distance is %f\n", time_difference * velocity);
			printf("displacement excess is %f\n", displacement_excess);
			last_time = current_time;

			if (spatflag == 1)
				if (get_spatial() == 0)
					return;

			int addflag = 0;
			if (comm->layout != LAYOUT_TILED) {
				if (comm->myloc[2] == comm->procgrid[2] - 1)
					addflag = 1;
			} else {
				if (comm->mysplit[2][1] == 1.0)
					addflag = 1;
			}

			if (addflag) {

				double bboxlo[3], bboxhi[3];

				bboxlo[0] = domain->sublo[0];
				bboxhi[0] = domain->subhi[0];
				bboxlo[1] = domain->sublo[1];
				bboxhi[1] = domain->subhi[1];
				bboxlo[2] = domain->subhi[2];
				bboxhi[2] = domain->subhi[2] + size;

				double xmin, ymin, zmin, xmax, ymax, zmax;
				xmin = ymin = zmin = BIG;
				xmax = ymax = zmax = -BIG;

				domain->lattice->bbox(1, bboxlo[0], bboxlo[1], bboxlo[2], xmin, ymin, zmin, xmax, ymax, zmax);
				domain->lattice->bbox(1, bboxhi[0], bboxlo[1], bboxlo[2], xmin, ymin, zmin, xmax, ymax, zmax);
				domain->lattice->bbox(1, bboxlo[0], bboxhi[1], bboxlo[2], xmin, ymin, zmin, xmax, ymax, zmax);
				domain->lattice->bbox(1, bboxhi[0], bboxhi[1], bboxlo[2], xmin, ymin, zmin, xmax, ymax, zmax);
				domain->lattice->bbox(1, bboxlo[0], bboxlo[1], bboxhi[2], xmin, ymin, zmin, xmax, ymax, zmax);
				domain->lattice->bbox(1, bboxhi[0], bboxlo[1], bboxhi[2], xmin, ymin, zmin, xmax, ymax, zmax);
				domain->lattice->bbox(1, bboxlo[0], bboxhi[1], bboxhi[2], xmin, ymin, zmin, xmax, ymax, zmax);
				domain->lattice->bbox(1, bboxhi[0], bboxhi[1], bboxhi[2], xmin, ymin, zmin, xmax, ymax, zmax);

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

				double **basis = domain->lattice->basis;
				double x[3];
				double *sublo = domain->sublo;
				double *subhi = domain->subhi;

				//printf("**** STILL WANT TO INSERT 0\n");

				int i, j, k, m;
				for (k = klo; k <= khi; k++) {
					for (j = jlo; j <= jhi; j++) {
						for (i = ilo; i <= ihi; i++) {
							for (m = 0; m < nbasis; m++) {
								x[0] = i + basis[m][0];
								x[1] = j + basis[m][1];
								x[2] = k + basis[m][2];

								if (!domain->regions[nregion]->match(x[0], x[1], x[2]))
									continue;

								int flag = 0;
								// convert from lattice coords to box coords
								domain->lattice->lattice2box(x[0], x[1], x[2]);

								// store insertion axis position
								if (fabs(x[2]) - insertion_height > 1.0e-16) {
									printf("insertion position should be at %g bust is at %g\n", insertion_height, x[2]);
									error->one(FLERR, "");
								}

								//printf("initial z = %f\n", x[2]);
								x[2] -= displacement_excess;
								last_insertion_height = x[2];
								//printf("final z = %f\n", x[2]);

//							if (x[0] >= sublo[0] && x[0] < subhi[0] && x[1] >= sublo[1] && x[1] < subhi[1] && x[2] >= subhi[2]
//									&& x[2] < subhi[2] + size)
//								flag = 1;
//							else if (domain->dimension == 2 && x[1] >= domain->boxhi[1] && comm->myloc[1] == comm->procgrid[1] - 1
//									&& x[0] >= sublo[0] && x[0] < subhi[0])
//								flag = 1;

								flag = 1;

								//printf("**** STILL WANT TO INSERT 1\n");

								if (flag) {
									first = false;

									//printf("**** STILL WANT TO INSERT 2\n");

									addnode++;
									atom->avec->create_atom(basistype[m], x);

									double volume_one = pow(domain->lattice->xlattice, 3); // particle volume
									double mass_one = rho * volume_one;

									int nlocal = atom->nlocal;
									printf("nlocal = %d, vol=%f, mass=%f\n", nlocal, volume_one, mass_one);
									int idx = nlocal - 1;
									double *vfrac = atom->vfrac;
									double *rmass = atom->rmass;
									double *radius = atom->radius;
									double **v = atom->v;
									double **vest = atom->vest;
									vfrac[idx] = volume_one;
									rmass[idx] = mass_one;
									v[idx][0] = vest[idx][0] = 0.0;
									v[idx][1] = vest[idx][1] = 0.0;
									v[idx][2] = vest[idx][2] = velocity;
									//radius[idx] = radius_one;

								}
							}
						}
					}
				}
			}
			int addtotal = 0;
			printf("proc %d is before barrier\n", comm->me);
			MPI_Barrier(world);
			MPI_Allreduce(&addnode, &addtotal, 1, MPI_INT, MPI_SUM, world);
			printf("proc %d is after barrier\n", comm->me);

			if (addtotal) {
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
