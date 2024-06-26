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

/* ----------------------------------------------------------------------
 Contributing author: Mike Parks (SNL)
 ------------------------------------------------------------------------- */

#include "math.h"
#include "float.h"
#include "stdlib.h"
#include "string.h"
#include "pair_smd_triangulated_surface.h"
#include "atom.h"
#include "domain.h"
#include "force.h"
#include "update.h"
#include "modify.h"
#include "fix.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"
#include <Eigen/Eigen>
#include <stdio.h>
#include <iostream>
#include "SmdMatDB.h"

using namespace std;
using namespace LAMMPS_NS;
using namespace Eigen;

#define SQRT2 1.414213562e0

/* ---------------------------------------------------------------------- */

PairTriSurf::PairTriSurf(LAMMPS *lmp) :
		Pair(lmp) {

	onerad_dynamic = onerad_frozen = maxrad_dynamic = maxrad_frozen = NULL;
	bulkmodulus = NULL;
	kn = NULL;
	frictionCoefficient = NULL;
	eta = NULL;
	scale = 1.0;
	nmax = 0; // make sure no atom on this proc such that initial memory allocation is correct
	ncontact = NULL;

	// this pair style needs to access the material constitutive law database:
	// heat capacity and stiffness are needed.
	int retcode = SmdMatDB::instance().ReadMaterials(atom->ntypes);
	if (retcode < 0) {
		error->one(FLERR, "failed to read material database");
	}
}

/* ---------------------------------------------------------------------- */

PairTriSurf::~PairTriSurf() {

	if (allocated) {
		memory->destroy(setflag);
		memory->destroy(cutsq);
		memory->destroy(bulkmodulus);
		memory->destroy(kn);
		memory->destroy(frictionCoefficient);
		memory->destroy(eta);

		delete[] onerad_dynamic;
		delete[] onerad_frozen;
		delete[] maxrad_dynamic;
		delete[] maxrad_frozen;
		delete[] ncontact;
	}
}

/* ---------------------------------------------------------------------- */

void PairTriSurf::compute(int eflag, int vflag) {
	int i, j, ii, jj, inum, jnum, itype, jtype, particle_type;
	double rsq, r, evdwl, normalForceMagnitude;
	int *ilist, *jlist, *numneigh, **firstneigh;
	double rcut, r_tri, r_particle, touch_distance;
	int tri, particle, region;
	Vector3d normal, x1, x2, x3, x4, x13, x23, x43, w, cp, x4cp, vnew, v_old;
	;
	Vector3d xi, x_center, dx;
	Matrix2d C;
	Vector2d w2d, rhs;

	evdwl = 0.0;
	if (eflag || vflag)
		ev_setup(eflag, vflag);
	else
		evflag = vflag_fdotr = 0;

	int *mol = atom->molecule;
	double **f = atom->f;
	double **smd_data_9 = atom->smd_data_9;
	double **x = atom->x;
	double **x0 = atom->x0;
	double **v = atom->v;
	double *rmass = atom->rmass;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	double *radius = atom->contact_radius;
	double *heat = atom->heat;
	double rcutSq;
	Vector3d offset;

	int newton_pair = force->newton_pair;
	int periodic = (domain->xperiodic || domain->yperiodic || domain->zperiodic);

	inum = list->inum;
	ilist = list->ilist;
	numneigh = list->numneigh;
	firstneigh = list->firstneigh;

	if (atom->nmax > nmax) {
		nmax = atom->nmax;
		delete[] ncontact;
		ncontact = new int[nmax];
	}

	for (i = 0; i < nlocal; i++) {
		ncontact[i] = 0;
	}

	int max_neighs = 0;
	stable_time_increment = 1.0e22;

	// loop over neighbors of my atoms using a half neighbor list
	for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];
		itype = type[i];
		jlist = firstneigh[i];
		jnum = numneigh[i];
		max_neighs = MAX(max_neighs, jnum);

		for (jj = 0; jj < jnum; jj++) {
			j = jlist[jj];

			j &= NEIGHMASK;

			jtype = type[j];

			/* decide which one of i, j is triangle and which is particle */
			if ((mol[i] < 65535) && (mol[j] >= 65535)) {
				particle = i;
				tri = j;
			} else if ((mol[j] < 65535) && (mol[i] >= 65535)) {
				particle = j;
				tri = i;
			} else {
				error->one(FLERR, "unknown case");
			}

			particle_type = type[particle];

			x_center << x[tri][0], x[tri][1], x[tri][2]; // center of triangle

			x4 << x[particle][0], x[particle][1], x[particle][2]; // particle coordinates.

			dx = x_center - x4; // distance from particle to the triangle's center
			if (periodic) {
				domain->minimum_image(dx(0), dx(1), dx(2));
			}
			rsq = dx.squaredNorm();

			r_tri = scale * radius[tri];
			r_particle = scale * radius[particle];
			rcut = r_tri + r_particle;
			rcutSq = rcut * rcut;

			/* 1. Neighborhood Check:  */
			if (rsq < rcutSq) {

				getTriangleInfo(x0[tri], smd_data_9[tri], x1, x2, x3, normal);

				/* projection of particle on triangle's plane */
				region = PointTriangleDistance(x4, x1, x2, x3, normal, cp, x4cp,
						r);

				/* 2. Region Check: is particle's projection inside the triangle's area? */
				if (region == 0) {

					Vector3d viscousFForce, coulumbFForce, totalForce;

					normalForceMagnitude = 0;
					viscousFForce.setZero();
					coulumbFForce.setZero();
					totalForce.setZero();

					/* Calculate Tangential Relative Velocity */
					Vector3d vtan = TangentialVelocity(particle, tri, normal);
					double vtan_norm = vtan.norm();

					/* V I S C O U S   F R I C T I O N   F O R C E */

					viscousFForce = -eta[itype][jtype] * vtan / (r * r);

					/* 3. Contact Check: is particle closer than its radius to the triangle plane? */
					if (fabs(dx.dot(normal)) < r_particle) {

						/* directly set particle temperature (specific thermal energy) to wall temperature */
						heat[particle] = wall_temperature[itype][jtype] * rmass[particle] * SmdMatDB::instance().gProps[particle_type].cp;;

						/* penalty force pushes particle away from triangle */
						if (r < r_particle) {

							/* guard against the case that r is very small */
							if (r < 1.0e-2 * r_particle) {
								continue;
							}
							ncontact[particle] += 1;

							NormalForce(r_particle, r,
									bulkmodulus[itype][jtype], rmass[particle],
									normalForceMagnitude, evdwl,
									stable_time_increment);

							/* C O U L U M ' S   F R I C T I O N   F O R C E */

							if (vtan_norm > 1.0e-16) {
								coulumbFForce = -normalForceMagnitude
										* frictionCoefficient[itype][jtype]
										* vtan / vtan_norm;
							} else {
								coulumbFForce.setZero();
							}

							if (evflag) {
								normalForceMagnitude /= r; // divide by r because ev_tally expects force * distance vector
								ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0,
										normalForceMagnitude, x4cp(0), x4cp(1),
										x4cp(2));
							}
						}

						/* if particle comes too close to triangle, reflect its velocity and */
						/* explicitly move it back to the predetermined "touch-distance"     */

						touch_distance = 0.4 * r_particle;
						if (r < touch_distance) {
							keepTouchDistance(cp, x4cp, v[particle], r,
									touch_distance, *&x[particle]);
						}
					}

					/* T O T A L   F O R C E */
					/* total force =   repulsive elastic force along normal */
					/*               + tangential frictional force          */

					totalForce = normalForceMagnitude * normal + coulumbFForce
							+ viscousFForce;

					if (particle < nlocal) {
						f[particle][0] += totalForce(0);
						f[particle][1] += totalForce(1);
						f[particle][2] += totalForce(2);
					}

					if (tri < nlocal) {
						f[tri][0] -= totalForce(0);
						f[tri][1] -= totalForce(1);
						f[tri][2] -= totalForce(2);
					}

				} else {
					continue; //skip contact because we only want contact with interior points
					// contact point is on edge
					error->warning(FLERR, "contact point is an edge");
				}
			}
		}
	}

//	int max_neighs_all = 0;
//	MPI_Allreduce(&max_neighs, &max_neighs_all, 1, MPI_INT, MPI_MAX, world);
//	if (comm->me == 0) {
//		printf("max. neighs in tri pair is %d\n", max_neighs_all);
//	}
//
//		double stable_time_increment_all = 0.0;
//		MPI_Allreduce(&stable_time_increment, &stable_time_increment_all, 1, MPI_DOUBLE, MPI_MIN, world);
//		if (comm->me == 0) {
//			printf("stable time step tri pair is %f\n", stable_time_increment_all);
//		}
}

/* ------------------------------------------------------------------------- */

Vector3d PairTriSurf::TangentialVelocity(const int particle, const int tri,
		const Vector3d normal) {
	double **v = atom->v;
	Map<const Vector3d> v_particle(v[particle]);
	Map<const Vector3d> v_tri(v[tri]);

	Vector3d vrel = v_particle - v_tri; // relative translational velocity, points from tri to node, like dx
	Vector3d vtan = vrel - normal * vrel.dot(normal); // tangential velocity component

	/*
	 * tangentail velocity and normal should be orthognal
	 */

	if (fabs(vtan.dot(normal)) > 1.0e-3) {
		cout << "vtan   is " << vtan.transpose() << endl;
		cout << "normal is " << normal.transpose() << endl;
		printf(
				"expected angle between tangential (projected in plane velocity) and triangle normal to be close to zero, but cosAngle is %f\n",
				vtan.dot(normal));
		error->one(FLERR, "");
	}

	return vtan;
}

/* ---------------------------------------------------------------------- */

void PairTriSurf::getTriangleInfo(const double *x0, const double *smd_data_9,
		Vector3d &x1, Vector3d &x2, Vector3d &x3, Vector3d &normal) {

	/*
	 * gather triangle information
	 */
	normal(0) = x0[0];
	normal(1) = x0[1];
	normal(2) = x0[2];

	/*
	 * get triangle vertices
	 */
	x1(0) = smd_data_9[0];
	x1(1) = smd_data_9[1];
	x1(2) = smd_data_9[2];
	x2(0) = smd_data_9[3];
	x2(1) = smd_data_9[4];
	x2(2) = smd_data_9[5];
	x3(0) = smd_data_9[6];
	x3(1) = smd_data_9[7];
	x3(2) = smd_data_9[8];

}

/* ---------------------------------------------------------------------- */

void PairTriSurf::NormalForce(const double r_particle, const double r,
		const double bulk_M, const double mass, double &normalForceNorm,
		double &evdwl, double &stable_dt) {
	double delta, r_geom, dt_crit;

	delta = r_particle - r; // overlap distance
	r_geom = r_particle;

	normalForceNorm = 1.066666667e0 * bulk_M * delta * sqrt(delta * r_geom);

	evdwl = r * normalForceNorm * 0.4e0 * delta; // GCG 25 April: this expression conserves total energy

	dt_crit = 3.14 * sqrt(mass / (normalForceNorm / delta));

	stable_dt = MIN(stable_dt, dt_crit);
}

/* ---------------------------------------------------------------------- */

void PairTriSurf::keepTouchDistance(const Vector3d cp, const Vector3d x4cp,
		double *&v, const double r, const double touch_distance, double *&x) {

	Vector3d normal, v_old, vnew;

	/*
	 * reflect velocity if it points toward triangle
	 */

	normal = x4cp / r;

	// printf("moving particle on top of triangle\n");
	x[0] = cp(0) + touch_distance * normal(0);
	x[1] = cp(1) + touch_distance * normal(1);
	x[2] = cp(2) + touch_distance * normal(2);

	//v_old << v[particle][0], v[particle][1], v[particle][2];
	v_old(0) = v[0];
	v_old(1) = v[1];
	v_old(2) = v[2];

	if (v_old.dot(normal) < 0.0) {
		//printf("flipping velocity\n");
		vnew = -1.0 * (-2.0 * v_old.dot(normal) * normal + v_old);
		v[0] = vnew(0);
		v[1] = vnew(1);
		v[2] = vnew(2);
	}

}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairTriSurf::allocate() {
	allocated = 1;
	int n = atom->ntypes;

	memory->create(setflag, n + 1, n + 1, "pair:setflag");
	for (int i = 1; i <= n; i++)
		for (int j = i; j <= n; j++)
			setflag[i][j] = 0;

	memory->create(bulkmodulus, n + 1, n + 1, "pair:kspring");
	memory->create(kn, n + 1, n + 1, "pair:kn");
	memory->create(wall_temperature, n + 1, n + 1, "pair:wall_temperature");
	memory->create(frictionCoefficient, n + 1, n + 1,
			"pair:frictionCoefficient");
	memory->create(eta, n + 1, n + 1, "pair:eta");

	memory->create(cutsq, n + 1, n + 1, "pair:cutsq"); // always needs to be allocated, even with granular neighborlist

	onerad_dynamic = new double[n + 1];
	onerad_frozen = new double[n + 1];
	maxrad_dynamic = new double[n + 1];
	maxrad_frozen = new double[n + 1];
}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairTriSurf::settings(int narg, char **arg) {
	if (narg != 1)
		error->all(FLERR,
				"Illegal number of args for pair_style smd/tri_surface");

	scale = force->numeric(FLERR, arg[0]);
	if (comm->me == 0) {
		printf(
				"\n>>========>>========>>========>>========>>========>>========>>========>>========\n");
		printf("SMD/TRI_SURFACE CONTACT SETTINGS:\n");
		printf("... effective contact radius is scaled by %f\n", scale);
		printf(
				">>========>>========>>========>>========>>========>>========>>========>>========\n");
	}

}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairTriSurf::coeff(int narg, char **arg) {
	if (narg != 6)
		error->all(FLERR,
				"Incorrect args for pair coefficients. Expected: i, j, contact_stiffness, wall_temperature, Coulomb friction_coefficient, viscous friction coeff");
	if (!allocated)
		allocate();

	int ilo, ihi, jlo, jhi;
	force->bounds(arg[0], atom->ntypes, ilo, ihi);
	force->bounds(arg[1], atom->ntypes, jlo, jhi);

	double bulkmodulus_one = atof(arg[2]);
	double wall_temperature_one = atof(arg[3]);
	double frictionCoefficient_one = atof(arg[4]);
	double eta_one = atof(arg[5]);

	// set short-range force constant
	double kn_one = 0.0;
	if (domain->dimension == 3) {
		kn_one = (16. / 15.) * bulkmodulus_one; //assuming poisson ratio = 1/4 for 3d
	} else {
		kn_one = 0.251856195 * (2. / 3.) * bulkmodulus_one; //assuming poisson ratio = 1/3 for 2d
	}

	int count = 0;
	for (int i = ilo; i <= ihi; i++) {
		for (int j = MAX(jlo, i); j <= jhi; j++) {
			bulkmodulus[i][j] = bulkmodulus_one;
			kn[i][j] = kn_one;
			wall_temperature[i][j] = wall_temperature_one;
			frictionCoefficient[i][j] = frictionCoefficient_one;
			eta[i][j] = eta_one;
			setflag[i][j] = 1;
			count++;
		}
	}

	if (count == 0)
		error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairTriSurf::init_one(int i, int j) {

	if (!allocated)
		allocate();

	if (setflag[i][j] == 0)
		error->all(FLERR, "All pair coeffs are not set");

	bulkmodulus[j][i] = bulkmodulus[i][j];
	kn[j][i] = kn[i][j];
	wall_temperature[j][i] = wall_temperature[i][j];
	frictionCoefficient[j][i] = frictionCoefficient[i][j];
	eta[j][i] = eta[i][j];

	// cutoff = sum of max I,J radii for
	// dynamic/dynamic & dynamic/frozen interactions, but not frozen/frozen

	double cutoff = maxrad_dynamic[i] + maxrad_dynamic[j];
	cutoff = MAX(cutoff, maxrad_frozen[i] + maxrad_dynamic[j]);
	cutoff = MAX(cutoff, maxrad_dynamic[i] + maxrad_frozen[j]);

	if (comm->me == 0) {
		printf("cutoff for pair smd/smd/tri_surface = %f\n", cutoff);
	}
	return cutoff;
}

/* ----------------------------------------------------------------------
 init specific to this pair style
 ------------------------------------------------------------------------- */

void PairTriSurf::init_style() {
	int i;

	// error checks

	if (!atom->contact_radius_flag)
		error->all(FLERR,
				"Pair style smd/smd/tri_surface requires atom style with contact_radius");

	// old: half list
	int irequest = neighbor->request(this);
	neighbor->requests[irequest]->half = 0;
	neighbor->requests[irequest]->gran = 1;

	// need a full neighbor list
//	int irequest = neighbor->request(this);
//	neighbor->requests[irequest]->half = 0;
//	neighbor->requests[irequest]->full = 1;

	// set maxrad_dynamic and maxrad_frozen for each type
	// include future Fix pour particles as dynamic

	for (i = 1; i <= atom->ntypes; i++)
		onerad_dynamic[i] = onerad_frozen[i] = 0.0;

	double *radius = atom->radius;
	int *type = atom->type;
	int nlocal = atom->nlocal;

	for (i = 0; i < nlocal; i++) {
		onerad_dynamic[type[i]] = MAX(onerad_dynamic[type[i]], radius[i]);
	}

	MPI_Allreduce(&onerad_dynamic[1], &maxrad_dynamic[1], atom->ntypes,
			MPI_DOUBLE, MPI_MAX, world);
	MPI_Allreduce(&onerad_frozen[1], &maxrad_frozen[1], atom->ntypes,
			MPI_DOUBLE, MPI_MAX, world);
}

/* ----------------------------------------------------------------------
 neighbor callback to inform pair style of neighbor list to use
 optional granular history list
 ------------------------------------------------------------------------- */

void PairTriSurf::init_list(int id, NeighList *ptr) {
	if (id == 0)
		list = ptr;
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based arrays
 ------------------------------------------------------------------------- */

double PairTriSurf::memory_usage() {

	return 0.0;
}

/*
 * distance between triangle and point
 */
/*
 function [dist,PP0] = pointTriangleDistance(TRI,P)
 % calculate distance between a point and a triangle in 3D
 % SYNTAX
 %   dist = pointTriangleDistance(TRI,P)
 %   [dist,PP0] = pointTriangleDistance(TRI,P)
 %
 % DESCRIPTION
 %   Calculate the distance of a given point P from a triangle TRI.
 %   Point P is a row vector of the form 1x3. The triangle is a matrix
 %   formed by three rows of points TRI = [P1;P2;P3] each of size 1x3.
 %   dist = pointTriangleDistance(TRI,P) returns the distance of the point P
 %   to the triangle TRI.
 %   [dist,PP0] = pointTriangleDistance(TRI,P) additionally returns the
 %   closest point PP0 to P on the triangle TRI.
 %
 % Author: Gwendolyn Fischer
 % Release: 1.0
 % Release date: 09/02/02
 % Release: 1.1 Fixed Bug because of normalization
 % Release: 1.2 Fixed Bug because of typo in region 5 20101013
 % Release: 1.3 Fixed Bug because of typo in region 2 20101014

 % Possible extention could be a version tailored not to return the distance
 % and additionally the closest point, but instead return only the closest
 % point. Could lead to a small speed gain.

 % Example:
 % %% The Problem
 % P0 = [0.5 -0.3 0.5];
 %
 % P1 = [0 -1 0];
 % P2 = [1  0 0];
 % P3 = [0  0 0];
 %
 % vertices = [P1; P2; P3];
 % faces = [1 2 3];
 %
 % %% The Engine
 % [dist,PP0] = pointTriangleDistance([P1;P2;P3],P0);
 %
 % %% Visualization
 % [x,y,z] = sphere(20);
 % x = dist*x+P0(1);
 % y = dist*y+P0(2);
 % z = dist*z+P0(3);
 %
 % figure
 % hold all
 % patch('Vertices',vertices,'Faces',faces,'FaceColor','r','FaceAlpha',0.8);
 % plot3(P0(1),P0(2),P0(3),'b*');
 % plot3(PP0(1),PP0(2),PP0(3),'*g')
 % surf(x,y,z,'FaceColor','b','FaceAlpha',0.3)
 % view(3)

 % The algorithm is based on
 % "David Eberly, 'Distance Between Point and Triangle in 3D',
 % Geometric Tools, LLC, (1999)"
 % http:\\www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
 %
 %        ^t
 %  \     |
 %   \reg2|
 %    \   |
 %     \  |
 %      \ |
 %       \|
 %        *P2
 %        |\
%        | \
%  reg3  |  \ reg1
 %        |   \
%        |reg0\
%        |     \
%        |      \ P1
 % -------*-------*------->s
 %        |P0      \
%  reg4  | reg5    \ reg6
 */

//void PairTriSurf::PointTriangleDistance(const Vector3d P, const Vector3d TRI1, const Vector3d TRI2, const Vector3d TRI3,
//		Vector3d &CP, double &dist) {
//
//	Vector3d B, E0, E1, D;
//	double a, b, c, d, e, f;
//	double det, s, t, sqrDistance, tmp0, tmp1, numer, denom, invDet;
//
//	// rewrite triangle in normal form
//	B = TRI1;
//	E0 = TRI2 - B;
//	E1 = TRI3 - B;
//
//	D = B - P;
//	a = E0.dot(E0);
//	b = E0.dot(E1);
//	c = E1.dot(E1);
//	d = E0.dot(D);
//	e = E1.dot(D);
//	f = D.dot(D);
//
//	det = a * c - b * b;
//	//% do we have to use abs here?
//	s = b * e - c * d;
//	t = b * d - a * e;
//
//	//% Terible tree of conditionals to determine in which region of the diagram
//	//% shown above the projection of the point into the triangle-plane lies.
//	if ((s + t) <= det) {
//		if (s < 0) {
//			if (t < 0) {
//				// %region4
//				if (d < 0) {
//					t = 0;
//					if (-d >= a) {
//						s = 1;
//						sqrDistance = a + 2 * d + f;
//					} else {
//						s = -d / a;
//						sqrDistance = d * s + f;
//					}
//				} else {
//					s = 0;
//					if (e >= 0) {
//						t = 0;
//						sqrDistance = f;
//					} else {
//						if (-e >= c) {
//							t = 1;
//							sqrDistance = c + 2 * e + f;
//						} else {
//							t = -e / c;
//							sqrDistance = e * t + f;
//						}
//					}
//				}
//				// end % of region 4
//			} else {
//				// % region 3
//				s = 0;
//				if (e >= 0) {
//					t = 0;
//					sqrDistance = f;
//				} else {
//					if (-e >= c) {
//						t = 1;
//						sqrDistance = c + 2 * e + f;
//					} else {
//						t = -e / c;
//						sqrDistance = e * t + f;
//					}
//				}
//			}
//			// end of region 3
//		} else {
//			if (t < 0) {
//				//% region 5
//				t = 0;
//				if (d >= 0) {
//					s = 0;
//					sqrDistance = f;
//				} else {
//					if (-d >= a) {
//						s = 1;
//						sqrDistance = a + 2 * d + f;
//					} else {
//						s = -d / a;
//						sqrDistance = d * s + f;
//					}
//				}
//			} else {
//				// region 0
//				invDet = 1 / det;
//				s = s * invDet;
//				t = t * invDet;
//				sqrDistance = s * (a * s + b * t + 2 * d) + t * (b * s + c * t + 2 * e) + f;
//			}
//		}
//	} else {
//		if (s < 0) {
//			// % region 2
//			tmp0 = b + d;
//			tmp1 = c + e;
//			if (tmp1 > tmp0) { //% minimum on edge s+t=1
//				numer = tmp1 - tmp0;
//				denom = a - 2 * b + c;
//				if (numer >= denom) {
//					s = 1;
//					t = 0;
//					sqrDistance = a + 2 * d + f;
//				} else {
//					s = numer / denom;
//					t = 1 - s;
//					sqrDistance = s * (a * s + b * t + 2 * d) + t * (b * s + c * t + 2 * e) + f;
//				}
//			} else
//				// % minimum on edge s=0
//				s = 0;
//			if (tmp1 <= 0) {
//				t = 1;
//				sqrDistance = c + 2 * e + f;
//			} else {
//				if (e >= 0) {
//					t = 0;
//					sqrDistance = f;
//				} else {
//					t = -e / c;
//					sqrDistance = e * t + f;
//				}
//			}
//		} //end % of region	2
//		else {
//			if (t < 0) {
//				// %region6
//				tmp0 = b + e;
//				tmp1 = a + d;
//				if (tmp1 > tmp0) {
//					numer = tmp1 - tmp0;
//					denom = a - 2 * b + c;
//					if (numer >= denom) {
//						t = 1;
//						s = 0;
//						sqrDistance = c + 2 * e + f;
//					} else {
//						t = numer / denom;
//						s = 1 - t;
//						sqrDistance = s * (a * s + b * t + 2 * d) + t * (b * s + c * t + 2 * e) + f;
//					}
//				} else {
//					t = 0;
//					if (tmp1 <= 0) {
//						s = 1;
//						sqrDistance = a + 2 * d + f;
//					} else {
//						if (d >= 0) {
//							s = 0;
//							sqrDistance = f;
//						} else {
//							s = -d / a;
//							sqrDistance = d * s + f;
//						}
//					}
//				} // % end region 6
//			} else {
//				//% region 1
//				numer = c + e - b - d;
//				if (numer <= 0) {
//					s = 0;
//					t = 1;
//					sqrDistance = c + 2 * e + f;
//				} else {
//					denom = a - 2 * b + c;
//					if (numer >= denom) {
//						s = 1;
//						t = 0;
//						sqrDistance = a + 2 * d + f;
//					} else {
//						s = numer / denom;
//						t = 1 - s;
//						sqrDistance = s * (a * s + b * t + 2 * d) + t * (b * s + c * t + 2 * e) + f;
//					}
//				} //% end of region 1
//			}
//		}
//	}
//
//	// % account for numerical round-off error
//	if (sqrDistance < 0) {
//		sqrDistance = 0;
//	}
//
//	dist = sqrt(sqrDistance);
//
//	// closest point
//	CP = B + s * E0 + t * E1;
//
//}
/*
 * % The algorithm is based on
 % "David Eberly, 'Distance Between Point and Triangle in 3D',
 % Geometric Tools, LLC, (1999)"
 % http:\\www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
 */

int PairTriSurf::PointTriangleDistance(const Vector3d sourcePosition,
		const Vector3d TRI0, const Vector3d TRI1, const Vector3d TRI2,
		Vector3d &normal, Vector3d &CP, Vector3d &x4cp, double &dist) {

	Vector3d edge0 = TRI1 - TRI0;
	Vector3d edge1 = TRI2 - TRI0;
	Vector3d v0 = TRI0 - sourcePosition;

	double a = edge0.dot(edge0);
	double b = edge0.dot(edge1);
	double c = edge1.dot(edge1);
	double d = edge0.dot(v0);
	double e = edge1.dot(v0);

	double det = a * c - b * b;
	double s = b * e - c * d;
	double t = b * d - a * e;
	double cosAngle = 0;

	int region = 1; // region is 0 if closest point is within triangle, 1 if on triangle edges

	if (s + t < det) {
		if (s < 0.f) {
			if (t < 0.f) {
				if (d < 0.f) {
					s = clamp(-d / a, 0.f, 1.f);
					t = 0.f;
				} else {
					s = 0.f;
					t = clamp(-e / c, 0.f, 1.f);
				}
			} else {
				s = 0.f;
				t = clamp(-e / c, 0.f, 1.f);
			}
		} else if (t < 0.f) {
			s = clamp(-d / a, 0.f, 1.f);
			t = 0.f;
		} else {
			float invDet = 1.f / det;
			s *= invDet;
			t *= invDet;
			// this should be region 0, i.e., contact point is within triangle
			region = 0;
		}
	} else {
		if (s < 0.f) {
			float tmp0 = b + d;
			float tmp1 = c + e;
			if (tmp1 > tmp0) {
				float numer = tmp1 - tmp0;
				float denom = a - 2 * b + c;
				s = clamp(numer / denom, 0.f, 1.f);
				t = 1 - s;
			} else {
				t = clamp(-e / c, 0.f, 1.f);
				s = 0.f;
			}
		} else if (t < 0.f) {
			if (a + d > b + e) {
				float numer = c + e - b - d;
				float denom = a - 2 * b + c;
				s = clamp(numer / denom, 0.f, 1.f);
				t = 1 - s;
			} else {
				s = clamp(-e / c, 0.f, 1.f);
				t = 0.f;
			}
		} else {
			float numer = c + e - b - d;
			float denom = a - 2 * b + c;
			s = clamp(numer / denom, 0.f, 1.f);
			t = 1.f - s;
		}
	}

	CP = TRI0 + s * edge0 + t * edge1;

	/* distance vector to closest point on triangle */
	x4cp = sourcePosition - CP;
	dist = x4cp.norm();

	/* flip normal to point in direction of x4cp */
	if (x4cp.dot(normal) < 0.0) {
		normal *= -1.0;
	}

	if (region == 0) {
		/* check validity of result: x4cp should be parallel to triangle normal */
		cosAngle = x4cp.dot(normal) / dist;
		if (fabs(fabs(cosAngle) - 1.0) > 1.0e-3) {
			cout << "x4cp   is " << x4cp.transpose() << endl;
			cout << "normal is " << normal.transpose() << endl;
			printf("region is %d, expected |cosAngle = 1| but cosAngle is %f\n",
					region, cosAngle);
			error->one(FLERR, "");
		}
	}

	return region;

}

double PairTriSurf::clamp(const double a, const double min, const double max) {
	if (a < min) {
		return min;
	} else if (a > max) {
		return max;
	} else {
		return a;
	}
}

void *PairTriSurf::extract(const char *str, int &i) {
	//printf("in PairTriSurf::extract\n");
	if (strcmp(str, "smd/tri_surface/stable_time_increment_ptr") == 0) {
		return (void *) &stable_time_increment;
	} else if (strcmp(str, "smd/tri_surface/ncontact_ptr") == 0) {
		return (void *) ncontact;
	}

	return NULL;

}
