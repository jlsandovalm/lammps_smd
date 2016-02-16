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

#ifdef PAIR_CLASS

PairStyle(smd/tri_surface,PairTriSurf)

#else

#ifndef LMP_SMD_TRI_SURFACE_H
#define LMP_SMD_TRI_SURFACE_H

#include "pair.h"
#include <Eigen/Eigen>
using namespace Eigen;


namespace LAMMPS_NS {

class PairTriSurf : public Pair {
 public:
  PairTriSurf(class LAMMPS *);
  virtual ~PairTriSurf();
  virtual void compute(int, int);
  void getTriangleInfo(const double *x0, const double *smd_data_9,
		  	  	  	  	  Vector3d &x1, Vector3d &x2, Vector3d &x3, Vector3d &normal);
  void NormalForce(const double r_particle, const double r, const double bulk_M, const double mass,
		  	  	  	  double &normalForceNorm, double &evdwl, double &stable_dt);
  void keepTouchDistance(const Vector3d cp, const Vector3d x4cp, double *&v,
  										const double r, const double touch_distance, double *&x);
  void settings(int, char **);
  void coeff(int, char **);
  double init_one(int, int);
  void init_style();
  void init_list(int, class NeighList *);
  virtual double memory_usage();
  int PointTriangleDistance(const Vector3d sourcePosition, const Vector3d TRI0, const Vector3d TRI1, const Vector3d TRI2,
							Vector3d &normal, Vector3d &CP, Vector3d &x4cp, double &dist);
  double clamp(const double a, const double min, const double max);
  void *extract(const char *, int &);
  Vector3d TangentialVelocity(const int, const int, const Vector3d);

 protected:
  double **bulkmodulus;
  double **kn;
  double **wall_temperature;
  double **frictionCoefficient;
  double **eta;

  int nmax;
  int *ncontact; // number of contacting triangles per particle

  double *onerad_dynamic,*onerad_frozen;
  double *maxrad_dynamic,*maxrad_frozen;

  double scale;
  double stable_time_increment; // stable time step size

  void allocate();
};

}

#endif
#endif

