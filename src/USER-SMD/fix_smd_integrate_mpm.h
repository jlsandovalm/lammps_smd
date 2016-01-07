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

#ifdef FIX_CLASS

FixStyle(smd/integrate_mpm,FixSMDIntegrateMpm)

#else

#ifndef LMP_FIX_SMD_INTEGRATE_MPM_H
#define LMP_FIX_SMD_INTEGRATE_MPM_H

#include "fix.h"

namespace LAMMPS_NS {

class FixSMDIntegrateMpm : public Fix {
 public:
  FixSMDIntegrateMpm(class LAMMPS *, int, char **);
  int setmask();
  virtual void init();
  virtual void final_integrate();
  void reset_dt();

 private:
  class NeighList *list;
  int nregion, region_flag;
  char *idregion;

 protected:
  enum {DEFAULT_INTEGRATION, CONSTANT_VELOCITY, PRESCRIBED_VELOCITY};
  double dtv, vlimit, vlimitsq;;
  int mass_require;
  double FLIP_contribution, PIC_contribution;
  double const_vx, const_vy, const_vz;
  bool flag3d; // integrate z degree of freedom?

  class Pair *pair;
};

}

#endif
#endif
