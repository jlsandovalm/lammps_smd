/* -*- c++ -*- ----------------------------------------------------------
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

FixStyle(smd/inflow,FixSmdInflow)

#else

#ifndef FIX_SMD_INFLOW_H
#define FIX_SMD_INFLOW_H

#include "fix.h"

namespace LAMMPS_NS {

class FixSmdInflow : public Fix {
 public:
  FixSmdInflow(class LAMMPS *, int, char **);
  ~FixSmdInflow();
  int setmask();
  void setup(int);
  void pre_exchange();
  void initial_integrate(int);
  void post_force(int);

 private:
  int xflag, yflag, zflag, freq; // velocity direction of atoms
  int nbasis;
  int *basistype;
  int nregion;

  bool first;

  int type_one;
  int insertion_dimension;
  int rho_flag, velocity_flag, region_flag, freq_flag, type_flag;
  double rho, radius_one, contact_radius_one, heat_one;
  double volume_one, mass_one;
  double last_time;
  double particle_spacing;
  double velocity;
  double insertion_height;
  double last_insertion_height;
  double extent_xlo, extent_xhi, extent_ylo, extent_yhi, extent_zlo, extent_zhi;

  char *idregion;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Fix append/atoms requires a lattice be defined

Use the lattice command for this purpose.

E: Only zhi currently implemented for fix append/atoms

Self-explanatory.

E: Append boundary must be shrink/minimum

The boundary style of the face where atoms are added
must be of type m (shrink/minimum).

E: Bad fix ID in fix append/atoms command

The value of the fix_id for keyword spatial must start with the suffix
f_.

E: Invalid basis setting in fix append/atoms command

The basis index must be between 1 to N where N is the number of basis
atoms in the lattice.  The type index must be between 1 to N where N
is the number of atom types.

E: Cannot use append/atoms in periodic dimension

The boundary style of the face where atoms are added can not be of
type p (periodic).

E: Cannot append atoms to a triclinic box

The simulation box must be defined with edges alligned with the
Cartesian axes.

E: Fix ID for fix ave/spatial does not exist

Self-explanatory.

E: Too many total atoms

See the setting for bigint in the src/lmptype.h file.

*/
