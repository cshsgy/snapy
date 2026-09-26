#pragma once

// base
#include <configure.h>

namespace snap {

#if NMASS > 0  // use legacy Athena++ indexing scheme

//! \brief Indexing scheme for hydro and reconstruction variables
//!
//! These enumerations define the array indices for accessing
//! hydrodynamic and reconstruction variables in legacy Athena++ format.
enum {
  // hydro variables
  IDN = 0,          //!< Density index
  ICY = 1,          //!< Composition/species index (first mass fraction)
  IVX = 1 + NMASS,  //!< X-velocity index
  IVY = 2 + NMASS,  //!< Y-velocity index
  IVZ = 3 + NMASS,  //!< Z-velocity index
  IPR = 4 + NMASS,  //!< Pressure index

  // reconstruction variables
  ILT = 0,  //!< Left interface
  IRT = 1,  //!< Right interface
};

#else  // use new indexing scheme

//! \brief Indexing scheme for hydro and reconstruction variables
//!
//! These enumerations define the array indices for accessing
//! hydrodynamic and reconstruction variables in new format.
enum {
  // hydro variables
  IDN = 0,  //!< Density index
  IVX = 1,  //!< X-velocity index
  IVY = 2,  //!< Y-velocity index
  IVZ = 3,  //!< Z-velocity index
  IPR = 4,  //!< Pressure index
  ICY = 5,  //!< Composition/species index (first mass fraction)

  // reconstruction variables
  ILT = 0,  //!< Left interface
  IRT = 1,  //!< Right interface
};

#endif  // index scheme

// Species rows start on the row after pressure. NMASS>0 (ICY == 1) fails this.
static_assert(ICY == IPR + 1,
              "species rows must start on the row after pressure");

//! \brief Variable type enumeration
//!
//! Defines different types of variables used in the simulation.
enum {
  // variable type
  kPrimitive = 0,  //!< Primitive variables (rho, v, P)
  kConserved = 1,  //!< Conserved variables (rho, rho*v, E)
  kScalar = 2,     //!< Scalar variables

  // temperature, pressure, mass fraction with LR states
  kTPMassLR = 5,  //!< Temperature, pressure, mass with left-right states
  kDPMassLR = 6,  //!< Density, pressure, mass with left-right states
};

//! \brief Velocity component enumeration
//!
//! Indices for accessing velocity components.
enum {
  VEL1 = 0,  //!< First velocity component
  VEL2 = 1,  //!< Second velocity component
  VEL3 = 2   //!< Third velocity component
};

}  // namespace snap
