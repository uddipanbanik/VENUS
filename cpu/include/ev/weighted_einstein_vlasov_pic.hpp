#pragma once

// Expose the two weighted Einstein--Vlasov coordinate formulations.
// The Schwarzschild/polar-areal API is in namespace ev; the ingoing
// Eddington--Finkelstein API is in ev::eddington_finkelstein.
// Each included module defines its own momentum variable, metric constraints
// and characteristic equations. A caller selects a chart explicitly rather than
// identifying their distinct time slices or momentum coordinates.

#include "ev/weighted_einstein_vlasov_pic_schwarzschild.hpp"
#include "ev/weighted_einstein_vlasov_pic_eddington_finkelstein.hpp"
