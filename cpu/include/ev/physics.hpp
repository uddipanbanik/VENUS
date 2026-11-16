#pragma once

// Collect the CPU moment, metric, timestep, Vlasov--Poisson and
// Boltzmann--Poisson modules for callers that use the complete field solver.
// The individual headers define their physical variables and numerical steps.
// A caller needing only one operation can include its narrower module directly;
// this convenience header itself introduces no state or additional evolution.

#include "ev/moments.hpp"
#include "ev/metric.hpp"
#include "ev/timestep.hpp"
#include "ev/vlasov_poisson.hpp"
#include "ev/boltzmann_poisson.hpp"
