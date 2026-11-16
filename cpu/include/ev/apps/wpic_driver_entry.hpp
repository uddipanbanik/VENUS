#pragma once

// Declare the shared weighted-marker application entry points.
// Each function receives the executable arguments and returns its process status.
// The selected driver owns physical parameter parsing, initial or restart state,
// field construction, time evolution and output. These declarations let the
// small task executables use that same implementation without sharing mutable
// simulation state between separate processes.

namespace ev::apps {

int run_weighted_einstein_vlasov_pic(int argc, char** argv);
int run_expanding_weighted_vlasov_pic(int argc, char** argv);
int run_static_sidm_weighted_vlasov_pic(int argc, char** argv);

}  // namespace ev::apps
