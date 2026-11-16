#pragma once

// Check output chronology before resuming a simulation into an existing
// directory. Inputs are the chosen restart time and the named diagnostic
// histories. Existing CSV histories and radial profiles must contain only
// times at or before that restart, within the stated roundoff allowance.
// Checking before opening output streams preserves later saved branches.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ev {

inline void require_output_precedes_restart(
    const std::filesystem::path& directory, double restart_time,
    const std::vector<std::string>& histories) {
    if (!std::filesystem::exists(directory)) return;
    if (!std::isfinite(restart_time)) {
        throw std::runtime_error("Restart time must be finite before output validation.");
    }
    const auto check_csv = [&](const std::filesystem::path& path) {
        if (!std::filesystem::exists(path)) return;
        std::ifstream input(path);
        std::string line;
        if (!input || !std::getline(input, line)) {
            throw std::runtime_error("Could not read existing diagnostic: " + path.string());
        }
        std::stringstream header(line);
        std::string token;
        std::size_t time_column = 0;
        bool found_time = false;
        // Locate the time column by name rather than assuming a particular
        // diagnostic schema or column position.
        while (std::getline(header, token, ',')) {
            if (token == "time" || token == "time\r") {
                found_time = true;
                break;
            }
            ++time_column;
        }
        if (!found_time) {
            throw std::runtime_error("Existing diagnostic has no time column: " + path.string());
        }
        while (std::getline(input, line)) {
            if (line.empty()) continue;
            std::stringstream row(line);
            for (std::size_t column = 0; column <= time_column; ++column) {
                if (!std::getline(row, token, ',')) {
                    throw std::runtime_error("Missing diagnostic time in " + path.string());
                }
            }
            std::size_t consumed = 0;
            // Parse the whole numeric token and require a finite time. A malformed
            // row cannot silently pass the restart chronology check.
            const double saved_time = std::stod(token, &consumed);
            if (!std::isfinite(saved_time)
                || token.find_first_not_of(" \t\r", consumed) != std::string::npos) {
                throw std::runtime_error("Invalid diagnostic time in " + path.string());
            }
            // Allow only roundoff at the scale of the two recorded times. A
            // resolved later observation requires a separate output directory so
            // its physical branch is retained.
            const double tolerance = 128.0 * std::numeric_limits<double>::epsilon()
                * std::max({1.0, std::abs(restart_time), std::abs(saved_time)});
            if (saved_time > restart_time + tolerance) {
                std::ostringstream message;
                message << std::setprecision(17) << "Existing diagnostic "
                        << path.string() << " contains time " << saved_time
                        << " later than restart time " << restart_time
                        << ". Continue in a separate output directory.";
                throw std::runtime_error(message.str());
            }
        }
        if (!input.eof()) {
            throw std::runtime_error("Failed reading existing diagnostic: " + path.string());
        }
    };
    for (const auto& name : histories) check_csv(directory / name);
    // Profiles also contain physical epochs. Check them as well as the
    // explicit history list before any continuation overwrites old files.
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        const auto name = entry.path().filename().string();
        if (entry.is_regular_file() && name.compare(0, 8, "profile_") == 0
            && entry.path().extension() == ".csv") {
            check_csv(entry.path());
        }
    }
}

} // namespace ev
