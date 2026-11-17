#!/usr/bin/env bash
# Use the compiler's default SDK when it links successfully. If installed
# command-line tools and SDK stubs differ, try the compatible macOS 15 SDK.
venus_select_macos_sdk() {
  [[ $(uname -s) == Darwin ]] || return 0
  [[ -z ${BASE_CXXFLAGS:-} && -z ${SDKROOT:-} ]] || return 0
  local probe compiler sdk chosen
  compiler=${CXX:-c++}
  probe=$(mktemp -d "${TMPDIR:-/tmp}/venus-compiler.XXXXXX")
  printf '#include <vector>\nint main() { std::vector<int> a; return a.size(); }\n' > "$probe/check.cpp"
  if "$compiler" -std=c++17 "$probe/check.cpp" -o "$probe/check" 2> "$probe/error"; then
    rm -rf "$probe"
    return 0
  fi
  chosen=''
  for sdk in /Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk /Library/Developer/CommandLineTools/SDKs/MacOSX15.sdk; do
    [[ -d "$sdk" ]] || continue
    if "$compiler" -std=c++17 -isysroot "$sdk" "$probe/check.cpp" -o "$probe/check" 2> "$probe/error"; then
      chosen=$sdk
      break
    fi
  done
  if [[ -z "$chosen" ]]; then
    cat "$probe/error" >&2
    rm -rf "$probe"
    echo 'Select a compatible C++ compiler and SDK through CXX and BASE_CXXFLAGS.' >&2
    return 1
  fi
  export SDKROOT="$chosen"
  export BASE_CXXFLAGS="-std=c++17 -O3 -Wall -Wextra -Wpedantic -ffp-contract=off -isysroot $chosen"
  rm -rf "$probe"
  printf 'Using compatible macOS SDK: %s\n' "$chosen"
}
