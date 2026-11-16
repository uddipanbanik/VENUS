#pragma once

// Compute SHA-256 digests for scientific files and restart/source inventories.
// Input bytes are processed in bounded blocks and returned as a hexadecimal
// 256-bit digest. Equality with a trusted expected digest establishes byte
// correspondence. The unkeyed digest itself supplies no claim about who wrote
// the file; source ownership and expected-record trust belong to the caller.

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ev::integrity {

class SHA256 {
public:
    SHA256() { reset(); }

    // Accumulate arbitrary input fragments into 64-byte compression blocks.
    // The total byte count tracks only original input, not final padding.
    void update(const void* source, std::size_t size) {
        const auto* data = static_cast<const std::uint8_t*>(source);
        total_bytes_ += size;
        while (size > 0) {
            const std::size_t take = std::min(size, block_.size() - used_);
            for (std::size_t i = 0; i < take; ++i) {
                block_[used_ + i] = data[i];
            }
            used_ += take;
            data += take;
            size -= take;
            if (used_ == block_.size()) {
                compress(block_.data());
                used_ = 0;
            }
        }
    }

    void update(const std::string& value) {
        update(value.data(), value.size());
    }

    std::string finish_hex() {
        // SHA-256 appends one set bit, enough zero bits to reach byte 56 in a
        // block, and the original length as a 64-bit big-endian bit count.
        const std::uint64_t bit_count = total_bytes_ * UINT64_C(8);
        const std::uint8_t one = 0x80;
        update_padding(&one, 1);
        const std::uint8_t zero = 0;
        while (used_ != 56) {
            update_padding(&zero, 1);
        }
        std::array<std::uint8_t, 8> length{};
        for (int i = 0; i < 8; ++i) {
            length[static_cast<std::size_t>(7 - i)] =
                static_cast<std::uint8_t>(bit_count >> (8 * i));
        }
        update_padding(length.data(), length.size());
        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (std::uint32_t value : state_) {
            output << std::setw(8) << value;
        }
        return output.str();
    }

private:
    std::array<std::uint32_t, 8> state_{};
    std::array<std::uint8_t, 64> block_{};
    std::size_t used_ = 0;
    std::uint64_t total_bytes_ = 0;

    static std::uint32_t rotate_right(std::uint32_t x, int shift) {
        return (x >> shift) | (x << (32 - shift));
    }

    void reset() {
        state_ = {
            UINT32_C(0x6a09e667), UINT32_C(0xbb67ae85),
            UINT32_C(0x3c6ef372), UINT32_C(0xa54ff53a),
            UINT32_C(0x510e527f), UINT32_C(0x9b05688c),
            UINT32_C(0x1f83d9ab), UINT32_C(0x5be0cd19),
        };
        block_.fill(0);
        used_ = 0;
        total_bytes_ = 0;
    }

    // Padding uses the same block compression but leaves total_bytes_
    // unchanged, preserving the length of the original scientific byte stream.
    void update_padding(const void* source, std::size_t size) {
        const auto* data = static_cast<const std::uint8_t*>(source);
        while (size > 0) {
            const std::size_t take = std::min(size, block_.size() - used_);
            for (std::size_t i = 0; i < take; ++i) {
                block_[used_ + i] = data[i];
            }
            used_ += take;
            data += take;
            size -= take;
            if (used_ == block_.size()) {
                compress(block_.data());
                used_ = 0;
            }
        }
    }

    void compress(const std::uint8_t* data) {
        static constexpr std::array<std::uint32_t, 64> k = {
            UINT32_C(0x428a2f98), UINT32_C(0x71374491),
            UINT32_C(0xb5c0fbcf), UINT32_C(0xe9b5dba5),
            UINT32_C(0x3956c25b), UINT32_C(0x59f111f1),
            UINT32_C(0x923f82a4), UINT32_C(0xab1c5ed5),
            UINT32_C(0xd807aa98), UINT32_C(0x12835b01),
            UINT32_C(0x243185be), UINT32_C(0x550c7dc3),
            UINT32_C(0x72be5d74), UINT32_C(0x80deb1fe),
            UINT32_C(0x9bdc06a7), UINT32_C(0xc19bf174),
            UINT32_C(0xe49b69c1), UINT32_C(0xefbe4786),
            UINT32_C(0x0fc19dc6), UINT32_C(0x240ca1cc),
            UINT32_C(0x2de92c6f), UINT32_C(0x4a7484aa),
            UINT32_C(0x5cb0a9dc), UINT32_C(0x76f988da),
            UINT32_C(0x983e5152), UINT32_C(0xa831c66d),
            UINT32_C(0xb00327c8), UINT32_C(0xbf597fc7),
            UINT32_C(0xc6e00bf3), UINT32_C(0xd5a79147),
            UINT32_C(0x06ca6351), UINT32_C(0x14292967),
            UINT32_C(0x27b70a85), UINT32_C(0x2e1b2138),
            UINT32_C(0x4d2c6dfc), UINT32_C(0x53380d13),
            UINT32_C(0x650a7354), UINT32_C(0x766a0abb),
            UINT32_C(0x81c2c92e), UINT32_C(0x92722c85),
            UINT32_C(0xa2bfe8a1), UINT32_C(0xa81a664b),
            UINT32_C(0xc24b8b70), UINT32_C(0xc76c51a3),
            UINT32_C(0xd192e819), UINT32_C(0xd6990624),
            UINT32_C(0xf40e3585), UINT32_C(0x106aa070),
            UINT32_C(0x19a4c116), UINT32_C(0x1e376c08),
            UINT32_C(0x2748774c), UINT32_C(0x34b0bcb5),
            UINT32_C(0x391c0cb3), UINT32_C(0x4ed8aa4a),
            UINT32_C(0x5b9cca4f), UINT32_C(0x682e6ff3),
            UINT32_C(0x748f82ee), UINT32_C(0x78a5636f),
            UINT32_C(0x84c87814), UINT32_C(0x8cc70208),
            UINT32_C(0x90befffa), UINT32_C(0xa4506ceb),
            UINT32_C(0xbef9a3f7), UINT32_C(0xc67178f2),
        };
        // Read the first 16 big-endian words, then extend them to the 64-word
        // message schedule using the standard rotated/shifted SHA-256 functions.
        std::array<std::uint32_t, 64> w{};
        for (int i = 0; i < 16; ++i) {
            const std::size_t j = static_cast<std::size_t>(4 * i);
            w[static_cast<std::size_t>(i)] =
                (static_cast<std::uint32_t>(data[j]) << 24)
                | (static_cast<std::uint32_t>(data[j + 1]) << 16)
                | (static_cast<std::uint32_t>(data[j + 2]) << 8)
                | static_cast<std::uint32_t>(data[j + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotate_right(w[i - 15], 7)
                ^ rotate_right(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotate_right(w[i - 2], 17)
                ^ rotate_right(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];
        // Each standard compression round combines the schedule word, its
        // fixed constant, the choice/majority functions and rotated state words.
        // Unsigned 32-bit arithmetic provides the specified modular sums.
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t sum1 = rotate_right(e, 6)
                ^ rotate_right(e, 11) ^ rotate_right(e, 25);
            const std::uint32_t choice = (e & f) ^ ((~e) & g);
            const std::uint32_t t1 = h + sum1 + choice + k[i] + w[i];
            const std::uint32_t sum0 = rotate_right(a, 2)
                ^ rotate_right(a, 13) ^ rotate_right(a, 22);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }
};

inline std::string sha256_string(const std::string& value) {
    SHA256 digest;
    digest.update(value);
    return digest.finish_hex();
}

// Read the complete file in fixed-size binary chunks. Distinguish a
// normal EOF from an input error before returning a digest, so a failed
// read cannot masquerade as a complete-file identity.
inline std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "Could not open artifact for SHA-256: " + path.string());
    }
    SHA256 digest;
    std::array<char, 65536> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) {
            digest.update(buffer.data(), static_cast<std::size_t>(count));
        }
    }
    if (!input.eof()) {
        throw std::runtime_error(
            "Could not read artifact for SHA-256: " + path.string());
    }
    return digest.finish_hex();
}

}  // namespace ev::integrity
