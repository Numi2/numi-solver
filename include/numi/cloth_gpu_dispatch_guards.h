#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace numi::cloth {

// ClothBag.metal's self-cell buffer and bitonic sort each have 4096 entries.
inline constexpr std::size_t kSelfCellEntryCapacity = 4096u;

inline void requireSelfCellCapacity(std::size_t authoredSegments,
                                    std::uint32_t declaredSegments) {
    if (authoredSegments > kSelfCellEntryCapacity ||
        declaredSegments > kSelfCellEntryCapacity) {
        throw std::logic_error("Metal cloth self-cell segment capacity exceeded");
    }
}

template <typename Counts>
inline void requireFixedLocalContactCounts(const Counts& requested,
                                           const Counts& initial) {
    static_assert(std::is_trivially_copyable_v<Counts>);
    static_assert(sizeof(Counts) == 4u * sizeof(std::uint32_t));
    if (std::memcmp(&requested, &initial, sizeof(Counts)) != 0) {
        throw std::logic_error("Metal cloth trajectory changes fixed local-contact topology");
    }
}

} // namespace numi::cloth
