#include "numi/cloth_gpu_dispatch_guards.h"
#include "numi/cloth_bag_gpu.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

template <typename Action>
void mustReject(Action action, const char* description) {
    try {
        action();
    } catch (const std::logic_error&) {
        return;
    }
    throw std::runtime_error(description);
}

int main() {
    using numi::cloth::requireFixedLocalContactCounts;
    using numi::cloth::requireSelfCellCapacity;
    requireSelfCellCapacity(0, 0);
    requireSelfCellCapacity(2904, 2904);
    requireSelfCellCapacity(4096, 4096);
    mustReject([] { requireSelfCellCapacity(4097, 4096); },
               "authored segment overflow accepted");
    mustReject([] { requireSelfCellCapacity(4096, 4097); },
               "declared segment overflow accepted");

    const mr_uint4 initial{5754, 12, 0, 0};
    requireFixedLocalContactCounts(initial, initial);
    const std::array<mr_uint4, 4> changedCounts{{
        {5755, 12, 0, 0}, {5754, 13, 0, 0},
        {5754, 12, 1, 0}, {5754, 12, 0, 1}
    }};
    for (const mr_uint4& changed : changedCounts) {
        mustReject([&] { requireFixedLocalContactCounts(changed, initial); },
                   "local-contact topology mutation accepted");
    }
    std::cout << "dispatch_guard_host_passed=1 accepted_capacity_boundary=4096"
              << " rejected_capacity_overflows=2 rejected_local_count_mutations=4\n";
}
