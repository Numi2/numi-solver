#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/cloth_bag_gpu.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef NUMI_TEMPORAL_CONE_METALLIB
#error "NUMI_TEMPORAL_CONE_METALLIB must name the built solver metallib"
#endif

namespace {

constexpr std::uint32_t kAround = 48u;
constexpr std::uint32_t kLevels = 28u;
constexpr std::uint32_t kBottomGrid = 13u;
constexpr std::uint32_t kBottomInterior = kBottomGrid - 2u;
constexpr std::uint32_t kParticleCount =
    kAround * kLevels + kBottomInterior * kBottomInterior;
constexpr std::uint32_t kDistanceCount = 2904u;
constexpr std::uint32_t kGripCount = 10u;
constexpr std::uint32_t kKnotCount = 1369u;
constexpr std::uint32_t kBendCount = 2834u;
constexpr float kOrdinaryMass = 0.000050f;
constexpr float kHemMass = 0.000100f;
constexpr float kGripCompliance = 2.0e-4f;
constexpr float kStrainLimit = 0.285f;
constexpr float kTimestep = 1.0f / 5760.0f;

struct DVec3 {
    double x{};
    double y{};
    double z{};
};

DVec3 operator+(const DVec3 a, const DVec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

DVec3 operator-(const DVec3 a, const DVec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

DVec3 operator*(const DVec3 a, const double scale) {
    return {a.x * scale, a.y * scale, a.z * scale};
}

DVec3 operator/(const DVec3 a, const double scale) {
    return a * (1.0 / scale);
}

DVec3& operator+=(DVec3& a, const DVec3 b) {
    a = a + b;
    return a;
}

DVec3& operator-=(DVec3& a, const DVec3 b) {
    a = a - b;
    return a;
}

double dot(const DVec3 a, const DVec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

double length(const DVec3 value) {
    return std::sqrt(dot(value, value));
}

mr_float4 f4(
    const float x,
    const float y,
    const float z,
    const float w = 0.0f
) {
    return {x, y, z, w};
}

mr_uint4 u4(
    const std::uint32_t x,
    const std::uint32_t y,
    const std::uint32_t z = 0u,
    const std::uint32_t w = 0u
) {
    return {x, y, z, w};
}

DVec3 d3(const mr_float4 value) {
    return {
        static_cast<double>(value.x),
        static_cast<double>(value.y),
        static_cast<double>(value.z),
    };
}

double smoothstep(const double value) {
    const double x = std::clamp(value, 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
}

std::uint32_t nodeIndex(
    const std::uint32_t level,
    const std::uint32_t ring
) {
    return level * kAround + ring % kAround;
}

DVec3 authoredPosition(
    const std::uint32_t level,
    const std::uint32_t ring
) {
    const double vertical = static_cast<double>(level) /
        static_cast<double>(kLevels - 1u);
    const double body = smoothstep(vertical / 0.55);
    const double fold = std::clamp((vertical - 0.72) / 0.28, 0.0, 1.0);
    const double skirt = 1.0 - smoothstep(vertical / 0.18);
    const double angle0 = 2.0 * std::numbers::pi *
        static_cast<double>(ring) / static_cast<double>(kAround);
    const double baseRadius =
        0.266 + 0.030 * body - 0.116 * fold + 0.045 * skirt;
    const double angle = angle0 + 0.08 * vertical;
    const double wrinkle = 1.0 +
        0.035 * std::sin(5.0 * angle + 5.0 * std::numbers::pi * vertical) +
        0.018 * std::sin(9.0 * angle - 3.0 * std::numbers::pi * vertical);
    const double looseRim = fold * (
        0.007 * std::sin(3.0 * angle + 0.35) +
        0.004 * std::sin(7.0 * angle - 0.80)
    );
    const double looseSkirt = skirt * (
        0.020 * std::sin(3.0 * angle - 0.40) +
        0.012 * std::sin(7.0 * angle + 0.70)
    );
    const double rimSag = fold * (
        0.016 * std::sin(angle + 0.55) +
        0.008 * std::sin(3.0 * angle - 0.30) +
        0.004 * std::sin(6.0 * angle + 0.90)
    );
    const double radius = baseRadius * wrinkle + looseRim + looseSkirt;
    const double bodySag = body * (
        0.018 * std::sin(2.0 * angle + 0.30) +
        0.009 * std::sin(5.0 * angle - 0.70)
    );
    const double skirtSag = skirt * (
        0.012 * std::sin(4.0 * angle + 0.20) +
        0.006 * std::sin(9.0 * angle - 0.50)
    );
    const double bodyHeight =
        0.012 + 0.155 * std::min(vertical / 0.72, 1.0) + bodySag;
    const double foldedHeight = fold < 0.35
        ? 0.167 + 0.038 * (fold / 0.35)
        : 0.205 - 0.025 * ((fold - 0.35) / 0.65);
    return {
        radius * std::cos(angle),
        radius * std::sin(angle),
        (vertical < 0.72 ? bodyHeight : foldedHeight) +
            rimSag + skirtSag + 0.009,
    };
}

std::pair<double, double> concentricBottomCoordinate(
    const std::uint32_t row,
    const std::uint32_t column
) {
    const double half = 0.5 * static_cast<double>(kBottomGrid - 1u);
    const double u = (static_cast<double>(column) - half) / half;
    const double v = (static_cast<double>(row) - half) / half;
    if (std::abs(u) < 1.0e-12 && std::abs(v) < 1.0e-12) {
        return {0.0, 0.0};
    }
    double radius{};
    double angle{};
    if (std::abs(u) >= std::abs(v)) {
        radius = u;
        angle = (std::numbers::pi / 4.0) * (v / u);
    } else {
        radius = v;
        angle = std::numbers::pi / 2.0 -
            (std::numbers::pi / 4.0) * (u / v);
    }
    return {radius * std::cos(angle), radius * std::sin(angle)};
}

bool bottomBoundary(
    const std::uint32_t row,
    const std::uint32_t column
) {
    return row == 0u || column == 0u ||
        row + 1u == kBottomGrid || column + 1u == kBottomGrid;
}

std::uint32_t bottomGridIndex(
    const std::uint32_t row,
    const std::uint32_t column
) {
    if (bottomBoundary(row, column)) {
        const auto [x, y] = concentricBottomCoordinate(row, column);
        double angle = std::atan2(y, x);
        if (angle < 0.0) {
            angle += 2.0 * std::numbers::pi;
        }
        return nodeIndex(
            0u,
            static_cast<std::uint32_t>(std::llround(
                angle * static_cast<double>(kAround) /
                (2.0 * std::numbers::pi)
            )) % kAround
        );
    }
    return kAround * kLevels +
        (row - 1u) * kBottomInterior + column - 1u;
}

DVec3 authoredBottomPosition(
    const std::uint32_t row,
    const std::uint32_t column
) {
    const auto [x, y] = concentricBottomCoordinate(row, column);
    const double radial = std::hypot(x, y);
    if (radial < 1.0e-12) {
        return {0.0, 0.0, 0.021};
    }
    double angle = std::atan2(y, x);
    if (angle < 0.0) {
        angle += 2.0 * std::numbers::pi;
    }
    const double ringCoordinate = angle * static_cast<double>(kAround) /
        (2.0 * std::numbers::pi);
    const std::uint32_t firstRing = static_cast<std::uint32_t>(
        std::floor(ringCoordinate)
    ) % kAround;
    const std::uint32_t secondRing = (firstRing + 1u) % kAround;
    const double fraction = ringCoordinate - std::floor(ringCoordinate);
    const DVec3 first = authoredPosition(0u, firstRing);
    const DVec3 second = authoredPosition(0u, secondRing);
    const DVec3 outer = first * (1.0 - fraction) + second * fraction;
    return {
        outer.x * radial,
        outer.y * radial,
        0.021 + radial * (outer.z - 0.021),
    };
}

struct InitialState {
    NumiClothBagGPUConfig config{};
    std::vector<NumiClothBagGPUParticle> particles;
    std::vector<NumiClothBagGPUDistance> distances;
    std::vector<NumiClothBagGPUGrip> grips;
    std::vector<NumiClothBagGPUKnot> knots;
    std::vector<NumiClothBagGPUBend> bends;
    std::vector<NumiClothBagGPUBatch> distanceBatches;
    std::vector<NumiClothBagGPUBatch> knotBatches;
    std::vector<NumiClothBagGPUBatch> bendBatches;
};

struct EdgeSpec {
    std::uint32_t first{};
    std::uint32_t second{};
    float compliance{};
    std::uint32_t kind{};
    std::uint32_t color{};
};

struct KnotSpec {
    std::array<std::uint32_t, 4> particles{};
    std::uint32_t color{};
};

struct BendSpec {
    std::uint32_t first{};
    std::uint32_t middle{};
    std::uint32_t third{};
    float compliance{};
    std::uint32_t color{};
};

InitialState makeInitialState() {
    InitialState result;
    result.particles.reserve(kParticleCount);
    const auto addParticle = [&result](const DVec3 position, const float mass) {
        NumiClothBagGPUParticle particle{};
        particle.positionAndInverseMass = f4(
            static_cast<float>(position.x),
            static_cast<float>(position.y),
            static_cast<float>(position.z),
            1.0f / mass
        );
        particle.previousAndMass = f4(
            static_cast<float>(position.x),
            static_cast<float>(position.y),
            static_cast<float>(position.z),
            mass
        );
        particle.velocity = f4(0.0f, 0.0f, 0.0f);
        result.particles.push_back(particle);
    };
    for (std::uint32_t level = 0u; level < kLevels; ++level) {
        for (std::uint32_t ring = 0u; ring < kAround; ++ring) {
            addParticle(
                authoredPosition(level, ring),
                level + 2u >= kLevels ? kHemMass : kOrdinaryMass
            );
        }
    }
    for (std::uint32_t row = 1u; row + 1u < kBottomGrid; ++row) {
        for (std::uint32_t column = 1u;
             column + 1u < kBottomGrid;
             ++column) {
            addParticle(authoredBottomPosition(row, column), kOrdinaryMass);
        }
    }
    if (result.particles.size() != kParticleCount) {
        throw std::logic_error("cloth particle topology count changed");
    }

    std::vector<EdgeSpec> edges;
    edges.reserve(kDistanceCount);
    const auto addEdge = [&edges](
        const std::uint32_t first,
        const std::uint32_t second,
        const float compliance,
        const std::uint32_t kind
    ) {
        edges.push_back({first, second, compliance, kind, 0u});
    };
    for (std::uint32_t level = 0u; level < kLevels; ++level) {
        for (std::uint32_t ring = 0u; ring < kAround; ++ring) {
            addEdge(
                nodeIndex(level, ring),
                nodeIndex(level, ring + 1u),
                level + 2u >= kLevels ? 1.0e-9f : 1.0e-8f,
                1u
            );
        }
    }
    for (std::uint32_t level = 0u; level + 1u < kLevels; ++level) {
        for (std::uint32_t ring = 0u; ring < kAround; ++ring) {
            addEdge(
                nodeIndex(level, ring),
                nodeIndex(level + 1u, ring),
                1.0e-8f,
                0u
            );
        }
    }
    for (std::uint32_t row = 0u; row < kBottomGrid; ++row) {
        for (std::uint32_t column = 0u;
             column + 1u < kBottomGrid;
             ++column) {
            if (!(bottomBoundary(row, column) &&
                  bottomBoundary(row, column + 1u))) {
                addEdge(
                    bottomGridIndex(row, column),
                    bottomGridIndex(row, column + 1u),
                    1.0e-8f,
                    2u
                );
            }
        }
    }
    for (std::uint32_t column = 0u; column < kBottomGrid; ++column) {
        for (std::uint32_t row = 0u; row + 1u < kBottomGrid; ++row) {
            if (!(bottomBoundary(row, column) &&
                  bottomBoundary(row + 1u, column))) {
                addEdge(
                    bottomGridIndex(row, column),
                    bottomGridIndex(row + 1u, column),
                    1.0e-8f,
                    2u
                );
            }
        }
    }
    if (edges.size() != kDistanceCount) {
        throw std::logic_error("cloth distance topology count changed");
    }

    std::vector<std::uint64_t> particleColors(kParticleCount, 0u);
    std::uint32_t colorCount = 0u;
    for (EdgeSpec& edge : edges) {
        const std::uint64_t unavailable =
            particleColors[edge.first] | particleColors[edge.second];
        const std::uint64_t available = ~unavailable;
        if (available == 0u) {
            throw std::logic_error("cloth edge coloring exceeds 64 colors");
        }
        edge.color = static_cast<std::uint32_t>(std::countr_zero(available));
        colorCount = std::max(colorCount, edge.color + 1u);
        const std::uint64_t bit = std::uint64_t{1u} << edge.color;
        particleColors[edge.first] |= bit;
        particleColors[edge.second] |= bit;
    }
    std::stable_sort(
        edges.begin(),
        edges.end(),
        [](const EdgeSpec& first, const EdgeSpec& second) {
            return first.color < second.color;
        }
    );
    result.distances.reserve(edges.size());
    std::uint32_t batchStart = 0u;
    for (std::uint32_t color = 0u; color < colorCount; ++color) {
        const std::uint32_t first = batchStart;
        while (batchStart < edges.size() &&
               edges[batchStart].color == color) {
            const EdgeSpec& edge = edges[batchStart];
            const double restLength = length(
                d3(result.particles[edge.second].positionAndInverseMass) -
                d3(result.particles[edge.first].positionAndInverseMass)
            );
            NumiClothBagGPUDistance distance{};
            distance.particlesAndColor = u4(
                edge.first, edge.second, edge.color, edge.kind
            );
            distance.material = f4(
                static_cast<float>(restLength),
                edge.compliance,
                0.0f,
                kStrainLimit
            );
            result.distances.push_back(distance);
            ++batchStart;
        }
        result.distanceBatches.push_back({
            u4(first, batchStart - first, color, 0u)
        });
    }

    std::vector<KnotSpec> knotSpecs;
    knotSpecs.reserve(kKnotCount);
    const auto addKnot = [&knotSpecs](
        const std::uint32_t warpFirst,
        const std::uint32_t warpSecond,
        const std::uint32_t weftFirst,
        const std::uint32_t weftSecond
    ) {
        knotSpecs.push_back({
            {{warpFirst, warpSecond, weftFirst, weftSecond}}, 0u
        });
    };
    for (std::uint32_t level = 1u; level + 1u < kLevels; ++level) {
        for (std::uint32_t ring = 0u; ring < kAround; ++ring) {
            addKnot(
                nodeIndex(level - 1u, ring),
                nodeIndex(level + 1u, ring),
                nodeIndex(level, ring + kAround - 1u),
                nodeIndex(level, ring + 1u)
            );
        }
    }
    for (std::uint32_t row = 1u; row + 1u < kBottomGrid; ++row) {
        for (std::uint32_t column = 1u;
             column + 1u < kBottomGrid;
             ++column) {
            addKnot(
                bottomGridIndex(row - 1u, column),
                bottomGridIndex(row + 1u, column),
                bottomGridIndex(row, column - 1u),
                bottomGridIndex(row, column + 1u)
            );
        }
    }
    if (knotSpecs.size() != kKnotCount) {
        throw std::logic_error("cloth knot topology count changed");
    }
    particleColors.assign(kParticleCount, 0u);
    std::uint32_t knotColorCount = 0u;
    for (KnotSpec& knot : knotSpecs) {
        std::uint64_t unavailable = 0u;
        for (const std::uint32_t particleIndex : knot.particles) {
            unavailable |= particleColors[particleIndex];
        }
        const std::uint64_t available = ~unavailable;
        if (available == 0u) {
            throw std::logic_error("cloth knot coloring exceeds 64 colors");
        }
        knot.color = static_cast<std::uint32_t>(
            std::countr_zero(available)
        );
        knotColorCount = std::max(knotColorCount, knot.color + 1u);
        const std::uint64_t bit = std::uint64_t{1u} << knot.color;
        for (const std::uint32_t particleIndex : knot.particles) {
            particleColors[particleIndex] |= bit;
        }
    }
    std::stable_sort(
        knotSpecs.begin(),
        knotSpecs.end(),
        [](const KnotSpec& first, const KnotSpec& second) {
            return first.color < second.color;
        }
    );
    result.knots.reserve(knotSpecs.size());
    batchStart = 0u;
    for (std::uint32_t color = 0u; color < knotColorCount; ++color) {
        const std::uint32_t first = batchStart;
        while (batchStart < knotSpecs.size() &&
               knotSpecs[batchStart].color == color) {
            const KnotSpec& source = knotSpecs[batchStart];
            const DVec3 warpVector =
                d3(result.particles[source.particles[1]].positionAndInverseMass) -
                d3(result.particles[source.particles[0]].positionAndInverseMass);
            const DVec3 weftVector =
                d3(result.particles[source.particles[3]].positionAndInverseMass) -
                d3(result.particles[source.particles[2]].positionAndInverseMass);
            NumiClothBagGPUKnot knot{};
            knot.particles = u4(
                source.particles[0],
                source.particles[1],
                source.particles[2],
                source.particles[3]
            );
            knot.control = u4(source.color, 0u, 0u, 0u);
            knot.material = f4(
                static_cast<float>(dot(
                    warpVector / length(warpVector),
                    weftVector / length(weftVector)
                )),
                2.0e-6f,
                0.0f,
                0.0f
            );
            result.knots.push_back(knot);
            ++batchStart;
        }
        result.knotBatches.push_back({
            u4(first, batchStart - first, color, 0u)
        });
    }

    std::vector<BendSpec> bendSpecs;
    bendSpecs.reserve(kBendCount);
    const auto addBend = [&bendSpecs](
        const std::uint32_t first,
        const std::uint32_t middle,
        const std::uint32_t third,
        const float compliance
    ) {
        bendSpecs.push_back({first, middle, third, compliance, 0u});
    };
    for (std::uint32_t level = 0u; level < kLevels; ++level) {
        for (std::uint32_t ring = 0u; ring < kAround; ++ring) {
            addBend(
                nodeIndex(level, ring + kAround - 1u),
                nodeIndex(level, ring),
                nodeIndex(level, ring + 1u),
                level + 2u >= kLevels ? 1.0e-8f : 8.0e-2f
            );
        }
    }
    for (std::uint32_t level = 1u; level + 1u < kLevels; ++level) {
        for (std::uint32_t ring = 0u; ring < kAround; ++ring) {
            addBend(
                nodeIndex(level - 1u, ring),
                nodeIndex(level, ring),
                nodeIndex(level + 1u, ring),
                8.0e-2f
            );
        }
    }
    for (std::uint32_t row = 1u; row + 1u < kBottomGrid; ++row) {
        for (std::uint32_t column = 1u;
             column + 1u < kBottomGrid;
             ++column) {
            addBend(
                bottomGridIndex(row, column - 1u),
                bottomGridIndex(row, column),
                bottomGridIndex(row, column + 1u),
                8.0e-2f
            );
            addBend(
                bottomGridIndex(column - 1u, row),
                bottomGridIndex(column, row),
                bottomGridIndex(column + 1u, row),
                8.0e-2f
            );
        }
    }
    if (bendSpecs.size() != kBendCount) {
        throw std::logic_error("cloth bend topology count changed");
    }
    particleColors.assign(kParticleCount, 0u);
    std::uint32_t bendColorCount = 0u;
    for (BendSpec& bend : bendSpecs) {
        const std::uint64_t unavailable =
            particleColors[bend.first] | particleColors[bend.third];
        const std::uint64_t available = ~unavailable;
        if (available == 0u) {
            throw std::logic_error("cloth bend coloring exceeds 64 colors");
        }
        bend.color = static_cast<std::uint32_t>(
            std::countr_zero(available)
        );
        bendColorCount = std::max(bendColorCount, bend.color + 1u);
        const std::uint64_t bit = std::uint64_t{1u} << bend.color;
        particleColors[bend.first] |= bit;
        particleColors[bend.third] |= bit;
    }
    std::stable_sort(
        bendSpecs.begin(),
        bendSpecs.end(),
        [](const BendSpec& first, const BendSpec& second) {
            return first.color < second.color;
        }
    );
    result.bends.reserve(bendSpecs.size());
    batchStart = 0u;
    for (std::uint32_t color = 0u; color < bendColorCount; ++color) {
        const std::uint32_t first = batchStart;
        while (batchStart < bendSpecs.size() &&
               bendSpecs[batchStart].color == color) {
            const BendSpec& source = bendSpecs[batchStart];
            const DVec3 firstPosition = d3(
                result.particles[source.first].positionAndInverseMass
            );
            const DVec3 middlePosition = d3(
                result.particles[source.middle].positionAndInverseMass
            );
            const DVec3 thirdPosition = d3(
                result.particles[source.third].positionAndInverseMass
            );
            NumiClothBagGPUBend bend{};
            bend.particlesAndColor = u4(
                source.first, source.middle, source.third, source.color
            );
            bend.material = f4(
                static_cast<float>(length(thirdPosition - firstPosition)),
                static_cast<float>(
                    length(middlePosition - firstPosition) +
                    length(thirdPosition - middlePosition)
                ),
                source.compliance,
                0.0f
            );
            result.bends.push_back(bend);
            ++batchStart;
        }
        result.bendBatches.push_back({
            u4(first, batchStart - first, color, 0u)
        });
    }

    const DVec3 base = authoredPosition(kLevels - 1u, 0u);
    result.config.control = u4(
        NUMI_CLOTH_BAG_GPU_ABI_VERSION,
        kParticleCount,
        kDistanceCount,
        kGripCount
    );
    result.config.constraintCounts = u4(
        kKnotCount, kBendCount, 0u, 0u
    );
    result.config.gravityAndTimestep = f4(0.0f, 0.0f, -9.81f, kTimestep);
    result.config.gripTargetAndActive = f4(
        static_cast<float>(base.x + 0.002),
        static_cast<float>(base.y - 0.001),
        static_cast<float>(base.z + 0.006),
        1.0f
    );
    result.config.clothMaterial = f4(0.004f, 0.0f, 0.0f, 0.0f);
    result.grips.reserve(kGripCount);
    for (std::uint32_t level = kLevels - 2u; level < kLevels; ++level) {
        for (const int offset : {-2, -1, 0, 1, 2}) {
            const std::uint32_t ring = static_cast<std::uint32_t>(
                (static_cast<int>(kAround) + offset) %
                static_cast<int>(kAround)
            );
            const std::uint32_t particleIndex = nodeIndex(level, ring);
            const DVec3 rest = d3(
                result.particles[particleIndex].positionAndInverseMass
            );
            NumiClothBagGPUGrip grip{};
            grip.particle = u4(particleIndex, 0u, 0u, 0u);
            grip.targetOffsetAndCompliance = f4(
                static_cast<float>(rest.x - base.x),
                static_cast<float>(rest.y - base.y),
                static_cast<float>(rest.z - base.z),
                kGripCompliance
            );
            grip.lambda = f4(0.0f, 0.0f, 0.0f);
            result.grips.push_back(grip);
        }
    }
    return result;
}

InitialState makeStrainProbeState() {
    InitialState result;
    result.config.control = u4(
        NUMI_CLOTH_BAG_GPU_ABI_VERSION, 4u, 2u, 0u
    );
    result.config.constraintCounts = u4(0u, 0u, 0u, 0u);
    result.config.gravityAndTimestep = f4(0.0f, 0.0f, 0.0f, 0.01f);
    result.config.gripTargetAndActive = f4(0.0f, 0.0f, 0.0f, 0.0f);
    result.config.clothMaterial = f4(0.004f, 0.0f, 0.0f, 0.0f);
    const auto particle = [](const float x, const float inverseMass) {
        NumiClothBagGPUParticle value{};
        value.positionAndInverseMass = f4(x, 0.0f, 0.0f, inverseMass);
        value.previousAndMass = f4(
            x, 0.0f, 0.0f, 1.0f / inverseMass
        );
        value.velocity = f4(0.0f, 0.0f, 0.0f);
        return value;
    };
    result.particles = {
        particle(0.0f, 1.0f),
        particle(1.5f, 2.0f),
        particle(3.0f, 1.0f),
        particle(3.6f, 2.0f),
    };
    NumiClothBagGPUDistance extension{};
    extension.particlesAndColor = u4(0u, 1u, 0u, 0u);
    extension.material = f4(1.0f, 0.0f, 0.0f, kStrainLimit);
    NumiClothBagGPUDistance compression{};
    compression.particlesAndColor = u4(2u, 3u, 0u, 0u);
    compression.material = f4(1.0f, 0.0f, 0.0f, kStrainLimit);
    result.distances = {extension, compression};
    result.distanceBatches = {{u4(0u, 2u, 0u, 0u)}};
    return result;
}

InitialState makeGroundBendProbeState() {
    InitialState result;
    result.config.control = u4(
        NUMI_CLOTH_BAG_GPU_ABI_VERSION, 3u, 0u, 0u
    );
    result.config.constraintCounts = u4(0u, 1u, 1u, 0u);
    result.config.gravityAndTimestep = f4(0.0f, 0.0f, 0.0f, 0.01f);
    result.config.gripTargetAndActive = f4(0.0f, 0.0f, 0.0f, 0.0f);
    result.config.clothMaterial = f4(0.004f, 0.0f, 0.0f, 0.0f);
    const auto particle = [](const DVec3 position) {
        NumiClothBagGPUParticle value{};
        value.positionAndInverseMass = f4(
            static_cast<float>(position.x),
            static_cast<float>(position.y),
            static_cast<float>(position.z),
            1.0f
        );
        value.previousAndMass = f4(
            static_cast<float>(position.x),
            static_cast<float>(position.y),
            static_cast<float>(position.z),
            1.0f
        );
        value.velocity = f4(0.0f, 0.0f, 0.0f);
        return value;
    };
    result.particles = {
        particle({0.0, 0.0, 0.004}),
        particle({0.5, 0.0, 0.054}),
        particle({1.0, 0.0, 0.104}),
    };
    NumiClothBagGPUBend bend{};
    bend.particlesAndColor = u4(0u, 1u, 2u, 0u);
    bend.material = f4(1.2f, 1.2f, 0.0f, 0.0f);
    result.bends = {bend};
    result.bendBatches = {{u4(0u, 1u, 0u, 0u)}};
    return result;
}

bool verifyColoring(const InitialState& state) {
    for (const NumiClothBagGPUBatch& batch : state.distanceBatches) {
        std::vector<bool> used(state.particles.size(), false);
        for (std::uint32_t local = 0u; local < batch.control.y; ++local) {
            const auto& constraint = state.distances[
                batch.control.x + local
            ];
            if (constraint.particlesAndColor.z != batch.control.z ||
                used[constraint.particlesAndColor.x] ||
                used[constraint.particlesAndColor.y]) {
                return false;
            }
            used[constraint.particlesAndColor.x] = true;
            used[constraint.particlesAndColor.y] = true;
        }
    }
    for (const NumiClothBagGPUBatch& batch : state.knotBatches) {
        std::vector<bool> used(state.particles.size(), false);
        for (std::uint32_t local = 0u; local < batch.control.y; ++local) {
            const auto& constraint = state.knots[batch.control.x + local];
            if (constraint.control.x != batch.control.z) {
                return false;
            }
            const std::array<std::uint32_t, 4> participants{{
                constraint.particles.x,
                constraint.particles.y,
                constraint.particles.z,
                constraint.particles.w,
            }};
            for (const std::uint32_t particleIndex : participants) {
                if (particleIndex >= used.size() || used[particleIndex]) {
                    return false;
                }
                used[particleIndex] = true;
            }
        }
    }
    for (const NumiClothBagGPUBatch& batch : state.bendBatches) {
        std::vector<bool> used(state.particles.size(), false);
        for (std::uint32_t local = 0u; local < batch.control.y; ++local) {
            const auto& constraint = state.bends[batch.control.x + local];
            if (constraint.particlesAndColor.w != batch.control.z ||
                constraint.particlesAndColor.x >= used.size() ||
                constraint.particlesAndColor.z >= used.size() ||
                used[constraint.particlesAndColor.x] ||
                used[constraint.particlesAndColor.z]) {
                return false;
            }
            used[constraint.particlesAndColor.x] = true;
            used[constraint.particlesAndColor.z] = true;
        }
    }
    std::vector<bool> gripUsed(state.particles.size(), false);
    for (const auto& grip : state.grips) {
        if (grip.particle.x >= gripUsed.size() || gripUsed[grip.particle.x]) {
            return false;
        }
        gripUsed[grip.particle.x] = true;
    }
    return true;
}

struct OracleParticle {
    DVec3 position{};
    DVec3 previous{};
    DVec3 velocity{};
    double inverseMass{};
};

struct OracleDistance {
    std::uint32_t first{};
    std::uint32_t second{};
    double restLength{};
    double compliance{};
    double lambda{};
    double extensionLimit{};
};

struct OracleGrip {
    std::uint32_t particle{};
    DVec3 offset{};
    DVec3 lambda{};
    double compliance{};
};

struct OracleKnot {
    std::array<std::uint32_t, 4> particles{};
    double restCosine{};
    double compliance{};
    double lambda{};
};

struct OracleBend {
    std::uint32_t first{};
    std::uint32_t middle{};
    std::uint32_t third{};
    double restChord{};
    double restArc{};
    double compliance{};
    double lambda{};
};

struct OracleResult {
    std::vector<OracleParticle> particles;
    std::vector<OracleDistance> distances;
    std::vector<OracleGrip> grips;
    std::vector<OracleKnot> knots;
    std::vector<OracleBend> bends;
};

OracleResult runOracle(
    const InitialState& initial,
    const std::uint32_t iterations,
    const std::uint32_t strainSweeps
) {
    OracleResult result;
    result.particles.reserve(initial.particles.size());
    for (const auto& source : initial.particles) {
        result.particles.push_back({
            d3(source.positionAndInverseMass),
            d3(source.previousAndMass),
            d3(source.velocity),
            static_cast<double>(source.positionAndInverseMass.w),
        });
    }
    result.distances.reserve(initial.distances.size());
    for (const auto& source : initial.distances) {
        result.distances.push_back({
            source.particlesAndColor.x,
            source.particlesAndColor.y,
            static_cast<double>(source.material.x),
            static_cast<double>(source.material.y),
            0.0,
            static_cast<double>(source.material.w),
        });
    }
    result.grips.reserve(initial.grips.size());
    for (const auto& source : initial.grips) {
        result.grips.push_back({
            source.particle.x,
            d3(source.targetOffsetAndCompliance),
            {},
            static_cast<double>(source.targetOffsetAndCompliance.w),
        });
    }
    result.knots.reserve(initial.knots.size());
    for (const auto& source : initial.knots) {
        result.knots.push_back({
            {{
                source.particles.x,
                source.particles.y,
                source.particles.z,
                source.particles.w,
            }},
            static_cast<double>(source.material.x),
            static_cast<double>(source.material.y),
            0.0,
        });
    }
    result.bends.reserve(initial.bends.size());
    for (const auto& source : initial.bends) {
        result.bends.push_back({
            source.particlesAndColor.x,
            source.particlesAndColor.y,
            source.particlesAndColor.z,
            static_cast<double>(source.material.x),
            static_cast<double>(source.material.y),
            static_cast<double>(source.material.z),
            0.0,
        });
    }
    const double timestep = initial.config.gravityAndTimestep.w;
    const DVec3 gravity = d3(initial.config.gravityAndTimestep);
    const DVec3 gripTarget = d3(initial.config.gripTargetAndActive);
    for (OracleParticle& particle : result.particles) {
        particle.previous = particle.position;
        if (particle.inverseMass > 0.0) {
            particle.velocity += gravity * timestep;
            particle.position += particle.velocity * timestep;
        }
    }
    for (std::uint32_t iteration = 0u; iteration < iterations; ++iteration) {
        for (const NumiClothBagGPUBatch& batch : initial.distanceBatches) {
            for (std::uint32_t local = 0u; local < batch.control.y; ++local) {
                OracleDistance& constraint =
                    result.distances[batch.control.x + local];
                OracleParticle& first = result.particles[constraint.first];
                OracleParticle& second = result.particles[constraint.second];
                const DVec3 difference = second.position - first.position;
                const double currentLength = length(difference);
                const double alpha = constraint.compliance /
                    (timestep * timestep);
                const double denominator =
                    first.inverseMass + second.inverseMass + alpha;
                if (!(currentLength > 1.0e-12) || !(denominator > 0.0)) {
                    continue;
                }
                const double value = currentLength - constraint.restLength;
                const double deltaLambda =
                    (-value - alpha * constraint.lambda) / denominator;
                constraint.lambda += deltaLambda;
                const DVec3 correction = difference *
                    (deltaLambda / currentLength);
                first.position -= correction * first.inverseMass;
                second.position += correction * second.inverseMass;
            }
        }
        for (const NumiClothBagGPUBatch& batch : initial.knotBatches) {
            for (std::uint32_t local = 0u; local < batch.control.y; ++local) {
                OracleKnot& constraint =
                    result.knots[batch.control.x + local];
                OracleParticle& warpFirst =
                    result.particles[constraint.particles[0]];
                OracleParticle& warpSecond =
                    result.particles[constraint.particles[1]];
                OracleParticle& weftFirst =
                    result.particles[constraint.particles[2]];
                OracleParticle& weftSecond =
                    result.particles[constraint.particles[3]];
                const DVec3 warpVector =
                    warpSecond.position - warpFirst.position;
                const DVec3 weftVector =
                    weftSecond.position - weftFirst.position;
                const double warpLength = length(warpVector);
                const double weftLength = length(weftVector);
                if (!(warpLength > 1.0e-12) ||
                    !(weftLength > 1.0e-12)) {
                    continue;
                }
                const DVec3 warp = warpVector / warpLength;
                const DVec3 weft = weftVector / weftLength;
                const double cosine = std::clamp(dot(warp, weft), -1.0, 1.0);
                const double value = cosine - constraint.restCosine;
                const DVec3 warpGradient =
                    (weft - warp * cosine) / warpLength;
                const DVec3 weftGradient =
                    (warp - weft * cosine) / weftLength;
                const std::array<DVec3, 4> gradients{{
                    warpGradient * -1.0,
                    warpGradient,
                    weftGradient * -1.0,
                    weftGradient,
                }};
                const std::array<double, 4> inverseMasses{{
                    warpFirst.inverseMass,
                    warpSecond.inverseMass,
                    weftFirst.inverseMass,
                    weftSecond.inverseMass,
                }};
                const double alpha = constraint.compliance /
                    (timestep * timestep);
                double denominator = alpha;
                for (std::size_t participant = 0u;
                     participant < gradients.size();
                     ++participant) {
                    denominator += inverseMasses[participant] *
                        dot(gradients[participant], gradients[participant]);
                }
                if (!(denominator > 0.0)) {
                    continue;
                }
                const double deltaLambda =
                    (-value - alpha * constraint.lambda) / denominator;
                constraint.lambda += deltaLambda;
                warpFirst.position += gradients[0] *
                    (inverseMasses[0] * deltaLambda);
                warpSecond.position += gradients[1] *
                    (inverseMasses[1] * deltaLambda);
                weftFirst.position += gradients[2] *
                    (inverseMasses[2] * deltaLambda);
                weftSecond.position += gradients[3] *
                    (inverseMasses[3] * deltaLambda);
            }
        }
        if (iteration % 2u == 0u) {
            for (const NumiClothBagGPUBatch& batch : initial.bendBatches) {
                for (std::uint32_t local = 0u;
                     local < batch.control.y;
                     ++local) {
                    OracleBend& constraint =
                        result.bends[batch.control.x + local];
                    OracleParticle& first =
                        result.particles[constraint.first];
                    OracleParticle& third =
                        result.particles[constraint.third];
                    const DVec3 difference = third.position - first.position;
                    const double currentChord = length(difference);
                    if (!(currentChord > 1.0e-12)) {
                        continue;
                    }
                    const double value =
                        currentChord - constraint.restChord;
                    const DVec3 direction = difference / currentChord;
                    const std::array<DVec3, 2> gradients{{
                        direction * -1.0, direction
                    }};
                    const std::array<double, 2> inverseMasses{{
                        first.inverseMass, third.inverseMass
                    }};
                    const double alpha = constraint.compliance /
                        (timestep * timestep);
                    double freeDenominator = alpha;
                    for (std::size_t participant = 0u;
                         participant < gradients.size();
                         ++participant) {
                        freeDenominator += inverseMasses[participant] *
                            dot(
                                gradients[participant],
                                gradients[participant]
                            );
                    }
                    if (!(freeDenominator >= 1.0e-16)) {
                        continue;
                    }
                    const double numerator =
                        -value - alpha * constraint.lambda;
                    const double freeDeltaLambda =
                        numerator / freeDenominator;
                    const bool groundEnabled =
                        initial.config.constraintCounts.z != 0u;
                    const double groundHeight =
                        initial.config.clothMaterial.x;
                    const std::array<DVec3, 2> positions{{
                        first.position, third.position
                    }};
                    std::array<bool, 2> groundActive{};
                    double denominator = alpha;
                    for (std::size_t participant = 0u;
                         participant < gradients.size();
                         ++participant) {
                        groundActive[participant] = groundEnabled &&
                            positions[participant].z <=
                                groundHeight + 1.0e-9 &&
                            gradients[participant].z * freeDeltaLambda < 0.0;
                        denominator += inverseMasses[participant] * (
                            dot(
                                gradients[participant],
                                gradients[participant]
                            ) -
                            (groundActive[participant]
                                ? gradients[participant].z *
                                    gradients[participant].z
                                : 0.0)
                        );
                    }
                    if (!(denominator >= 1.0e-16)) {
                        continue;
                    }
                    const double deltaLambda = numerator / denominator;
                    double fraction = 1.0;
                    if (groundEnabled) {
                        for (std::size_t participant = 0u;
                             participant < gradients.size();
                             ++participant) {
                            if (groundActive[participant]) {
                                continue;
                            }
                            const double verticalCorrection =
                                gradients[participant].z *
                                inverseMasses[participant] * deltaLambda;
                            if (verticalCorrection < 0.0) {
                                fraction = std::min(
                                    fraction,
                                    std::max(
                                        0.0,
                                        positions[participant].z - groundHeight
                                    ) / -verticalCorrection
                                );
                            }
                        }
                    }
                    const double appliedLambda = deltaLambda * fraction;
                    constraint.lambda += appliedLambda;
                    DVec3 firstCorrection = gradients[0] *
                        (inverseMasses[0] * appliedLambda);
                    DVec3 thirdCorrection = gradients[1] *
                        (inverseMasses[1] * appliedLambda);
                    if (groundActive[0]) {
                        firstCorrection.z = 0.0;
                    }
                    if (groundActive[1]) {
                        thirdCorrection.z = 0.0;
                    }
                    first.position += firstCorrection;
                    third.position += thirdCorrection;
                    if (groundEnabled) {
                        first.position.z = std::max(
                            first.position.z, groundHeight
                        );
                        third.position.z = std::max(
                            third.position.z, groundHeight
                        );
                    }
                }
            }
        }
        for (OracleGrip& grip : result.grips) {
            OracleParticle& particle = result.particles[grip.particle];
            const double alpha = grip.compliance / (timestep * timestep);
            const double denominator = particle.inverseMass + alpha;
            if (!(denominator > 0.0)) {
                continue;
            }
            const DVec3 value = particle.position - (gripTarget + grip.offset);
            const DVec3 deltaLambda =
                (value * -1.0 - grip.lambda * alpha) / denominator;
            grip.lambda += deltaLambda;
            particle.position += deltaLambda * particle.inverseMass;
        }
    }
    for (std::uint32_t sweep = 0u; sweep < strainSweeps; ++sweep) {
        for (const NumiClothBagGPUBatch& batch : initial.distanceBatches) {
            for (std::uint32_t local = 0u; local < batch.control.y; ++local) {
                const OracleDistance& constraint =
                    result.distances[batch.control.x + local];
                OracleParticle& first = result.particles[constraint.first];
                OracleParticle& second = result.particles[constraint.second];
                const DVec3 difference = second.position - first.position;
                const double currentLength = length(difference);
                const double maximumLength = constraint.restLength *
                    (1.0 + constraint.extensionLimit);
                if (!(currentLength > maximumLength) ||
                    !(currentLength > 1.0e-12)) {
                    continue;
                }
                const double denominator =
                    first.inverseMass + second.inverseMass;
                if (!(denominator > 0.0)) {
                    continue;
                }
                const double correctionMagnitude =
                    currentLength - maximumLength;
                const DVec3 direction = difference / currentLength;
                first.position += direction *
                    (first.inverseMass * correctionMagnitude / denominator);
                second.position -= direction *
                    (second.inverseMass * correctionMagnitude / denominator);
            }
        }
    }
    for (OracleParticle& particle : result.particles) {
        particle.velocity =
            (particle.position - particle.previous) / timestep;
    }
    return result;
}

std::string errorText(NSError* error) {
    return error == nil
        ? std::string("unknown Metal error")
        : std::string(error.localizedDescription.UTF8String);
}

id<MTLComputePipelineState> makePipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString* name
) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) {
        throw std::runtime_error(
            "solver metallib lacks kernel: " +
            std::string(name.UTF8String)
        );
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:function
                                      error:&error];
    if (pipeline == nil) {
        throw std::runtime_error(
            "failed to create pipeline: " + errorText(error)
        );
    }
    return pipeline;
}

void dispatch(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    const NSUInteger count
) {
    if (count == 0u) {
        return;
    }
    const NSUInteger width = std::min<NSUInteger>(
        pipeline.maxTotalThreadsPerThreadgroup,
        256u
    );
    [encoder setComputePipelineState:pipeline];
    [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
          threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
}

struct GPUResult {
    std::vector<NumiClothBagGPUParticle> particles;
    std::vector<NumiClothBagGPUDistance> distances;
    std::vector<NumiClothBagGPUGrip> grips;
    std::vector<NumiClothBagGPUKnot> knots;
    std::vector<NumiClothBagGPUBend> bends;
    std::uint32_t failure{};
    double seconds{};
};

struct Pipelines {
    id<MTLComputePipelineState> begin;
    id<MTLComputePipelineState> distance;
    id<MTLComputePipelineState> knot;
    id<MTLComputePipelineState> bend;
    id<MTLComputePipelineState> grip;
    id<MTLComputePipelineState> strain;
    id<MTLComputePipelineState> finalize;
};

GPUResult runGPU(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    const Pipelines& pipelines,
    const InitialState& initial,
    const std::uint32_t iterations,
    const std::uint32_t strainSweeps
) {
    const auto makeBytes = [device](const auto& values) {
        using Value = typename std::decay_t<decltype(values)>::value_type;
        const NSUInteger bytes = values.size() * sizeof(Value);
        id<MTLBuffer> buffer = [device
            newBufferWithLength:std::max<NSUInteger>(bytes, 1u)
                        options:MTLResourceStorageModeShared];
        if (buffer != nil && bytes > 0u) {
            std::memcpy(buffer.contents, values.data(), bytes);
        }
        return buffer;
    };
    id<MTLBuffer> configBuffer = [device
        newBufferWithBytes:&initial.config
                   length:sizeof(initial.config)
                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> particleBuffer = makeBytes(initial.particles);
    id<MTLBuffer> distanceBuffer = makeBytes(initial.distances);
    id<MTLBuffer> gripBuffer = makeBytes(initial.grips);
    id<MTLBuffer> knotBuffer = makeBytes(initial.knots);
    id<MTLBuffer> bendBuffer = makeBytes(initial.bends);
    std::uint32_t zero = 0u;
    id<MTLBuffer> failureBuffer = [device
        newBufferWithBytes:&zero
                   length:sizeof(zero)
                  options:MTLResourceStorageModeShared];
    if (configBuffer == nil || particleBuffer == nil ||
        distanceBuffer == nil || gripBuffer == nil ||
        knotBuffer == nil || bendBuffer == nil ||
        failureBuffer == nil) {
        throw std::runtime_error("failed to allocate Metal cloth buffers");
    }

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (commandBuffer == nil || encoder == nil) {
        throw std::runtime_error("failed to create Metal cloth encoder");
    }
    [encoder setBuffer:configBuffer offset:0 atIndex:0];
    [encoder setBuffer:particleBuffer offset:0 atIndex:1];
    [encoder setBuffer:distanceBuffer offset:0 atIndex:2];
    [encoder setBuffer:gripBuffer offset:0 atIndex:3];
    [encoder setBuffer:knotBuffer offset:0 atIndex:4];
    [encoder setBuffer:bendBuffer offset:0 atIndex:5];
    [encoder setBuffer:failureBuffer offset:0 atIndex:6];
    dispatch(
        encoder,
        pipelines.begin,
        std::max({
            initial.particles.size(),
            initial.distances.size(),
            initial.grips.size(),
            initial.knots.size(),
            initial.bends.size(),
        })
    );

    for (std::uint32_t iteration = 0u; iteration < iterations; ++iteration) {
        for (const NumiClothBagGPUBatch& batch : initial.distanceBatches) {
            [encoder setComputePipelineState:pipelines.distance];
            [encoder setBuffer:configBuffer offset:0 atIndex:0];
            [encoder setBuffer:particleBuffer offset:0 atIndex:1];
            [encoder setBuffer:distanceBuffer offset:0 atIndex:2];
            [encoder setBytes:&batch length:sizeof(batch) atIndex:3];
            [encoder setBuffer:failureBuffer offset:0 atIndex:4];
            dispatch(encoder, pipelines.distance, batch.control.y);
        }
        for (const NumiClothBagGPUBatch& batch : initial.knotBatches) {
            [encoder setComputePipelineState:pipelines.knot];
            [encoder setBuffer:configBuffer offset:0 atIndex:0];
            [encoder setBuffer:particleBuffer offset:0 atIndex:1];
            [encoder setBuffer:knotBuffer offset:0 atIndex:2];
            [encoder setBytes:&batch length:sizeof(batch) atIndex:3];
            [encoder setBuffer:failureBuffer offset:0 atIndex:4];
            dispatch(encoder, pipelines.knot, batch.control.y);
        }
        if (iteration % 2u == 0u) {
            for (const NumiClothBagGPUBatch& batch : initial.bendBatches) {
                [encoder setComputePipelineState:pipelines.bend];
                [encoder setBuffer:configBuffer offset:0 atIndex:0];
                [encoder setBuffer:particleBuffer offset:0 atIndex:1];
                [encoder setBuffer:bendBuffer offset:0 atIndex:2];
                [encoder setBytes:&batch length:sizeof(batch) atIndex:3];
                [encoder setBuffer:failureBuffer offset:0 atIndex:4];
                dispatch(encoder, pipelines.bend, batch.control.y);
            }
        }
        [encoder setComputePipelineState:pipelines.grip];
        [encoder setBuffer:configBuffer offset:0 atIndex:0];
        [encoder setBuffer:particleBuffer offset:0 atIndex:1];
        [encoder setBuffer:gripBuffer offset:0 atIndex:2];
        [encoder setBuffer:failureBuffer offset:0 atIndex:3];
        dispatch(encoder, pipelines.grip, initial.grips.size());
    }
    for (std::uint32_t sweep = 0u; sweep < strainSweeps; ++sweep) {
        for (const NumiClothBagGPUBatch& batch : initial.distanceBatches) {
            [encoder setComputePipelineState:pipelines.strain];
            [encoder setBuffer:configBuffer offset:0 atIndex:0];
            [encoder setBuffer:particleBuffer offset:0 atIndex:1];
            [encoder setBuffer:distanceBuffer offset:0 atIndex:2];
            [encoder setBytes:&batch length:sizeof(batch) atIndex:3];
            [encoder setBuffer:failureBuffer offset:0 atIndex:4];
            dispatch(encoder, pipelines.strain, batch.control.y);
        }
    }
    [encoder setComputePipelineState:pipelines.finalize];
    [encoder setBuffer:configBuffer offset:0 atIndex:0];
    [encoder setBuffer:particleBuffer offset:0 atIndex:1];
    [encoder setBuffer:failureBuffer offset:0 atIndex:2];
    dispatch(encoder, pipelines.finalize, initial.particles.size());
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted ||
        commandBuffer.error != nil) {
        throw std::runtime_error(
            "Metal cloth command failed: " + errorText(commandBuffer.error)
        );
    }

    GPUResult result;
    const auto assign = [](auto& output, id<MTLBuffer> buffer, const auto& source) {
        using Value = typename std::decay_t<decltype(output)>::value_type;
        const auto* values = static_cast<const Value*>(buffer.contents);
        output.assign(values, values + source.size());
    };
    assign(result.particles, particleBuffer, initial.particles);
    assign(result.distances, distanceBuffer, initial.distances);
    assign(result.grips, gripBuffer, initial.grips);
    assign(result.knots, knotBuffer, initial.knots);
    assign(result.bends, bendBuffer, initial.bends);
    result.failure = *static_cast<const std::uint32_t*>(failureBuffer.contents);
    if (commandBuffer.GPUEndTime >= commandBuffer.GPUStartTime) {
        result.seconds = commandBuffer.GPUEndTime - commandBuffer.GPUStartTime;
    }
    return result;
}

template <typename Value>
bool bitwiseEqual(
    const std::vector<Value>& first,
    const std::vector<Value>& second
) {
    return first.size() == second.size() &&
        std::memcmp(
            first.data(),
            second.data(),
            first.size() * sizeof(Value)
        ) == 0;
}

std::uint64_t hashGPUResult(const GPUResult& result) {
    std::uint64_t hash = 1469598103934665603ull;
    const auto append = [&hash](const auto& values) {
        const auto* bytes = reinterpret_cast<const std::uint8_t*>(
            values.data()
        );
        const std::size_t count = values.size() * sizeof(values.front());
        for (std::size_t index = 0u; index < count; ++index) {
            hash ^= bytes[index];
            hash *= 1099511628211ull;
        }
    };
    append(result.particles);
    append(result.distances);
    append(result.grips);
    append(result.knots);
    append(result.bends);
    hash ^= result.failure;
    hash *= 1099511628211ull;
    return hash;
}

double maximumStrainViolation(
    const std::vector<NumiClothBagGPUParticle>& particles,
    const std::vector<NumiClothBagGPUDistance>& distances
) {
    double maximum = 0.0;
    for (const auto& constraint : distances) {
        const double currentLength = length(
            d3(particles[constraint.particlesAndColor.y].positionAndInverseMass) -
            d3(particles[constraint.particlesAndColor.x].positionAndInverseMass)
        );
        const double limit = static_cast<double>(constraint.material.x) *
            (1.0 + static_cast<double>(constraint.material.w));
        maximum = std::max(maximum, currentLength - limit);
    }
    return maximum;
}

int run(const int argc, const char* const* argv) {
    std::uint32_t replays = 2u;
    std::uint32_t iterations = 32u;
    std::uint32_t strainSweeps = 3u;
    std::string metallibPath = NUMI_TEMPORAL_CONE_METALLIB;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string_view value(argv[argument]);
        if (value == "--replays" && argument + 1 < argc) {
            replays = static_cast<std::uint32_t>(
                std::stoul(argv[++argument])
            );
        } else if (value == "--iterations" && argument + 1 < argc) {
            iterations = static_cast<std::uint32_t>(
                std::stoul(argv[++argument])
            );
        } else if (value == "--strain-sweeps" && argument + 1 < argc) {
            strainSweeps = static_cast<std::uint32_t>(
                std::stoul(argv[++argument])
            );
        } else if (value == "--metallib" && argument + 1 < argc) {
            metallibPath = argv[++argument];
        } else if (value == "--help") {
            std::cout
                << "usage: numi-solver-cloth-metal [--replays N] "
                   "[--iterations N] [--strain-sweeps N] "
                   "[--metallib PATH]\n";
            return 0;
        } else {
            throw std::runtime_error(
                "unknown argument: " + std::string(value)
            );
        }
    }
    replays = std::max(replays, 2u);
    if (iterations == 0u || strainSweeps == 0u) {
        throw std::runtime_error("iterations and strain sweeps must be positive");
    }

    const InitialState initial = makeInitialState();
    const bool coloringExact = verifyColoring(initial);
    const OracleResult oracle = runOracle(initial, iterations, strainSweeps);

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        throw std::runtime_error("no Apple Metal device is available");
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
        throw std::runtime_error("failed to create Metal command queue");
    }
    NSError* error = nil;
    NSString* path = [NSString stringWithUTF8String:metallibPath.c_str()];
    id<MTLLibrary> library = [device
        newLibraryWithURL:[NSURL fileURLWithPath:path]
                    error:&error];
    if (library == nil) {
        throw std::runtime_error(
            "failed to load solver metallib: " + errorText(error)
        );
    }
    const Pipelines pipelines{
        makePipeline(device, library, @"numi_cloth_bag_begin_substep"),
        makePipeline(device, library, @"numi_cloth_bag_solve_distance"),
        makePipeline(device, library, @"numi_cloth_bag_solve_knot"),
        makePipeline(device, library, @"numi_cloth_bag_solve_bend"),
        makePipeline(device, library, @"numi_cloth_bag_solve_grip"),
        makePipeline(device, library, @"numi_cloth_bag_limit_strain"),
        makePipeline(device, library, @"numi_cloth_bag_finalize_substep"),
    };

    std::vector<GPUResult> gpuResults;
    gpuResults.reserve(replays);
    for (std::uint32_t replay = 0u; replay < replays; ++replay) {
        gpuResults.push_back(runGPU(
            device,
            queue,
            pipelines,
            initial,
            iterations,
            strainSweeps
        ));
    }
    const InitialState strainInitial = makeStrainProbeState();
    const OracleResult strainOracle = runOracle(strainInitial, 0u, 1u);
    const GPUResult strainGPU = runGPU(
        device,
        queue,
        pipelines,
        strainInitial,
        0u,
        1u
    );
    const InitialState groundBendInitial = makeGroundBendProbeState();
    const OracleResult groundBendOracle = runOracle(
        groundBendInitial, 1u, 0u
    );
    const GPUResult groundBendGPU = runGPU(
        device,
        queue,
        pipelines,
        groundBendInitial,
        1u,
        0u
    );
    const GPUResult& gpu = gpuResults.front();
    bool deterministic = true;
    for (std::size_t replay = 1u; replay < gpuResults.size(); ++replay) {
        deterministic = deterministic &&
            gpu.failure == gpuResults[replay].failure &&
            bitwiseEqual(gpu.particles, gpuResults[replay].particles) &&
            bitwiseEqual(gpu.distances, gpuResults[replay].distances) &&
            bitwiseEqual(gpu.grips, gpuResults[replay].grips) &&
            bitwiseEqual(gpu.knots, gpuResults[replay].knots) &&
            bitwiseEqual(gpu.bends, gpuResults[replay].bends);
    }

    double maximumPositionError = 0.0;
    double maximumVelocityError = 0.0;
    double maximumDistanceLambdaError = 0.0;
    double maximumGripLambdaError = 0.0;
    double maximumKnotLambdaError = 0.0;
    double maximumBendLambdaError = 0.0;
    double maximumKnotLambda = 0.0;
    double maximumBendLambda = 0.0;
    double maximumDisplacement = 0.0;
    for (std::size_t index = 0u; index < gpu.particles.size(); ++index) {
        const DVec3 gpuPosition = d3(
            gpu.particles[index].positionAndInverseMass
        );
        const DVec3 gpuVelocity = d3(gpu.particles[index].velocity);
        const DVec3 positionDelta = gpuPosition - oracle.particles[index].position;
        const DVec3 velocityDelta = gpuVelocity - oracle.particles[index].velocity;
        maximumPositionError = std::max(
            maximumPositionError,
            std::max({
                std::abs(positionDelta.x),
                std::abs(positionDelta.y),
                std::abs(positionDelta.z),
            })
        );
        maximumVelocityError = std::max(
            maximumVelocityError,
            std::max({
                std::abs(velocityDelta.x),
                std::abs(velocityDelta.y),
                std::abs(velocityDelta.z),
            })
        );
        maximumDisplacement = std::max(
            maximumDisplacement,
            length(gpuPosition - d3(
                initial.particles[index].positionAndInverseMass
            ))
        );
    }
    for (std::size_t index = 0u; index < gpu.distances.size(); ++index) {
        maximumDistanceLambdaError = std::max(
            maximumDistanceLambdaError,
            std::abs(
                static_cast<double>(gpu.distances[index].material.z) -
                oracle.distances[index].lambda
            )
        );
    }
    for (std::size_t index = 0u; index < gpu.knots.size(); ++index) {
        const double gpuLambda = gpu.knots[index].material.z;
        maximumKnotLambdaError = std::max(
            maximumKnotLambdaError,
            std::abs(gpuLambda - oracle.knots[index].lambda)
        );
        maximumKnotLambda = std::max(maximumKnotLambda, std::abs(gpuLambda));
    }
    for (std::size_t index = 0u; index < gpu.bends.size(); ++index) {
        const double gpuLambda = gpu.bends[index].material.w;
        maximumBendLambdaError = std::max(
            maximumBendLambdaError,
            std::abs(gpuLambda - oracle.bends[index].lambda)
        );
        maximumBendLambda = std::max(maximumBendLambda, std::abs(gpuLambda));
    }
    double gripForce = 0.0;
    for (std::size_t index = 0u; index < gpu.grips.size(); ++index) {
        const DVec3 gpuLambda = d3(gpu.grips[index].lambda);
        const DVec3 lambdaDelta = gpuLambda - oracle.grips[index].lambda;
        maximumGripLambdaError = std::max(
            maximumGripLambdaError,
            std::max({
                std::abs(lambdaDelta.x),
                std::abs(lambdaDelta.y),
                std::abs(lambdaDelta.z),
            })
        );
        gripForce += length(gpuLambda) /
            (static_cast<double>(kTimestep) * kTimestep);
    }
    const double strainViolation = maximumStrainViolation(
        gpu.particles,
        gpu.distances
    );
    const double probeInitialViolation = maximumStrainViolation(
        strainInitial.particles,
        strainInitial.distances
    );
    const double probeFinalViolation = maximumStrainViolation(
        strainGPU.particles,
        strainGPU.distances
    );
    double probePositionError = 0.0;
    for (std::size_t index = 0u;
         index < strainGPU.particles.size();
         ++index) {
        const DVec3 delta =
            d3(strainGPU.particles[index].positionAndInverseMass) -
            strainOracle.particles[index].position;
        probePositionError = std::max(
            probePositionError,
            std::max({std::abs(delta.x), std::abs(delta.y), std::abs(delta.z)})
        );
    }
    const double compressedInitialLength = length(
        d3(strainInitial.particles[3].positionAndInverseMass) -
        d3(strainInitial.particles[2].positionAndInverseMass)
    );
    const double compressedFinalLength = length(
        d3(strainGPU.particles[3].positionAndInverseMass) -
        d3(strainGPU.particles[2].positionAndInverseMass)
    );
    const auto pairCenterOfMass = [](const auto& particles) {
        const double firstMass = 1.0 /
            particles[0].positionAndInverseMass.w;
        const double secondMass = 1.0 /
            particles[1].positionAndInverseMass.w;
        return (
            d3(particles[0].positionAndInverseMass) * firstMass +
            d3(particles[1].positionAndInverseMass) * secondMass
        ) / (firstMass + secondMass);
    };
    const double probeCenterOfMassError = length(
        pairCenterOfMass(strainGPU.particles) -
        pairCenterOfMass(strainInitial.particles)
    );
    double groundBendPositionError = 0.0;
    for (std::size_t index = 0u;
         index < groundBendGPU.particles.size();
         ++index) {
        const DVec3 delta =
            d3(groundBendGPU.particles[index].positionAndInverseMass) -
            groundBendOracle.particles[index].position;
        groundBendPositionError = std::max(
            groundBendPositionError,
            std::max({std::abs(delta.x), std::abs(delta.y), std::abs(delta.z)})
        );
    }
    const double supportedHeight =
        groundBendGPU.particles[0].positionAndInverseMass.z;
    const double freeEndpointRise =
        groundBendGPU.particles[2].positionAndInverseMass.z -
        groundBendInitial.particles[2].positionAndInverseMass.z;
    double averageSeconds = 0.0;
    for (const GPUResult& replay : gpuResults) {
        averageSeconds += replay.seconds;
    }
    averageSeconds /= static_cast<double>(gpuResults.size());

    const bool passed =
        initial.particles.size() == kParticleCount &&
        initial.distances.size() == kDistanceCount &&
        initial.grips.size() == kGripCount &&
        initial.knots.size() == kKnotCount &&
        initial.bends.size() == kBendCount &&
        coloringExact && gpu.failure == NUMI_CLOTH_BAG_GPU_FAILURE_NONE &&
        deterministic && maximumPositionError <= 2.0e-5 &&
        maximumVelocityError <= 0.12 &&
        maximumDistanceLambdaError <= 2.0e-8 &&
        maximumGripLambdaError <= 2.0e-8 &&
        maximumKnotLambdaError <= 2.0e-8 &&
        maximumBendLambdaError <= 2.0e-8 &&
        maximumKnotLambda > 1.0e-12 && maximumBendLambda > 1.0e-12 &&
        strainViolation <= 2.0e-6 &&
        maximumDisplacement > 1.0e-4 && gripForce > 1.0 &&
        strainGPU.failure == NUMI_CLOTH_BAG_GPU_FAILURE_NONE &&
        probeInitialViolation > 0.20 && probeFinalViolation <= 2.0e-7 &&
        probePositionError <= 2.0e-7 &&
        std::abs(compressedFinalLength - compressedInitialLength) <= 1.0e-7 &&
        probeCenterOfMassError <= 1.0e-7 &&
        groundBendGPU.failure == NUMI_CLOTH_BAG_GPU_FAILURE_NONE &&
        groundBendPositionError <= 2.0e-7 &&
        supportedHeight >= 0.004 - 1.0e-8 && freeEndpointRise > 0.0;

    std::cout << std::fixed << std::setprecision(12)
              << "device=" << device.name.UTF8String << '\n'
              << "abi=" << NUMI_CLOTH_BAG_GPU_ABI_VERSION
              << " particles=" << initial.particles.size()
              << " distances=" << initial.distances.size()
              << " grips=" << initial.grips.size()
              << " knots=" << initial.knots.size()
              << " bends=" << initial.bends.size() << '\n'
              << "distance_colors=" << initial.distanceBatches.size()
              << " knot_colors=" << initial.knotBatches.size()
              << " bend_colors=" << initial.bendBatches.size() << '\n'
              << "iterations=" << iterations
              << " strain_sweeps=" << strainSweeps
              << " replays=" << replays << '\n'
              << "coloring_exact=" << std::boolalpha << coloringExact
              << " failure_flags=" << gpu.failure
              << " deterministic=" << deterministic << '\n'
              << "max_position_error=" << maximumPositionError
              << " max_velocity_error=" << maximumVelocityError << '\n'
              << "max_distance_lambda_error="
              << maximumDistanceLambdaError
              << " max_grip_lambda_error=" << maximumGripLambdaError << '\n'
              << "max_knot_lambda_error=" << maximumKnotLambdaError
              << " max_bend_lambda_error=" << maximumBendLambdaError
              << " max_knot_lambda=" << maximumKnotLambda
              << " max_bend_lambda=" << maximumBendLambda << '\n'
              << "max_strain_violation=" << strainViolation
              << " max_displacement=" << maximumDisplacement
              << " grip_force=" << gripForce << '\n'
              << "strain_probe_initial_violation=" << probeInitialViolation
              << " strain_probe_final_violation=" << probeFinalViolation
              << " strain_probe_position_error=" << probePositionError
              << " compression_length_change="
              << compressedFinalLength - compressedInitialLength
              << " strain_probe_com_error=" << probeCenterOfMassError
              << " strain_probe_failure_flags=" << strainGPU.failure << '\n'
              << "ground_bend_position_error=" << groundBendPositionError
              << " supported_height=" << supportedHeight
              << " free_endpoint_rise=" << freeEndpointRise
              << " ground_bend_failure_flags=" << groundBendGPU.failure
              << '\n'
              << "average_gpu_seconds=" << averageSeconds
              << " state_hash=0x" << std::hex << hashGPUResult(gpu)
              << std::dec << '\n'
              << "result=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}

} // namespace

int main(const int argc, const char* const* argv) {
    @autoreleasepool {
        try {
            return run(argc, argv);
        } catch (const std::exception& exception) {
            std::cerr << "error: " << exception.what() << '\n';
            return 2;
        }
    }
}
