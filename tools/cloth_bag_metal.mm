#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/cloth_bag_gpu.h"
#include "numi/cloth_material.h"
#include "numi/grip_trajectory.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
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
constexpr std::uint32_t kFruitCount = 12u;
constexpr std::uint32_t kFruitPairCount =
    kFruitCount * (kFruitCount - 1u) / 2u;
constexpr std::uint32_t kFruitYarnCount = kFruitCount * kDistanceCount;
constexpr std::uint32_t kReconciliationPasses = 8u;
constexpr std::uint32_t kFinalContactPasses = 2u;
constexpr std::uint32_t kCertificatePasses = 8u;
numi::ClothMaterialArtifact gClothMaterial{};
float kFruitPairFriction = static_cast<float>(
    gClothMaterial.values.fruitPairFriction
);
float kFruitGroundFriction = static_cast<float>(
    gClothMaterial.values.fruitGroundFriction
);
float kFruitRollingResistance = static_cast<float>(
    gClothMaterial.values.fruitRollingResistance
);
float kFruitYarnFriction = static_cast<float>(
    gClothMaterial.values.fruitYarnFriction
);
float kClothGroundFriction = static_cast<float>(
    gClothMaterial.values.clothGroundFriction
);
float kClothSelfFriction = static_cast<float>(
    gClothMaterial.values.clothSelfFriction
);
constexpr float kAirDensity = 1.225f;
float kYarnCrossflowDrag = static_cast<float>(
    gClothMaterial.values.yarnCrossflowDrag
);
float kYarnSkinFriction = static_cast<float>(
    gClothMaterial.values.yarnSkinFriction
);
float kFruitDrag = static_cast<float>(gClothMaterial.values.fruitDrag);
float kFruitRotationalDrag = static_cast<float>(
    gClothMaterial.values.fruitRotationalDrag
);
constexpr std::uint32_t kPickupSubstepsPerFrame = 48u;
constexpr std::uint32_t kGroundedQualificationFrames = 120u;
constexpr std::uint32_t kSpinQualificationFrames = 60u;
constexpr std::uint32_t kPickupMotionFrames = 240u;
constexpr std::uint32_t kPickupQualificationFrames =
    kPickupMotionFrames + 240u;
constexpr double kSpinAirborneLift = 0.52;

enum class TrajectoryScenario : std::uint8_t {
    grounded,
    spin,
    pickup,
    recorded,
};

std::size_t packedSelfPairIndex(
    const std::uint32_t first,
    const std::uint32_t second,
    const std::uint32_t segmentCount
) {
    const std::uint32_t lower = std::min(first, second);
    const std::uint32_t upper = std::max(first, second);
    if (lower == upper || upper >= segmentCount) {
        throw std::logic_error("invalid packed yarn-pair lookup index");
    }
    return static_cast<std::size_t>(lower) *
            (2u * segmentCount - lower - 1u) / 2u +
        (upper - lower - 1u);
}
float kOrdinaryMass = static_cast<float>(
    gClothMaterial.values.ordinaryNodeMassKg
);
float kHemMass = static_cast<float>(gClothMaterial.values.hemNodeMassKg);
float kClothRadius = static_cast<float>(gClothMaterial.values.yarnRadiusM);
float kAxialBodyCompliance = static_cast<float>(
    gClothMaterial.values.axialBodyComplianceMPerN
);
float kAxialCuffCompliance = static_cast<float>(
    gClothMaterial.values.axialCuffComplianceMPerN
);
float kKnotCompliance = static_cast<float>(
    gClothMaterial.values.knotCompliance
);
float kBendBodyCompliance = static_cast<float>(
    gClothMaterial.values.bendBodyComplianceMPerN
);
float kBendCuffCompliance = static_cast<float>(
    gClothMaterial.values.bendCuffComplianceMPerN
);
float kGripCompliance = static_cast<float>(
    gClothMaterial.values.gripComplianceMPerN
);
constexpr float kGripCaptureRadius = 0.12f;
constexpr float kStrainLimit = 0.285f;
constexpr float kTimestep = 1.0f / 5760.0f;

void applyClothMaterial(const numi::ClothMaterialArtifact& artifact) {
    gClothMaterial = artifact;
    kOrdinaryMass = static_cast<float>(artifact.values.ordinaryNodeMassKg);
    kHemMass = static_cast<float>(artifact.values.hemNodeMassKg);
    kClothRadius = static_cast<float>(artifact.values.yarnRadiusM);
    kAxialBodyCompliance = static_cast<float>(
        artifact.values.axialBodyComplianceMPerN
    );
    kAxialCuffCompliance = static_cast<float>(
        artifact.values.axialCuffComplianceMPerN
    );
    kKnotCompliance = static_cast<float>(artifact.values.knotCompliance);
    kBendBodyCompliance = static_cast<float>(
        artifact.values.bendBodyComplianceMPerN
    );
    kBendCuffCompliance = static_cast<float>(
        artifact.values.bendCuffComplianceMPerN
    );
    kGripCompliance = static_cast<float>(
        artifact.values.gripComplianceMPerN
    );
    kClothGroundFriction = static_cast<float>(
        artifact.values.clothGroundFriction
    );
    kClothSelfFriction = static_cast<float>(
        artifact.values.clothSelfFriction
    );
    kFruitYarnFriction = static_cast<float>(
        artifact.values.fruitYarnFriction
    );
    kFruitGroundFriction = static_cast<float>(
        artifact.values.fruitGroundFriction
    );
    kFruitPairFriction = static_cast<float>(
        artifact.values.fruitPairFriction
    );
    kFruitRollingResistance = static_cast<float>(
        artifact.values.fruitRollingResistance
    );
    kYarnCrossflowDrag = static_cast<float>(
        artifact.values.yarnCrossflowDrag
    );
    kYarnSkinFriction = static_cast<float>(
        artifact.values.yarnSkinFriction
    );
    kFruitDrag = static_cast<float>(artifact.values.fruitDrag);
    kFruitRotationalDrag = static_cast<float>(
        artifact.values.fruitRotationalDrag
    );
}

struct DVec3 {
    double x{};
    double y{};
    double z{};
};

struct TrajectoryGripPose {
    DVec3 target{};
    numi::GripTrajectoryQuaternion orientation{};
    bool active{};
    std::uint32_t attachmentGeneration{1u};
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

DVec3 cross(const DVec3 a, const DVec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

double length(const DVec3 value) {
    return std::sqrt(dot(value, value));
}

DVec3 normalized(const DVec3 value) {
    const double magnitude = length(value);
    return magnitude > 1.0e-14
        ? value / magnitude
        : DVec3{1.0, 0.0, 0.0};
}

mr_float4 f4(
    const float x,
    const float y,
    const float z,
    const float w = 0.0f
) {
    return {x, y, z, w};
}

mr_float4 gripQuaternion4(
    const numi::GripTrajectoryQuaternion quaternion
) {
    return f4(
        static_cast<float>(quaternion.x),
        static_cast<float>(quaternion.y),
        static_cast<float>(quaternion.z),
        static_cast<float>(quaternion.w)
    );
}

DVec3 rotateGripOffset(
    const mr_float4 quaternion,
    const DVec3 offset
) {
    const numi::GripTrajectoryVector3 rotated = numi::rotateGripVector(
        {quaternion.x, quaternion.y, quaternion.z, quaternion.w},
        {offset.x, offset.y, offset.z}
    );
    return {rotated.x, rotated.y, rotated.z};
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

DVec3 pickupGripTarget(const double time) {
    const DVec3 base = authoredPosition(kLevels - 1u, 0u);
    const double lift = smoothstep(time / 0.80);
    const double firstSnap = smoothstep((time - 1.00) / 0.23);
    const double firstRecovery = smoothstep((time - 1.45) / 0.50);
    return base + DVec3{
        -0.10 * firstSnap,
        0.04 * std::sin(std::numbers::pi * firstSnap),
        1.25 * lift - 0.85 * firstSnap + 0.75 * firstRecovery,
    };
}

DVec3 spinGripTarget(const double time) {
    DVec3 base = authoredPosition(kLevels - 1u, 0u);
    base.z += kSpinAirborneLift;
    constexpr double radius = 0.28;
    constexpr double angularSpeed = 4.8;
    constexpr double rampTime = 0.18;
    const double angle = angularSpeed * (
        time - rampTime * (1.0 - std::exp(-time / rampTime))
    );
    return base + DVec3{
        radius * (std::cos(angle) - 1.0),
        radius * std::sin(angle),
        0.035 * std::sin(0.5 * angle),
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
    std::vector<NumiClothBagGPUFruit> fruits;
    std::vector<NumiClothBagGPUFruitPair> fruitPairs;
    std::vector<NumiClothBagGPUYarnContact> yarnContacts;
    std::vector<NumiClothBagGPUSelfPair> selfPairs;
    std::vector<std::uint32_t> selfPairLookup;
    std::vector<std::uint32_t> mouthRimParticles;
    std::vector<std::uint32_t> cuffParticles;
    std::vector<NumiClothBagGPUBatch> distanceBatches;
    std::vector<NumiClothBagGPUBatch> knotBatches;
    std::vector<NumiClothBagGPUBatch> bendBatches;
    std::vector<NumiClothBagGPUBatch> fruitPairBatches;
    std::vector<NumiClothBagGPUBatch> selfBatches;
    std::uint32_t maximumSelfBatchSize{};
    std::uint32_t initialMouthCandidateMask{};
    std::uint32_t initialReleasedMask{};
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

struct FruitPairSpec {
    std::uint32_t first{};
    std::uint32_t second{};
    std::uint32_t stableIndex{};
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
                level + 2u >= kLevels
                    ? kAxialCuffCompliance
                    : kAxialBodyCompliance,
                1u
            );
        }
    }
    for (std::uint32_t level = 0u; level + 1u < kLevels; ++level) {
        for (std::uint32_t ring = 0u; ring < kAround; ++ring) {
            addEdge(
                nodeIndex(level, ring),
                nodeIndex(level + 1u, ring),
                kAxialBodyCompliance,
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
                    kAxialBodyCompliance,
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
                    kAxialBodyCompliance,
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
                kKnotCompliance,
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
                level + 2u >= kLevels
                    ? kBendCuffCompliance
                    : kBendBodyCompliance
            );
        }
    }
    for (std::uint32_t level = 1u; level + 1u < kLevels; ++level) {
        for (std::uint32_t ring = 0u; ring < kAround; ++ring) {
            addBend(
                nodeIndex(level - 1u, ring),
                nodeIndex(level, ring),
                nodeIndex(level + 1u, ring),
                kBendBodyCompliance
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
                kBendBodyCompliance
            );
            addBend(
                bottomGridIndex(column - 1u, row),
                bottomGridIndex(column, row),
                bottomGridIndex(column + 1u, row),
                kBendBodyCompliance
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
        kKnotCount, kBendCount, 0u, kFruitCount
    );
    result.config.contactCounts = u4(
        kFruitPairCount, kFruitYarnCount, 0u, 0u
    );
    result.config.gravityAndTimestep = f4(0.0f, 0.0f, -9.81f, kTimestep);
    result.config.gripTargetAndActive = f4(
        static_cast<float>(base.x + 0.002),
        static_cast<float>(base.y - 0.001),
        static_cast<float>(base.z + 0.006),
        1.0f
    );
    result.config.gripOrientation = f4(0.0f, 0.0f, 0.0f, 1.0f);
    result.config.gripControl = u4(1u, 0u, kAround, 2u);
    result.config.gripMaterial = f4(kGripCaptureRadius, 0.0f, 0.0f, 0.0f);
    float maximumLimitedYarnLength = 0.0f;
    for (const NumiClothBagGPUDistance& distance : result.distances) {
        maximumLimitedYarnLength = std::max(
            maximumLimitedYarnLength,
            distance.material.x * (1.0f + distance.material.w)
        );
    }
    result.config.clothMaterial = f4(
        kClothRadius,
        std::max(0.1f, maximumLimitedYarnLength + 0.008001f),
        kClothGroundFriction,
        kClothSelfFriction
    );
    result.config.fruitMaterial = f4(
        kFruitPairFriction,
        kFruitGroundFriction,
        kFruitRollingResistance,
        kFruitYarnFriction
    );
    result.config.airVelocityAndDensity = f4(
        0.0f, 0.0f, 0.0f, kAirDensity
    );
    result.config.aerodynamicCoefficients = f4(
        kYarnCrossflowDrag,
        kYarnSkinFriction,
        kFruitDrag,
        kFruitRotationalDrag
    );
    result.config.mouthControl = u4(
        kAround,
        bottomGridIndex(kBottomGrid / 2u, kBottomGrid / 2u),
        0u,
        0u
    );
    result.config.mouthMaterial = f4(0.025f, 0.0f, 0.0f, 0.0f);
    result.mouthRimParticles.reserve(kAround);
    result.cuffParticles.reserve(2u * kAround);
    for (std::uint32_t ring = 0u; ring < kAround; ++ring) {
        result.mouthRimParticles.push_back(nodeIndex(kLevels - 1u, ring));
    }
    for (std::uint32_t level = kLevels - 2u;
         level < kLevels;
         ++level) {
        for (std::uint32_t ring = 0u; ring < kAround; ++ring) {
            result.cuffParticles.push_back(nodeIndex(level, ring));
        }
    }
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
            grip.particle = u4(particleIndex, 1u, 0u, 0u);
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

    constexpr std::array<DVec3, kFruitCount> fruitPositions{{
        {0.190, 0.000, 0.088},
        {0.118, 0.149, 0.083},
        {-0.042, 0.185, 0.093},
        {-0.171, 0.082, 0.078},
        {-0.171, -0.082, 0.086},
        {-0.042, -0.185, 0.090},
        {0.118, -0.149, 0.081},
        {0.000, 0.000, 0.084},
        {0.105, 0.000, 0.210},
        {0.000, 0.105, 0.210},
        {-0.105, 0.000, 0.215},
        {0.000, -0.105, 0.210},
    }};
    constexpr std::array<float, kFruitCount> fruitRadii{{
        0.070f, 0.065f, 0.075f, 0.060f, 0.068f, 0.072f,
        0.063f, 0.066f, 0.072f, 0.064f, 0.070f, 0.067f,
    }};
    constexpr std::array<float, kFruitCount> fruitMasses{{
        0.21f, 0.17f, 0.26f, 0.14f, 0.19f, 0.23f,
        0.16f, 0.18f, 0.23f, 0.17f, 0.21f, 0.18f,
    }};
    constexpr std::array<std::uint32_t, kFruitCount> appearances{{
        0u, 1u, 2u, 1u, 3u, 0u,
        2u, 3u, 0u, 1u, 2u, 3u,
    }};
    result.fruits.reserve(kFruitCount);
    for (std::size_t index = 0u; index < fruitPositions.size(); ++index) {
        const DVec3 position = fruitPositions[index] + DVec3{0.0, 0.0, 0.009};
        NumiClothBagGPUFruit fruit{};
        fruit.positionAndInverseMass = f4(
            static_cast<float>(position.x),
            static_cast<float>(position.y),
            static_cast<float>(position.z),
            1.0f / fruitMasses[index]
        );
        fruit.previousAndRadius = f4(
            static_cast<float>(position.x),
            static_cast<float>(position.y),
            static_cast<float>(position.z),
            fruitRadii[index]
        );
        fruit.velocityAndGroundImpulse = f4(0.0f, 0.0f, 0.0f, 0.0f);
        fruit.angularVelocity = f4(0.0f, 0.0f, 0.0f, 0.0f);
        fruit.orientation = f4(0.0f, 0.0f, 0.0f, 1.0f);
        fruit.identity = u4(appearances[index], 0u, 0u, 0u);
        result.fruits.push_back(fruit);
    }

    std::vector<FruitPairSpec> pairSpecs;
    pairSpecs.reserve(kFruitPairCount);
    std::uint32_t stablePair = 0u;
    for (std::uint32_t first = 0u; first < kFruitCount; ++first) {
        for (std::uint32_t second = first + 1u;
             second < kFruitCount;
             ++second, ++stablePair) {
            pairSpecs.push_back({first, second, stablePair, 0u});
        }
    }
    particleColors.assign(kFruitCount, 0u);
    std::uint32_t pairColorCount = 0u;
    for (FruitPairSpec& pair : pairSpecs) {
        const std::uint64_t unavailable =
            particleColors[pair.first] | particleColors[pair.second];
        const std::uint64_t available = ~unavailable;
        if (available == 0u) {
            throw std::logic_error("fruit pair coloring exceeds 64 colors");
        }
        pair.color = static_cast<std::uint32_t>(
            std::countr_zero(available)
        );
        pairColorCount = std::max(pairColorCount, pair.color + 1u);
        const std::uint64_t bit = std::uint64_t{1u} << pair.color;
        particleColors[pair.first] |= bit;
        particleColors[pair.second] |= bit;
    }
    std::stable_sort(
        pairSpecs.begin(),
        pairSpecs.end(),
        [](const FruitPairSpec& first, const FruitPairSpec& second) {
            return first.color < second.color;
        }
    );
    result.fruitPairs.reserve(pairSpecs.size());
    batchStart = 0u;
    for (std::uint32_t color = 0u; color < pairColorCount; ++color) {
        const std::uint32_t first = batchStart;
        while (batchStart < pairSpecs.size() &&
               pairSpecs[batchStart].color == color) {
            const FruitPairSpec& source = pairSpecs[batchStart];
            NumiClothBagGPUFruitPair pair{};
            pair.fruitsAndColor = u4(
                source.first,
                source.second,
                source.color,
                source.stableIndex
            );
            pair.contact = f4(0.0f, 0.0f, 0.0f, 0.0f);
            result.fruitPairs.push_back(pair);
            ++batchStart;
        }
        result.fruitPairBatches.push_back({
            u4(first, batchStart - first, color, 0u)
        });
    }
    result.yarnContacts.reserve(kFruitYarnCount);
    for (std::uint32_t fruit = 0u; fruit < kFruitCount; ++fruit) {
        for (std::uint32_t segment = 0u;
             segment < kDistanceCount;
             ++segment) {
            const NumiClothBagGPUDistance& yarn = result.distances[segment];
            NumiClothBagGPUYarnContact contact{};
            contact.identity = u4(
                fruit,
                segment,
                yarn.particlesAndColor.x,
                yarn.particlesAndColor.y
            );
            result.yarnContacts.push_back(contact);
        }
    }

    std::vector<std::vector<std::uint32_t>> directTopology(kParticleCount);
    for (const NumiClothBagGPUDistance& distance : result.distances) {
        directTopology[distance.particlesAndColor.x].push_back(
            distance.particlesAndColor.y
        );
        directTopology[distance.particlesAndColor.y].push_back(
            distance.particlesAndColor.x
        );
    }
    std::vector<std::vector<std::uint32_t>> localTopology(kParticleCount);
    for (std::uint32_t particle = 0u;
         particle < kParticleCount;
         ++particle) {
        std::vector<std::uint32_t>& local = localTopology[particle];
        local.push_back(particle);
        for (const std::uint32_t firstHop : directTopology[particle]) {
            local.push_back(firstHop);
            for (const std::uint32_t secondHop : directTopology[firstHop]) {
                local.push_back(secondHop);
            }
        }
        std::sort(local.begin(), local.end());
        local.erase(std::unique(local.begin(), local.end()), local.end());
    }
    const auto localPair = [&localTopology](
        const std::uint32_t first,
        const std::uint32_t second
    ) {
        const std::vector<std::uint32_t>& local = localTopology[first];
        return std::binary_search(local.begin(), local.end(), second);
    };
    const auto localEdgePair = [&result, &localPair](
        const std::uint32_t firstIndex,
        const std::uint32_t secondIndex
    ) {
        const mr_uint4 first =
            result.distances[firstIndex].particlesAndColor;
        const mr_uint4 second =
            result.distances[secondIndex].particlesAndColor;
        return localPair(first.x, second.x) ||
            localPair(first.x, second.y) ||
            localPair(first.y, second.x) ||
            localPair(first.y, second.y);
    };
    struct PhaseSelfPair {
        NumiClothBagGPUSelfPair pair{};
        std::uint32_t firstSegment{};
        std::uint32_t secondSegment{};
        std::uint32_t color{};
    };
    static_assert(kDistanceCount % 2u == 0u);
    const std::uint32_t rotatingCount = kDistanceCount - 1u;
    const std::uint32_t fixedSegment = kDistanceCount - 1u;
    result.selfPairs.reserve(
        static_cast<std::size_t>(kDistanceCount) *
        static_cast<std::size_t>(kDistanceCount - 1u) / 2u
    );
    result.selfPairLookup.assign(
        static_cast<std::size_t>(kDistanceCount) *
            (kDistanceCount - 1u) / 2u,
        std::numeric_limits<std::uint32_t>::max()
    );
    std::vector<std::uint64_t> particleColorMasks(kParticleCount, 0u);
    std::vector<PhaseSelfPair> phasePairs;
    phasePairs.reserve(kDistanceCount / 2u);
    for (std::uint32_t phase = 0u; phase < rotatingCount; ++phase) {
        std::fill(
            particleColorMasks.begin(), particleColorMasks.end(), 0u
        );
        phasePairs.clear();
        for (std::uint32_t slot = 0u;
             slot < kDistanceCount / 2u;
             ++slot) {
            std::uint32_t firstIndex{};
            std::uint32_t secondIndex{};
            if (slot == 0u) {
                firstIndex = fixedSegment;
                secondIndex = phase;
            } else {
                firstIndex = (phase + slot) % rotatingCount;
                secondIndex = (
                    phase + rotatingCount - slot
                ) % rotatingCount;
            }
            if (localEdgePair(firstIndex, secondIndex)) {
                continue;
            }
            const mr_uint4 first =
                result.distances[firstIndex].particlesAndColor;
            const mr_uint4 second =
                result.distances[secondIndex].particlesAndColor;
            const std::uint64_t unavailable =
                particleColorMasks[first.x] |
                particleColorMasks[first.y] |
                particleColorMasks[second.x] |
                particleColorMasks[second.y];
            const std::uint64_t available = ~unavailable;
            if (available == 0u) {
                throw std::logic_error(
                    "yarn self-contact phase exceeds 64 subcolors"
                );
            }
            const std::uint32_t color = static_cast<std::uint32_t>(
                std::countr_zero(available)
            );
            const std::uint64_t colorBit = std::uint64_t{1u} << color;
            particleColorMasks[first.x] |= colorBit;
            particleColorMasks[first.y] |= colorBit;
            particleColorMasks[second.x] |= colorBit;
            particleColorMasks[second.y] |= colorBit;
            phasePairs.push_back({
                {firstIndex, secondIndex},
                firstIndex,
                secondIndex,
                color
            });
        }
        std::stable_sort(
            phasePairs.begin(),
            phasePairs.end(),
            [](const PhaseSelfPair& first, const PhaseSelfPair& second) {
                return first.color < second.color;
            }
        );
        std::size_t phaseIndex = 0u;
        while (phaseIndex < phasePairs.size()) {
            const std::uint32_t color = phasePairs[phaseIndex].color;
            while (phaseIndex < phasePairs.size() &&
                   phasePairs[phaseIndex].color == color) {
                const std::uint32_t start = static_cast<std::uint32_t>(
                    result.selfPairs.size()
                );
                std::uint32_t count = 0u;
                while (phaseIndex < phasePairs.size() &&
                       phasePairs[phaseIndex].color == color &&
                       count < 256u) {
                    const std::uint32_t pairIndex =
                        static_cast<std::uint32_t>(result.selfPairs.size());
                    const PhaseSelfPair& source = phasePairs[phaseIndex];
                    result.selfPairs.push_back(phasePairs[phaseIndex].pair);
                    result.selfPairLookup[packedSelfPairIndex(
                        source.firstSegment,
                        source.secondSegment,
                        kDistanceCount
                    )] = pairIndex;
                    ++phaseIndex;
                    ++count;
                }
                result.selfBatches.push_back({
                    u4(start, count, phase, color)
                });
                result.maximumSelfBatchSize = std::max(
                    result.maximumSelfBatchSize, count
                );
            }
        }
    }
    result.config.contactCounts.z = static_cast<std::uint32_t>(
        result.selfPairs.size()
    );
    result.config.contactCounts.w = static_cast<std::uint32_t>(
        result.selfBatches.size()
    );
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
    result.config.clothMaterial = f4(
        kClothRadius, 0.0f, kClothGroundFriction, kClothSelfFriction
    );
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
    result.config.clothMaterial = f4(
        kClothRadius, 0.0f, kClothGroundFriction, kClothSelfFriction
    );
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

NumiClothBagGPUFruit makeProbeFruit(
    const DVec3 position,
    const float inverseMass,
    const float radius
) {
    NumiClothBagGPUFruit fruit{};
    fruit.positionAndInverseMass = f4(
        static_cast<float>(position.x),
        static_cast<float>(position.y),
        static_cast<float>(position.z),
        inverseMass
    );
    fruit.previousAndRadius = f4(
        static_cast<float>(position.x),
        static_cast<float>(position.y),
        static_cast<float>(position.z),
        radius
    );
    fruit.velocityAndGroundImpulse = f4(0.0f, 0.0f, 0.0f, 0.0f);
    fruit.angularVelocity = f4(0.0f, 0.0f, 0.0f, 0.0f);
    fruit.orientation = f4(0.0f, 0.0f, 0.0f, 1.0f);
    fruit.identity = u4(0u, 0u, 0u, 0u);
    return fruit;
}

InitialState makeFruitPairProbeState() {
    InitialState result;
    result.config.control = u4(
        NUMI_CLOTH_BAG_GPU_ABI_VERSION, 0u, 0u, 0u
    );
    result.config.constraintCounts = u4(0u, 0u, 0u, 2u);
    result.config.contactCounts = u4(1u, 0u, 0u, 0u);
    result.config.gravityAndTimestep = f4(0.0f, 0.0f, 0.0f, 0.01f);
    result.config.gripTargetAndActive = f4(0.0f, 0.0f, 0.0f, 0.0f);
    result.config.clothMaterial = f4(
        kClothRadius, 0.0f, kClothGroundFriction, kClothSelfFriction
    );
    result.config.fruitMaterial = f4(
        kFruitPairFriction,
        kFruitGroundFriction,
        kFruitRollingResistance,
        kFruitYarnFriction
    );
    result.fruits = {
        makeProbeFruit({0.0, 0.0, 2.0}, 1.0f, 1.0f),
        makeProbeFruit({1.5, 0.0, 2.0}, 2.0f, 1.0f),
    };
    result.fruits[0].velocityAndGroundImpulse.y = 1.0f;
    result.fruits[1].velocityAndGroundImpulse.y = -2.0f;
    NumiClothBagGPUFruitPair pair{};
    pair.fruitsAndColor = u4(0u, 1u, 0u, 0u);
    pair.contact = f4(0.0f, 0.0f, 0.0f, 0.0f);
    result.fruitPairs = {pair};
    result.fruitPairBatches = {{u4(0u, 1u, 0u, 0u)}};
    return result;
}

InitialState makeGroundContactProbeState() {
    InitialState result;
    result.config.control = u4(
        NUMI_CLOTH_BAG_GPU_ABI_VERSION, 1u, 0u, 0u
    );
    result.config.constraintCounts = u4(0u, 0u, 1u, 1u);
    result.config.contactCounts = u4(0u, 0u, 0u, 0u);
    result.config.gravityAndTimestep = f4(0.0f, 0.0f, 0.0f, 0.01f);
    result.config.gripTargetAndActive = f4(0.0f, 0.0f, 0.0f, 0.0f);
    result.config.clothMaterial = f4(
        kClothRadius, 0.0f, kClothGroundFriction, kClothSelfFriction
    );
    result.config.fruitMaterial = f4(
        kFruitPairFriction,
        kFruitGroundFriction,
        kFruitRollingResistance,
        kFruitYarnFriction
    );
    NumiClothBagGPUParticle particle{};
    particle.positionAndInverseMass = f4(0.0f, 0.0f, 0.0f, 1.0f);
    particle.previousAndMass = f4(0.0f, 0.0f, 0.0f, 1.0f);
    particle.velocity = f4(1.0f, 0.0f, 0.0f, 0.0f);
    result.particles = {particle};
    result.fruits = {
        makeProbeFruit({0.0, 0.0, 0.5}, 2.0f, 1.0f)
    };
    result.fruits[0].velocityAndGroundImpulse.x = 3.0f;
    result.fruits[0].angularVelocity.x = 2.0f;
    return result;
}

InitialState makeYarnCCDProbeState() {
    InitialState result;
    result.config.control = u4(
        NUMI_CLOTH_BAG_GPU_ABI_VERSION, 2u, 1u, 0u
    );
    result.config.constraintCounts = u4(0u, 0u, 0u, 1u);
    result.config.contactCounts = u4(0u, 1u, 0u, 0u);
    result.config.gravityAndTimestep = f4(0.0f, 0.0f, 0.0f, 0.01f);
    result.config.gripTargetAndActive = f4(0.0f, 0.0f, 0.0f, 0.0f);
    result.config.clothMaterial = f4(
        kClothRadius, 0.0f, kClothGroundFriction, kClothSelfFriction
    );
    result.config.fruitMaterial = f4(
        kFruitPairFriction,
        kFruitGroundFriction,
        kFruitRollingResistance,
        kFruitYarnFriction
    );
    const auto fixedParticle = [](const DVec3 position) {
        NumiClothBagGPUParticle value{};
        value.positionAndInverseMass = f4(
            static_cast<float>(position.x),
            static_cast<float>(position.y),
            static_cast<float>(position.z),
            0.0f
        );
        value.previousAndMass = f4(
            static_cast<float>(position.x),
            static_cast<float>(position.y),
            static_cast<float>(position.z),
            0.0f
        );
        value.velocity = f4(0.0f, 0.0f, 0.0f, 0.0f);
        return value;
    };
    result.particles = {
        fixedParticle({0.0, -0.1, 1.0}),
        fixedParticle({0.0, 0.1, 1.0}),
    };
    NumiClothBagGPUDistance distance{};
    distance.particlesAndColor = u4(0u, 1u, 0u, 0u);
    distance.material = f4(0.2f, 0.0f, 0.0f, kStrainLimit);
    result.distances = {distance};
    result.distanceBatches = {{u4(0u, 1u, 0u, 0u)}};
    NumiClothBagGPUFruit fruit = makeProbeFruit(
        {-0.1, 0.0, 1.0}, 1.0f, 0.02f
    );
    fruit.velocityAndGroundImpulse = f4(20.0f, 2.0f, 0.0f, 0.0f);
    result.fruits = {fruit};
    NumiClothBagGPUYarnContact contact{};
    contact.identity = u4(0u, 0u, 0u, 1u);
    result.yarnContacts = {contact};
    return result;
}

InitialState makeSelfCCDProbeState() {
    InitialState result;
    result.config.control = u4(
        NUMI_CLOTH_BAG_GPU_ABI_VERSION, 4u, 2u, 0u
    );
    result.config.constraintCounts = u4(0u, 0u, 0u, 0u);
    result.config.contactCounts = u4(0u, 0u, 1u, 1u);
    result.config.gravityAndTimestep = f4(0.0f, 0.0f, 0.0f, 0.01f);
    result.config.gripTargetAndActive = f4(0.0f, 0.0f, 0.0f, 0.0f);
    result.config.clothMaterial = f4(
        kClothRadius,
        2.0f + 2.0f * kClothRadius + 1.0e-6f,
        kClothGroundFriction,
        kClothSelfFriction
    );
    const auto particle = [](
        const DVec3 position,
        const DVec3 velocity,
        const float inverseMass
    ) {
        NumiClothBagGPUParticle value{};
        value.positionAndInverseMass = f4(
            static_cast<float>(position.x),
            static_cast<float>(position.y),
            static_cast<float>(position.z),
            inverseMass
        );
        value.previousAndMass = f4(
            static_cast<float>(position.x),
            static_cast<float>(position.y),
            static_cast<float>(position.z),
            inverseMass > 0.0f ? 1.0f / inverseMass : 0.0f
        );
        value.velocity = f4(
            static_cast<float>(velocity.x),
            static_cast<float>(velocity.y),
            static_cast<float>(velocity.z),
            0.0f
        );
        return value;
    };
    result.particles = {
        particle({-1.0, 0.0, 0.0}, {}, 1.0f),
        particle({1.0, 0.0, 0.0}, {}, 1.0f),
        particle({0.0, -1.0, 0.08}, {2.0, 0.0, -16.0}, 1.0f),
        particle({0.0, 1.0, 0.08}, {2.0, 0.0, -16.0}, 1.0f),
    };
    NumiClothBagGPUDistance firstDistance{};
    firstDistance.particlesAndColor = u4(0u, 1u, 0u, 0u);
    NumiClothBagGPUDistance secondDistance{};
    secondDistance.particlesAndColor = u4(2u, 3u, 0u, 0u);
    result.distances = {firstDistance, secondDistance};
    result.selfPairs = {{0u, 1u}};
    result.selfPairLookup = {0u};
    result.selfBatches = {{u4(0u, 1u, 0u, 0u)}};
    result.maximumSelfBatchSize = 1u;
    return result;
}

InitialState makeYarnAerodynamicsProbeState(
    const bool refined,
    const bool coMovingAir
) {
    InitialState result;
    const std::uint32_t particleCount = refined ? 3u : 2u;
    const std::uint32_t distanceCount = refined ? 2u : 1u;
    result.config.control = u4(
        NUMI_CLOTH_BAG_GPU_ABI_VERSION,
        particleCount,
        distanceCount,
        0u
    );
    result.config.gravityAndTimestep = f4(0.0f, 0.0f, 0.0f, 0.01f);
    result.config.gripTargetAndActive = f4(0.0f, 0.0f, 0.0f, 0.0f);
    result.config.clothMaterial = f4(
        kClothRadius, 0.0f, kClothGroundFriction, kClothSelfFriction
    );
    result.config.airVelocityAndDensity = coMovingAir
        ? f4(3.0f, 4.0f, 0.0f, kAirDensity)
        : f4(0.0f, 0.0f, 0.0f, kAirDensity);
    result.config.aerodynamicCoefficients = f4(
        kYarnCrossflowDrag,
        kYarnSkinFriction,
        kFruitDrag,
        kFruitRotationalDrag
    );
    const auto particle = [](const double x, const double mass) {
        NumiClothBagGPUParticle value{};
        value.positionAndInverseMass = f4(
            static_cast<float>(x), 0.0f, 2.0f,
            static_cast<float>(1.0 / mass)
        );
        value.previousAndMass = f4(
            static_cast<float>(x), 0.0f, 2.0f,
            static_cast<float>(mass)
        );
        value.velocity = f4(3.0f, 4.0f, 0.0f, 0.0f);
        return value;
    };
    if (refined) {
        result.particles = {
            particle(-1.0, 0.5),
            particle(0.0, 1.0),
            particle(1.0, 0.5),
        };
        NumiClothBagGPUDistance first{};
        first.particlesAndColor = u4(0u, 1u, 0u, 0u);
        first.material = f4(1.0f, 0.0f, 0.0f, kStrainLimit);
        NumiClothBagGPUDistance second{};
        second.particlesAndColor = u4(1u, 2u, 1u, 0u);
        second.material = f4(1.0f, 0.0f, 0.0f, kStrainLimit);
        result.distances = {first, second};
        result.distanceBatches = {
            {u4(0u, 1u, 0u, 0u)},
            {u4(1u, 1u, 1u, 0u)},
        };
    } else {
        result.particles = {particle(-1.0, 1.0), particle(1.0, 1.0)};
        NumiClothBagGPUDistance distance{};
        distance.particlesAndColor = u4(0u, 1u, 0u, 0u);
        distance.material = f4(2.0f, 0.0f, 0.0f, kStrainLimit);
        result.distances = {distance};
        result.distanceBatches = {{u4(0u, 1u, 0u, 0u)}};
    }
    return result;
}

InitialState makeFruitAerodynamicsProbeState() {
    InitialState result;
    result.config.control = u4(
        NUMI_CLOTH_BAG_GPU_ABI_VERSION, 0u, 0u, 0u
    );
    result.config.constraintCounts = u4(0u, 0u, 0u, 1u);
    result.config.gravityAndTimestep = f4(0.0f, 0.0f, 0.0f, 0.01f);
    result.config.gripTargetAndActive = f4(0.0f, 0.0f, 0.0f, 0.0f);
    result.config.clothMaterial = f4(
        kClothRadius, 0.0f, kClothGroundFriction, kClothSelfFriction
    );
    result.config.airVelocityAndDensity = f4(
        0.0f, 0.0f, 0.0f, kAirDensity
    );
    result.config.aerodynamicCoefficients = f4(
        kYarnCrossflowDrag,
        kYarnSkinFriction,
        kFruitDrag,
        kFruitRotationalDrag
    );
    result.fruits = {
        makeProbeFruit({0.0, 0.0, 10.0}, 1.0f / 0.21f, 0.07f)
    };
    result.fruits[0].velocityAndGroundImpulse =
        f4(5.0f, -1.0f, 2.0f, 0.0f);
    result.fruits[0].angularVelocity = f4(3.0f, 4.0f, 2.0f, 0.0f);
    return result;
}

InitialState makeMouthReleaseProbeState(
    const DVec3 fruitPosition,
    const bool rotated,
    const std::uint32_t initialCandidateMask = 0u
) {
    InitialState result;
    constexpr std::uint32_t bottomCenter = kAround;
    result.config.control = u4(
        NUMI_CLOTH_BAG_GPU_ABI_VERSION, kAround + 1u, 0u, 0u
    );
    result.config.constraintCounts = u4(0u, 0u, 0u, 1u);
    result.config.gravityAndTimestep = f4(0.0f, 0.0f, 0.0f, 0.01f);
    result.config.gripTargetAndActive = f4(0.0f, 0.0f, 0.0f, 0.0f);
    result.config.clothMaterial = f4(
        kClothRadius, 0.0f, kClothGroundFriction, kClothSelfFriction
    );
    result.config.mouthControl = u4(kAround, bottomCenter, 0u, 0u);
    result.config.mouthMaterial = f4(0.025f, 0.0f, 0.0f, 0.0f);
    result.initialMouthCandidateMask = initialCandidateMask;
    result.particles.reserve(kAround + 1u);
    result.mouthRimParticles.reserve(kAround);
    const auto fixedParticle = [](const DVec3 position) {
        NumiClothBagGPUParticle particle{};
        particle.positionAndInverseMass = f4(
            static_cast<float>(position.x),
            static_cast<float>(position.y),
            static_cast<float>(position.z),
            0.0f
        );
        particle.previousAndMass = f4(
            static_cast<float>(position.x),
            static_cast<float>(position.y),
            static_cast<float>(position.z),
            0.0f
        );
        particle.velocity = f4(0.0f, 0.0f, 0.0f, 0.0f);
        return particle;
    };
    for (std::uint32_t ring = 0u; ring < kAround; ++ring) {
        const double angle = 2.0 * std::numbers::pi *
            static_cast<double>(ring) / static_cast<double>(kAround);
        const DVec3 position = rotated
            ? DVec3{1.0, std::cos(angle), std::sin(angle)}
            : DVec3{std::cos(angle), std::sin(angle), 1.0};
        result.particles.push_back(fixedParticle(position));
        result.mouthRimParticles.push_back(ring);
    }
    result.particles.push_back(fixedParticle({0.0, 0.0, 0.0}));
    result.fruits = {makeProbeFruit(fruitPosition, 1.0f, 0.10f)};
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
    for (const NumiClothBagGPUBatch& batch : state.fruitPairBatches) {
        std::vector<bool> used(state.fruits.size(), false);
        for (std::uint32_t local = 0u; local < batch.control.y; ++local) {
            const auto& pair = state.fruitPairs[batch.control.x + local];
            if (pair.fruitsAndColor.z != batch.control.z ||
                pair.fruitsAndColor.x >= used.size() ||
                pair.fruitsAndColor.y >= used.size() ||
                used[pair.fruitsAndColor.x] ||
                used[pair.fruitsAndColor.y]) {
                return false;
            }
            used[pair.fruitsAndColor.x] = true;
            used[pair.fruitsAndColor.y] = true;
        }
    }
    std::uint32_t observedMaximumSelfBatch = 0u;
    for (const NumiClothBagGPUBatch& batch : state.selfBatches) {
        if (batch.control.x + batch.control.y > state.selfPairs.size()) {
            return false;
        }
        observedMaximumSelfBatch = std::max(
            observedMaximumSelfBatch, batch.control.y
        );
        std::vector<bool> used(state.particles.size(), false);
        for (std::uint32_t local = 0u; local < batch.control.y; ++local) {
            const NumiClothBagGPUSelfPair& pair =
                state.selfPairs[batch.control.x + local];
            if (pair.firstSegment >= state.distances.size() ||
                pair.secondSegment >= state.distances.size() ||
                pair.firstSegment == pair.secondSegment) {
                return false;
            }
            const mr_uint4 first = state.distances[
                pair.firstSegment
            ].particlesAndColor;
            const mr_uint4 second = state.distances[
                pair.secondSegment
            ].particlesAndColor;
            const std::array<std::uint32_t, 4> participants{{
                first.x, first.y, second.x, second.y,
            }};
            for (const std::uint32_t particleIndex : participants) {
                if (particleIndex >= used.size() || used[particleIndex]) {
                    return false;
                }
                used[particleIndex] = true;
            }
        }
    }
    if (observedMaximumSelfBatch != state.maximumSelfBatchSize ||
        state.selfPairs.size() != state.config.contactCounts.z ||
        state.selfBatches.size() != state.config.contactCounts.w) {
        return false;
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
    double predictedVerticalVelocity{};
    double groundNormalImpulse{};
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

struct OracleFruit {
    DVec3 position{};
    DVec3 previous{};
    DVec3 velocity{};
    DVec3 angularVelocity{};
    std::array<double, 4> orientation{{0.0, 0.0, 0.0, 1.0}};
    double inverseMass{};
    double radius{};
    double groundNormalImpulse{};
};

struct OracleFruitPair {
    std::uint32_t first{};
    std::uint32_t second{};
    DVec3 weightedNormal{};
    double normalImpulse{};
};

struct OracleYarnContact {
    std::uint32_t fruit{};
    std::uint32_t segment{};
    std::uint32_t first{};
    std::uint32_t second{};
    DVec3 currentNormal{};
    DVec3 sweptNormal{};
    double currentSeparation{};
    double currentWeight{};
    double sweptWeight{};
    double impactTime{};
    double combinedRadius{};
    double removedAdvance{};
    bool currentOverlap{};
    bool sweptImpact{};
    bool degenerateCurrent{};
    DVec3 fruitNormalImpulse{};
    double normalImpulse{};
    std::array<double, 2> segmentImpulse{};
    std::uint32_t responseCount{};
    double firstSweptTime{};
    double sweptAdvance{};
};

struct OracleSelfImpulse {
    DVec3 normalOnFirst{};
    double normalImpulse{};
    std::array<double, 2> firstEndpointImpulses{};
    std::array<double, 2> secondEndpointImpulses{};
};

struct OracleResult {
    std::vector<OracleParticle> particles;
    std::vector<OracleDistance> distances;
    std::vector<OracleGrip> grips;
    std::vector<OracleKnot> knots;
    std::vector<OracleBend> bends;
    std::vector<OracleFruit> fruits;
    std::vector<OracleFruitPair> fruitPairs;
    std::vector<OracleYarnContact> yarnContacts;
    std::uint64_t presentSelfContacts{};
    std::uint64_t sweptSelfContacts{};
    double maximumSelfCorrection{};
    std::unordered_map<std::uint32_t, OracleSelfImpulse> selfImpulses;
    std::uint64_t selfFrictionContacts{};
    std::array<std::uint64_t, 4> frictionContacts{};
    std::uint64_t rollingContacts{};
    double maximumFrictionConeRatio{};
    double maximumRollingResistanceRatio{};
    double maximumYarnAerodynamicForce{};
    double maximumFruitAerodynamicForce{};
    double maximumFruitAerodynamicTorque{};
    std::uint32_t mouthCandidateMask{};
    std::uint32_t releasedMask{};
    std::vector<double> maximumMouthClearanceByFruit;
};

struct OraclePointSegmentSample {
    DVec3 closest{};
    double weight{};
    double distance{};
};

struct OracleSegmentSegmentSample {
    DVec3 firstPoint{};
    DVec3 secondPoint{};
    double firstWeight{};
    double secondWeight{};
    double distance{};
};

OraclePointSegmentSample samplePointSegment(
    const DVec3 point,
    const DVec3 first,
    const DVec3 second
) {
    const DVec3 direction = second - first;
    const double lengthSquared = dot(direction, direction);
    const double weight = lengthSquared > 1.0e-20
        ? std::clamp(
            dot(direction, point - first) / lengthSquared,
            0.0,
            1.0
        )
        : 0.0;
    const DVec3 closest = first + direction * weight;
    return {closest, weight, length(closest - point)};
}

OracleSegmentSegmentSample sampleSegments(
    const DVec3 firstStart,
    const DVec3 firstEnd,
    const DVec3 secondStart,
    const DVec3 secondEnd
) {
    const DVec3 firstDirection = firstEnd - firstStart;
    const DVec3 secondDirection = secondEnd - secondStart;
    const DVec3 offset = firstStart - secondStart;
    const double firstLengthSquared = dot(firstDirection, firstDirection);
    const double secondLengthSquared = dot(secondDirection, secondDirection);
    const double secondProjection = dot(secondDirection, offset);
    double firstWeight = 0.0;
    double secondWeight = 0.0;
    if (firstLengthSquared <= 1.0e-20 &&
        secondLengthSquared <= 1.0e-20) {
        return {
            firstStart,
            secondStart,
            0.0,
            0.0,
            length(secondStart - firstStart),
        };
    }
    if (firstLengthSquared <= 1.0e-20) {
        secondWeight = std::clamp(
            secondProjection / secondLengthSquared, 0.0, 1.0
        );
    } else {
        const double firstProjection = dot(firstDirection, offset);
        if (secondLengthSquared <= 1.0e-20) {
            firstWeight = std::clamp(
                -firstProjection / firstLengthSquared, 0.0, 1.0
            );
        } else {
            const double crossProjection = dot(
                firstDirection, secondDirection
            );
            const double denominator =
                firstLengthSquared * secondLengthSquared -
                crossProjection * crossProjection;
            if (denominator > 1.0e-20) {
                firstWeight = std::clamp(
                    (crossProjection * secondProjection -
                     firstProjection * secondLengthSquared) / denominator,
                    0.0,
                    1.0
                );
            }
            secondWeight = (
                crossProjection * firstWeight + secondProjection
            ) / secondLengthSquared;
            if (secondWeight < 0.0) {
                secondWeight = 0.0;
                firstWeight = std::clamp(
                    -firstProjection / firstLengthSquared, 0.0, 1.0
                );
            } else if (secondWeight > 1.0) {
                secondWeight = 1.0;
                firstWeight = std::clamp(
                    (crossProjection - firstProjection) /
                        firstLengthSquared,
                    0.0,
                    1.0
                );
            }
        }
    }
    const DVec3 firstPoint = firstStart + firstDirection * firstWeight;
    const DVec3 secondPoint = secondStart + secondDirection * secondWeight;
    return {
        firstPoint,
        secondPoint,
        firstWeight,
        secondWeight,
        length(secondPoint - firstPoint),
    };
}

DVec3 selfContactNormal(
    const DVec3 separation,
    const DVec3 firstDirection,
    const DVec3 secondDirection,
    const DVec3 previousOffset
) {
    const double separationLength = length(separation);
    if (separationLength > 1.0e-12) {
        return separation / separationLength;
    }
    DVec3 normal = normalized(cross(firstDirection, secondDirection));
    if (dot(previousOffset, normal) > 0.0) {
        normal = normal * -1.0;
    }
    return normal;
}

bool edgeBoundsOverlap(
    const OracleParticle& firstStart,
    const OracleParticle& firstEnd,
    const OracleParticle& secondStart,
    const OracleParticle& secondEnd,
    const bool swept,
    const double expansion
) {
    const auto minimum = [](const DVec3 first, const DVec3 second) {
        return DVec3{
            std::min(first.x, second.x),
            std::min(first.y, second.y),
            std::min(first.z, second.z),
        };
    };
    const auto maximum = [](const DVec3 first, const DVec3 second) {
        return DVec3{
            std::max(first.x, second.x),
            std::max(first.y, second.y),
            std::max(first.z, second.z),
        };
    };
    DVec3 firstMinimum = minimum(firstStart.position, firstEnd.position);
    DVec3 firstMaximum = maximum(firstStart.position, firstEnd.position);
    DVec3 secondMinimum = minimum(secondStart.position, secondEnd.position);
    DVec3 secondMaximum = maximum(secondStart.position, secondEnd.position);
    if (swept) {
        firstMinimum = minimum(
            firstMinimum, minimum(firstStart.previous, firstEnd.previous)
        );
        firstMaximum = maximum(
            firstMaximum, maximum(firstStart.previous, firstEnd.previous)
        );
        secondMinimum = minimum(
            secondMinimum, minimum(secondStart.previous, secondEnd.previous)
        );
        secondMaximum = maximum(
            secondMaximum, maximum(secondStart.previous, secondEnd.previous)
        );
    }
    const DVec3 amount{expansion, expansion, expansion};
    firstMinimum -= amount;
    firstMaximum += amount;
    secondMinimum -= amount;
    secondMaximum += amount;
    return firstMinimum.x <= secondMaximum.x &&
        firstMaximum.x >= secondMinimum.x &&
        firstMinimum.y <= secondMaximum.y &&
        firstMaximum.y >= secondMinimum.y &&
        firstMinimum.z <= secondMaximum.z &&
        firstMaximum.z >= secondMinimum.z;
}

std::vector<OracleYarnContact> buildOracleYarnContacts(
    const InitialState& initial,
    const OracleResult& state,
    const std::vector<OracleYarnContact>* accumulated = nullptr
) {
    std::vector<OracleYarnContact> result;
    result.reserve(initial.yarnContacts.size());
    const double clothRadius = initial.config.clothMaterial.x;
    for (const NumiClothBagGPUYarnContact& source : initial.yarnContacts) {
        const std::uint32_t fruitIndex = source.identity.x;
        const std::uint32_t segmentIndex = source.identity.y;
        const OracleFruit& fruit = state.fruits[fruitIndex];
        const OracleDistance& segment = state.distances[segmentIndex];
        const OracleParticle& first = state.particles[segment.first];
        const OracleParticle& second = state.particles[segment.second];
        const double target = fruit.radius + clothRadius;
        const OraclePointSegmentSample current = samplePointSegment(
            fruit.position, first.position, second.position
        );
        const bool degenerateCurrent = current.distance < 1.0e-12;
        DVec3 currentNormal{};
        if (degenerateCurrent) {
            const DVec3 segmentDirection = normalized(
                second.position - first.position
            );
            currentNormal = normalized(cross(
                segmentDirection,
                std::abs(segmentDirection.z) < 0.9
                    ? DVec3{0.0, 0.0, 1.0}
                    : DVec3{1.0, 0.0, 0.0}
            ));
        } else {
            currentNormal =
                (current.closest - fruit.position) / current.distance;
        }

        double impactTime = 0.0;
        DVec3 sweptNormal{};
        double sweptWeight = 0.0;
        double removedAdvance = 0.0;
        bool sweptImpact = false;
        const double motionBound =
            length(fruit.position - fruit.previous) +
            length(first.position - first.previous) +
            length(second.position - second.previous);
        if (motionBound >= 1.0e-14) {
            const auto sweptSample = [&fruit, &first, &second](
                const double time
            ) {
                const DVec3 ball = fruit.previous +
                    (fruit.position - fruit.previous) * time;
                const DVec3 firstPosition = first.previous +
                    (first.position - first.previous) * time;
                const DVec3 secondPosition = second.previous +
                    (second.position - second.previous) * time;
                return samplePointSegment(
                    ball, firstPosition, secondPosition
                );
            };
            constexpr double distanceTolerance = 1.0e-9;
            OraclePointSegmentSample impact = sweptSample(0.0);
            bool found = impact.distance <= target + distanceTolerance;
            if (!found) {
                for (std::uint32_t iteration = 0u;
                     iteration < 80u;
                     ++iteration) {
                    impact = sweptSample(impactTime);
                    const double gap = impact.distance - target;
                    if (gap <= distanceTolerance) {
                        found = true;
                        break;
                    }
                    const double advance = 0.9 * gap / motionBound;
                    if (!std::isfinite(advance) || !(advance > 0.0) ||
                        impactTime + advance >= 1.0) {
                        break;
                    }
                    impactTime += std::max(advance, 1.0e-10);
                }
            }
            if (found && impact.distance >= 1.0e-12) {
                const DVec3 ballAtImpact = fruit.previous +
                    (fruit.position - fruit.previous) * impactTime;
                const DVec3 firstAtImpact = first.previous +
                    (first.position - first.previous) * impactTime;
                const DVec3 secondAtImpact = second.previous +
                    (second.position - second.previous) * impactTime;
                sweptNormal =
                    (impact.closest - ballAtImpact) / impact.distance;
                sweptWeight = impact.weight;
                const DVec3 segmentRemaining =
                    (first.position - firstAtImpact) *
                        (1.0 - sweptWeight) +
                    (second.position - secondAtImpact) * sweptWeight;
                const DVec3 ballRemaining =
                    fruit.position - ballAtImpact;
                removedAdvance = dot(
                    ballRemaining - segmentRemaining, sweptNormal
                );
                sweptImpact = removedAdvance > 0.0;
                if (!sweptImpact) {
                    removedAdvance = 0.0;
                }
            }
        }
        OracleYarnContact contact{
            fruitIndex,
            segmentIndex,
            segment.first,
            segment.second,
            currentNormal,
            sweptNormal,
            current.distance - target,
            current.weight,
            sweptWeight,
            impactTime,
            target,
            removedAdvance,
            current.distance < target,
            sweptImpact,
            degenerateCurrent,
        };
        if (accumulated != nullptr && result.size() < accumulated->size()) {
            const OracleYarnContact& previous = (*accumulated)[result.size()];
            contact.fruitNormalImpulse = previous.fruitNormalImpulse;
            contact.normalImpulse = previous.normalImpulse;
            contact.segmentImpulse = previous.segmentImpulse;
            contact.responseCount = previous.responseCount;
            contact.firstSweptTime = previous.firstSweptTime;
            contact.sweptAdvance = previous.sweptAdvance;
        }
        result.push_back(contact);
    }
    return result;
}

double applyOracleYarnCorrection(
    const InitialState& initial,
    OracleResult& state,
    const std::uint32_t firstIndex,
    const std::uint32_t secondIndex,
    OracleFruit& fruit,
    const double firstWeight,
    const double secondWeight,
    const DVec3 normal,
    const double correctionDistance
) {
    OracleParticle& first = state.particles[firstIndex];
    OracleParticle& second = state.particles[secondIndex];
    const bool groundEnabled = initial.config.constraintCounts.z != 0u;
    const double groundHeight = initial.config.clothMaterial.x;
    bool firstGroundActive = groundEnabled && normal.z < 0.0 &&
        firstWeight > 0.0 && first.position.z <= groundHeight + 1.0e-9;
    bool secondGroundActive = groundEnabled && normal.z < 0.0 &&
        secondWeight > 0.0 && second.position.z <= groundHeight + 1.0e-9;
    if (firstGroundActive) {
        first.position.z = groundHeight;
    }
    if (secondGroundActive) {
        second.position.z = groundHeight;
    }
    double remaining = correctionDistance;
    double accumulatedLambda = 0.0;
    for (std::uint32_t activeSetIteration = 0u;
         activeSetIteration < 3u && remaining > 1.0e-14;
         ++activeSetIteration) {
        DVec3 firstResponse = normal * first.inverseMass;
        DVec3 secondResponse = normal * second.inverseMass;
        if (firstGroundActive) {
            firstResponse.z = 0.0;
        }
        if (secondGroundActive) {
            secondResponse.z = 0.0;
        }
        const double denominator = fruit.inverseMass +
            dot(normal, firstResponse) * firstWeight * firstWeight +
            dot(normal, secondResponse) * secondWeight * secondWeight;
        if (!(denominator > 0.0)) {
            break;
        }
        const double unconstrainedLambda = remaining / denominator;
        double stepLambda = unconstrainedLambda;
        if (groundEnabled && normal.z < 0.0) {
            const double firstVertical = firstResponse.z * firstWeight;
            if (!firstGroundActive && firstVertical < 0.0) {
                stepLambda = std::min(
                    stepLambda,
                    std::max(0.0, first.position.z - groundHeight) /
                        -firstVertical
                );
            }
            const double secondVertical = secondResponse.z * secondWeight;
            if (!secondGroundActive && secondVertical < 0.0) {
                stepLambda = std::min(
                    stepLambda,
                    std::max(0.0, second.position.z - groundHeight) /
                        -secondVertical
                );
            }
        }
        fruit.position -= normal * (fruit.inverseMass * stepLambda);
        first.position += firstResponse * (firstWeight * stepLambda);
        second.position += secondResponse * (secondWeight * stepLambda);
        accumulatedLambda += stepLambda;
        remaining = std::max(
            0.0,
            remaining - denominator * stepLambda
        );
        if (stepLambda >= unconstrainedLambda - 1.0e-14) {
            break;
        }
        bool activated = false;
        if (!firstGroundActive && firstWeight > 0.0 &&
            first.position.z <= groundHeight + 1.0e-9) {
            first.position.z = groundHeight;
            firstGroundActive = true;
            activated = true;
        }
        if (!secondGroundActive && secondWeight > 0.0 &&
            second.position.z <= groundHeight + 1.0e-9) {
            second.position.z = groundHeight;
            secondGroundActive = true;
            activated = true;
        }
        if (!activated) {
            break;
        }
    }
    return accumulatedLambda;
}

void solveOracleYarnBatches(
    const InitialState& initial,
    OracleResult& state,
    const bool swept
) {
    if (state.yarnContacts.empty() || state.fruits.empty()) {
        return;
    }
    const double timestep = initial.config.gravityAndTimestep.w;
    const double clothRadius = initial.config.clothMaterial.x;
    for (const NumiClothBagGPUBatch& batch : initial.distanceBatches) {
        for (std::uint32_t phase = 0u; phase < batch.control.y; ++phase) {
            for (std::uint32_t fruitIndex = 0u;
                 fruitIndex < state.fruits.size();
                 ++fruitIndex) {
                const std::uint32_t localSegment =
                    (phase + fruitIndex) % batch.control.y;
                const std::uint32_t segmentIndex =
                    batch.control.x + localSegment;
                const OracleDistance& segment =
                    state.distances[segmentIndex];
                OracleFruit& fruit = state.fruits[fruitIndex];
                OracleYarnContact& contact = state.yarnContacts[
                    fruitIndex * state.distances.size() + segmentIndex
                ];
                DVec3 normal{};
                double firstWeight = 0.0;
                double secondWeight = 0.0;
                double correction = 0.0;
                if (swept) {
                    if (contact.sweptImpact) {
                        normal = contact.sweptNormal;
                        secondWeight = contact.sweptWeight;
                        firstWeight = 1.0 - secondWeight;
                        correction = contact.removedAdvance;
                    }
                } else {
                    const OraclePointSegmentSample current =
                        samplePointSegment(
                            fruit.position,
                            state.particles[segment.first].position,
                            state.particles[segment.second].position
                        );
                    const double target = fruit.radius + clothRadius;
                    if (current.distance < target) {
                        if (current.distance < 1.0e-12) {
                            const DVec3 segmentDirection = normalized(
                                state.particles[segment.second].position -
                                state.particles[segment.first].position
                            );
                            normal = normalized(cross(
                                segmentDirection,
                                std::abs(segmentDirection.z) < 0.9
                                    ? DVec3{0.0, 0.0, 1.0}
                                    : DVec3{1.0, 0.0, 0.0}
                            ));
                        } else {
                            normal = (
                                current.closest - fruit.position
                            ) / current.distance;
                        }
                        secondWeight = current.weight;
                        firstWeight = 1.0 - secondWeight;
                        correction = target - current.distance;
                    }
                }
                if (!(correction > 0.0)) {
                    continue;
                }
                const double lambda = applyOracleYarnCorrection(
                    initial,
                    state,
                    segment.first,
                    segment.second,
                    fruit,
                    firstWeight,
                    secondWeight,
                    normal,
                    correction
                );
                if (!(lambda > 0.0)) {
                    continue;
                }
                const double impulse = lambda / timestep;
                contact.fruitNormalImpulse -= normal * impulse;
                contact.normalImpulse += impulse;
                contact.segmentImpulse[0] += firstWeight * impulse;
                contact.segmentImpulse[1] += secondWeight * impulse;
                if (swept) {
                    contact.firstSweptTime = contact.impactTime;
                    contact.sweptAdvance += correction;
                }
                ++contact.responseCount;
            }
        }
    }
}

void solveOracleSelfContact(
    const InitialState& initial,
    OracleResult& state,
    const bool swept
) {
    if (initial.selfPairs.empty()) {
        return;
    }
    const double target = 2.0 * initial.config.clothMaterial.x;
    constexpr double distanceTolerance = 1.0e-9;
    for (const NumiClothBagGPUBatch& batch : initial.selfBatches) {
        for (std::uint32_t localIndex = 0u;
             localIndex < batch.control.y;
             ++localIndex) {
            const std::uint32_t pairIndex = batch.control.x + localIndex;
            const NumiClothBagGPUSelfPair& pair =
                initial.selfPairs[pairIndex];
            const mr_uint4 first = initial.distances[
                pair.firstSegment
            ].particlesAndColor;
            const mr_uint4 second = initial.distances[
                pair.secondSegment
            ].particlesAndColor;
            const mr_uint4 indices = u4(
                first.x, first.y, second.x, second.y
            );
            OracleParticle& firstStart = state.particles[indices.x];
            OracleParticle& firstEnd = state.particles[indices.y];
            OracleParticle& secondStart = state.particles[indices.z];
            OracleParticle& secondEnd = state.particles[indices.w];
            if (!edgeBoundsOverlap(
                firstStart,
                firstEnd,
                secondStart,
                secondEnd,
                swept,
                initial.config.clothMaterial.x
            )) {
                continue;
            }
            double firstWeight = 0.0;
            double secondWeight = 0.0;
            DVec3 normal{};
            double correction = 0.0;
            if (!swept) {
                const OracleSegmentSegmentSample closest = sampleSegments(
                    firstStart.position,
                    firstEnd.position,
                    secondStart.position,
                    secondEnd.position
                );
                if (!(closest.distance < target)) {
                    continue;
                }
                firstWeight = closest.firstWeight;
                secondWeight = closest.secondWeight;
                const DVec3 firstPrevious =
                    firstStart.previous * (1.0 - firstWeight) +
                    firstEnd.previous * firstWeight;
                const DVec3 secondPrevious =
                    secondStart.previous * (1.0 - secondWeight) +
                    secondEnd.previous * secondWeight;
                normal = selfContactNormal(
                    closest.secondPoint - closest.firstPoint,
                    firstEnd.position - firstStart.position,
                    secondEnd.position - secondStart.position,
                    firstPrevious - secondPrevious
                );
                correction = target - closest.distance;
            } else {
                const double motionBound =
                    length(firstStart.position - firstStart.previous) +
                    length(firstEnd.position - firstEnd.previous) +
                    length(secondStart.position - secondStart.previous) +
                    length(secondEnd.position - secondEnd.previous);
                if (motionBound < 1.0e-14) {
                    continue;
                }
                const auto sampleSwept = [&](const double time) {
                    return sampleSegments(
                        firstStart.previous +
                            (firstStart.position - firstStart.previous) * time,
                        firstEnd.previous +
                            (firstEnd.position - firstEnd.previous) * time,
                        secondStart.previous +
                            (secondStart.position - secondStart.previous) * time,
                        secondEnd.previous +
                            (secondEnd.position - secondEnd.previous) * time
                    );
                };
                double time = 0.0;
                OracleSegmentSegmentSample impact = sampleSwept(0.0);
                bool found = impact.distance <= target + distanceTolerance;
                if (!found) {
                    for (std::uint32_t iteration = 0u;
                         iteration < 80u;
                         ++iteration) {
                        impact = sampleSwept(time);
                        const double gap = impact.distance - target;
                        if (gap <= distanceTolerance) {
                            found = true;
                            break;
                        }
                        const double advance = 0.9 * gap / motionBound;
                        if (!std::isfinite(advance) || !(advance > 0.0) ||
                            time + advance >= 1.0) {
                            break;
                        }
                        time += std::max(advance, 1.0e-10);
                    }
                }
                if (!found) {
                    continue;
                }
                firstWeight = impact.firstWeight;
                secondWeight = impact.secondWeight;
                const DVec3 firstPrevious =
                    firstStart.previous * (1.0 - firstWeight) +
                    firstEnd.previous * firstWeight;
                const DVec3 secondPrevious =
                    secondStart.previous * (1.0 - secondWeight) +
                    secondEnd.previous * secondWeight;
                const DVec3 firstImpactStart = firstStart.previous +
                    (firstStart.position - firstStart.previous) * time;
                const DVec3 firstImpactEnd = firstEnd.previous +
                    (firstEnd.position - firstEnd.previous) * time;
                const DVec3 secondImpactStart = secondStart.previous +
                    (secondStart.position - secondStart.previous) * time;
                const DVec3 secondImpactEnd = secondEnd.previous +
                    (secondEnd.position - secondEnd.previous) * time;
                normal = selfContactNormal(
                    impact.secondPoint - impact.firstPoint,
                    firstImpactEnd - firstImpactStart,
                    secondImpactEnd - secondImpactStart,
                    firstPrevious - secondPrevious
                );
                const DVec3 firstRemaining =
                    (firstStart.position - firstImpactStart) *
                        (1.0 - firstWeight) +
                    (firstEnd.position - firstImpactEnd) * firstWeight;
                const DVec3 secondRemaining =
                    (secondStart.position - secondImpactStart) *
                        (1.0 - secondWeight) +
                    (secondEnd.position - secondImpactEnd) * secondWeight;
                correction = dot(firstRemaining - secondRemaining, normal);
                if (!(correction > 0.0)) {
                    continue;
                }
            }
            const double firstStartWeight = 1.0 - firstWeight;
            const double secondStartWeight = 1.0 - secondWeight;
            const double denominator =
                firstStart.inverseMass *
                    firstStartWeight * firstStartWeight +
                firstEnd.inverseMass * firstWeight * firstWeight +
                secondStart.inverseMass *
                    secondStartWeight * secondStartWeight +
                secondEnd.inverseMass * secondWeight * secondWeight;
            if (!(denominator > 0.0)) {
                continue;
            }
            const double lambda = correction / denominator;
            firstStart.position -= normal *
                (firstStart.inverseMass * firstStartWeight * lambda);
            firstEnd.position -= normal *
                (firstEnd.inverseMass * firstWeight * lambda);
            secondStart.position += normal *
                (secondStart.inverseMass * secondStartWeight * lambda);
            secondEnd.position += normal *
                (secondEnd.inverseMass * secondWeight * lambda);
            if (swept) {
                ++state.sweptSelfContacts;
            } else {
                ++state.presentSelfContacts;
            }
            state.maximumSelfCorrection = std::max(
                state.maximumSelfCorrection, correction
            );
            const double impulse = lambda /
                initial.config.gravityAndTimestep.w;
            OracleSelfImpulse& response = state.selfImpulses[pairIndex];
            response.normalOnFirst -= normal * impulse;
            response.normalImpulse += impulse;
            response.firstEndpointImpulses[0] +=
                firstStartWeight * impulse;
            response.firstEndpointImpulses[1] += firstWeight * impulse;
            response.secondEndpointImpulses[0] +=
                secondStartWeight * impulse;
            response.secondEndpointImpulses[1] += secondWeight * impulse;
        }
    }
}

void solveOracleFruitPairs(
    const InitialState& initial,
    OracleResult& state
) {
    const double timestep = initial.config.gravityAndTimestep.w;
    for (const NumiClothBagGPUBatch& batch : initial.fruitPairBatches) {
        for (std::uint32_t local = 0u; local < batch.control.y; ++local) {
            OracleFruitPair& pair =
                state.fruitPairs[batch.control.x + local];
            OracleFruit& first = state.fruits[pair.first];
            OracleFruit& second = state.fruits[pair.second];
            const DVec3 difference = second.position - first.position;
            const double currentLength = length(difference);
            const double target = first.radius + second.radius;
            if (!(currentLength < target) ||
                !(currentLength > 1.0e-12)) {
                continue;
            }
            const double denominator =
                first.inverseMass + second.inverseMass;
            if (!(denominator > 0.0)) {
                continue;
            }
            const double lambda = (target - currentLength) / denominator;
            const DVec3 normal = difference / currentLength;
            const DVec3 correction = normal * lambda;
            first.position -= correction * first.inverseMass;
            second.position += correction * second.inverseMass;
            const double impulseMagnitude = lambda / timestep;
            pair.weightedNormal += normal * impulseMagnitude;
            pair.normalImpulse += impulseMagnitude;
        }
    }
}

void solveOracleGround(
    const InitialState& initial,
    OracleResult& state
) {
    if (initial.config.constraintCounts.z == 0u) {
        return;
    }
    const double timestep = initial.config.gravityAndTimestep.w;
    const double clothRadius = initial.config.clothMaterial.x;
    for (OracleParticle& particle : state.particles) {
        particle.position.z = std::max(particle.position.z, clothRadius);
    }
    for (OracleFruit& fruit : state.fruits) {
        const double penetration = fruit.radius - fruit.position.z;
        if (penetration > 0.0) {
            fruit.groundNormalImpulse += penetration /
                (fruit.inverseMass * timestep);
            fruit.position.z = fruit.radius;
        }
    }
}

void solveOracleStrainLimits(
    const InitialState& initial,
    OracleResult& state,
    const std::uint32_t sweeps
) {
    for (std::uint32_t sweep = 0u; sweep < sweeps; ++sweep) {
        for (const NumiClothBagGPUBatch& batch : initial.distanceBatches) {
            for (std::uint32_t local = 0u; local < batch.control.y; ++local) {
                const OracleDistance& constraint =
                    state.distances[batch.control.x + local];
                OracleParticle& first = state.particles[constraint.first];
                OracleParticle& second = state.particles[constraint.second];
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
}

double oracleFruitInverseInertia(const OracleFruit& fruit) {
    return 2.5 * fruit.inverseMass / (fruit.radius * fruit.radius);
}

void applyOracleFruitImpulse(
    OracleFruit& fruit,
    const DVec3 impulse,
    const DVec3 contactOffset
) {
    fruit.velocity += impulse * fruit.inverseMass;
    fruit.angularVelocity += cross(contactOffset, impulse) *
        oracleFruitInverseInertia(fruit);
}

void recordOracleFriction(
    OracleResult& state,
    const std::size_t counter,
    const double tangentialImpulse,
    const double frictionLimit
) {
    ++state.frictionContacts[counter];
    if (frictionLimit > 0.0) {
        state.maximumFrictionConeRatio = std::max(
            state.maximumFrictionConeRatio,
            tangentialImpulse / frictionLimit
        );
    }
}

void applyOracleClothGroundFriction(
    const InitialState& initial,
    OracleResult& state
) {
    if (initial.config.constraintCounts.z == 0u) {
        return;
    }
    const double friction = initial.config.clothMaterial.z;
    for (OracleParticle& particle : state.particles) {
        if (!(particle.groundNormalImpulse > 0.0) ||
            !(particle.inverseMass > 0.0)) {
            continue;
        }
        const DVec3 tangentVelocity{
            particle.velocity.x, particle.velocity.y, 0.0
        };
        const double slipSpeed = length(tangentVelocity);
        if (!(slipSpeed > 1.0e-10)) {
            continue;
        }
        const double frictionLimit =
            friction * particle.groundNormalImpulse;
        const double tangentialImpulse = std::min(
            slipSpeed / particle.inverseMass, frictionLimit
        );
        if (!(tangentialImpulse > 0.0)) {
            continue;
        }
        particle.velocity -= tangentVelocity * (
            tangentialImpulse * particle.inverseMass / slipSpeed
        );
        recordOracleFriction(
            state, 2u, tangentialImpulse, frictionLimit
        );
    }
}

void applyOracleSelfFriction(
    const InitialState& initial,
    OracleResult& state
) {
    const double friction = initial.config.clothMaterial.w;
    for (const NumiClothBagGPUBatch& batch : initial.selfBatches) {
        for (std::uint32_t local = 0u; local < batch.control.y; ++local) {
            const std::uint32_t pairIndex = batch.control.x + local;
            const auto responseIterator = state.selfImpulses.find(pairIndex);
            if (responseIterator == state.selfImpulses.end()) {
                continue;
            }
            const OracleSelfImpulse& response = responseIterator->second;
            const double normalLength = length(response.normalOnFirst);
            const double firstSum =
                response.firstEndpointImpulses[0] +
                response.firstEndpointImpulses[1];
            const double secondSum =
                response.secondEndpointImpulses[0] +
                response.secondEndpointImpulses[1];
            if (!(response.normalImpulse > 0.0) ||
                !(normalLength > 1.0e-10) ||
                !(firstSum > 1.0e-12) || !(secondSum > 1.0e-12)) {
                continue;
            }
            const NumiClothBagGPUSelfPair& pair =
                initial.selfPairs[pairIndex];
            const OracleDistance& firstDistance =
                state.distances[pair.firstSegment];
            const OracleDistance& secondDistance =
                state.distances[pair.secondSegment];
            OracleParticle& firstStart =
                state.particles[firstDistance.first];
            OracleParticle& firstEnd =
                state.particles[firstDistance.second];
            OracleParticle& secondStart =
                state.particles[secondDistance.first];
            OracleParticle& secondEnd =
                state.particles[secondDistance.second];
            const DVec3 normal = response.normalOnFirst / normalLength;
            const std::array<double, 2> firstWeights{{
                response.firstEndpointImpulses[0] / firstSum,
                response.firstEndpointImpulses[1] / firstSum,
            }};
            const std::array<double, 2> secondWeights{{
                response.secondEndpointImpulses[0] / secondSum,
                response.secondEndpointImpulses[1] / secondSum,
            }};
            const DVec3 firstVelocity =
                firstStart.velocity * firstWeights[0] +
                firstEnd.velocity * firstWeights[1];
            const DVec3 secondVelocity =
                secondStart.velocity * secondWeights[0] +
                secondEnd.velocity * secondWeights[1];
            const DVec3 relativeVelocity = firstVelocity - secondVelocity;
            const DVec3 tangentVelocity = relativeVelocity -
                normal * dot(relativeVelocity, normal);
            const double slipSpeed = length(tangentVelocity);
            if (!(slipSpeed > 1.0e-10)) {
                continue;
            }
            const DVec3 tangent = tangentVelocity / slipSpeed;
            const double denominator =
                firstStart.inverseMass * firstWeights[0] * firstWeights[0] +
                firstEnd.inverseMass * firstWeights[1] * firstWeights[1] +
                secondStart.inverseMass *
                    secondWeights[0] * secondWeights[0] +
                secondEnd.inverseMass *
                    secondWeights[1] * secondWeights[1];
            if (!(denominator > 0.0)) {
                continue;
            }
            const double frictionLimit =
                friction * response.normalImpulse;
            const double tangentialImpulse = std::min(
                slipSpeed / denominator, frictionLimit
            );
            if (!(tangentialImpulse > 0.0)) {
                continue;
            }
            const DVec3 impulseOnFirst = tangent * -tangentialImpulse;
            firstStart.velocity += impulseOnFirst *
                (firstStart.inverseMass * firstWeights[0]);
            firstEnd.velocity += impulseOnFirst *
                (firstEnd.inverseMass * firstWeights[1]);
            secondStart.velocity -= impulseOnFirst *
                (secondStart.inverseMass * secondWeights[0]);
            secondEnd.velocity -= impulseOnFirst *
                (secondEnd.inverseMass * secondWeights[1]);
            ++state.selfFrictionContacts;
            if (frictionLimit > 0.0) {
                state.maximumFrictionConeRatio = std::max(
                    state.maximumFrictionConeRatio,
                    tangentialImpulse / frictionLimit
                );
            }
        }
    }
}

void applyOracleYarnFriction(
    const InitialState& initial,
    OracleResult& state
) {
    const double friction = initial.config.fruitMaterial.w;
    const bool groundEnabled = initial.config.constraintCounts.z != 0u;
    const double groundHeight = initial.config.clothMaterial.x;
    for (const NumiClothBagGPUBatch& batch : initial.distanceBatches) {
        for (std::uint32_t phase = 0u; phase < batch.control.y; ++phase) {
            for (std::uint32_t fruitIndex = 0u;
                 fruitIndex < state.fruits.size();
                 ++fruitIndex) {
                const std::uint32_t localSegment =
                    (phase + fruitIndex) % batch.control.y;
                const std::uint32_t segmentIndex =
                    batch.control.x + localSegment;
                const OracleDistance& segment =
                    state.distances[segmentIndex];
                const OracleYarnContact& contact = state.yarnContacts[
                    fruitIndex * state.distances.size() + segmentIndex
                ];
                if (!(contact.normalImpulse > 0.0) ||
                    !(length(contact.fruitNormalImpulse) > 1.0e-10)) {
                    continue;
                }
                const DVec3 normal = normalized(
                    contact.fruitNormalImpulse
                );
                std::array<double, 2> weights{{
                    contact.segmentImpulse[0] / contact.normalImpulse,
                    contact.segmentImpulse[1] / contact.normalImpulse,
                }};
                const double weightSum = weights[0] + weights[1];
                if (!(weightSum > 1.0e-12)) {
                    continue;
                }
                weights[0] /= weightSum;
                weights[1] /= weightSum;
                OracleParticle& first = state.particles[segment.first];
                OracleParticle& second = state.particles[segment.second];
                OracleFruit& fruit = state.fruits[fruitIndex];
                const DVec3 yarnVelocity =
                    first.velocity * weights[0] +
                    second.velocity * weights[1];
                const DVec3 ballOffset = normal * -fruit.radius;
                const DVec3 ballContactVelocity = fruit.velocity +
                    cross(fruit.angularVelocity, ballOffset);
                const DVec3 relativeVelocity =
                    ballContactVelocity - yarnVelocity;
                const DVec3 tangentVelocity = relativeVelocity -
                    normal * dot(relativeVelocity, normal);
                const double slipSpeed = length(tangentVelocity);
                if (!(slipSpeed > 1.0e-10)) {
                    continue;
                }
                const DVec3 tangent = tangentVelocity / slipSpeed;
                const DVec3 ballLever = cross(ballOffset, tangent);
                double denominator = fruit.inverseMass +
                    oracleFruitInverseInertia(fruit) *
                        dot(ballLever, ballLever);
                std::array<DVec3, 2> responses{{
                    tangent * first.inverseMass,
                    tangent * second.inverseMass,
                }};
                const std::array<OracleParticle*, 2> particles{{
                    &first, &second,
                }};
                for (std::size_t index = 0u; index < 2u; ++index) {
                    if (groundEnabled &&
                        particles[index]->position.z <=
                            groundHeight + 1.0e-6 &&
                        responses[index].z < 0.0) {
                        responses[index].z = 0.0;
                    }
                    denominator += dot(tangent, responses[index]) *
                        weights[index] * weights[index];
                }
                if (!(denominator > 0.0)) {
                    continue;
                }
                const double frictionLimit =
                    friction * contact.normalImpulse;
                const double tangentialImpulse = std::min(
                    slipSpeed / denominator, frictionLimit
                );
                if (!(tangentialImpulse > 0.0)) {
                    continue;
                }
                applyOracleFruitImpulse(
                    fruit,
                    tangent * -tangentialImpulse,
                    ballOffset
                );
                first.velocity += responses[0] *
                    (tangentialImpulse * weights[0]);
                second.velocity += responses[1] *
                    (tangentialImpulse * weights[1]);
                recordOracleFriction(
                    state, 1u, tangentialImpulse, frictionLimit
                );
            }
        }
    }
}

void applyOracleFruitPairFriction(
    const InitialState& initial,
    OracleResult& state
) {
    const double friction = initial.config.fruitMaterial.x;
    for (const NumiClothBagGPUBatch& batch : initial.fruitPairBatches) {
        for (std::uint32_t local = 0u; local < batch.control.y; ++local) {
            const OracleFruitPair& pair =
                state.fruitPairs[batch.control.x + local];
            if (!(pair.normalImpulse > 0.0) ||
                !(length(pair.weightedNormal) > 1.0e-10)) {
                continue;
            }
            OracleFruit& first = state.fruits[pair.first];
            OracleFruit& second = state.fruits[pair.second];
            const DVec3 normal = normalized(pair.weightedNormal);
            const DVec3 firstOffset = normal * first.radius;
            const DVec3 secondOffset = normal * -second.radius;
            const DVec3 firstContactVelocity = first.velocity +
                cross(first.angularVelocity, firstOffset);
            const DVec3 secondContactVelocity = second.velocity +
                cross(second.angularVelocity, secondOffset);
            const DVec3 relativeVelocity =
                secondContactVelocity - firstContactVelocity;
            const DVec3 tangentVelocity = relativeVelocity -
                normal * dot(relativeVelocity, normal);
            const double slipSpeed = length(tangentVelocity);
            if (!(slipSpeed > 1.0e-10)) {
                continue;
            }
            const DVec3 tangent = tangentVelocity / slipSpeed;
            const DVec3 firstLever = cross(firstOffset, tangent);
            const DVec3 secondLever = cross(secondOffset, tangent);
            const double denominator =
                first.inverseMass + second.inverseMass +
                oracleFruitInverseInertia(first) *
                    dot(firstLever, firstLever) +
                oracleFruitInverseInertia(second) *
                    dot(secondLever, secondLever);
            if (!(denominator > 0.0)) {
                continue;
            }
            const double frictionLimit = friction * pair.normalImpulse;
            const double tangentialImpulse = std::min(
                slipSpeed / denominator, frictionLimit
            );
            if (!(tangentialImpulse > 0.0)) {
                continue;
            }
            const DVec3 impulseOnSecond = tangent * -tangentialImpulse;
            applyOracleFruitImpulse(second, impulseOnSecond, secondOffset);
            applyOracleFruitImpulse(first, impulseOnSecond * -1.0, firstOffset);
            recordOracleFriction(
                state, 0u, tangentialImpulse, frictionLimit
            );
        }
    }
}

void applyOracleFruitGroundFriction(
    const InitialState& initial,
    OracleResult& state
) {
    if (initial.config.constraintCounts.z == 0u) {
        return;
    }
    const double friction = initial.config.fruitMaterial.y;
    const double rollingResistance = initial.config.fruitMaterial.z;
    const DVec3 normal{0.0, 0.0, 1.0};
    for (OracleFruit& fruit : state.fruits) {
        const double normalImpulse = fruit.groundNormalImpulse;
        if (!(normalImpulse > 0.0)) {
            continue;
        }
        const DVec3 contactOffset{0.0, 0.0, -fruit.radius};
        const DVec3 contactVelocity = fruit.velocity +
            cross(fruit.angularVelocity, contactOffset);
        const DVec3 tangentVelocity = contactVelocity -
            normal * dot(contactVelocity, normal);
        const double slipSpeed = length(tangentVelocity);
        if (slipSpeed > 1.0e-10) {
            const DVec3 tangent = tangentVelocity / slipSpeed;
            const DVec3 lever = cross(contactOffset, tangent);
            const double denominator = fruit.inverseMass +
                oracleFruitInverseInertia(fruit) * dot(lever, lever);
            const double frictionLimit = friction * normalImpulse;
            const double tangentialImpulse = std::min(
                slipSpeed / denominator, frictionLimit
            );
            if (tangentialImpulse > 0.0) {
                applyOracleFruitImpulse(
                    fruit,
                    tangent * -tangentialImpulse,
                    contactOffset
                );
                recordOracleFriction(
                    state, 3u, tangentialImpulse, frictionLimit
                );
            }
        }
        const DVec3 rollingAngularVelocity{
            fruit.angularVelocity.x, fruit.angularVelocity.y, 0.0
        };
        const double rollingSpeed = length(rollingAngularVelocity);
        if (rollingSpeed > 1.0e-12 && rollingResistance > 0.0) {
            const double inverseInertia = oracleFruitInverseInertia(fruit);
            const double requiredAngularImpulse =
                rollingSpeed / inverseInertia;
            const double rollingImpulseLimit =
                rollingResistance * fruit.radius * normalImpulse;
            const double angularImpulse = std::min(
                requiredAngularImpulse, rollingImpulseLimit
            );
            fruit.angularVelocity -= rollingAngularVelocity *
                (angularImpulse * inverseInertia / rollingSpeed);
            ++state.rollingContacts;
            if (rollingImpulseLimit > 0.0) {
                state.maximumRollingResistanceRatio = std::max(
                    state.maximumRollingResistanceRatio,
                    angularImpulse / rollingImpulseLimit
                );
            }
        }
    }
}

void integrateOracleFruitOrientation(
    const InitialState& initial,
    OracleResult& state
) {
    const double halfTimestep =
        0.5 * initial.config.gravityAndTimestep.w;
    for (OracleFruit& fruit : state.fruits) {
        const DVec3 vectorPart{
            fruit.orientation[0],
            fruit.orientation[1],
            fruit.orientation[2],
        };
        const double scalarPart = fruit.orientation[3];
        const DVec3 vectorDerivative =
            fruit.angularVelocity * scalarPart +
            cross(fruit.angularVelocity, vectorPart);
        const double scalarDerivative =
            -dot(fruit.angularVelocity, vectorPart);
        std::array<double, 4> advanced{{
            fruit.orientation[0] + vectorDerivative.x * halfTimestep,
            fruit.orientation[1] + vectorDerivative.y * halfTimestep,
            fruit.orientation[2] + vectorDerivative.z * halfTimestep,
            fruit.orientation[3] + scalarDerivative * halfTimestep,
        }};
        const double magnitude = std::sqrt(
            advanced[0] * advanced[0] +
            advanced[1] * advanced[1] +
            advanced[2] * advanced[2] +
            advanced[3] * advanced[3]
        );
        for (double& component : advanced) {
            component /= magnitude;
        }
        fruit.orientation = advanced;
    }
}

void applyOracleAerodynamics(
    const InitialState& initial,
    OracleResult& state
) {
    const DVec3 airVelocity = d3(initial.config.airVelocityAndDensity);
    const double density = initial.config.airVelocityAndDensity.w;
    const double timestep = initial.config.gravityAndTimestep.w;
    std::vector<DVec3> forces(state.particles.size());
    for (const OracleDistance& distance : state.distances) {
        const OracleParticle& first = state.particles[distance.first];
        const OracleParticle& second = state.particles[distance.second];
        const DVec3 span = second.position - first.position;
        const double spanLength = length(span);
        if (!(spanLength > 1.0e-12)) {
            continue;
        }
        const DVec3 axis = span / spanLength;
        const DVec3 relativeVelocity =
            (first.velocity + second.velocity) * 0.5 - airVelocity;
        const DVec3 axialVelocity =
            axis * dot(relativeVelocity, axis);
        const DVec3 crossflowVelocity =
            relativeVelocity - axialVelocity;
        const double crossflowCoefficient =
            0.5 * density * initial.config.aerodynamicCoefficients.x *
            (2.0 * initial.config.clothMaterial.x * spanLength);
        const double skinCoefficient =
            0.5 * density * initial.config.aerodynamicCoefficients.y *
            (2.0 * std::numbers::pi *
                initial.config.clothMaterial.x * spanLength);
        const DVec3 force = crossflowVelocity * (
            -crossflowCoefficient * length(crossflowVelocity)
        ) + axialVelocity * (-skinCoefficient * length(axialVelocity));
        forces[distance.first] += force * 0.5;
        forces[distance.second] += force * 0.5;
        state.maximumYarnAerodynamicForce = std::max(
            state.maximumYarnAerodynamicForce,
            length(force)
        );
    }
    double dragPower = 0.0;
    double inverseMassWeightedForceSquared = 0.0;
    for (std::size_t index = 0u; index < state.particles.size(); ++index) {
        const OracleParticle& particle = state.particles[index];
        if (!(particle.inverseMass > 0.0)) {
            continue;
        }
        dragPower += dot(
            forces[index], particle.velocity - airVelocity
        );
        inverseMassWeightedForceSquared +=
            particle.inverseMass * dot(forces[index], forces[index]);
    }
    double attenuation = 1.0;
    if (inverseMassWeightedForceSquared > 0.0) {
        attenuation = dragPower < 0.0
            ? 1.0 / (
                1.0 + timestep * inverseMassWeightedForceSquared /
                    -dragPower
            )
            : 0.0;
    }
    for (std::size_t index = 0u; index < state.particles.size(); ++index) {
        OracleParticle& particle = state.particles[index];
        if (particle.inverseMass > 0.0) {
            particle.velocity += forces[index] * (
                attenuation * particle.inverseMass * timestep
            );
        }
    }
    for (OracleFruit& fruit : state.fruits) {
        const DVec3 relativeVelocity = fruit.velocity - airVelocity;
        const double speed = length(relativeVelocity);
        const double translationalCoefficient =
            0.5 * density * initial.config.aerodynamicCoefficients.z *
            std::numbers::pi * fruit.radius * fruit.radius;
        fruit.velocity = airVelocity + relativeVelocity * (
            1.0 / (
                1.0 + fruit.inverseMass * translationalCoefficient *
                    speed * timestep
            )
        );
        const double angularSpeed = length(fruit.angularVelocity);
        const double rotationalCoefficient =
            (3.0 * std::numbers::pi * std::numbers::pi / 8.0) *
            density * initial.config.aerodynamicCoefficients.w *
            std::pow(fruit.radius, 5.0);
        fruit.angularVelocity = fruit.angularVelocity * (
            1.0 / (
                1.0 + oracleFruitInverseInertia(fruit) *
                    rotationalCoefficient * angularSpeed * timestep
            )
        );
        state.maximumFruitAerodynamicForce = std::max(
            state.maximumFruitAerodynamicForce,
            translationalCoefficient * speed * speed
        );
        state.maximumFruitAerodynamicTorque = std::max(
            state.maximumFruitAerodynamicTorque,
            rotationalCoefficient * angularSpeed * angularSpeed
        );
    }
}

void updateOracleReleasedFruit(
    const InitialState& initial,
    OracleResult& state
) {
    const std::uint32_t rimCount = initial.config.mouthControl.x;
    if (rimCount == 0u) {
        return;
    }
    std::vector<DVec3> ring;
    ring.reserve(rimCount);
    DVec3 center{};
    for (std::uint32_t index = 0u; index < rimCount; ++index) {
        const DVec3 position = state.particles[
            initial.mouthRimParticles[index]
        ].position;
        ring.push_back(position);
        center += position;
    }
    center = center / static_cast<double>(rimCount);
    DVec3 areaNormal{};
    for (std::uint32_t index = 0u; index < rimCount; ++index) {
        areaNormal += cross(
            ring[index] - center,
            ring[(index + 1u) % rimCount] - center
        );
    }
    if (!(length(areaNormal) > 1.0e-8)) {
        return;
    }
    DVec3 normal = normalized(areaNormal);
    const DVec3 interiorDirection = center - state.particles[
        initial.config.mouthControl.y
    ].position;
    if (dot(normal, interiorDirection) < 0.0) {
        normal = normal * -1.0;
    }
    DVec3 tangent = ring[0] - center;
    tangent -= normal * dot(tangent, normal);
    if (!(length(tangent) > 1.0e-8)) {
        tangent = cross(
            normal,
            std::abs(normal.z) < 0.9
                ? DVec3{0.0, 0.0, 1.0}
                : DVec3{1.0, 0.0, 0.0}
        );
    }
    tangent = normalized(tangent);
    const DVec3 bitangent = normalized(cross(normal, tangent));
    const auto projectedInside = [&] (
        const DVec3 point,
        const double edgeClearance
    ) {
        const DVec3 relativePoint = point - center;
        const double pointX = dot(relativePoint, tangent);
        const double pointY = dot(relativePoint, bitangent);
        bool inside = false;
        double minimumEdgeDistanceSquared =
            std::numeric_limits<double>::infinity();
        for (std::uint32_t first = 0u, second = rimCount - 1u;
             first < rimCount;
             second = first++) {
            const DVec3 firstRelative = ring[first] - center;
            const DVec3 secondRelative = ring[second] - center;
            const double firstX = dot(firstRelative, tangent);
            const double firstY = dot(firstRelative, bitangent);
            const double secondX = dot(secondRelative, tangent);
            const double secondY = dot(secondRelative, bitangent);
            const double edgeX = secondX - firstX;
            const double edgeY = secondY - firstY;
            const double edgeLengthSquared =
                edgeX * edgeX + edgeY * edgeY;
            const double fraction = edgeLengthSquared > 1.0e-20
                ? std::clamp(
                    ((pointX - firstX) * edgeX +
                     (pointY - firstY) * edgeY) / edgeLengthSquared,
                    0.0,
                    1.0
                )
                : 0.0;
            const double separationX =
                pointX - (firstX + fraction * edgeX);
            const double separationY =
                pointY - (firstY + fraction * edgeY);
            minimumEdgeDistanceSquared = std::min(
                minimumEdgeDistanceSquared,
                separationX * separationX + separationY * separationY
            );
            const bool crosses = (firstY > pointY) != (secondY > pointY);
            if (crosses) {
                const double crossingX = firstX +
                    (secondX - firstX) *
                        (pointY - firstY) / (secondY - firstY);
                if (pointX < crossingX) {
                    inside = !inside;
                }
            }
        }
        return inside || minimumEdgeDistanceSquared <=
            edgeClearance * edgeClearance;
    };
    const auto minimumRimDistance = [&] (const DVec3 point) {
        double minimumDistanceSquared =
            std::numeric_limits<double>::infinity();
        for (std::uint32_t first = 0u; first < rimCount; ++first) {
            const DVec3 start = ring[first];
            const DVec3 edge = ring[(first + 1u) % rimCount] - start;
            const double edgeLengthSquared = dot(edge, edge);
            const double fraction = edgeLengthSquared > 1.0e-20
                ? std::clamp(
                    dot(point - start, edge) / edgeLengthSquared,
                    0.0,
                    1.0
                )
                : 0.0;
            const DVec3 separation = point - (start + edge * fraction);
            minimumDistanceSquared = std::min(
                minimumDistanceSquared, dot(separation, separation)
            );
        }
        return std::sqrt(minimumDistanceSquared);
    };
    for (std::size_t index = 0u; index < state.fruits.size(); ++index) {
        const OracleFruit& fruit = state.fruits[index];
        const double requiredClearance =
            fruit.radius + initial.config.clothMaterial.x;
        const bool insideProjection = projectedInside(fruit.position, 0.0);
        const bool nearProjection = projectedInside(
            fruit.position, requiredClearance
        );
        const double outwardDistance = dot(
            fruit.position - center, normal
        );
        const std::uint32_t mask = 1u << index;
        if (insideProjection && outwardDistance > 0.0) {
            state.mouthCandidateMask |= mask;
        }
        const double axialClearance =
            outwardDistance - requiredClearance;
        const double rimClearance =
            minimumRimDistance(fruit.position) - requiredClearance;
        const double hysteresis = initial.config.mouthMaterial.x;
        const bool fullyClearThroughCap = nearProjection &&
            axialClearance > hysteresis;
        const bool fullyClearAroundRim =
            (state.mouthCandidateMask & mask) != 0u &&
            !insideProjection && outwardDistance > 0.0 &&
            rimClearance > hysteresis;
        const double clearance = fullyClearThroughCap
            ? axialClearance
            : (fullyClearAroundRim ? rimClearance : axialClearance);
        state.maximumMouthClearanceByFruit[index] = std::max(
            state.maximumMouthClearanceByFruit[index], clearance
        );
        if (fullyClearThroughCap || fullyClearAroundRim) {
            state.releasedMask |= mask;
        }
    }
}

OracleResult runOracle(
    const InitialState& initial,
    const std::uint32_t iterations,
    const std::uint32_t strainSweeps
) {
    @autoreleasepool {
    OracleResult result;
    result.mouthCandidateMask = initial.initialMouthCandidateMask;
    result.releasedMask = initial.initialReleasedMask;
    result.maximumMouthClearanceByFruit.assign(
        initial.fruits.size(), 0.0
    );
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
    result.fruits.reserve(initial.fruits.size());
    for (const auto& source : initial.fruits) {
        result.fruits.push_back({
            d3(source.positionAndInverseMass),
            d3(source.previousAndRadius),
            d3(source.velocityAndGroundImpulse),
            d3(source.angularVelocity),
            {{
                source.orientation.x,
                source.orientation.y,
                source.orientation.z,
                source.orientation.w,
            }},
            static_cast<double>(source.positionAndInverseMass.w),
            static_cast<double>(source.previousAndRadius.w),
            0.0,
        });
    }
    result.fruitPairs.reserve(initial.fruitPairs.size());
    for (const auto& source : initial.fruitPairs) {
        result.fruitPairs.push_back({
            source.fruitsAndColor.x,
            source.fruitsAndColor.y,
            {},
            0.0,
        });
    }
    const double timestep = initial.config.gravityAndTimestep.w;
    const DVec3 gravity = d3(initial.config.gravityAndTimestep);
    const DVec3 gripTarget = d3(initial.config.gripTargetAndActive);
    for (OracleParticle& particle : result.particles) {
        particle.previous = particle.position;
        particle.groundNormalImpulse = 0.0;
        if (particle.inverseMass > 0.0) {
            particle.velocity += gravity * timestep;
            particle.predictedVerticalVelocity = particle.velocity.z;
        }
    }
    for (OracleFruit& fruit : result.fruits) {
        fruit.previous = fruit.position;
        fruit.velocity += gravity * timestep;
        fruit.groundNormalImpulse = 0.0;
    }
    applyOracleAerodynamics(initial, result);
    for (OracleParticle& particle : result.particles) {
        if (particle.inverseMass > 0.0) {
            particle.predictedVerticalVelocity = particle.velocity.z;
            particle.position += particle.velocity * timestep;
        }
    }
    for (OracleFruit& fruit : result.fruits) {
        fruit.position += fruit.velocity * timestep;
    }
    solveOracleSelfContact(initial, result, true);
    result.yarnContacts = buildOracleYarnContacts(initial, result);
    solveOracleYarnBatches(initial, result, true);
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
        if (initial.config.gripTargetAndActive.w > 0.0f) {
            for (OracleGrip& grip : result.grips) {
                OracleParticle& particle = result.particles[grip.particle];
                const double alpha = grip.compliance / (timestep * timestep);
                const double denominator = particle.inverseMass + alpha;
                if (!(denominator > 0.0)) {
                    continue;
                }
                const DVec3 value = particle.position - (
                    gripTarget + rotateGripOffset(
                        initial.config.gripOrientation,
                        grip.offset
                    )
                );
                const DVec3 deltaLambda =
                    (value * -1.0 - grip.lambda * alpha) / denominator;
                grip.lambda += deltaLambda;
                particle.position += deltaLambda * particle.inverseMass;
            }
        }
        for (const NumiClothBagGPUBatch& batch : initial.fruitPairBatches) {
            for (std::uint32_t local = 0u; local < batch.control.y; ++local) {
                OracleFruitPair& pair =
                    result.fruitPairs[batch.control.x + local];
                OracleFruit& first = result.fruits[pair.first];
                OracleFruit& second = result.fruits[pair.second];
                const DVec3 difference = second.position - first.position;
                const double currentLength = length(difference);
                const double target = first.radius + second.radius;
                if (!(currentLength < target) ||
                    !(currentLength > 1.0e-12)) {
                    continue;
                }
                const double denominator =
                    first.inverseMass + second.inverseMass;
                if (!(denominator > 0.0)) {
                    continue;
                }
                const double lambda =
                    (target - currentLength) / denominator;
                const DVec3 normal = difference / currentLength;
                const DVec3 correction = normal * lambda;
                first.position -= correction * first.inverseMass;
                second.position += correction * second.inverseMass;
                const double impulseMagnitude = lambda / timestep;
                pair.weightedNormal += normal * impulseMagnitude;
                pair.normalImpulse += impulseMagnitude;
            }
        }
        solveOracleYarnBatches(initial, result, false);
        solveOracleSelfContact(initial, result, false);
        if (initial.config.constraintCounts.z != 0u) {
            const double clothRadius = initial.config.clothMaterial.x;
            for (OracleParticle& particle : result.particles) {
                particle.position.z = std::max(
                    particle.position.z, clothRadius
                );
            }
            for (OracleFruit& fruit : result.fruits) {
                const double penetration = fruit.radius - fruit.position.z;
                if (penetration > 0.0) {
                    fruit.groundNormalImpulse += penetration /
                        (fruit.inverseMass * timestep);
                    fruit.position.z = fruit.radius;
                }
            }
        }
    }
    solveOracleStrainLimits(initial, result, strainSweeps);
    const auto solveFinalContacts = [&] {
        solveOracleFruitPairs(initial, result);
        solveOracleYarnBatches(initial, result, false);
        solveOracleGround(initial, result);
    };
    const auto solveEndpointSelfAndStrain = [&] {
        solveOracleSelfContact(initial, result, false);
        solveOracleStrainLimits(initial, result, strainSweeps);
    };
    if (strainSweeps != 0u) {
        for (std::uint32_t pass = 0u;
             pass < kReconciliationPasses;
             ++pass) {
            solveOracleFruitPairs(initial, result);
            solveOracleYarnBatches(initial, result, false);
            solveOracleStrainLimits(initial, result, strainSweeps);
            solveOracleGround(initial, result);
        }
        solveEndpointSelfAndStrain();
        for (std::uint32_t pass = 0u; pass < kFinalContactPasses; ++pass) {
            solveFinalContacts();
        }
        for (std::uint32_t certificate = 0u;
             certificate < kCertificatePasses;
             ++certificate) {
            solveEndpointSelfAndStrain();
            for (std::uint32_t pass = 0u;
                 pass < kFinalContactPasses;
                 ++pass) {
                solveFinalContacts();
            }
        }
    }
    for (OracleParticle& particle : result.particles) {
        particle.velocity =
            (particle.position - particle.previous) / timestep;
        particle.groundNormalImpulse = 0.0;
        if (initial.config.constraintCounts.z != 0u &&
            particle.inverseMass > 0.0 &&
            particle.position.z <=
                initial.config.clothMaterial.x + 1.0e-6) {
            particle.groundNormalImpulse = std::max(
                0.0,
                (particle.velocity.z - particle.predictedVerticalVelocity) /
                    particle.inverseMass
            );
        }
    }
    for (OracleFruit& fruit : result.fruits) {
        fruit.velocity = (fruit.position - fruit.previous) / timestep;
    }
    applyOracleClothGroundFriction(initial, result);
    applyOracleSelfFriction(initial, result);
    applyOracleYarnFriction(initial, result);
    applyOracleFruitPairFriction(initial, result);
    applyOracleFruitGroundFriction(initial, result);
    integrateOracleFruitOrientation(initial, result);
    updateOracleReleasedFruit(initial, result);
    result.yarnContacts = buildOracleYarnContacts(
        initial, result, &result.yarnContacts
    );
        return result;
    }
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
    std::vector<NumiClothBagGPUFruit> fruits;
    std::vector<NumiClothBagGPUFruitPair> fruitPairs;
    std::vector<NumiClothBagGPUYarnContact> yarnContacts;
    NumiClothBagGPUSelfStatus selfStatus{};
    NumiClothBagGPUFrictionStatus frictionStatus{};
    NumiClothBagGPUAerodynamicsStatus aerodynamicsStatus{};
    NumiClothBagGPUReleaseStatus releaseStatus{};
    std::vector<std::uint32_t> maximumMouthClearanceBits;
    std::uint32_t failure{};
    double seconds{};
};

struct Pipelines {
    id<MTLComputePipelineState> prepareTrajectorySubstep;
    id<MTLComputePipelineState> begin;
    id<MTLComputePipelineState> updateGripAttachment;
    id<MTLComputePipelineState> yarnAerodynamicsClear;
    id<MTLComputePipelineState> yarnAerodynamicsAccumulate;
    id<MTLComputePipelineState> yarnAerodynamicsReduce;
    id<MTLComputePipelineState> yarnAerodynamicsApply;
    id<MTLComputePipelineState> fruitAerodynamics;
    id<MTLComputePipelineState> advance;
    id<MTLComputePipelineState> distance;
    id<MTLComputePipelineState> knot;
    id<MTLComputePipelineState> bend;
    id<MTLComputePipelineState> grip;
    id<MTLComputePipelineState> fruitPair;
    id<MTLComputePipelineState> ground;
    id<MTLComputePipelineState> strain;
    id<MTLComputePipelineState> yarnSolve;
    id<MTLComputePipelineState> yarnContacts;
    id<MTLComputePipelineState> selfCells;
    id<MTLComputePipelineState> selfCellSort;
    id<MTLComputePipelineState> selfSpatialDetect;
    id<MTLComputePipelineState> selfBatchCount;
    id<MTLComputePipelineState> selfBatchCompact;
    id<MTLComputePipelineState> selfDetect;
    id<MTLComputePipelineState> selfContact;
    id<MTLComputePipelineState> selfImpulseSort;
    id<MTLComputePipelineState> selfImpulseClear;
    id<MTLComputePipelineState> selfImpulseAggregate;
    id<MTLComputePipelineState> selfFrictionBatchCount;
    id<MTLComputePipelineState> selfFriction;
    id<MTLComputePipelineState> finalize;
    id<MTLComputePipelineState> finalizeFruit;
    id<MTLComputePipelineState> fruitOrientation;
    id<MTLComputePipelineState> clothGroundFriction;
    id<MTLComputePipelineState> yarnFriction;
    id<MTLComputePipelineState> fruitPairFriction;
    id<MTLComputePipelineState> fruitGroundFriction;
    id<MTLComputePipelineState> release;
};

GPUResult runGPU(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    const Pipelines& pipelines,
    const InitialState& initial,
    const std::uint32_t iterations,
    const std::uint32_t strainSweeps,
    const std::vector<NumiClothBagGPUConfig>& requestedTrajectoryConfigs = {}
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
    const auto makeZeroed = [device](const NSUInteger requestedBytes) {
        const NSUInteger bytes = std::max<NSUInteger>(requestedBytes, 1u);
        id<MTLBuffer> buffer = [device
            newBufferWithLength:bytes
                        options:MTLResourceStorageModeShared];
        if (buffer != nil) {
            std::memset(buffer.contents, 0, bytes);
        }
        return buffer;
    };
    const std::vector<NumiClothBagGPUConfig> trajectoryConfigs =
        requestedTrajectoryConfigs.empty()
        ? std::vector<NumiClothBagGPUConfig>{initial.config}
        : requestedTrajectoryConfigs;
    if (trajectoryConfigs.size() >
        std::numeric_limits<std::uint32_t>::max()) {
        throw std::logic_error("Metal cloth trajectory is too long");
    }
    for (const NumiClothBagGPUConfig& config : trajectoryConfigs) {
        if (config.control.x != NUMI_CLOTH_BAG_GPU_ABI_VERSION ||
            config.control.y != initial.config.control.y ||
            config.control.z != initial.config.control.z ||
            config.control.w != initial.config.control.w ||
            config.gripControl.z != initial.config.gripControl.z ||
            config.gripControl.w != initial.config.gripControl.w ||
            std::memcmp(
                &config.constraintCounts,
                &initial.config.constraintCounts,
                sizeof(config.constraintCounts)
            ) != 0 ||
            std::memcmp(
                &config.contactCounts,
                &initial.config.contactCounts,
                sizeof(config.contactCounts)
            ) != 0 ||
            std::memcmp(
                &config.mouthControl,
                &initial.config.mouthControl,
                sizeof(config.mouthControl)
            ) != 0) {
            throw std::logic_error(
                "Metal cloth trajectory changes fixed topology"
            );
        }
        if (config.gripControl.y == 1u &&
            initial.cuffParticles.size() !=
                2u * static_cast<std::size_t>(config.gripControl.z)) {
            throw std::logic_error(
                "Metal cloth trajectory has incomplete cuff topology"
            );
        }
    }
    if (!std::all_of(
            initial.cuffParticles.begin(),
            initial.cuffParticles.end(),
            [&](const std::uint32_t particle) {
                return particle < initial.particles.size();
            }
        )) {
        throw std::logic_error("Metal cloth cuff topology is out of range");
    }
    id<MTLBuffer> configBuffer = [device
        newBufferWithBytes:&trajectoryConfigs.front()
                   length:sizeof(trajectoryConfigs.front())
                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> trajectoryConfigBuffer = makeBytes(trajectoryConfigs);
    id<MTLBuffer> particleBuffer = makeBytes(initial.particles);
    id<MTLBuffer> distanceBuffer = makeBytes(initial.distances);
    id<MTLBuffer> gripBuffer = makeBytes(initial.grips);
    id<MTLBuffer> knotBuffer = makeBytes(initial.knots);
    id<MTLBuffer> bendBuffer = makeBytes(initial.bends);
    id<MTLBuffer> fruitBuffer = makeBytes(initial.fruits);
    id<MTLBuffer> fruitPairBuffer = makeBytes(initial.fruitPairs);
    id<MTLBuffer> yarnContactBuffer = makeBytes(initial.yarnContacts);
    id<MTLBuffer> yarnAerodynamicForceBuffer = makeZeroed(
        initial.particles.size() * sizeof(mr_float4)
    );
    id<MTLBuffer> yarnAerodynamicReductionBuffer = makeZeroed(
        sizeof(mr_float4)
    );
    id<MTLBuffer> selfPairBuffer = makeBytes(initial.selfPairs);
    id<MTLBuffer> selfBatchBuffer = makeBytes(initial.selfBatches);
    id<MTLBuffer> selfPairLookupBuffer = makeBytes(initial.selfPairLookup);
    id<MTLBuffer> mouthRimBuffer = makeBytes(initial.mouthRimParticles);
    id<MTLBuffer> cuffParticleBuffer = makeBytes(initial.cuffParticles);
    id<MTLBuffer> selfCellBuffer = makeZeroed(
        4096u * sizeof(std::uint64_t)
    );
    id<MTLBuffer> selfActiveFlagBuffer = makeZeroed(
        initial.selfPairs.size() * sizeof(std::uint32_t)
    );
    const NSUInteger selfBatchBytes =
        initial.selfBatches.size() * sizeof(std::uint32_t);
    id<MTLBuffer> selfActiveBatchCountBuffer = makeZeroed(selfBatchBytes);
    id<MTLBuffer> selfActiveBatchIndexBuffer = makeZeroed(selfBatchBytes);
    id<MTLBuffer> activeSelfBatchCountBuffer = makeZeroed(
        sizeof(std::uint32_t)
    );
    id<MTLBuffer> selfImpulseRecordBuffer = makeZeroed(
        NUMI_CLOTH_BAG_GPU_SELF_IMPULSE_CAPACITY *
            sizeof(NumiClothBagGPUSelfImpulse)
    );
    id<MTLBuffer> selfImpulseAggregateBuffer = makeZeroed(
        NUMI_CLOTH_BAG_GPU_SELF_IMPULSE_CAPACITY *
            sizeof(NumiClothBagGPUSelfImpulse)
    );
    const std::vector<std::uint64_t> emptySelfImpulseKeys(
        NUMI_CLOTH_BAG_GPU_SELF_IMPULSE_CAPACITY,
        std::numeric_limits<std::uint64_t>::max()
    );
    id<MTLBuffer> selfImpulseKeyBuffer = makeBytes(emptySelfImpulseKeys);
    id<MTLBuffer> selfImpulseCountBuffer = makeZeroed(
        sizeof(std::uint32_t)
    );
    NumiClothBagGPUSelfStatus zeroSelfStatus{};
    id<MTLBuffer> selfStatusBuffer = [device
        newBufferWithBytes:&zeroSelfStatus
                   length:sizeof(zeroSelfStatus)
                  options:MTLResourceStorageModeShared];
    NumiClothBagGPUFrictionStatus zeroFrictionStatus{};
    id<MTLBuffer> frictionStatusBuffer = [device
        newBufferWithBytes:&zeroFrictionStatus
                   length:sizeof(zeroFrictionStatus)
                  options:MTLResourceStorageModeShared];
    NumiClothBagGPUAerodynamicsStatus zeroAerodynamicsStatus{};
    id<MTLBuffer> aerodynamicsStatusBuffer = [device
        newBufferWithBytes:&zeroAerodynamicsStatus
                   length:sizeof(zeroAerodynamicsStatus)
                  options:MTLResourceStorageModeShared];
    NumiClothBagGPUReleaseStatus initialReleaseStatus{};
    initialReleaseStatus.masks = u4(
        initial.initialMouthCandidateMask,
        initial.initialReleasedMask,
        0u,
        0u
    );
    id<MTLBuffer> releaseStatusBuffer = [device
        newBufferWithBytes:&initialReleaseStatus
                   length:sizeof(initialReleaseStatus)
                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> maximumMouthClearanceBuffer = makeZeroed(
        initial.fruits.size() * sizeof(std::uint32_t)
    );
    std::uint32_t zero = 0u;
    id<MTLBuffer> failureBuffer = [device
        newBufferWithBytes:&zero
                   length:sizeof(zero)
                  options:MTLResourceStorageModeShared];
    if (configBuffer == nil || trajectoryConfigBuffer == nil ||
        particleBuffer == nil ||
        distanceBuffer == nil || gripBuffer == nil ||
        knotBuffer == nil || bendBuffer == nil ||
        fruitBuffer == nil || fruitPairBuffer == nil ||
        yarnContactBuffer == nil || yarnAerodynamicForceBuffer == nil ||
        yarnAerodynamicReductionBuffer == nil ||
        selfPairBuffer == nil || selfBatchBuffer == nil ||
        selfPairLookupBuffer == nil || mouthRimBuffer == nil ||
        cuffParticleBuffer == nil ||
        selfCellBuffer == nil ||
        selfActiveFlagBuffer == nil || selfActiveBatchCountBuffer == nil ||
        selfActiveBatchIndexBuffer == nil ||
        activeSelfBatchCountBuffer == nil ||
        selfImpulseRecordBuffer == nil ||
        selfImpulseAggregateBuffer == nil ||
        selfImpulseKeyBuffer == nil || selfImpulseCountBuffer == nil ||
        selfStatusBuffer == nil ||
        frictionStatusBuffer == nil ||
        aerodynamicsStatusBuffer == nil ||
        releaseStatusBuffer == nil ||
        maximumMouthClearanceBuffer == nil ||
        failureBuffer == nil) {
        throw std::runtime_error("failed to allocate Metal cloth buffers");
    }

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (commandBuffer == nil || encoder == nil) {
        throw std::runtime_error("failed to create Metal cloth encoder");
    }
    std::uint32_t selfContactEpoch = 0u;
    for (std::uint32_t trajectorySubstep = 0u;
         trajectorySubstep < trajectoryConfigs.size();
         ++trajectorySubstep) {
    [encoder setComputePipelineState:pipelines.prepareTrajectorySubstep];
    [encoder setBuffer:configBuffer offset:0 atIndex:0];
    [encoder setBuffer:trajectoryConfigBuffer offset:0 atIndex:1];
    [encoder setBytes:&trajectorySubstep
                length:sizeof(trajectorySubstep)
               atIndex:2];
    [encoder setBuffer:selfImpulseKeyBuffer offset:0 atIndex:3];
    [encoder setBuffer:selfImpulseCountBuffer offset:0 atIndex:4];
    [encoder setBuffer:failureBuffer offset:0 atIndex:5];
    dispatch(
        encoder,
        pipelines.prepareTrajectorySubstep,
        NUMI_CLOTH_BAG_GPU_SELF_IMPULSE_CAPACITY
    );
    [encoder setBuffer:configBuffer offset:0 atIndex:0];
    [encoder setBuffer:particleBuffer offset:0 atIndex:1];
    [encoder setBuffer:distanceBuffer offset:0 atIndex:2];
    [encoder setBuffer:gripBuffer offset:0 atIndex:3];
    [encoder setBuffer:knotBuffer offset:0 atIndex:4];
    [encoder setBuffer:bendBuffer offset:0 atIndex:5];
    [encoder setBuffer:fruitBuffer offset:0 atIndex:6];
    [encoder setBuffer:fruitPairBuffer offset:0 atIndex:7];
    [encoder setBuffer:failureBuffer offset:0 atIndex:8];
    [encoder setBuffer:yarnContactBuffer offset:0 atIndex:9];
    dispatch(
        encoder,
        pipelines.begin,
        std::max({
            initial.particles.size(),
            initial.distances.size(),
            initial.grips.size(),
            initial.knots.size(),
            initial.bends.size(),
            initial.fruits.size(),
            initial.fruitPairs.size(),
        })
    );
    [encoder setComputePipelineState:pipelines.updateGripAttachment];
    [encoder setBuffer:configBuffer offset:0 atIndex:0];
    [encoder setBuffer:particleBuffer offset:0 atIndex:1];
    [encoder setBuffer:gripBuffer offset:0 atIndex:2];
    [encoder setBuffer:failureBuffer offset:0 atIndex:3];
    [encoder setBuffer:cuffParticleBuffer offset:0 atIndex:4];
    dispatch(
        encoder,
        pipelines.updateGripAttachment,
        initial.grips.size()
    );
    [encoder setComputePipelineState:pipelines.yarnAerodynamicsClear];
    [encoder setBuffer:configBuffer offset:0 atIndex:0];
    [encoder setBuffer:yarnAerodynamicForceBuffer offset:0 atIndex:1];
    [encoder setBuffer:failureBuffer offset:0 atIndex:2];
    dispatch(
        encoder,
        pipelines.yarnAerodynamicsClear,
        initial.particles.size()
    );
    for (const NumiClothBagGPUBatch& batch : initial.distanceBatches) {
        [encoder setComputePipelineState:pipelines.yarnAerodynamicsAccumulate];
        [encoder setBuffer:configBuffer offset:0 atIndex:0];
        [encoder setBuffer:particleBuffer offset:0 atIndex:1];
        [encoder setBuffer:distanceBuffer offset:0 atIndex:2];
        [encoder setBytes:&batch length:sizeof(batch) atIndex:3];
        [encoder setBuffer:yarnAerodynamicForceBuffer offset:0 atIndex:4];
        [encoder setBuffer:aerodynamicsStatusBuffer offset:0 atIndex:5];
        [encoder setBuffer:failureBuffer offset:0 atIndex:6];
        dispatch(
            encoder,
            pipelines.yarnAerodynamicsAccumulate,
            batch.control.y
        );
    }
    [encoder setComputePipelineState:pipelines.yarnAerodynamicsReduce];
    [encoder setBuffer:configBuffer offset:0 atIndex:0];
    [encoder setBuffer:particleBuffer offset:0 atIndex:1];
    [encoder setBuffer:yarnAerodynamicForceBuffer offset:0 atIndex:2];
    [encoder setBuffer:yarnAerodynamicReductionBuffer offset:0 atIndex:3];
    [encoder setBuffer:failureBuffer offset:0 atIndex:4];
    dispatch(encoder, pipelines.yarnAerodynamicsReduce, 256u);
    [encoder setComputePipelineState:pipelines.yarnAerodynamicsApply];
    [encoder setBuffer:configBuffer offset:0 atIndex:0];
    [encoder setBuffer:particleBuffer offset:0 atIndex:1];
    [encoder setBuffer:yarnAerodynamicForceBuffer offset:0 atIndex:2];
    [encoder setBuffer:yarnAerodynamicReductionBuffer offset:0 atIndex:3];
    [encoder setBuffer:failureBuffer offset:0 atIndex:4];
    dispatch(
        encoder,
        pipelines.yarnAerodynamicsApply,
        initial.particles.size()
    );
    [encoder setComputePipelineState:pipelines.fruitAerodynamics];
    [encoder setBuffer:configBuffer offset:0 atIndex:0];
    [encoder setBuffer:fruitBuffer offset:0 atIndex:1];
    [encoder setBuffer:aerodynamicsStatusBuffer offset:0 atIndex:2];
    [encoder setBuffer:failureBuffer offset:0 atIndex:3];
    dispatch(
        encoder,
        pipelines.fruitAerodynamics,
        initial.fruits.size()
    );
    [encoder setComputePipelineState:pipelines.advance];
    [encoder setBuffer:configBuffer offset:0 atIndex:0];
    [encoder setBuffer:particleBuffer offset:0 atIndex:1];
    [encoder setBuffer:fruitBuffer offset:0 atIndex:2];
    [encoder setBuffer:failureBuffer offset:0 atIndex:3];
    dispatch(
        encoder,
        pipelines.advance,
        std::max(initial.particles.size(), initial.fruits.size())
    );

    const auto buildYarnContacts = [&] {
        [encoder setComputePipelineState:pipelines.yarnContacts];
        [encoder setBuffer:configBuffer offset:0 atIndex:0];
        [encoder setBuffer:particleBuffer offset:0 atIndex:1];
        [encoder setBuffer:distanceBuffer offset:0 atIndex:2];
        [encoder setBuffer:fruitBuffer offset:0 atIndex:3];
        [encoder setBuffer:yarnContactBuffer offset:0 atIndex:4];
        [encoder setBuffer:failureBuffer offset:0 atIndex:5];
        dispatch(
            encoder,
            pipelines.yarnContacts,
            initial.yarnContacts.size()
        );
    };
    const auto solveYarnBatches = [&](const std::uint32_t mode) {
        for (NumiClothBagGPUBatch batch : initial.distanceBatches) {
            batch.control.w = mode;
            [encoder setComputePipelineState:pipelines.yarnSolve];
            [encoder setBuffer:configBuffer offset:0 atIndex:0];
            [encoder setBuffer:particleBuffer offset:0 atIndex:1];
            [encoder setBuffer:distanceBuffer offset:0 atIndex:2];
            [encoder setBuffer:fruitBuffer offset:0 atIndex:3];
            [encoder setBuffer:yarnContactBuffer offset:0 atIndex:4];
            [encoder setBytes:&batch length:sizeof(batch) atIndex:5];
            [encoder setBuffer:failureBuffer offset:0 atIndex:6];
            dispatch(encoder, pipelines.yarnSolve, initial.fruits.size());
        }
    };
    const auto solveSelfContact = [&](const std::uint32_t mode) {
        if (initial.selfBatches.empty()) {
            return;
        }
        const std::uint32_t epoch = ++selfContactEpoch;
        const NSUInteger detectionWidth = std::min<NSUInteger>(
            pipelines.selfDetect.maxTotalThreadsPerThreadgroup, 256u
        );
        if (initial.maximumSelfBatchSize > detectionWidth) {
            throw std::runtime_error(
                "Metal self-contact detection width is below batch capacity"
            );
        }
        if (mode == 1u) {
            [encoder setComputePipelineState:pipelines.selfDetect];
            [encoder setBuffer:configBuffer offset:0 atIndex:0];
            [encoder setBuffer:particleBuffer offset:0 atIndex:1];
            [encoder setBuffer:selfPairBuffer offset:0 atIndex:2];
            [encoder setBuffer:selfBatchBuffer offset:0 atIndex:3];
            [encoder setBuffer:selfActiveFlagBuffer offset:0 atIndex:4];
            [encoder setBuffer:selfActiveBatchCountBuffer offset:0 atIndex:5];
            [encoder setBuffer:failureBuffer offset:0 atIndex:6];
            [encoder setBytes:&mode length:sizeof(mode) atIndex:7];
            [encoder setBuffer:distanceBuffer offset:0 atIndex:8];
            [encoder setBytes:&epoch length:sizeof(epoch) atIndex:9];
            [encoder dispatchThreadgroups:MTLSizeMake(
                initial.selfBatches.size(), 1u, 1u
            ) threadsPerThreadgroup:MTLSizeMake(detectionWidth, 1u, 1u)];
        } else {
            if (initial.selfPairLookup.size() !=
                initial.distances.size() *
                    (initial.distances.size() - 1u) / 2u) {
                throw std::runtime_error(
                    "Metal self-contact spatial lookup is incomplete"
                );
            }
            [encoder setComputePipelineState:pipelines.selfCells];
            [encoder setBuffer:configBuffer offset:0 atIndex:0];
            [encoder setBuffer:particleBuffer offset:0 atIndex:1];
            [encoder setBuffer:distanceBuffer offset:0 atIndex:2];
            [encoder setBuffer:selfCellBuffer offset:0 atIndex:3];
            [encoder setBuffer:failureBuffer offset:0 atIndex:4];
            dispatch(encoder, pipelines.selfCells, 4096u);
            [encoder setComputePipelineState:pipelines.selfCellSort];
            [encoder setBuffer:selfCellBuffer offset:0 atIndex:0];
            dispatch(encoder, pipelines.selfCellSort, 256u);
            [encoder setComputePipelineState:pipelines.selfSpatialDetect];
            [encoder setBuffer:configBuffer offset:0 atIndex:0];
            [encoder setBuffer:particleBuffer offset:0 atIndex:1];
            [encoder setBuffer:distanceBuffer offset:0 atIndex:2];
            [encoder setBuffer:selfCellBuffer offset:0 atIndex:3];
            [encoder setBuffer:selfPairLookupBuffer offset:0 atIndex:4];
            [encoder setBuffer:selfActiveFlagBuffer offset:0 atIndex:5];
            [encoder setBuffer:failureBuffer offset:0 atIndex:6];
            [encoder setBytes:&epoch length:sizeof(epoch) atIndex:7];
            dispatch(
                encoder,
                pipelines.selfSpatialDetect,
                initial.distances.size()
            );
            [encoder setComputePipelineState:pipelines.selfBatchCount];
            [encoder setBuffer:configBuffer offset:0 atIndex:0];
            [encoder setBuffer:selfBatchBuffer offset:0 atIndex:1];
            [encoder setBuffer:selfActiveFlagBuffer offset:0 atIndex:2];
            [encoder setBuffer:selfActiveBatchCountBuffer offset:0 atIndex:3];
            [encoder setBuffer:failureBuffer offset:0 atIndex:4];
            [encoder setBytes:&epoch length:sizeof(epoch) atIndex:5];
            dispatch(
                encoder,
                pipelines.selfBatchCount,
                initial.selfBatches.size()
            );
        }
        [encoder setComputePipelineState:pipelines.selfBatchCompact];
        [encoder setBuffer:configBuffer offset:0 atIndex:0];
        [encoder setBuffer:selfActiveBatchCountBuffer offset:0 atIndex:1];
        [encoder setBuffer:selfActiveBatchIndexBuffer offset:0 atIndex:2];
        [encoder setBuffer:activeSelfBatchCountBuffer offset:0 atIndex:3];
        dispatch(encoder, pipelines.selfBatchCompact, 256u);
        [encoder setComputePipelineState:pipelines.selfContact];
        [encoder setBuffer:configBuffer offset:0 atIndex:0];
        [encoder setBuffer:particleBuffer offset:0 atIndex:1];
        [encoder setBuffer:selfPairBuffer offset:0 atIndex:2];
        [encoder setBuffer:selfBatchBuffer offset:0 atIndex:3];
        [encoder setBuffer:selfActiveFlagBuffer offset:0 atIndex:4];
        [encoder setBuffer:selfActiveBatchIndexBuffer offset:0 atIndex:5];
        [encoder setBuffer:activeSelfBatchCountBuffer offset:0 atIndex:6];
        [encoder setBuffer:selfStatusBuffer offset:0 atIndex:7];
        [encoder setBuffer:failureBuffer offset:0 atIndex:8];
        [encoder setBytes:&mode length:sizeof(mode) atIndex:9];
        [encoder setBytes:&epoch length:sizeof(epoch) atIndex:10];
        [encoder setBuffer:distanceBuffer offset:0 atIndex:11];
        [encoder setBuffer:selfImpulseRecordBuffer offset:0 atIndex:12];
        [encoder setBuffer:selfImpulseKeyBuffer offset:0 atIndex:13];
        [encoder setBuffer:selfImpulseCountBuffer offset:0 atIndex:14];
        dispatch(
            encoder,
            pipelines.selfContact,
            initial.maximumSelfBatchSize
        );
    };
    solveSelfContact(1u);
    buildYarnContacts();
    solveYarnBatches(1u);

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
        for (const NumiClothBagGPUBatch& batch : initial.fruitPairBatches) {
            [encoder setComputePipelineState:pipelines.fruitPair];
            [encoder setBuffer:configBuffer offset:0 atIndex:0];
            [encoder setBuffer:fruitBuffer offset:0 atIndex:1];
            [encoder setBuffer:fruitPairBuffer offset:0 atIndex:2];
            [encoder setBytes:&batch length:sizeof(batch) atIndex:3];
            [encoder setBuffer:failureBuffer offset:0 atIndex:4];
            dispatch(encoder, pipelines.fruitPair, batch.control.y);
        }
        solveYarnBatches(0u);
        solveSelfContact(0u);
        if (initial.config.constraintCounts.z != 0u) {
            [encoder setComputePipelineState:pipelines.ground];
            [encoder setBuffer:configBuffer offset:0 atIndex:0];
            [encoder setBuffer:particleBuffer offset:0 atIndex:1];
            [encoder setBuffer:fruitBuffer offset:0 atIndex:2];
            [encoder setBuffer:failureBuffer offset:0 atIndex:3];
            dispatch(
                encoder,
                pipelines.ground,
                std::max(initial.particles.size(), initial.fruits.size())
            );
        }
    }
    const auto solveFruitPairs = [&] {
        for (const NumiClothBagGPUBatch& batch : initial.fruitPairBatches) {
            [encoder setComputePipelineState:pipelines.fruitPair];
            [encoder setBuffer:configBuffer offset:0 atIndex:0];
            [encoder setBuffer:fruitBuffer offset:0 atIndex:1];
            [encoder setBuffer:fruitPairBuffer offset:0 atIndex:2];
            [encoder setBytes:&batch length:sizeof(batch) atIndex:3];
            [encoder setBuffer:failureBuffer offset:0 atIndex:4];
            dispatch(encoder, pipelines.fruitPair, batch.control.y);
        }
    };
    const auto solveGround = [&] {
        if (initial.config.constraintCounts.z == 0u) {
            return;
        }
        [encoder setComputePipelineState:pipelines.ground];
        [encoder setBuffer:configBuffer offset:0 atIndex:0];
        [encoder setBuffer:particleBuffer offset:0 atIndex:1];
        [encoder setBuffer:fruitBuffer offset:0 atIndex:2];
        [encoder setBuffer:failureBuffer offset:0 atIndex:3];
        dispatch(
            encoder,
            pipelines.ground,
            std::max(initial.particles.size(), initial.fruits.size())
        );
    };
    const auto solveStrainLimits = [&] {
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
    };
    solveStrainLimits();
    const auto solveFinalContacts = [&] {
        solveFruitPairs();
        solveYarnBatches(0u);
        solveGround();
    };
    const auto solveEndpointSelfAndStrain = [&] {
        solveSelfContact(0u);
        solveStrainLimits();
    };
    if (strainSweeps != 0u) {
        for (std::uint32_t pass = 0u;
             pass < kReconciliationPasses;
             ++pass) {
            solveFruitPairs();
            solveYarnBatches(0u);
            solveStrainLimits();
            solveGround();
        }
        solveEndpointSelfAndStrain();
        for (std::uint32_t pass = 0u; pass < kFinalContactPasses; ++pass) {
            solveFinalContacts();
        }
        for (std::uint32_t certificate = 0u;
             certificate < kCertificatePasses;
             ++certificate) {
            solveEndpointSelfAndStrain();
            for (std::uint32_t pass = 0u;
                 pass < kFinalContactPasses;
                 ++pass) {
                solveFinalContacts();
            }
        }
    }
    [encoder setComputePipelineState:pipelines.selfImpulseSort];
    [encoder setBuffer:selfImpulseKeyBuffer offset:0 atIndex:0];
    dispatch(encoder, pipelines.selfImpulseSort, 256u);
    [encoder setComputePipelineState:pipelines.selfImpulseClear];
    [encoder setBuffer:configBuffer offset:0 atIndex:0];
    [encoder setBuffer:selfActiveFlagBuffer offset:0 atIndex:1];
    [encoder setBuffer:failureBuffer offset:0 atIndex:2];
    dispatch(
        encoder, pipelines.selfImpulseClear, initial.selfPairs.size()
    );
    [encoder setComputePipelineState:pipelines.selfImpulseAggregate];
    [encoder setBuffer:configBuffer offset:0 atIndex:0];
    [encoder setBuffer:selfImpulseKeyBuffer offset:0 atIndex:1];
    [encoder setBuffer:selfImpulseRecordBuffer offset:0 atIndex:2];
    [encoder setBuffer:selfImpulseCountBuffer offset:0 atIndex:3];
    [encoder setBuffer:selfImpulseAggregateBuffer offset:0 atIndex:4];
    [encoder setBuffer:selfActiveFlagBuffer offset:0 atIndex:5];
    [encoder setBuffer:failureBuffer offset:0 atIndex:6];
    dispatch(
        encoder,
        pipelines.selfImpulseAggregate,
        NUMI_CLOTH_BAG_GPU_SELF_IMPULSE_CAPACITY
    );
    [encoder setComputePipelineState:pipelines.selfFrictionBatchCount];
    [encoder setBuffer:configBuffer offset:0 atIndex:0];
    [encoder setBuffer:selfBatchBuffer offset:0 atIndex:1];
    [encoder setBuffer:selfActiveFlagBuffer offset:0 atIndex:2];
    [encoder setBuffer:selfActiveBatchCountBuffer offset:0 atIndex:3];
    [encoder setBuffer:failureBuffer offset:0 atIndex:4];
    dispatch(
        encoder,
        pipelines.selfFrictionBatchCount,
        initial.selfBatches.size()
    );
    [encoder setComputePipelineState:pipelines.selfBatchCompact];
    [encoder setBuffer:configBuffer offset:0 atIndex:0];
    [encoder setBuffer:selfActiveBatchCountBuffer offset:0 atIndex:1];
    [encoder setBuffer:selfActiveBatchIndexBuffer offset:0 atIndex:2];
    [encoder setBuffer:activeSelfBatchCountBuffer offset:0 atIndex:3];
    dispatch(encoder, pipelines.selfBatchCompact, 256u);
    buildYarnContacts();
    [encoder setComputePipelineState:pipelines.finalize];
    [encoder setBuffer:configBuffer offset:0 atIndex:0];
    [encoder setBuffer:particleBuffer offset:0 atIndex:1];
    [encoder setBuffer:failureBuffer offset:0 atIndex:2];
    dispatch(encoder, pipelines.finalize, initial.particles.size());
    [encoder setComputePipelineState:pipelines.finalizeFruit];
    [encoder setBuffer:configBuffer offset:0 atIndex:0];
    [encoder setBuffer:fruitBuffer offset:0 atIndex:1];
    [encoder setBuffer:failureBuffer offset:0 atIndex:2];
    dispatch(encoder, pipelines.finalizeFruit, initial.fruits.size());
    [encoder setComputePipelineState:pipelines.clothGroundFriction];
    [encoder setBuffer:configBuffer offset:0 atIndex:0];
    [encoder setBuffer:particleBuffer offset:0 atIndex:1];
    [encoder setBuffer:frictionStatusBuffer offset:0 atIndex:2];
    [encoder setBuffer:failureBuffer offset:0 atIndex:3];
    dispatch(
        encoder, pipelines.clothGroundFriction, initial.particles.size()
    );
    [encoder setComputePipelineState:pipelines.selfFriction];
    [encoder setBuffer:configBuffer offset:0 atIndex:0];
    [encoder setBuffer:particleBuffer offset:0 atIndex:1];
    [encoder setBuffer:distanceBuffer offset:0 atIndex:2];
    [encoder setBuffer:selfPairBuffer offset:0 atIndex:3];
    [encoder setBuffer:selfBatchBuffer offset:0 atIndex:4];
    [encoder setBuffer:selfActiveFlagBuffer offset:0 atIndex:5];
    [encoder setBuffer:selfImpulseAggregateBuffer offset:0 atIndex:6];
    [encoder setBuffer:selfActiveBatchIndexBuffer offset:0 atIndex:7];
    [encoder setBuffer:activeSelfBatchCountBuffer offset:0 atIndex:8];
    [encoder setBuffer:frictionStatusBuffer offset:0 atIndex:9];
    [encoder setBuffer:failureBuffer offset:0 atIndex:10];
    dispatch(
        encoder, pipelines.selfFriction, initial.maximumSelfBatchSize
    );
    for (const NumiClothBagGPUBatch& batch : initial.distanceBatches) {
        [encoder setComputePipelineState:pipelines.yarnFriction];
        [encoder setBuffer:configBuffer offset:0 atIndex:0];
        [encoder setBuffer:particleBuffer offset:0 atIndex:1];
        [encoder setBuffer:distanceBuffer offset:0 atIndex:2];
        [encoder setBuffer:fruitBuffer offset:0 atIndex:3];
        [encoder setBuffer:yarnContactBuffer offset:0 atIndex:4];
        [encoder setBytes:&batch length:sizeof(batch) atIndex:5];
        [encoder setBuffer:frictionStatusBuffer offset:0 atIndex:6];
        [encoder setBuffer:failureBuffer offset:0 atIndex:7];
        dispatch(encoder, pipelines.yarnFriction, initial.fruits.size());
    }
    for (const NumiClothBagGPUBatch& batch : initial.fruitPairBatches) {
        [encoder setComputePipelineState:pipelines.fruitPairFriction];
        [encoder setBuffer:configBuffer offset:0 atIndex:0];
        [encoder setBuffer:fruitBuffer offset:0 atIndex:1];
        [encoder setBuffer:fruitPairBuffer offset:0 atIndex:2];
        [encoder setBytes:&batch length:sizeof(batch) atIndex:3];
        [encoder setBuffer:frictionStatusBuffer offset:0 atIndex:4];
        [encoder setBuffer:failureBuffer offset:0 atIndex:5];
        dispatch(encoder, pipelines.fruitPairFriction, batch.control.y);
    }
    [encoder setComputePipelineState:pipelines.fruitGroundFriction];
    [encoder setBuffer:configBuffer offset:0 atIndex:0];
    [encoder setBuffer:fruitBuffer offset:0 atIndex:1];
    [encoder setBuffer:frictionStatusBuffer offset:0 atIndex:2];
    [encoder setBuffer:failureBuffer offset:0 atIndex:3];
    dispatch(
        encoder, pipelines.fruitGroundFriction, initial.fruits.size()
    );
    [encoder setComputePipelineState:pipelines.fruitOrientation];
    [encoder setBuffer:configBuffer offset:0 atIndex:0];
    [encoder setBuffer:fruitBuffer offset:0 atIndex:1];
    [encoder setBuffer:failureBuffer offset:0 atIndex:2];
    dispatch(encoder, pipelines.fruitOrientation, initial.fruits.size());
    [encoder setComputePipelineState:pipelines.release];
    [encoder setBuffer:configBuffer offset:0 atIndex:0];
    [encoder setBuffer:particleBuffer offset:0 atIndex:1];
    [encoder setBuffer:mouthRimBuffer offset:0 atIndex:2];
    [encoder setBuffer:fruitBuffer offset:0 atIndex:3];
    [encoder setBuffer:releaseStatusBuffer offset:0 atIndex:4];
    [encoder setBuffer:maximumMouthClearanceBuffer offset:0 atIndex:5];
    [encoder setBuffer:failureBuffer offset:0 atIndex:6];
    dispatch(encoder, pipelines.release, initial.fruits.size());
    }
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
    assign(result.fruits, fruitBuffer, initial.fruits);
    assign(result.fruitPairs, fruitPairBuffer, initial.fruitPairs);
    assign(result.yarnContacts, yarnContactBuffer, initial.yarnContacts);
    result.selfStatus = *static_cast<const NumiClothBagGPUSelfStatus*>(
        selfStatusBuffer.contents
    );
    result.frictionStatus =
        *static_cast<const NumiClothBagGPUFrictionStatus*>(
            frictionStatusBuffer.contents
        );
    result.aerodynamicsStatus =
        *static_cast<const NumiClothBagGPUAerodynamicsStatus*>(
            aerodynamicsStatusBuffer.contents
        );
    result.releaseStatus =
        *static_cast<const NumiClothBagGPUReleaseStatus*>(
            releaseStatusBuffer.contents
        );
    const auto* mouthClearanceValues =
        static_cast<const std::uint32_t*>(
            maximumMouthClearanceBuffer.contents
        );
    result.maximumMouthClearanceBits.assign(
        mouthClearanceValues,
        mouthClearanceValues + initial.fruits.size()
    );
    result.failure = *static_cast<const std::uint32_t*>(failureBuffer.contents);
    if (commandBuffer.GPUEndTime >= commandBuffer.GPUStartTime) {
        result.seconds = commandBuffer.GPUEndTime - commandBuffer.GPUStartTime;
    }
    return result;
}

InitialState continuedInitialState(
    const InitialState& source,
    const GPUResult& result,
    const NumiClothBagGPUConfig& nextConfig
) {
    InitialState next = source;
    next.config = nextConfig;
    next.particles = result.particles;
    next.distances = result.distances;
    next.grips = result.grips;
    next.knots = result.knots;
    next.bends = result.bends;
    next.fruits = result.fruits;
    next.fruitPairs = result.fruitPairs;
    next.yarnContacts = result.yarnContacts;
    next.initialMouthCandidateMask = result.releaseStatus.masks.x;
    next.initialReleasedMask = result.releaseStatus.masks.y;
    return next;
}

bool bitwiseEqualPhysicalState(
    const GPUResult& first,
    const GPUResult& second
) {
    const auto equal = [](const auto& left, const auto& right) {
        using Value = typename std::decay_t<decltype(left)>::value_type;
        return left.size() == right.size() &&
            std::memcmp(
                left.data(), right.data(), left.size() * sizeof(Value)
            ) == 0;
    };
    return first.failure == second.failure &&
        equal(first.particles, second.particles) &&
        equal(first.distances, second.distances) &&
        equal(first.grips, second.grips) &&
        equal(first.knots, second.knots) &&
        equal(first.bends, second.bends) &&
        equal(first.fruits, second.fruits) &&
        equal(first.fruitPairs, second.fruitPairs) &&
        equal(first.yarnContacts, second.yarnContacts) &&
        first.releaseStatus.masks.x == second.releaseStatus.masks.x &&
        first.releaseStatus.masks.y == second.releaseStatus.masks.y;
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
    append(result.fruits);
    append(result.fruitPairs);
    append(result.yarnContacts);
    const std::array<NumiClothBagGPUSelfStatus, 1> selfStatus{{
        result.selfStatus
    }};
    append(selfStatus);
    const std::array<NumiClothBagGPUFrictionStatus, 1> frictionStatus{{
        result.frictionStatus
    }};
    append(frictionStatus);
    const std::array<NumiClothBagGPUAerodynamicsStatus, 1>
        aerodynamicsStatus{{result.aerodynamicsStatus}};
    append(aerodynamicsStatus);
    const std::array<NumiClothBagGPUReleaseStatus, 1> releaseStatus{{
        result.releaseStatus
    }};
    append(releaseStatus);
    append(result.maximumMouthClearanceBits);
    hash ^= result.failure;
    hash *= 1099511628211ull;
    return hash;
}

void dumpGPUOBJ(
    const std::string& path,
    const std::vector<NumiClothBagGPUParticle>& particles,
    const std::vector<NumiClothBagGPUFruit>& fruits,
    const std::vector<NumiClothBagGPUGrip>& grips,
    const NumiClothBagGPUConfig& config
) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open Metal cloth OBJ: " + path);
    }
    output << std::setprecision(9);
    output << "# Numi Solver explicit-yarn Metal cloth bag\n";
    output << "# vertices " << particles.size()
           << " render_triangles "
           << (2u * kAround * (kLevels - 1u) +
               2u * (kBottomGrid - 1u) * (kBottomGrid - 1u))
           << '\n';
    for (const NumiClothBagGPUParticle& particle : particles) {
        output << "v " << particle.positionAndInverseMass.x << ' '
               << particle.positionAndInverseMass.y << ' '
               << particle.positionAndInverseMass.z << '\n';
    }
    for (std::uint32_t level = 0u; level + 1u < kLevels; ++level) {
        for (std::uint32_t ring = 0u; ring < kAround; ++ring) {
            const std::uint32_t next = (ring + 1u) % kAround;
            const std::uint32_t a = nodeIndex(level, ring) + 1u;
            const std::uint32_t b = nodeIndex(level, next) + 1u;
            const std::uint32_t c = nodeIndex(level + 1u, ring) + 1u;
            const std::uint32_t d = nodeIndex(level + 1u, next) + 1u;
            output << "f " << a << ' ' << b << ' ' << c << '\n';
            output << "f " << b << ' ' << d << ' ' << c << '\n';
        }
    }
    for (std::uint32_t row = 0u; row + 1u < kBottomGrid; ++row) {
        for (std::uint32_t column = 0u;
             column + 1u < kBottomGrid;
             ++column) {
            const std::uint32_t a = bottomGridIndex(row, column) + 1u;
            const std::uint32_t b = bottomGridIndex(row, column + 1u) + 1u;
            const std::uint32_t c = bottomGridIndex(row + 1u, column) + 1u;
            const std::uint32_t d =
                bottomGridIndex(row + 1u, column + 1u) + 1u;
            if ((row + column) % 2u == 0u) {
                output << "f " << a << ' ' << c << ' ' << b << '\n';
                output << "f " << b << ' ' << c << ' ' << d << '\n';
            } else {
                output << "f " << a << ' ' << d << ' ' << b << '\n';
                output << "f " << a << ' ' << c << ' ' << d << '\n';
            }
        }
    }
    output << "# grip center " << config.gripTargetAndActive.x << ' '
           << config.gripTargetAndActive.y << ' '
           << config.gripTargetAndActive.z << " active "
           << (config.gripTargetAndActive.w > 0.0f ? 1 : 0)
           << " orientation "
           << config.gripOrientation.w << ' '
           << config.gripOrientation.x << ' '
           << config.gripOrientation.y << ' '
           << config.gripOrientation.z << " patch_center "
           << (grips.empty() ? 0u : grips.front().particle.w) << '\n';
    for (std::size_t index = 0u; index < fruits.size(); ++index) {
        const NumiClothBagGPUFruit& fruit = fruits[index];
        output << "# ball " << index << " center "
               << fruit.positionAndInverseMass.x << ' '
               << fruit.positionAndInverseMass.y << ' '
               << fruit.positionAndInverseMass.z << " radius "
               << fruit.previousAndRadius.w << " appearance "
               << fruit.identity.x << " orientation "
               << fruit.orientation.w << ' ' << fruit.orientation.x << ' '
               << fruit.orientation.y << ' ' << fruit.orientation.z
               << " angular_velocity " << fruit.angularVelocity.x << ' '
               << fruit.angularVelocity.y << ' '
               << fruit.angularVelocity.z << '\n';
    }
}

struct TrajectoryReplay {
    GPUResult final;
    std::vector<std::uint64_t> frameHashes;
    bool failureFree{true};
    double gpuSeconds{};
    double maximumHandleLag{};
};

const char* trajectoryName(const TrajectoryScenario scenario) {
    switch (scenario) {
        case TrajectoryScenario::grounded:
            return "grounded";
        case TrajectoryScenario::spin:
            return "spin";
        case TrajectoryScenario::pickup:
            return "pickup";
        case TrajectoryScenario::recorded:
            return "recorded";
    }
    throw std::logic_error("unknown cloth trajectory scenario");
}

TrajectoryGripPose trajectoryGripPose(
    const TrajectoryScenario scenario,
    const double time,
    const numi::GripTrajectory* trajectory
) {
    switch (scenario) {
        case TrajectoryScenario::grounded:
            return {
                .target = authoredPosition(kLevels - 1u, 0u),
                .orientation = {},
                .active = false,
            };
        case TrajectoryScenario::spin:
            return {
                .target = spinGripTarget(time),
                .orientation = {},
                .active = true,
            };
        case TrajectoryScenario::pickup:
            return {
                .target = pickupGripTarget(time),
                .orientation = {},
                .active = true,
            };
        case TrajectoryScenario::recorded: {
            if (trajectory == nullptr) {
                throw std::logic_error(
                    "recorded Metal trajectory is missing its pose data"
                );
            }
            const numi::GripTrajectoryPose pose =
                numi::sampleGripTrajectory(*trajectory, time);
            const DVec3 base = authoredPosition(kLevels - 1u, 0u);
            return {
                .target = base + DVec3{
                    pose.translationMeters.x,
                    pose.translationMeters.y,
                    pose.translationMeters.z,
                },
                .orientation = pose.orientation,
                .active = pose.active,
                .attachmentGeneration = pose.attachmentGeneration,
            };
        }
    }
    throw std::logic_error("unknown cloth trajectory scenario");
}

InitialState makeTrajectoryInitialState(
    const InitialState& initial,
    const TrajectoryScenario scenario,
    const numi::GripTrajectory* trajectory = nullptr
) {
    InitialState state = initial;
    if (scenario == TrajectoryScenario::spin) {
        for (NumiClothBagGPUParticle& particle : state.particles) {
            particle.positionAndInverseMass.z +=
                static_cast<float>(kSpinAirborneLift);
            particle.previousAndMass.z +=
                static_cast<float>(kSpinAirborneLift);
        }
        for (NumiClothBagGPUFruit& fruit : state.fruits) {
            fruit.positionAndInverseMass.z +=
                static_cast<float>(kSpinAirborneLift);
            fruit.previousAndRadius.z +=
                static_cast<float>(kSpinAirborneLift);
        }
    }
    state.config.constraintCounts.z =
        scenario == TrajectoryScenario::spin ? 0u : 1u;
    const TrajectoryGripPose pose = trajectoryGripPose(
        scenario, 0.0, trajectory
    );
    state.config.gripTargetAndActive = f4(
        static_cast<float>(pose.target.x),
        static_cast<float>(pose.target.y),
        static_cast<float>(pose.target.z),
        pose.active ? 1.0f : 0.0f
    );
    state.config.gripOrientation = gripQuaternion4(pose.orientation);
    state.config.gripControl.x = pose.attachmentGeneration;
    state.config.gripControl.y = trajectory != nullptr &&
            trajectory->selectNearestCuffPatch
        ? 1u
        : 0u;
    return state;
}

TrajectoryReplay runTrajectoryReplay(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    const Pipelines& pipelines,
    const InitialState& initial,
    const TrajectoryScenario scenario,
    const std::uint32_t iterations,
    const std::uint32_t strainSweeps,
    const std::uint32_t steps,
    const std::uint32_t dumpEvery,
    const std::uint32_t replayIndex,
    const std::string& dumpPrefix,
    const numi::GripTrajectory* trajectory = nullptr
) {
    InitialState state = makeTrajectoryInitialState(
        initial, scenario, trajectory
    );
    TrajectoryReplay replay;
    replay.frameHashes.reserve(steps);
    if (!dumpPrefix.empty()) {
        dumpGPUOBJ(
            dumpPrefix + "-0.obj",
            state.particles,
            state.fruits,
            state.grips,
            state.config
        );
    }
    for (std::uint32_t step = 0u; step < steps; ++step) {
        std::vector<NumiClothBagGPUConfig> configs(
            kPickupSubstepsPerFrame, state.config
        );
        for (std::uint32_t substep = 0u;
             substep < kPickupSubstepsPerFrame;
             ++substep) {
            const std::uint64_t completedSubsteps =
                static_cast<std::uint64_t>(step) *
                    kPickupSubstepsPerFrame +
                substep + 1u;
            const TrajectoryGripPose pose = trajectoryGripPose(
                scenario,
                static_cast<double>(completedSubsteps) * kTimestep,
                trajectory
            );
            configs[substep].gripTargetAndActive = f4(
                static_cast<float>(pose.target.x),
                static_cast<float>(pose.target.y),
                static_cast<float>(pose.target.z),
                pose.active ? 1.0f : 0.0f
            );
            configs[substep].gripOrientation = gripQuaternion4(
                pose.orientation
            );
            configs[substep].gripControl.x = pose.attachmentGeneration;
            configs[substep].gripControl.y = trajectory != nullptr &&
                    trajectory->selectNearestCuffPatch
                ? 1u
                : 0u;
        }
        @autoreleasepool {
            replay.final = runGPU(
                device,
                queue,
                pipelines,
                state,
                iterations,
                strainSweeps,
                configs
            );
        }
        replay.failureFree = replay.failureFree &&
            replay.final.failure == NUMI_CLOTH_BAG_GPU_FAILURE_NONE;
        replay.gpuSeconds += replay.final.seconds;
        replay.frameHashes.push_back(hashGPUResult(replay.final));
        if (scenario != TrajectoryScenario::grounded &&
            configs.back().gripTargetAndActive.w > 0.0f) {
            for (const NumiClothBagGPUGrip& grip : replay.final.grips) {
                const DVec3 position = d3(
                    replay.final.particles[grip.particle.x]
                        .positionAndInverseMass
                );
                const DVec3 target =
                    d3(configs.back().gripTargetAndActive) +
                    rotateGripOffset(
                        configs.back().gripOrientation,
                        d3(grip.targetOffsetAndCompliance)
                    );
                replay.maximumHandleLag = std::max(
                    replay.maximumHandleLag,
                    length(position - target)
                );
            }
        }
        state = continuedInitialState(
            state, replay.final, configs.back()
        );
        const std::uint32_t completedSteps = step + 1u;
        if (!dumpPrefix.empty() &&
            (completedSteps % dumpEvery == 0u ||
             completedSteps == steps)) {
            dumpGPUOBJ(
                dumpPrefix + "-" + std::to_string(completedSteps) + ".obj",
                replay.final.particles,
                replay.final.fruits,
                replay.final.grips,
                configs.back()
            );
        }
        if (completedSteps % dumpEvery == 0u || completedSteps == steps) {
            std::cout << (scenario == TrajectoryScenario::pickup
                              ? "pickup_progress"
                              : "trajectory_progress")
                      << " scenario=" << trajectoryName(scenario)
                      << " replay=" << replayIndex
                      << " step=" << completedSteps << '/' << steps
                      << " released_mask="
                      << replay.final.releaseStatus.masks.y
                      << " gpu_seconds=" << replay.gpuSeconds << std::endl;
        }
        if (!replay.failureFree) {
            break;
        }
    }
    return replay;
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

double maximumGroundPenetration(
    const std::vector<NumiClothBagGPUParticle>& particles,
    const std::vector<NumiClothBagGPUFruit>& fruits,
    const double clothRadius
) {
    double maximum = 0.0;
    for (const NumiClothBagGPUParticle& particle : particles) {
        maximum = std::max(
            maximum,
            clothRadius -
                static_cast<double>(particle.positionAndInverseMass.z)
        );
    }
    for (const NumiClothBagGPUFruit& fruit : fruits) {
        maximum = std::max(
            maximum,
            static_cast<double>(fruit.previousAndRadius.w) -
                static_cast<double>(fruit.positionAndInverseMass.z)
        );
    }
    return maximum;
}

double maximumSelfPenetration(
    const std::vector<NumiClothBagGPUParticle>& particles,
    const std::vector<NumiClothBagGPUDistance>& distances,
    const std::vector<NumiClothBagGPUSelfPair>& pairs,
    const double clothRadius
) {
    double maximum = 0.0;
    const double target = 2.0 * clothRadius;
    for (const NumiClothBagGPUSelfPair& pair : pairs) {
        const mr_uint4 first = distances[
            pair.firstSegment
        ].particlesAndColor;
        const mr_uint4 second = distances[
            pair.secondSegment
        ].particlesAndColor;
        const mr_uint4 indices = u4(
            first.x, first.y, second.x, second.y
        );
        const DVec3 firstStart = d3(
            particles[indices.x].positionAndInverseMass
        );
        const DVec3 firstEnd = d3(
            particles[indices.y].positionAndInverseMass
        );
        const DVec3 secondStart = d3(
            particles[indices.z].positionAndInverseMass
        );
        const DVec3 secondEnd = d3(
            particles[indices.w].positionAndInverseMass
        );
        const auto axisSeparated = [target](
            const double firstA,
            const double firstB,
            const double secondA,
            const double secondB
        ) {
            return std::max(firstA, firstB) + target <
                    std::min(secondA, secondB) ||
                std::max(secondA, secondB) + target <
                    std::min(firstA, firstB);
        };
        if (axisSeparated(
                firstStart.x, firstEnd.x, secondStart.x, secondEnd.x
            ) ||
            axisSeparated(
                firstStart.y, firstEnd.y, secondStart.y, secondEnd.y
            ) ||
            axisSeparated(
                firstStart.z, firstEnd.z, secondStart.z, secondEnd.z
            )) {
            continue;
        }
        maximum = std::max(
            maximum,
            target - sampleSegments(
                firstStart, firstEnd, secondStart, secondEnd
            ).distance
        );
    }
    return maximum;
}

double minimumClothHeight(const GPUResult& result) {
    double minimum = std::numeric_limits<double>::infinity();
    for (const NumiClothBagGPUParticle& particle : result.particles) {
        minimum = std::min(
            minimum,
            static_cast<double>(particle.positionAndInverseMass.z)
        );
    }
    return minimum;
}

double minimumFruitClearance(const GPUResult& result) {
    double minimum = std::numeric_limits<double>::infinity();
    for (const NumiClothBagGPUFruit& fruit : result.fruits) {
        minimum = std::min(
            minimum,
            static_cast<double>(
                fruit.positionAndInverseMass.z -
                fruit.previousAndRadius.w
            )
        );
    }
    return minimum;
}

std::uint32_t groundedClothCount(
    const GPUResult& result,
    const double yarnRadius
) {
    std::uint32_t count = 0u;
    for (const NumiClothBagGPUParticle& particle : result.particles) {
        if (std::abs(
            static_cast<double>(particle.positionAndInverseMass.z) -
            yarnRadius
        ) <= 2.0e-6) {
            ++count;
        }
    }
    return count;
}

std::uint32_t groundedFruitCount(const GPUResult& result) {
    std::uint32_t count = 0u;
    for (const NumiClothBagGPUFruit& fruit : result.fruits) {
        if (std::abs(static_cast<double>(
            fruit.positionAndInverseMass.z - fruit.previousAndRadius.w
        )) <= 2.0e-6) {
            ++count;
        }
    }
    return count;
}

std::uint32_t trajectoryEscapeMask(
    const GPUResult& result,
    const TrajectoryScenario scenario
) {
    std::uint32_t mask = 0u;
    for (std::size_t index = 0u; index < result.fruits.size(); ++index) {
        const DVec3 position = d3(
            result.fruits[index].positionAndInverseMass
        );
        const double radial = std::hypot(position.x, position.y);
        const bool outside = scenario == TrajectoryScenario::grounded
            ? position.z > 0.90 || position.z < 0.0 || radial > 0.75
            : position.z > 5.0 || position.z < -5.0 || radial > 5.0;
        if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
            !std::isfinite(position.z) || outside) {
            mask |= 1u << index;
        }
    }
    return mask;
}

double maximumShapeDistanceChange(
    const InitialState& initial,
    const GPUResult& result
) {
    double maximum = 0.0;
    for (std::size_t first = 0u;
         first < result.particles.size();
         first += 8u) {
        const DVec3 initialFirst = d3(
            initial.particles[first].positionAndInverseMass
        );
        const DVec3 finalFirst = d3(
            result.particles[first].positionAndInverseMass
        );
        for (std::size_t second = first + 8u;
             second < result.particles.size();
             second += 8u) {
            const double initialDistance = length(
                d3(initial.particles[second].positionAndInverseMass) -
                initialFirst
            );
            const double finalDistance = length(
                d3(result.particles[second].positionAndInverseMass) -
                finalFirst
            );
            maximum = std::max(
                maximum,
                std::abs(finalDistance - initialDistance)
            );
        }
    }
    return maximum;
}

double kineticEnergy(const GPUResult& result) {
    double energy = 0.0;
    for (const NumiClothBagGPUParticle& particle : result.particles) {
        const DVec3 velocity = d3(particle.velocity);
        energy += 0.5 * static_cast<double>(particle.previousAndMass.w) *
            dot(velocity, velocity);
    }
    for (const NumiClothBagGPUFruit& fruit : result.fruits) {
        const double mass = 1.0 /
            static_cast<double>(fruit.positionAndInverseMass.w);
        const double radius = static_cast<double>(fruit.previousAndRadius.w);
        const DVec3 linearVelocity = d3(fruit.velocityAndGroundImpulse);
        const DVec3 angularVelocity = d3(fruit.angularVelocity);
        energy += 0.5 * mass * dot(linearVelocity, linearVelocity);
        energy += 0.2 * mass * radius * radius *
            dot(angularVelocity, angularVelocity);
    }
    return energy;
}

int run(const int argc, const char* const* argv) {
    std::uint32_t replays = 2u;
    std::uint32_t iterations = 32u;
    std::uint32_t strainSweeps = 3u;
    std::uint32_t groundedSteps = kGroundedQualificationFrames;
    std::uint32_t groundedDumpEvery = 10u;
    std::string groundedPrefix;
    std::uint32_t spinSteps = kSpinQualificationFrames;
    std::uint32_t spinDumpEvery = 5u;
    std::string spinPrefix;
    std::uint32_t pickupSteps = kPickupQualificationFrames;
    std::uint32_t pickupDumpEvery = 10u;
    std::string pickupPrefix;
    std::uint32_t recordedSteps = 0u;
    std::uint32_t recordedDumpEvery = 1u;
    std::string recordedPrefix;
    std::string gripTrajectoryPath;
    std::string materialPath;
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
        } else if (value == "--material" && argument + 1 < argc) {
            materialPath = argv[++argument];
        } else if (value == "--grip-trajectory" && argument + 1 < argc) {
            gripTrajectoryPath = argv[++argument];
        } else if (value == "--grounded-prefix" && argument + 1 < argc) {
            groundedPrefix = argv[++argument];
        } else if (value == "--grounded-steps" && argument + 1 < argc) {
            groundedSteps = static_cast<std::uint32_t>(
                std::stoul(argv[++argument])
            );
        } else if (value == "--grounded-dump-every" &&
                   argument + 1 < argc) {
            groundedDumpEvery = static_cast<std::uint32_t>(
                std::stoul(argv[++argument])
            );
        } else if (value == "--spin-prefix" && argument + 1 < argc) {
            spinPrefix = argv[++argument];
        } else if (value == "--spin-steps" && argument + 1 < argc) {
            spinSteps = static_cast<std::uint32_t>(
                std::stoul(argv[++argument])
            );
        } else if (value == "--spin-dump-every" &&
                   argument + 1 < argc) {
            spinDumpEvery = static_cast<std::uint32_t>(
                std::stoul(argv[++argument])
            );
        } else if (value == "--pickup-prefix" && argument + 1 < argc) {
            pickupPrefix = argv[++argument];
        } else if (value == "--pickup-steps" && argument + 1 < argc) {
            pickupSteps = static_cast<std::uint32_t>(
                std::stoul(argv[++argument])
            );
        } else if (value == "--pickup-dump-every" &&
                   argument + 1 < argc) {
            pickupDumpEvery = static_cast<std::uint32_t>(
                std::stoul(argv[++argument])
            );
        } else if (value == "--recorded-prefix" && argument + 1 < argc) {
            recordedPrefix = argv[++argument];
        } else if (value == "--recorded-steps" && argument + 1 < argc) {
            recordedSteps = static_cast<std::uint32_t>(
                std::stoul(argv[++argument])
            );
        } else if (value == "--recorded-dump-every" &&
                   argument + 1 < argc) {
            recordedDumpEvery = static_cast<std::uint32_t>(
                std::stoul(argv[++argument])
            );
        } else if (value == "--help") {
            std::cout
                << "usage: numi-solver-cloth-metal [--replays N] "
                   "[--iterations N] [--strain-sweeps N] "
                   "[--metallib PATH] [--material FILE] "
                   "[--grip-trajectory FILE] [--recorded-steps N] "
                   "[--recorded-prefix PATH] "
                   "[--recorded-dump-every N] "
                   "[--grounded-prefix PATH] "
                   "[--grounded-steps N] [--grounded-dump-every N] "
                   "[--spin-prefix PATH] [--spin-steps N] "
                   "[--spin-dump-every N] [--pickup-prefix PATH] "
                   "[--pickup-steps N] [--pickup-dump-every N]\n";
            return 0;
        } else {
            throw std::runtime_error(
                "unknown argument: " + std::string(value)
            );
        }
    }
    if (!materialPath.empty()) {
        applyClothMaterial(numi::loadClothMaterialArtifact(materialPath));
    }
    numi::GripTrajectory gripTrajectory;
    const numi::GripTrajectory* gripTrajectoryPointer = nullptr;
    if (!gripTrajectoryPath.empty()) {
        gripTrajectory = numi::loadGripTrajectory(gripTrajectoryPath);
        gripTrajectoryPointer = &gripTrajectory;
    }
    replays = std::max(replays, 2u);
    if (iterations == 0u || strainSweeps == 0u) {
        throw std::runtime_error("iterations and strain sweeps must be positive");
    }
    if (!groundedPrefix.empty() &&
        (groundedSteps == 0u ||
         groundedSteps > kGroundedQualificationFrames ||
         groundedDumpEvery == 0u)) {
        throw std::runtime_error(
            "Metal grounded requires 1..120 steps and positive dump cadence"
        );
    }
    if (!spinPrefix.empty() &&
        (spinSteps == 0u || spinSteps > kSpinQualificationFrames ||
         spinDumpEvery == 0u)) {
        throw std::runtime_error(
            "Metal spin requires 1..60 steps and positive dump cadence"
        );
    }
    if (!pickupPrefix.empty() &&
        (pickupSteps == 0u || pickupSteps > kPickupQualificationFrames ||
         pickupDumpEvery == 0u)) {
        throw std::runtime_error(
            "Metal pickup requires 1..480 steps and positive dump cadence"
        );
    }
    if (gripTrajectoryPointer == nullptr &&
        (recordedSteps != 0u || !recordedPrefix.empty())) {
        throw std::runtime_error(
            "Metal recorded replay requires --grip-trajectory"
        );
    }
    if (gripTrajectoryPointer != nullptr &&
        (recordedSteps == 0u || recordedDumpEvery == 0u ||
         !numi::gripTrajectoryCovers(
             gripTrajectory,
             static_cast<double>(recordedSteps) *
                 kPickupSubstepsPerFrame * kTimestep
         ))) {
        throw std::runtime_error(
            "Metal recorded replay requires positive covered steps and dump "
            "cadence"
        );
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
        makePipeline(
            device,
            library,
            @"numi_cloth_bag_prepare_trajectory_substep"
        ),
        makePipeline(device, library, @"numi_cloth_bag_begin_substep"),
        makePipeline(
            device,
            library,
            @"numi_cloth_bag_update_grip_attachment"
        ),
        makePipeline(
            device,
            library,
            @"numi_cloth_bag_clear_yarn_aerodynamic_forces"
        ),
        makePipeline(
            device,
            library,
            @"numi_cloth_bag_accumulate_yarn_aerodynamics"
        ),
        makePipeline(
            device,
            library,
            @"numi_cloth_bag_reduce_yarn_aerodynamics"
        ),
        makePipeline(
            device,
            library,
            @"numi_cloth_bag_apply_yarn_aerodynamics"
        ),
        makePipeline(
            device,
            library,
            @"numi_cloth_bag_apply_fruit_aerodynamics"
        ),
        makePipeline(device, library, @"numi_cloth_bag_advance_positions"),
        makePipeline(device, library, @"numi_cloth_bag_solve_distance"),
        makePipeline(device, library, @"numi_cloth_bag_solve_knot"),
        makePipeline(device, library, @"numi_cloth_bag_solve_bend"),
        makePipeline(device, library, @"numi_cloth_bag_solve_grip"),
        makePipeline(device, library, @"numi_cloth_bag_solve_fruit_pair"),
        makePipeline(device, library, @"numi_cloth_bag_solve_ground"),
        makePipeline(device, library, @"numi_cloth_bag_limit_strain"),
        makePipeline(device, library, @"numi_cloth_bag_solve_yarn_batch"),
        makePipeline(device, library, @"numi_cloth_bag_build_yarn_contacts"),
        makePipeline(device, library, @"numi_cloth_bag_build_self_cells"),
        makePipeline(device, library, @"numi_cloth_bag_sort_self_cells"),
        makePipeline(device, library, @"numi_cloth_bag_detect_self_spatial"),
        makePipeline(device, library, @"numi_cloth_bag_count_self_batches"),
        makePipeline(device, library, @"numi_cloth_bag_compact_self_batches"),
        makePipeline(device, library, @"numi_cloth_bag_detect_self_contact"),
        makePipeline(device, library, @"numi_cloth_bag_solve_self_contact"),
        makePipeline(device, library, @"numi_cloth_bag_sort_self_impulses"),
        makePipeline(
            device, library, @"numi_cloth_bag_clear_self_impulse_map"
        ),
        makePipeline(
            device, library, @"numi_cloth_bag_aggregate_self_impulses"
        ),
        makePipeline(
            device, library, @"numi_cloth_bag_count_self_friction_batches"
        ),
        makePipeline(device, library, @"numi_cloth_bag_apply_self_friction"),
        makePipeline(device, library, @"numi_cloth_bag_finalize_substep"),
        makePipeline(device, library, @"numi_cloth_bag_finalize_fruit"),
        makePipeline(
            device, library, @"numi_cloth_bag_integrate_fruit_orientation"
        ),
        makePipeline(
            device, library, @"numi_cloth_bag_apply_cloth_ground_friction"
        ),
        makePipeline(device, library, @"numi_cloth_bag_apply_yarn_friction"),
        makePipeline(
            device, library, @"numi_cloth_bag_apply_fruit_pair_friction"
        ),
        makePipeline(
            device, library, @"numi_cloth_bag_apply_fruit_ground_friction"
        ),
        makePipeline(
            device, library, @"numi_cloth_bag_update_released_fruit"
        ),
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
    const InitialState fruitPairInitial = makeFruitPairProbeState();
    const OracleResult fruitPairOracle = runOracle(
        fruitPairInitial, 1u, 0u
    );
    const GPUResult fruitPairGPU = runGPU(
        device,
        queue,
        pipelines,
        fruitPairInitial,
        1u,
        0u
    );
    const InitialState groundContactInitial = makeGroundContactProbeState();
    const OracleResult groundContactOracle = runOracle(
        groundContactInitial, 1u, 0u
    );
    const GPUResult groundContactGPU = runGPU(
        device,
        queue,
        pipelines,
        groundContactInitial,
        1u,
        0u
    );
    const InitialState yarnCCDInitial = makeYarnCCDProbeState();
    const OracleResult yarnCCDOracle = runOracle(
        yarnCCDInitial, 0u, 0u
    );
    const GPUResult yarnCCDGPU = runGPU(
        device,
        queue,
        pipelines,
        yarnCCDInitial,
        0u,
        0u
    );
    const InitialState selfCCDInitial = makeSelfCCDProbeState();
    const OracleResult selfCCDOracle = runOracle(
        selfCCDInitial, 0u, 0u
    );
    const GPUResult selfCCDGPU = runGPU(
        device,
        queue,
        pipelines,
        selfCCDInitial,
        0u,
        0u
    );
    const InitialState yarnAerodynamicsInitial =
        makeYarnAerodynamicsProbeState(false, false);
    const OracleResult yarnAerodynamicsOracle = runOracle(
        yarnAerodynamicsInitial, 0u, 0u
    );
    const GPUResult yarnAerodynamicsGPU = runGPU(
        device,
        queue,
        pipelines,
        yarnAerodynamicsInitial,
        0u,
        0u
    );
    const InitialState refinedYarnAerodynamicsInitial =
        makeYarnAerodynamicsProbeState(true, false);
    const GPUResult refinedYarnAerodynamicsGPU = runGPU(
        device,
        queue,
        pipelines,
        refinedYarnAerodynamicsInitial,
        0u,
        0u
    );
    const InitialState coMovingYarnAerodynamicsInitial =
        makeYarnAerodynamicsProbeState(false, true);
    const GPUResult coMovingYarnAerodynamicsGPU = runGPU(
        device,
        queue,
        pipelines,
        coMovingYarnAerodynamicsInitial,
        0u,
        0u
    );
    const InitialState fruitAerodynamicsInitial =
        makeFruitAerodynamicsProbeState();
    const OracleResult fruitAerodynamicsOracle = runOracle(
        fruitAerodynamicsInitial, 0u, 0u
    );
    const GPUResult fruitAerodynamicsGPU = runGPU(
        device,
        queue,
        pipelines,
        fruitAerodynamicsInitial,
        0u,
        0u
    );
    const auto runMouthProbe = [&] (
        const DVec3 fruitPosition,
        const bool rotated,
        const std::uint32_t initialCandidateMask = 0u
    ) {
        const InitialState probe = makeMouthReleaseProbeState(
            fruitPosition, rotated, initialCandidateMask
        );
        return std::pair{
            runOracle(probe, 0u, 0u),
            runGPU(device, queue, pipelines, probe, 0u, 0u),
        };
    };
    const auto mouthInside = runMouthProbe(
        {0.0, 0.0, 0.50}, false
    );
    const auto mouthReleased = runMouthProbe(
        {0.0, 0.0, 1.130}, false
    );
    const auto mouthOutside = runMouthProbe(
        {1.50, 0.0, 1.20}, false
    );
    const auto mouthGrazing = runMouthProbe(
        {0.0, 0.0, 1.114}, false
    );
    const auto mouthEdgeClearance = runMouthProbe(
        {1.05, 0.0, 1.130}, false
    );
    const auto mouthEdgeExit = runMouthProbe(
        {1.30, 0.0, 1.02}, false, 1u
    );
    const auto mouthRotated = runMouthProbe(
        {1.130, 0.0, 0.0}, true
    );
    std::vector<NumiClothBagGPUConfig> seamTrajectoryConfigs(
        3u, initial.config
    );
    for (std::size_t substep = 0u;
         substep < seamTrajectoryConfigs.size();
         ++substep) {
        seamTrajectoryConfigs[substep].gripTargetAndActive.z =
            initial.config.gripTargetAndActive.z +
            0.0005f * static_cast<float>(substep + 1u);
    }
    const GPUResult seamTrajectoryGPU = runGPU(
        device,
        queue,
        pipelines,
        initial,
        iterations,
        strainSweeps,
        seamTrajectoryConfigs
    );
    const GPUResult seamTrajectoryReplayGPU = runGPU(
        device,
        queue,
        pipelines,
        initial,
        iterations,
        strainSweeps,
        seamTrajectoryConfigs
    );
    InitialState seamSequentialState = initial;
    GPUResult seamSequentialGPU;
    for (const NumiClothBagGPUConfig& config : seamTrajectoryConfigs) {
        seamSequentialState.config = config;
        seamSequentialGPU = runGPU(
            device,
            queue,
            pipelines,
            seamSequentialState,
            iterations,
            strainSweeps
        );
        seamSequentialState = continuedInitialState(
            seamSequentialState,
            seamSequentialGPU,
            config
        );
    }
    const bool seamTrajectoryReplayExact =
        hashGPUResult(seamTrajectoryGPU) ==
            hashGPUResult(seamTrajectoryReplayGPU);
    const bool seamTrajectorySplitExact = bitwiseEqualPhysicalState(
        seamTrajectoryGPU, seamSequentialGPU
    );
    double seamAverageLift = 0.0;
    double seamMaximumHandleLag = 0.0;
    for (const NumiClothBagGPUGrip& grip : initial.grips) {
        const std::uint32_t particle = grip.particle.x;
        const DVec3 finalPosition = d3(
            seamTrajectoryGPU.particles[particle].positionAndInverseMass
        );
        const DVec3 initialPosition = d3(
            initial.particles[particle].positionAndInverseMass
        );
        const DVec3 target =
            d3(seamTrajectoryConfigs.back().gripTargetAndActive) +
            rotateGripOffset(
                seamTrajectoryConfigs.back().gripOrientation,
                d3(grip.targetOffsetAndCompliance)
            );
        seamAverageLift += finalPosition.z - initialPosition.z;
        seamMaximumHandleLag = std::max(
            seamMaximumHandleLag, length(finalPosition - target)
        );
    }
    seamAverageLift /= static_cast<double>(initial.grips.size());

    InitialState gripRotationInitial = initial;
    constexpr double gripProbeHalfAngle = std::numbers::pi / 180.0;
    gripRotationInitial.config.gripOrientation = f4(
        0.0f,
        static_cast<float>(std::sin(gripProbeHalfAngle)),
        0.0f,
        static_cast<float>(std::cos(gripProbeHalfAngle))
    );
    const OracleResult gripRotationOracle = runOracle(
        gripRotationInitial, iterations, strainSweeps
    );
    const GPUResult gripRotationGPU = runGPU(
        device,
        queue,
        pipelines,
        gripRotationInitial,
        iterations,
        strainSweeps
    );
    const GPUResult gripRotationReplayGPU = runGPU(
        device,
        queue,
        pipelines,
        gripRotationInitial,
        iterations,
        strainSweeps
    );
    double gripRotationPositionError = 0.0;
    double gripRotationDisplacement = 0.0;
    for (const NumiClothBagGPUGrip& grip : initial.grips) {
        const std::uint32_t particle = grip.particle.x;
        gripRotationPositionError = std::max(
            gripRotationPositionError,
            length(
                d3(gripRotationGPU.particles[particle]
                    .positionAndInverseMass) -
                gripRotationOracle.particles[particle].position
            )
        );
        gripRotationDisplacement = std::max(
            gripRotationDisplacement,
            length(
                d3(gripRotationGPU.particles[particle]
                    .positionAndInverseMass) -
                d3(initial.particles[particle].positionAndInverseMass)
            )
        );
    }
    const bool gripRotationReplayExact =
        hashGPUResult(gripRotationGPU) ==
        hashGPUResult(gripRotationReplayGPU);
    InitialState distantRegrabInitial = initial;
    distantRegrabInitial.config.gripControl.x = 2u;
    distantRegrabInitial.config.gripControl.y = 1u;
    distantRegrabInitial.config.gripTargetAndActive.x +=
        2.0f * kGripCaptureRadius;
    const GPUResult distantRegrabGPU = runGPU(
        device,
        queue,
        pipelines,
        distantRegrabInitial,
        iterations,
        strainSweeps
    );
    const bool distantRegrabRejected =
        (distantRegrabGPU.failure &
         NUMI_CLOTH_BAG_GPU_FAILURE_GRIP_CAPTURE) != 0u;
    const bool groundedRequested = !groundedPrefix.empty();
    TrajectoryReplay groundedFirst;
    TrajectoryReplay groundedSecond;
    bool groundedReplayExact = true;
    bool groundedComplete = false;
    std::uint32_t groundedReleasedMask = 0u;
    std::uint32_t groundedEscapeMask = 0u;
    std::uint32_t groundedClothContacts = 0u;
    std::uint32_t groundedFruitContacts = 0u;
    double groundedMinimumClothHeight = 0.0;
    double groundedMinimumFruitClearance = 0.0;
    double groundedStrainViolation = 0.0;
    double groundedGroundPenetration = 0.0;
    double groundedSelfPenetration = 0.0;
    double groundedShapeChange = 0.0;
    double groundedKineticEnergy = 0.0;
    if (groundedRequested) {
        groundedFirst = runTrajectoryReplay(
            device,
            queue,
            pipelines,
            initial,
            TrajectoryScenario::grounded,
            iterations,
            strainSweeps,
            groundedSteps,
            groundedDumpEvery,
            1u,
            groundedPrefix
        );
        groundedSecond = runTrajectoryReplay(
            device,
            queue,
            pipelines,
            initial,
            TrajectoryScenario::grounded,
            iterations,
            strainSweeps,
            groundedSteps,
            groundedDumpEvery,
            2u,
            {}
        );
        groundedReplayExact = groundedFirst.frameHashes ==
                groundedSecond.frameHashes &&
            bitwiseEqualPhysicalState(
                groundedFirst.final, groundedSecond.final
            );
        groundedComplete = groundedSteps ==
            kGroundedQualificationFrames;
        groundedReleasedMask =
            groundedFirst.final.releaseStatus.masks.y;
        groundedEscapeMask = trajectoryEscapeMask(
            groundedFirst.final, TrajectoryScenario::grounded
        );
        groundedClothContacts = groundedClothCount(
            groundedFirst.final,
            initial.config.clothMaterial.x
        );
        groundedFruitContacts = groundedFruitCount(groundedFirst.final);
        groundedMinimumClothHeight = minimumClothHeight(
            groundedFirst.final
        );
        groundedMinimumFruitClearance = minimumFruitClearance(
            groundedFirst.final
        );
        groundedStrainViolation = maximumStrainViolation(
            groundedFirst.final.particles,
            groundedFirst.final.distances
        );
        groundedGroundPenetration = maximumGroundPenetration(
            groundedFirst.final.particles,
            groundedFirst.final.fruits,
            initial.config.clothMaterial.x
        );
        groundedSelfPenetration = maximumSelfPenetration(
            groundedFirst.final.particles,
            groundedFirst.final.distances,
            initial.selfPairs,
            initial.config.clothMaterial.x
        );
        groundedShapeChange = maximumShapeDistanceChange(
            makeTrajectoryInitialState(
                initial, TrajectoryScenario::grounded
            ),
            groundedFirst.final
        );
        groundedKineticEnergy = kineticEnergy(groundedFirst.final);
    }
    const bool groundedQualified = !groundedRequested ||
        (groundedComplete && groundedFirst.failureFree &&
         groundedSecond.failureFree && groundedReplayExact &&
         groundedReleasedMask == 0u && groundedEscapeMask == 0u &&
         groundedClothContacts > 0u &&
         groundedMinimumClothHeight >=
            static_cast<double>(initial.config.clothMaterial.x) - 1.0e-6 &&
         groundedMinimumFruitClearance >= -1.0e-6 &&
         groundedStrainViolation <= 2.0e-6 &&
         groundedGroundPenetration <= 1.0e-6 &&
         groundedSelfPenetration <= 2.0e-6 &&
         groundedShapeChange > 1.0e-3);

    const bool spinRequested = !spinPrefix.empty();
    TrajectoryReplay spinFirst;
    TrajectoryReplay spinSecond;
    bool spinReplayExact = true;
    bool spinComplete = false;
    std::uint32_t spinReleasedMask = 0u;
    std::uint32_t spinEscapeMask = 0u;
    double spinMinimumClothHeight = 0.0;
    double spinMinimumFruitClearance = 0.0;
    double spinStrainViolation = 0.0;
    double spinSelfPenetration = 0.0;
    double spinShapeChange = 0.0;
    double spinKineticEnergy = 0.0;
    double spinHandleTravel = 0.0;
    if (spinRequested) {
        spinFirst = runTrajectoryReplay(
            device,
            queue,
            pipelines,
            initial,
            TrajectoryScenario::spin,
            iterations,
            strainSweeps,
            spinSteps,
            spinDumpEvery,
            1u,
            spinPrefix
        );
        spinSecond = runTrajectoryReplay(
            device,
            queue,
            pipelines,
            initial,
            TrajectoryScenario::spin,
            iterations,
            strainSweeps,
            spinSteps,
            spinDumpEvery,
            2u,
            {}
        );
        spinReplayExact = spinFirst.frameHashes == spinSecond.frameHashes &&
            bitwiseEqualPhysicalState(spinFirst.final, spinSecond.final);
        spinComplete = spinSteps == kSpinQualificationFrames;
        spinReleasedMask = spinFirst.final.releaseStatus.masks.y;
        spinEscapeMask = trajectoryEscapeMask(
            spinFirst.final, TrajectoryScenario::spin
        );
        spinMinimumClothHeight = minimumClothHeight(spinFirst.final);
        spinMinimumFruitClearance = minimumFruitClearance(spinFirst.final);
        spinStrainViolation = maximumStrainViolation(
            spinFirst.final.particles,
            spinFirst.final.distances
        );
        spinSelfPenetration = maximumSelfPenetration(
            spinFirst.final.particles,
            spinFirst.final.distances,
            initial.selfPairs,
            initial.config.clothMaterial.x
        );
        spinShapeChange = maximumShapeDistanceChange(
            makeTrajectoryInitialState(initial, TrajectoryScenario::spin),
            spinFirst.final
        );
        spinKineticEnergy = kineticEnergy(spinFirst.final);
        spinHandleTravel = length(
            spinGripTarget(
                static_cast<double>(spinSteps) *
                kPickupSubstepsPerFrame * kTimestep
            ) - spinGripTarget(0.0)
        );
    }
    const bool spinQualified = !spinRequested ||
        (spinComplete && spinFirst.failureFree && spinSecond.failureFree &&
         spinReplayExact && spinReleasedMask == 0u && spinEscapeMask == 0u &&
         spinMinimumClothHeight > -5.0 &&
         spinMinimumFruitClearance > -5.0 &&
         spinStrainViolation <= 2.0e-6 &&
         spinSelfPenetration <= 2.0e-6 && spinHandleTravel > 0.20 &&
         spinFirst.maximumHandleLag > 1.0e-3 &&
         spinFirst.maximumHandleLag < 0.25 && spinShapeChange > 0.01);

    const bool pickupRequested = !pickupPrefix.empty();
    TrajectoryReplay pickupFirst;
    TrajectoryReplay pickupSecond;
    bool pickupReplayExact = true;
    bool pickupComplete = false;
    double pickupStrainViolation = 0.0;
    double pickupGroundPenetration = 0.0;
    double pickupSelfPenetration = 0.0;
    std::uint32_t pickupReleasedMask = 0u;
    std::uint32_t pickupGroundedReleasedCount = 0u;
    if (pickupRequested) {
        pickupFirst = runTrajectoryReplay(
            device,
            queue,
            pipelines,
            initial,
            TrajectoryScenario::pickup,
            iterations,
            strainSweeps,
            pickupSteps,
            pickupDumpEvery,
            1u,
            pickupPrefix
        );
        pickupSecond = runTrajectoryReplay(
            device,
            queue,
            pipelines,
            initial,
            TrajectoryScenario::pickup,
            iterations,
            strainSweeps,
            pickupSteps,
            pickupDumpEvery,
            2u,
            {}
        );
        pickupReplayExact = pickupFirst.frameHashes ==
                pickupSecond.frameHashes &&
            bitwiseEqualPhysicalState(
                pickupFirst.final, pickupSecond.final
            );
        pickupComplete = pickupSteps == kPickupQualificationFrames;
        pickupReleasedMask = pickupFirst.final.releaseStatus.masks.y;
        for (std::size_t index = 0u;
             index < pickupFirst.final.fruits.size();
             ++index) {
            const NumiClothBagGPUFruit& fruit =
                pickupFirst.final.fruits[index];
            if ((pickupReleasedMask & (1u << index)) != 0u &&
                std::abs(
                    static_cast<double>(
                        fruit.positionAndInverseMass.z -
                        fruit.previousAndRadius.w
                    )
                ) <= 2.0e-6) {
                ++pickupGroundedReleasedCount;
            }
        }
        pickupStrainViolation = maximumStrainViolation(
            pickupFirst.final.particles,
            pickupFirst.final.distances
        );
        pickupGroundPenetration = maximumGroundPenetration(
            pickupFirst.final.particles,
            pickupFirst.final.fruits,
            initial.config.clothMaterial.x
        );
        pickupSelfPenetration = maximumSelfPenetration(
            pickupFirst.final.particles,
            pickupFirst.final.distances,
            initial.selfPairs,
            initial.config.clothMaterial.x
        );
    }
    const bool pickupQualified = !pickupRequested ||
        (pickupComplete && pickupFirst.failureFree &&
         pickupSecond.failureFree && pickupReplayExact &&
         std::popcount(pickupReleasedMask) >= 2 &&
         pickupGroundedReleasedCount >= 2u &&
         pickupStrainViolation <= 2.0e-6 &&
         pickupGroundPenetration <= 1.0e-6 &&
         pickupSelfPenetration <= 2.0e-6);

    const bool recordedRequested = gripTrajectoryPointer != nullptr;
    TrajectoryReplay recordedFirst;
    TrajectoryReplay recordedSecond;
    bool recordedReplayExact = true;
    std::uint32_t recordedReleasedMask = 0u;
    std::uint32_t recordedEscapeMask = 0u;
    double recordedStrainViolation = 0.0;
    double recordedGroundPenetration = 0.0;
    double recordedSelfPenetration = 0.0;
    double recordedShapeChange = 0.0;
    std::uint32_t recordedAttachmentGenerations = 0u;
    std::uint64_t recordedInactiveSubsteps = 0u;
    bool recordedAttachmentGenerationsExact = true;
    double recordedMaximumCaptureDistance = 0.0;
    double recordedMaximumCaptureError = 0.0;
    bool recordedDynamicPatchSelection = false;
    bool recordedPatchTopologyExact = true;
    bool recordedPatchCenterExact = true;
    std::uint32_t recordedPatchCenterRing = 0u;
    if (recordedRequested) {
        recordedAttachmentGenerations =
            numi::gripTrajectoryAttachmentGenerations(gripTrajectory);
        recordedDynamicPatchSelection =
            gripTrajectory.selectNearestCuffPatch;
        for (std::uint64_t completedSubstep = 1u;
             completedSubstep <= static_cast<std::uint64_t>(recordedSteps) *
                 kPickupSubstepsPerFrame;
             ++completedSubstep) {
            if (!numi::sampleGripTrajectory(
                    gripTrajectory,
                    static_cast<double>(completedSubstep) * kTimestep
                ).active) {
                ++recordedInactiveSubsteps;
            }
        }
        recordedFirst = runTrajectoryReplay(
            device,
            queue,
            pipelines,
            initial,
            TrajectoryScenario::recorded,
            iterations,
            strainSweeps,
            recordedSteps,
            recordedDumpEvery,
            1u,
            recordedPrefix,
            gripTrajectoryPointer
        );
        recordedSecond = runTrajectoryReplay(
            device,
            queue,
            pipelines,
            initial,
            TrajectoryScenario::recorded,
            iterations,
            strainSweeps,
            recordedSteps,
            recordedDumpEvery,
            2u,
            {},
            gripTrajectoryPointer
        );
        recordedReplayExact = recordedFirst.frameHashes ==
                recordedSecond.frameHashes &&
            bitwiseEqualPhysicalState(
                recordedFirst.final, recordedSecond.final
            );
        recordedReleasedMask = recordedFirst.final.releaseStatus.masks.y;
        recordedEscapeMask = trajectoryEscapeMask(
            recordedFirst.final, TrajectoryScenario::recorded
        );
        recordedStrainViolation = maximumStrainViolation(
            recordedFirst.final.particles,
            recordedFirst.final.distances
        );
        recordedGroundPenetration = maximumGroundPenetration(
            recordedFirst.final.particles,
            recordedFirst.final.fruits,
            initial.config.clothMaterial.x
        );
        recordedSelfPenetration = maximumSelfPenetration(
            recordedFirst.final.particles,
            recordedFirst.final.distances,
            initial.selfPairs,
            initial.config.clothMaterial.x
        );
        recordedShapeChange = maximumShapeDistanceChange(
            makeTrajectoryInitialState(
                initial,
                TrajectoryScenario::recorded,
                gripTrajectoryPointer
            ),
            recordedFirst.final
        );
        for (const NumiClothBagGPUGrip& grip : recordedFirst.final.grips) {
            recordedAttachmentGenerationsExact =
                recordedAttachmentGenerationsExact &&
                grip.particle.y == recordedAttachmentGenerations;
            recordedMaximumCaptureDistance = std::max(
                recordedMaximumCaptureDistance,
                static_cast<double>(std::bit_cast<float>(grip.particle.z))
            );
            recordedMaximumCaptureError = std::max(
                recordedMaximumCaptureError,
                static_cast<double>(grip.lambda.w)
            );
        }
        if (recordedDynamicPatchSelection &&
            !recordedFirst.final.grips.empty()) {
            recordedPatchTopologyExact =
                recordedFirst.final.grips.size() == kGripCount;
            recordedPatchCenterRing =
                recordedFirst.final.grips.front().particle.w;
            constexpr std::uint32_t patchWidth = 5u;
            for (std::size_t index = 0u;
                 index < recordedFirst.final.grips.size();
                 ++index) {
                const NumiClothBagGPUGrip& grip =
                    recordedFirst.final.grips[index];
                recordedPatchCenterExact = recordedPatchCenterExact &&
                    grip.particle.w == recordedPatchCenterRing;
                const std::uint32_t row = static_cast<std::uint32_t>(
                    index / patchWidth
                );
                const std::uint32_t slot = static_cast<std::uint32_t>(
                    index % patchWidth
                );
                const std::uint32_t ring = (
                    recordedPatchCenterRing + kAround + slot - 2u
                ) % kAround;
                recordedPatchTopologyExact = recordedPatchTopologyExact &&
                    row < 2u &&
                    grip.particle.x ==
                        initial.cuffParticles[row * kAround + ring];
            }
        }
    }
    const bool recordedQualified = !recordedRequested ||
        (recordedFirst.failureFree && recordedSecond.failureFree &&
         recordedReplayExact && recordedEscapeMask == 0u &&
         recordedStrainViolation <= 2.0e-6 &&
         recordedGroundPenetration <= 1.0e-6 &&
         recordedSelfPenetration <= 2.0e-6 &&
         recordedAttachmentGenerationsExact &&
         recordedMaximumCaptureDistance <=
             initial.config.gripMaterial.x + 1.0e-6 &&
         recordedMaximumCaptureError <= 2.0e-6 &&
         (recordedAttachmentGenerations <= 1u ||
         recordedInactiveSubsteps > 0u) &&
         (!recordedDynamicPatchSelection ||
          (recordedPatchCenterExact && recordedPatchTopologyExact)));
    const GPUResult& gpu = gpuResults.front();
    bool deterministic = true;
    for (std::size_t replay = 1u; replay < gpuResults.size(); ++replay) {
        deterministic = deterministic &&
            gpu.failure == gpuResults[replay].failure &&
            bitwiseEqual(gpu.particles, gpuResults[replay].particles) &&
            bitwiseEqual(gpu.distances, gpuResults[replay].distances) &&
            bitwiseEqual(gpu.grips, gpuResults[replay].grips) &&
            bitwiseEqual(gpu.knots, gpuResults[replay].knots) &&
            bitwiseEqual(gpu.bends, gpuResults[replay].bends) &&
            bitwiseEqual(gpu.fruits, gpuResults[replay].fruits) &&
            bitwiseEqual(gpu.fruitPairs, gpuResults[replay].fruitPairs) &&
            bitwiseEqual(
                gpu.yarnContacts,
                gpuResults[replay].yarnContacts
            ) &&
            std::memcmp(
                &gpu.selfStatus,
                &gpuResults[replay].selfStatus,
                sizeof(gpu.selfStatus)
            ) == 0 &&
            std::memcmp(
                &gpu.frictionStatus,
                &gpuResults[replay].frictionStatus,
                sizeof(gpu.frictionStatus)
            ) == 0 &&
            std::memcmp(
                &gpu.aerodynamicsStatus,
                &gpuResults[replay].aerodynamicsStatus,
                sizeof(gpu.aerodynamicsStatus)
            ) == 0 &&
            std::memcmp(
                &gpu.releaseStatus,
                &gpuResults[replay].releaseStatus,
                sizeof(gpu.releaseStatus)
            ) == 0 &&
            gpu.maximumMouthClearanceBits ==
                gpuResults[replay].maximumMouthClearanceBits;
    }

    double maximumPositionError = 0.0;
    double maximumVelocityError = 0.0;
    double maximumDistanceLambdaError = 0.0;
    double maximumGripLambdaError = 0.0;
    double maximumKnotLambdaError = 0.0;
    double maximumBendLambdaError = 0.0;
    double maximumKnotLambda = 0.0;
    double maximumBendLambda = 0.0;
    double maximumFruitPositionError = 0.0;
    double maximumFruitVelocityError = 0.0;
    double maximumFruitAngularVelocityError = 0.0;
    double maximumFruitOrientationError = 0.0;
    double maximumFruitOrientationNormError = 0.0;
    double maximumFruitPairContactError = 0.0;
    double maximumFruitPairImpulse = 0.0;
    double maximumFruitPairPenetration = 0.0;
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
    for (std::size_t index = 0u; index < gpu.fruits.size(); ++index) {
        const DVec3 positionDelta =
            d3(gpu.fruits[index].positionAndInverseMass) -
            oracle.fruits[index].position;
        const DVec3 velocityDelta =
            d3(gpu.fruits[index].velocityAndGroundImpulse) -
            oracle.fruits[index].velocity;
        const DVec3 angularVelocityDelta =
            d3(gpu.fruits[index].angularVelocity) -
            oracle.fruits[index].angularVelocity;
        maximumFruitPositionError = std::max(
            maximumFruitPositionError,
            std::max({
                std::abs(positionDelta.x),
                std::abs(positionDelta.y),
                std::abs(positionDelta.z),
            })
        );
        maximumFruitVelocityError = std::max(
            maximumFruitVelocityError,
            std::max({
                std::abs(velocityDelta.x),
                std::abs(velocityDelta.y),
                std::abs(velocityDelta.z),
            })
        );
        maximumFruitAngularVelocityError = std::max(
            maximumFruitAngularVelocityError,
            std::max({
                std::abs(angularVelocityDelta.x),
                std::abs(angularVelocityDelta.y),
                std::abs(angularVelocityDelta.z),
            })
        );
        const std::array<double, 4> orientation{{
            gpu.fruits[index].orientation.x,
            gpu.fruits[index].orientation.y,
            gpu.fruits[index].orientation.z,
            gpu.fruits[index].orientation.w,
        }};
        double orientationNormSquared = 0.0;
        for (std::size_t component = 0u; component < 4u; ++component) {
            maximumFruitOrientationError = std::max(
                maximumFruitOrientationError,
                std::abs(
                    orientation[component] -
                    oracle.fruits[index].orientation[component]
                )
            );
            orientationNormSquared +=
                orientation[component] * orientation[component];
        }
        maximumFruitOrientationNormError = std::max(
            maximumFruitOrientationNormError,
            std::abs(std::sqrt(orientationNormSquared) - 1.0)
        );
    }
    for (std::size_t index = 0u; index < gpu.fruitPairs.size(); ++index) {
        const DVec3 gpuWeightedNormal = d3(gpu.fruitPairs[index].contact);
        const DVec3 normalDelta =
            gpuWeightedNormal - oracle.fruitPairs[index].weightedNormal;
        maximumFruitPairContactError = std::max({
            maximumFruitPairContactError,
            std::abs(normalDelta.x),
            std::abs(normalDelta.y),
            std::abs(normalDelta.z),
            std::abs(
                static_cast<double>(gpu.fruitPairs[index].contact.w) -
                oracle.fruitPairs[index].normalImpulse
            ),
        });
        maximumFruitPairImpulse = std::max(
            maximumFruitPairImpulse,
            static_cast<double>(gpu.fruitPairs[index].contact.w)
        );
        const auto& pair = gpu.fruitPairs[index];
        const double separation = length(
            d3(gpu.fruits[pair.fruitsAndColor.y].positionAndInverseMass) -
            d3(gpu.fruits[pair.fruitsAndColor.x].positionAndInverseMass)
        );
        const double target =
            gpu.fruits[pair.fruitsAndColor.x].previousAndRadius.w +
            gpu.fruits[pair.fruitsAndColor.y].previousAndRadius.w;
        maximumFruitPairPenetration = std::max(
            maximumFruitPairPenetration,
            target - separation
        );
    }
    bool yarnIdentityExact =
        gpu.yarnContacts.size() == oracle.yarnContacts.size();
    bool yarnControlExact = yarnIdentityExact;
    bool yarnControlQualified = yarnIdentityExact;
    bool yarnResponseCountExact = yarnIdentityExact;
    std::uint64_t currentYarnOverlapCount = 0u;
    std::uint64_t sweptYarnImpactCount = 0u;
    std::uint64_t acceptedYarnResponseCount = 0u;
    std::uint64_t expectedYarnResponseCount = 0u;
    std::uint64_t yarnResponseCountMismatches = 0u;
    double maximumYarnSeparationError = 0.0;
    double maximumYarnCurrentNormalError = 0.0;
    double maximumYarnSweptNormalError = 0.0;
    double maximumYarnWeightError = 0.0;
    double maximumActiveYarnWeightError = 0.0;
    double maximumYarnImpactTimeError = 0.0;
    double maximumYarnAdvanceError = 0.0;
    double maximumYarnResponseError = 0.0;
    double maximumYarnNormalImpulse = 0.0;
    double maximumYarnPenetration = 0.0;
    for (std::size_t index = 0u;
         index < gpu.yarnContacts.size() && index < oracle.yarnContacts.size();
         ++index) {
        const NumiClothBagGPUYarnContact& actual = gpu.yarnContacts[index];
        const OracleYarnContact& expected = oracle.yarnContacts[index];
        yarnIdentityExact = yarnIdentityExact &&
            actual.identity.x == expected.fruit &&
            actual.identity.y == expected.segment &&
            actual.identity.z == expected.first &&
            actual.identity.w == expected.second;
        yarnControlExact = yarnControlExact &&
            (actual.control.x != 0u) == expected.currentOverlap &&
            (actual.control.y != 0u) == expected.sweptImpact &&
            (actual.control.z != 0u) == expected.degenerateCurrent;
        const double actualSeparation =
            actual.currentNormalAndSeparation.w;
        const bool overlapQualified =
            (actual.control.x != 0u) == expected.currentOverlap ||
            (std::abs(actualSeparation) <= 2.0e-6 &&
             std::abs(expected.currentSeparation) <= 2.0e-6);
        const bool sweptQualified =
            (actual.control.y != 0u) == expected.sweptImpact ||
            (std::abs(actual.sweptNormalAndAdvance.w) <= 2.0e-6 &&
             std::abs(expected.removedAdvance) <= 2.0e-6);
        yarnControlQualified = yarnControlQualified && overlapQualified &&
            sweptQualified &&
            (actual.control.z != 0u) == expected.degenerateCurrent;
        yarnResponseCountExact = yarnResponseCountExact &&
            actual.control.w == expected.responseCount;
        yarnResponseCountMismatches +=
            actual.control.w != expected.responseCount;
        currentYarnOverlapCount += actual.control.x != 0u;
        sweptYarnImpactCount += actual.control.y != 0u;
        acceptedYarnResponseCount += actual.control.w;
        expectedYarnResponseCount += expected.responseCount;
        maximumYarnPenetration = std::max(
            maximumYarnPenetration,
            std::max(0.0, -actualSeparation)
        );
        maximumYarnSeparationError = std::max(
            maximumYarnSeparationError,
            std::abs(
                static_cast<double>(
                    actual.currentNormalAndSeparation.w
                ) - expected.currentSeparation
            )
        );
        const DVec3 currentNormalDelta =
            d3(actual.currentNormalAndSeparation) - expected.currentNormal;
        maximumYarnCurrentNormalError = std::max({
            maximumYarnCurrentNormalError,
            std::abs(currentNormalDelta.x),
            std::abs(currentNormalDelta.y),
            std::abs(currentNormalDelta.z),
        });
        if (expected.sweptImpact) {
            const DVec3 sweptNormalDelta =
                d3(actual.sweptNormalAndAdvance) - expected.sweptNormal;
            maximumYarnSweptNormalError = std::max({
                maximumYarnSweptNormalError,
                std::abs(sweptNormalDelta.x),
                std::abs(sweptNormalDelta.y),
                std::abs(sweptNormalDelta.z),
            });
        }
        maximumYarnWeightError = std::max({
            maximumYarnWeightError,
            std::abs(
                static_cast<double>(actual.weightsAndTime.x) -
                expected.currentWeight
            ),
            std::abs(
                static_cast<double>(actual.weightsAndTime.y) -
                expected.sweptWeight
            ),
            std::abs(
                static_cast<double>(actual.weightsAndTime.w) -
                expected.combinedRadius
            ),
        });
        if (expected.currentOverlap || expected.sweptImpact) {
            maximumActiveYarnWeightError = std::max({
                maximumActiveYarnWeightError,
                std::abs(
                    static_cast<double>(actual.weightsAndTime.x) -
                    expected.currentWeight
                ),
                std::abs(
                    static_cast<double>(actual.weightsAndTime.y) -
                    expected.sweptWeight
                ),
            });
        }
        maximumYarnImpactTimeError = std::max(
            maximumYarnImpactTimeError,
            std::abs(
                static_cast<double>(actual.weightsAndTime.z) -
                expected.impactTime
            )
        );
        maximumYarnAdvanceError = std::max(
            maximumYarnAdvanceError,
            std::abs(
                static_cast<double>(actual.sweptNormalAndAdvance.w) -
                expected.removedAdvance
            )
        );
        const DVec3 fruitImpulseDelta =
            d3(actual.fruitNormalAndImpulse) -
            expected.fruitNormalImpulse;
        maximumYarnResponseError = std::max({
            maximumYarnResponseError,
            std::abs(fruitImpulseDelta.x),
            std::abs(fruitImpulseDelta.y),
            std::abs(fruitImpulseDelta.z),
            std::abs(
                static_cast<double>(actual.fruitNormalAndImpulse.w) -
                expected.normalImpulse
            ),
            std::abs(
                static_cast<double>(actual.segmentImpulse.x) -
                expected.segmentImpulse[0]
            ),
            std::abs(
                static_cast<double>(actual.segmentImpulse.y) -
                expected.segmentImpulse[1]
            ),
            std::abs(
                static_cast<double>(actual.segmentImpulse.z) -
                expected.firstSweptTime
            ),
            std::abs(
                static_cast<double>(actual.segmentImpulse.w) -
                expected.sweptAdvance
            ),
        });
        maximumYarnNormalImpulse = std::max(
            maximumYarnNormalImpulse,
            static_cast<double>(actual.fruitNormalAndImpulse.w)
        );
    }
    const std::uint64_t gpuPresentSelfContacts =
        gpu.selfStatus.counters.x;
    const std::uint64_t gpuSweptSelfContacts =
        gpu.selfStatus.counters.y;
    const double gpuMaximumSelfCorrection = std::bit_cast<float>(
        gpu.selfStatus.counters.z
    );
    const std::uint64_t presentSelfContactCountDifference =
        gpuPresentSelfContacts > oracle.presentSelfContacts
        ? gpuPresentSelfContacts - oracle.presentSelfContacts
        : oracle.presentSelfContacts - gpuPresentSelfContacts;
    const std::uint64_t sweptSelfContactCountDifference =
        gpuSweptSelfContacts > oracle.sweptSelfContacts
        ? gpuSweptSelfContacts - oracle.sweptSelfContacts
        : oracle.sweptSelfContacts - gpuSweptSelfContacts;
    const double maximumSelfCorrectionError = std::abs(
        gpuMaximumSelfCorrection - oracle.maximumSelfCorrection
    );
    const double finalSelfPenetration = maximumSelfPenetration(
        gpu.particles,
        initial.distances,
        initial.selfPairs,
        initial.config.clothMaterial.x
    );
    const double finalGroundPenetration = maximumGroundPenetration(
        gpu.particles,
        gpu.fruits,
        initial.config.clothMaterial.x
    );
    const std::array<std::uint64_t, 4> gpuFrictionContacts{{
        gpu.frictionStatus.counters.x,
        gpu.frictionStatus.counters.y,
        gpu.frictionStatus.counters.z,
        gpu.frictionStatus.counters.w,
    }};
    const bool frictionContactCountsExact =
        gpuFrictionContacts == oracle.frictionContacts;
    const std::uint64_t gpuSelfFrictionContacts =
        gpu.frictionStatus.metrics.w;
    const std::uint64_t selfFrictionContactCountDifference =
        gpuSelfFrictionContacts > oracle.selfFrictionContacts
        ? gpuSelfFrictionContacts - oracle.selfFrictionContacts
        : oracle.selfFrictionContacts - gpuSelfFrictionContacts;
    std::uint64_t maximumFrictionContactCountDifference = 0u;
    for (std::size_t index = 0u; index < gpuFrictionContacts.size(); ++index) {
        const std::uint64_t actual = gpuFrictionContacts[index];
        const std::uint64_t expected = oracle.frictionContacts[index];
        maximumFrictionContactCountDifference = std::max(
            maximumFrictionContactCountDifference,
            actual > expected ? actual - expected : expected - actual
        );
    }
    const double gpuMaximumFrictionConeRatio = std::bit_cast<float>(
        gpu.frictionStatus.metrics.x
    );
    const double gpuMaximumRollingResistanceRatio = std::bit_cast<float>(
        gpu.frictionStatus.metrics.y
    );
    const double maximumFrictionConeRatioError = std::abs(
        gpuMaximumFrictionConeRatio - oracle.maximumFrictionConeRatio
    );
    const double maximumRollingResistanceRatioError = std::abs(
        gpuMaximumRollingResistanceRatio -
        oracle.maximumRollingResistanceRatio
    );
    const bool rollingContactCountExact =
        gpu.frictionStatus.metrics.z == oracle.rollingContacts;
    const double gpuMaximumYarnAerodynamicForce = std::bit_cast<float>(
        gpu.aerodynamicsStatus.metrics.x
    );
    const double gpuMaximumFruitAerodynamicForce = std::bit_cast<float>(
        gpu.aerodynamicsStatus.metrics.y
    );
    const double gpuMaximumFruitAerodynamicTorque = std::bit_cast<float>(
        gpu.aerodynamicsStatus.metrics.z
    );
    const double maximumYarnAerodynamicForceError = std::abs(
        gpuMaximumYarnAerodynamicForce -
        oracle.maximumYarnAerodynamicForce
    );
    const double maximumFruitAerodynamicForceError = std::abs(
        gpuMaximumFruitAerodynamicForce -
        oracle.maximumFruitAerodynamicForce
    );
    const double maximumFruitAerodynamicTorqueError = std::abs(
        gpuMaximumFruitAerodynamicTorque -
        oracle.maximumFruitAerodynamicTorque
    );
    bool releaseClearanceQualified =
        gpu.maximumMouthClearanceBits.size() ==
            oracle.maximumMouthClearanceByFruit.size();
    double maximumReleaseClearanceError = 0.0;
    for (std::size_t index = 0u;
         index < gpu.maximumMouthClearanceBits.size() &&
             index < oracle.maximumMouthClearanceByFruit.size();
         ++index) {
        maximumReleaseClearanceError = std::max(
            maximumReleaseClearanceError,
            std::abs(
                static_cast<double>(std::bit_cast<float>(
                    gpu.maximumMouthClearanceBits[index]
                )) - oracle.maximumMouthClearanceByFruit[index]
            )
        );
    }
    releaseClearanceQualified = releaseClearanceQualified &&
        maximumReleaseClearanceError <= 2.0e-6;
    const bool releaseMasksExact =
        gpu.releaseStatus.masks.x == oracle.mouthCandidateMask &&
        gpu.releaseStatus.masks.y == oracle.releasedMask;
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
    double fruitPairProbePositionError = 0.0;
    for (std::size_t index = 0u; index < fruitPairGPU.fruits.size(); ++index) {
        const DVec3 delta =
            d3(fruitPairGPU.fruits[index].positionAndInverseMass) -
            fruitPairOracle.fruits[index].position;
        fruitPairProbePositionError = std::max(
            fruitPairProbePositionError,
            std::max({std::abs(delta.x), std::abs(delta.y), std::abs(delta.z)})
        );
    }
    const double fruitPairProbeSeparation = length(
        d3(fruitPairGPU.fruits[1].positionAndInverseMass) -
        d3(fruitPairGPU.fruits[0].positionAndInverseMass)
    );
    const double fruitPairProbeCenterError = length(
        pairCenterOfMass(fruitPairGPU.fruits) -
        pairCenterOfMass(fruitPairInitial.fruits)
    );
    const double fruitPairProbeImpulseError = std::abs(
        static_cast<double>(fruitPairGPU.fruitPairs[0].contact.w) -
        fruitPairOracle.fruitPairs[0].normalImpulse
    );
    const double fruitPairProbeImpulse =
        fruitPairGPU.fruitPairs[0].contact.w;
    double fruitPairProbeVelocityError = 0.0;
    double fruitPairProbeAngularVelocityError = 0.0;
    double fruitPairProbeOrientationError = 0.0;
    double fruitPairProbeOrientationChange = 0.0;
    for (std::size_t index = 0u; index < fruitPairGPU.fruits.size(); ++index) {
        const DVec3 velocityDelta =
            d3(fruitPairGPU.fruits[index].velocityAndGroundImpulse) -
            fruitPairOracle.fruits[index].velocity;
        const DVec3 angularDelta =
            d3(fruitPairGPU.fruits[index].angularVelocity) -
            fruitPairOracle.fruits[index].angularVelocity;
        fruitPairProbeVelocityError = std::max({
            fruitPairProbeVelocityError,
            std::abs(velocityDelta.x),
            std::abs(velocityDelta.y),
            std::abs(velocityDelta.z),
        });
        fruitPairProbeAngularVelocityError = std::max({
            fruitPairProbeAngularVelocityError,
            std::abs(angularDelta.x),
            std::abs(angularDelta.y),
            std::abs(angularDelta.z),
        });
        const std::array<double, 4> orientation{{
            fruitPairGPU.fruits[index].orientation.x,
            fruitPairGPU.fruits[index].orientation.y,
            fruitPairGPU.fruits[index].orientation.z,
            fruitPairGPU.fruits[index].orientation.w,
        }};
        const std::array<double, 4> initialOrientation{{
            fruitPairInitial.fruits[index].orientation.x,
            fruitPairInitial.fruits[index].orientation.y,
            fruitPairInitial.fruits[index].orientation.z,
            fruitPairInitial.fruits[index].orientation.w,
        }};
        for (std::size_t component = 0u; component < 4u; ++component) {
            fruitPairProbeOrientationError = std::max(
                fruitPairProbeOrientationError,
                std::abs(
                    orientation[component] -
                    fruitPairOracle.fruits[index].orientation[component]
                )
            );
            fruitPairProbeOrientationChange = std::max(
                fruitPairProbeOrientationChange,
                std::abs(
                    orientation[component] -
                    initialOrientation[component]
                )
            );
        }
    }
    const auto tangentialSpeed = [](
        const DVec3 relativeVelocity,
        const DVec3 normal
    ) {
        return length(
            relativeVelocity - normal * dot(relativeVelocity, normal)
        );
    };
    const auto fruitEnergy = [](
        const NumiClothBagGPUFruit& source,
        const DVec3 velocity,
        const DVec3 angularVelocity
    ) {
        const double mass = 1.0 / source.positionAndInverseMass.w;
        const double radius = source.previousAndRadius.w;
        const double inertia = 0.4 * mass * radius * radius;
        return 0.5 * mass * dot(velocity, velocity) +
            0.5 * inertia * dot(angularVelocity, angularVelocity);
    };
    const DVec3 fruitPairNormal = normalized(
        d3(fruitPairGPU.fruitPairs[0].contact)
    );
    std::array<DVec3, 2> fruitPairPreVelocity{};
    std::array<DVec3, 2> fruitPairPostVelocity{};
    std::array<DVec3, 2> fruitPairPreAngular{};
    std::array<DVec3, 2> fruitPairPostAngular{};
    for (std::size_t index = 0u; index < 2u; ++index) {
        fruitPairPreVelocity[index] = (
            d3(fruitPairGPU.fruits[index].positionAndInverseMass) -
            d3(fruitPairInitial.fruits[index].positionAndInverseMass)
        ) / fruitPairInitial.config.gravityAndTimestep.w;
        fruitPairPostVelocity[index] = d3(
            fruitPairGPU.fruits[index].velocityAndGroundImpulse
        );
        fruitPairPreAngular[index] = d3(
            fruitPairInitial.fruits[index].angularVelocity
        );
        fruitPairPostAngular[index] = d3(
            fruitPairGPU.fruits[index].angularVelocity
        );
    }
    const DVec3 fruitPairFirstOffset = fruitPairNormal *
        fruitPairGPU.fruits[0].previousAndRadius.w;
    const DVec3 fruitPairSecondOffset = fruitPairNormal *
        -fruitPairGPU.fruits[1].previousAndRadius.w;
    const double fruitPairSlipBefore = tangentialSpeed(
        fruitPairPreVelocity[1] +
            cross(fruitPairPreAngular[1], fruitPairSecondOffset) -
            fruitPairPreVelocity[0] -
            cross(fruitPairPreAngular[0], fruitPairFirstOffset),
        fruitPairNormal
    );
    const double fruitPairSlipAfter = tangentialSpeed(
        fruitPairPostVelocity[1] +
            cross(fruitPairPostAngular[1], fruitPairSecondOffset) -
            fruitPairPostVelocity[0] -
            cross(fruitPairPostAngular[0], fruitPairFirstOffset),
        fruitPairNormal
    );
    DVec3 fruitPairMomentumBefore{};
    DVec3 fruitPairMomentumAfter{};
    double fruitPairEnergyBefore = 0.0;
    double fruitPairEnergyAfter = 0.0;
    for (std::size_t index = 0u; index < 2u; ++index) {
        const double mass = 1.0 /
            fruitPairGPU.fruits[index].positionAndInverseMass.w;
        fruitPairMomentumBefore += fruitPairPreVelocity[index] * mass;
        fruitPairMomentumAfter += fruitPairPostVelocity[index] * mass;
        fruitPairEnergyBefore += fruitEnergy(
            fruitPairGPU.fruits[index],
            fruitPairPreVelocity[index],
            fruitPairPreAngular[index]
        );
        fruitPairEnergyAfter += fruitEnergy(
            fruitPairGPU.fruits[index],
            fruitPairPostVelocity[index],
            fruitPairPostAngular[index]
        );
    }
    const double fruitPairMomentumError = length(
        fruitPairMomentumAfter - fruitPairMomentumBefore
    );
    const double groundClothHeight =
        groundContactGPU.particles[0].positionAndInverseMass.z;
    const double groundFruitHeight =
        groundContactGPU.fruits[0].positionAndInverseMass.z;
    const double groundFruitImpulse =
        groundContactGPU.fruits[0].velocityAndGroundImpulse.w;
    const double groundContactPositionError = std::max(
        std::abs(
            groundClothHeight -
            groundContactOracle.particles[0].position.z
        ),
        std::abs(
            groundFruitHeight -
            groundContactOracle.fruits[0].position.z
        )
    );
    const double groundContactImpulseError = std::abs(
        groundFruitImpulse -
        groundContactOracle.fruits[0].groundNormalImpulse
    );
    const double groundClothImpulseError = std::abs(
        static_cast<double>(groundContactGPU.particles[0].velocity.w) -
        groundContactOracle.particles[0].groundNormalImpulse
    );
    const DVec3 groundClothVelocityDelta =
        d3(groundContactGPU.particles[0].velocity) -
        groundContactOracle.particles[0].velocity;
    const DVec3 groundFruitVelocityDelta =
        d3(groundContactGPU.fruits[0].velocityAndGroundImpulse) -
        groundContactOracle.fruits[0].velocity;
    const DVec3 groundFruitAngularVelocityDelta =
        d3(groundContactGPU.fruits[0].angularVelocity) -
        groundContactOracle.fruits[0].angularVelocity;
    const double groundFrictionVelocityError = std::max({
        std::abs(groundClothVelocityDelta.x),
        std::abs(groundClothVelocityDelta.y),
        std::abs(groundClothVelocityDelta.z),
        std::abs(groundFruitVelocityDelta.x),
        std::abs(groundFruitVelocityDelta.y),
        std::abs(groundFruitVelocityDelta.z),
        std::abs(groundFruitAngularVelocityDelta.x),
        std::abs(groundFruitAngularVelocityDelta.y),
        std::abs(groundFruitAngularVelocityDelta.z),
    });
    const double groundProbeTimestep =
        groundContactInitial.config.gravityAndTimestep.w;
    const DVec3 groundClothVelocityBefore = (
        d3(groundContactGPU.particles[0].positionAndInverseMass) -
        d3(groundContactInitial.particles[0].positionAndInverseMass)
    ) / groundProbeTimestep;
    const DVec3 groundClothVelocityAfter = d3(
        groundContactGPU.particles[0].velocity
    );
    const DVec3 groundFruitVelocityBefore = (
        d3(groundContactGPU.fruits[0].positionAndInverseMass) -
        d3(groundContactInitial.fruits[0].positionAndInverseMass)
    ) / groundProbeTimestep;
    const DVec3 groundFruitVelocityAfter = d3(
        groundContactGPU.fruits[0].velocityAndGroundImpulse
    );
    const DVec3 groundFruitAngularBefore = d3(
        groundContactInitial.fruits[0].angularVelocity
    );
    const DVec3 groundFruitAngularAfter = d3(
        groundContactGPU.fruits[0].angularVelocity
    );
    const DVec3 groundOffset{
        0.0, 0.0, -groundContactGPU.fruits[0].previousAndRadius.w
    };
    const DVec3 groundNormal{0.0, 0.0, 1.0};
    const double groundClothSlipBefore = tangentialSpeed(
        groundClothVelocityBefore, groundNormal
    );
    const double groundClothSlipAfter = tangentialSpeed(
        groundClothVelocityAfter, groundNormal
    );
    const double groundFruitSlipBefore = tangentialSpeed(
        groundFruitVelocityBefore +
            cross(groundFruitAngularBefore, groundOffset),
        groundNormal
    );
    const double groundFruitSlipAfter = tangentialSpeed(
        groundFruitVelocityAfter +
            cross(groundFruitAngularAfter, groundOffset),
        groundNormal
    );
    const double groundRollingSpeedBefore = std::hypot(
        groundFruitAngularBefore.x, groundFruitAngularBefore.y
    );
    const double groundRollingSpeedAfter = std::hypot(
        groundFruitAngularAfter.x, groundFruitAngularAfter.y
    );
    const NumiClothBagGPUYarnContact& yarnCCDContact =
        yarnCCDGPU.yarnContacts.front();
    const OracleYarnContact& yarnCCDExpected =
        yarnCCDOracle.yarnContacts.front();
    const DVec3 yarnCCDCurrentNormalDelta =
        d3(yarnCCDContact.currentNormalAndSeparation) -
        yarnCCDExpected.currentNormal;
    const DVec3 yarnCCDSweptNormalDelta =
        d3(yarnCCDContact.sweptNormalAndAdvance) -
        yarnCCDExpected.sweptNormal;
    const double yarnCCDGeometryError = std::max({
        std::abs(yarnCCDCurrentNormalDelta.x),
        std::abs(yarnCCDCurrentNormalDelta.y),
        std::abs(yarnCCDCurrentNormalDelta.z),
        std::abs(yarnCCDSweptNormalDelta.x),
        std::abs(yarnCCDSweptNormalDelta.y),
        std::abs(yarnCCDSweptNormalDelta.z),
        std::abs(
            static_cast<double>(
                yarnCCDContact.currentNormalAndSeparation.w
            ) - yarnCCDExpected.currentSeparation
        ),
        std::abs(
            static_cast<double>(yarnCCDContact.weightsAndTime.x) -
            yarnCCDExpected.currentWeight
        ),
        std::abs(
            static_cast<double>(yarnCCDContact.weightsAndTime.y) -
            yarnCCDExpected.sweptWeight
        ),
        std::abs(
            static_cast<double>(yarnCCDContact.weightsAndTime.z) -
            yarnCCDExpected.impactTime
        ),
        std::abs(
            static_cast<double>(yarnCCDContact.sweptNormalAndAdvance.w) -
            yarnCCDExpected.removedAdvance
        ),
    });
    const bool yarnCCDFlagsQualified =
        ((yarnCCDContact.control.x != 0u) ==
            yarnCCDExpected.currentOverlap ||
         (std::abs(yarnCCDContact.currentNormalAndSeparation.w) <= 2.0e-6 &&
          std::abs(yarnCCDExpected.currentSeparation) <= 2.0e-6)) &&
        (yarnCCDContact.control.y != 0u) ==
            yarnCCDExpected.sweptImpact &&
        (yarnCCDContact.control.z != 0u) ==
            yarnCCDExpected.degenerateCurrent &&
        yarnCCDContact.control.w == yarnCCDExpected.responseCount;
    const DVec3 yarnCCDResponseDelta =
        d3(yarnCCDContact.fruitNormalAndImpulse) -
        yarnCCDExpected.fruitNormalImpulse;
    const double yarnCCDResponseError = std::max({
        std::abs(yarnCCDResponseDelta.x),
        std::abs(yarnCCDResponseDelta.y),
        std::abs(yarnCCDResponseDelta.z),
        std::abs(
            static_cast<double>(yarnCCDContact.fruitNormalAndImpulse.w) -
            yarnCCDExpected.normalImpulse
        ),
        std::abs(
            static_cast<double>(yarnCCDContact.segmentImpulse.x) -
            yarnCCDExpected.segmentImpulse[0]
        ),
        std::abs(
            static_cast<double>(yarnCCDContact.segmentImpulse.y) -
            yarnCCDExpected.segmentImpulse[1]
        ),
        std::abs(
            static_cast<double>(yarnCCDContact.segmentImpulse.z) -
            yarnCCDExpected.firstSweptTime
        ),
        std::abs(
            static_cast<double>(yarnCCDContact.segmentImpulse.w) -
            yarnCCDExpected.sweptAdvance
        ),
    });
    const double yarnCCDImpactTime = yarnCCDContact.segmentImpulse.z;
    const double yarnCCDRemovedAdvance =
        yarnCCDContact.segmentImpulse.w;
    const double yarnCCDFinalSeparation =
        yarnCCDContact.currentNormalAndSeparation.w;
    const double yarnCCDFinalFruitX =
        yarnCCDGPU.fruits[0].positionAndInverseMass.x;
    const double yarnCCDNormalImpulse =
        yarnCCDContact.fruitNormalAndImpulse.w;
    const DVec3 yarnCCDFruitVelocityDelta =
        d3(yarnCCDGPU.fruits[0].velocityAndGroundImpulse) -
        yarnCCDOracle.fruits[0].velocity;
    const DVec3 yarnCCDFruitAngularVelocityDelta =
        d3(yarnCCDGPU.fruits[0].angularVelocity) -
        yarnCCDOracle.fruits[0].angularVelocity;
    const double yarnCCDFrictionVelocityError = std::max({
        std::abs(yarnCCDFruitVelocityDelta.x),
        std::abs(yarnCCDFruitVelocityDelta.y),
        std::abs(yarnCCDFruitVelocityDelta.z),
        std::abs(yarnCCDFruitAngularVelocityDelta.x),
        std::abs(yarnCCDFruitAngularVelocityDelta.y),
        std::abs(yarnCCDFruitAngularVelocityDelta.z),
    });
    const DVec3 yarnCCDFrictionNormal = normalized(
        d3(yarnCCDContact.fruitNormalAndImpulse)
    );
    const DVec3 yarnCCDOffset = yarnCCDFrictionNormal *
        -yarnCCDGPU.fruits[0].previousAndRadius.w;
    const DVec3 yarnCCDFruitVelocityBefore = (
        d3(yarnCCDGPU.fruits[0].positionAndInverseMass) -
        d3(yarnCCDInitial.fruits[0].positionAndInverseMass)
    ) / yarnCCDInitial.config.gravityAndTimestep.w;
    const DVec3 yarnCCDFruitAngularBefore = d3(
        yarnCCDInitial.fruits[0].angularVelocity
    );
    const DVec3 yarnCCDFruitVelocityAfter = d3(
        yarnCCDGPU.fruits[0].velocityAndGroundImpulse
    );
    const DVec3 yarnCCDFruitAngularAfter = d3(
        yarnCCDGPU.fruits[0].angularVelocity
    );
    const double yarnCCDSlipBefore = tangentialSpeed(
        yarnCCDFruitVelocityBefore +
            cross(yarnCCDFruitAngularBefore, yarnCCDOffset),
        yarnCCDFrictionNormal
    );
    const double yarnCCDSlipAfter = tangentialSpeed(
        yarnCCDFruitVelocityAfter +
            cross(yarnCCDFruitAngularAfter, yarnCCDOffset),
        yarnCCDFrictionNormal
    );
    double selfCCDPositionError = 0.0;
    for (std::size_t index = 0u;
         index < selfCCDGPU.particles.size();
         ++index) {
        const DVec3 delta =
            d3(selfCCDGPU.particles[index].positionAndInverseMass) -
            selfCCDOracle.particles[index].position;
        selfCCDPositionError = std::max({
            selfCCDPositionError,
            std::abs(delta.x),
            std::abs(delta.y),
            std::abs(delta.z),
        });
    }
    const OracleSegmentSegmentSample selfCCDFinalGeometry = sampleSegments(
        d3(selfCCDGPU.particles[0].positionAndInverseMass),
        d3(selfCCDGPU.particles[1].positionAndInverseMass),
        d3(selfCCDGPU.particles[2].positionAndInverseMass),
        d3(selfCCDGPU.particles[3].positionAndInverseMass)
    );
    const double selfCCDFinalSeparation =
        selfCCDFinalGeometry.distance - 0.008;
    const double selfCCDFinalMovingHeight = 0.5 * (
        selfCCDGPU.particles[2].positionAndInverseMass.z +
        selfCCDGPU.particles[3].positionAndInverseMass.z
    );
    const double selfCCDMaximumCorrection = std::bit_cast<float>(
        selfCCDGPU.selfStatus.counters.z
    );
    double selfCCDFrictionVelocityError = 0.0;
    std::array<DVec3, 4> selfCCDVelocityBefore{};
    std::array<DVec3, 4> selfCCDVelocityAfter{};
    DVec3 selfCCDMomentumBefore{};
    DVec3 selfCCDMomentumAfter{};
    double selfCCDEnergyBefore = 0.0;
    double selfCCDEnergyAfter = 0.0;
    for (std::size_t index = 0u;
         index < selfCCDGPU.particles.size();
         ++index) {
        selfCCDVelocityBefore[index] = (
            d3(selfCCDGPU.particles[index].positionAndInverseMass) -
            d3(selfCCDInitial.particles[index].positionAndInverseMass)
        ) / selfCCDInitial.config.gravityAndTimestep.w;
        selfCCDVelocityAfter[index] = d3(
            selfCCDGPU.particles[index].velocity
        );
        const DVec3 velocityDelta = selfCCDVelocityAfter[index] -
            selfCCDOracle.particles[index].velocity;
        selfCCDFrictionVelocityError = std::max({
            selfCCDFrictionVelocityError,
            std::abs(velocityDelta.x),
            std::abs(velocityDelta.y),
            std::abs(velocityDelta.z),
        });
        const double inverseMass =
            selfCCDGPU.particles[index].positionAndInverseMass.w;
        if (inverseMass > 0.0) {
            const double mass = 1.0 / inverseMass;
            selfCCDMomentumBefore += selfCCDVelocityBefore[index] * mass;
            selfCCDMomentumAfter += selfCCDVelocityAfter[index] * mass;
            selfCCDEnergyBefore += 0.5 * mass *
                dot(selfCCDVelocityBefore[index], selfCCDVelocityBefore[index]);
            selfCCDEnergyAfter += 0.5 * mass *
                dot(selfCCDVelocityAfter[index], selfCCDVelocityAfter[index]);
        }
    }
    DVec3 selfCCDFrictionNormal{0.0, 0.0, 1.0};
    std::array<double, 2> selfCCDFirstWeights{{0.5, 0.5}};
    std::array<double, 2> selfCCDSecondWeights{{0.5, 0.5}};
    const auto selfCCDImpulse = selfCCDOracle.selfImpulses.find(0u);
    if (selfCCDImpulse != selfCCDOracle.selfImpulses.end()) {
        const OracleSelfImpulse& response = selfCCDImpulse->second;
        const double normalLength = length(response.normalOnFirst);
        const double firstSum =
            response.firstEndpointImpulses[0] +
            response.firstEndpointImpulses[1];
        const double secondSum =
            response.secondEndpointImpulses[0] +
            response.secondEndpointImpulses[1];
        if (normalLength > 1.0e-12) {
            selfCCDFrictionNormal = response.normalOnFirst / normalLength;
        }
        if (firstSum > 1.0e-12) {
            selfCCDFirstWeights = {{
                response.firstEndpointImpulses[0] / firstSum,
                response.firstEndpointImpulses[1] / firstSum,
            }};
        }
        if (secondSum > 1.0e-12) {
            selfCCDSecondWeights = {{
                response.secondEndpointImpulses[0] / secondSum,
                response.secondEndpointImpulses[1] / secondSum,
            }};
        }
    }
    const auto selfCCDContactVelocity = [](
        const std::array<DVec3, 4>& velocities,
        const std::array<double, 2>& firstWeights,
        const std::array<double, 2>& secondWeights
    ) {
        const DVec3 firstVelocity = velocities[0] * firstWeights[0] +
            velocities[1] * firstWeights[1];
        const DVec3 secondVelocity = velocities[2] * secondWeights[0] +
            velocities[3] * secondWeights[1];
        return firstVelocity - secondVelocity;
    };
    const double selfCCDSlipBefore = tangentialSpeed(
        selfCCDContactVelocity(
            selfCCDVelocityBefore,
            selfCCDFirstWeights,
            selfCCDSecondWeights
        ),
        selfCCDFrictionNormal
    );
    const double selfCCDSlipAfter = tangentialSpeed(
        selfCCDContactVelocity(
            selfCCDVelocityAfter,
            selfCCDFirstWeights,
            selfCCDSecondWeights
        ),
        selfCCDFrictionNormal
    );
    const double selfCCDMomentumError = length(
        selfCCDMomentumAfter - selfCCDMomentumBefore
    );
    const auto yarnRelativeEnergy = [](
        const InitialState& source,
        const std::vector<NumiClothBagGPUParticle>& particles
    ) {
        const DVec3 airVelocity = d3(source.config.airVelocityAndDensity);
        double energy = 0.0;
        for (std::size_t index = 0u; index < particles.size(); ++index) {
            const double mass = source.particles[index].previousAndMass.w;
            const DVec3 relativeVelocity =
                d3(particles[index].velocity) - airVelocity;
            energy += 0.5 * mass * dot(
                relativeVelocity, relativeVelocity
            );
        }
        return energy;
    };
    const auto yarnCenterVelocity = [](
        const InitialState& source,
        const GPUResult& result
    ) {
        DVec3 momentum{};
        double mass = 0.0;
        for (std::size_t index = 0u;
             index < result.particles.size();
             ++index) {
            const double particleMass =
                source.particles[index].previousAndMass.w;
            momentum += d3(result.particles[index].velocity) * particleMass;
            mass += particleMass;
        }
        return momentum / mass;
    };
    const double yarnAerodynamicsEnergyBefore = yarnRelativeEnergy(
        yarnAerodynamicsInitial, yarnAerodynamicsInitial.particles
    );
    const double yarnAerodynamicsEnergyAfter = yarnRelativeEnergy(
        yarnAerodynamicsInitial, yarnAerodynamicsGPU.particles
    );
    double yarnAerodynamicsVelocityError = 0.0;
    for (std::size_t index = 0u;
         index < yarnAerodynamicsGPU.particles.size();
         ++index) {
        const DVec3 delta = d3(
            yarnAerodynamicsGPU.particles[index].velocity
        ) - yarnAerodynamicsOracle.particles[index].velocity;
        yarnAerodynamicsVelocityError = std::max({
            yarnAerodynamicsVelocityError,
            std::abs(delta.x),
            std::abs(delta.y),
            std::abs(delta.z),
        });
    }
    const double yarnAerodynamicsRefinementError = length(
        yarnCenterVelocity(
            yarnAerodynamicsInitial, yarnAerodynamicsGPU
        ) -
        yarnCenterVelocity(
            refinedYarnAerodynamicsInitial,
            refinedYarnAerodynamicsGPU
        )
    );
    double coMovingYarnAerodynamicsDelta = 0.0;
    for (std::size_t index = 0u;
         index < coMovingYarnAerodynamicsGPU.particles.size();
         ++index) {
        const DVec3 delta = d3(
            coMovingYarnAerodynamicsGPU.particles[index].velocity
        ) - d3(coMovingYarnAerodynamicsInitial.particles[index].velocity);
        coMovingYarnAerodynamicsDelta = std::max({
            coMovingYarnAerodynamicsDelta,
            std::abs(delta.x),
            std::abs(delta.y),
            std::abs(delta.z),
        });
    }
    const double yarnAerodynamicsProbeForce = std::bit_cast<float>(
        yarnAerodynamicsGPU.aerodynamicsStatus.metrics.x
    );
    const double yarnAerodynamicsProbeForceError = std::abs(
        yarnAerodynamicsProbeForce -
        yarnAerodynamicsOracle.maximumYarnAerodynamicForce
    );
    const NumiClothBagGPUFruit& fruitAerodynamicsSource =
        fruitAerodynamicsInitial.fruits[0];
    const DVec3 fruitAerodynamicsVelocityBefore = d3(
        fruitAerodynamicsSource.velocityAndGroundImpulse
    );
    const DVec3 fruitAerodynamicsAngularBefore = d3(
        fruitAerodynamicsSource.angularVelocity
    );
    const DVec3 fruitAerodynamicsVelocityAfter = d3(
        fruitAerodynamicsGPU.fruits[0].velocityAndGroundImpulse
    );
    const DVec3 fruitAerodynamicsAngularAfter = d3(
        fruitAerodynamicsGPU.fruits[0].angularVelocity
    );
    const double fruitAerodynamicsMass =
        1.0 / fruitAerodynamicsSource.positionAndInverseMass.w;
    const double fruitAerodynamicsInertia =
        0.4 * fruitAerodynamicsMass *
        fruitAerodynamicsSource.previousAndRadius.w *
        fruitAerodynamicsSource.previousAndRadius.w;
    const double fruitAerodynamicsEnergyBefore =
        0.5 * fruitAerodynamicsMass * dot(
            fruitAerodynamicsVelocityBefore,
            fruitAerodynamicsVelocityBefore
        ) + 0.5 * fruitAerodynamicsInertia * dot(
            fruitAerodynamicsAngularBefore,
            fruitAerodynamicsAngularBefore
        );
    const double fruitAerodynamicsEnergyAfter =
        0.5 * fruitAerodynamicsMass * dot(
            fruitAerodynamicsVelocityAfter,
            fruitAerodynamicsVelocityAfter
        ) + 0.5 * fruitAerodynamicsInertia * dot(
            fruitAerodynamicsAngularAfter,
            fruitAerodynamicsAngularAfter
        );
    const DVec3 fruitAerodynamicsVelocityDelta =
        fruitAerodynamicsVelocityAfter -
        fruitAerodynamicsOracle.fruits[0].velocity;
    const DVec3 fruitAerodynamicsAngularDelta =
        fruitAerodynamicsAngularAfter -
        fruitAerodynamicsOracle.fruits[0].angularVelocity;
    const double fruitAerodynamicsVelocityError = std::max({
        std::abs(fruitAerodynamicsVelocityDelta.x),
        std::abs(fruitAerodynamicsVelocityDelta.y),
        std::abs(fruitAerodynamicsVelocityDelta.z),
        std::abs(fruitAerodynamicsAngularDelta.x),
        std::abs(fruitAerodynamicsAngularDelta.y),
        std::abs(fruitAerodynamicsAngularDelta.z),
    });
    const double fruitAerodynamicsProbeForce = std::bit_cast<float>(
        fruitAerodynamicsGPU.aerodynamicsStatus.metrics.y
    );
    const double fruitAerodynamicsProbeTorque = std::bit_cast<float>(
        fruitAerodynamicsGPU.aerodynamicsStatus.metrics.z
    );
    const double fruitAerodynamicsProbeForceError = std::abs(
        fruitAerodynamicsProbeForce -
        fruitAerodynamicsOracle.maximumFruitAerodynamicForce
    );
    const double fruitAerodynamicsProbeTorqueError = std::abs(
        fruitAerodynamicsProbeTorque -
        fruitAerodynamicsOracle.maximumFruitAerodynamicTorque
    );
    const auto mouthClearance = [](const GPUResult& result) {
        return result.maximumMouthClearanceBits.empty()
            ? 0.0
            : static_cast<double>(std::bit_cast<float>(
                result.maximumMouthClearanceBits[0]
            ));
    };
    const auto mouthMatchesOracle = [&mouthClearance](const auto& result) {
        return result.second.failure == NUMI_CLOTH_BAG_GPU_FAILURE_NONE &&
            result.second.releaseStatus.masks.x ==
                result.first.mouthCandidateMask &&
            result.second.releaseStatus.masks.y ==
                result.first.releasedMask &&
            std::abs(
                mouthClearance(result.second) -
                result.first.maximumMouthClearanceByFruit[0]
            ) <= 2.0e-6;
    };
    const bool mouthFP64Qualified =
        mouthMatchesOracle(mouthInside) &&
        mouthMatchesOracle(mouthReleased) &&
        mouthMatchesOracle(mouthOutside) &&
        mouthMatchesOracle(mouthGrazing) &&
        mouthMatchesOracle(mouthEdgeClearance) &&
        mouthMatchesOracle(mouthEdgeExit) &&
        mouthMatchesOracle(mouthRotated);
    const double mouthReleaseClearance =
        mouthClearance(mouthReleased.second);
    const double mouthRotatedClearance =
        mouthClearance(mouthRotated.second);
    const double fruitPairProbeConeRatio = std::bit_cast<float>(
        fruitPairGPU.frictionStatus.metrics.x
    );
    const double groundProbeConeRatio = std::bit_cast<float>(
        groundContactGPU.frictionStatus.metrics.x
    );
    const double groundProbeRollingRatio = std::bit_cast<float>(
        groundContactGPU.frictionStatus.metrics.y
    );
    const double yarnCCDProbeConeRatio = std::bit_cast<float>(
        yarnCCDGPU.frictionStatus.metrics.x
    );
    const double selfCCDProbeConeRatio = std::bit_cast<float>(
        selfCCDGPU.frictionStatus.metrics.x
    );
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
        initial.fruits.size() == kFruitCount &&
        initial.fruitPairs.size() == kFruitPairCount &&
        initial.yarnContacts.size() == kFruitYarnCount &&
        initial.selfPairs.size() > 4'000'000u &&
        !initial.selfBatches.empty() &&
        initial.maximumSelfBatchSize <= 256u &&
        coloringExact && gpu.failure == NUMI_CLOTH_BAG_GPU_FAILURE_NONE &&
        deterministic && maximumPositionError <= 2.0e-5 &&
        maximumVelocityError <= 0.12 &&
        maximumDistanceLambdaError <= 2.0e-8 &&
        maximumGripLambdaError <= 2.0e-8 &&
        maximumKnotLambdaError <= 2.0e-8 &&
        maximumBendLambdaError <= 2.0e-8 &&
        maximumKnotLambda > 1.0e-12 && maximumBendLambda > 1.0e-12 &&
        maximumFruitPositionError <= 2.0e-6 &&
        maximumFruitVelocityError <= 0.02 &&
        maximumFruitAngularVelocityError <= 2.0e-4 &&
        maximumFruitOrientationError <= 2.0e-6 &&
        maximumFruitOrientationNormError <= 2.0e-7 &&
        maximumYarnAerodynamicForceError <= 2.0e-6 &&
        maximumFruitAerodynamicForceError <= 2.0e-6 &&
        maximumFruitAerodynamicTorqueError <= 2.0e-8 &&
        gpuMaximumYarnAerodynamicForce > 0.0 &&
        gpuMaximumFruitAerodynamicForce > 0.0 &&
        releaseMasksExact && releaseClearanceQualified &&
        maximumFruitPairContactError <= 2.0e-5 &&
        maximumFruitPairPenetration <= 2.0e-6 &&
        yarnIdentityExact && yarnControlQualified &&
        maximumYarnSeparationError <= 2.0e-5 &&
        maximumYarnCurrentNormalError <= 5.0e-4 &&
        maximumYarnSweptNormalError <= 5.0e-4 &&
        maximumActiveYarnWeightError <= 2.0e-4 &&
        maximumYarnImpactTimeError <= 2.0e-4 &&
        maximumYarnAdvanceError <= 2.0e-5 &&
        maximumYarnResponseError <= 5.0e-2 &&
        maximumYarnPenetration <= 2.0e-6 &&
        acceptedYarnResponseCount > 0u && maximumYarnNormalImpulse > 0.0 &&
        presentSelfContactCountDifference <= 16u &&
        sweptSelfContactCountDifference <= 4u &&
        maximumSelfCorrectionError <= 2.0e-5 &&
        finalSelfPenetration <= 2.0e-6 &&
        finalGroundPenetration <= 1.0e-9 &&
        gpuPresentSelfContacts + gpuSweptSelfContacts > 0u &&
        maximumFrictionContactCountDifference <= 4u &&
        gpuFrictionContacts[1] > 0u &&
        gpuMaximumFrictionConeRatio <= 1.0 + 1.0e-6 &&
        maximumFrictionConeRatioError <= 2.0e-5 &&
        gpuMaximumRollingResistanceRatio <= 1.0 + 1.0e-6 &&
        maximumRollingResistanceRatioError <= 2.0e-5 &&
        rollingContactCountExact &&
        selfFrictionContactCountDifference <= 4u &&
        gpuSelfFrictionContacts > 0u &&
        strainViolation <= 2.0e-6 &&
        maximumDisplacement > 1.0e-4 && gripForce > 1.0 &&
        strainGPU.failure == NUMI_CLOTH_BAG_GPU_FAILURE_NONE &&
        probeInitialViolation > 0.20 && probeFinalViolation <= 2.0e-7 &&
        probePositionError <= 2.0e-7 &&
        std::abs(compressedFinalLength - compressedInitialLength) <= 1.0e-7 &&
        probeCenterOfMassError <= 1.0e-7 &&
        groundBendGPU.failure == NUMI_CLOTH_BAG_GPU_FAILURE_NONE &&
        groundBendPositionError <= 2.0e-7 &&
        supportedHeight >= 0.004 - 1.0e-8 && freeEndpointRise > 0.0 &&
        fruitPairGPU.failure == NUMI_CLOTH_BAG_GPU_FAILURE_NONE &&
        fruitPairProbePositionError <= 2.0e-7 &&
        std::abs(fruitPairProbeSeparation - 2.0) <= 2.0e-7 &&
        fruitPairProbeCenterError <= 1.0e-7 &&
        fruitPairProbeImpulseError <= 2.0e-6 &&
        fruitPairProbeImpulse > 0.0 &&
        fruitPairProbeVelocityError <= 1.0e-4 &&
        fruitPairProbeAngularVelocityError <= 1.0e-4 &&
        fruitPairProbeOrientationError <= 2.0e-6 &&
        fruitPairProbeOrientationChange > 1.0e-6 &&
        fruitPairGPU.frictionStatus.counters.x ==
            fruitPairOracle.frictionContacts[0] &&
        fruitPairGPU.frictionStatus.counters.x > 0u &&
        fruitPairProbeConeRatio <= 1.0 + 1.0e-6 &&
        fruitPairSlipAfter < fruitPairSlipBefore &&
        fruitPairEnergyAfter < fruitPairEnergyBefore &&
        fruitPairMomentumError <= 1.0e-5 &&
        groundContactGPU.failure == NUMI_CLOTH_BAG_GPU_FAILURE_NONE &&
        groundContactPositionError <= 1.0e-7 &&
        groundContactImpulseError <= 2.0e-6 &&
        groundClothImpulseError <= 2.0e-6 &&
        groundClothHeight >= 0.004 - 1.0e-8 &&
        groundFruitHeight >= 1.0 - 1.0e-8 &&
        groundFrictionVelocityError <= 1.0e-4 &&
        groundContactGPU.frictionStatus.counters.z ==
            groundContactOracle.frictionContacts[2] &&
        groundContactGPU.frictionStatus.counters.w ==
            groundContactOracle.frictionContacts[3] &&
        groundContactGPU.frictionStatus.counters.z > 0u &&
        groundContactGPU.frictionStatus.counters.w > 0u &&
        groundContactGPU.frictionStatus.metrics.z ==
            groundContactOracle.rollingContacts &&
        groundContactGPU.frictionStatus.metrics.z > 0u &&
        groundProbeConeRatio <= 1.0 + 1.0e-6 &&
        groundProbeRollingRatio <= 1.0 + 1.0e-6 &&
        groundClothSlipAfter < groundClothSlipBefore &&
        groundFruitSlipAfter < groundFruitSlipBefore &&
        groundRollingSpeedAfter < groundRollingSpeedBefore &&
        yarnCCDGPU.failure == NUMI_CLOTH_BAG_GPU_FAILURE_NONE &&
        yarnCCDFlagsQualified && yarnCCDGeometryError <= 5.0e-6 &&
        yarnCCDResponseError <= 1.0e-4 &&
        yarnCCDImpactTime > 0.30 && yarnCCDImpactTime < 0.50 &&
        yarnCCDRemovedAdvance > 0.05 &&
        std::abs(yarnCCDFinalSeparation) <= 2.0e-6 &&
        yarnCCDFinalFruitX < 0.0 && yarnCCDNormalImpulse > 1.0 &&
        yarnCCDFrictionVelocityError <= 1.0e-3 &&
        yarnCCDGPU.frictionStatus.counters.y ==
            yarnCCDOracle.frictionContacts[1] &&
        yarnCCDGPU.frictionStatus.counters.y > 0u &&
        yarnCCDProbeConeRatio <= 1.0 + 1.0e-6 &&
        yarnCCDSlipAfter < yarnCCDSlipBefore &&
        selfCCDGPU.failure == NUMI_CLOTH_BAG_GPU_FAILURE_NONE &&
        selfCCDGPU.selfStatus.counters.x == 0u &&
        selfCCDGPU.selfStatus.counters.y == 1u &&
        selfCCDPositionError <= 2.0e-6 &&
        std::abs(selfCCDFinalSeparation) <= 5.0e-6 &&
        selfCCDMaximumCorrection > 0.08 &&
        selfCCDFrictionVelocityError <= 1.0e-3 &&
        selfCCDGPU.frictionStatus.metrics.w ==
            selfCCDOracle.selfFrictionContacts &&
        selfCCDGPU.frictionStatus.metrics.w > 0u &&
        selfCCDProbeConeRatio <= 1.0 + 1.0e-6 &&
        selfCCDSlipAfter < selfCCDSlipBefore &&
        selfCCDEnergyAfter < selfCCDEnergyBefore &&
        selfCCDMomentumError <= 1.0e-5 &&
        yarnAerodynamicsGPU.failure == NUMI_CLOTH_BAG_GPU_FAILURE_NONE &&
        refinedYarnAerodynamicsGPU.failure ==
            NUMI_CLOTH_BAG_GPU_FAILURE_NONE &&
        coMovingYarnAerodynamicsGPU.failure ==
            NUMI_CLOTH_BAG_GPU_FAILURE_NONE &&
        yarnAerodynamicsVelocityError <= 1.0e-5 &&
        yarnAerodynamicsProbeForceError <= 1.0e-6 &&
        yarnAerodynamicsProbeForce > 0.0 &&
        yarnAerodynamicsEnergyAfter < yarnAerodynamicsEnergyBefore &&
        yarnAerodynamicsRefinementError <= 1.0e-6 &&
        coMovingYarnAerodynamicsDelta <= 5.0e-6 &&
        fruitAerodynamicsGPU.failure == NUMI_CLOTH_BAG_GPU_FAILURE_NONE &&
        fruitAerodynamicsVelocityError <= 5.0e-5 &&
        fruitAerodynamicsProbeForceError <= 1.0e-6 &&
        fruitAerodynamicsProbeTorqueError <= 1.0e-8 &&
        fruitAerodynamicsProbeForce > 0.0 &&
        fruitAerodynamicsProbeTorque > 0.0 &&
        fruitAerodynamicsEnergyAfter < fruitAerodynamicsEnergyBefore &&
        mouthFP64Qualified &&
        mouthInside.second.releaseStatus.masks.y == 0u &&
        mouthReleased.second.releaseStatus.masks.y == 1u &&
        mouthOutside.second.releaseStatus.masks.y == 0u &&
        mouthGrazing.second.releaseStatus.masks.y == 0u &&
        mouthEdgeClearance.second.releaseStatus.masks.y == 1u &&
        mouthEdgeExit.second.releaseStatus.masks.y == 1u &&
        mouthRotated.second.releaseStatus.masks.y == 1u &&
        std::abs(mouthReleaseClearance - 0.026) <= 2.0e-6 &&
        std::abs(mouthRotatedClearance - 0.026) <= 2.0e-6 &&
        seamTrajectoryGPU.failure == NUMI_CLOTH_BAG_GPU_FAILURE_NONE &&
        seamTrajectoryReplayExact && seamTrajectorySplitExact &&
        seamAverageLift > 0.0 && seamMaximumHandleLag < 0.02 &&
        gripRotationGPU.failure == NUMI_CLOTH_BAG_GPU_FAILURE_NONE &&
        gripRotationReplayExact && gripRotationPositionError <= 2.0e-6 &&
        gripRotationDisplacement > 1.0e-4 && distantRegrabRejected &&
        groundedQualified && spinQualified && pickupQualified &&
        recordedQualified;

    std::cout << std::fixed << std::setprecision(12)
              << "device=" << device.name.UTF8String << '\n'
              << "material_schema=" << numi::kClothMaterialSchema
              << " material_artifact_loaded=" << std::boolalpha
              << gClothMaterial.loaded
              << " parameters_hash=" << gClothMaterial.parametersHash
              << " observations_hash=" << gClothMaterial.observationsHash
              << '\n'
              << "grip_trajectory_loaded=" << recordedRequested
              << " schema="
              << (recordedRequested
                      ? gripTrajectory.schema
                      : "none")
              << " content_fingerprint="
              << (recordedRequested
                      ? gripTrajectory.contentFingerprint
                      : "none")
              << " poses="
              << (recordedRequested ? gripTrajectory.poses.size() : 0u)
              << " duration_seconds="
              << (recordedRequested
                      ? gripTrajectory.poses.back().timeSeconds
                      : 0.0)
              << " maximum_rotation_radians="
              << (recordedRequested
                      ? numi::maximumGripTrajectoryRotation(gripTrajectory)
                      : 0.0)
              << " attachment_generations="
              << (recordedRequested
                      ? recordedAttachmentGenerations
                      : 0u)
              << " selection_mode="
              << (recordedDynamicPatchSelection
                      ? "nearest_cuff_patch"
                      : "fixed_patch")
              << '\n'
              << "abi=" << NUMI_CLOTH_BAG_GPU_ABI_VERSION
              << " particles=" << initial.particles.size()
              << " distances=" << initial.distances.size()
              << " grips=" << initial.grips.size()
              << " knots=" << initial.knots.size()
              << " bends=" << initial.bends.size()
              << " fruits=" << initial.fruits.size()
              << " fruit_pairs=" << initial.fruitPairs.size()
              << " fruit_yarn_candidates=" << initial.yarnContacts.size()
              << '\n'
              << "distance_colors=" << initial.distanceBatches.size()
              << " knot_colors=" << initial.knotBatches.size()
              << " bend_colors=" << initial.bendBatches.size()
              << " fruit_pair_colors="
              << initial.fruitPairBatches.size() << '\n'
              << "self_pairs=" << initial.selfPairs.size()
              << " self_batches=" << initial.selfBatches.size()
              << " max_self_batch=" << initial.maximumSelfBatchSize << '\n'
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
              << "max_fruit_position_error=" << maximumFruitPositionError
              << " max_fruit_velocity_error=" << maximumFruitVelocityError
              << " max_fruit_angular_velocity_error="
              << maximumFruitAngularVelocityError
              << " max_fruit_orientation_error="
              << maximumFruitOrientationError
              << " max_fruit_orientation_norm_error="
              << maximumFruitOrientationNormError
              << " max_fruit_pair_contact_error="
              << maximumFruitPairContactError
              << " max_fruit_pair_impulse=" << maximumFruitPairImpulse
              << " max_fruit_pair_penetration="
              << maximumFruitPairPenetration << '\n'
              << "max_yarn_aerodynamic_force="
              << gpuMaximumYarnAerodynamicForce
              << " expected=" << oracle.maximumYarnAerodynamicForce
              << " error=" << maximumYarnAerodynamicForceError
              << " max_fruit_aerodynamic_force="
              << gpuMaximumFruitAerodynamicForce
              << " expected_fruit="
              << oracle.maximumFruitAerodynamicForce
              << " fruit_error=" << maximumFruitAerodynamicForceError
              << " max_fruit_aerodynamic_torque="
              << gpuMaximumFruitAerodynamicTorque
              << " expected_torque="
              << oracle.maximumFruitAerodynamicTorque
              << " torque_error=" << maximumFruitAerodynamicTorqueError
              << '\n'
              << "mouth_candidate_mask="
              << gpu.releaseStatus.masks.x
              << " expected_candidate_mask=" << oracle.mouthCandidateMask
              << " released_mask=" << gpu.releaseStatus.masks.y
              << " expected_released_mask=" << oracle.releasedMask
              << " max_release_clearance_error="
              << maximumReleaseClearanceError << '\n'
              << "yarn_identity_exact=" << yarnIdentityExact
              << " yarn_control_exact=" << yarnControlExact
              << " yarn_control_qualified=" << yarnControlQualified
              << " current_yarn_overlaps=" << currentYarnOverlapCount
              << " swept_yarn_impacts=" << sweptYarnImpactCount << '\n'
              << "max_yarn_separation_error="
              << maximumYarnSeparationError
              << " max_yarn_current_normal_error="
              << maximumYarnCurrentNormalError
              << " max_yarn_swept_normal_error="
              << maximumYarnSweptNormalError << '\n'
              << "max_yarn_weight_error=" << maximumYarnWeightError
              << " max_active_yarn_weight_error="
              << maximumActiveYarnWeightError
              << " max_yarn_impact_time_error="
              << maximumYarnImpactTimeError
              << " max_yarn_advance_error="
              << maximumYarnAdvanceError << '\n'
              << "yarn_response_count_exact=" << yarnResponseCountExact
              << " accepted_yarn_responses=" << acceptedYarnResponseCount
              << " expected_yarn_responses=" << expectedYarnResponseCount
              << " response_count_mismatches="
              << yarnResponseCountMismatches
              << " max_yarn_normal_impulse="
              << maximumYarnNormalImpulse
              << " max_yarn_response_error="
              << maximumYarnResponseError
              << " max_yarn_penetration=" << maximumYarnPenetration << '\n'
              << "present_self_contacts=" << gpuPresentSelfContacts
              << " expected_present_self_contacts="
              << oracle.presentSelfContacts
              << " swept_self_contacts=" << gpuSweptSelfContacts
              << " expected_swept_self_contacts="
              << oracle.sweptSelfContacts << '\n'
              << "present_self_count_difference="
              << presentSelfContactCountDifference
              << " swept_self_count_difference="
              << sweptSelfContactCountDifference
              << " max_self_correction=" << gpuMaximumSelfCorrection
              << " max_self_correction_error="
              << maximumSelfCorrectionError
              << " final_self_penetration=" << finalSelfPenetration
              << " final_ground_penetration="
              << finalGroundPenetration << '\n'
              << "max_strain_violation=" << strainViolation
              << " max_displacement=" << maximumDisplacement
              << " grip_force=" << gripForce << '\n'
              << "reconciliation_passes=" << kReconciliationPasses
              << " final_contact_passes=" << kFinalContactPasses
              << " certificate_passes=" << kCertificatePasses << '\n'
              << "friction_contacts_pair=" << gpuFrictionContacts[0]
              << " expected_pair=" << oracle.frictionContacts[0]
              << " yarn=" << gpuFrictionContacts[1]
              << " expected_yarn=" << oracle.frictionContacts[1]
              << " cloth_ground=" << gpuFrictionContacts[2]
              << " expected_cloth_ground=" << oracle.frictionContacts[2]
              << " fruit_ground=" << gpuFrictionContacts[3]
              << " expected_fruit_ground=" << oracle.frictionContacts[3]
              << " self=" << gpuSelfFrictionContacts
              << " expected_self=" << oracle.selfFrictionContacts
              << " count_exact=" << frictionContactCountsExact
              << " max_count_difference="
              << maximumFrictionContactCountDifference
              << " self_count_difference="
              << selfFrictionContactCountDifference << '\n'
              << "max_friction_cone_ratio="
              << gpuMaximumFrictionConeRatio
              << " expected=" << oracle.maximumFrictionConeRatio
              << " error=" << maximumFrictionConeRatioError
              << " max_rolling_ratio="
              << gpuMaximumRollingResistanceRatio
              << " expected_rolling="
              << oracle.maximumRollingResistanceRatio
              << " rolling_error="
              << maximumRollingResistanceRatioError
              << " rolling_contacts="
              << gpu.frictionStatus.metrics.z
              << " expected_rolling_contacts=" << oracle.rollingContacts
              << '\n'
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
              << "fruit_pair_probe_position_error="
              << fruitPairProbePositionError
              << " separation=" << fruitPairProbeSeparation
              << " center_error=" << fruitPairProbeCenterError
              << " impulse_error=" << fruitPairProbeImpulseError
              << " normal_impulse=" << fruitPairProbeImpulse
              << " velocity_error=" << fruitPairProbeVelocityError
              << " angular_velocity_error="
              << fruitPairProbeAngularVelocityError
              << " orientation_error="
              << fruitPairProbeOrientationError
              << " orientation_change="
              << fruitPairProbeOrientationChange
              << " friction_contacts="
              << fruitPairGPU.frictionStatus.counters.x
              << " cone_ratio=" << fruitPairProbeConeRatio
              << " slip_before=" << fruitPairSlipBefore
              << " slip_after=" << fruitPairSlipAfter
              << " energy_before=" << fruitPairEnergyBefore
              << " energy_after=" << fruitPairEnergyAfter
              << " momentum_error=" << fruitPairMomentumError
              << " fruit_pair_probe_failure_flags=" << fruitPairGPU.failure
              << '\n'
              << "ground_contact_position_error="
              << groundContactPositionError
              << " ground_contact_impulse_error="
              << groundContactImpulseError
              << " cloth_impulse_error=" << groundClothImpulseError
              << " cloth_height=" << groundClothHeight
              << " fruit_height=" << groundFruitHeight
              << " fruit_normal_impulse=" << groundFruitImpulse
              << " friction_velocity_error="
              << groundFrictionVelocityError
              << " cloth_friction_contacts="
              << groundContactGPU.frictionStatus.counters.z
              << " fruit_friction_contacts="
              << groundContactGPU.frictionStatus.counters.w
              << " rolling_contacts="
              << groundContactGPU.frictionStatus.metrics.z
              << " cone_ratio=" << groundProbeConeRatio
              << " rolling_ratio=" << groundProbeRollingRatio
              << " cloth_slip_before=" << groundClothSlipBefore
              << " cloth_slip_after=" << groundClothSlipAfter
              << " fruit_slip_before=" << groundFruitSlipBefore
              << " fruit_slip_after=" << groundFruitSlipAfter
              << " rolling_speed_before=" << groundRollingSpeedBefore
              << " rolling_speed_after=" << groundRollingSpeedAfter
              << " ground_contact_failure_flags=" << groundContactGPU.failure
              << '\n'
              << "yarn_ccd_geometry_error=" << yarnCCDGeometryError
              << " impact_time=" << yarnCCDImpactTime
              << " removed_advance=" << yarnCCDRemovedAdvance
              << " current_overlap="
              << (yarnCCDContact.control.x != 0u)
              << " swept_impact=" << (yarnCCDContact.control.y != 0u)
              << " response_count=" << yarnCCDContact.control.w
              << " normal_impulse=" << yarnCCDNormalImpulse
              << " final_separation=" << yarnCCDFinalSeparation
              << " final_fruit_x=" << yarnCCDFinalFruitX
              << " response_error=" << yarnCCDResponseError
              << " friction_velocity_error="
              << yarnCCDFrictionVelocityError
              << " friction_contacts="
              << yarnCCDGPU.frictionStatus.counters.y
              << " cone_ratio=" << yarnCCDProbeConeRatio
              << " slip_before=" << yarnCCDSlipBefore
              << " slip_after=" << yarnCCDSlipAfter
              << " yarn_ccd_failure_flags=" << yarnCCDGPU.failure << '\n'
              << "self_ccd_position_error=" << selfCCDPositionError
              << " final_separation=" << selfCCDFinalSeparation
              << " final_moving_height=" << selfCCDFinalMovingHeight
              << " max_correction=" << selfCCDMaximumCorrection
              << " present_contacts="
              << selfCCDGPU.selfStatus.counters.x
              << " swept_contacts=" << selfCCDGPU.selfStatus.counters.y
              << " friction_velocity_error="
              << selfCCDFrictionVelocityError
              << " friction_contacts="
              << selfCCDGPU.frictionStatus.metrics.w
              << " cone_ratio=" << selfCCDProbeConeRatio
              << " slip_before=" << selfCCDSlipBefore
              << " slip_after=" << selfCCDSlipAfter
              << " energy_before=" << selfCCDEnergyBefore
              << " energy_after=" << selfCCDEnergyAfter
              << " momentum_error=" << selfCCDMomentumError
              << " self_ccd_failure_flags=" << selfCCDGPU.failure << '\n'
              << "yarn_aerodynamics_velocity_error="
              << yarnAerodynamicsVelocityError
              << " force=" << yarnAerodynamicsProbeForce
              << " force_error=" << yarnAerodynamicsProbeForceError
              << " energy_before=" << yarnAerodynamicsEnergyBefore
              << " energy_after=" << yarnAerodynamicsEnergyAfter
              << " refinement_error="
              << yarnAerodynamicsRefinementError
              << " co_moving_delta=" << coMovingYarnAerodynamicsDelta
              << " yarn_aerodynamics_failure_flags="
              << yarnAerodynamicsGPU.failure << '\n'
              << "fruit_aerodynamics_velocity_error="
              << fruitAerodynamicsVelocityError
              << " force=" << fruitAerodynamicsProbeForce
              << " force_error=" << fruitAerodynamicsProbeForceError
              << " torque=" << fruitAerodynamicsProbeTorque
              << " torque_error=" << fruitAerodynamicsProbeTorqueError
              << " energy_before=" << fruitAerodynamicsEnergyBefore
              << " energy_after=" << fruitAerodynamicsEnergyAfter
              << " fruit_aerodynamics_failure_flags="
              << fruitAerodynamicsGPU.failure << '\n'
              << "mouth_probe_inside_mask="
              << mouthInside.second.releaseStatus.masks.y
              << " released_mask="
              << mouthReleased.second.releaseStatus.masks.y
              << " outside_mask="
              << mouthOutside.second.releaseStatus.masks.y
              << " grazing_mask="
              << mouthGrazing.second.releaseStatus.masks.y
              << " edge_clearance_mask="
              << mouthEdgeClearance.second.releaseStatus.masks.y
              << " edge_exit_mask="
              << mouthEdgeExit.second.releaseStatus.masks.y
              << " rotated_mask="
              << mouthRotated.second.releaseStatus.masks.y
              << " clearance=" << mouthReleaseClearance
              << " rotated_clearance=" << mouthRotatedClearance
              << " fp64_qualified=" << mouthFP64Qualified << '\n'
              << "seam_trajectory_substeps="
              << seamTrajectoryConfigs.size()
              << " replay_exact=" << seamTrajectoryReplayExact
              << " split_exact=" << seamTrajectorySplitExact
              << " average_lift=" << seamAverageLift
              << " maximum_handle_lag=" << seamMaximumHandleLag
              << " failure_flags=" << seamTrajectoryGPU.failure
              << " state_hash=0x" << std::hex
              << hashGPUResult(seamTrajectoryGPU) << std::dec << '\n'
              << "grip_rotation_angle_radians="
              << 2.0 * gripProbeHalfAngle
              << " position_error=" << gripRotationPositionError
              << " seam_displacement=" << gripRotationDisplacement
              << " replay_exact=" << gripRotationReplayExact
              << " failure_flags=" << gripRotationGPU.failure
              << " state_hash=0x" << std::hex
              << hashGPUResult(gripRotationGPU) << std::dec << '\n'
              << "distant_regrab_rejected=" << distantRegrabRejected
              << " failure_flags=" << distantRegrabGPU.failure << '\n'
              << "grounded_requested=" << groundedRequested
              << " complete=" << groundedComplete
              << " steps=" << (groundedRequested ? groundedSteps : 0u)
              << " replay_exact=" << groundedReplayExact
              << " released_mask=" << groundedReleasedMask
              << " escape_mask=" << groundedEscapeMask
              << " grounded_cloth_count=" << groundedClothContacts
              << " grounded_fruit_count=" << groundedFruitContacts
              << " minimum_cloth_height=" << groundedMinimumClothHeight
              << " minimum_fruit_clearance="
              << groundedMinimumFruitClearance
              << " strain_violation=" << groundedStrainViolation
              << " ground_penetration=" << groundedGroundPenetration
              << " self_penetration=" << groundedSelfPenetration
              << " shape_change=" << groundedShapeChange
              << " kinetic_energy=" << groundedKineticEnergy
              << " first_gpu_seconds=" << groundedFirst.gpuSeconds
              << " second_gpu_seconds=" << groundedSecond.gpuSeconds
              << " qualified=" << groundedQualified << '\n'
              << "spin_requested=" << spinRequested
              << " complete=" << spinComplete
              << " steps=" << (spinRequested ? spinSteps : 0u)
              << " replay_exact=" << spinReplayExact
              << " released_mask=" << spinReleasedMask
              << " released_count=" << std::popcount(spinReleasedMask)
              << " escape_mask=" << spinEscapeMask
              << " minimum_cloth_height=" << spinMinimumClothHeight
              << " minimum_fruit_clearance=" << spinMinimumFruitClearance
              << " strain_violation=" << spinStrainViolation
              << " self_penetration=" << spinSelfPenetration
              << " handle_travel=" << spinHandleTravel
              << " maximum_handle_lag=" << spinFirst.maximumHandleLag
              << " shape_change=" << spinShapeChange
              << " kinetic_energy=" << spinKineticEnergy
              << " first_gpu_seconds=" << spinFirst.gpuSeconds
              << " second_gpu_seconds=" << spinSecond.gpuSeconds
              << " qualified=" << spinQualified << '\n'
              << "pickup_requested=" << pickupRequested
              << " complete=" << pickupComplete
              << " steps=" << (pickupRequested ? pickupSteps : 0u)
              << " motion_frames=" << kPickupMotionFrames
              << " settling_frames="
              << (pickupRequested && pickupSteps > kPickupMotionFrames
                      ? pickupSteps - kPickupMotionFrames
                      : 0u)
              << " replay_exact=" << pickupReplayExact
              << " released_mask=" << pickupReleasedMask
              << " released_count=" << std::popcount(pickupReleasedMask)
              << " grounded_released_count="
              << pickupGroundedReleasedCount
              << " strain_violation=" << pickupStrainViolation
              << " ground_penetration=" << pickupGroundPenetration
              << " self_penetration=" << pickupSelfPenetration
              << " first_gpu_seconds=" << pickupFirst.gpuSeconds
              << " second_gpu_seconds=" << pickupSecond.gpuSeconds
              << " qualified=" << pickupQualified << '\n'
              << "recorded_requested=" << recordedRequested
              << " steps=" << (recordedRequested ? recordedSteps : 0u)
              << " replay_exact=" << recordedReplayExact
              << " released_mask=" << recordedReleasedMask
              << " released_count="
              << std::popcount(recordedReleasedMask)
              << " escape_mask=" << recordedEscapeMask
              << " strain_violation=" << recordedStrainViolation
              << " ground_penetration=" << recordedGroundPenetration
              << " self_penetration=" << recordedSelfPenetration
              << " maximum_handle_lag="
              << recordedFirst.maximumHandleLag
              << " regrab_count="
              << (recordedAttachmentGenerations > 0u
                      ? recordedAttachmentGenerations - 1u
                      : 0u)
              << " inactive_grip_substeps="
              << recordedInactiveSubsteps
              << " attachment_generations_exact="
              << recordedAttachmentGenerationsExact
              << " maximum_regrab_capture_distance="
              << recordedMaximumCaptureDistance
              << " maximum_regrab_capture_error="
              << recordedMaximumCaptureError
              << " patch_selection_count="
              << (recordedDynamicPatchSelection &&
                          recordedAttachmentGenerations > 0u
                      ? recordedAttachmentGenerations - 1u
                      : 0u)
              << " selected_patch_center_ring="
              << recordedPatchCenterRing
              << " patch_center_exact="
              << recordedPatchCenterExact
              << " patch_topology_exact="
              << recordedPatchTopologyExact
              << " shape_change=" << recordedShapeChange
              << " first_gpu_seconds=" << recordedFirst.gpuSeconds
              << " second_gpu_seconds=" << recordedSecond.gpuSeconds
              << " qualified=" << recordedQualified
              << " state_hash=0x" << std::hex
              << (recordedRequested
                      ? hashGPUResult(recordedFirst.final)
                      : 0u)
              << std::dec << '\n'
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
