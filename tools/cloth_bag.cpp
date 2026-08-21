#include "numi/cloth_material.h"
#include "numi/grip_trajectory.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kAround = 48;
constexpr std::uint32_t kLevels = 28;
constexpr std::uint32_t kBottomGrid = 13;
constexpr std::uint32_t kBottomInterior = kBottomGrid - 2u;
constexpr std::size_t kFruitCount = 12;
constexpr double kAirborneLift = 0.52;
constexpr double kInitialGroundLift = 0.009;
numi::ClothMaterialArtifact gClothMaterial{};
double kClothRadius = gClothMaterial.values.yarnRadiusM;
double kClothNodeMass = gClothMaterial.values.ordinaryNodeMassKg;
double kClothHemNodeMass = gClothMaterial.values.hemNodeMassKg;
double kClothMass =
    static_cast<double>(
        kAround * (kLevels - 2u) + kBottomInterior * kBottomInterior
    ) * kClothNodeMass +
    static_cast<double>(2u * kAround) * kClothHemNodeMass;
double kFruitGroundFriction = gClothMaterial.values.fruitGroundFriction;
double kClothGroundFriction = gClothMaterial.values.clothGroundFriction;
double kFruitPairFriction = gClothMaterial.values.fruitPairFriction;
double kFruitClothFriction = gClothMaterial.values.fruitYarnFriction;
double kClothSelfFriction = gClothMaterial.values.clothSelfFriction;
double kFruitRollingResistanceCoefficient =
    gClothMaterial.values.fruitRollingResistance;
constexpr double kAirDensity = 1.225;
double kYarnCrossflowDragCoefficient =
    gClothMaterial.values.yarnCrossflowDrag;
double kYarnSkinFrictionCoefficient =
    gClothMaterial.values.yarnSkinFriction;
double kFruitDragCoefficient = gClothMaterial.values.fruitDrag;
double kFruitRotationalDragCoefficient =
    gClothMaterial.values.fruitRotationalDrag;
double kAxialBodyCompliance =
    gClothMaterial.values.axialBodyComplianceMPerN;
double kAxialCuffCompliance =
    gClothMaterial.values.axialCuffComplianceMPerN;
double kKnotCompliance = gClothMaterial.values.knotCompliance;
double kBendBodyCompliance =
    gClothMaterial.values.bendBodyComplianceMPerN;
double kBendCuffCompliance =
    gClothMaterial.values.bendCuffComplianceMPerN;
double kGripCompliance = gClothMaterial.values.gripComplianceMPerN;
constexpr double kGripCaptureRadius = 0.12;
constexpr double kMouthReleaseHysteresis = 0.025;
constexpr std::size_t kBallPairCount =
    kFruitCount * (kFruitCount - 1u) / 2u;

enum class Scenario : std::uint8_t {
    grounded,
    spin,
    pickup,
    recorded,
};

void applyClothMaterial(const numi::ClothMaterialArtifact& artifact) {
    gClothMaterial = artifact;
    kClothRadius = artifact.values.yarnRadiusM;
    kClothNodeMass = artifact.values.ordinaryNodeMassKg;
    kClothHemNodeMass = artifact.values.hemNodeMassKg;
    kClothMass = static_cast<double>(
        kAround * (kLevels - 2u) + kBottomInterior * kBottomInterior
    ) * kClothNodeMass +
        static_cast<double>(2u * kAround) * kClothHemNodeMass;
    kFruitGroundFriction = artifact.values.fruitGroundFriction;
    kClothGroundFriction = artifact.values.clothGroundFriction;
    kFruitPairFriction = artifact.values.fruitPairFriction;
    kFruitClothFriction = artifact.values.fruitYarnFriction;
    kClothSelfFriction = artifact.values.clothSelfFriction;
    kFruitRollingResistanceCoefficient =
        artifact.values.fruitRollingResistance;
    kYarnCrossflowDragCoefficient = artifact.values.yarnCrossflowDrag;
    kYarnSkinFrictionCoefficient = artifact.values.yarnSkinFriction;
    kFruitDragCoefficient = artifact.values.fruitDrag;
    kFruitRotationalDragCoefficient = artifact.values.fruitRotationalDrag;
    kAxialBodyCompliance = artifact.values.axialBodyComplianceMPerN;
    kAxialCuffCompliance = artifact.values.axialCuffComplianceMPerN;
    kKnotCompliance = artifact.values.knotCompliance;
    kBendBodyCompliance = artifact.values.bendBodyComplianceMPerN;
    kBendCuffCompliance = artifact.values.bendCuffComplianceMPerN;
    kGripCompliance = artifact.values.gripComplianceMPerN;
}

struct Vec3 {
    double x{};
    double y{};
    double z{};
};

Vec3 operator+(const Vec3 a, const Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator-(const Vec3 a, const Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 operator*(const Vec3 a, const double s) {
    return {a.x * s, a.y * s, a.z * s};
}

Vec3 operator/(const Vec3 a, const double s) {
    return a * (1.0 / s);
}

Vec3& operator+=(Vec3& a, const Vec3 b) {
    a = a + b;
    return a;
}

Vec3& operator-=(Vec3& a, const Vec3 b) {
    a = a - b;
    return a;
}

double dot(const Vec3 a, const Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(const Vec3 a, const Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

double lengthSquared(const Vec3 v) {
    return dot(v, v);
}

double length(const Vec3 v) {
    return std::sqrt(lengthSquared(v));
}

Vec3 normalized(const Vec3 v) {
    const double magnitude = length(v);
    return magnitude > 1.0e-14 ? v / magnitude : Vec3{1.0, 0.0, 0.0};
}

bool finite(const Vec3 v) {
    return std::isfinite(v.x) && std::isfinite(v.y) &&
           std::isfinite(v.z);
}

struct Quaternion {
    double w{1.0};
    double x{};
    double y{};
    double z{};
};

Quaternion operator+(const Quaternion a, const Quaternion b) {
    return {a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z};
}

Quaternion operator*(const Quaternion q, const double s) {
    return {q.w * s, q.x * s, q.y * s, q.z * s};
}

Quaternion operator*(const Quaternion a, const Quaternion b) {
    return {
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    };
}

double quaternionLength(const Quaternion q) {
    return std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
}

Quaternion normalized(const Quaternion q) {
    const double magnitude = quaternionLength(q);
    return magnitude > 1.0e-14
        ? q * (1.0 / magnitude)
        : Quaternion{};
}

bool finite(const Quaternion q) {
    return std::isfinite(q.w) && std::isfinite(q.x) &&
        std::isfinite(q.y) && std::isfinite(q.z);
}

Vec3 rotateVector(const Quaternion q, const Vec3 value) {
    const Vec3 vectorPart{q.x, q.y, q.z};
    const Vec3 twiceCross = cross(vectorPart, value) * 2.0;
    return value + twiceCross * q.w + cross(vectorPart, twiceCross);
}

Vec3 inverseRotateVector(const Quaternion q, const Vec3 value) {
    return rotateVector({q.w, -q.x, -q.y, -q.z}, value);
}

Quaternion gripQuaternion(
    const numi::GripTrajectoryQuaternion quaternion
) {
    return {
        quaternion.w,
        quaternion.x,
        quaternion.y,
        quaternion.z,
    };
}

struct Particle {
    Vec3 position{};
    Vec3 previous{};
    Vec3 velocity{};
    Vec3 rest{};
    double inverseMass{};
    double mass{};
};

struct Ball {
    Vec3 position{};
    Vec3 previous{};
    Vec3 velocity{};
    Vec3 angularVelocity{};
    Quaternion orientation{};
    double radius{0.11};
    double inverseMass{1.0};
    std::uint32_t appearance{};
};

struct BallPairContactImpulse {
    Vec3 weightedNormal{};
    double normalImpulse{};
};

struct BallYarnContactImpulse {
    Vec3 weightedNormalOnBall{};
    std::array<double, 2> weightedSegment{};
    double normalImpulse{};
};

enum class DistanceKind : std::uint8_t {
    warp,
    weft,
    bottom,
};

struct DistanceConstraint {
    std::uint32_t first{};
    std::uint32_t second{};
    double restLength{};
    double compliance{};
    double lambda{};
    DistanceKind kind{};
};

struct Triangle {
    std::uint32_t first{};
    std::uint32_t second{};
    std::uint32_t third{};
};

struct Edge {
    std::uint32_t first{};
    std::uint32_t second{};
};

struct SegmentClosest {
    Vec3 firstPoint{};
    Vec3 secondPoint{};
    double firstWeight{};
    double secondWeight{};
};

SegmentClosest closestPointsOnSegments(
    Vec3 firstStart,
    Vec3 firstEnd,
    Vec3 secondStart,
    Vec3 secondEnd
);

double solveBallYarn(
    std::vector<Particle>& particles,
    Edge segment,
    Ball& ball,
    double timestep,
    bool groundEnabled,
    BallYarnContactImpulse& contactImpulse,
    std::uint64_t& contactCount
);

struct SelfEdgeEdgeContactImpulse {
    Edge first{};
    Edge second{};
    Vec3 weightedNormalOnFirst{};
    std::array<double, 2> weightedFirst{};
    std::array<double, 2> weightedSecond{};
    double normalImpulse{};
};

struct SelfContactImpulses {
    std::unordered_map<std::uint64_t, SelfEdgeEdgeContactImpulse> edgeEdge;
};

struct YarnBendConstraint {
    std::uint32_t first{};
    std::uint32_t middle{};
    std::uint32_t third{};
    double restChord{};
    double restArc{};
    double compliance{kBendBodyCompliance};
    double lambda{};
};

struct KnotConstraint {
    std::uint32_t warpFirst{};
    std::uint32_t warpSecond{};
    std::uint32_t weftFirst{};
    std::uint32_t weftSecond{};
    double restCosine{};
    double compliance{kKnotCompliance};
    double lambda{};
};

struct GripConstraint {
    std::uint32_t particle{};
    Vec3 targetOffset{};
    Vec3 lambda{};
    double compliance{kGripCompliance};
};

struct ClothModel {
    std::vector<Particle> particles;
    std::vector<DistanceConstraint> distances;
    std::vector<Triangle> renderTriangles;
    std::vector<Edge> yarnSegments;
    std::vector<YarnBendConstraint> bends;
    std::vector<KnotConstraint> knots;
    std::vector<GripConstraint> grips;
    std::vector<std::vector<std::uint32_t>> localTopology;
    std::uint32_t bottomCenter{};
    Vec3 gripTarget{};
    Vec3 gripPrevious{};
    Quaternion gripOrientation{};
    bool gripActive{};
    std::uint32_t gripAttachmentGeneration{1u};
    std::uint32_t gripPatchCenterRing{};
};

struct Metrics {
    double maximumWarpStrain{};
    double maximumWeftStrain{};
    double maximumWarpExtension{};
    double maximumWarpCompression{};
    double maximumWeftExtension{};
    double maximumWeftCompression{};
    double maximumBottomStrain{};
    double maximumBottomExtension{};
    double maximumBottomCompression{};
    std::uint32_t maximumWarpExtensionFirst{};
    std::uint32_t maximumWarpExtensionSecond{};
    double maximumBendError{};
    double maximumKnotAngleError{};
    double maximumBallPenetration{};
    double maximumPublishedBallPenetration{};
    double maximumPublishedPrimitiveSelfPenetration{};
    double maximumPublishedStrainLimitViolation{};
    double maximumSelfPenetration{};
    double maximumGroundPenetration{};
    double maximumClothGroundCorrection{};
    double maximumBallGroundCorrection{};
    double maximumPublishedGroundPenetration{};
    std::uint32_t maximumContactReconciliationPasses{};
    std::uint32_t maximumPrimitiveCertificatePasses{};
    double maximumSweptGroundAdvance{};
    double maximumSweptBallAdvance{};
    double minimumTriangleArea{std::numeric_limits<double>::infinity()};
    double maximumSpeed{};
    double maximumAngularSpeed{};
    double maximumYarnAerodynamicForce{};
    double maximumFruitAerodynamicForce{};
    double maximumFruitAerodynamicTorque{};
    double aerodynamicDissipation{};
    double maximumRollingResistanceRatio{};
    double maximumFrictionConeRatio{};
    double accumulatedTangentialImpulse{};
    double maximumGripForce{};
    double maximumGripImpulse{};
    double maximumRegrabCaptureDistance{};
    double maximumRegrabCaptureError{};
    std::uint32_t regrabCount{};
    std::uint64_t inactiveGripSubsteps{};
    std::uint32_t gripPatchSelectionCount{};
    std::uint32_t maximumGripPatchRingShift{};
    std::uint64_t ballYarnContacts{};
    std::uint64_t sweptBallYarnContacts{};
    std::uint64_t selfContacts{};
    std::uint64_t edgeEdgeSelfContacts{};
    std::uint64_t sweptEdgeEdgeSelfContacts{};
    std::uint64_t edgeEdgeCandidatePairs{};
    std::uint64_t edgeEdgeSphereCandidatePairs{};
    std::uint64_t clothSelfFrictionContacts{};
    std::uint64_t clothGroundFrictionContacts{};
    double maximumSweptSelfAdvance{};
    double finalPrimitiveSelfPenetration{};
    double maximumStrainLimitCorrection{};
    double finalStrainLimitViolation{};
    std::uint64_t ballClothFrictionContacts{};
    std::uint64_t ballPairFrictionContacts{};
    std::uint64_t ballGroundFrictionContacts{};
    std::uint64_t ballRollingResistanceContacts{};
    std::uint32_t escapedMask{};
    std::uint32_t spilledMask{};
    std::uint32_t mouthCandidateMask{};
    std::uint32_t releasedMask{};
    std::array<double, kFruitCount> maximumBallPenetrationByFruit{};
    std::array<double, kFruitCount> maximumMouthClearanceByFruit{};
    double ballClothSolveSeconds{};
    double primitiveSelfSolveSeconds{};
    double pointSelfSolveSeconds{};
    double strainLimitSolveSeconds{};
    double primitiveCertificateSeconds{};
};

template <typename Function>
auto accumulateSeconds(double& total, Function&& function) {
    const auto start = std::chrono::steady_clock::now();
    auto result = function();
    total += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start
    ).count();
    return result;
}

struct SimulationResult {
    ClothModel cloth;
    std::array<Ball, kFruitCount> balls{};
    Metrics metrics{};
    Scenario scenario{Scenario::grounded};
};

std::uint32_t nodeIndex(const std::uint32_t level, const std::uint32_t ring) {
    return level * kAround + ring % kAround;
}

std::uint32_t cyclicRingDistance(
    const std::uint32_t first,
    const std::uint32_t second
) {
    const std::uint32_t direct = first > second ? first - second : second - first;
    return std::min(direct, kAround - direct);
}

double smoothstep(const double value) {
    const double x = std::clamp(value, 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
}

Vec3 authoredPosition(const std::uint32_t level, const std::uint32_t ring) {
    const double vertical = static_cast<double>(level) /
        static_cast<double>(kLevels - 1);
    const double body = smoothstep(vertical / 0.55);
    const double fold = std::clamp((vertical - 0.72) / 0.28, 0.0, 1.0);
    const double skirt = 1.0 - smoothstep(vertical / 0.18);
    const double angle0 = 2.0 * std::numbers::pi *
        static_cast<double>(ring) / static_cast<double>(kAround);
    const double baseRadius =
        0.266 + 0.030 * body - 0.116 * fold + 0.045 * skirt;
    const double angle = angle0 + 0.08 * vertical;
    const double wrinkle =
        1.0 +
        0.035 * std::sin(
            5.0 * angle + 5.0 * std::numbers::pi * vertical
        ) +
        0.018 * std::sin(
            9.0 * angle - 3.0 * std::numbers::pi * vertical
        );
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
            rimSag + skirtSag + kInitialGroundLift,
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

bool bottomBoundary(const std::uint32_t row, const std::uint32_t column) {
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

Vec3 authoredBottomPosition(
    const std::uint32_t row,
    const std::uint32_t column
) {
    const auto [x, y] = concentricBottomCoordinate(row, column);
    const double radial = std::hypot(x, y);
    if (radial < 1.0e-12) {
        return {0.0, 0.0, 0.012 + kInitialGroundLift};
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
    const Vec3 first = authoredPosition(0u, firstRing);
    const Vec3 second = authoredPosition(0u, secondRing);
    const Vec3 outer = first * (1.0 - fraction) + second * fraction;
    return {
        outer.x * radial,
        outer.y * radial,
        0.012 + kInitialGroundLift +
            radial * (outer.z - 0.012 - kInitialGroundLift),
    };
}

ClothModel makeCloth(const Scenario scenario) {
    ClothModel model;
    std::array<std::uint32_t, kAround> bottomBoundaryUse{};
    for (std::uint32_t row = 0u; row < kBottomGrid; ++row) {
        for (std::uint32_t column = 0u; column < kBottomGrid; ++column) {
            if (bottomBoundary(row, column)) {
                ++bottomBoundaryUse[bottomGridIndex(row, column)];
            }
        }
    }
    if (!std::all_of(
        bottomBoundaryUse.begin(),
        bottomBoundaryUse.end(),
        [](const std::uint32_t uses) { return uses == 1u; }
    )) {
        throw std::logic_error(
            "bottom weave must map one-to-one onto the 48-node wall boundary"
        );
    }
    model.particles.reserve(
        kAround * kLevels + kBottomInterior * kBottomInterior
    );
    for (std::uint32_t level = 0; level < kLevels; ++level) {
        for (std::uint32_t ring = 0; ring < kAround; ++ring) {
            Vec3 position = authoredPosition(level, ring);
            if (scenario == Scenario::spin) {
                position.z += kAirborneLift;
            }
            const double mass = level + 2u >= kLevels
                ? kClothHemNodeMass
                : kClothNodeMass;
            model.particles.push_back({
                position,
                position,
                {},
                position,
                1.0 / mass,
                mass,
            });
        }
    }
    for (std::uint32_t row = 1u; row + 1u < kBottomGrid; ++row) {
        for (std::uint32_t column = 1u;
             column + 1u < kBottomGrid;
             ++column) {
            Vec3 position = authoredBottomPosition(row, column);
            if (scenario == Scenario::spin) {
                position.z += kAirborneLift;
            }
            model.particles.push_back({
                position,
                position,
                {},
                position,
                1.0 / kClothNodeMass,
                kClothNodeMass,
            });
        }
    }
    model.bottomCenter = bottomGridIndex(
        (kBottomGrid - 1u) / 2u,
        (kBottomGrid - 1u) / 2u
    );
    if (scenario != Scenario::grounded) {
        const std::uint32_t centerIndex = nodeIndex(kLevels - 1u, 0u);
        model.gripTarget = model.particles[centerIndex].rest;
        model.gripPrevious = model.gripTarget;
        model.gripActive = true;
        for (std::uint32_t level = kLevels - 2u;
             level < kLevels;
             ++level) {
            for (const int offset : {-2, -1, 0, 1, 2}) {
                const std::uint32_t ring = static_cast<std::uint32_t>(
                    (static_cast<int>(kAround) + offset) %
                    static_cast<int>(kAround)
                );
                const std::uint32_t index = nodeIndex(level, ring);
                model.grips.push_back({
                    .particle = index,
                    .targetOffset =
                        model.particles[index].rest - model.gripTarget,
                });
            }
        }
    }

    const auto addDistance = [&model](
        const std::uint32_t first,
        const std::uint32_t second,
        const double compliance,
        const DistanceKind kind,
        const bool isYarnSegment = true
    ) {
        model.distances.push_back({
            first,
            second,
            length(model.particles[second].rest - model.particles[first].rest),
            compliance,
            0.0,
            kind,
        });
        if (isYarnSegment) {
            model.yarnSegments.push_back({first, second});
        }
    };

    for (std::uint32_t level = 0; level < kLevels; ++level) {
        for (std::uint32_t ring = 0; ring < kAround; ++ring) {
            addDistance(
                nodeIndex(level, ring),
                nodeIndex(level, ring + 1u),
                level + 2u >= kLevels
                    ? kAxialCuffCompliance
                    : kAxialBodyCompliance,
                DistanceKind::weft
            );
        }
    }
    for (std::uint32_t level = 0; level + 1u < kLevels; ++level) {
        for (std::uint32_t ring = 0; ring < kAround; ++ring) {
            addDistance(
                nodeIndex(level, ring),
                nodeIndex(level + 1u, ring),
                kAxialBodyCompliance,
                DistanceKind::warp
            );
        }
    }
    for (std::uint32_t row = 0u; row < kBottomGrid; ++row) {
        for (std::uint32_t column = 0u;
             column + 1u < kBottomGrid;
             ++column) {
            if (!(bottomBoundary(row, column) &&
                  bottomBoundary(row, column + 1u))) {
                addDistance(
                    bottomGridIndex(row, column),
                    bottomGridIndex(row, column + 1u),
                    kAxialBodyCompliance,
                    DistanceKind::bottom
                );
            }
        }
    }
    for (std::uint32_t column = 0u; column < kBottomGrid; ++column) {
        for (std::uint32_t row = 0u; row + 1u < kBottomGrid; ++row) {
            if (!(bottomBoundary(row, column) &&
                  bottomBoundary(row + 1u, column))) {
                addDistance(
                    bottomGridIndex(row, column),
                    bottomGridIndex(row + 1u, column),
                    kAxialBodyCompliance,
                    DistanceKind::bottom
                );
            }
        }
    }

    const auto addKnot = [&model](
        const std::uint32_t warpFirst,
        const std::uint32_t warpSecond,
        const std::uint32_t weftFirst,
        const std::uint32_t weftSecond
    ) {
        const Vec3 warp = normalized(
            model.particles[warpSecond].rest -
            model.particles[warpFirst].rest
        );
        const Vec3 weft = normalized(
            model.particles[weftSecond].rest -
            model.particles[weftFirst].rest
        );
        model.knots.push_back({
            .warpFirst = warpFirst,
            .warpSecond = warpSecond,
            .weftFirst = weftFirst,
            .weftSecond = weftSecond,
            .restCosine = dot(warp, weft),
            .compliance = kKnotCompliance,
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

    for (std::uint32_t level = 0; level + 1u < kLevels; ++level) {
        for (std::uint32_t ring = 0; ring < kAround; ++ring) {
            const std::uint32_t next = (ring + 1u) % kAround;
            const std::uint32_t a = nodeIndex(level, ring);
            const std::uint32_t b = nodeIndex(level, next);
            const std::uint32_t c = nodeIndex(level + 1u, ring);
            const std::uint32_t d = nodeIndex(level + 1u, next);
            model.renderTriangles.push_back({a, b, c});
            model.renderTriangles.push_back({b, d, c});
        }
    }
    for (std::uint32_t row = 0u; row + 1u < kBottomGrid; ++row) {
        for (std::uint32_t column = 0u;
             column + 1u < kBottomGrid;
             ++column) {
            const std::uint32_t a = bottomGridIndex(row, column);
            const std::uint32_t b = bottomGridIndex(row, column + 1u);
            const std::uint32_t c = bottomGridIndex(row + 1u, column);
            const std::uint32_t d = bottomGridIndex(row + 1u, column + 1u);
            if ((row + column) % 2u == 0u) {
                model.renderTriangles.push_back({a, c, b});
                model.renderTriangles.push_back({b, c, d});
            } else {
                model.renderTriangles.push_back({a, d, b});
                model.renderTriangles.push_back({a, c, d});
            }
        }
    }

    std::vector<std::vector<std::uint32_t>> directTopology(
        model.particles.size()
    );
    for (const DistanceConstraint& constraint : model.distances) {
        directTopology[constraint.first].push_back(constraint.second);
        directTopology[constraint.second].push_back(constraint.first);
    }
    model.localTopology.resize(model.particles.size());
    for (std::uint32_t index = 0; index < model.particles.size(); ++index) {
        std::vector<std::uint32_t>& local = model.localTopology[index];
        local.push_back(index);
        for (const std::uint32_t firstHop : directTopology[index]) {
            local.push_back(firstHop);
            for (const std::uint32_t secondHop : directTopology[firstHop]) {
                local.push_back(secondHop);
            }
        }
        std::sort(local.begin(), local.end());
        local.erase(std::unique(local.begin(), local.end()), local.end());
    }

    const auto addYarnBend = [&model](
        const std::uint32_t first,
        const std::uint32_t middle,
        const std::uint32_t third,
        const double compliance
    ) {
        model.bends.push_back({
            .first = first,
            .middle = middle,
            .third = third,
            .restChord = length(
                model.particles[third].rest - model.particles[first].rest
            ),
            .restArc =
                length(
                    model.particles[middle].rest -
                    model.particles[first].rest
                ) +
                length(
                    model.particles[third].rest -
                    model.particles[middle].rest
                ),
            .compliance = compliance,
        });
    };
    for (std::uint32_t level = 0u; level < kLevels; ++level) {
        for (std::uint32_t ring = 0u; ring < kAround; ++ring) {
            addYarnBend(
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
            addYarnBend(
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
            addYarnBend(
                bottomGridIndex(row, column - 1u),
                bottomGridIndex(row, column),
                bottomGridIndex(row, column + 1u),
                kBendBodyCompliance
            );
            addYarnBend(
                bottomGridIndex(column - 1u, row),
                bottomGridIndex(column, row),
                bottomGridIndex(column + 1u, row),
                kBendBodyCompliance
            );
        }
    }
    return model;
}

std::array<Ball, kFruitCount> makeBalls(const Scenario scenario) {
    constexpr std::array<Vec3, kFruitCount> groundedPositions{{
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
    constexpr std::array<double, kFruitCount> radii{{
        0.070, 0.065, 0.075, 0.060, 0.068, 0.072,
        0.063, 0.066, 0.072, 0.064, 0.070, 0.067,
    }};
    constexpr std::array<double, kFruitCount> masses{{
        0.21, 0.17, 0.26, 0.14, 0.19, 0.23,
        0.16, 0.18, 0.23, 0.17, 0.21, 0.18,
    }};
    constexpr std::array<std::uint32_t, kFruitCount> appearances{{
        0u, 1u, 2u, 1u, 3u, 0u,
        2u, 3u, 0u, 1u, 2u, 3u,
    }};
    std::array<Ball, kFruitCount> balls{};
    for (std::size_t index = 0; index < balls.size(); ++index) {
        const Vec3 position = groundedPositions[index] + Vec3{
            0.0,
            0.0,
            kInitialGroundLift +
                (scenario == Scenario::spin ? kAirborneLift : 0.0),
        };
        balls[index].position = position;
        balls[index].previous = position;
        balls[index].radius = radii[index];
        balls[index].inverseMass = 1.0 / masses[index];
        balls[index].appearance = appearances[index];
    }
    return balls;
}

double inverseInertia(const Ball& ball) {
    return 2.5 * ball.inverseMass / (ball.radius * ball.radius);
}

void integrateOrientation(Ball& ball, const double timestep) {
    const Quaternion angular{
        0.0,
        ball.angularVelocity.x,
        ball.angularVelocity.y,
        ball.angularVelocity.z,
    };
    ball.orientation = normalized(
        ball.orientation + (angular * ball.orientation) * (0.5 * timestep)
    );
}

Vec3 quadraticDragForce(
    const Vec3 relativeVelocity,
    const double coefficient
) {
    const double speed = length(relativeVelocity);
    if (speed < 1.0e-14 || coefficient <= 0.0) {
        return {};
    }
    return relativeVelocity * (-coefficient * speed);
}

double quadraticDragAttenuation(
    const double speed,
    const double coefficient,
    const double inverseMass,
    const double timestep
) {
    return 1.0 / (
        1.0 + inverseMass * coefficient * speed * timestep
    );
}

void applyYarnAerodynamics(
    ClothModel& cloth,
    const Vec3 airVelocity,
    const double timestep,
    Metrics* metrics
) {
    std::vector<Vec3> forces(cloth.particles.size());
    double relativeEnergyBefore = 0.0;
    if (metrics != nullptr) {
        for (const Particle& particle : cloth.particles) {
            relativeEnergyBefore += 0.5 * particle.mass * lengthSquared(
                particle.velocity - airVelocity
            );
        }
    }
    for (const Edge segment : cloth.yarnSegments) {
        const Particle& first = cloth.particles[segment.first];
        const Particle& second = cloth.particles[segment.second];
        const Vec3 span = second.position - first.position;
        const double spanLength = length(span);
        if (spanLength < 1.0e-12) {
            continue;
        }
        const Vec3 axis = span / spanLength;
        const Vec3 relativeVelocity =
            (first.velocity + second.velocity) * 0.5 - airVelocity;
        const Vec3 axialVelocity = axis * dot(relativeVelocity, axis);
        const Vec3 crossflowVelocity = relativeVelocity - axialVelocity;
        const double crossflowArea = 2.0 * kClothRadius * spanLength;
        const double wettedArea =
            2.0 * std::numbers::pi * kClothRadius * spanLength;
        const Vec3 crossflowForce = quadraticDragForce(
            crossflowVelocity,
            0.5 * kAirDensity * kYarnCrossflowDragCoefficient *
                crossflowArea
        );
        const Vec3 skinForce = quadraticDragForce(
            axialVelocity,
            0.5 * kAirDensity * kYarnSkinFrictionCoefficient * wettedArea
        );
        const Vec3 force = crossflowForce + skinForce;
        forces[segment.first] += force * 0.5;
        forces[segment.second] += force * 0.5;
        if (metrics != nullptr) {
            metrics->maximumYarnAerodynamicForce = std::max(
                metrics->maximumYarnAerodynamicForce,
                length(force)
            );
        }
    }
    double dragPower = 0.0;
    double inverseMassWeightedForceSquared = 0.0;
    for (std::size_t index = 0u; index < cloth.particles.size(); ++index) {
        const Particle& particle = cloth.particles[index];
        if (particle.inverseMass <= 0.0) {
            continue;
        }
        dragPower += dot(
            forces[index],
            particle.velocity - airVelocity
        );
        inverseMassWeightedForceSquared +=
            particle.inverseMass * lengthSquared(forces[index]);
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
    for (std::size_t index = 0u; index < cloth.particles.size(); ++index) {
        Particle& particle = cloth.particles[index];
        if (particle.inverseMass > 0.0) {
            particle.velocity +=
                forces[index] * (
                    attenuation * particle.inverseMass * timestep
                );
        }
    }
    if (metrics != nullptr) {
        double relativeEnergyAfter = 0.0;
        for (const Particle& particle : cloth.particles) {
            relativeEnergyAfter += 0.5 * particle.mass * lengthSquared(
                particle.velocity - airVelocity
            );
        }
        metrics->aerodynamicDissipation += std::max(
            0.0,
            relativeEnergyBefore - relativeEnergyAfter
        );
    }
}

void applyFruitAerodynamics(
    std::array<Ball, kFruitCount>& balls,
    const Vec3 airVelocity,
    const double timestep,
    Metrics* metrics
) {
    for (Ball& ball : balls) {
        if (ball.inverseMass <= 0.0) {
            continue;
        }
        const double inverseRotationalInertia = inverseInertia(ball);
        const double relativeEnergyBefore =
            0.5 / ball.inverseMass * lengthSquared(
                ball.velocity - airVelocity
            ) +
            0.5 / inverseRotationalInertia *
                lengthSquared(ball.angularVelocity);
        const Vec3 relativeVelocity = ball.velocity - airVelocity;
        const double frontalArea =
            std::numbers::pi * ball.radius * ball.radius;
        const double translationalCoefficient =
            0.5 * kAirDensity * kFruitDragCoefficient * frontalArea;
        const Vec3 force = quadraticDragForce(
            relativeVelocity,
            translationalCoefficient
        );
        const double translationalAttenuation = quadraticDragAttenuation(
            length(relativeVelocity),
            translationalCoefficient,
            ball.inverseMass,
            timestep
        );
        ball.velocity = airVelocity +
            relativeVelocity * translationalAttenuation;

        const double rotationalCoefficient =
            (3.0 * std::numbers::pi * std::numbers::pi / 8.0) *
            kAirDensity * kFruitRotationalDragCoefficient *
            std::pow(ball.radius, 5.0);
        const Vec3 torque = quadraticDragForce(
            ball.angularVelocity,
            rotationalCoefficient
        );
        ball.angularVelocity = ball.angularVelocity *
            quadraticDragAttenuation(
                length(ball.angularVelocity),
                rotationalCoefficient,
                inverseRotationalInertia,
                timestep
            );

        if (metrics != nullptr) {
            const double relativeEnergyAfter =
                0.5 / ball.inverseMass * lengthSquared(
                    ball.velocity - airVelocity
                ) +
                0.5 / inverseRotationalInertia *
                    lengthSquared(ball.angularVelocity);
            metrics->maximumFruitAerodynamicForce = std::max(
                metrics->maximumFruitAerodynamicForce,
                length(force)
            );
            metrics->maximumFruitAerodynamicTorque = std::max(
                metrics->maximumFruitAerodynamicTorque,
                length(torque)
            );
            metrics->aerodynamicDissipation += std::max(
                0.0,
                relativeEnergyBefore - relativeEnergyAfter
            );
        }
    }
}

struct AerodynamicsProbeResult {
    Vec3 coarseForce{};
    Vec3 refinedForce{};
    double yarnEnergyBefore{};
    double yarnEnergyAfter{};
    double fruitForce{};
    double fruitTorque{};
    double fruitSpeed{};
    double fruitAngularSpeed{};
    double coMovingDelta{};
    double yarnTemporalRefinementError{};
    double fruitTemporalRefinementError{};
    double dissipation{};
};

AerodynamicsProbeResult runAerodynamicsProbeOnce() {
    constexpr double timestep = 0.01;
    constexpr Vec3 yarnVelocity{3.0, 4.0, 0.0};
    const auto makeParticle = [yarnVelocity](
        const Vec3 position,
        const double mass
    ) {
        Particle particle;
        particle.position = position;
        particle.previous = position;
        particle.rest = position;
        particle.velocity = yarnVelocity;
        particle.inverseMass = 1.0 / mass;
        particle.mass = mass;
        return particle;
    };
    const auto momentum = [](const ClothModel& cloth) {
        Vec3 value{};
        for (const Particle& particle : cloth.particles) {
            value += particle.velocity * particle.mass;
        }
        return value;
    };
    const auto kineticEnergy = [](const ClothModel& cloth) {
        double value = 0.0;
        for (const Particle& particle : cloth.particles) {
            value += 0.5 * particle.mass * lengthSquared(particle.velocity);
        }
        return value;
    };

    ClothModel coarse;
    coarse.particles = {
        makeParticle({-0.5, 0.0, 0.0}, 0.5),
        makeParticle({0.5, 0.0, 0.0}, 0.5),
    };
    coarse.yarnSegments = {{0u, 1u}};
    const Vec3 coarseMomentumBefore = momentum(coarse);
    const double energyBefore = kineticEnergy(coarse);
    Metrics metrics;
    applyYarnAerodynamics(coarse, {}, timestep, &metrics);
    const Vec3 coarseForce =
        (momentum(coarse) - coarseMomentumBefore) / timestep;

    ClothModel refined;
    refined.particles = {
        makeParticle({-0.5, 0.0, 0.0}, 0.25),
        makeParticle({0.0, 0.0, 0.0}, 0.50),
        makeParticle({0.5, 0.0, 0.0}, 0.25),
    };
    refined.yarnSegments = {{0u, 1u}, {1u, 2u}};
    const Vec3 refinedMomentumBefore = momentum(refined);
    applyYarnAerodynamics(refined, {}, timestep, nullptr);
    const Vec3 refinedForce =
        (momentum(refined) - refinedMomentumBefore) / timestep;

    std::array<Ball, kFruitCount> balls{};
    Ball& fruit = balls[0];
    fruit.radius = 0.10;
    fruit.inverseMass = 1.0;
    fruit.velocity = {5.0, 0.0, 0.0};
    fruit.angularVelocity = {0.0, 0.0, 10.0};
    Metrics fruitMetrics;
    applyFruitAerodynamics(balls, {}, timestep, &fruitMetrics);

    ClothModel coMoving;
    coMoving.particles = {
        makeParticle({-0.5, 0.0, 0.0}, 0.5),
        makeParticle({0.5, 0.0, 0.0}, 0.5),
    };
    coMoving.yarnSegments = {{0u, 1u}};
    std::array<Ball, kFruitCount> coMovingBalls{};
    coMovingBalls[0].radius = 0.10;
    coMovingBalls[0].inverseMass = 1.0;
    coMovingBalls[0].velocity = yarnVelocity;
    const Vec3 coMovingMomentumBefore = momentum(coMoving);
    const Vec3 coMovingBallVelocityBefore = coMovingBalls[0].velocity;
    applyYarnAerodynamics(
        coMoving,
        yarnVelocity,
        timestep,
        nullptr
    );
    applyFruitAerodynamics(
        coMovingBalls,
        yarnVelocity,
        timestep,
        nullptr
    );

    ClothModel temporalYarnCoarse;
    temporalYarnCoarse.particles = {
        makeParticle({-0.5, 0.0, 0.0}, 0.5),
        makeParticle({0.5, 0.0, 0.0}, 0.5),
    };
    for (Particle& particle : temporalYarnCoarse.particles) {
        particle.velocity = {0.0, 4.0, 0.0};
    }
    temporalYarnCoarse.yarnSegments = {{0u, 1u}};
    ClothModel temporalYarnRefined = temporalYarnCoarse;
    applyYarnAerodynamics(temporalYarnCoarse, {}, 0.20, nullptr);
    for (std::uint32_t step = 0u; step < 20u; ++step) {
        applyYarnAerodynamics(temporalYarnRefined, {}, 0.01, nullptr);
    }
    double yarnTemporalRefinementError = 0.0;
    for (std::size_t index = 0u;
         index < temporalYarnCoarse.particles.size();
         ++index) {
        yarnTemporalRefinementError = std::max(
            yarnTemporalRefinementError,
            length(
                temporalYarnCoarse.particles[index].velocity -
                temporalYarnRefined.particles[index].velocity
            )
        );
    }

    std::array<Ball, kFruitCount> temporalFruitCoarse{};
    temporalFruitCoarse[0] = fruit;
    temporalFruitCoarse[0].velocity = {5.0, 0.0, 0.0};
    temporalFruitCoarse[0].angularVelocity = {0.0, 0.0, 10.0};
    auto temporalFruitRefined = temporalFruitCoarse;
    applyFruitAerodynamics(temporalFruitCoarse, {}, 0.20, nullptr);
    for (std::uint32_t step = 0u; step < 20u; ++step) {
        applyFruitAerodynamics(temporalFruitRefined, {}, 0.01, nullptr);
    }
    const double fruitTemporalRefinementError =
        length(
            temporalFruitCoarse[0].velocity -
            temporalFruitRefined[0].velocity
        ) + length(
            temporalFruitCoarse[0].angularVelocity -
            temporalFruitRefined[0].angularVelocity
        );

    return {
        .coarseForce = coarseForce,
        .refinedForce = refinedForce,
        .yarnEnergyBefore = energyBefore,
        .yarnEnergyAfter = kineticEnergy(coarse),
        .fruitForce = fruitMetrics.maximumFruitAerodynamicForce,
        .fruitTorque = fruitMetrics.maximumFruitAerodynamicTorque,
        .fruitSpeed = length(fruit.velocity),
        .fruitAngularSpeed = length(fruit.angularVelocity),
        .coMovingDelta = length(momentum(coMoving) - coMovingMomentumBefore) +
            length(coMovingBalls[0].velocity - coMovingBallVelocityBefore),
        .yarnTemporalRefinementError = yarnTemporalRefinementError,
        .fruitTemporalRefinementError = fruitTemporalRefinementError,
        .dissipation = metrics.aerodynamicDissipation +
            fruitMetrics.aerodynamicDissipation,
    };
}

bool runAerodynamicsProbe() {
    const AerodynamicsProbeResult first = runAerodynamicsProbeOnce();
    const AerodynamicsProbeResult replay = runAerodynamicsProbeOnce();
    constexpr double lengthMeters = 1.0;
    constexpr double crossflowSpeed = 4.0;
    constexpr double axialSpeed = 3.0;
    const double expectedCrossflowForce =
        -0.5 * kAirDensity * kYarnCrossflowDragCoefficient *
        (2.0 * kClothRadius * lengthMeters) *
        crossflowSpeed * crossflowSpeed;
    const double expectedAxialForce =
        -0.5 * kAirDensity * kYarnSkinFrictionCoefficient *
        (2.0 * std::numbers::pi * kClothRadius * lengthMeters) *
        axialSpeed * axialSpeed;
    const double expectedYarnDragPower =
        expectedAxialForce * axialSpeed +
        expectedCrossflowForce * crossflowSpeed;
    const double expectedYarnForceSquared =
        expectedAxialForce * expectedAxialForce +
        expectedCrossflowForce * expectedCrossflowForce;
    const double expectedYarnAttenuation = 1.0 / (
        1.0 + 0.01 * expectedYarnForceSquared /
            -expectedYarnDragPower
    );
    const double expectedAppliedCrossflowForce =
        expectedCrossflowForce * expectedYarnAttenuation;
    const double expectedAppliedAxialForce =
        expectedAxialForce * expectedYarnAttenuation;
    constexpr double fruitRadius = 0.10;
    constexpr double fruitSpeed = 5.0;
    constexpr double fruitAngularSpeed = 10.0;
    const double expectedFruitForce =
        0.5 * kAirDensity * kFruitDragCoefficient *
        std::numbers::pi * fruitRadius * fruitRadius *
        fruitSpeed * fruitSpeed;
    const double expectedFruitTorque =
        (3.0 * std::numbers::pi * std::numbers::pi / 8.0) *
        kAirDensity * kFruitRotationalDragCoefficient *
        std::pow(fruitRadius, 5.0) *
        fruitAngularSpeed * fruitAngularSpeed;
    const double expectedFruitSpeed = fruitSpeed /
        (1.0 + expectedFruitForce / fruitSpeed * 0.01);
    const double expectedFruitAngularSpeed = fruitAngularSpeed /
        (1.0 + expectedFruitTorque / fruitAngularSpeed *
            (2.5 / (fruitRadius * fruitRadius)) * 0.01);
    const bool deterministic =
        first.coarseForce.x == replay.coarseForce.x &&
        first.coarseForce.y == replay.coarseForce.y &&
        first.refinedForce.x == replay.refinedForce.x &&
        first.refinedForce.y == replay.refinedForce.y &&
        first.yarnEnergyAfter == replay.yarnEnergyAfter &&
        first.fruitForce == replay.fruitForce &&
        first.fruitTorque == replay.fruitTorque &&
        first.fruitSpeed == replay.fruitSpeed &&
        first.fruitAngularSpeed == replay.fruitAngularSpeed &&
        first.coMovingDelta == replay.coMovingDelta &&
        first.yarnTemporalRefinementError ==
            replay.yarnTemporalRefinementError &&
        first.fruitTemporalRefinementError ==
            replay.fruitTemporalRefinementError;
    const bool pass = deterministic &&
        std::abs(first.coarseForce.x - expectedAppliedAxialForce) <
            1.0e-12 &&
        std::abs(first.coarseForce.y - expectedAppliedCrossflowForce) <
            1.0e-12 &&
        length(first.coarseForce - first.refinedForce) < 1.0e-12 &&
        first.yarnEnergyAfter < first.yarnEnergyBefore &&
        std::abs(first.fruitForce - expectedFruitForce) < 1.0e-12 &&
        std::abs(first.fruitTorque - expectedFruitTorque) < 1.0e-12 &&
        std::abs(first.fruitSpeed - expectedFruitSpeed) < 1.0e-12 &&
        std::abs(
            first.fruitAngularSpeed - expectedFruitAngularSpeed
        ) < 1.0e-12 &&
        first.coMovingDelta < 1.0e-12 &&
        first.yarnTemporalRefinementError < 1.0e-12 &&
        first.fruitTemporalRefinementError < 1.0e-12 &&
        first.dissipation > 0.0;
    std::cout << std::fixed << std::setprecision(12)
              << "probe=explicit_yarn_and_fruit_aerodynamics"
              << " air_density=" << kAirDensity
              << " yarn_crossflow_force=" << -first.coarseForce.y
              << " yarn_axial_force=" << -first.coarseForce.x
              << " refinement_force_error="
              << length(first.coarseForce - first.refinedForce)
              << " yarn_energy_before=" << first.yarnEnergyBefore
              << " yarn_energy_after=" << first.yarnEnergyAfter
              << " fruit_force=" << first.fruitForce
              << " fruit_torque=" << first.fruitTorque
              << " fruit_speed=" << first.fruitSpeed
              << " fruit_angular_speed=" << first.fruitAngularSpeed
              << " co_moving_delta=" << first.coMovingDelta
              << " yarn_temporal_refinement_error="
              << first.yarnTemporalRefinementError
              << " fruit_temporal_refinement_error="
              << first.fruitTemporalRefinementError
              << " dissipation=" << first.dissipation
              << " deterministic=" << std::boolalpha << deterministic
              << '\n'
              << "result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass;
}

void applyBallImpulse(
    Ball& ball,
    const Vec3 impulse,
    const Vec3 contactOffset
) {
    ball.velocity += impulse * ball.inverseMass;
    ball.angularVelocity += cross(contactOffset, impulse) * inverseInertia(ball);
}

void recordFrictionImpulse(
    Metrics& metrics,
    const double tangentialImpulse,
    const double frictionLimit
) {
    metrics.accumulatedTangentialImpulse += tangentialImpulse;
    if (frictionLimit > 0.0) {
        metrics.maximumFrictionConeRatio = std::max(
            metrics.maximumFrictionConeRatio,
            tangentialImpulse / frictionLimit
        );
    }
}

Vec3 spinGripTarget(const double time) {
    Vec3 base = authoredPosition(kLevels - 1u, 0u);
    base.z += kAirborneLift;
    constexpr double radius = 0.28;
    constexpr double angularSpeed = 4.8;
    constexpr double rampTime = 0.18;
    const double angle = angularSpeed * (
        time - rampTime * (1.0 - std::exp(-time / rampTime))
    );
    return base + Vec3{
        radius * (std::cos(angle) - 1.0),
        radius * std::sin(angle),
        0.035 * std::sin(0.5 * angle),
    };
}

Vec3 pickupGripTarget(const double time) {
    const Vec3 base = authoredPosition(kLevels - 1u, 0u);
    const double lift = smoothstep(time / 0.80);
    const double firstSnap = smoothstep((time - 1.00) / 0.23);
    const double firstRecovery = smoothstep((time - 1.45) / 0.50);
    return base + Vec3{
        -0.10 * firstSnap,
        0.04 * std::sin(std::numbers::pi * firstSnap),
        1.25 * lift - 0.85 * firstSnap + 0.75 * firstRecovery,
    };
}

void updateGrip(
    ClothModel& cloth,
    Metrics& metrics,
    const Scenario scenario,
    const double time,
    const numi::GripTrajectory* trajectory
) {
    cloth.gripPrevious = cloth.gripTarget;
    if (trajectory != nullptr) {
        const numi::GripTrajectoryPose pose =
            numi::sampleGripTrajectory(*trajectory, time);
        const Vec3 base = authoredPosition(kLevels - 1u, 0u);
        cloth.gripTarget = base + Vec3{
            pose.translationMeters.x,
            pose.translationMeters.y,
            pose.translationMeters.z,
        };
        cloth.gripOrientation = gripQuaternion(pose.orientation);
        if (pose.active && pose.attachmentGeneration !=
                cloth.gripAttachmentGeneration) {
            if (cloth.gripAttachmentGeneration ==
                    std::numeric_limits<std::uint32_t>::max() ||
                pose.attachmentGeneration !=
                    cloth.gripAttachmentGeneration + 1u) {
                throw std::runtime_error(
                    "grip reattachment generation is not sequential"
                );
            }
            if (trajectory->selectNearestCuffPatch) {
                std::uint32_t centerRing = 0u;
                double nearestDistanceSquared =
                    std::numeric_limits<double>::infinity();
                for (std::uint32_t ring = 0u; ring < kAround; ++ring) {
                    const Vec3 separation = cloth.particles[
                        nodeIndex(kLevels - 1u, ring)
                    ].position - cloth.gripTarget;
                    const double distanceSquared = lengthSquared(separation);
                    if (distanceSquared < nearestDistanceSquared) {
                        nearestDistanceSquared = distanceSquared;
                        centerRing = ring;
                    }
                }
                constexpr std::uint32_t patchWidth = 5u;
                if (cloth.grips.size() != 2u * patchWidth) {
                    throw std::logic_error(
                        "dynamic cuff selection requires ten grip constraints"
                    );
                }
                for (std::uint32_t row = 0u; row < 2u; ++row) {
                    for (std::uint32_t slot = 0u;
                         slot < patchWidth;
                         ++slot) {
                        const std::uint32_t ring = (
                            centerRing + kAround + slot - 2u
                        ) % kAround;
                        cloth.grips[row * patchWidth + slot].particle =
                            nodeIndex(kLevels - 2u + row, ring);
                    }
                }
                metrics.maximumGripPatchRingShift = std::max(
                    metrics.maximumGripPatchRingShift,
                    cyclicRingDistance(
                        cloth.gripPatchCenterRing,
                        centerRing
                    )
                );
                cloth.gripPatchCenterRing = centerRing;
                ++metrics.gripPatchSelectionCount;
            }
            double maximumCaptureDistance = 0.0;
            for (const GripConstraint& constraint : cloth.grips) {
                maximumCaptureDistance = std::max(
                    maximumCaptureDistance,
                    length(
                        cloth.particles[constraint.particle].position -
                        cloth.gripTarget
                    )
                );
            }
            if (maximumCaptureDistance > kGripCaptureRadius) {
                throw std::runtime_error(
                    "grip reattachment exceeds capture radius"
                );
            }
            double maximumCaptureError = 0.0;
            for (GripConstraint& constraint : cloth.grips) {
                const Vec3 position =
                    cloth.particles[constraint.particle].position;
                constraint.targetOffset = inverseRotateVector(
                    cloth.gripOrientation,
                    position - cloth.gripTarget
                );
                constraint.lambda = {};
                const Vec3 reconstructed = cloth.gripTarget + rotateVector(
                    cloth.gripOrientation,
                    constraint.targetOffset
                );
                maximumCaptureError = std::max(
                    maximumCaptureError,
                    length(position - reconstructed)
                );
            }
            cloth.gripAttachmentGeneration = pose.attachmentGeneration;
            metrics.maximumRegrabCaptureDistance = std::max(
                metrics.maximumRegrabCaptureDistance,
                maximumCaptureDistance
            );
            metrics.maximumRegrabCaptureError = std::max(
                metrics.maximumRegrabCaptureError,
                maximumCaptureError
            );
            ++metrics.regrabCount;
        }
        cloth.gripActive = pose.active;
        if (!pose.active) {
            ++metrics.inactiveGripSubsteps;
        }
    } else {
        cloth.gripTarget = scenario == Scenario::spin
            ? spinGripTarget(time)
            : pickupGripTarget(time);
        cloth.gripOrientation = {};
        cloth.gripActive = true;
    }
}

void solveGrip(
    std::vector<Particle>& particles,
    GripConstraint& constraint,
    const Vec3 target,
    const double timestep
) {
    Particle& particle = particles[constraint.particle];
    const double alpha = constraint.compliance / (timestep * timestep);
    const double denominator = particle.inverseMass + alpha;
    if (denominator <= 0.0) {
        return;
    }
    const Vec3 value = particle.position - target;
    const Vec3 deltaLambda =
        (value * -1.0 - constraint.lambda * alpha) / denominator;
    constraint.lambda += deltaLambda;
    particle.position += deltaLambda * particle.inverseMass;
}

void solveDistance(
    std::vector<Particle>& particles,
    DistanceConstraint& constraint,
    const double timestep
) {
    Particle& first = particles[constraint.first];
    Particle& second = particles[constraint.second];
    const Vec3 difference = second.position - first.position;
    const double currentLength = length(difference);
    if (currentLength < 1.0e-12) {
        return;
    }
    const double alpha = constraint.compliance / (timestep * timestep);
    const double denominator = first.inverseMass + second.inverseMass + alpha;
    if (denominator <= 0.0) {
        return;
    }
    const double value = currentLength - constraint.restLength;
    const double deltaLambda =
        (-value - alpha * constraint.lambda) / denominator;
    constraint.lambda += deltaLambda;
    const Vec3 correction = difference * (deltaLambda / currentLength);
    first.position -= correction * first.inverseMass;
    second.position += correction * second.inverseMass;
}

void solveKnot(
    std::vector<Particle>& particles,
    KnotConstraint& constraint,
    const double timestep
) {
    Particle& warpFirst = particles[constraint.warpFirst];
    Particle& warpSecond = particles[constraint.warpSecond];
    Particle& weftFirst = particles[constraint.weftFirst];
    Particle& weftSecond = particles[constraint.weftSecond];
    const Vec3 warpVector = warpSecond.position - warpFirst.position;
    const Vec3 weftVector = weftSecond.position - weftFirst.position;
    const double warpLength = length(warpVector);
    const double weftLength = length(weftVector);
    if (warpLength < 1.0e-12 || weftLength < 1.0e-12) {
        return;
    }
    const Vec3 warp = warpVector / warpLength;
    const Vec3 weft = weftVector / weftLength;
    const double cosine = std::clamp(dot(warp, weft), -1.0, 1.0);
    const double value = cosine - constraint.restCosine;
    const Vec3 warpGradient = (weft - warp * cosine) / warpLength;
    const Vec3 weftGradient = (warp - weft * cosine) / weftLength;
    const std::array<Vec3, 4> gradients{{
        warpGradient * -1.0,
        warpGradient,
        weftGradient * -1.0,
        weftGradient,
    }};
    const std::array<std::uint32_t, 4> indices{{
        constraint.warpFirst,
        constraint.warpSecond,
        constraint.weftFirst,
        constraint.weftSecond,
    }};
    const double alpha = constraint.compliance / (timestep * timestep);
    double denominator = alpha;
    for (std::size_t index = 0u; index < indices.size(); ++index) {
        denominator += particles[indices[index]].inverseMass *
            lengthSquared(gradients[index]);
    }
    if (denominator <= 0.0) {
        return;
    }
    const double deltaLambda =
        (-value - alpha * constraint.lambda) / denominator;
    constraint.lambda += deltaLambda;
    for (std::size_t index = 0u; index < indices.size(); ++index) {
        particles[indices[index]].position += gradients[index] *
            (particles[indices[index]].inverseMass * deltaLambda);
    }
}

double strainExtensionLimit(const DistanceKind kind) {
    static_cast<void>(kind);
    return 0.285;
}

double solveStrainLimits(
    std::vector<Particle>& particles,
    const std::vector<DistanceConstraint>& constraints
) {
    double maximumCorrection = 0.0;
    for (const DistanceConstraint& constraint : constraints) {
        Particle& first = particles[constraint.first];
        Particle& second = particles[constraint.second];
        const Vec3 difference = second.position - first.position;
        const double currentLength = length(difference);
        const double maximumLength = constraint.restLength *
            (1.0 + strainExtensionLimit(constraint.kind));
        if (currentLength <= maximumLength || currentLength < 1.0e-12) {
            continue;
        }
        const double denominator = first.inverseMass + second.inverseMass;
        if (denominator <= 0.0) {
            continue;
        }
        const double correctionMagnitude = currentLength - maximumLength;
        const Vec3 direction = difference / currentLength;
        first.position += direction *
            (first.inverseMass * correctionMagnitude / denominator);
        second.position -= direction *
            (second.inverseMass * correctionMagnitude / denominator);
        maximumCorrection = std::max(
            maximumCorrection,
            correctionMagnitude
        );
    }
    return maximumCorrection;
}

double measureStrainLimitViolation(const ClothModel& cloth) {
    double maximumViolation = 0.0;
    for (const DistanceConstraint& constraint : cloth.distances) {
        const double currentLength = length(
            cloth.particles[constraint.second].position -
            cloth.particles[constraint.first].position
        );
        const double maximumLength = constraint.restLength *
            (1.0 + strainExtensionLimit(constraint.kind));
        maximumViolation = std::max(
            maximumViolation,
            currentLength - maximumLength
        );
    }
    return maximumViolation;
}

void solveBend(
    std::vector<Particle>& particles,
    YarnBendConstraint& constraint,
    const double timestep,
    const bool groundEnabled
) {
    const std::array<std::uint32_t, 2> indices{{
        constraint.first,
        constraint.third,
    }};
    const Vec3 difference = particles[constraint.third].position -
        particles[constraint.first].position;
    const double currentChord = length(difference);
    if (currentChord < 1.0e-12) {
        return;
    }
    const double value = currentChord - constraint.restChord;
    const Vec3 direction = difference / currentChord;
    const std::array<Vec3, 2> gradients{{direction * -1.0, direction}};
    const double alpha = constraint.compliance / (timestep * timestep);
    double freeDenominator = alpha;
    for (std::size_t index = 0; index < indices.size(); ++index) {
        freeDenominator += particles[indices[index]].inverseMass *
            lengthSquared(gradients[index]);
    }
    if (freeDenominator < 1.0e-16) {
        return;
    }
    const double numerator = -value - alpha * constraint.lambda;
    const double freeDeltaLambda = numerator / freeDenominator;
    std::array<bool, 2> groundActive{};
    double denominator = alpha;
    for (std::size_t index = 0; index < indices.size(); ++index) {
        const Particle& particle = particles[indices[index]];
        groundActive[index] = groundEnabled &&
            particle.position.z <= kClothRadius + 1.0e-9 &&
            gradients[index].z * freeDeltaLambda < 0.0;
        denominator += particle.inverseMass * (
            lengthSquared(gradients[index]) -
            (groundActive[index]
                ? gradients[index].z * gradients[index].z
                : 0.0)
        );
    }
    if (denominator < 1.0e-16) {
        return;
    }
    const double deltaLambda = numerator / denominator;
    double fraction = 1.0;
    if (groundEnabled) {
        for (std::size_t index = 0; index < indices.size(); ++index) {
            if (groundActive[index]) {
                continue;
            }
            const Particle& particle = particles[indices[index]];
            const double verticalCorrection = gradients[index].z *
                particle.inverseMass * deltaLambda;
            if (verticalCorrection >= 0.0) {
                continue;
            }
            fraction = std::min(
                fraction,
                std::max(0.0, particle.position.z - kClothRadius) /
                    -verticalCorrection
            );
        }
    }
    const double appliedLambda = deltaLambda * fraction;
    constraint.lambda += appliedLambda;
    for (std::size_t index = 0; index < indices.size(); ++index) {
        Vec3 correction = gradients[index] *
            (particles[indices[index]].inverseMass * appliedLambda);
        if (groundActive[index]) {
            correction.z = 0.0;
        }
        particles[indices[index]].position += correction;
        if (groundEnabled && particles[indices[index]].position.z <
            kClothRadius) {
            particles[indices[index]].position.z = kClothRadius;
        }
    }
}

Vec3 deformableParticleResponse(
    const Particle& particle,
    const Vec3 load,
    const bool groundEnabled
) {
    Vec3 response = load * particle.inverseMass;
    if (groundEnabled &&
        particle.position.z <= kClothRadius + 1.0e-6 &&
        response.z < 0.0) {
        response.z = 0.0;
    }
    return response;
}


double measureGroundPenetration(
    const std::vector<Particle>& particles,
    const std::array<Ball, kFruitCount>& balls
) {
    double maximum = 0.0;
    for (const Particle& particle : particles) {
        maximum = std::max(
            maximum,
            kClothRadius - particle.position.z
        );
    }
    for (const Ball& ball : balls) {
        maximum = std::max(maximum, ball.radius - ball.position.z);
    }
    return maximum;
}

double solveBallPair(
    Ball& first,
    Ball& second,
    const double timestep,
    BallPairContactImpulse& contactImpulse
) {
    const Vec3 difference = second.position - first.position;
    const double currentLength = length(difference);
    const double target = first.radius + second.radius;
    if (currentLength >= target || currentLength < 1.0e-12) {
        return 0.0;
    }
    const double denominator = first.inverseMass + second.inverseMass;
    const double lambda = (target - currentLength) / denominator;
    const Vec3 normal = difference / currentLength;
    const Vec3 correction = normal * lambda;
    first.position -= correction * first.inverseMass;
    second.position += correction * second.inverseMass;
    const double impulseMagnitude = lambda / timestep;
    contactImpulse.weightedNormal += normal * impulseMagnitude;
    contactImpulse.normalImpulse += impulseMagnitude;
    return target - currentLength;
}

void applyBallPairFriction(
    std::array<Ball, kFruitCount>& balls,
    const std::array<BallPairContactImpulse, kBallPairCount>& contacts,
    Metrics& metrics
) {
    std::size_t pairIndex = 0u;
    for (std::size_t firstIndex = 0; firstIndex < balls.size(); ++firstIndex) {
        for (std::size_t secondIndex = firstIndex + 1u;
             secondIndex < balls.size();
             ++secondIndex, ++pairIndex) {
            const BallPairContactImpulse& contact = contacts[pairIndex];
            if (contact.normalImpulse <= 0.0 ||
                lengthSquared(contact.weightedNormal) < 1.0e-20) {
                continue;
            }
            Ball& first = balls[firstIndex];
            Ball& second = balls[secondIndex];
            const Vec3 normal = normalized(contact.weightedNormal);
            const Vec3 firstOffset = normal * first.radius;
            const Vec3 secondOffset = normal * -second.radius;
            const Vec3 firstContactVelocity = first.velocity +
                cross(first.angularVelocity, firstOffset);
            const Vec3 secondContactVelocity = second.velocity +
                cross(second.angularVelocity, secondOffset);
            const Vec3 relativeVelocity =
                secondContactVelocity - firstContactVelocity;
            const Vec3 tangentVelocity = relativeVelocity -
                normal * dot(relativeVelocity, normal);
            const double slipSpeed = length(tangentVelocity);
            if (slipSpeed < 1.0e-10) {
                continue;
            }
            const Vec3 tangent = tangentVelocity / slipSpeed;
            const double denominator =
                first.inverseMass + second.inverseMass +
                inverseInertia(first) *
                    lengthSquared(cross(firstOffset, tangent)) +
                inverseInertia(second) *
                    lengthSquared(cross(secondOffset, tangent));
            if (denominator <= 0.0) {
                continue;
            }
            const double frictionLimit =
                kFruitPairFriction * contact.normalImpulse;
            const double tangentialImpulse = std::min(
                slipSpeed / denominator,
                frictionLimit
            );
            if (tangentialImpulse <= 0.0) {
                continue;
            }
            const Vec3 impulseOnSecond = tangent * -tangentialImpulse;
            applyBallImpulse(second, impulseOnSecond, secondOffset);
            applyBallImpulse(first, impulseOnSecond * -1.0, firstOffset);
            recordFrictionImpulse(
                metrics,
                tangentialImpulse,
                frictionLimit
            );
            ++metrics.ballPairFrictionContacts;
        }
    }
}

void applyBallRollingResistance(
    Ball& ball,
    const double normalImpulse,
    const double rollingResistanceCoefficient,
    Metrics& metrics
) {
    const Vec3 rollingAngularVelocity{
        ball.angularVelocity.x,
        ball.angularVelocity.y,
        0.0,
    };
    const double rollingSpeed = length(rollingAngularVelocity);
    if (normalImpulse <= 0.0 || rollingSpeed <= 1.0e-12 ||
        rollingResistanceCoefficient <= 0.0) {
        return;
    }
    const double requiredAngularImpulse =
        rollingSpeed / inverseInertia(ball);
    const double rollingImpulseLimit =
        rollingResistanceCoefficient * ball.radius * normalImpulse;
    const double angularImpulse = std::min(
        requiredAngularImpulse,
        rollingImpulseLimit
    );
    ball.angularVelocity -= rollingAngularVelocity * (
        angularImpulse * inverseInertia(ball) / rollingSpeed
    );
    if (rollingImpulseLimit > 0.0) {
        metrics.maximumRollingResistanceRatio = std::max(
            metrics.maximumRollingResistanceRatio,
            angularImpulse / rollingImpulseLimit
        );
    }
    ++metrics.ballRollingResistanceContacts;
}

void applyBallGroundFriction(
    std::array<Ball, kFruitCount>& balls,
    const std::array<double, kFruitCount>& normalImpulses,
    const double timestep,
    Metrics& metrics,
    const double rollingResistanceCoefficient
) {
    static_cast<void>(timestep);
    const Vec3 normal{0.0, 0.0, 1.0};
    for (std::size_t index = 0; index < balls.size(); ++index) {
        Ball& ball = balls[index];
        const double normalImpulse = normalImpulses[index];
        if (normalImpulse <= 0.0) {
            continue;
        }
        const Vec3 contactOffset{0.0, 0.0, -ball.radius};
        const Vec3 contactVelocity = ball.velocity +
            cross(ball.angularVelocity, contactOffset);
        const Vec3 tangentVelocity = contactVelocity -
            normal * dot(contactVelocity, normal);
        const double slipSpeed = length(tangentVelocity);
        if (slipSpeed > 1.0e-10) {
            const Vec3 tangent = tangentVelocity / slipSpeed;
            const double denominator = ball.inverseMass +
                inverseInertia(ball) *
                    lengthSquared(cross(contactOffset, tangent));
            const double frictionLimit =
                kFruitGroundFriction * normalImpulse;
            const double tangentialImpulse = std::min(
                slipSpeed / denominator,
                frictionLimit
            );
            if (tangentialImpulse > 0.0) {
                applyBallImpulse(
                    ball,
                    tangent * -tangentialImpulse,
                    contactOffset
                );
                recordFrictionImpulse(
                    metrics,
                    tangentialImpulse,
                    frictionLimit
                );
                ++metrics.ballGroundFrictionContacts;
            }
        }
        applyBallRollingResistance(
            ball,
            normalImpulse,
            rollingResistanceCoefficient,
            metrics
        );
    }
}

void applyClothGroundFriction(
    std::vector<Particle>& particles,
    const std::vector<double>& normalImpulses,
    Metrics& metrics
) {
    for (std::size_t index = 0u; index < particles.size(); ++index) {
        Particle& particle = particles[index];
        const double normalImpulse = normalImpulses[index];
        if (normalImpulse <= 0.0 || particle.inverseMass <= 0.0) {
            continue;
        }
        const Vec3 tangentVelocity{
            particle.velocity.x,
            particle.velocity.y,
            0.0,
        };
        const double slipSpeed = length(tangentVelocity);
        if (slipSpeed < 1.0e-10) {
            continue;
        }
        const double frictionLimit =
            kClothGroundFriction * normalImpulse;
        const double tangentialImpulse = std::min(
            slipSpeed / particle.inverseMass,
            frictionLimit
        );
        if (tangentialImpulse <= 0.0) {
            continue;
        }
        particle.velocity -= tangentVelocity * (
            tangentialImpulse * particle.inverseMass / slipSpeed
        );
        recordFrictionImpulse(
            metrics,
            tangentialImpulse,
            frictionLimit
        );
        ++metrics.clothGroundFrictionContacts;
    }
}

struct RollingProbeResult {
    double linearSpeed{};
    double angularSpeed{};
    double contactSlipSpeed{};
    double energyRatio{};
    Metrics metrics{};
    Quaternion orientation{};
};

RollingProbeResult runRollingProbeOnce() {
    constexpr double timestep = 1.0 / 480.0;
    constexpr std::uint32_t steps = 240u;
    constexpr double mass = 0.21;
    std::array<Ball, kFruitCount> balls{};
    Ball& ball = balls[0];
    ball.radius = 0.07;
    ball.inverseMass = 1.0 / mass;
    ball.position = {0.0, 0.0, ball.radius};
    ball.previous = ball.position;
    ball.velocity = {1.0, 0.0, 0.0};
    Metrics metrics;
    const double initialEnergy =
        0.5 * mass * lengthSquared(ball.velocity);
    for (std::uint32_t step = 0; step < steps; ++step) {
        std::array<double, kFruitCount> normalImpulses{};
        normalImpulses[0] = mass * 9.81 * timestep;
        applyBallGroundFriction(
            balls,
            normalImpulses,
            timestep,
            metrics,
            0.0
        );
        integrateOrientation(ball, timestep);
        ball.position += ball.velocity * timestep;
    }
    const Vec3 contactOffset{0.0, 0.0, -ball.radius};
    const Vec3 contactVelocity = ball.velocity +
        cross(ball.angularVelocity, contactOffset);
    const double finalEnergy =
        0.5 * mass * lengthSquared(ball.velocity) +
        0.5 * lengthSquared(ball.angularVelocity) / inverseInertia(ball);
    return {
        .linearSpeed = length(ball.velocity),
        .angularSpeed = length(ball.angularVelocity),
        .contactSlipSpeed = std::hypot(contactVelocity.x, contactVelocity.y),
        .energyRatio = finalEnergy / initialEnergy,
        .metrics = metrics,
        .orientation = ball.orientation,
    };
}

bool runRollingProbe() {
    const RollingProbeResult first = runRollingProbeOnce();
    const RollingProbeResult replay = runRollingProbeOnce();
    constexpr double expectedLinearSpeed = 5.0 / 7.0;
    const bool deterministic =
        first.linearSpeed == replay.linearSpeed &&
        first.angularSpeed == replay.angularSpeed &&
        first.contactSlipSpeed == replay.contactSlipSpeed &&
        first.energyRatio == replay.energyRatio &&
        first.orientation.w == replay.orientation.w &&
        first.orientation.x == replay.orientation.x &&
        first.orientation.y == replay.orientation.y &&
        first.orientation.z == replay.orientation.z;
    const bool pass = deterministic &&
        std::abs(first.linearSpeed - expectedLinearSpeed) < 1.0e-9 &&
        std::abs(first.angularSpeed * 0.07 - expectedLinearSpeed) < 1.0e-9 &&
        first.contactSlipSpeed < 1.0e-9 &&
        std::abs(first.energyRatio - 5.0 / 7.0) < 1.0e-9 &&
        first.metrics.maximumFrictionConeRatio <= 1.0 + 1.0e-12;
    std::cout << std::fixed << std::setprecision(12)
              << "probe=solid_sphere_slide_to_roll"
              << " initial_speed=1.000000000000"
              << " final_speed=" << first.linearSpeed
              << " expected_speed=" << expectedLinearSpeed
              << " radius_times_omega=" << first.angularSpeed * 0.07
              << " contact_slip=" << first.contactSlipSpeed
              << " energy_ratio=" << first.energyRatio
              << " friction_cone_ratio="
              << first.metrics.maximumFrictionConeRatio
              << " deterministic=" << std::boolalpha << deterministic
              << '\n'
              << "result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass;
}

struct RollingResistanceProbeCase {
    double horizontalAngularSpeed{};
    double verticalAngularSpeed{};
    double energy{};
    double ratio{};
    std::uint64_t contacts{};
};

RollingResistanceProbeCase runRollingResistanceProbeCase(
    const double normalImpulse
) {
    Ball ball;
    ball.radius = 0.10;
    ball.inverseMass = 1.0;
    ball.angularVelocity = {3.0, 4.0, 2.0};
    Metrics metrics;
    applyBallRollingResistance(
        ball,
        normalImpulse,
        kFruitRollingResistanceCoefficient,
        metrics
    );
    return {
        .horizontalAngularSpeed = std::hypot(
            ball.angularVelocity.x,
            ball.angularVelocity.y
        ),
        .verticalAngularSpeed = ball.angularVelocity.z,
        .energy = 0.5 * lengthSquared(ball.angularVelocity) /
            inverseInertia(ball),
        .ratio = metrics.maximumRollingResistanceRatio,
        .contacts = metrics.ballRollingResistanceContacts,
    };
}

bool runRollingResistanceProbe() {
    const RollingResistanceProbeCase limited =
        runRollingResistanceProbeCase(2.0);
    const RollingResistanceProbeCase sticking =
        runRollingResistanceProbeCase(100.0);
    const RollingResistanceProbeCase unloaded =
        runRollingResistanceProbeCase(0.0);
    const RollingResistanceProbeCase limitedReplay =
        runRollingResistanceProbeCase(2.0);
    const RollingResistanceProbeCase stickingReplay =
        runRollingResistanceProbeCase(100.0);
    const RollingResistanceProbeCase unloadedReplay =
        runRollingResistanceProbeCase(0.0);
    const bool deterministic =
        limited.horizontalAngularSpeed ==
            limitedReplay.horizontalAngularSpeed &&
        limited.verticalAngularSpeed == limitedReplay.verticalAngularSpeed &&
        limited.energy == limitedReplay.energy &&
        limited.ratio == limitedReplay.ratio &&
        sticking.horizontalAngularSpeed ==
            stickingReplay.horizontalAngularSpeed &&
        sticking.verticalAngularSpeed == stickingReplay.verticalAngularSpeed &&
        unloaded.horizontalAngularSpeed ==
            unloadedReplay.horizontalAngularSpeed;
    const bool pass = deterministic &&
        std::abs(limited.horizontalAngularSpeed - 4.25) < 1.0e-12 &&
        std::abs(limited.verticalAngularSpeed - 2.0) < 1.0e-12 &&
        std::abs(limited.ratio - 1.0) < 1.0e-12 &&
        limited.contacts == 1u &&
        sticking.horizontalAngularSpeed < 1.0e-12 &&
        std::abs(sticking.verticalAngularSpeed - 2.0) < 1.0e-12 &&
        sticking.contacts == 1u &&
        std::abs(unloaded.horizontalAngularSpeed - 5.0) < 1.0e-12 &&
        std::abs(unloaded.verticalAngularSpeed - 2.0) < 1.0e-12 &&
        unloaded.contacts == 0u &&
        limited.energy < unloaded.energy;
    std::cout << std::fixed << std::setprecision(12)
              << "probe=load_dependent_fruit_rolling_resistance"
              << " coefficient=" << kFruitRollingResistanceCoefficient
              << " limited_horizontal_omega="
              << limited.horizontalAngularSpeed
              << " limited_vertical_omega="
              << limited.verticalAngularSpeed
              << " limited_ratio=" << limited.ratio
              << " sticking_horizontal_omega="
              << sticking.horizontalAngularSpeed
              << " sticking_vertical_omega="
              << sticking.verticalAngularSpeed
              << " unloaded_horizontal_omega="
              << unloaded.horizontalAngularSpeed
              << " deterministic=" << std::boolalpha << deterministic
              << '\n'
              << "result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass;
}

struct ClothGroundFrictionProbeCase {
    double speed{};
    double energy{};
    double coneRatio{};
    std::uint64_t contacts{};
};

ClothGroundFrictionProbeCase runClothGroundFrictionProbeCase(
    const double normalImpulse
) {
    std::vector<Particle> particles(1);
    particles[0].mass = 1.0;
    particles[0].inverseMass = 1.0;
    particles[0].velocity = {1.0, 0.0, 0.0};
    Metrics metrics;
    applyClothGroundFriction(
        particles,
        {normalImpulse},
        metrics
    );
    return {
        .speed = length(particles[0].velocity),
        .energy = 0.5 * lengthSquared(particles[0].velocity),
        .coneRatio = metrics.maximumFrictionConeRatio,
        .contacts = metrics.clothGroundFrictionContacts,
    };
}

bool runClothGroundFrictionProbe() {
    const ClothGroundFrictionProbeCase sticking =
        runClothGroundFrictionProbeCase(10.0);
    const ClothGroundFrictionProbeCase sliding =
        runClothGroundFrictionProbeCase(1.0);
    const ClothGroundFrictionProbeCase unloaded =
        runClothGroundFrictionProbeCase(0.0);
    const ClothGroundFrictionProbeCase stickingReplay =
        runClothGroundFrictionProbeCase(10.0);
    const ClothGroundFrictionProbeCase slidingReplay =
        runClothGroundFrictionProbeCase(1.0);
    const ClothGroundFrictionProbeCase unloadedReplay =
        runClothGroundFrictionProbeCase(0.0);
    const bool deterministic =
        sticking.speed == stickingReplay.speed &&
        sticking.energy == stickingReplay.energy &&
        sticking.coneRatio == stickingReplay.coneRatio &&
        sliding.speed == slidingReplay.speed &&
        sliding.energy == slidingReplay.energy &&
        sliding.coneRatio == slidingReplay.coneRatio &&
        unloaded.speed == unloadedReplay.speed &&
        unloaded.energy == unloadedReplay.energy;
    const bool pass = deterministic &&
        sticking.speed < 1.0e-12 &&
        sticking.energy < 1.0e-12 &&
        sticking.contacts == 1u &&
        std::abs(sliding.speed - 0.55) < 1.0e-12 &&
        std::abs(sliding.energy - 0.15125) < 1.0e-12 &&
        std::abs(sliding.coneRatio - 1.0) < 1.0e-12 &&
        sliding.contacts == 1u &&
        std::abs(unloaded.speed - 1.0) < 1.0e-12 &&
        unloaded.contacts == 0u;
    std::cout << std::fixed << std::setprecision(12)
              << "probe=cloth_ground_coulomb_friction"
              << " coefficient=" << kClothGroundFriction
              << " sticking_speed=" << sticking.speed
              << " sticking_energy=" << sticking.energy
              << " sliding_speed=" << sliding.speed
              << " sliding_energy=" << sliding.energy
              << " sliding_cone_ratio=" << sliding.coneRatio
              << " unloaded_speed=" << unloaded.speed
              << " deterministic=" << std::boolalpha << deterministic
              << '\n'
              << "result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass;
}


struct DeformableResponseProbeCase {
    double separationError{};
    double ballAdvance{};
    double clothAdvance{};
    double centerShift{};
    double groundViolation{};
    std::uint64_t contacts{};
};

DeformableResponseProbeCase runDeformableResponseProbeCase(
    const bool groundEnabled,
    const double clearance
) {
    constexpr double penetration = 0.008;
    constexpr double ballMass = 0.20;
    constexpr double ballRadius = 0.10;
    const double yarnHeight = groundEnabled
        ? kClothRadius + clearance
        : 0.0;
    std::vector<Particle> particles(2);
    const std::array<Vec3, 2> positions{{
        {-1.0, 0.0, yarnHeight},
        {1.0, 0.0, yarnHeight},
    }};
    for (std::size_t index = 0; index < particles.size(); ++index) {
        particles[index].position = positions[index];
        particles[index].previous = positions[index];
        particles[index].inverseMass = 1.0 / kClothNodeMass;
        particles[index].mass = kClothNodeMass;
    }
    Ball ball;
    ball.radius = ballRadius;
    ball.inverseMass = 1.0 / ballMass;
    ball.position = {
        0.0,
        0.0,
        yarnHeight + ballRadius + kClothRadius - penetration,
    };
    ball.previous = ball.position;
    const double ballStart = ball.position.z;
    const double clothStart = yarnHeight;
    const double totalMass = ballMass + 2.0 * kClothNodeMass;
    const double centerBefore = (
        ballMass * ball.position.z +
        kClothNodeMass * (
            particles[0].position.z + particles[1].position.z
        )
    ) / totalMass;
    BallYarnContactImpulse contact;
    std::uint64_t contacts = 0u;
    solveBallYarn(
        particles,
        Edge{0u, 1u},
        ball,
        1.0 / 480.0,
        groundEnabled,
        contact,
        contacts
    );
    const SegmentClosest closest = closestPointsOnSegments(
        ball.position, ball.position,
        particles[0].position, particles[1].position
    );
    const double clothHeight = (
        particles[0].position.z + particles[1].position.z
    ) / 2.0;
    const double centerAfter = (
        ballMass * ball.position.z +
        kClothNodeMass * (
            particles[0].position.z + particles[1].position.z
        )
    ) / totalMass;
    double groundViolation = 0.0;
    if (groundEnabled) {
        for (const Particle& particle : particles) {
            groundViolation = std::max(
                groundViolation,
                kClothRadius - particle.position.z
            );
        }
    }
    return {
        .separationError = std::abs(
            length(closest.secondPoint - ball.position) -
            (ballRadius + kClothRadius)
        ),
        .ballAdvance = ball.position.z - ballStart,
        .clothAdvance = clothStart - clothHeight,
        .centerShift = centerAfter - centerBefore,
        .groundViolation = groundViolation,
        .contacts = contacts,
    };
}

bool runDeformableResponseProbe() {
    constexpr double penetration = 0.008;
    constexpr double crossingClearance = 0.001;
    constexpr double ballInverseMass = 1.0 / 0.20;
    const double clothPointInverseMass =
        1.0 / (2.0 * kClothNodeMass);
    const double freeBallAdvance = penetration * ballInverseMass /
        (ballInverseMass + clothPointInverseMass);
    const DeformableResponseProbeCase free =
        runDeformableResponseProbeCase(false, 0.0);
    const DeformableResponseProbeCase supported =
        runDeformableResponseProbeCase(true, 0.0);
    const DeformableResponseProbeCase crossing =
        runDeformableResponseProbeCase(true, crossingClearance);
    const DeformableResponseProbeCase freeReplay =
        runDeformableResponseProbeCase(false, 0.0);
    const DeformableResponseProbeCase supportedReplay =
        runDeformableResponseProbeCase(true, 0.0);
    const DeformableResponseProbeCase crossingReplay =
        runDeformableResponseProbeCase(true, crossingClearance);
    const bool deterministic =
        free.ballAdvance == freeReplay.ballAdvance &&
        free.clothAdvance == freeReplay.clothAdvance &&
        free.centerShift == freeReplay.centerShift &&
        supported.ballAdvance == supportedReplay.ballAdvance &&
        supported.clothAdvance == supportedReplay.clothAdvance &&
        crossing.ballAdvance == crossingReplay.ballAdvance &&
        crossing.clothAdvance == crossingReplay.clothAdvance;
    const bool pass = deterministic &&
        free.contacts == 1u && supported.contacts == 1u &&
        crossing.contacts == 1u &&
        free.separationError < 1.0e-12 &&
        supported.separationError < 1.0e-12 &&
        crossing.separationError < 1.0e-12 &&
        std::abs(free.ballAdvance - freeBallAdvance) < 1.0e-12 &&
        std::abs(free.centerShift) < 1.0e-12 &&
        std::abs(supported.ballAdvance - penetration) < 1.0e-12 &&
        std::abs(supported.clothAdvance) < 1.0e-12 &&
        supported.groundViolation < 1.0e-12 &&
        std::abs(crossing.ballAdvance -
            (penetration - crossingClearance)) < 1.0e-12 &&
        std::abs(crossing.clothAdvance - crossingClearance) < 1.0e-12 &&
        crossing.groundViolation < 1.0e-12;
    std::cout << std::fixed << std::setprecision(12)
              << "probe=coupled_yarn_deformable_contact_response"
              << " node_mass_kg=" << kClothNodeMass
              << " free_ball_advance=" << free.ballAdvance
              << " free_cloth_advance=" << free.clothAdvance
              << " free_center_shift=" << free.centerShift
              << " supported_ball_advance=" << supported.ballAdvance
              << " supported_cloth_advance=" << supported.clothAdvance
              << " crossing_ball_advance=" << crossing.ballAdvance
              << " crossing_cloth_advance=" << crossing.clothAdvance
              << " maximum_separation_error=" << std::max({
                    free.separationError,
                    supported.separationError,
                    crossing.separationError,
                 })
              << " maximum_ground_violation=" << std::max(
                    supported.groundViolation,
                    crossing.groundViolation
                 )
              << " deterministic=" << std::boolalpha << deterministic
              << '\n'
              << "result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass;
}

double solveSweptEdgeEdge(
    ClothModel& cloth,
    Edge first,
    Edge second,
    std::uint64_t& contactCount,
    SelfEdgeEdgeContactImpulse* contactImpulse = nullptr,
    double timestep = 1.0
);

struct SelfCCDProbeResult {
    double edgeHeight{};
    double edgeRemovedAdvance{};
    std::uint64_t edgeContacts{};
};

SelfCCDProbeResult runSelfCCDProbeOnce() {
    ClothModel edgeCloth;
    edgeCloth.particles.resize(4);
    edgeCloth.particles[0].position = {-1.0, 0.0, 0.0};
    edgeCloth.particles[1].position = {1.0, 0.0, 0.0};
    edgeCloth.particles[2].previous = {0.0, -1.0, 0.02};
    edgeCloth.particles[2].position = {0.0, -1.0, -0.02};
    edgeCloth.particles[3].previous = {0.0, 1.0, 0.02};
    edgeCloth.particles[3].position = {0.0, 1.0, -0.02};
    edgeCloth.particles[2].inverseMass = 1.0;
    edgeCloth.particles[3].inverseMass = 1.0;
    edgeCloth.particles[0].previous = edgeCloth.particles[0].position;
    edgeCloth.particles[1].previous = edgeCloth.particles[1].position;
    edgeCloth.localTopology = {{0u}, {1u}, {2u}, {3u}};
    std::uint64_t edgeContacts = 0u;
    const double edgeRemovedAdvance = solveSweptEdgeEdge(
        edgeCloth,
        Edge{0u, 1u},
        Edge{2u, 3u},
        edgeContacts
    );
    return {
        .edgeHeight = 0.5 * (
            edgeCloth.particles[2].position.z +
            edgeCloth.particles[3].position.z
        ),
        .edgeRemovedAdvance = edgeRemovedAdvance,
        .edgeContacts = edgeContacts,
    };
}

bool runSelfCCDProbe() {
    const SelfCCDProbeResult first = runSelfCCDProbeOnce();
    const SelfCCDProbeResult replay = runSelfCCDProbeOnce();
    const bool deterministic =
        first.edgeHeight == replay.edgeHeight &&
        first.edgeRemovedAdvance == replay.edgeRemovedAdvance &&
        first.edgeContacts == replay.edgeContacts;
    const bool pass = deterministic &&
        first.edgeContacts == 1u &&
        std::abs(first.edgeHeight - 2.0 * kClothRadius) < 2.0e-9 &&
        first.edgeRemovedAdvance > 0.027;
    std::cout << std::fixed << std::setprecision(12)
              << "probe=continuous_yarn_capsule_self_contact"
              << " edge_start=0.020000000000"
              << " edge_predicted=-0.020000000000"
              << " edge_contact_height=" << first.edgeHeight
              << " edge_removed_advance=" << first.edgeRemovedAdvance
              << " edge_contacts=" << first.edgeContacts
              << " deterministic=" << std::boolalpha << deterministic
              << '\n'
              << "result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass;
}

struct StrainProbeResult {
    double extension{};
    double correction{};
    double center{};
};

StrainProbeResult runStrainProbeOnce() {
    std::vector<Particle> particles(2);
    particles[0].position = {-0.75, 0.0, 0.0};
    particles[1].position = {0.75, 0.0, 0.0};
    particles[0].inverseMass = 1.0;
    particles[1].inverseMass = 1.0;
    const DistanceConstraint constraint{
        .first = 0u,
        .second = 1u,
        .restLength = 1.0,
        .kind = DistanceKind::warp,
    };
    const double correction = solveStrainLimits(particles, {constraint});
    return {
        .extension = length(
            particles[1].position - particles[0].position
        ) - 1.0,
        .correction = correction,
        .center = 0.5 * (particles[0].position.x + particles[1].position.x),
    };
}

bool runStrainProbe() {
    const StrainProbeResult first = runStrainProbeOnce();
    const StrainProbeResult replay = runStrainProbeOnce();
    const bool deterministic =
        first.extension == replay.extension &&
        first.correction == replay.correction &&
        first.center == replay.center;
    const bool pass = deterministic &&
        std::abs(first.extension - 0.285) < 1.0e-12 &&
        std::abs(first.correction - 0.215) < 1.0e-12 &&
        std::abs(first.center) < 1.0e-12;
    std::cout << std::fixed << std::setprecision(12)
              << "probe=unilateral_cloth_strain_limit"
              << " predicted_extension=0.500000000000"
              << " limited_extension=" << first.extension
              << " maximum_extension=0.285000000000"
              << " correction=" << first.correction
              << " center_shift=" << first.center
              << " deterministic=" << std::boolalpha << deterministic
              << '\n'
              << "result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass;
}

double solveGround(
    std::vector<Particle>& particles,
    std::array<Ball, kFruitCount>& balls,
    const double timestep,
    std::array<double, kFruitCount>& ballNormalImpulses,
    double& maximumClothCorrection,
    double& maximumBallCorrection
) {
    double maximumPenetration = 0.0;
    for (Particle& particle : particles) {
        const double penetration = kClothRadius - particle.position.z;
        if (penetration > 0.0) {
            maximumPenetration = std::max(maximumPenetration, penetration);
            maximumClothCorrection = std::max(
                maximumClothCorrection,
                penetration
            );
            particle.position.z = kClothRadius;
        }
    }
    for (std::size_t index = 0; index < balls.size(); ++index) {
        Ball& ball = balls[index];
        const double penetration = ball.radius - ball.position.z;
        if (penetration > 0.0) {
            maximumPenetration = std::max(maximumPenetration, penetration);
            maximumBallCorrection = std::max(
                maximumBallCorrection,
                penetration
            );
            ballNormalImpulses[index] +=
                penetration / (ball.inverseMass * timestep);
            ball.position.z = ball.radius;
        }
    }
    return maximumPenetration;
}

double sweepGroundPrediction(
    std::vector<Particle>& particles,
    std::array<Ball, kFruitCount>& balls,
    const double timestep,
    std::array<double, kFruitCount>& ballNormalImpulses
) {
    double maximumAdvance = 0.0;
    for (Particle& particle : particles) {
        if (particle.previous.z >= kClothRadius &&
            particle.position.z < kClothRadius) {
            maximumAdvance = std::max(
                maximumAdvance,
                kClothRadius - particle.position.z
            );
            particle.position.z = kClothRadius;
        }
    }
    for (std::size_t index = 0; index < balls.size(); ++index) {
        Ball& ball = balls[index];
        if (ball.previous.z >= ball.radius &&
            ball.position.z < ball.radius) {
            const double removedAdvance = ball.radius - ball.position.z;
            maximumAdvance = std::max(maximumAdvance, removedAdvance);
            ballNormalImpulses[index] +=
                removedAdvance / (ball.inverseMass * timestep);
            ball.position.z = ball.radius;
        }
    }
    return maximumAdvance;
}

bool localTopologyPair(
    const ClothModel& cloth,
    const std::uint32_t first,
    const std::uint32_t second
) {
    const std::vector<std::uint32_t>& local = cloth.localTopology[first];
    return std::binary_search(local.begin(), local.end(), second);
}

std::uint64_t spatialKey(const int x, const int y, const int z) {
    constexpr int bias = 1 << 20;
    const auto encode = [&](const int value) {
        return static_cast<std::uint64_t>(value + bias) & 0x1fffffu;
    };
    return encode(x) | (encode(y) << 21u) | (encode(z) << 42u);
}

bool edgePairLocal(
    const ClothModel& cloth,
    const Edge first,
    const Edge second
) {
    return localTopologyPair(cloth, first.first, second.first) ||
        localTopologyPair(cloth, first.first, second.second) ||
        localTopologyPair(cloth, first.second, second.first) ||
        localTopologyPair(cloth, first.second, second.second);
}

SegmentClosest closestPointsOnSegments(
    const Vec3 firstStart,
    const Vec3 firstEnd,
    const Vec3 secondStart,
    const Vec3 secondEnd
) {
    const Vec3 firstDirection = firstEnd - firstStart;
    const Vec3 secondDirection = secondEnd - secondStart;
    const Vec3 offset = firstStart - secondStart;
    const double firstLengthSquared = lengthSquared(firstDirection);
    const double secondLengthSquared = lengthSquared(secondDirection);
    const double secondProjection = dot(secondDirection, offset);
    double firstWeight = 0.0;
    double secondWeight = 0.0;
    if (firstLengthSquared <= 1.0e-20 &&
        secondLengthSquared <= 1.0e-20) {
        return {firstStart, secondStart, 0.0, 0.0};
    }
    if (firstLengthSquared <= 1.0e-20) {
        secondWeight = std::clamp(
            secondProjection / secondLengthSquared,
            0.0,
            1.0
        );
    } else {
        const double firstProjection = dot(firstDirection, offset);
        if (secondLengthSquared <= 1.0e-20) {
            firstWeight = std::clamp(
                -firstProjection / firstLengthSquared,
                0.0,
                1.0
            );
        } else {
            const double crossProjection =
                dot(firstDirection, secondDirection);
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
                    -firstProjection / firstLengthSquared,
                    0.0,
                    1.0
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
    return {
        firstStart + firstDirection * firstWeight,
        secondStart + secondDirection * secondWeight,
        firstWeight,
        secondWeight,
    };
}

struct SweptBallYarnSample {
    Vec3 ballPosition{};
    std::array<Vec3, 2> segment{};
    Vec3 closest{};
    double segmentWeight{};
    double distance{};
};

SweptBallYarnSample sampleSweptBallYarn(
    const std::vector<Particle>& particles,
    const Edge segment,
    const Ball& ball,
    const double time
) {
    SweptBallYarnSample sample;
    sample.ballPosition = ball.previous +
        (ball.position - ball.previous) * time;
    const std::array<std::uint32_t, 2> indices{{
        segment.first, segment.second,
    }};
    for (std::size_t index = 0u; index < indices.size(); ++index) {
        const Particle& particle = particles[indices[index]];
        sample.segment[index] = particle.previous +
            (particle.position - particle.previous) * time;
    }
    const SegmentClosest closest = closestPointsOnSegments(
        sample.ballPosition,
        sample.ballPosition,
        sample.segment[0],
        sample.segment[1]
    );
    sample.closest = closest.secondPoint;
    sample.segmentWeight = closest.secondWeight;
    sample.distance = length(sample.closest - sample.ballPosition);
    return sample;
}

double applyBallYarnCorrection(
    std::vector<Particle>& particles,
    const Edge segment,
    const std::array<double, 2> weights,
    Ball& ball,
    const Vec3 normal,
    const double correctionDistance,
    const bool groundEnabled
) {
    const std::array<std::uint32_t, 2> indices{{
        segment.first, segment.second,
    }};
    std::array<bool, 2> groundActive{};
    if (groundEnabled && normal.z < 0.0) {
        for (std::size_t index = 0u; index < indices.size(); ++index) {
            groundActive[index] = weights[index] > 0.0 &&
                particles[indices[index]].position.z <=
                    kClothRadius + 1.0e-9;
            if (groundActive[index]) {
                particles[indices[index]].position.z = kClothRadius;
            }
        }
    }

    double remaining = correctionDistance;
    double accumulatedLambda = 0.0;
    for (std::size_t activeSetIteration = 0u;
         activeSetIteration < 3u && remaining > 1.0e-14;
         ++activeSetIteration) {
        std::array<Vec3, 2> response{};
        double denominator = ball.inverseMass;
        for (std::size_t index = 0u; index < indices.size(); ++index) {
            response[index] = normal *
                particles[indices[index]].inverseMass;
            if (groundActive[index]) {
                response[index].z = 0.0;
            }
            denominator += dot(normal, response[index]) *
                weights[index] * weights[index];
        }
        if (denominator <= 0.0) {
            break;
        }

        const double unconstrainedLambda = remaining / denominator;
        double stepLambda = unconstrainedLambda;
        if (groundEnabled && normal.z < 0.0) {
            for (std::size_t index = 0u; index < indices.size(); ++index) {
                const double verticalResponse = response[index].z *
                    weights[index];
                if (groundActive[index] || verticalResponse >= 0.0) {
                    continue;
                }
                const double distanceToGround = std::max(
                    0.0,
                    particles[indices[index]].position.z - kClothRadius
                );
                stepLambda = std::min(
                    stepLambda,
                    distanceToGround / -verticalResponse
                );
            }
        }

        ball.position -= normal * (ball.inverseMass * stepLambda);
        for (std::size_t index = 0u; index < indices.size(); ++index) {
            particles[indices[index]].position += response[index] *
                (weights[index] * stepLambda);
        }
        accumulatedLambda += stepLambda;
        remaining = std::max(0.0, remaining - denominator * stepLambda);
        if (stepLambda >= unconstrainedLambda - 1.0e-14) {
            break;
        }

        bool activated = false;
        for (std::size_t index = 0u; index < indices.size(); ++index) {
            if (!groundActive[index] && weights[index] > 0.0 &&
                particles[indices[index]].position.z <=
                    kClothRadius + 1.0e-9) {
                particles[indices[index]].position.z = kClothRadius;
                groundActive[index] = true;
                activated = true;
            }
        }
        if (!activated) {
            break;
        }
    }
    return accumulatedLambda;
}

double solveBallYarn(
    std::vector<Particle>& particles,
    const Edge segment,
    Ball& ball,
    const double timestep,
    const bool groundEnabled,
    BallYarnContactImpulse& contactImpulse,
    std::uint64_t& contactCount
) {
    const Vec3 first = particles[segment.first].position;
    const Vec3 second = particles[segment.second].position;
    const double target = ball.radius + kClothRadius;
    if (ball.position.x < std::min(first.x, second.x) - target ||
        ball.position.x > std::max(first.x, second.x) + target ||
        ball.position.y < std::min(first.y, second.y) - target ||
        ball.position.y > std::max(first.y, second.y) + target ||
        ball.position.z < std::min(first.z, second.z) - target ||
        ball.position.z > std::max(first.z, second.z) + target) {
        return 0.0;
    }
    const SegmentClosest closest = closestPointsOnSegments(
        ball.position, ball.position, first, second
    );
    Vec3 separation = closest.secondPoint - ball.position;
    double distance = length(separation);
    if (distance >= target) {
        return 0.0;
    }
    if (distance < 1.0e-12) {
        const Vec3 segmentDirection = normalized(second - first);
        separation = normalized(cross(
            segmentDirection,
            std::abs(segmentDirection.z) < 0.9
                ? Vec3{0.0, 0.0, 1.0}
                : Vec3{1.0, 0.0, 0.0}
        ));
        distance = 0.0;
    }
    const Vec3 normal = normalized(separation);
    const std::array<double, 2> weights{{
        1.0 - closest.secondWeight,
        closest.secondWeight,
    }};
    const double correction = target - distance;
    const double lambda = applyBallYarnCorrection(
        particles,
        segment,
        weights,
        ball,
        normal,
        correction,
        groundEnabled
    );
    if (lambda <= 0.0) {
        return correction;
    }
    const double impulseMagnitude = lambda / timestep;
    contactImpulse.weightedNormalOnBall -= normal * impulseMagnitude;
    for (std::size_t index = 0u; index < weights.size(); ++index) {
        contactImpulse.weightedSegment[index] +=
            weights[index] * impulseMagnitude;
    }
    contactImpulse.normalImpulse += impulseMagnitude;
    ++contactCount;
    return correction;
}

double solveSweptBallYarn(
    std::vector<Particle>& particles,
    const Edge segment,
    Ball& ball,
    const double timestep,
    const bool groundEnabled,
    BallYarnContactImpulse& contactImpulse,
    std::uint64_t& contactCount
) {
    const double target = ball.radius + kClothRadius;
    const std::array<std::uint32_t, 2> indices{{
        segment.first, segment.second,
    }};
    double motionBound = length(ball.position - ball.previous);
    for (const std::uint32_t index : indices) {
        motionBound += length(
            particles[index].position - particles[index].previous
        );
    }
    if (motionBound < 1.0e-14) {
        return 0.0;
    }
    constexpr double distanceTolerance = 1.0e-9;
    double time = 0.0;
    SweptBallYarnSample impact = sampleSweptBallYarn(
        particles, segment, ball, 0.0
    );
    bool found = impact.distance <= target + distanceTolerance;
    if (!found) {
        for (std::uint32_t iteration = 0u; iteration < 80u; ++iteration) {
            impact = sampleSweptBallYarn(
                particles, segment, ball, time
            );
            const double gap = impact.distance - target;
            if (gap <= distanceTolerance) {
                found = true;
                break;
            }
            const double advance = 0.9 * gap / motionBound;
            if (!std::isfinite(advance) || advance <= 0.0 ||
                time + advance >= 1.0) {
                break;
            }
            time += std::max(advance, 1.0e-10);
        }
    }
    if (!found || impact.distance < 1.0e-12) {
        return 0.0;
    }
    const Vec3 normal =
        (impact.closest - impact.ballPosition) / impact.distance;
    const std::array<double, 2> weights{{
        1.0 - impact.segmentWeight,
        impact.segmentWeight,
    }};
    const Vec3 segmentRemaining =
        (particles[segment.first].position - impact.segment[0]) * weights[0] +
        (particles[segment.second].position - impact.segment[1]) * weights[1];
    const Vec3 ballRemaining = ball.position - impact.ballPosition;
    const double removedAdvance = dot(
        ballRemaining - segmentRemaining,
        normal
    );
    if (removedAdvance <= 0.0) {
        return 0.0;
    }
    const double lambda = applyBallYarnCorrection(
        particles,
        segment,
        weights,
        ball,
        normal,
        removedAdvance,
        groundEnabled
    );
    if (lambda <= 0.0) {
        return 0.0;
    }
    const double impulseMagnitude = lambda / timestep;
    contactImpulse.weightedNormalOnBall -= normal * impulseMagnitude;
    for (std::size_t index = 0u; index < weights.size(); ++index) {
        contactImpulse.weightedSegment[index] +=
            weights[index] * impulseMagnitude;
    }
    contactImpulse.normalImpulse += impulseMagnitude;
    ++contactCount;
    return removedAdvance;
}

double measureBallYarnPenetration(
    const ClothModel& cloth,
    const std::array<Ball, kFruitCount>& balls
) {
    double maximum = 0.0;
    for (const Ball& ball : balls) {
        const double target = ball.radius + kClothRadius;
        for (const Edge segment : cloth.yarnSegments) {
            const Vec3 first = cloth.particles[segment.first].position;
            const Vec3 second = cloth.particles[segment.second].position;
            if (ball.position.x < std::min(first.x, second.x) - target ||
                ball.position.x > std::max(first.x, second.x) + target ||
                ball.position.y < std::min(first.y, second.y) - target ||
                ball.position.y > std::max(first.y, second.y) + target ||
                ball.position.z < std::min(first.z, second.z) - target ||
                ball.position.z > std::max(first.z, second.z) + target) {
                continue;
            }
            const SegmentClosest closest = closestPointsOnSegments(
                ball.position, ball.position, first, second
            );
            maximum = std::max(
                maximum,
                target - length(closest.secondPoint - ball.position)
            );
        }
    }
    return maximum;
}

void solveBallYarnContactSweep(
    ClothModel& cloth,
    std::array<Ball, kFruitCount>& balls,
    const double timestep,
    const bool groundEnabled,
    std::vector<BallYarnContactImpulse>& contacts,
    Metrics& metrics
) {
    for (std::size_t ballIndex = 0u;
         ballIndex < balls.size();
         ++ballIndex) {
        Ball& ball = balls[ballIndex];
        for (std::size_t segmentIndex = 0u;
             segmentIndex < cloth.yarnSegments.size();
             ++segmentIndex) {
            const double penetration = solveBallYarn(
                cloth.particles,
                cloth.yarnSegments[segmentIndex],
                ball,
                timestep,
                groundEnabled,
                contacts[ballIndex * cloth.yarnSegments.size() +
                    segmentIndex],
                metrics.ballYarnContacts
            );
            metrics.maximumBallPenetration = std::max(
                metrics.maximumBallPenetration,
                penetration
            );
            metrics.maximumBallPenetrationByFruit[ballIndex] = std::max(
                metrics.maximumBallPenetrationByFruit[ballIndex],
                penetration
            );
        }
    }
}

void applyBallYarnFriction(
    ClothModel& cloth,
    std::array<Ball, kFruitCount>& balls,
    const std::vector<BallYarnContactImpulse>& contacts,
    Metrics& metrics,
    const bool groundEnabled
) {
    for (std::size_t ballIndex = 0u;
         ballIndex < balls.size();
         ++ballIndex) {
        Ball& ball = balls[ballIndex];
        for (std::size_t segmentIndex = 0u;
             segmentIndex < cloth.yarnSegments.size();
             ++segmentIndex) {
            const BallYarnContactImpulse& contact = contacts[
                ballIndex * cloth.yarnSegments.size() + segmentIndex
            ];
            if (contact.normalImpulse <= 0.0 ||
                lengthSquared(contact.weightedNormalOnBall) < 1.0e-20) {
                continue;
            }
            const Edge segment = cloth.yarnSegments[segmentIndex];
            const std::array<std::uint32_t, 2> indices{{
                segment.first, segment.second,
            }};
            const Vec3 normal = normalized(contact.weightedNormalOnBall);
            std::array<double, 2> weights{};
            double weightSum = 0.0;
            Vec3 yarnVelocity{};
            for (std::size_t index = 0u; index < weights.size(); ++index) {
                weights[index] = contact.weightedSegment[index] /
                    contact.normalImpulse;
                weightSum += weights[index];
            }
            if (weightSum <= 1.0e-12) {
                continue;
            }
            for (std::size_t index = 0u; index < weights.size(); ++index) {
                weights[index] /= weightSum;
                yarnVelocity += cloth.particles[indices[index]].velocity *
                    weights[index];
            }
            const Vec3 ballOffset = normal * -ball.radius;
            const Vec3 ballContactVelocity = ball.velocity +
                cross(ball.angularVelocity, ballOffset);
            const Vec3 relativeVelocity =
                ballContactVelocity - yarnVelocity;
            const Vec3 tangentVelocity = relativeVelocity -
                normal * dot(relativeVelocity, normal);
            const double slipSpeed = length(tangentVelocity);
            if (slipSpeed < 1.0e-10) {
                continue;
            }
            const Vec3 tangent = tangentVelocity / slipSpeed;
            double denominator = ball.inverseMass + inverseInertia(ball) *
                lengthSquared(cross(ballOffset, tangent));
            std::array<Vec3, 2> yarnResponse{};
            for (std::size_t index = 0u; index < weights.size(); ++index) {
                yarnResponse[index] = deformableParticleResponse(
                    cloth.particles[indices[index]],
                    tangent,
                    groundEnabled
                );
                denominator += dot(tangent, yarnResponse[index]) *
                    weights[index] * weights[index];
            }
            if (denominator <= 0.0) {
                continue;
            }
            const double frictionLimit =
                kFruitClothFriction * contact.normalImpulse;
            const double tangentialImpulse = std::min(
                slipSpeed / denominator,
                frictionLimit
            );
            if (tangentialImpulse <= 0.0) {
                continue;
            }
            applyBallImpulse(ball, tangent * -tangentialImpulse, ballOffset);
            for (std::size_t index = 0u; index < weights.size(); ++index) {
                cloth.particles[indices[index]].velocity +=
                    yarnResponse[index] *
                    (tangentialImpulse * weights[index]);
            }
            recordFrictionImpulse(
                metrics,
                tangentialImpulse,
                frictionLimit
            );
            ++metrics.ballClothFrictionContacts;
        }
    }
}

struct YarnMechanicsProbeResult {
    double contactHeight{};
    double removedAdvance{};
    double normalImpulse{};
    std::uint64_t contacts{};
    double knotAngleError{};
    double knotCenterShift{};
};

YarnMechanicsProbeResult runYarnMechanicsProbeOnce() {
    constexpr double timestep = 1.0e-3;
    std::vector<Particle> contactParticles(2);
    contactParticles[0].position = {-1.0, 0.0, 0.0};
    contactParticles[1].position = {1.0, 0.0, 0.0};
    for (Particle& particle : contactParticles) {
        particle.previous = particle.position;
        particle.inverseMass = 0.0;
    }
    Ball ball;
    ball.radius = 0.02;
    ball.inverseMass = 1.0;
    ball.previous = {0.0, 0.0, 0.08};
    ball.position = {0.0, 0.0, -0.08};
    ball.velocity = {0.0, 0.0, -160.0};
    BallYarnContactImpulse contact;
    std::uint64_t contacts = 0u;
    const double removedAdvance = solveSweptBallYarn(
        contactParticles,
        Edge{0u, 1u},
        ball,
        timestep,
        false,
        contact,
        contacts
    );

    std::vector<Particle> knotParticles(4);
    constexpr double initialAngle = std::numbers::pi / 3.0;
    const Vec3 weftDirection{
        std::cos(initialAngle),
        std::sin(initialAngle),
        0.0,
    };
    knotParticles[0].position = {-1.0, 0.0, 0.0};
    knotParticles[1].position = {1.0, 0.0, 0.0};
    knotParticles[2].position = weftDirection * -1.0;
    knotParticles[3].position = weftDirection;
    for (Particle& particle : knotParticles) {
        particle.previous = particle.position;
        particle.inverseMass = 1.0;
    }
    const Vec3 centerBefore = (
        knotParticles[0].position + knotParticles[1].position +
        knotParticles[2].position + knotParticles[3].position
    ) / 4.0;
    KnotConstraint knot{
        .warpFirst = 0u,
        .warpSecond = 1u,
        .weftFirst = 2u,
        .weftSecond = 3u,
        .restCosine = 0.0,
        .compliance = 2.0e-6,
    };
    for (std::uint32_t iteration = 0u; iteration < 12u; ++iteration) {
        solveKnot(knotParticles, knot, timestep);
    }
    const Vec3 finalWarp = normalized(
        knotParticles[1].position - knotParticles[0].position
    );
    const Vec3 finalWeft = normalized(
        knotParticles[3].position - knotParticles[2].position
    );
    const Vec3 centerAfter = (
        knotParticles[0].position + knotParticles[1].position +
        knotParticles[2].position + knotParticles[3].position
    ) / 4.0;
    return {
        .contactHeight = ball.position.z,
        .removedAdvance = removedAdvance,
        .normalImpulse = contact.normalImpulse,
        .contacts = contacts,
        .knotAngleError = std::abs(
            std::numbers::pi / 2.0 - std::acos(std::clamp(
                dot(finalWarp, finalWeft), -1.0, 1.0
            ))
        ),
        .knotCenterShift = length(centerAfter - centerBefore),
    };
}

bool runYarnMechanicsProbe() {
    const YarnMechanicsProbeResult first = runYarnMechanicsProbeOnce();
    const YarnMechanicsProbeResult replay = runYarnMechanicsProbeOnce();
    const double expectedHeight = 0.02 + kClothRadius;
    const double initialKnotAngleError = std::numbers::pi / 6.0;
    const bool deterministic =
        first.contactHeight == replay.contactHeight &&
        first.removedAdvance == replay.removedAdvance &&
        first.normalImpulse == replay.normalImpulse &&
        first.contacts == replay.contacts &&
        first.knotAngleError == replay.knotAngleError &&
        first.knotCenterShift == replay.knotCenterShift;
    const bool pass = deterministic && first.contacts == 1u &&
        std::abs(first.contactHeight - expectedHeight) < 2.0e-9 &&
        first.removedAdvance > 0.10 && first.normalImpulse > 100.0 &&
        first.knotAngleError < 0.75 * initialKnotAngleError &&
        first.knotAngleError > 1.0e-6 &&
        first.knotCenterShift < 1.0e-12;
    std::cout << std::fixed << std::setprecision(12)
              << "probe=explicit_yarn_capsule_and_knot"
              << " sphere_start_height=0.080000000000"
              << " sphere_predicted_height=-0.080000000000"
              << " yarn_contact_height=" << first.contactHeight
              << " expected_height=" << expectedHeight
              << " removed_advance=" << first.removedAdvance
              << " normal_impulse=" << first.normalImpulse
              << " contacts=" << first.contacts
              << " knot_initial_angle=" << std::numbers::pi / 3.0
              << " knot_target_angle=" << std::numbers::pi / 2.0
              << " knot_compliance=0.000002000000"
              << " knot_angle_error=" << first.knotAngleError
              << " knot_center_shift=" << first.knotCenterShift
              << " deterministic=" << std::boolalpha << deterministic
              << '\n'
              << "result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass;
}

Vec3 selfContactNormal(
    const Vec3 separation,
    const Vec3 fallbackFirst,
    const Vec3 fallbackSecond,
    const Vec3 previousOffset
) {
    const double separationLength = length(separation);
    if (separationLength > 1.0e-12) {
        return separation / separationLength;
    }
    Vec3 normal = normalized(cross(fallbackFirst, fallbackSecond));
    if (dot(previousOffset, normal) > 0.0) {
        normal = normal * -1.0;
    }
    return normal;
}

struct EdgeEdgeSample {
    std::array<Vec3, 2> first{};
    std::array<Vec3, 2> second{};
    SegmentClosest closest{};
    double distance{};
};

EdgeEdgeSample sampleEdgeEdge(
    const ClothModel& cloth,
    const Edge first,
    const Edge second,
    const double time
) {
    EdgeEdgeSample sample;
    const std::array<std::uint32_t, 2> firstIndices{{first.first, first.second}};
    const std::array<std::uint32_t, 2> secondIndices{{
        second.first, second.second,
    }};
    for (std::size_t index = 0; index < 2; ++index) {
        const Particle& firstParticle = cloth.particles[firstIndices[index]];
        const Particle& secondParticle = cloth.particles[secondIndices[index]];
        sample.first[index] = firstParticle.previous +
            (firstParticle.position - firstParticle.previous) * time;
        sample.second[index] = secondParticle.previous +
            (secondParticle.position - secondParticle.previous) * time;
    }
    sample.closest = closestPointsOnSegments(
        sample.first[0], sample.first[1], sample.second[0], sample.second[1]
    );
    sample.distance = length(
        sample.closest.secondPoint - sample.closest.firstPoint
    );
    return sample;
}

double solveSweptEdgeEdge(
    ClothModel& cloth,
    const Edge first,
    const Edge second,
    std::uint64_t& contactCount,
    SelfEdgeEdgeContactImpulse* contactImpulse,
    const double timestep
) {
    const double target = 2.0 * kClothRadius;
    constexpr double tolerance = 1.0e-9;
    const std::array<std::uint32_t, 4> indices{{
        first.first, first.second, second.first, second.second,
    }};
    double motionBound = 0.0;
    for (const std::uint32_t index : indices) {
        motionBound += length(
            cloth.particles[index].position -
            cloth.particles[index].previous
        );
    }
    if (motionBound < 1.0e-14) {
        return 0.0;
    }
    double time = 0.0;
    EdgeEdgeSample impact = sampleEdgeEdge(cloth, first, second, 0.0);
    bool found = impact.distance <= target + tolerance;
    if (!found) {
        for (std::uint32_t iteration = 0; iteration < 80u; ++iteration) {
            impact = sampleEdgeEdge(cloth, first, second, time);
            const double gap = impact.distance - target;
            if (gap <= tolerance) {
                found = true;
                break;
            }
            const double advance = 0.9 * gap / motionBound;
            if (!std::isfinite(advance) || advance <= 0.0 ||
                time + advance >= 1.0) {
                break;
            }
            time += std::max(advance, 1.0e-10);
        }
    }
    if (!found) {
        return 0.0;
    }
    const std::array<double, 2> firstWeights{{
        1.0 - impact.closest.firstWeight,
        impact.closest.firstWeight,
    }};
    const std::array<double, 2> secondWeights{{
        1.0 - impact.closest.secondWeight,
        impact.closest.secondWeight,
    }};
    const Vec3 firstPrevious =
        cloth.particles[first.first].previous * firstWeights[0] +
        cloth.particles[first.second].previous * firstWeights[1];
    const Vec3 secondPrevious =
        cloth.particles[second.first].previous * secondWeights[0] +
        cloth.particles[second.second].previous * secondWeights[1];
    const Vec3 normal = selfContactNormal(
        impact.closest.secondPoint - impact.closest.firstPoint,
        impact.first[1] - impact.first[0],
        impact.second[1] - impact.second[0],
        firstPrevious - secondPrevious
    );
    Vec3 firstRemaining{};
    Vec3 secondRemaining{};
    for (std::size_t index = 0; index < 2; ++index) {
        firstRemaining += (
            cloth.particles[indices[index]].position - impact.first[index]
        ) * firstWeights[index];
        secondRemaining += (
            cloth.particles[indices[index + 2u]].position - impact.second[index]
        ) * secondWeights[index];
    }
    const double removedAdvance = dot(
        firstRemaining - secondRemaining,
        normal
    );
    if (removedAdvance <= 0.0) {
        return 0.0;
    }
    double denominator = 0.0;
    for (std::size_t index = 0; index < 2; ++index) {
        denominator += cloth.particles[indices[index]].inverseMass *
            firstWeights[index] * firstWeights[index];
        denominator += cloth.particles[indices[index + 2u]].inverseMass *
            secondWeights[index] * secondWeights[index];
    }
    if (denominator <= 0.0) {
        return 0.0;
    }
    const double lambda = removedAdvance / denominator;
    for (std::size_t index = 0; index < 2; ++index) {
        cloth.particles[indices[index]].position -= normal *
            (cloth.particles[indices[index]].inverseMass *
             firstWeights[index] * lambda);
        cloth.particles[indices[index + 2u]].position += normal *
            (cloth.particles[indices[index + 2u]].inverseMass *
             secondWeights[index] * lambda);
    }
    if (contactImpulse != nullptr) {
        const double impulseMagnitude = lambda / timestep;
        contactImpulse->weightedNormalOnFirst -=
            normal * impulseMagnitude;
        for (std::size_t index = 0; index < 2; ++index) {
            contactImpulse->weightedFirst[index] +=
                firstWeights[index] * impulseMagnitude;
            contactImpulse->weightedSecond[index] +=
                secondWeights[index] * impulseMagnitude;
        }
        contactImpulse->normalImpulse += impulseMagnitude;
    }
    ++contactCount;
    return removedAdvance;
}

double solveEdgeEdge(
    ClothModel& cloth,
    const Edge first,
    const Edge second,
    std::uint64_t& contactCount,
    SelfEdgeEdgeContactImpulse* contactImpulse,
    const double timestep
) {
    const double target = 2.0 * kClothRadius;
    const SegmentClosest closest = closestPointsOnSegments(
        cloth.particles[first.first].position,
        cloth.particles[first.second].position,
        cloth.particles[second.first].position,
        cloth.particles[second.second].position
    );
    const Vec3 separation = closest.secondPoint - closest.firstPoint;
    const double distance = length(separation);
    if (distance >= target) {
        return 0.0;
    }
    const std::array<double, 2> firstWeights{{
        1.0 - closest.firstWeight, closest.firstWeight,
    }};
    const std::array<double, 2> secondWeights{{
        1.0 - closest.secondWeight, closest.secondWeight,
    }};
    const Vec3 firstPrevious =
        cloth.particles[first.first].previous * firstWeights[0] +
        cloth.particles[first.second].previous * firstWeights[1];
    const Vec3 secondPrevious =
        cloth.particles[second.first].previous * secondWeights[0] +
        cloth.particles[second.second].previous * secondWeights[1];
    const Vec3 normal = selfContactNormal(
        separation,
        cloth.particles[first.second].position -
            cloth.particles[first.first].position,
        cloth.particles[second.second].position -
            cloth.particles[second.first].position,
        firstPrevious - secondPrevious
    );
    const std::array<std::uint32_t, 4> indices{{
        first.first, first.second, second.first, second.second,
    }};
    double denominator = 0.0;
    for (std::size_t index = 0; index < 2; ++index) {
        denominator += cloth.particles[indices[index]].inverseMass *
            firstWeights[index] * firstWeights[index];
        denominator += cloth.particles[indices[index + 2u]].inverseMass *
            secondWeights[index] * secondWeights[index];
    }
    if (denominator <= 0.0) {
        return target - distance;
    }
    const double lambda = (target - distance) / denominator;
    for (std::size_t index = 0; index < 2; ++index) {
        cloth.particles[indices[index]].position -= normal *
            (cloth.particles[indices[index]].inverseMass *
             firstWeights[index] * lambda);
        cloth.particles[indices[index + 2u]].position += normal *
            (cloth.particles[indices[index + 2u]].inverseMass *
             secondWeights[index] * lambda);
    }
    if (contactImpulse != nullptr) {
        const double impulseMagnitude = lambda / timestep;
        contactImpulse->weightedNormalOnFirst -=
            normal * impulseMagnitude;
        for (std::size_t index = 0; index < 2; ++index) {
            contactImpulse->weightedFirst[index] +=
                firstWeights[index] * impulseMagnitude;
            contactImpulse->weightedSecond[index] +=
                secondWeights[index] * impulseMagnitude;
        }
        contactImpulse->normalImpulse += impulseMagnitude;
    }
    ++contactCount;
    return target - distance;
}

struct CollisionBounds {
    Vec3 minimum{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
    Vec3 maximum{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
};

void includePoint(CollisionBounds& bounds, const Vec3 point) {
    bounds.minimum.x = std::min(bounds.minimum.x, point.x);
    bounds.minimum.y = std::min(bounds.minimum.y, point.y);
    bounds.minimum.z = std::min(bounds.minimum.z, point.z);
    bounds.maximum.x = std::max(bounds.maximum.x, point.x);
    bounds.maximum.y = std::max(bounds.maximum.y, point.y);
    bounds.maximum.z = std::max(bounds.maximum.z, point.z);
}

void expandBounds(CollisionBounds& bounds, const double expansion) {
    const Vec3 amount{expansion, expansion, expansion};
    bounds.minimum -= amount;
    bounds.maximum += amount;
}

CollisionBounds edgeBounds(
    const ClothModel& cloth,
    const Edge edge,
    const bool swept,
    const double expansion
) {
    CollisionBounds bounds;
    for (const std::uint32_t index : {edge.first, edge.second}) {
        includePoint(bounds, cloth.particles[index].position);
        if (swept) {
            includePoint(bounds, cloth.particles[index].previous);
        }
    }
    expandBounds(bounds, expansion);
    return bounds;
}

struct CollisionSphere {
    Vec3 center{};
    double radius{};
};

CollisionSphere edgeCollisionSphere(
    const ClothModel& cloth,
    const Edge edge,
    const bool swept,
    const double expansion
) {
    std::array<Vec3, 4> points{{
        cloth.particles[edge.first].position,
        cloth.particles[edge.second].position,
        cloth.particles[edge.first].previous,
        cloth.particles[edge.second].previous,
    }};
    const std::size_t pointCount = swept ? 4u : 2u;
    Vec3 center{};
    for (std::size_t index = 0u; index < pointCount; ++index) {
        center += points[index];
    }
    center = center / static_cast<double>(pointCount);
    double radius = 0.0;
    for (std::size_t index = 0u; index < pointCount; ++index) {
        radius = std::max(radius, length(points[index] - center));
    }
    return {center, radius + expansion};
}

bool overlapSpheres(
    const CollisionSphere& first,
    const CollisionSphere& second
) {
    const double radius = first.radius + second.radius;
    return lengthSquared(first.center - second.center) <= radius * radius;
}

struct CollisionPrimitiveBounds {
    CollisionBounds bounds;
    std::uint32_t primitive{};
};

struct CollisionBVHNode {
    CollisionBounds bounds;
    std::uint32_t begin{};
    std::uint32_t end{};
    std::uint32_t left{std::numeric_limits<std::uint32_t>::max()};
    std::uint32_t right{std::numeric_limits<std::uint32_t>::max()};
};

struct CollisionBVH {
    std::vector<CollisionPrimitiveBounds> primitives;
    std::vector<CollisionBVHNode> nodes;
};

void includeBounds(
    CollisionBounds& destination,
    const CollisionBounds& source
) {
    includePoint(destination, source.minimum);
    includePoint(destination, source.maximum);
}

bool overlapBounds(
    const CollisionBounds& first,
    const CollisionBounds& second
) {
    return first.minimum.x <= second.maximum.x &&
        first.maximum.x >= second.minimum.x &&
        first.minimum.y <= second.maximum.y &&
        first.maximum.y >= second.minimum.y &&
        first.minimum.z <= second.maximum.z &&
        first.maximum.z >= second.minimum.z;
}

double boundsCenterAxis(
    const CollisionBounds& bounds,
    const std::uint32_t axis
) {
    if (axis == 0u) {
        return 0.5 * (bounds.minimum.x + bounds.maximum.x);
    }
    if (axis == 1u) {
        return 0.5 * (bounds.minimum.y + bounds.maximum.y);
    }
    return 0.5 * (bounds.minimum.z + bounds.maximum.z);
}

std::uint32_t buildCollisionBVHNode(
    CollisionBVH& tree,
    const std::uint32_t begin,
    const std::uint32_t end
) {
    constexpr std::uint32_t leafCapacity = 8u;
    const std::uint32_t nodeIndex =
        static_cast<std::uint32_t>(tree.nodes.size());
    tree.nodes.push_back({});
    CollisionBounds bounds;
    CollisionBounds centerBounds;
    for (std::uint32_t index = begin; index < end; ++index) {
        includeBounds(bounds, tree.primitives[index].bounds);
        const Vec3 center{
            boundsCenterAxis(tree.primitives[index].bounds, 0u),
            boundsCenterAxis(tree.primitives[index].bounds, 1u),
            boundsCenterAxis(tree.primitives[index].bounds, 2u),
        };
        includePoint(centerBounds, center);
    }
    tree.nodes[nodeIndex].bounds = bounds;
    tree.nodes[nodeIndex].begin = begin;
    tree.nodes[nodeIndex].end = end;
    if (end - begin <= leafCapacity) {
        return nodeIndex;
    }
    const Vec3 extent = centerBounds.maximum - centerBounds.minimum;
    const std::uint32_t axis = extent.y > extent.x
        ? (extent.z > extent.y ? 2u : 1u)
        : (extent.z > extent.x ? 2u : 0u);
    std::stable_sort(
        tree.primitives.begin() + begin,
        tree.primitives.begin() + end,
        [axis](
            const CollisionPrimitiveBounds& first,
            const CollisionPrimitiveBounds& second
        ) {
            const double firstCenter = boundsCenterAxis(first.bounds, axis);
            const double secondCenter = boundsCenterAxis(second.bounds, axis);
            return firstCenter < secondCenter ||
                (firstCenter == secondCenter &&
                 first.primitive < second.primitive);
        }
    );
    const std::uint32_t middle = begin + (end - begin) / 2u;
    tree.nodes[nodeIndex].left = buildCollisionBVHNode(tree, begin, middle);
    tree.nodes[nodeIndex].right = buildCollisionBVHNode(tree, middle, end);
    return nodeIndex;
}

template <typename BoundsFunction>
CollisionBVH makeCollisionBVH(
    const std::uint32_t primitiveCount,
    BoundsFunction&& boundsFunction
) {
    CollisionBVH tree;
    tree.primitives.reserve(primitiveCount);
    tree.nodes.reserve(2u * primitiveCount);
    for (std::uint32_t index = 0u; index < primitiveCount; ++index) {
        tree.primitives.push_back({boundsFunction(index), index});
    }
    if (primitiveCount != 0u) {
        buildCollisionBVHNode(tree, 0u, primitiveCount);
    }
    return tree;
}

void gatherCollisionCandidates(
    const CollisionBVH& tree,
    const CollisionBounds& bounds,
    std::vector<std::uint32_t>& candidates
) {
    candidates.clear();
    if (tree.nodes.empty()) {
        return;
    }
    std::vector<std::uint32_t> stack{0u};
    while (!stack.empty()) {
        const std::uint32_t nodeIndex = stack.back();
        stack.pop_back();
        const CollisionBVHNode& node = tree.nodes[nodeIndex];
        if (!overlapBounds(node.bounds, bounds)) {
            continue;
        }
        if (node.left == std::numeric_limits<std::uint32_t>::max()) {
            for (std::uint32_t index = node.begin; index < node.end; ++index) {
                const CollisionPrimitiveBounds& primitive =
                    tree.primitives[index];
                if (overlapBounds(primitive.bounds, bounds)) {
                    candidates.push_back(primitive.primitive);
                }
            }
            continue;
        }
        stack.push_back(node.right);
        stack.push_back(node.left);
    }
}

double solvePrimitiveSelfCollision(
    ClothModel& cloth,
    const bool swept,
    Metrics& metrics,
    const double timestep,
    SelfContactImpulses* contactImpulses
) {
    const double target = 2.0 * kClothRadius;
    const double broadphaseExpansion = 0.5 * target;
    double maximum = 0.0;
    std::vector<std::uint32_t> candidates;
    candidates.reserve(64u);
    const CollisionBVH edgeTree = makeCollisionBVH(
        static_cast<std::uint32_t>(cloth.yarnSegments.size()),
        [&](const std::uint32_t index) {
            return edgeBounds(
                cloth,
                cloth.yarnSegments[index],
                swept,
                broadphaseExpansion
            );
        }
    );
    std::vector<CollisionSphere> edgeSpheres;
    edgeSpheres.reserve(cloth.yarnSegments.size());
    for (const Edge edge : cloth.yarnSegments) {
        edgeSpheres.push_back(edgeCollisionSphere(
            cloth,
            edge,
            swept,
            broadphaseExpansion
        ));
    }
    std::uint64_t edgeContacts = 0u;
    for (std::uint32_t firstIndex = 0;
         firstIndex < cloth.yarnSegments.size();
         ++firstIndex) {
        const Edge first = cloth.yarnSegments[firstIndex];
        gatherCollisionCandidates(
            edgeTree,
            edgeBounds(cloth, first, swept, broadphaseExpansion),
            candidates
        );
        metrics.edgeEdgeCandidatePairs += candidates.size();
        for (const std::uint32_t secondIndex : candidates) {
            if (secondIndex <= firstIndex) {
                continue;
            }
            if (!overlapSpheres(
                edgeSpheres[firstIndex],
                edgeSpheres[secondIndex]
            )) {
                continue;
            }
            ++metrics.edgeEdgeSphereCandidatePairs;
            const Edge second = cloth.yarnSegments[secondIndex];
            if (edgePairLocal(cloth, first, second)) {
                continue;
            }
            SelfEdgeEdgeContactImpulse localImpulse{
                .first = first,
                .second = second,
            };
            SelfEdgeEdgeContactImpulse* contactImpulse =
                contactImpulses != nullptr ? &localImpulse : nullptr;
            const double correction = swept
                ? solveSweptEdgeEdge(
                    cloth,
                    first,
                    second,
                    edgeContacts,
                    contactImpulse,
                    timestep
                )
                : solveEdgeEdge(
                    cloth,
                    first,
                    second,
                    edgeContacts,
                    contactImpulse,
                    timestep
                );
            maximum = std::max(maximum, correction);
            if (contactImpulses != nullptr &&
                localImpulse.normalImpulse > 0.0) {
                const std::uint64_t key =
                    (static_cast<std::uint64_t>(firstIndex) << 32u) |
                    secondIndex;
                auto [found, inserted] =
                    contactImpulses->edgeEdge.try_emplace(key);
                if (inserted) {
                    found->second.first = first;
                    found->second.second = second;
                }
                found->second.weightedNormalOnFirst +=
                    localImpulse.weightedNormalOnFirst;
                for (std::size_t index = 0u; index < 2u; ++index) {
                    found->second.weightedFirst[index] +=
                        localImpulse.weightedFirst[index];
                    found->second.weightedSecond[index] +=
                        localImpulse.weightedSecond[index];
                }
                found->second.normalImpulse += localImpulse.normalImpulse;
            }
        }
    }
    if (swept) {
        metrics.sweptEdgeEdgeSelfContacts += edgeContacts;
    } else {
        metrics.edgeEdgeSelfContacts += edgeContacts;
    }
    metrics.selfContacts += edgeContacts;
    return maximum;
}

void applyClothSelfFriction(
    ClothModel& cloth,
    const SelfContactImpulses& contacts,
    Metrics& metrics
) {
    std::vector<std::uint64_t> edgeKeys;
    edgeKeys.reserve(contacts.edgeEdge.size());
    for (const auto& entry : contacts.edgeEdge) {
        edgeKeys.push_back(entry.first);
    }
    std::sort(edgeKeys.begin(), edgeKeys.end());
    for (const std::uint64_t key : edgeKeys) {
        const SelfEdgeEdgeContactImpulse& contact = contacts.edgeEdge.at(key);
        if (contact.normalImpulse <= 0.0 ||
            lengthSquared(contact.weightedNormalOnFirst) < 1.0e-20) {
            continue;
        }
        const Vec3 normal = normalized(contact.weightedNormalOnFirst);
        std::array<double, 2> firstWeights{};
        std::array<double, 2> secondWeights{};
        double firstSum = 0.0;
        double secondSum = 0.0;
        for (std::size_t index = 0; index < 2; ++index) {
            firstWeights[index] =
                contact.weightedFirst[index] / contact.normalImpulse;
            secondWeights[index] =
                contact.weightedSecond[index] / contact.normalImpulse;
            firstSum += firstWeights[index];
            secondSum += secondWeights[index];
        }
        if (firstSum <= 1.0e-12 || secondSum <= 1.0e-12) {
            continue;
        }
        Vec3 firstVelocity{};
        Vec3 secondVelocity{};
        const std::array<std::uint32_t, 2> firstIndices{{
            contact.first.first, contact.first.second,
        }};
        const std::array<std::uint32_t, 2> secondIndices{{
            contact.second.first, contact.second.second,
        }};
        for (std::size_t index = 0; index < 2; ++index) {
            firstWeights[index] /= firstSum;
            secondWeights[index] /= secondSum;
            firstVelocity += cloth.particles[firstIndices[index]].velocity *
                firstWeights[index];
            secondVelocity += cloth.particles[secondIndices[index]].velocity *
                secondWeights[index];
        }
        const Vec3 relativeVelocity = firstVelocity - secondVelocity;
        const Vec3 tangentVelocity = relativeVelocity -
            normal * dot(relativeVelocity, normal);
        const double slipSpeed = length(tangentVelocity);
        if (slipSpeed < 1.0e-10) {
            continue;
        }
        const Vec3 tangent = tangentVelocity / slipSpeed;
        double denominator = 0.0;
        for (std::size_t index = 0; index < 2; ++index) {
            denominator += cloth.particles[firstIndices[index]].inverseMass *
                firstWeights[index] * firstWeights[index];
            denominator += cloth.particles[secondIndices[index]].inverseMass *
                secondWeights[index] * secondWeights[index];
        }
        if (denominator <= 0.0) {
            continue;
        }
        const double frictionLimit =
            kClothSelfFriction * contact.normalImpulse;
        const double tangentialImpulse = std::min(
            slipSpeed / denominator,
            frictionLimit
        );
        if (tangentialImpulse <= 0.0) {
            continue;
        }
        const Vec3 impulseOnFirst = tangent * -tangentialImpulse;
        for (std::size_t index = 0; index < 2; ++index) {
            cloth.particles[firstIndices[index]].velocity += impulseOnFirst *
                (cloth.particles[firstIndices[index]].inverseMass *
                 firstWeights[index]);
            cloth.particles[secondIndices[index]].velocity -= impulseOnFirst *
                (cloth.particles[secondIndices[index]].inverseMass *
                 secondWeights[index]);
        }
        recordFrictionImpulse(
            metrics,
            tangentialImpulse,
            frictionLimit
        );
        ++metrics.clothSelfFrictionContacts;
    }
}

struct SelfFrictionProbeCase {
    double slipSpeed{};
    double momentum{};
    double energy{};
    double coneRatio{};
    std::uint64_t contacts{};
};

SelfFrictionProbeCase runSelfFrictionProbeCase(
    const double normalImpulse
) {
    ClothModel cloth;
    cloth.particles.resize(4);
    for (Particle& particle : cloth.particles) {
        particle.inverseMass = 1.0;
    }
    cloth.particles[0].velocity = {1.0, 0.0, 0.0};
    cloth.particles[1].velocity = {1.0, 0.0, 0.0};
    SelfContactImpulses contacts;
    SelfEdgeEdgeContactImpulse contact;
    contact.first = {0u, 1u};
    contact.second = {2u, 3u};
    contact.weightedNormalOnFirst = {0.0, 0.0, normalImpulse};
    contact.weightedFirst = {
        normalImpulse / 2.0,
        normalImpulse / 2.0,
    };
    contact.weightedSecond = contact.weightedFirst;
    contact.normalImpulse = normalImpulse;
    contacts.edgeEdge.emplace(0u, contact);
    Metrics metrics;
    applyClothSelfFriction(cloth, contacts, metrics);
    const Vec3 firstVelocity = (
        cloth.particles[0].velocity +
        cloth.particles[1].velocity
    ) / 2.0;
    const Vec3 secondVelocity = (
        cloth.particles[2].velocity +
        cloth.particles[3].velocity
    ) / 2.0;
    double momentum = 0.0;
    double energy = 0.0;
    for (const Particle& particle : cloth.particles) {
        momentum += particle.velocity.x;
        energy += 0.5 * lengthSquared(particle.velocity);
    }
    return {
        .slipSpeed = length(firstVelocity - secondVelocity),
        .momentum = momentum,
        .energy = energy,
        .coneRatio = metrics.maximumFrictionConeRatio,
        .contacts = metrics.clothSelfFrictionContacts,
    };
}

bool runSelfFrictionProbe() {
    const SelfFrictionProbeCase sticking = runSelfFrictionProbeCase(10.0);
    const SelfFrictionProbeCase sliding = runSelfFrictionProbeCase(1.0);
    const SelfFrictionProbeCase stickingReplay =
        runSelfFrictionProbeCase(10.0);
    const SelfFrictionProbeCase slidingReplay =
        runSelfFrictionProbeCase(1.0);
    const bool deterministic =
        sticking.slipSpeed == stickingReplay.slipSpeed &&
        sticking.momentum == stickingReplay.momentum &&
        sticking.energy == stickingReplay.energy &&
        sticking.coneRatio == stickingReplay.coneRatio &&
        sliding.slipSpeed == slidingReplay.slipSpeed &&
        sliding.momentum == slidingReplay.momentum &&
        sliding.energy == slidingReplay.energy &&
        sliding.coneRatio == slidingReplay.coneRatio;
    const bool pass = deterministic &&
        sticking.slipSpeed < 1.0e-12 &&
        std::abs(sticking.momentum - 2.0) < 1.0e-12 &&
        std::abs(sticking.energy - 0.5) < 1.0e-12 &&
        sliding.slipSpeed > 0.5 &&
        std::abs(sliding.momentum - 2.0) < 1.0e-12 &&
        std::abs(sliding.coneRatio - 1.0) < 1.0e-12 &&
        sticking.contacts == 1u && sliding.contacts == 1u;
    std::cout << std::fixed << std::setprecision(12)
              << "probe=cloth_self_coulomb_friction"
              << " sticking_slip=" << sticking.slipSpeed
              << " sticking_momentum=" << sticking.momentum
              << " sticking_energy=" << sticking.energy
              << " sliding_slip=" << sliding.slipSpeed
              << " sliding_momentum=" << sliding.momentum
              << " sliding_cone_ratio=" << sliding.coneRatio
              << " deterministic=" << std::boolalpha << deterministic
              << '\n'
              << "result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass;
}

double measurePrimitiveSelfPenetration(const ClothModel& cloth) {
    const double target = 2.0 * kClothRadius;
    const double broadphaseExpansion = 0.5 * target;
    double maximumPenetration = 0.0;
    std::vector<std::uint32_t> candidates;
    candidates.reserve(64u);
    const CollisionBVH edgeTree = makeCollisionBVH(
        static_cast<std::uint32_t>(cloth.yarnSegments.size()),
        [&](const std::uint32_t index) {
            return edgeBounds(
                cloth,
                cloth.yarnSegments[index],
                false,
                broadphaseExpansion
            );
        }
    );
    std::vector<CollisionSphere> edgeSpheres;
    edgeSpheres.reserve(cloth.yarnSegments.size());
    for (const Edge edge : cloth.yarnSegments) {
        edgeSpheres.push_back(edgeCollisionSphere(
            cloth,
            edge,
            false,
            broadphaseExpansion
        ));
    }
    for (std::uint32_t firstIndex = 0;
         firstIndex < cloth.yarnSegments.size();
         ++firstIndex) {
        const Edge first = cloth.yarnSegments[firstIndex];
        gatherCollisionCandidates(
            edgeTree,
            edgeBounds(cloth, first, false, broadphaseExpansion),
            candidates
        );
        for (const std::uint32_t secondIndex : candidates) {
            if (secondIndex <= firstIndex) {
                continue;
            }
            if (!overlapSpheres(
                edgeSpheres[firstIndex],
                edgeSpheres[secondIndex]
            )) {
                continue;
            }
            const Edge second = cloth.yarnSegments[secondIndex];
            if (edgePairLocal(cloth, first, second)) {
                continue;
            }
            const SegmentClosest closest = closestPointsOnSegments(
                cloth.particles[first.first].position,
                cloth.particles[first.second].position,
                cloth.particles[second.first].position,
                cloth.particles[second.second].position
            );
            maximumPenetration = std::max(
                maximumPenetration,
                target - length(
                    closest.secondPoint - closest.firstPoint
                )
            );
        }
    }
    return maximumPenetration;
}

double solveSelfCollision(
    ClothModel& cloth,
    std::uint64_t& contactCount
) {
    std::vector<Particle>& particles = cloth.particles;
    const double target = 2.0 * kClothRadius;
    const double inverseCell = 1.0 / target;
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> cells;
    cells.reserve(particles.size() * 2u);
    for (std::uint32_t index = 0; index < particles.size(); ++index) {
        const Vec3 position = particles[index].position;
        const int x = static_cast<int>(std::floor(position.x * inverseCell));
        const int y = static_cast<int>(std::floor(position.y * inverseCell));
        const int z = static_cast<int>(std::floor(position.z * inverseCell));
        cells[spatialKey(x, y, z)].push_back(index);
    }
    double maximumPenetration = 0.0;
    for (std::uint32_t firstIndex = 0; firstIndex < particles.size(); ++firstIndex) {
        const Vec3 position = particles[firstIndex].position;
        const int baseX = static_cast<int>(std::floor(position.x * inverseCell));
        const int baseY = static_cast<int>(std::floor(position.y * inverseCell));
        const int baseZ = static_cast<int>(std::floor(position.z * inverseCell));
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const auto found = cells.find(spatialKey(
                        baseX + dx,
                        baseY + dy,
                        baseZ + dz
                    ));
                    if (found == cells.end()) {
                        continue;
                    }
                    for (const std::uint32_t secondIndex : found->second) {
                        if (secondIndex <= firstIndex ||
                            localTopologyPair(cloth, firstIndex, secondIndex)) {
                            continue;
                        }
                        Particle& first = particles[firstIndex];
                        Particle& second = particles[secondIndex];
                        const Vec3 difference = second.position - first.position;
                        const double currentLength = length(difference);
                        if (currentLength >= target || currentLength < 1.0e-12) {
                            continue;
                        }
                        const double denominator =
                            first.inverseMass + second.inverseMass;
                        if (denominator <= 0.0) {
                            continue;
                        }
                        const Vec3 correction = difference *
                            ((target - currentLength) /
                             (currentLength * denominator));
                        first.position -= correction * first.inverseMass;
                        second.position += correction * second.inverseMass;
                        maximumPenetration = std::max(
                            maximumPenetration,
                            target - currentLength
                        );
                        ++contactCount;
                    }
                }
            }
        }
    }
    return maximumPenetration;
}

void updateMetrics(
    const ClothModel& cloth,
    const std::array<Ball, kFruitCount>& balls,
    Metrics& metrics
) {
    for (const DistanceConstraint& constraint : cloth.distances) {
        const double currentLength = length(
            cloth.particles[constraint.second].position -
            cloth.particles[constraint.first].position
        );
        const double signedStrain =
            currentLength / constraint.restLength - 1.0;
        const double strain = std::abs(signedStrain);
        const double extension = std::max(0.0, signedStrain);
        const double compression = std::max(0.0, -signedStrain);
        if (constraint.kind == DistanceKind::warp) {
            metrics.maximumWarpStrain = std::max(metrics.maximumWarpStrain, strain);
            if (extension > metrics.maximumWarpExtension) {
                metrics.maximumWarpExtension = extension;
                metrics.maximumWarpExtensionFirst = constraint.first;
                metrics.maximumWarpExtensionSecond = constraint.second;
            }
            metrics.maximumWarpCompression = std::max(
                metrics.maximumWarpCompression,
                compression
            );
        } else if (constraint.kind == DistanceKind::weft) {
            metrics.maximumWeftStrain = std::max(metrics.maximumWeftStrain, strain);
            metrics.maximumWeftExtension = std::max(
                metrics.maximumWeftExtension,
                extension
            );
            metrics.maximumWeftCompression = std::max(
                metrics.maximumWeftCompression,
                compression
            );
        } else if (constraint.kind == DistanceKind::bottom) {
            metrics.maximumBottomStrain = std::max(
                metrics.maximumBottomStrain,
                strain
            );
            metrics.maximumBottomExtension = std::max(
                metrics.maximumBottomExtension,
                extension
            );
            metrics.maximumBottomCompression = std::max(
                metrics.maximumBottomCompression,
                compression
            );
        }
    }
    for (const YarnBendConstraint& bend : cloth.bends) {
        const double chord = length(
            cloth.particles[bend.third].position -
            cloth.particles[bend.first].position
        );
        metrics.maximumBendError = std::max(
            metrics.maximumBendError,
            std::abs(chord - bend.restChord) / bend.restArc
        );
    }
    for (const KnotConstraint& knot : cloth.knots) {
        const Vec3 warp = normalized(
            cloth.particles[knot.warpSecond].position -
            cloth.particles[knot.warpFirst].position
        );
        const Vec3 weft = normalized(
            cloth.particles[knot.weftSecond].position -
            cloth.particles[knot.weftFirst].position
        );
        const double currentAngle = std::acos(std::clamp(
            dot(warp, weft), -1.0, 1.0
        ));
        const double restAngle = std::acos(std::clamp(
            knot.restCosine, -1.0, 1.0
        ));
        metrics.maximumKnotAngleError = std::max(
            metrics.maximumKnotAngleError,
            std::abs(currentAngle - restAngle)
        );
    }
    for (const Triangle triangle : cloth.renderTriangles) {
        const Vec3 first = cloth.particles[triangle.first].position;
        const Vec3 second = cloth.particles[triangle.second].position;
        const Vec3 third = cloth.particles[triangle.third].position;
        metrics.minimumTriangleArea = std::min(
            metrics.minimumTriangleArea,
            0.5 * length(cross(second - first, third - first))
        );
    }
    for (const Particle& particle : cloth.particles) {
        metrics.maximumSpeed = std::max(metrics.maximumSpeed, length(particle.velocity));
    }
    for (const Ball& ball : balls) {
        metrics.maximumSpeed = std::max(metrics.maximumSpeed, length(ball.velocity));
        metrics.maximumAngularSpeed = std::max(
            metrics.maximumAngularSpeed,
            length(ball.angularVelocity)
        );
    }
}

struct MouthFrame {
    Vec3 center{};
    Vec3 normal{};
    Vec3 tangent{};
    Vec3 bitangent{};
    std::array<Vec3, kAround> ring{};
    bool valid{};
};

MouthFrame makeMouthFrame(const ClothModel& cloth) {
    MouthFrame frame;
    for (std::uint32_t ring = 0u; ring < kAround; ++ring) {
        frame.ring[ring] =
            cloth.particles[nodeIndex(kLevels - 1u, ring)].position;
        frame.center += frame.ring[ring];
    }
    frame.center = frame.center / static_cast<double>(kAround);
    Vec3 areaNormal{};
    for (std::uint32_t ring = 0u; ring < kAround; ++ring) {
        areaNormal += cross(
            frame.ring[ring] - frame.center,
            frame.ring[(ring + 1u) % kAround] - frame.center
        );
    }
    if (lengthSquared(areaNormal) < 1.0e-16) {
        return frame;
    }
    frame.normal = normalized(areaNormal);
    const Vec3 interiorDirection =
        frame.center - cloth.particles[cloth.bottomCenter].position;
    if (dot(frame.normal, interiorDirection) < 0.0) {
        frame.normal = frame.normal * -1.0;
    }
    Vec3 tangent = frame.ring[0] - frame.center;
    tangent -= frame.normal * dot(tangent, frame.normal);
    if (lengthSquared(tangent) < 1.0e-16) {
        tangent = cross(
            frame.normal,
            std::abs(frame.normal.z) < 0.9
                ? Vec3{0.0, 0.0, 1.0}
                : Vec3{1.0, 0.0, 0.0}
        );
    }
    frame.tangent = normalized(tangent);
    frame.bitangent = normalized(cross(frame.normal, frame.tangent));
    frame.valid = true;
    return frame;
}

bool projectedInsideMouth(
    const MouthFrame& frame,
    const Vec3 point,
    const double edgeClearance = 0.0
) {
    const Vec3 relativePoint = point - frame.center;
    const double pointX = dot(relativePoint, frame.tangent);
    const double pointY = dot(relativePoint, frame.bitangent);
    bool inside = false;
    double minimumEdgeDistanceSquared =
        std::numeric_limits<double>::infinity();
    for (std::uint32_t first = 0u, second = kAround - 1u;
         first < kAround;
         second = first++) {
        const Vec3 firstRelative = frame.ring[first] - frame.center;
        const Vec3 secondRelative = frame.ring[second] - frame.center;
        const double firstX = dot(firstRelative, frame.tangent);
        const double firstY = dot(firstRelative, frame.bitangent);
        const double secondX = dot(secondRelative, frame.tangent);
        const double secondY = dot(secondRelative, frame.bitangent);
        const double edgeX = secondX - firstX;
        const double edgeY = secondY - firstY;
        const double edgeLengthSquared = edgeX * edgeX + edgeY * edgeY;
        const double fraction = edgeLengthSquared > 1.0e-20
            ? std::clamp(
                ((pointX - firstX) * edgeX +
                 (pointY - firstY) * edgeY) / edgeLengthSquared,
                0.0,
                1.0
            )
            : 0.0;
        const double separationX = pointX - (firstX + fraction * edgeX);
        const double separationY = pointY - (firstY + fraction * edgeY);
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
}

double minimumMouthRimDistance(
    const MouthFrame& frame,
    const Vec3 point
) {
    double minimumDistanceSquared =
        std::numeric_limits<double>::infinity();
    for (std::uint32_t first = 0u; first < kAround; ++first) {
        const Vec3 start = frame.ring[first];
        const Vec3 edge =
            frame.ring[(first + 1u) % kAround] - start;
        const double edgeLengthSquared = lengthSquared(edge);
        const double fraction = edgeLengthSquared > 1.0e-20
            ? std::clamp(
                dot(point - start, edge) / edgeLengthSquared,
                0.0,
                1.0
            )
            : 0.0;
        minimumDistanceSquared = std::min(
            minimumDistanceSquared,
            lengthSquared(point - (start + edge * fraction))
        );
    }
    return std::sqrt(minimumDistanceSquared);
}

void updateReleasedFruit(
    const ClothModel& cloth,
    const std::array<Ball, kFruitCount>& balls,
    Metrics& metrics
) {
    const MouthFrame mouth = makeMouthFrame(cloth);
    if (!mouth.valid) {
        return;
    }
    for (std::size_t index = 0u; index < balls.size(); ++index) {
        const Ball& ball = balls[index];
        const double requiredClearance = ball.radius + kClothRadius;
        const bool insideMouthProjection = projectedInsideMouth(
            mouth,
            ball.position
        );
        const bool nearMouthProjection = projectedInsideMouth(
            mouth,
            ball.position,
            requiredClearance
        );
        const double outwardDistance = dot(
            ball.position - mouth.center,
            mouth.normal
        );
        const std::uint32_t mask = 1u << index;
        if (insideMouthProjection && outwardDistance > 0.0) {
            metrics.mouthCandidateMask |= mask;
        }
        const double axialClearance =
            outwardDistance - requiredClearance;
        const double rimClearance =
            minimumMouthRimDistance(mouth, ball.position) -
            requiredClearance;
        const bool fullyClearThroughCap = nearMouthProjection &&
            axialClearance > kMouthReleaseHysteresis;
        const bool fullyClearAroundRim =
            (metrics.mouthCandidateMask & mask) != 0u &&
            !insideMouthProjection &&
            outwardDistance > 0.0 &&
            rimClearance > kMouthReleaseHysteresis;
        const double clearance = fullyClearThroughCap
            ? axialClearance
            : (fullyClearAroundRim ? rimClearance : axialClearance);
        metrics.maximumMouthClearanceByFruit[index] = std::max(
            metrics.maximumMouthClearanceByFruit[index],
            clearance
        );
        if (fullyClearThroughCap || fullyClearAroundRim) {
            metrics.releasedMask |= mask;
        }
    }
}

struct MouthReleaseProbeResult {
    std::uint32_t insideMask{};
    std::uint32_t releasedMask{};
    std::uint32_t outsideProjectionMask{};
    std::uint32_t grazingMask{};
    std::uint32_t edgeClearanceMask{};
    std::uint32_t edgeExitMask{};
    std::uint32_t rotatedMask{};
    double clearance{};
    double rotatedClearance{};
};

MouthReleaseProbeResult runMouthReleaseProbeOnce() {
    const auto circularMouth = [](const bool rotated) {
        ClothModel cloth = makeCloth(Scenario::grounded);
        cloth.particles[cloth.bottomCenter].position = {};
        for (std::uint32_t ring = 0u; ring < kAround; ++ring) {
            const double angle = 2.0 * std::numbers::pi *
                static_cast<double>(ring) / static_cast<double>(kAround);
            cloth.particles[nodeIndex(kLevels - 1u, ring)].position = rotated
                ? Vec3{1.0, std::cos(angle), std::sin(angle)}
                : Vec3{std::cos(angle), std::sin(angle), 1.0};
        }
        return cloth;
    };

    std::array<Ball, kFruitCount> balls{};
    balls[0].radius = 0.10;
    balls[0].position = {0.0, 0.0, 0.50};
    ClothModel upright = circularMouth(false);
    Metrics uprightMetrics;
    updateReleasedFruit(upright, balls, uprightMetrics);
    const std::uint32_t insideMask = uprightMetrics.releasedMask;
    balls[0].position = {0.0, 0.0, 1.130};
    updateReleasedFruit(upright, balls, uprightMetrics);

    Metrics outsideMetrics;
    balls[0].position = {1.50, 0.0, 1.20};
    updateReleasedFruit(upright, balls, outsideMetrics);

    Metrics grazingMetrics;
    balls[0].position = {0.0, 0.0, 1.114};
    updateReleasedFruit(upright, balls, grazingMetrics);

    Metrics edgeClearanceMetrics;
    balls[0].position = {1.05, 0.0, 1.130};
    updateReleasedFruit(upright, balls, edgeClearanceMetrics);

    Metrics edgeExitMetrics;
    balls[0].position = {0.0, 0.0, 1.05};
    updateReleasedFruit(upright, balls, edgeExitMetrics);
    balls[0].position = {1.30, 0.0, 1.02};
    updateReleasedFruit(upright, balls, edgeExitMetrics);

    ClothModel rotated = circularMouth(true);
    Metrics rotatedMetrics;
    balls[0].position = {1.130, 0.0, 0.0};
    updateReleasedFruit(rotated, balls, rotatedMetrics);
    return {
        .insideMask = insideMask,
        .releasedMask = uprightMetrics.releasedMask,
        .outsideProjectionMask = outsideMetrics.releasedMask,
        .grazingMask = grazingMetrics.releasedMask,
        .edgeClearanceMask = edgeClearanceMetrics.releasedMask,
        .edgeExitMask = edgeExitMetrics.releasedMask,
        .rotatedMask = rotatedMetrics.releasedMask,
        .clearance = uprightMetrics.maximumMouthClearanceByFruit[0],
        .rotatedClearance =
            rotatedMetrics.maximumMouthClearanceByFruit[0],
    };
}

bool runMouthReleaseProbe() {
    const MouthReleaseProbeResult first = runMouthReleaseProbeOnce();
    const MouthReleaseProbeResult replay = runMouthReleaseProbeOnce();
    const bool deterministic =
        first.insideMask == replay.insideMask &&
        first.releasedMask == replay.releasedMask &&
        first.outsideProjectionMask == replay.outsideProjectionMask &&
        first.grazingMask == replay.grazingMask &&
        first.edgeClearanceMask == replay.edgeClearanceMask &&
        first.edgeExitMask == replay.edgeExitMask &&
        first.rotatedMask == replay.rotatedMask &&
        first.clearance == replay.clearance &&
        first.rotatedClearance == replay.rotatedClearance;
    const bool pass = deterministic &&
        first.insideMask == 0u &&
        first.releasedMask == 1u &&
        first.outsideProjectionMask == 0u &&
        first.grazingMask == 0u &&
        first.edgeClearanceMask == 1u &&
        first.edgeExitMask == 1u &&
        first.rotatedMask == 1u &&
        std::abs(first.clearance - 0.026) < 1.0e-12 &&
        std::abs(first.rotatedClearance - 0.026) < 1.0e-12;
    std::cout << std::fixed << std::setprecision(12)
              << "probe=topology_aware_mouth_release"
              << " inside_mask=" << first.insideMask
              << " released_mask=" << first.releasedMask
              << " outside_projection_mask="
              << first.outsideProjectionMask
              << " grazing_mask=" << first.grazingMask
              << " edge_clearance_mask=" << first.edgeClearanceMask
              << " edge_exit_mask=" << first.edgeExitMask
              << " rotated_mask=" << first.rotatedMask
              << " clearance=" << first.clearance
              << " rotated_clearance=" << first.rotatedClearance
              << " deterministic=" << std::boolalpha << deterministic
              << '\n'
              << "result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass;
}

SimulationResult simulate(
    const std::uint32_t steps,
    const double frameTimestep,
    const std::uint32_t substeps,
    const std::uint32_t iterations,
    const Scenario scenario,
    const std::vector<std::uint32_t>* captureSteps = nullptr,
    std::vector<SimulationResult>* captures = nullptr,
    const numi::GripTrajectory* gripTrajectory = nullptr
) {
    if ((scenario == Scenario::recorded) != (gripTrajectory != nullptr)) {
        throw std::runtime_error(
            "recorded scenario requires exactly one grip trajectory"
        );
    }
    if (gripTrajectory != nullptr && !numi::gripTrajectoryCovers(
        *gripTrajectory,
        static_cast<double>(steps) * frameTimestep
    )) {
        throw std::runtime_error(
            "grip trajectory does not cover the requested simulation"
        );
    }
    SimulationResult result;
    result.scenario = scenario;
    result.cloth = makeCloth(scenario);
    result.balls = makeBalls(scenario);
    const double timestep = frameTimestep / static_cast<double>(substeps);
    const Vec3 gravity{0.0, 0.0, -9.81};
    std::vector<Vec3> predictedClothVelocities(
        result.cloth.particles.size()
    );
    std::vector<double> clothGroundNormalImpulses(
        result.cloth.particles.size()
    );
    const auto capture = [&](const std::uint32_t completedSteps) {
        if (captureSteps != nullptr && captures != nullptr &&
            std::find(
                captureSteps->begin(),
                captureSteps->end(),
                completedSteps
            ) != captureSteps->end()) {
            captures->push_back(result);
        }
    };
    capture(0u);

    for (std::uint32_t step = 0; step < steps; ++step) {
        double publishedPrimitiveResidual =
            std::numeric_limits<double>::infinity();
        for (std::uint32_t substep = 0; substep < substeps; ++substep) {
            std::fill(
                clothGroundNormalImpulses.begin(),
                clothGroundNormalImpulses.end(),
                0.0
            );
            std::array<BallPairContactImpulse, kBallPairCount> pairContacts{};
            std::vector<BallYarnContactImpulse> yarnContacts(
                result.balls.size() * result.cloth.yarnSegments.size()
            );
            std::array<double, kFruitCount> groundNormalImpulses{};
            SelfContactImpulses selfContactImpulses;
            selfContactImpulses.edgeEdge.reserve(256u);
            if (scenario != Scenario::grounded) {
                const double time = (
                    static_cast<double>(step * substeps + substep + 1u) *
                    timestep
                );
                updateGrip(
                    result.cloth,
                    result.metrics,
                    scenario,
                    time,
                    gripTrajectory
                );
            }
            for (Particle& particle : result.cloth.particles) {
                particle.previous = particle.position;
                if (particle.inverseMass == 0.0) {
                    particle.position = particle.rest;
                    particle.velocity =
                        (particle.position - particle.previous) / timestep;
                    continue;
                }
                particle.velocity += gravity * timestep;
            }
            applyYarnAerodynamics(
                result.cloth,
                {},
                timestep,
                &result.metrics
            );
            for (std::size_t index = 0u;
                 index < result.cloth.particles.size();
                 ++index) {
                Particle& particle = result.cloth.particles[index];
                predictedClothVelocities[index] = particle.velocity;
                if (particle.inverseMass == 0.0) {
                    continue;
                }
                particle.position += particle.velocity * timestep;
            }
            for (Ball& ball : result.balls) {
                ball.previous = ball.position;
                ball.velocity += gravity * timestep;
            }
            applyFruitAerodynamics(
                result.balls,
                {},
                timestep,
                &result.metrics
            );
            for (Ball& ball : result.balls) {
                ball.position += ball.velocity * timestep;
            }
            if (scenario != Scenario::spin) {
                result.metrics.maximumSweptGroundAdvance = std::max(
                    result.metrics.maximumSweptGroundAdvance,
                    sweepGroundPrediction(
                        result.cloth.particles,
                        result.balls,
                        timestep,
                        groundNormalImpulses
                    )
                );
            }
            for (DistanceConstraint& constraint : result.cloth.distances) {
                constraint.lambda = 0.0;
            }
            for (YarnBendConstraint& constraint : result.cloth.bends) {
                constraint.lambda = 0.0;
            }
            for (KnotConstraint& constraint : result.cloth.knots) {
                constraint.lambda = 0.0;
            }
            for (GripConstraint& constraint : result.cloth.grips) {
                constraint.lambda = {};
            }

            result.metrics.maximumSweptSelfAdvance = std::max(
                result.metrics.maximumSweptSelfAdvance,
                accumulateSeconds(
                    result.metrics.primitiveSelfSolveSeconds,
                    [&] {
                        return solvePrimitiveSelfCollision(
                            result.cloth,
                            true,
                            result.metrics,
                            timestep,
                            &selfContactImpulses
                        );
                    }
                )
            );

            for (std::size_t ballIndex = 0;
                 ballIndex < result.balls.size();
                 ++ballIndex) {
                for (std::size_t segmentIndex = 0;
                     segmentIndex < result.cloth.yarnSegments.size();
                     ++segmentIndex) {
                    result.metrics.maximumSweptBallAdvance = std::max(
                        result.metrics.maximumSweptBallAdvance,
                        solveSweptBallYarn(
                            result.cloth.particles,
                            result.cloth.yarnSegments[segmentIndex],
                            result.balls[ballIndex],
                            timestep,
                            scenario != Scenario::spin,
                            yarnContacts[
                                ballIndex * result.cloth.yarnSegments.size() +
                                segmentIndex
                            ],
                            result.metrics.sweptBallYarnContacts
                        )
                    );
                }
            }

            for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
                for (DistanceConstraint& constraint : result.cloth.distances) {
                    solveDistance(result.cloth.particles, constraint, timestep);
                }
                for (KnotConstraint& constraint : result.cloth.knots) {
                    solveKnot(result.cloth.particles, constraint, timestep);
                }
                if (iteration % 2u == 0u) {
                    for (YarnBendConstraint& constraint : result.cloth.bends) {
                        solveBend(
                            result.cloth.particles,
                            constraint,
                            timestep,
                            scenario != Scenario::spin
                        );
                    }
                }
                if (result.cloth.gripActive) {
                    for (GripConstraint& constraint : result.cloth.grips) {
                        solveGrip(
                            result.cloth.particles,
                            constraint,
                            result.cloth.gripTarget +
                                rotateVector(
                                    result.cloth.gripOrientation,
                                    constraint.targetOffset
                                ),
                            timestep
                        );
                    }
                }
                std::size_t pairIndex = 0u;
                for (std::size_t first = 0; first < result.balls.size(); ++first) {
                    for (std::size_t second = first + 1u;
                         second < result.balls.size();
                         ++second, ++pairIndex) {
                        solveBallPair(
                            result.balls[first],
                            result.balls[second],
                            timestep,
                            pairContacts[pairIndex]
                        );
                    }
                }
                accumulateSeconds(
                    result.metrics.ballClothSolveSeconds,
                    [&] {
                        solveBallYarnContactSweep(
                            result.cloth,
                            result.balls,
                            timestep,
                            scenario != Scenario::spin,
                            yarnContacts,
                            result.metrics
                        );
                        return 0;
                    }
                );
                result.metrics.maximumSelfPenetration = std::max(
                    result.metrics.maximumSelfPenetration,
                    accumulateSeconds(
                        result.metrics.pointSelfSolveSeconds,
                        [&] {
                            return solveSelfCollision(
                                result.cloth,
                                result.metrics.selfContacts
                            );
                        }
                    )
                );
                if (scenario != Scenario::spin) {
                    result.metrics.maximumGroundPenetration = std::max(
                        result.metrics.maximumGroundPenetration,
                        solveGround(
                            result.cloth.particles,
                            result.balls,
                            timestep,
                            groundNormalImpulses,
                            result.metrics.maximumClothGroundCorrection,
                            result.metrics.maximumBallGroundCorrection
                        )
                    );
                }
            }
            constexpr std::uint32_t maximumContactReconciliationPasses = 8u;
            std::uint32_t reconciliationPasses = 0u;
            for (std::uint32_t pass = 0;
                 pass < maximumContactReconciliationPasses;
                 ++pass) {
                std::size_t pairIndex = 0u;
                for (std::size_t first = 0;
                     first < result.balls.size();
                     ++first) {
                    for (std::size_t second = first + 1u;
                         second < result.balls.size();
                         ++second, ++pairIndex) {
                        solveBallPair(
                            result.balls[first],
                            result.balls[second],
                            timestep,
                            pairContacts[pairIndex]
                        );
                    }
                }
                accumulateSeconds(
                    result.metrics.ballClothSolveSeconds,
                    [&] {
                        solveBallYarnContactSweep(
                            result.cloth,
                            result.balls,
                            timestep,
                            scenario != Scenario::spin,
                            yarnContacts,
                            result.metrics
                        );
                        return 0;
                    }
                );
                for (std::uint32_t strainSweep = 0u;
                     strainSweep < 3u;
                     ++strainSweep) {
                    result.metrics.maximumStrainLimitCorrection = std::max(
                        result.metrics.maximumStrainLimitCorrection,
                        accumulateSeconds(
                            result.metrics.strainLimitSolveSeconds,
                            [&] {
                                return solveStrainLimits(
                                    result.cloth.particles,
                                    result.cloth.distances
                                );
                            }
                        )
                    );
                }
                if (scenario != Scenario::spin) {
                    result.metrics.maximumGroundPenetration = std::max(
                        result.metrics.maximumGroundPenetration,
                        solveGround(
                            result.cloth.particles,
                            result.balls,
                            timestep,
                            groundNormalImpulses,
                            result.metrics.maximumClothGroundCorrection,
                            result.metrics.maximumBallGroundCorrection
                        )
                    );
                }
                reconciliationPasses = pass + 1u;
                const double ballResidual = measureBallYarnPenetration(
                    result.cloth,
                    result.balls
                );
                const double groundResidual = scenario == Scenario::spin
                    ? 0.0
                    : measureGroundPenetration(
                        result.cloth.particles,
                        result.balls
                    );
                if (ballResidual < 1.0e-6 &&
                    groundResidual < 1.0e-9 &&
                    measureStrainLimitViolation(result.cloth) < 1.0e-6) {
                    break;
                }
            }
            result.metrics.maximumContactReconciliationPasses = std::max(
                result.metrics.maximumContactReconciliationPasses,
                reconciliationPasses
            );
            const auto solveEndpointPrimitive = [&] {
                result.metrics.maximumSelfPenetration = std::max(
                    result.metrics.maximumSelfPenetration,
                    accumulateSeconds(
                        result.metrics.primitiveSelfSolveSeconds,
                        [&] {
                            return solvePrimitiveSelfCollision(
                                result.cloth,
                                false,
                                result.metrics,
                                timestep,
                                &selfContactImpulses
                            );
                        }
                    )
                );
                for (std::uint32_t strainSweep = 0u;
                     strainSweep < 3u;
                     ++strainSweep) {
                    result.metrics.maximumStrainLimitCorrection = std::max(
                        result.metrics.maximumStrainLimitCorrection,
                        accumulateSeconds(
                            result.metrics.strainLimitSolveSeconds,
                            [&] {
                                return solveStrainLimits(
                                    result.cloth.particles,
                                    result.cloth.distances
                                );
                            }
                        )
                    );
                }
            };
            const auto solveFinalContacts = [&] {
                constexpr std::uint32_t finalContactPasses = 2u;
                for (std::uint32_t pass = 0u;
                     pass < finalContactPasses;
                     ++pass) {
                    std::size_t pairIndex = 0u;
                    for (std::size_t first = 0u;
                         first < result.balls.size();
                         ++first) {
                        for (std::size_t second = first + 1u;
                             second < result.balls.size();
                             ++second, ++pairIndex) {
                            solveBallPair(
                                result.balls[first],
                                result.balls[second],
                                timestep,
                                pairContacts[pairIndex]
                            );
                        }
                    }
                    accumulateSeconds(
                        result.metrics.ballClothSolveSeconds,
                        [&] {
                            solveBallYarnContactSweep(
                                result.cloth,
                                result.balls,
                                timestep,
                                scenario != Scenario::spin,
                                yarnContacts,
                                result.metrics
                            );
                            return 0;
                        }
                    );
                    if (scenario != Scenario::spin) {
                        result.metrics.maximumGroundPenetration = std::max(
                            result.metrics.maximumGroundPenetration,
                            solveGround(
                                result.cloth.particles,
                                result.balls,
                                timestep,
                                groundNormalImpulses,
                                result.metrics.maximumClothGroundCorrection,
                                result.metrics.maximumBallGroundCorrection
                            )
                        );
                    }
                }
            };
            solveEndpointPrimitive();
            solveFinalContacts();
            if (substep + 1u == substeps) {
                constexpr std::uint32_t maximumCertificatePasses = 128u;
                std::uint32_t certificatePasses = 0u;
                for (std::uint32_t pass = 0u;
                     pass < maximumCertificatePasses;
                     ++pass) {
                    solveEndpointPrimitive();
                    solveFinalContacts();
                    certificatePasses = pass + 1u;
                    publishedPrimitiveResidual = accumulateSeconds(
                        result.metrics.primitiveCertificateSeconds,
                        [&] {
                            return measurePrimitiveSelfPenetration(
                                result.cloth
                            );
                        }
                    );
                    const double publishedBallResidual =
                        measureBallYarnPenetration(
                            result.cloth,
                            result.balls
                        );
                    const double publishedGroundResidual =
                        scenario == Scenario::spin
                            ? 0.0
                            : measureGroundPenetration(
                                result.cloth.particles,
                                result.balls
                            );
                    const double publishedStrainResidual =
                        measureStrainLimitViolation(result.cloth);
                    if (publishedPrimitiveResidual < 1.0e-6 &&
                        publishedBallResidual < 1.0e-6 &&
                        publishedGroundResidual < 1.0e-9 &&
                        publishedStrainResidual < 1.0e-6) {
                        break;
                    }
                }
                result.metrics.maximumPrimitiveCertificatePasses = std::max(
                    result.metrics.maximumPrimitiveCertificatePasses,
                    certificatePasses
                );
            }
            for (std::size_t index = 0u;
                 index < result.cloth.particles.size();
                 ++index) {
                Particle& particle = result.cloth.particles[index];
                if (particle.inverseMass == 0.0) {
                    particle.position = particle.rest;
                    particle.velocity =
                        (particle.position - particle.previous) / timestep;
                } else {
                    particle.velocity =
                        (particle.position - particle.previous) / timestep;
                    if (scenario != Scenario::spin &&
                        particle.position.z <= kClothRadius + 1.0e-6) {
                        const double normalVelocityChange =
                            particle.velocity.z -
                            predictedClothVelocities[index].z;
                        clothGroundNormalImpulses[index] = std::max(
                            0.0,
                            normalVelocityChange / particle.inverseMass
                        );
                    }
                }
            }
            for (Ball& ball : result.balls) {
                ball.velocity = (ball.position - ball.previous) / timestep;
            }
            applyClothGroundFriction(
                result.cloth.particles,
                clothGroundNormalImpulses,
                result.metrics
            );
            applyClothSelfFriction(
                result.cloth,
                selfContactImpulses,
                result.metrics
            );
            applyBallYarnFriction(
                result.cloth,
                result.balls,
                yarnContacts,
                result.metrics,
                scenario != Scenario::spin
            );
            applyBallPairFriction(result.balls, pairContacts, result.metrics);
            applyBallGroundFriction(
                result.balls,
                groundNormalImpulses,
                timestep,
                result.metrics,
                kFruitRollingResistanceCoefficient
            );
            if (scenario != Scenario::grounded) {
                updateReleasedFruit(
                    result.cloth,
                    result.balls,
                    result.metrics
                );
            }
            for (Ball& ball : result.balls) {
                integrateOrientation(ball, timestep);
            }
            double gripForce = 0.0;
            double gripImpulse = 0.0;
            for (const GripConstraint& constraint : result.cloth.grips) {
                gripForce += length(constraint.lambda) /
                    (timestep * timestep);
                gripImpulse += length(constraint.lambda) / timestep;
            }
            result.metrics.maximumGripForce = std::max(
                result.metrics.maximumGripForce,
                gripForce
            );
            result.metrics.maximumGripImpulse = std::max(
                result.metrics.maximumGripImpulse,
                gripImpulse
            );
        }
        result.metrics.maximumPublishedPrimitiveSelfPenetration = std::max(
            result.metrics.maximumPublishedPrimitiveSelfPenetration,
            publishedPrimitiveResidual
        );
        result.metrics.maximumPublishedBallPenetration = std::max(
            result.metrics.maximumPublishedBallPenetration,
            measureBallYarnPenetration(
                result.cloth,
                result.balls
            )
        );
        result.metrics.maximumPublishedGroundPenetration = std::max(
            result.metrics.maximumPublishedGroundPenetration,
            scenario == Scenario::spin ? 0.0 : measureGroundPenetration(
                result.cloth.particles,
                result.balls
            )
        );
        result.metrics.maximumPublishedStrainLimitViolation = std::max(
            result.metrics.maximumPublishedStrainLimitViolation,
            measureStrainLimitViolation(result.cloth)
        );
        updateMetrics(result.cloth, result.balls, result.metrics);
        capture(step + 1u);
    }

    for (std::size_t ballIndex = 0; ballIndex < result.balls.size(); ++ballIndex) {
        const Ball& ball = result.balls[ballIndex];
        const double radial = std::hypot(ball.position.x, ball.position.y);
        const bool outsideScenarioBounds = scenario == Scenario::grounded
            ? ball.position.z > 0.90 || ball.position.z < 0.0 || radial > 0.75
            : ball.position.z > 5.0 || ball.position.z < -5.0 || radial > 5.0;
        if (!finite(ball.position) || outsideScenarioBounds) {
            result.metrics.escapedMask |= 1u << ballIndex;
        }
        if (scenario == Scenario::grounded &&
            (ball.position.z > 0.45 || radial > 0.48)) {
            result.metrics.spilledMask |= 1u << ballIndex;
        }
    }
    result.metrics.finalPrimitiveSelfPenetration = accumulateSeconds(
        result.metrics.primitiveCertificateSeconds,
        [&] { return measurePrimitiveSelfPenetration(result.cloth); }
    );
    result.metrics.finalStrainLimitViolation =
        measureStrainLimitViolation(result.cloth);
    return result;
}

std::uint64_t hashResult(const SimulationResult& result) {
    std::uint64_t hash = 1469598103934665603ull;
    const auto append = [&](const double value) {
        std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
        for (std::size_t byte = 0; byte < sizeof(bits); ++byte) {
            hash ^= static_cast<std::uint8_t>(bits >> (8u * byte));
            hash *= 1099511628211ull;
        }
    };
    for (const Particle& particle : result.cloth.particles) {
        append(particle.position.x);
        append(particle.position.y);
        append(particle.position.z);
        append(particle.velocity.x);
        append(particle.velocity.y);
        append(particle.velocity.z);
    }
    for (const Ball& ball : result.balls) {
        append(ball.position.x);
        append(ball.position.y);
        append(ball.position.z);
        append(ball.velocity.x);
        append(ball.velocity.y);
        append(ball.velocity.z);
        append(ball.angularVelocity.x);
        append(ball.angularVelocity.y);
        append(ball.angularVelocity.z);
        append(ball.orientation.w);
        append(ball.orientation.x);
        append(ball.orientation.y);
        append(ball.orientation.z);
    }
    append(result.cloth.gripTarget.x);
    append(result.cloth.gripTarget.y);
    append(result.cloth.gripTarget.z);
    append(result.cloth.gripActive ? 1.0 : 0.0);
    if (result.cloth.gripAttachmentGeneration > 1u) {
        append(static_cast<double>(result.cloth.gripAttachmentGeneration));
        for (const GripConstraint& grip : result.cloth.grips) {
            if (result.metrics.gripPatchSelectionCount > 0u) {
                append(static_cast<double>(grip.particle));
            }
            append(grip.targetOffset.x);
            append(grip.targetOffset.y);
            append(grip.targetOffset.z);
        }
    }
    return hash;
}

void dumpOBJ(const std::string& path, const SimulationResult& result) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open OBJ output: " + path);
    }
    output << std::setprecision(9);
    output << "# Numi Solver explicit-yarn cloth bag reference\n";
    output << "# vertices " << result.cloth.particles.size()
           << " render_triangles " << result.cloth.renderTriangles.size()
           << '\n';
    for (const Particle& particle : result.cloth.particles) {
        output << "v " << particle.position.x << ' '
               << particle.position.y << ' ' << particle.position.z << '\n';
    }
    for (const Triangle triangle : result.cloth.renderTriangles) {
        output << "f " << triangle.first + 1u << ' '
               << triangle.second + 1u << ' '
               << triangle.third + 1u << '\n';
    }
    if (result.scenario != Scenario::grounded) {
        output << "# grip center " << result.cloth.gripTarget.x << ' '
               << result.cloth.gripTarget.y << ' '
               << result.cloth.gripTarget.z << " active "
               << (result.cloth.gripActive ? 1 : 0)
               << " orientation "
               << result.cloth.gripOrientation.w << ' '
               << result.cloth.gripOrientation.x << ' '
               << result.cloth.gripOrientation.y << ' '
               << result.cloth.gripOrientation.z
               << " patch_center "
               << result.cloth.gripPatchCenterRing << '\n';
    }
    for (std::size_t index = 0; index < result.balls.size(); ++index) {
        const Ball& ball = result.balls[index];
        output << "# ball " << index << " center "
               << ball.position.x << ' ' << ball.position.y << ' '
               << ball.position.z << " radius " << ball.radius
               << " appearance " << ball.appearance
               << " orientation " << ball.orientation.w << ' '
               << ball.orientation.x << ' ' << ball.orientation.y << ' '
               << ball.orientation.z
               << " angular_velocity " << ball.angularVelocity.x << ' '
               << ball.angularVelocity.y << ' '
               << ball.angularVelocity.z << '\n';
    }
}

bool gripParticlesUnique(const ClothModel& cloth) {
    for (std::size_t first = 0u; first < cloth.grips.size(); ++first) {
        for (std::size_t second = first + 1u;
             second < cloth.grips.size();
             ++second) {
            if (cloth.grips[first].particle == cloth.grips[second].particle) {
                return false;
            }
        }
    }
    return true;
}

bool gripPatchTopologyExact(const ClothModel& cloth) {
    constexpr std::uint32_t patchWidth = 5u;
    if (cloth.grips.size() != 2u * patchWidth) {
        return false;
    }
    for (std::uint32_t row = 0u; row < 2u; ++row) {
        for (std::uint32_t slot = 0u; slot < patchWidth; ++slot) {
            const std::uint32_t ring = (
                cloth.gripPatchCenterRing + kAround + slot - 2u
            ) % kAround;
            if (cloth.grips[row * patchWidth + slot].particle !=
                nodeIndex(kLevels - 2u + row, ring)) {
                return false;
            }
        }
    }
    return true;
}

bool acceptable(const SimulationResult& result, const bool deterministic) {
    bool allFinite = finite(result.cloth.gripOrientation) &&
        std::abs(
            quaternionLength(result.cloth.gripOrientation) - 1.0
        ) < 1.0e-9;
    for (const Particle& particle : result.cloth.particles) {
        allFinite = allFinite && finite(particle.position) && finite(particle.velocity);
    }
    for (const Ball& ball : result.balls) {
        allFinite = allFinite && finite(ball.position) && finite(ball.velocity) &&
            finite(ball.angularVelocity) && finite(ball.orientation) &&
            std::abs(quaternionLength(ball.orientation) - 1.0) < 1.0e-9;
    }
    const bool groundValid = result.scenario == Scenario::spin ||
        result.metrics.maximumGroundPenetration < kClothRadius + 1.0e-6;
    const bool pickupOutcome = result.scenario != Scenario::pickup ||
        std::popcount(result.metrics.releasedMask) >= 2;
    double clothMass = 0.0;
    for (const Particle& particle : result.cloth.particles) {
        clothMass += particle.mass;
    }
    const bool spatialPatchValid =
        result.metrics.gripPatchSelectionCount == 0u ||
        (result.metrics.gripPatchSelectionCount ==
             result.metrics.regrabCount &&
         gripParticlesUnique(result.cloth) &&
         gripPatchTopologyExact(result.cloth));
    return allFinite && deterministic && pickupOutcome &&
        spatialPatchValid &&
        result.metrics.escapedMask == 0u &&
        result.metrics.spilledMask == 0u && groundValid &&
        std::abs(clothMass - kClothMass) < 1.0e-12 &&
        result.metrics.maximumRegrabCaptureDistance <=
            kGripCaptureRadius + 1.0e-12 &&
        result.metrics.maximumRegrabCaptureError <= 1.0e-9 &&
        result.metrics.minimumTriangleArea > 1.0e-8 &&
        result.metrics.maximumWarpExtension < 0.30 &&
        result.metrics.maximumWarpCompression < 0.60 &&
        result.metrics.maximumWeftExtension < 0.30 &&
        result.metrics.maximumWeftCompression < 0.60 &&
        result.metrics.maximumBottomExtension < 0.30 &&
        result.metrics.maximumBottomCompression < 0.60 &&
        result.metrics.maximumKnotAngleError < 0.80 &&
        result.metrics.maximumBallPenetration < 0.010 &&
        result.metrics.maximumPublishedBallPenetration < 2.0e-6 &&
        result.metrics.maximumPublishedGroundPenetration < 2.0e-6 &&
        result.metrics.maximumPublishedPrimitiveSelfPenetration < 2.0e-6 &&
        result.metrics.maximumPublishedStrainLimitViolation < 2.0e-6 &&
        result.metrics.maximumSelfPenetration < 2.0 * kClothRadius &&
        result.metrics.finalPrimitiveSelfPenetration < 2.0e-6 &&
        result.metrics.finalStrainLimitViolation < 2.0e-6 &&
        result.metrics.maximumSpeed < 30.0 &&
        result.metrics.maximumAngularSpeed < 200.0 &&
        result.metrics.maximumGripForce < 500.0 &&
        result.metrics.maximumFrictionConeRatio <= 1.0 + 1.0e-12 &&
        result.metrics.maximumRollingResistanceRatio <= 1.0 + 1.0e-12;
}

}  // namespace

int main(int argc, char** argv) try {
    std::uint32_t steps = 120u;
    std::uint32_t substeps = 24u;
    std::uint32_t iterations = 32u;
    std::uint32_t replays = 2u;
    double timestep = 1.0 / 120.0;
    std::string dumpPath;
    std::string framePrefix;
    std::string materialPath;
    std::string gripTrajectoryPath;
    std::uint32_t frameStride = 0u;
    bool rollingProbe = false;
    bool selfCCDProbe = false;
    bool strainProbe = false;
    bool selfFrictionProbe = false;
    bool deformableResponseProbe = false;
    bool yarnMechanicsProbe = false;
    bool aerodynamicsProbe = false;
    bool clothGroundFrictionProbe = false;
    bool rollingResistanceProbe = false;
    bool mouthReleaseProbe = false;
    Scenario scenario = Scenario::grounded;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string value = argv[argument];
        const auto nextUnsigned = [&](std::uint32_t& target) {
            if (argument + 1 >= argc) {
                throw std::invalid_argument("missing value for " + value);
            }
            target = static_cast<std::uint32_t>(std::stoul(argv[++argument]));
        };
        if (value == "--steps") {
            nextUnsigned(steps);
        } else if (value == "--substeps") {
            nextUnsigned(substeps);
        } else if (value == "--iterations") {
            nextUnsigned(iterations);
        } else if (value == "--replays") {
            nextUnsigned(replays);
        } else if (value == "--timestep" && argument + 1 < argc) {
            timestep = std::stod(argv[++argument]);
        } else if (value == "--dump-obj" && argument + 1 < argc) {
            dumpPath = argv[++argument];
        } else if (value == "--dump-frames" && argument + 1 < argc) {
            framePrefix = argv[++argument];
        } else if (value == "--dump-every") {
            nextUnsigned(frameStride);
        } else if (value == "--material" && argument + 1 < argc) {
            materialPath = argv[++argument];
        } else if (value == "--grip-trajectory" && argument + 1 < argc) {
            gripTrajectoryPath = argv[++argument];
        } else if (value == "--rolling-probe") {
            rollingProbe = true;
        } else if (value == "--self-ccd-probe") {
            selfCCDProbe = true;
        } else if (value == "--strain-probe") {
            strainProbe = true;
        } else if (value == "--self-friction-probe") {
            selfFrictionProbe = true;
        } else if (value == "--deformable-response-probe") {
            deformableResponseProbe = true;
        } else if (value == "--yarn-mechanics-probe") {
            yarnMechanicsProbe = true;
        } else if (value == "--aerodynamics-probe") {
            aerodynamicsProbe = true;
        } else if (value == "--cloth-ground-friction-probe") {
            clothGroundFrictionProbe = true;
        } else if (value == "--rolling-resistance-probe") {
            rollingResistanceProbe = true;
        } else if (value == "--mouth-release-probe") {
            mouthReleaseProbe = true;
        } else if (value == "--scenario" && argument + 1 < argc) {
            const std::string name = argv[++argument];
            if (name == "grounded") {
                scenario = Scenario::grounded;
            } else if (name == "spin") {
                scenario = Scenario::spin;
            } else if (name == "pickup") {
                scenario = Scenario::pickup;
            } else if (name == "recorded") {
                scenario = Scenario::recorded;
            } else {
                throw std::invalid_argument("unknown scenario: " + name);
            }
        } else if (value == "--help") {
            std::cout << "usage: numi-solver-cloth-bag "
                         "[--steps N] [--substeps N] [--iterations N] "
                         "[--replays 1|2] "
                         "[--timestep DT] "
                         "[--scenario grounded|spin|pickup|recorded] "
                         "[--material FILE] [--grip-trajectory FILE] "
                         "[--dump-obj PATH] [--dump-frames PREFIX] "
                         "[--dump-every N] [--rolling-probe] "
                         "[--self-ccd-probe] [--strain-probe] "
                         "[--self-friction-probe] "
                         "[--deformable-response-probe] "
                         "[--yarn-mechanics-probe] "
                         "[--aerodynamics-probe] "
                         "[--cloth-ground-friction-probe] "
                         "[--rolling-resistance-probe] "
                         "[--mouth-release-probe]\n";
            return 0;
        } else {
            throw std::invalid_argument("unknown argument: " + value);
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
    if ((scenario == Scenario::recorded) !=
        (gripTrajectoryPointer != nullptr)) {
        throw std::invalid_argument(
            "--scenario recorded and --grip-trajectory must be used together"
        );
    }
    if (rollingProbe) {
        return runRollingProbe() ? 0 : 1;
    }
    if (selfCCDProbe) {
        return runSelfCCDProbe() ? 0 : 1;
    }
    if (strainProbe) {
        return runStrainProbe() ? 0 : 1;
    }
    if (selfFrictionProbe) {
        return runSelfFrictionProbe() ? 0 : 1;
    }
    if (deformableResponseProbe) {
        return runDeformableResponseProbe() ? 0 : 1;
    }
    if (yarnMechanicsProbe) {
        return runYarnMechanicsProbe() ? 0 : 1;
    }
    if (aerodynamicsProbe) {
        return runAerodynamicsProbe() ? 0 : 1;
    }
    if (clothGroundFrictionProbe) {
        return runClothGroundFrictionProbe() ? 0 : 1;
    }
    if (rollingResistanceProbe) {
        return runRollingResistanceProbe() ? 0 : 1;
    }
    if (mouthReleaseProbe) {
        return runMouthReleaseProbe() ? 0 : 1;
    }
    if (steps == 0u || substeps == 0u || iterations == 0u ||
        (replays != 1u && replays != 2u) ||
        !std::isfinite(timestep) || timestep <= 0.0) {
        throw std::invalid_argument("simulation controls must be positive");
    }
    if (frameStride != 0u && framePrefix.empty()) {
        throw std::invalid_argument("--dump-every requires --dump-frames");
    }

    std::vector<std::uint32_t> captureSteps;
    std::vector<SimulationResult> captures;
    if (!framePrefix.empty()) {
        if (frameStride == 0u) {
            for (std::uint32_t quarter = 0u; quarter <= 4u; ++quarter) {
                const std::uint32_t completed =
                    (steps * quarter + 2u) / 4u;
                if (captureSteps.empty() || captureSteps.back() != completed) {
                    captureSteps.push_back(completed);
                }
            }
        } else {
            for (std::uint32_t completed = 0u;
                 completed < steps;
                 completed += frameStride) {
                captureSteps.push_back(completed);
            }
            captureSteps.push_back(steps);
        }
    }
    const SimulationResult first = simulate(
        steps,
        timestep,
        substeps,
        iterations,
        scenario,
        framePrefix.empty() ? nullptr : &captureSteps,
        framePrefix.empty() ? nullptr : &captures,
        gripTrajectoryPointer
    );
    const std::uint64_t firstHash = hashResult(first);
    std::uint64_t replayHash = 0u;
    if (replays == 2u) {
        replayHash = hashResult(simulate(
            steps,
            timestep,
            substeps,
            iterations,
            scenario,
            nullptr,
            nullptr,
            gripTrajectoryPointer
        ));
    }
    const bool deterministic = replays == 2u && firstHash == replayHash;
    if (!dumpPath.empty()) {
        dumpOBJ(dumpPath, first);
    }
    for (std::size_t index = 0; index < captures.size(); ++index) {
        dumpOBJ(
            framePrefix + "-" + std::to_string(captureSteps[index]) + ".obj",
            captures[index]
        );
    }

    const Metrics& metrics = first.metrics;
    double clothMass = 0.0;
    for (const Particle& particle : first.cloth.particles) {
        clothMass += particle.mass;
    }
    double fruitMass = 0.0;
    for (const Ball& ball : first.balls) {
        fruitMass += 1.0 / ball.inverseMass;
    }
    std::cout << std::fixed << std::setprecision(9);
    const char* scenarioName = scenario == Scenario::grounded ? "grounded" :
        scenario == Scenario::spin ? "spin" :
        scenario == Scenario::pickup ? "pickup" : "recorded";
    std::cout << "material_schema=" << numi::kClothMaterialSchema
              << " material_artifact_loaded=" << std::boolalpha
              << gClothMaterial.loaded
              << " parameters_hash=" << gClothMaterial.parametersHash
              << " observations_hash=" << gClothMaterial.observationsHash
              << '\n';
    if (gripTrajectoryPointer != nullptr) {
        std::cout << "grip_trajectory_schema="
                  << gripTrajectory.schema
                  << " content_fingerprint="
                  << gripTrajectory.contentFingerprint
                  << " poses=" << gripTrajectory.poses.size()
                  << " duration_seconds="
                  << gripTrajectory.poses.back().timeSeconds
                  << " maximum_rotation_radians="
                  << numi::maximumGripTrajectoryRotation(gripTrajectory)
                  << " attachment_generations="
                  << numi::gripTrajectoryAttachmentGenerations(
                         gripTrajectory
                     )
                  << " selection_mode="
                  << (gripTrajectory.selectNearestCuffPatch
                          ? "nearest_cuff_patch"
                          : "fixed_patch")
                  << '\n';
    }
    std::cout << "model=explicit_yarn_cloth_reference"
              << " scenario=" << scenarioName
              << " nodes=" << first.cloth.particles.size()
              << " render_triangles="
              << first.cloth.renderTriangles.size()
              << " stretch_constraints=" << first.cloth.distances.size()
              << " bend_constraints=" << first.cloth.bends.size()
              << " yarn_segments=" << first.cloth.yarnSegments.size()
              << " knot_constraints=" << first.cloth.knots.size()
              << " balls=" << first.balls.size()
              << " cloth_mass_kg=" << clothMass
              << " fruit_mass_kg=" << fruitMass
              << " steps=" << steps
              << " substeps=" << substeps
              << " iterations=" << iterations
              << " replays=" << replays
              << " simulated_seconds=" << steps * timestep << '\n';
    if (!captures.empty()) {
        std::cout << "captured_frames=" << captures.size()
                  << " frame_prefix=" << framePrefix << '\n';
    }
    std::cout << "max_warp_strain=" << metrics.maximumWarpStrain
              << " max_weft_strain=" << metrics.maximumWeftStrain
              << " max_knot_angle_error="
              << metrics.maximumKnotAngleError
              << " max_yarn_bend_chord_error="
              << metrics.maximumBendError
              << " min_triangle_area=" << metrics.minimumTriangleArea << '\n';
    std::cout << "max_warp_extension=" << metrics.maximumWarpExtension
              << " max_warp_compression=" << metrics.maximumWarpCompression
              << " max_weft_extension=" << metrics.maximumWeftExtension
              << " max_weft_compression=" << metrics.maximumWeftCompression
              << " max_warp_extension_edge="
              << metrics.maximumWarpExtensionFirst << ':'
              << metrics.maximumWarpExtensionSecond
              << '\n';
    std::cout << "max_bottom_strain=" << metrics.maximumBottomStrain
              << " max_bottom_extension=" << metrics.maximumBottomExtension
              << " max_bottom_compression=" << metrics.maximumBottomCompression
              << '\n';
    std::cout << "max_ball_contact_correction="
              << metrics.maximumBallPenetration
              << " max_published_ball_penetration="
              << metrics.maximumPublishedBallPenetration
              << " max_published_primitive_self_penetration="
              << metrics.maximumPublishedPrimitiveSelfPenetration
              << " max_published_strain_limit_violation="
              << metrics.maximumPublishedStrainLimitViolation
              << " max_self_penetration=" << metrics.maximumSelfPenetration
              << " ball_yarn_contacts=" << metrics.ballYarnContacts
              << " self_contacts=" << metrics.selfContacts
              << " escaped_mask=" << metrics.escapedMask
              << " spilled_mask=" << metrics.spilledMask
              << " mouth_candidate_mask="
              << metrics.mouthCandidateMask
              << " released_mask=" << metrics.releasedMask << '\n';
    std::cout << "max_ball_contact_correction_by_fruit=";
    for (std::size_t index = 0;
         index < metrics.maximumBallPenetrationByFruit.size();
         ++index) {
        if (index != 0u) {
            std::cout << ',';
        }
        std::cout << metrics.maximumBallPenetrationByFruit[index];
    }
    std::cout << '\n';
    std::cout << "max_mouth_clearance_by_fruit=";
    for (std::size_t index = 0;
         index < metrics.maximumMouthClearanceByFruit.size();
         ++index) {
        if (index != 0u) {
            std::cout << ',';
        }
        std::cout << metrics.maximumMouthClearanceByFruit[index];
    }
    std::cout << '\n';
    std::cout << "max_ground_contact_correction="
              << metrics.maximumGroundPenetration
              << " max_cloth_ground_correction="
              << metrics.maximumClothGroundCorrection
              << " max_ball_ground_correction="
              << metrics.maximumBallGroundCorrection
              << " max_published_ground_penetration="
              << metrics.maximumPublishedGroundPenetration
              << " max_contact_reconciliation_passes="
              << metrics.maximumContactReconciliationPasses
              << " max_primitive_certificate_passes="
              << metrics.maximumPrimitiveCertificatePasses
              << " max_swept_ground_advance="
              << metrics.maximumSweptGroundAdvance
              << " max_swept_ball_advance="
              << metrics.maximumSweptBallAdvance
              << " max_swept_self_advance="
              << metrics.maximumSweptSelfAdvance
              << " final_primitive_self_penetration="
              << metrics.finalPrimitiveSelfPenetration
              << " max_strain_limit_correction="
              << metrics.maximumStrainLimitCorrection
              << " final_strain_limit_violation="
              << metrics.finalStrainLimitViolation
              << " max_speed=" << metrics.maximumSpeed
              << " max_angular_speed=" << metrics.maximumAngularSpeed
              << " max_yarn_aerodynamic_force="
              << metrics.maximumYarnAerodynamicForce
              << " max_fruit_aerodynamic_force="
              << metrics.maximumFruitAerodynamicForce
              << " max_fruit_aerodynamic_torque="
              << metrics.maximumFruitAerodynamicTorque
              << " aerodynamic_dissipation="
              << metrics.aerodynamicDissipation
              << " max_grip_force=" << metrics.maximumGripForce
              << " max_grip_impulse=" << metrics.maximumGripImpulse
              << " regrab_count=" << metrics.regrabCount
              << " max_regrab_capture_distance="
              << metrics.maximumRegrabCaptureDistance
              << " max_regrab_capture_error="
              << metrics.maximumRegrabCaptureError
              << " inactive_grip_substeps="
              << metrics.inactiveGripSubsteps
              << " patch_selection_count="
              << metrics.gripPatchSelectionCount
              << " maximum_patch_ring_shift="
              << metrics.maximumGripPatchRingShift
              << " selected_patch_center_ring="
              << first.cloth.gripPatchCenterRing
              << " patch_particles_unique="
              << gripParticlesUnique(first.cloth)
              << " patch_topology_exact="
              << gripPatchTopologyExact(first.cloth)
              << " max_friction_cone_ratio="
              << metrics.maximumFrictionConeRatio
              << " max_rolling_resistance_ratio="
              << metrics.maximumRollingResistanceRatio
              << " tangential_impulse="
              << metrics.accumulatedTangentialImpulse
              << " cloth_friction_contacts="
              << metrics.ballClothFrictionContacts
              << " pair_friction_contacts="
              << metrics.ballPairFrictionContacts
              << " ground_friction_contacts="
              << metrics.ballGroundFrictionContacts
              << " rolling_resistance_contacts="
              << metrics.ballRollingResistanceContacts
              << " swept_ball_yarn_contacts="
              << metrics.sweptBallYarnContacts
              << " edge_edge_self_contacts="
              << metrics.edgeEdgeSelfContacts
              << " swept_edge_edge_self_contacts="
              << metrics.sweptEdgeEdgeSelfContacts
              << " edge_edge_candidate_pairs="
              << metrics.edgeEdgeCandidatePairs
              << " edge_edge_sphere_candidate_pairs="
              << metrics.edgeEdgeSphereCandidatePairs
              << " cloth_self_friction_contacts="
              << metrics.clothSelfFrictionContacts
              << " cloth_ground_friction_contacts="
              << metrics.clothGroundFrictionContacts
              << " deterministic=" << std::boolalpha << deterministic
              << " state_hash=0x" << std::hex << firstHash << std::dec << '\n';
    std::cout << "ball_cloth_solve_seconds=" << metrics.ballClothSolveSeconds
              << " primitive_self_solve_seconds="
              << metrics.primitiveSelfSolveSeconds
              << " point_self_solve_seconds="
              << metrics.pointSelfSolveSeconds
              << " strain_limit_solve_seconds="
              << metrics.strainLimitSolveSeconds
              << " primitive_certificate_seconds="
              << metrics.primitiveCertificateSeconds << '\n';
    const bool pass = acceptable(first, deterministic);
    std::cout << "result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << "numi-solver-cloth-bag: " << error.what() << '\n';
    return 2;
}
