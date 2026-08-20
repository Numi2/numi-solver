#include <algorithm>
#include <array>
#include <bit>
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
constexpr double kClothRadius = 0.004;
constexpr double kAirborneLift = 0.52;
constexpr double kInitialGroundLift = 0.009;
constexpr double kClothNodeMass = 0.000050;
constexpr double kClothHemNodeMass = 0.000100;
constexpr double kClothMass =
    static_cast<double>(
        kAround * (kLevels - 2u) + kBottomInterior * kBottomInterior
    ) * kClothNodeMass +
    static_cast<double>(2u * kAround) * kClothHemNodeMass;
constexpr double kClothContactPatchMass = kClothMass;
constexpr double kFruitGroundFriction = 0.42;
constexpr double kFruitPairFriction = 0.30;
constexpr double kFruitClothFriction = 0.36;
constexpr double kFruitRollingResistanceRate = 0.35;
constexpr std::size_t kBallPairCount =
    kFruitCount * (kFruitCount - 1u) / 2u;

enum class Scenario : std::uint8_t {
    grounded,
    spin,
    pickup,
};

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

struct BallTriangleContactImpulse {
    Vec3 weightedNormalOnBall{};
    std::array<double, 3> weightedBarycentric{};
    double normalImpulse{};
};

enum class DistanceKind : std::uint8_t {
    warp,
    weft,
    shear,
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

struct BendConstraint {
    std::uint32_t edgeFirst{};
    std::uint32_t edgeSecond{};
    std::uint32_t oppositeFirst{};
    std::uint32_t oppositeSecond{};
    double restAngle{};
    double compliance{};
    double lambda{};
};

struct GripConstraint {
    std::uint32_t particle{};
    Vec3 targetOffset{};
    Vec3 lambda{};
    double compliance{2.0e-3};
};

struct ClothModel {
    std::vector<Particle> particles;
    std::vector<DistanceConstraint> distances;
    std::vector<Triangle> triangles;
    std::vector<Edge> collisionEdges;
    std::vector<BendConstraint> bends;
    std::vector<GripConstraint> grips;
    std::vector<std::vector<std::uint32_t>> localTopology;
    std::uint32_t bottomCenter{};
    Vec3 gripTarget{};
    Vec3 gripPrevious{};
};

struct Metrics {
    double maximumWarpStrain{};
    double maximumWeftStrain{};
    double maximumShearStrain{};
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
    double maximumBallPenetration{};
    double maximumSelfPenetration{};
    double maximumGroundPenetration{};
    double maximumSweptGroundAdvance{};
    double maximumSweptBallAdvance{};
    double minimumTriangleArea{std::numeric_limits<double>::infinity()};
    double maximumSpeed{};
    double maximumAngularSpeed{};
    double maximumFrictionConeRatio{};
    double accumulatedTangentialImpulse{};
    double maximumGripForce{};
    double maximumGripImpulse{};
    std::uint64_t ballTriangleContacts{};
    std::uint64_t sweptBallTriangleContacts{};
    std::uint64_t selfContacts{};
    std::uint64_t vertexTriangleSelfContacts{};
    std::uint64_t edgeEdgeSelfContacts{};
    std::uint64_t sweptVertexTriangleSelfContacts{};
    std::uint64_t sweptEdgeEdgeSelfContacts{};
    double maximumSweptSelfAdvance{};
    double finalPrimitiveSelfPenetration{};
    double maximumStrainLimitCorrection{};
    double finalStrainLimitViolation{};
    std::uint64_t ballClothFrictionContacts{};
    std::uint64_t ballPairFrictionContacts{};
    std::uint64_t ballGroundFrictionContacts{};
    std::uint32_t escapedMask{};
    std::uint32_t spilledMask{};
    std::uint32_t releasedMask{};
    std::array<double, kFruitCount> maximumBallPenetrationByFruit{};
};

struct SimulationResult {
    ClothModel cloth;
    std::array<Ball, kFruitCount> balls{};
    Metrics metrics{};
    Scenario scenario{Scenario::grounded};
};

std::uint32_t nodeIndex(const std::uint32_t level, const std::uint32_t ring) {
    return level * kAround + ring % kAround;
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

double wrapAngle(double angle) {
    while (angle > std::numbers::pi) {
        angle -= 2.0 * std::numbers::pi;
    }
    while (angle < -std::numbers::pi) {
        angle += 2.0 * std::numbers::pi;
    }
    return angle;
}

double signedDihedral(
    const Vec3 edgeFirst,
    const Vec3 edgeSecond,
    const Vec3 oppositeFirst,
    const Vec3 oppositeSecond
) {
    const Vec3 edge = edgeSecond - edgeFirst;
    const Vec3 direction = normalized(edge);
    const Vec3 firstNormal = normalized(cross(edge, oppositeFirst - edgeFirst));
    const Vec3 secondNormal = normalized(cross(oppositeSecond - edgeFirst, edge));
    return std::atan2(
        dot(cross(firstNormal, secondNormal), direction),
        std::clamp(dot(firstNormal, secondNormal), -1.0, 1.0)
    );
}

std::uint64_t edgeKey(std::uint32_t first, std::uint32_t second) {
    if (first > second) {
        std::swap(first, second);
    }
    return (static_cast<std::uint64_t>(first) << 32u) | second;
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
        for (const int offset : {-2, -1, 0, 1, 2}) {
            const std::uint32_t ring = static_cast<std::uint32_t>(
                (static_cast<int>(kAround) + offset) %
                static_cast<int>(kAround)
            );
            const std::uint32_t index = nodeIndex(kLevels - 1u, ring);
            model.grips.push_back({
                .particle = index,
                .targetOffset =
                    model.particles[index].rest - model.gripTarget,
            });
        }
    }

    const auto addDistance = [&](
        const std::uint32_t first,
        const std::uint32_t second,
        const double compliance,
        const DistanceKind kind
    ) {
        model.distances.push_back({
            first,
            second,
            length(model.particles[second].rest - model.particles[first].rest),
            compliance,
            0.0,
            kind,
        });
    };

    for (std::uint32_t level = 0; level < kLevels; ++level) {
        for (std::uint32_t ring = 0; ring < kAround; ++ring) {
            addDistance(
                nodeIndex(level, ring),
                nodeIndex(level, ring + 1u),
                level + 2u >= kLevels ? 1.0e-9 : 1.0e-8,
                DistanceKind::weft
            );
        }
    }
    for (std::uint32_t level = 0; level + 1u < kLevels; ++level) {
        for (std::uint32_t ring = 0; ring < kAround; ++ring) {
            addDistance(
                nodeIndex(level, ring),
                nodeIndex(level + 1u, ring),
                1.0e-8,
                DistanceKind::warp
            );
            addDistance(
                nodeIndex(level, ring),
                nodeIndex(level + 1u, ring + 1u),
                4.0e-8,
                DistanceKind::shear
            );
            addDistance(
                nodeIndex(level, ring + 1u),
                nodeIndex(level + 1u, ring),
                4.0e-8,
                DistanceKind::shear
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
                    1.0e-8,
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
                    1.0e-8,
                    DistanceKind::bottom
                );
            }
        }
    }
    for (std::uint32_t row = 0u; row + 1u < kBottomGrid; ++row) {
        for (std::uint32_t column = 0u;
             column + 1u < kBottomGrid;
             ++column) {
            addDistance(
                bottomGridIndex(row, column),
                bottomGridIndex(row + 1u, column + 1u),
                4.0e-8,
                DistanceKind::bottom
            );
            addDistance(
                bottomGridIndex(row, column + 1u),
                bottomGridIndex(row + 1u, column),
                4.0e-8,
                DistanceKind::bottom
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
            model.triangles.push_back({a, b, c});
            model.triangles.push_back({b, d, c});
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
                model.triangles.push_back({a, c, b});
                model.triangles.push_back({b, c, d});
            } else {
                model.triangles.push_back({a, d, b});
                model.triangles.push_back({a, c, d});
            }
        }
    }

    std::unordered_map<std::uint64_t, bool> collisionEdgeKeys;
    collisionEdgeKeys.reserve(model.triangles.size() * 2u);
    for (const Triangle triangle : model.triangles) {
        const std::array<std::array<std::uint32_t, 2>, 3> edges{{
            {{triangle.first, triangle.second}},
            {{triangle.second, triangle.third}},
            {{triangle.third, triangle.first}},
        }};
        for (const auto edge : edges) {
            const std::uint64_t key = edgeKey(edge[0], edge[1]);
            if (collisionEdgeKeys.emplace(key, true).second) {
                model.collisionEdges.push_back({edge[0], edge[1]});
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

    struct HalfEdge {
        std::uint32_t edgeFirst{};
        std::uint32_t edgeSecond{};
        std::uint32_t opposite{};
    };
    std::unordered_map<std::uint64_t, HalfEdge> edges;
    edges.reserve(model.triangles.size() * 2u);
    for (const Triangle triangle : model.triangles) {
        const std::array<std::array<std::uint32_t, 3>, 3> triangleEdges{{
            {{triangle.first, triangle.second, triangle.third}},
            {{triangle.second, triangle.third, triangle.first}},
            {{triangle.third, triangle.first, triangle.second}},
        }};
        for (const auto entry : triangleEdges) {
            const std::uint64_t key = edgeKey(entry[0], entry[1]);
            const auto found = edges.find(key);
            if (found == edges.end()) {
                edges.emplace(key, HalfEdge{entry[0], entry[1], entry[2]});
                continue;
            }
            const HalfEdge first = found->second;
            const auto& particles = model.particles;
            const double restAngle = signedDihedral(
                particles[first.edgeFirst].rest,
                particles[first.edgeSecond].rest,
                particles[first.opposite].rest,
                particles[entry[2]].rest
            );
            model.bends.push_back({
                first.edgeFirst,
                first.edgeSecond,
                first.opposite,
                entry[2],
                restAngle,
                8.0e-2,
                0.0,
            });
            edges.erase(found);
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
    const double lift = smoothstep(time / 0.58);
    const double pour = smoothstep((time - 0.48) / 0.62);
    return base + Vec3{
        -0.54 * pour,
        0.12 * std::sin(std::numbers::pi * pour),
        0.72 * lift + 0.08 * std::sin(std::numbers::pi * pour) - 0.12 * pour,
    };
}

void updateGrip(
    ClothModel& cloth,
    const Scenario scenario,
    const double time
) {
    cloth.gripPrevious = cloth.gripTarget;
    cloth.gripTarget = scenario == Scenario::spin
        ? spinGripTarget(time)
        : pickupGripTarget(time);
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

double strainExtensionLimit(const DistanceKind kind) {
    return kind == DistanceKind::shear ? 0.38 : 0.285;
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

double bendAngle(const std::array<Vec3, 4>& positions) {
    return signedDihedral(positions[0], positions[1], positions[2], positions[3]);
}

void solveBend(
    std::vector<Particle>& particles,
    BendConstraint& constraint,
    const double timestep
) {
    const std::array<std::uint32_t, 4> indices{{
        constraint.edgeFirst,
        constraint.edgeSecond,
        constraint.oppositeFirst,
        constraint.oppositeSecond,
    }};
    std::array<Vec3, 4> positions{};
    for (std::size_t index = 0; index < 4; ++index) {
        positions[index] = particles[indices[index]].position;
    }
    const double value = wrapAngle(bendAngle(positions) - constraint.restAngle);
    constexpr double epsilon = 2.0e-6;
    std::array<Vec3, 4> gradients{};
    for (std::size_t vertex = 0; vertex < 4; ++vertex) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            std::array<Vec3, 4> plus = positions;
            std::array<Vec3, 4> minus = positions;
            double* plusValue = axis == 0 ? &plus[vertex].x :
                axis == 1 ? &plus[vertex].y : &plus[vertex].z;
            double* minusValue = axis == 0 ? &minus[vertex].x :
                axis == 1 ? &minus[vertex].y : &minus[vertex].z;
            *plusValue += epsilon;
            *minusValue -= epsilon;
            const double derivative = wrapAngle(
                bendAngle(plus) - bendAngle(minus)
            ) / (2.0 * epsilon);
            double* target = axis == 0 ? &gradients[vertex].x :
                axis == 1 ? &gradients[vertex].y : &gradients[vertex].z;
            *target = derivative;
        }
    }
    const double alpha = constraint.compliance / (timestep * timestep);
    double denominator = alpha;
    for (std::size_t index = 0; index < 4; ++index) {
        denominator += particles[indices[index]].inverseMass *
            lengthSquared(gradients[index]);
    }
    if (denominator < 1.0e-16) {
        return;
    }
    const double deltaLambda =
        (-value - alpha * constraint.lambda) / denominator;
    constraint.lambda += deltaLambda;
    for (std::size_t index = 0; index < 4; ++index) {
        particles[indices[index]].position += gradients[index] *
            (particles[indices[index]].inverseMass * deltaLambda);
    }
}

struct ClosestPoint {
    Vec3 point{};
    std::array<double, 3> barycentric{};
};

ClosestPoint closestPointOnTriangle(
    const Vec3 point,
    const Vec3 first,
    const Vec3 second,
    const Vec3 third
) {
    const Vec3 firstSecond = second - first;
    const Vec3 firstThird = third - first;
    const Vec3 firstPoint = point - first;
    const double d1 = dot(firstSecond, firstPoint);
    const double d2 = dot(firstThird, firstPoint);
    if (d1 <= 0.0 && d2 <= 0.0) {
        return {first, {1.0, 0.0, 0.0}};
    }
    const Vec3 secondPoint = point - second;
    const double d3 = dot(firstSecond, secondPoint);
    const double d4 = dot(firstThird, secondPoint);
    if (d3 >= 0.0 && d4 <= d3) {
        return {second, {0.0, 1.0, 0.0}};
    }
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        const double v = d1 / (d1 - d3);
        return {first + firstSecond * v, {1.0 - v, v, 0.0}};
    }
    const Vec3 thirdPoint = point - third;
    const double d5 = dot(firstSecond, thirdPoint);
    const double d6 = dot(firstThird, thirdPoint);
    if (d6 >= 0.0 && d5 <= d6) {
        return {third, {0.0, 0.0, 1.0}};
    }
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        const double w = d2 / (d2 - d6);
        return {first + firstThird * w, {1.0 - w, 0.0, w}};
    }
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        const double w = (d4 - d3) /
            ((d4 - d3) + (d5 - d6));
        return {second + (third - second) * w, {0.0, 1.0 - w, w}};
    }
    const double denominator = 1.0 / (va + vb + vc);
    const double v = vb * denominator;
    const double w = vc * denominator;
    return {
        first + firstSecond * v + firstThird * w,
        {1.0 - v - w, v, w},
    };
}

struct SweptTriangleSample {
    Vec3 ballPosition{};
    std::array<Vec3, 3> trianglePositions{};
    ClosestPoint closest{};
    double distance{};
};

SweptTriangleSample sampleSweptTriangle(
    const std::vector<Particle>& particles,
    const std::array<std::uint32_t, 3> indices,
    const Ball& ball,
    const double time
) {
    SweptTriangleSample sample;
    sample.ballPosition = ball.previous +
        (ball.position - ball.previous) * time;
    for (std::size_t index = 0; index < 3; ++index) {
        const Particle& particle = particles[indices[index]];
        sample.trianglePositions[index] = particle.previous +
            (particle.position - particle.previous) * time;
    }
    sample.closest = closestPointOnTriangle(
        sample.ballPosition,
        sample.trianglePositions[0],
        sample.trianglePositions[1],
        sample.trianglePositions[2]
    );
    sample.distance = length(sample.closest.point - sample.ballPosition);
    return sample;
}

double solveSweptBallTriangle(
    std::vector<Particle>& particles,
    const Triangle triangle,
    Ball& ball,
    const double timestep,
    BallTriangleContactImpulse& contactImpulse,
    std::uint64_t& contactCount
) {
    const std::array<std::uint32_t, 3> indices{{
        triangle.first,
        triangle.second,
        triangle.third,
    }};
    const double target = ball.radius + kClothRadius;
    const Vec3 ballMinimum{
        std::min(ball.previous.x, ball.position.x) - target,
        std::min(ball.previous.y, ball.position.y) - target,
        std::min(ball.previous.z, ball.position.z) - target,
    };
    const Vec3 ballMaximum{
        std::max(ball.previous.x, ball.position.x) + target,
        std::max(ball.previous.y, ball.position.y) + target,
        std::max(ball.previous.z, ball.position.z) + target,
    };
    Vec3 triangleMinimum{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
    Vec3 triangleMaximum{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
    double motionBound = length(ball.position - ball.previous);
    for (const std::uint32_t index : indices) {
        const Particle& particle = particles[index];
        triangleMinimum.x = std::min({
            triangleMinimum.x, particle.previous.x, particle.position.x,
        });
        triangleMinimum.y = std::min({
            triangleMinimum.y, particle.previous.y, particle.position.y,
        });
        triangleMinimum.z = std::min({
            triangleMinimum.z, particle.previous.z, particle.position.z,
        });
        triangleMaximum.x = std::max({
            triangleMaximum.x, particle.previous.x, particle.position.x,
        });
        triangleMaximum.y = std::max({
            triangleMaximum.y, particle.previous.y, particle.position.y,
        });
        triangleMaximum.z = std::max({
            triangleMaximum.z, particle.previous.z, particle.position.z,
        });
        motionBound += length(particle.position - particle.previous);
    }
    if (ballMaximum.x < triangleMinimum.x ||
        ballMinimum.x > triangleMaximum.x ||
        ballMaximum.y < triangleMinimum.y ||
        ballMinimum.y > triangleMaximum.y ||
        ballMaximum.z < triangleMinimum.z ||
        ballMinimum.z > triangleMaximum.z || motionBound < 1.0e-14) {
        return 0.0;
    }
    constexpr double distanceTolerance = 1.0e-9;
    const SweptTriangleSample start = sampleSweptTriangle(
        particles, indices, ball, 0.0
    );
    double time = 0.0;
    SweptTriangleSample impact = start;
    bool found = start.distance <= target + distanceTolerance;
    if (!found) {
        for (std::uint32_t iteration = 0; iteration < 80u; ++iteration) {
            impact = sampleSweptTriangle(particles, indices, ball, time);
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
        (impact.closest.point - impact.ballPosition) / impact.distance;
    Vec3 triangleRemaining{};
    for (std::size_t index = 0; index < 3; ++index) {
        triangleRemaining +=
            (particles[indices[index]].position -
             impact.trianglePositions[index]) *
            impact.closest.barycentric[index];
    }
    const Vec3 ballRemaining = ball.position - impact.ballPosition;
    const double removedAdvance = dot(
        ballRemaining - triangleRemaining,
        normal
    );
    if (removedAdvance <= 0.0) {
        return 0.0;
    }
    std::array<double, 3> clothContactInverseMass{};
    double denominator = ball.inverseMass;
    for (std::size_t index = 0; index < 3; ++index) {
        clothContactInverseMass[index] = std::min(
            particles[indices[index]].inverseMass,
            1.0 / kClothContactPatchMass
        );
        denominator += clothContactInverseMass[index] *
            impact.closest.barycentric[index] *
            impact.closest.barycentric[index];
    }
    if (denominator <= 0.0) {
        return 0.0;
    }
    const double lambda = removedAdvance / denominator;
    ball.position -= normal * (ball.inverseMass * lambda);
    for (std::size_t index = 0; index < 3; ++index) {
        particles[indices[index]].position += normal *
            (clothContactInverseMass[index] *
             impact.closest.barycentric[index] * lambda);
    }
    const double impulseMagnitude = lambda / timestep;
    contactImpulse.weightedNormalOnBall -= normal * impulseMagnitude;
    for (std::size_t index = 0; index < 3; ++index) {
        contactImpulse.weightedBarycentric[index] +=
            impact.closest.barycentric[index] * impulseMagnitude;
    }
    contactImpulse.normalImpulse += impulseMagnitude;
    ++contactCount;
    return removedAdvance;
}

double solveBallTriangle(
    std::vector<Particle>& particles,
    const Triangle triangle,
    Ball& ball,
    const double timestep,
    BallTriangleContactImpulse& contactImpulse,
    std::uint64_t& contactCount
) {
    const std::array<std::uint32_t, 3> indices{{
        triangle.first,
        triangle.second,
        triangle.third,
    }};
    const Vec3 first = particles[indices[0]].position;
    const Vec3 second = particles[indices[1]].position;
    const Vec3 third = particles[indices[2]].position;
    const double target = ball.radius + kClothRadius;
    if (ball.position.x < std::min({first.x, second.x, third.x}) - target ||
        ball.position.x > std::max({first.x, second.x, third.x}) + target ||
        ball.position.y < std::min({first.y, second.y, third.y}) - target ||
        ball.position.y > std::max({first.y, second.y, third.y}) + target ||
        ball.position.z < std::min({first.z, second.z, third.z}) - target ||
        ball.position.z > std::max({first.z, second.z, third.z}) + target) {
        return 0.0;
    }
    const ClosestPoint closest = closestPointOnTriangle(
        ball.position,
        first,
        second,
        third
    );
    Vec3 separation = closest.point - ball.position;
    double distance = length(separation);
    if (distance >= target) {
        return 0.0;
    }
    if (distance < 1.0e-10) {
        separation = normalized(cross(second - first, third - first));
        distance = 0.0;
    }
    const Vec3 normal = normalized(separation);
    std::array<double, 3> clothContactInverseMass{};
    double denominator = ball.inverseMass;
    for (std::size_t index = 0; index < 3; ++index) {
        clothContactInverseMass[index] = std::min(
            particles[indices[index]].inverseMass,
            1.0 / kClothContactPatchMass
        );
        denominator += clothContactInverseMass[index] *
            closest.barycentric[index] * closest.barycentric[index];
    }
    if (denominator <= 0.0) {
        return target - distance;
    }
    const double correction = (target - distance) / denominator;
    for (std::size_t index = 0; index < 3; ++index) {
        particles[indices[index]].position += normal *
            (clothContactInverseMass[index] *
             closest.barycentric[index] * correction);
    }
    ball.position -= normal * (ball.inverseMass * correction);
    const double impulseMagnitude = correction / timestep;
    contactImpulse.weightedNormalOnBall -= normal * impulseMagnitude;
    for (std::size_t index = 0; index < 3; ++index) {
        contactImpulse.weightedBarycentric[index] +=
            closest.barycentric[index] * impulseMagnitude;
    }
    contactImpulse.normalImpulse += impulseMagnitude;
    ++contactCount;
    return target - distance;
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

void applyBallClothFriction(
    std::vector<Particle>& particles,
    const std::vector<Triangle>& triangles,
    std::array<Ball, kFruitCount>& balls,
    const std::vector<BallTriangleContactImpulse>& contacts,
    Metrics& metrics
) {
    for (std::size_t ballIndex = 0; ballIndex < balls.size(); ++ballIndex) {
        Ball& ball = balls[ballIndex];
        for (std::size_t triangleIndex = 0;
             triangleIndex < triangles.size();
             ++triangleIndex) {
            const BallTriangleContactImpulse& contact = contacts[
                ballIndex * triangles.size() + triangleIndex
            ];
            if (contact.normalImpulse <= 0.0 ||
                lengthSquared(contact.weightedNormalOnBall) < 1.0e-20) {
                continue;
            }
            const Triangle triangle = triangles[triangleIndex];
            const std::array<std::uint32_t, 3> indices{{
                triangle.first,
                triangle.second,
                triangle.third,
            }};
            const Vec3 normal = normalized(contact.weightedNormalOnBall);
            std::array<double, 3> barycentric{};
            Vec3 clothVelocity{};
            double barycentricSum = 0.0;
            for (std::size_t index = 0; index < 3; ++index) {
                barycentric[index] =
                    contact.weightedBarycentric[index] / contact.normalImpulse;
                barycentricSum += barycentric[index];
            }
            if (barycentricSum <= 1.0e-12) {
                continue;
            }
            for (std::size_t index = 0; index < 3; ++index) {
                barycentric[index] /= barycentricSum;
                clothVelocity += particles[indices[index]].velocity *
                    barycentric[index];
            }
            const Vec3 ballOffset = normal * -ball.radius;
            const Vec3 ballContactVelocity = ball.velocity +
                cross(ball.angularVelocity, ballOffset);
            const Vec3 relativeVelocity = ballContactVelocity - clothVelocity;
            const Vec3 tangentVelocity = relativeVelocity -
                normal * dot(relativeVelocity, normal);
            const double slipSpeed = length(tangentVelocity);
            if (slipSpeed < 1.0e-10) {
                continue;
            }
            const Vec3 tangent = tangentVelocity / slipSpeed;
            double denominator = ball.inverseMass + inverseInertia(ball) *
                lengthSquared(cross(ballOffset, tangent));
            std::array<double, 3> clothInverseMass{};
            for (std::size_t index = 0; index < 3; ++index) {
                clothInverseMass[index] = std::min(
                    particles[indices[index]].inverseMass,
                    1.0 / kClothContactPatchMass
                );
                denominator += clothInverseMass[index] *
                    barycentric[index] * barycentric[index];
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
            const Vec3 impulse = tangent * -tangentialImpulse;
            applyBallImpulse(ball, impulse, ballOffset);
            for (std::size_t index = 0; index < 3; ++index) {
                particles[indices[index]].velocity -= impulse *
                    (clothInverseMass[index] * barycentric[index]);
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

void applyBallGroundFriction(
    std::array<Ball, kFruitCount>& balls,
    const std::array<double, kFruitCount>& normalImpulses,
    const double timestep,
    Metrics& metrics,
    const double rollingResistanceRate
) {
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
        ball.angularVelocity = ball.angularVelocity *
            std::exp(-rollingResistanceRate * timestep);
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

struct CCDProbeResult {
    double finalHeight{};
    double removedAdvance{};
    double normalImpulse{};
    std::uint64_t contacts{};
};

CCDProbeResult runCCDProbeOnce() {
    constexpr double timestep = 1.0e-3;
    std::vector<Particle> particles(3);
    particles[0].position = {-1.0, -1.0, 0.0};
    particles[1].position = {1.0, -1.0, 0.0};
    particles[2].position = {0.0, 1.0, 0.0};
    for (Particle& particle : particles) {
        particle.previous = particle.position;
        particle.inverseMass = 0.0;
    }
    Ball ball;
    ball.radius = 0.02;
    ball.inverseMass = 1.0;
    ball.previous = {0.0, 0.0, 0.08};
    ball.position = {0.0, 0.0, -0.08};
    ball.velocity = {0.0, 0.0, -160.0};
    BallTriangleContactImpulse contact;
    std::uint64_t contacts = 0u;
    const double removedAdvance = solveSweptBallTriangle(
        particles,
        Triangle{0u, 1u, 2u},
        ball,
        timestep,
        contact,
        contacts
    );
    return {
        .finalHeight = ball.position.z,
        .removedAdvance = removedAdvance,
        .normalImpulse = contact.normalImpulse,
        .contacts = contacts,
    };
}

bool runCCDProbe() {
    const CCDProbeResult first = runCCDProbeOnce();
    const CCDProbeResult replay = runCCDProbeOnce();
    const double targetHeight = 0.02 + kClothRadius;
    const bool deterministic =
        first.finalHeight == replay.finalHeight &&
        first.removedAdvance == replay.removedAdvance &&
        first.normalImpulse == replay.normalImpulse &&
        first.contacts == replay.contacts;
    const bool pass = deterministic && first.contacts == 1u &&
        std::abs(first.finalHeight - targetHeight) < 2.0e-9 &&
        first.removedAdvance > 0.10 && first.normalImpulse > 100.0;
    std::cout << std::fixed << std::setprecision(12)
              << "probe=swept_sphere_triangle"
              << " start_height=0.080000000000"
              << " predicted_height=-0.080000000000"
              << " contact_height=" << first.finalHeight
              << " expected_height=" << targetHeight
              << " removed_advance=" << first.removedAdvance
              << " normal_impulse=" << first.normalImpulse
              << " contacts=" << first.contacts
              << " deterministic=" << std::boolalpha << deterministic
              << '\n'
              << "result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass;
}

double solveSweptVertexTriangle(
    ClothModel& cloth,
    std::uint32_t vertexIndex,
    Triangle triangle,
    std::uint64_t& contactCount
);

double solveSweptEdgeEdge(
    ClothModel& cloth,
    Edge first,
    Edge second,
    std::uint64_t& contactCount
);

struct SelfCCDProbeResult {
    double vertexHeight{};
    double vertexRemovedAdvance{};
    std::uint64_t vertexContacts{};
    double edgeHeight{};
    double edgeRemovedAdvance{};
    std::uint64_t edgeContacts{};
};

SelfCCDProbeResult runSelfCCDProbeOnce() {
    ClothModel vertexCloth;
    vertexCloth.particles.resize(4);
    vertexCloth.particles[0].previous = {0.0, 0.0, 0.02};
    vertexCloth.particles[0].position = {0.0, 0.0, -0.02};
    vertexCloth.particles[0].inverseMass = 1.0;
    vertexCloth.particles[1].position = {-1.0, -1.0, 0.0};
    vertexCloth.particles[2].position = {1.0, -1.0, 0.0};
    vertexCloth.particles[3].position = {0.0, 1.0, 0.0};
    for (std::size_t index = 1; index < 4; ++index) {
        vertexCloth.particles[index].previous =
            vertexCloth.particles[index].position;
    }
    vertexCloth.localTopology = {{0u}, {1u}, {2u}, {3u}};
    std::uint64_t vertexContacts = 0u;
    const double vertexRemovedAdvance = solveSweptVertexTriangle(
        vertexCloth,
        0u,
        Triangle{1u, 2u, 3u},
        vertexContacts
    );

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
        .vertexHeight = vertexCloth.particles[0].position.z,
        .vertexRemovedAdvance = vertexRemovedAdvance,
        .vertexContacts = vertexContacts,
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
        first.vertexHeight == replay.vertexHeight &&
        first.vertexRemovedAdvance == replay.vertexRemovedAdvance &&
        first.vertexContacts == replay.vertexContacts &&
        first.edgeHeight == replay.edgeHeight &&
        first.edgeRemovedAdvance == replay.edgeRemovedAdvance &&
        first.edgeContacts == replay.edgeContacts;
    const bool pass = deterministic &&
        first.vertexContacts == 1u && first.edgeContacts == 1u &&
        std::abs(first.vertexHeight - 2.0 * kClothRadius) < 2.0e-9 &&
        std::abs(first.edgeHeight - 2.0 * kClothRadius) < 2.0e-9 &&
        first.vertexRemovedAdvance > 0.027 &&
        first.edgeRemovedAdvance > 0.027;
    std::cout << std::fixed << std::setprecision(12)
              << "probe=continuous_cloth_self_contact"
              << " vertex_start=0.020000000000"
              << " vertex_predicted=-0.020000000000"
              << " vertex_contact_height=" << first.vertexHeight
              << " vertex_removed_advance=" << first.vertexRemovedAdvance
              << " vertex_contacts=" << first.vertexContacts
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
    std::array<double, kFruitCount>& ballNormalImpulses
) {
    double maximumPenetration = 0.0;
    for (Particle& particle : particles) {
        const double penetration = kClothRadius - particle.position.z;
        if (penetration > 0.0) {
            maximumPenetration = std::max(maximumPenetration, penetration);
            particle.position.z = kClothRadius;
        }
    }
    for (std::size_t index = 0; index < balls.size(); ++index) {
        Ball& ball = balls[index];
        const double penetration = ball.radius - ball.position.z;
        if (penetration > 0.0) {
            maximumPenetration = std::max(maximumPenetration, penetration);
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

bool vertexTriangleLocal(
    const ClothModel& cloth,
    const std::uint32_t vertex,
    const Triangle triangle
) {
    return localTopologyPair(cloth, vertex, triangle.first) ||
        localTopologyPair(cloth, vertex, triangle.second) ||
        localTopologyPair(cloth, vertex, triangle.third);
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

struct SegmentClosest {
    Vec3 firstPoint{};
    Vec3 secondPoint{};
    double firstWeight{};
    double secondWeight{};
};

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

struct VertexTriangleSample {
    Vec3 vertex{};
    std::array<Vec3, 3> triangle{};
    ClosestPoint closest{};
    double distance{};
};

VertexTriangleSample sampleVertexTriangle(
    const ClothModel& cloth,
    const std::uint32_t vertexIndex,
    const Triangle triangle,
    const double time
) {
    VertexTriangleSample sample;
    const Particle& vertex = cloth.particles[vertexIndex];
    sample.vertex = vertex.previous +
        (vertex.position - vertex.previous) * time;
    const std::array<std::uint32_t, 3> indices{{
        triangle.first, triangle.second, triangle.third,
    }};
    for (std::size_t index = 0; index < 3; ++index) {
        const Particle& particle = cloth.particles[indices[index]];
        sample.triangle[index] = particle.previous +
            (particle.position - particle.previous) * time;
    }
    sample.closest = closestPointOnTriangle(
        sample.vertex,
        sample.triangle[0],
        sample.triangle[1],
        sample.triangle[2]
    );
    sample.distance = length(sample.closest.point - sample.vertex);
    return sample;
}

double solveSweptVertexTriangle(
    ClothModel& cloth,
    const std::uint32_t vertexIndex,
    const Triangle triangle,
    std::uint64_t& contactCount
) {
    constexpr double target = 2.0 * kClothRadius;
    constexpr double tolerance = 1.0e-9;
    Particle& vertex = cloth.particles[vertexIndex];
    const std::array<std::uint32_t, 3> indices{{
        triangle.first, triangle.second, triangle.third,
    }};
    double motionBound = length(vertex.position - vertex.previous);
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
    VertexTriangleSample impact = sampleVertexTriangle(
        cloth, vertexIndex, triangle, 0.0
    );
    bool found = impact.distance <= target + tolerance;
    if (!found) {
        for (std::uint32_t iteration = 0; iteration < 80u; ++iteration) {
            impact = sampleVertexTriangle(
                cloth, vertexIndex, triangle, time
            );
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
    const Vec3 previousTrianglePoint =
        cloth.particles[indices[0]].previous * impact.closest.barycentric[0] +
        cloth.particles[indices[1]].previous * impact.closest.barycentric[1] +
        cloth.particles[indices[2]].previous * impact.closest.barycentric[2];
    const Vec3 normal = selfContactNormal(
        impact.closest.point - impact.vertex,
        impact.triangle[1] - impact.triangle[0],
        impact.triangle[2] - impact.triangle[0],
        vertex.previous - previousTrianglePoint
    );
    Vec3 triangleRemaining{};
    for (std::size_t index = 0; index < 3; ++index) {
        triangleRemaining += (
            cloth.particles[indices[index]].position - impact.triangle[index]
        ) * impact.closest.barycentric[index];
    }
    const double removedAdvance = dot(
        (vertex.position - impact.vertex) - triangleRemaining,
        normal
    );
    if (removedAdvance <= 0.0) {
        return 0.0;
    }
    double denominator = vertex.inverseMass;
    for (std::size_t index = 0; index < 3; ++index) {
        denominator += cloth.particles[indices[index]].inverseMass *
            impact.closest.barycentric[index] *
            impact.closest.barycentric[index];
    }
    if (denominator <= 0.0) {
        return 0.0;
    }
    const double lambda = removedAdvance / denominator;
    vertex.position -= normal * (vertex.inverseMass * lambda);
    for (std::size_t index = 0; index < 3; ++index) {
        cloth.particles[indices[index]].position += normal *
            (cloth.particles[indices[index]].inverseMass *
             impact.closest.barycentric[index] * lambda);
    }
    ++contactCount;
    return removedAdvance;
}

double solveVertexTriangle(
    ClothModel& cloth,
    const std::uint32_t vertexIndex,
    const Triangle triangle,
    std::uint64_t& contactCount
) {
    constexpr double target = 2.0 * kClothRadius;
    Particle& vertex = cloth.particles[vertexIndex];
    const std::array<std::uint32_t, 3> indices{{
        triangle.first, triangle.second, triangle.third,
    }};
    const ClosestPoint closest = closestPointOnTriangle(
        vertex.position,
        cloth.particles[indices[0]].position,
        cloth.particles[indices[1]].position,
        cloth.particles[indices[2]].position
    );
    const Vec3 separation = closest.point - vertex.position;
    const double distance = length(separation);
    if (distance >= target) {
        return 0.0;
    }
    const Vec3 previousTrianglePoint =
        cloth.particles[indices[0]].previous * closest.barycentric[0] +
        cloth.particles[indices[1]].previous * closest.barycentric[1] +
        cloth.particles[indices[2]].previous * closest.barycentric[2];
    const Vec3 normal = selfContactNormal(
        separation,
        cloth.particles[indices[1]].position -
            cloth.particles[indices[0]].position,
        cloth.particles[indices[2]].position -
            cloth.particles[indices[0]].position,
        vertex.previous - previousTrianglePoint
    );
    double denominator = vertex.inverseMass;
    for (std::size_t index = 0; index < 3; ++index) {
        denominator += cloth.particles[indices[index]].inverseMass *
            closest.barycentric[index] * closest.barycentric[index];
    }
    if (denominator <= 0.0) {
        return target - distance;
    }
    const double lambda = (target - distance) / denominator;
    vertex.position -= normal * (vertex.inverseMass * lambda);
    for (std::size_t index = 0; index < 3; ++index) {
        cloth.particles[indices[index]].position += normal *
            (cloth.particles[indices[index]].inverseMass *
             closest.barycentric[index] * lambda);
    }
    ++contactCount;
    return target - distance;
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
    std::uint64_t& contactCount
) {
    constexpr double target = 2.0 * kClothRadius;
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
    ++contactCount;
    return removedAdvance;
}

double solveEdgeEdge(
    ClothModel& cloth,
    const Edge first,
    const Edge second,
    std::uint64_t& contactCount
) {
    constexpr double target = 2.0 * kClothRadius;
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

CollisionBounds vertexBounds(
    const Particle& particle,
    const bool swept,
    const double expansion
) {
    CollisionBounds bounds;
    includePoint(bounds, particle.position);
    if (swept) {
        includePoint(bounds, particle.previous);
    }
    expandBounds(bounds, expansion);
    return bounds;
}

CollisionBounds triangleBounds(
    const ClothModel& cloth,
    const Triangle triangle,
    const bool swept,
    const double expansion
) {
    CollisionBounds bounds;
    for (const std::uint32_t index : {
        triangle.first, triangle.second, triangle.third,
    }) {
        includePoint(bounds, cloth.particles[index].position);
        if (swept) {
            includePoint(bounds, cloth.particles[index].previous);
        }
    }
    expandBounds(bounds, expansion);
    return bounds;
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

using CollisionGrid = std::unordered_map<
    std::uint64_t,
    std::vector<std::uint32_t>
>;

void insertCollisionBounds(
    CollisionGrid& grid,
    const CollisionBounds& bounds,
    const double inverseCell,
    const std::uint32_t primitive
) {
    const int minimumX = static_cast<int>(
        std::floor(bounds.minimum.x * inverseCell)
    );
    const int minimumY = static_cast<int>(
        std::floor(bounds.minimum.y * inverseCell)
    );
    const int minimumZ = static_cast<int>(
        std::floor(bounds.minimum.z * inverseCell)
    );
    const int maximumX = static_cast<int>(
        std::floor(bounds.maximum.x * inverseCell)
    );
    const int maximumY = static_cast<int>(
        std::floor(bounds.maximum.y * inverseCell)
    );
    const int maximumZ = static_cast<int>(
        std::floor(bounds.maximum.z * inverseCell)
    );
    for (int z = minimumZ; z <= maximumZ; ++z) {
        for (int y = minimumY; y <= maximumY; ++y) {
            for (int x = minimumX; x <= maximumX; ++x) {
                grid[spatialKey(x, y, z)].push_back(primitive);
            }
        }
    }
}

void gatherCollisionCandidates(
    const CollisionGrid& grid,
    const CollisionBounds& bounds,
    const double inverseCell,
    std::vector<std::uint32_t>& stamps,
    const std::uint32_t stamp,
    std::vector<std::uint32_t>& candidates
) {
    candidates.clear();
    const int minimumX = static_cast<int>(
        std::floor(bounds.minimum.x * inverseCell)
    );
    const int minimumY = static_cast<int>(
        std::floor(bounds.minimum.y * inverseCell)
    );
    const int minimumZ = static_cast<int>(
        std::floor(bounds.minimum.z * inverseCell)
    );
    const int maximumX = static_cast<int>(
        std::floor(bounds.maximum.x * inverseCell)
    );
    const int maximumY = static_cast<int>(
        std::floor(bounds.maximum.y * inverseCell)
    );
    const int maximumZ = static_cast<int>(
        std::floor(bounds.maximum.z * inverseCell)
    );
    for (int z = minimumZ; z <= maximumZ; ++z) {
        for (int y = minimumY; y <= maximumY; ++y) {
            for (int x = minimumX; x <= maximumX; ++x) {
                const auto found = grid.find(spatialKey(x, y, z));
                if (found == grid.end()) {
                    continue;
                }
                for (const std::uint32_t primitive : found->second) {
                    if (stamps[primitive] == stamp) {
                        continue;
                    }
                    stamps[primitive] = stamp;
                    candidates.push_back(primitive);
                }
            }
        }
    }
}

double solvePrimitiveSelfCollision(
    ClothModel& cloth,
    const bool swept,
    Metrics& metrics
) {
    constexpr double target = 2.0 * kClothRadius;
    constexpr double cellSize = 5.0 * target;
    constexpr double inverseCell = 1.0 / cellSize;
    CollisionGrid triangleGrid;
    triangleGrid.reserve(cloth.triangles.size() * 3u);
    for (std::uint32_t index = 0; index < cloth.triangles.size(); ++index) {
        insertCollisionBounds(
            triangleGrid,
            triangleBounds(cloth, cloth.triangles[index], swept, target),
            inverseCell,
            index
        );
    }
    double maximum = 0.0;
    std::uint64_t vertexContacts = 0u;
    std::vector<std::uint32_t> triangleStamps(cloth.triangles.size(), 0u);
    std::vector<std::uint32_t> candidates;
    candidates.reserve(64u);
    for (std::uint32_t vertex = 0; vertex < cloth.particles.size(); ++vertex) {
        gatherCollisionCandidates(
            triangleGrid,
            vertexBounds(cloth.particles[vertex], swept, target),
            inverseCell,
            triangleStamps,
            vertex + 1u,
            candidates
        );
        for (const std::uint32_t triangleIndex : candidates) {
            const Triangle triangle = cloth.triangles[triangleIndex];
            if (vertexTriangleLocal(cloth, vertex, triangle)) {
                continue;
            }
            maximum = std::max(
                maximum,
                swept
                    ? solveSweptVertexTriangle(
                        cloth, vertex, triangle, vertexContacts
                    )
                    : solveVertexTriangle(
                        cloth, vertex, triangle, vertexContacts
                    )
            );
        }
    }
    if (swept) {
        metrics.sweptVertexTriangleSelfContacts += vertexContacts;
    } else {
        metrics.vertexTriangleSelfContacts += vertexContacts;
    }
    metrics.selfContacts += vertexContacts;

    CollisionGrid edgeGrid;
    edgeGrid.reserve(cloth.collisionEdges.size() * 3u);
    for (std::uint32_t index = 0;
         index < cloth.collisionEdges.size();
         ++index) {
        insertCollisionBounds(
            edgeGrid,
            edgeBounds(cloth, cloth.collisionEdges[index], swept, target),
            inverseCell,
            index
        );
    }
    std::uint64_t edgeContacts = 0u;
    std::vector<std::uint32_t> edgeStamps(cloth.collisionEdges.size(), 0u);
    for (std::uint32_t firstIndex = 0;
         firstIndex < cloth.collisionEdges.size();
         ++firstIndex) {
        const Edge first = cloth.collisionEdges[firstIndex];
        gatherCollisionCandidates(
            edgeGrid,
            edgeBounds(cloth, first, swept, target),
            inverseCell,
            edgeStamps,
            firstIndex + 1u,
            candidates
        );
        for (const std::uint32_t secondIndex : candidates) {
            if (secondIndex <= firstIndex) {
                continue;
            }
            const Edge second = cloth.collisionEdges[secondIndex];
            if (edgePairLocal(cloth, first, second)) {
                continue;
            }
            maximum = std::max(
                maximum,
                swept
                    ? solveSweptEdgeEdge(
                        cloth, first, second, edgeContacts
                    )
                    : solveEdgeEdge(cloth, first, second, edgeContacts)
            );
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

double measurePrimitiveSelfPenetration(const ClothModel& cloth) {
    constexpr double target = 2.0 * kClothRadius;
    constexpr double cellSize = 5.0 * target;
    constexpr double inverseCell = 1.0 / cellSize;
    CollisionGrid triangleGrid;
    triangleGrid.reserve(cloth.triangles.size() * 3u);
    for (std::uint32_t index = 0; index < cloth.triangles.size(); ++index) {
        insertCollisionBounds(
            triangleGrid,
            triangleBounds(cloth, cloth.triangles[index], false, target),
            inverseCell,
            index
        );
    }
    double maximumPenetration = 0.0;
    std::vector<std::uint32_t> stamps(cloth.triangles.size(), 0u);
    std::vector<std::uint32_t> candidates;
    candidates.reserve(64u);
    for (std::uint32_t vertex = 0; vertex < cloth.particles.size(); ++vertex) {
        gatherCollisionCandidates(
            triangleGrid,
            vertexBounds(cloth.particles[vertex], false, target),
            inverseCell,
            stamps,
            vertex + 1u,
            candidates
        );
        for (const std::uint32_t triangleIndex : candidates) {
            const Triangle triangle = cloth.triangles[triangleIndex];
            if (vertexTriangleLocal(cloth, vertex, triangle)) {
                continue;
            }
            const ClosestPoint closest = closestPointOnTriangle(
                cloth.particles[vertex].position,
                cloth.particles[triangle.first].position,
                cloth.particles[triangle.second].position,
                cloth.particles[triangle.third].position
            );
            maximumPenetration = std::max(
                maximumPenetration,
                target - length(
                    closest.point - cloth.particles[vertex].position
                )
            );
        }
    }
    CollisionGrid edgeGrid;
    edgeGrid.reserve(cloth.collisionEdges.size() * 3u);
    for (std::uint32_t index = 0;
         index < cloth.collisionEdges.size();
         ++index) {
        insertCollisionBounds(
            edgeGrid,
            edgeBounds(cloth, cloth.collisionEdges[index], false, target),
            inverseCell,
            index
        );
    }
    stamps.assign(cloth.collisionEdges.size(), 0u);
    for (std::uint32_t firstIndex = 0;
         firstIndex < cloth.collisionEdges.size();
         ++firstIndex) {
        const Edge first = cloth.collisionEdges[firstIndex];
        gatherCollisionCandidates(
            edgeGrid,
            edgeBounds(cloth, first, false, target),
            inverseCell,
            stamps,
            firstIndex + 1u,
            candidates
        );
        for (const std::uint32_t secondIndex : candidates) {
            if (secondIndex <= firstIndex) {
                continue;
            }
            const Edge second = cloth.collisionEdges[secondIndex];
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
    constexpr double target = 2.0 * kClothRadius;
    constexpr double inverseCell = 1.0 / target;
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
        } else if (constraint.kind == DistanceKind::shear) {
            metrics.maximumShearStrain = std::max(metrics.maximumShearStrain, strain);
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
    for (const BendConstraint& bend : cloth.bends) {
        const double angle = signedDihedral(
            cloth.particles[bend.edgeFirst].position,
            cloth.particles[bend.edgeSecond].position,
            cloth.particles[bend.oppositeFirst].position,
            cloth.particles[bend.oppositeSecond].position
        );
        metrics.maximumBendError = std::max(
            metrics.maximumBendError,
            std::abs(wrapAngle(angle - bend.restAngle))
        );
    }
    for (const Triangle triangle : cloth.triangles) {
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

SimulationResult simulate(
    const std::uint32_t steps,
    const double frameTimestep,
    const std::uint32_t substeps,
    const std::uint32_t iterations,
    const Scenario scenario,
    const std::vector<std::uint32_t>* captureSteps = nullptr,
    std::vector<SimulationResult>* captures = nullptr
) {
    SimulationResult result;
    result.scenario = scenario;
    result.cloth = makeCloth(scenario);
    result.balls = makeBalls(scenario);
    const double timestep = frameTimestep / static_cast<double>(substeps);
    const Vec3 gravity{0.0, 0.0, -9.81};
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
        for (std::uint32_t substep = 0; substep < substeps; ++substep) {
            std::array<BallPairContactImpulse, kBallPairCount> pairContacts{};
            std::vector<BallTriangleContactImpulse> triangleContacts(
                result.balls.size() * result.cloth.triangles.size()
            );
            std::array<double, kFruitCount> groundNormalImpulses{};
            if (scenario != Scenario::grounded) {
                const double time = (
                    static_cast<double>(step * substeps + substep + 1u) *
                    timestep
                );
                updateGrip(result.cloth, scenario, time);
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
                particle.velocity = particle.velocity * std::exp(-0.45 * timestep);
                particle.position += particle.velocity * timestep;
            }
            for (Ball& ball : result.balls) {
                ball.previous = ball.position;
                ball.velocity += gravity * timestep;
                ball.velocity = ball.velocity * std::exp(-0.08 * timestep);
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
            for (BendConstraint& constraint : result.cloth.bends) {
                constraint.lambda = 0.0;
            }
            for (GripConstraint& constraint : result.cloth.grips) {
                constraint.lambda = {};
            }

            result.metrics.maximumSweptSelfAdvance = std::max(
                result.metrics.maximumSweptSelfAdvance,
                solvePrimitiveSelfCollision(result.cloth, true, result.metrics)
            );

            for (std::size_t ballIndex = 0;
                 ballIndex < result.balls.size();
                 ++ballIndex) {
                for (std::size_t triangleIndex = 0;
                     triangleIndex < result.cloth.triangles.size();
                     ++triangleIndex) {
                    result.metrics.maximumSweptBallAdvance = std::max(
                        result.metrics.maximumSweptBallAdvance,
                        solveSweptBallTriangle(
                            result.cloth.particles,
                            result.cloth.triangles[triangleIndex],
                            result.balls[ballIndex],
                            timestep,
                            triangleContacts[
                                ballIndex * result.cloth.triangles.size() +
                                triangleIndex
                            ],
                            result.metrics.sweptBallTriangleContacts
                        )
                    );
                }
            }

            for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
                for (DistanceConstraint& constraint : result.cloth.distances) {
                    solveDistance(result.cloth.particles, constraint, timestep);
                }
                if (iteration % 2u == 0u) {
                    for (BendConstraint& constraint : result.cloth.bends) {
                        solveBend(result.cloth.particles, constraint, timestep);
                    }
                }
                for (GripConstraint& constraint : result.cloth.grips) {
                    solveGrip(
                        result.cloth.particles,
                        constraint,
                        result.cloth.gripTarget + constraint.targetOffset,
                        timestep
                    );
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
                for (std::size_t ballIndex = 0;
                     ballIndex < result.balls.size();
                     ++ballIndex) {
                    Ball& ball = result.balls[ballIndex];
                    for (std::size_t triangleIndex = 0;
                         triangleIndex < result.cloth.triangles.size();
                         ++triangleIndex) {
                        const Triangle triangle =
                            result.cloth.triangles[triangleIndex];
                        const double penetration = solveBallTriangle(
                            result.cloth.particles,
                            triangle,
                            ball,
                            timestep,
                            triangleContacts[
                                ballIndex * result.cloth.triangles.size() +
                                triangleIndex
                            ],
                            result.metrics.ballTriangleContacts
                        );
                        result.metrics.maximumBallPenetration = std::max(
                            result.metrics.maximumBallPenetration,
                            penetration
                        );
                        result.metrics.maximumBallPenetrationByFruit[ballIndex] =
                            std::max(
                                result.metrics.maximumBallPenetrationByFruit[ballIndex],
                                penetration
                            );
                    }
                }
                result.metrics.maximumSelfPenetration = std::max(
                    result.metrics.maximumSelfPenetration,
                    solveSelfCollision(
                        result.cloth,
                        result.metrics.selfContacts
                    )
                );
                if (iteration + 1u == iterations) {
                    const bool endpointCertificate =
                        (step == 0u && substep == 0u) ||
                        (step + 1u == steps && substep + 1u == substeps);
                    const std::uint32_t couplingCycles =
                        endpointCertificate ? 3u : 2u;
                    for (std::uint32_t pass = 0;
                         pass < couplingCycles;
                         ++pass) {
                        result.metrics.maximumSelfPenetration = std::max(
                            result.metrics.maximumSelfPenetration,
                            solvePrimitiveSelfCollision(
                                result.cloth,
                                false,
                                result.metrics
                            )
                        );
                        result.metrics.maximumStrainLimitCorrection = std::max(
                            result.metrics.maximumStrainLimitCorrection,
                            solveStrainLimits(
                                result.cloth.particles,
                                result.cloth.distances
                            )
                        );
                    }
                }
                if (scenario != Scenario::spin) {
                    result.metrics.maximumGroundPenetration = std::max(
                        result.metrics.maximumGroundPenetration,
                        solveGround(
                            result.cloth.particles,
                            result.balls,
                            timestep,
                            groundNormalImpulses
                        )
                    );
                }
            }
            for (Particle& particle : result.cloth.particles) {
                if (particle.inverseMass == 0.0) {
                    particle.position = particle.rest;
                    particle.velocity =
                        (particle.position - particle.previous) / timestep;
                } else {
                    particle.velocity =
                        (particle.position - particle.previous) / timestep;
                    if (particle.position.z <= kClothRadius + 1.0e-6) {
                        const double friction = std::exp(-12.0 * timestep);
                        particle.velocity.x *= friction;
                        particle.velocity.y *= friction;
                    }
                }
            }
            for (Ball& ball : result.balls) {
                ball.velocity = (ball.position - ball.previous) / timestep;
            }
            applyBallClothFriction(
                result.cloth.particles,
                result.cloth.triangles,
                result.balls,
                triangleContacts,
                result.metrics
            );
            applyBallPairFriction(result.balls, pairContacts, result.metrics);
            applyBallGroundFriction(
                result.balls,
                groundNormalImpulses,
                timestep,
                result.metrics,
                kFruitRollingResistanceRate
            );
            for (Ball& ball : result.balls) {
                ball.angularVelocity = ball.angularVelocity *
                    std::exp(-0.02 * timestep);
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
        updateMetrics(result.cloth, result.balls, result.metrics);
        if (scenario != Scenario::grounded) {
            for (std::size_t ballIndex = 0;
                 ballIndex < result.balls.size();
                 ++ballIndex) {
                if (length(
                    result.balls[ballIndex].position -
                    result.cloth.particles[result.cloth.bottomCenter].position
                ) > 0.45) {
                    result.metrics.releasedMask |= 1u << ballIndex;
                }
            }
        }
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
        if (scenario != Scenario::grounded && length(
            ball.position -
            result.cloth.particles[result.cloth.bottomCenter].position
        ) > 0.45) {
            result.metrics.releasedMask |= 1u << ballIndex;
        }
    }
    result.metrics.finalPrimitiveSelfPenetration =
        measurePrimitiveSelfPenetration(result.cloth);
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
    return hash;
}

void dumpOBJ(const std::string& path, const SimulationResult& result) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open OBJ output: " + path);
    }
    output << std::setprecision(9);
    output << "# Numi Solver dense cloth bag reference\n";
    output << "# vertices " << result.cloth.particles.size()
           << " triangles " << result.cloth.triangles.size() << '\n';
    for (const Particle& particle : result.cloth.particles) {
        output << "v " << particle.position.x << ' '
               << particle.position.y << ' ' << particle.position.z << '\n';
    }
    for (const Triangle triangle : result.cloth.triangles) {
        output << "f " << triangle.first + 1u << ' '
               << triangle.second + 1u << ' '
               << triangle.third + 1u << '\n';
    }
    if (result.scenario != Scenario::grounded) {
        output << "# grip center " << result.cloth.gripTarget.x << ' '
               << result.cloth.gripTarget.y << ' '
               << result.cloth.gripTarget.z << '\n';
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

bool acceptable(const SimulationResult& result, const bool deterministic) {
    bool allFinite = true;
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
    double clothMass = 0.0;
    for (const Particle& particle : result.cloth.particles) {
        clothMass += particle.mass;
    }
    return allFinite && deterministic && result.metrics.escapedMask == 0u &&
        result.metrics.spilledMask == 0u && groundValid &&
        std::abs(clothMass - kClothMass) < 1.0e-12 &&
        result.metrics.minimumTriangleArea > 1.0e-8 &&
        result.metrics.maximumWarpExtension < 0.30 &&
        result.metrics.maximumWarpCompression < 0.60 &&
        result.metrics.maximumWeftExtension < 0.30 &&
        result.metrics.maximumWeftCompression < 0.60 &&
        result.metrics.maximumBottomExtension < 0.30 &&
        result.metrics.maximumBottomCompression < 0.60 &&
        result.metrics.maximumShearStrain < 0.40 &&
        result.metrics.maximumBallPenetration < 0.010 &&
        result.metrics.maximumSelfPenetration < 2.0 * kClothRadius &&
        result.metrics.finalPrimitiveSelfPenetration < 2.0e-6 &&
        result.metrics.finalStrainLimitViolation < 2.0e-6 &&
        result.metrics.maximumSpeed < 30.0 &&
        result.metrics.maximumAngularSpeed < 200.0 &&
        result.metrics.maximumGripForce < 500.0 &&
        result.metrics.maximumFrictionConeRatio <= 1.0 + 1.0e-12;
}

}  // namespace

int main(int argc, char** argv) try {
    std::uint32_t steps = 120u;
    std::uint32_t substeps = 4u;
    std::uint32_t iterations = 12u;
    double timestep = 1.0 / 120.0;
    std::string dumpPath;
    std::string framePrefix;
    std::uint32_t frameStride = 0u;
    bool rollingProbe = false;
    bool ccdProbe = false;
    bool selfCCDProbe = false;
    bool strainProbe = false;
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
        } else if (value == "--timestep" && argument + 1 < argc) {
            timestep = std::stod(argv[++argument]);
        } else if (value == "--dump-obj" && argument + 1 < argc) {
            dumpPath = argv[++argument];
        } else if (value == "--dump-frames" && argument + 1 < argc) {
            framePrefix = argv[++argument];
        } else if (value == "--dump-every") {
            nextUnsigned(frameStride);
        } else if (value == "--rolling-probe") {
            rollingProbe = true;
        } else if (value == "--ccd-probe") {
            ccdProbe = true;
        } else if (value == "--self-ccd-probe") {
            selfCCDProbe = true;
        } else if (value == "--strain-probe") {
            strainProbe = true;
        } else if (value == "--scenario" && argument + 1 < argc) {
            const std::string name = argv[++argument];
            if (name == "grounded") {
                scenario = Scenario::grounded;
            } else if (name == "spin") {
                scenario = Scenario::spin;
            } else if (name == "pickup") {
                scenario = Scenario::pickup;
            } else {
                throw std::invalid_argument("unknown scenario: " + name);
            }
        } else if (value == "--help") {
            std::cout << "usage: numi-solver-cloth-bag "
                         "[--steps N] [--substeps N] [--iterations N] "
                         "[--timestep DT] [--scenario grounded|spin|pickup] "
                         "[--dump-obj PATH] [--dump-frames PREFIX] "
                         "[--dump-every N] [--rolling-probe] [--ccd-probe] "
                         "[--self-ccd-probe] [--strain-probe]\n";
            return 0;
        } else {
            throw std::invalid_argument("unknown argument: " + value);
        }
    }
    if (rollingProbe) {
        return runRollingProbe() ? 0 : 1;
    }
    if (ccdProbe) {
        return runCCDProbe() ? 0 : 1;
    }
    if (selfCCDProbe) {
        return runSelfCCDProbe() ? 0 : 1;
    }
    if (strainProbe) {
        return runStrainProbe() ? 0 : 1;
    }
    if (steps == 0u || substeps == 0u || iterations == 0u ||
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
        framePrefix.empty() ? nullptr : &captures
    );
    const SimulationResult replay = simulate(
        steps, timestep, substeps, iterations, scenario
    );
    const std::uint64_t firstHash = hashResult(first);
    const std::uint64_t replayHash = hashResult(replay);
    const bool deterministic = firstHash == replayHash;
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
        scenario == Scenario::spin ? "spin" : "pickup";
    std::cout << "model=dense_cloth_reference"
              << " scenario=" << scenarioName
              << " nodes=" << first.cloth.particles.size()
              << " triangles=" << first.cloth.triangles.size()
              << " stretch_constraints=" << first.cloth.distances.size()
              << " bend_constraints=" << first.cloth.bends.size()
              << " balls=" << first.balls.size()
              << " cloth_mass_kg=" << clothMass
              << " fruit_mass_kg=" << fruitMass
              << " steps=" << steps
              << " substeps=" << substeps
              << " iterations=" << iterations
              << " simulated_seconds=" << steps * timestep << '\n';
    if (!captures.empty()) {
        std::cout << "captured_frames=" << captures.size()
                  << " frame_prefix=" << framePrefix << '\n';
    }
    std::cout << "max_warp_strain=" << metrics.maximumWarpStrain
              << " max_weft_strain=" << metrics.maximumWeftStrain
              << " max_shear_strain=" << metrics.maximumShearStrain
              << " max_bend_error=" << metrics.maximumBendError
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
    std::cout << "max_ball_penetration=" << metrics.maximumBallPenetration
              << " max_self_penetration=" << metrics.maximumSelfPenetration
              << " ball_triangle_contacts=" << metrics.ballTriangleContacts
              << " self_contacts=" << metrics.selfContacts
              << " escaped_mask=" << metrics.escapedMask
              << " spilled_mask=" << metrics.spilledMask
              << " released_mask=" << metrics.releasedMask << '\n';
    std::cout << "max_ball_penetration_by_fruit=";
    for (std::size_t index = 0;
         index < metrics.maximumBallPenetrationByFruit.size();
         ++index) {
        if (index != 0u) {
            std::cout << ',';
        }
        std::cout << metrics.maximumBallPenetrationByFruit[index];
    }
    std::cout << '\n';
    std::cout << "max_ground_penetration=" << metrics.maximumGroundPenetration
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
              << " max_grip_force=" << metrics.maximumGripForce
              << " max_grip_impulse=" << metrics.maximumGripImpulse
              << " max_friction_cone_ratio="
              << metrics.maximumFrictionConeRatio
              << " tangential_impulse="
              << metrics.accumulatedTangentialImpulse
              << " cloth_friction_contacts="
              << metrics.ballClothFrictionContacts
              << " pair_friction_contacts="
              << metrics.ballPairFrictionContacts
              << " ground_friction_contacts="
              << metrics.ballGroundFrictionContacts
              << " swept_ball_triangle_contacts="
              << metrics.sweptBallTriangleContacts
              << " vertex_triangle_self_contacts="
              << metrics.vertexTriangleSelfContacts
              << " edge_edge_self_contacts="
              << metrics.edgeEdgeSelfContacts
              << " swept_vertex_triangle_self_contacts="
              << metrics.sweptVertexTriangleSelfContacts
              << " swept_edge_edge_self_contacts="
              << metrics.sweptEdgeEdgeSelfContacts
              << " deterministic=" << std::boolalpha << deterministic
              << " state_hash=0x" << std::hex << firstHash << std::dec << '\n';
    const bool pass = acceptable(first, deterministic);
    std::cout << "result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << "numi-solver-cloth-bag: " << error.what() << '\n';
    return 2;
}
