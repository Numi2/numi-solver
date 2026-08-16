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

constexpr std::uint32_t kAround = 32;
constexpr std::uint32_t kLevels = 20;
constexpr double kClothRadius = 0.006;
constexpr double kBallRadius = 0.14;

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

struct Particle {
    Vec3 position{};
    Vec3 previous{};
    Vec3 velocity{};
    Vec3 rest{};
    double inverseMass{};
};

struct Ball {
    Vec3 position{};
    Vec3 previous{};
    Vec3 velocity{};
    double radius{kBallRadius};
    double inverseMass{1.0};
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

struct BendConstraint {
    std::uint32_t edgeFirst{};
    std::uint32_t edgeSecond{};
    std::uint32_t oppositeFirst{};
    std::uint32_t oppositeSecond{};
    double restAngle{};
    double compliance{};
    double lambda{};
};

struct ClothModel {
    std::vector<Particle> particles;
    std::vector<DistanceConstraint> distances;
    std::vector<Triangle> triangles;
    std::vector<BendConstraint> bends;
    std::uint32_t bottomCenter{};
};

struct Metrics {
    double maximumWarpStrain{};
    double maximumWeftStrain{};
    double maximumShearStrain{};
    double maximumBendError{};
    double maximumBallPenetration{};
    double maximumSelfPenetration{};
    double maximumAnchorError{};
    double minimumTriangleArea{std::numeric_limits<double>::infinity()};
    double maximumSpeed{};
    std::uint64_t ballTriangleContacts{};
    std::uint64_t selfContacts{};
    std::uint32_t escapedMask{};
};

struct SimulationResult {
    ClothModel cloth;
    std::array<Ball, 6> balls{};
    Metrics metrics{};
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
    const double radius =
        0.10 +
        0.30 * smoothstep(vertical / 0.28) +
        0.12 * smoothstep((vertical - 0.72) / 0.28);
    const double angle = 2.0 * std::numbers::pi *
        static_cast<double>(ring) / static_cast<double>(kAround) +
        0.08 * vertical;
    return {
        radius * std::cos(angle),
        radius * std::sin(angle),
        1.08 * vertical,
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

ClothModel makeCloth() {
    ClothModel model;
    model.particles.reserve(kAround * kLevels + 1u);
    for (std::uint32_t level = 0; level < kLevels; ++level) {
        for (std::uint32_t ring = 0; ring < kAround; ++ring) {
            const Vec3 position = authoredPosition(level, ring);
            const bool anchored = level + 1u == kLevels;
            model.particles.push_back({
                position,
                position,
                {},
                position,
                anchored ? 0.0 : 1.0 / 0.006,
            });
        }
    }
    model.bottomCenter = static_cast<std::uint32_t>(model.particles.size());
    const Vec3 center{0.0, 0.0, 0.0};
    model.particles.push_back({center, center, {}, center, 1.0 / 0.025});

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
                1.0e-8,
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
    for (std::uint32_t ring = 0; ring < kAround; ++ring) {
        addDistance(
            model.bottomCenter,
            nodeIndex(0u, ring),
            1.0e-8,
            DistanceKind::bottom
        );
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
    for (std::uint32_t ring = 0; ring < kAround; ++ring) {
        model.triangles.push_back({
            model.bottomCenter,
            nodeIndex(0u, ring + 1u),
            nodeIndex(0u, ring),
        });
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
                2.0e-4,
                0.0,
            });
            edges.erase(found);
        }
    }
    return model;
}

std::array<Ball, 6> makeBalls() {
    constexpr std::array<Vec3, 6> positions{{
        {0.18, 0.00, 0.78},
        {-0.09, 0.155885, 0.78},
        {-0.09, -0.155885, 0.78},
        {0.085, 0.147224, 0.44},
        {-0.17, 0.00, 0.44},
        {0.085, -0.147224, 0.44},
    }};
    std::array<Ball, 6> balls{};
    for (std::size_t index = 0; index < balls.size(); ++index) {
        balls[index].position = positions[index];
        balls[index].previous = positions[index];
    }
    return balls;
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

double solveBallTriangle(
    std::vector<Particle>& particles,
    const Triangle triangle,
    Ball& ball,
    std::uint64_t& contactCount
) {
    const std::array<std::uint32_t, 3> indices{{
        triangle.first,
        triangle.second,
        triangle.third,
    }};
    const ClosestPoint closest = closestPointOnTriangle(
        ball.position,
        particles[indices[0]].position,
        particles[indices[1]].position,
        particles[indices[2]].position
    );
    Vec3 separation = closest.point - ball.position;
    double distance = length(separation);
    const double target = ball.radius + kClothRadius;
    if (distance >= target) {
        return 0.0;
    }
    if (distance < 1.0e-10) {
        const Vec3 first = particles[indices[0]].position;
        const Vec3 second = particles[indices[1]].position;
        const Vec3 third = particles[indices[2]].position;
        separation = normalized(cross(second - first, third - first));
        distance = 0.0;
    }
    const Vec3 normal = normalized(separation);
    double denominator = ball.inverseMass;
    for (std::size_t index = 0; index < 3; ++index) {
        denominator += particles[indices[index]].inverseMass *
            closest.barycentric[index] * closest.barycentric[index];
    }
    if (denominator <= 0.0) {
        return target - distance;
    }
    const double correction = (target - distance) / denominator;
    for (std::size_t index = 0; index < 3; ++index) {
        particles[indices[index]].position += normal *
            (particles[indices[index]].inverseMass *
             closest.barycentric[index] * correction);
    }
    ball.position -= normal * (ball.inverseMass * correction);
    ++contactCount;
    return target - distance;
}

double solveBallPair(Ball& first, Ball& second) {
    const Vec3 difference = second.position - first.position;
    const double currentLength = length(difference);
    const double target = first.radius + second.radius;
    if (currentLength >= target || currentLength < 1.0e-12) {
        return 0.0;
    }
    const double denominator = first.inverseMass + second.inverseMass;
    const Vec3 correction = difference *
        ((target - currentLength) / (currentLength * denominator));
    first.position -= correction * first.inverseMass;
    second.position += correction * second.inverseMass;
    return target - currentLength;
}

bool localTopologyPair(const std::uint32_t first, const std::uint32_t second) {
    const std::uint32_t gridCount = kAround * kLevels;
    if (first >= gridCount || second >= gridCount) {
        const std::uint32_t other = first >= gridCount ? second : first;
        return other < kAround;
    }
    const std::uint32_t firstLevel = first / kAround;
    const std::uint32_t secondLevel = second / kAround;
    const std::uint32_t firstRing = first % kAround;
    const std::uint32_t secondRing = second % kAround;
    const std::uint32_t ringDifference = std::min(
        (firstRing + kAround - secondRing) % kAround,
        (secondRing + kAround - firstRing) % kAround
    );
    const std::uint32_t levelDifference = firstLevel > secondLevel
        ? firstLevel - secondLevel
        : secondLevel - firstLevel;
    return levelDifference <= 1u && ringDifference <= 2u;
}

std::uint64_t spatialKey(const int x, const int y, const int z) {
    constexpr int bias = 1 << 20;
    const auto encode = [&](const int value) {
        return static_cast<std::uint64_t>(value + bias) & 0x1fffffu;
    };
    return encode(x) | (encode(y) << 21u) | (encode(z) << 42u);
}

double solveSelfCollision(
    std::vector<Particle>& particles,
    std::uint64_t& contactCount
) {
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
                            localTopologyPair(firstIndex, secondIndex)) {
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
    const std::array<Ball, 6>& balls,
    Metrics& metrics
) {
    for (const DistanceConstraint& constraint : cloth.distances) {
        const double currentLength = length(
            cloth.particles[constraint.second].position -
            cloth.particles[constraint.first].position
        );
        const double strain = std::abs(
            currentLength / constraint.restLength - 1.0
        );
        if (constraint.kind == DistanceKind::warp) {
            metrics.maximumWarpStrain = std::max(metrics.maximumWarpStrain, strain);
        } else if (constraint.kind == DistanceKind::weft) {
            metrics.maximumWeftStrain = std::max(metrics.maximumWeftStrain, strain);
        } else if (constraint.kind == DistanceKind::shear) {
            metrics.maximumShearStrain = std::max(metrics.maximumShearStrain, strain);
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
    for (std::uint32_t ring = 0; ring < kAround; ++ring) {
        const Particle& anchor = cloth.particles[nodeIndex(kLevels - 1u, ring)];
        metrics.maximumAnchorError = std::max(
            metrics.maximumAnchorError,
            length(anchor.position - anchor.rest)
        );
    }
    for (const Particle& particle : cloth.particles) {
        metrics.maximumSpeed = std::max(metrics.maximumSpeed, length(particle.velocity));
    }
    for (const Ball& ball : balls) {
        metrics.maximumSpeed = std::max(metrics.maximumSpeed, length(ball.velocity));
    }
}

SimulationResult simulate(
    const std::uint32_t steps,
    const double frameTimestep,
    const std::uint32_t substeps,
    const std::uint32_t iterations
) {
    SimulationResult result;
    result.cloth = makeCloth();
    result.balls = makeBalls();
    const double timestep = frameTimestep / static_cast<double>(substeps);
    const Vec3 gravity{0.0, 0.0, -9.81};

    for (std::uint32_t step = 0; step < steps; ++step) {
        for (std::uint32_t substep = 0; substep < substeps; ++substep) {
            for (Particle& particle : result.cloth.particles) {
                particle.previous = particle.position;
                if (particle.inverseMass == 0.0) {
                    particle.position = particle.rest;
                    particle.velocity = {};
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
            for (DistanceConstraint& constraint : result.cloth.distances) {
                constraint.lambda = 0.0;
            }
            for (BendConstraint& constraint : result.cloth.bends) {
                constraint.lambda = 0.0;
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
                for (std::size_t first = 0; first < result.balls.size(); ++first) {
                    for (std::size_t second = first + 1u;
                         second < result.balls.size();
                         ++second) {
                        solveBallPair(result.balls[first], result.balls[second]);
                    }
                }
                for (Ball& ball : result.balls) {
                    for (const Triangle triangle : result.cloth.triangles) {
                        result.metrics.maximumBallPenetration = std::max(
                            result.metrics.maximumBallPenetration,
                            solveBallTriangle(
                                result.cloth.particles,
                                triangle,
                                ball,
                                result.metrics.ballTriangleContacts
                            )
                        );
                    }
                }
                result.metrics.maximumSelfPenetration = std::max(
                    result.metrics.maximumSelfPenetration,
                    solveSelfCollision(
                        result.cloth.particles,
                        result.metrics.selfContacts
                    )
                );
            }
            for (Particle& particle : result.cloth.particles) {
                if (particle.inverseMass == 0.0) {
                    particle.position = particle.rest;
                    particle.velocity = {};
                } else {
                    particle.velocity =
                        (particle.position - particle.previous) / timestep;
                }
            }
            for (Ball& ball : result.balls) {
                ball.velocity = (ball.position - ball.previous) / timestep;
            }
        }
        updateMetrics(result.cloth, result.balls, result.metrics);
    }

    for (std::size_t ballIndex = 0; ballIndex < result.balls.size(); ++ballIndex) {
        const Ball& ball = result.balls[ballIndex];
        const double radial = std::hypot(ball.position.x, ball.position.y);
        if (!finite(ball.position) || ball.position.z > 1.08 + ball.radius ||
            ball.position.z < -1.5 || radial > 0.70) {
            result.metrics.escapedMask |= 1u << ballIndex;
        }
    }
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
    for (std::size_t index = 0; index < result.balls.size(); ++index) {
        const Ball& ball = result.balls[index];
        output << "# ball " << index << " center "
               << ball.position.x << ' ' << ball.position.y << ' '
               << ball.position.z << " radius " << ball.radius << '\n';
    }
}

bool acceptable(const SimulationResult& result, const bool deterministic) {
    bool allFinite = true;
    for (const Particle& particle : result.cloth.particles) {
        allFinite = allFinite && finite(particle.position) && finite(particle.velocity);
    }
    for (const Ball& ball : result.balls) {
        allFinite = allFinite && finite(ball.position) && finite(ball.velocity);
    }
    return allFinite && deterministic && result.metrics.escapedMask == 0u &&
        result.metrics.maximumAnchorError <= 1.0e-12 &&
        result.metrics.minimumTriangleArea > 1.0e-8 &&
        result.metrics.maximumWarpStrain < 0.30 &&
        result.metrics.maximumWeftStrain < 0.30 &&
        result.metrics.maximumShearStrain < 0.40 &&
        result.metrics.maximumBendError < 0.50 &&
        result.metrics.maximumBallPenetration < 0.010;
}

}  // namespace

int main(int argc, char** argv) try {
    std::uint32_t steps = 120u;
    std::uint32_t substeps = 2u;
    std::uint32_t iterations = 12u;
    double timestep = 1.0 / 120.0;
    std::string dumpPath;
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
        } else if (value == "--help") {
            std::cout << "usage: numi-solver-cloth-bag "
                         "[--steps N] [--substeps N] [--iterations N] "
                         "[--timestep DT] [--dump-obj PATH]\n";
            return 0;
        } else {
            throw std::invalid_argument("unknown argument: " + value);
        }
    }
    if (steps == 0u || substeps == 0u || iterations == 0u ||
        !std::isfinite(timestep) || timestep <= 0.0) {
        throw std::invalid_argument("simulation controls must be positive");
    }

    const SimulationResult first = simulate(steps, timestep, substeps, iterations);
    const SimulationResult replay = simulate(steps, timestep, substeps, iterations);
    const std::uint64_t firstHash = hashResult(first);
    const std::uint64_t replayHash = hashResult(replay);
    const bool deterministic = firstHash == replayHash;
    if (!dumpPath.empty()) {
        dumpOBJ(dumpPath, first);
    }

    const Metrics& metrics = first.metrics;
    std::cout << std::fixed << std::setprecision(9);
    std::cout << "model=dense_cloth_reference"
              << " nodes=" << first.cloth.particles.size()
              << " triangles=" << first.cloth.triangles.size()
              << " stretch_constraints=" << first.cloth.distances.size()
              << " bend_constraints=" << first.cloth.bends.size()
              << " balls=" << first.balls.size()
              << " steps=" << steps
              << " substeps=" << substeps
              << " iterations=" << iterations
              << " simulated_seconds=" << steps * timestep << '\n';
    std::cout << "max_warp_strain=" << metrics.maximumWarpStrain
              << " max_weft_strain=" << metrics.maximumWeftStrain
              << " max_shear_strain=" << metrics.maximumShearStrain
              << " max_bend_error=" << metrics.maximumBendError
              << " min_triangle_area=" << metrics.minimumTriangleArea << '\n';
    std::cout << "max_ball_penetration=" << metrics.maximumBallPenetration
              << " max_self_penetration=" << metrics.maximumSelfPenetration
              << " ball_triangle_contacts=" << metrics.ballTriangleContacts
              << " self_contacts=" << metrics.selfContacts
              << " escaped_mask=" << metrics.escapedMask << '\n';
    std::cout << "max_anchor_error=" << metrics.maximumAnchorError
              << " max_speed=" << metrics.maximumSpeed
              << " deterministic=" << std::boolalpha << deterministic
              << " state_hash=0x" << std::hex << firstHash << std::dec << '\n';
    const bool pass = acceptable(first, deterministic);
    std::cout << "result=" << (pass ? "PASS" : "FAIL") << '\n';
    return pass ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << "numi-solver-cloth-bag: " << error.what() << '\n';
    return 2;
}
