#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/temporal_cone_island.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef NUMI_TEMPORAL_CONE_METALLIB
#error "NUMI_TEMPORAL_CONE_METALLIB must name the built solver metallib"
#endif

namespace {

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<Vec3, 3>;

constexpr double kMatrixFloor = 1.0e-10;
constexpr double kConditioning = 1.0e-2;
constexpr double kConeEpsilon = 1.0e-7;
constexpr std::size_t kMaxContacts =
    NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS;
constexpr std::size_t kMaxRows = NUMI_TEMPORAL_CONE_ISLAND_MAX_ROWS;
constexpr std::size_t kMatrixElements =
    NUMI_TEMPORAL_CONE_ISLAND_MATRIX_ELEMENTS;

struct Batch {
    std::vector<NumiTemporalConeIslandHeader> headers;
    std::vector<float> matrices;
    std::vector<NumiTemporalConeIslandContact> contacts;
};

struct StreamBatch {
    std::vector<NumiTemporalConeStreamHeader> headers;
    std::vector<std::uint32_t> rowOffsets;
    std::vector<std::uint32_t> columnIndices;
    std::vector<float> blockValues;
    std::vector<NumiTemporalConeIslandContact> contacts;
};

struct GPUResult {
    std::vector<mr_float4> impulses;
    std::vector<NumiTemporalConeIslandStatus> statuses;
    double seconds = 0.0;
};

struct OracleResult {
    std::vector<Vec3> impulses;
    std::uint32_t status = NUMI_TEMPORAL_CONE_ISLAND_SUCCESS;
    std::uint32_t iterations = 0u;
    double kktResidual = 0.0;
    double coneViolation = 0.0;
    double objective = 0.0;
};

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
    const std::uint32_t z,
    const std::uint32_t w
) {
    return {x, y, z, w};
}

double ellipseFunction(
    const std::array<double, 2>& tangent,
    const std::array<double, 2>& radiiSquared,
    const double multiplier
) {
    double value = -1.0;
    for (std::size_t axis = 0; axis < 2; ++axis) {
        if (radiiSquared[axis] <= kMatrixFloor) {
            continue;
        }
        const double denominator = radiiSquared[axis] + multiplier;
        value += tangent[axis] * tangent[axis] * radiiSquared[axis] /
            (denominator * denominator);
    }
    return value;
}

std::array<double, 2> projectEllipse(
    const std::array<double, 2>& tangent,
    const std::array<double, 2>& radii
) {
    const std::array<double, 2> squared{{
        radii[0] * radii[0], radii[1] * radii[1],
    }};
    double normalized = 0.0;
    for (std::size_t axis = 0; axis < 2; ++axis) {
        if (squared[axis] > kMatrixFloor) {
            normalized += tangent[axis] * tangent[axis] / squared[axis];
        } else if (tangent[axis] != 0.0) {
            normalized = std::numeric_limits<double>::infinity();
        }
    }
    if (normalized <= 1.0) {
        return tangent;
    }
    const bool activeX = squared[0] > kMatrixFloor;
    const bool activeY = squared[1] > kMatrixFloor;
    if (activeX != activeY) {
        return {{
            activeX ? std::clamp(tangent[0], -radii[0], radii[0]) : 0.0,
            activeY ? std::clamp(tangent[1], -radii[1], radii[1]) : 0.0,
        }};
    }
    double lower = 0.0;
    double upper = std::max({
        1.0,
        std::abs(tangent[0]) * std::max(radii[0], 1.0),
        std::abs(tangent[1]) * std::max(radii[1], 1.0),
    });
    while (ellipseFunction(tangent, squared, upper) > 0.0) {
        upper *= 2.0;
    }
    for (std::uint32_t iteration = 0u; iteration < 100u; ++iteration) {
        const double middle = 0.5 * (lower + upper);
        if (ellipseFunction(tangent, squared, middle) > 0.0) {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    const double multiplier = 0.5 * (lower + upper);
    return {{
        squared[0] > kMatrixFloor
            ? tangent[0] * squared[0] / (squared[0] + multiplier)
            : 0.0,
        squared[1] > kMatrixFloor
            ? tangent[1] * squared[1] / (squared[1] + multiplier)
            : 0.0,
    }};
}

double anisotropicFunction(
    const Vec3& value,
    const std::array<double, 2>& frictionSquared,
    const double multiplier
) {
    const double normal = value[0] + multiplier;
    double result = -1.0;
    for (std::size_t axis = 0; axis < 2; ++axis) {
        if (frictionSquared[axis] <= kMatrixFloor) {
            continue;
        }
        const double denominator =
            frictionSquared[axis] * normal + multiplier;
        if (!(denominator > kMatrixFloor)) {
            return std::numeric_limits<double>::infinity();
        }
        result += value[axis + 1] * value[axis + 1] *
            frictionSquared[axis] / (denominator * denominator);
    }
    return result;
}

Vec3 projectCone(
    const Vec3& value,
    const double authoredFrictionU,
    const double authoredFrictionV,
    const double maximumNormal
) {
    const std::array<double, 2> friction{{
        std::max(authoredFrictionU, 0.0),
        std::max(authoredFrictionV, 0.0),
    }};
    if (friction[0] <= kConeEpsilon &&
        friction[1] <= kConeEpsilon) {
        return {{
            maximumNormal > 0.0
                ? std::clamp(value[0], 0.0, maximumNormal)
                : std::max(value[0], 0.0),
            0.0,
            0.0,
        }};
    }

    double normalized = 0.0;
    for (std::size_t axis = 0u; axis < 2u; ++axis) {
        if (friction[axis] > kConeEpsilon) {
            normalized += value[axis + 1u] * value[axis + 1u] /
                (friction[axis] * friction[axis]);
        } else if (value[axis + 1u] != 0.0) {
            normalized = std::numeric_limits<double>::infinity();
        }
    }
    Vec3 projected = value;
    if (!(value[0] >= 0.0 && normalized <= value[0] * value[0])) {
        const double dual = std::hypot(
            friction[0] * value[1],
            friction[1] * value[2]
        );
        if (value[0] + dual <= 0.0) {
            projected = {};
        } else {
            const bool activeU = friction[0] > kConeEpsilon;
            const bool activeV = friction[1] > kConeEpsilon;
            const bool isotropic =
                activeU && activeV &&
                std::abs(friction[0] - friction[1]) <=
                8.0 * std::numeric_limits<float>::epsilon() *
                    std::max({friction[0], friction[1], 1.0});
            if (activeU != activeV) {
                const double mu = activeU ? friction[0] : friction[1];
                const double tangent = activeU ? value[1] : value[2];
                const double normal =
                    (value[0] + mu * std::abs(tangent)) /
                    (1.0 + mu * mu);
                const double projectedTangent =
                    std::copysign(mu * normal, tangent);
                projected = {{
                    normal,
                    activeU ? projectedTangent : 0.0,
                    activeV ? projectedTangent : 0.0,
                }};
            } else if (isotropic) {
                const double tangent = std::hypot(value[1], value[2]);
                const double mu = 0.5 * (friction[0] + friction[1]);
                const double normal =
                    (value[0] + mu * tangent) / (1.0 + mu * mu);
                const double scale = tangent > kConeEpsilon
                    ? mu * normal / tangent
                    : 0.0;
                projected = {{normal, scale * value[1], scale * value[2]}};
            } else {
                const std::array<double, 2> frictionSquared{{
                    friction[0] * friction[0],
                    friction[1] * friction[1],
                }};
                double lower = std::max(0.0, -value[0]);
                double upper = std::max(
                    lower + std::max(dual, 1.0),
                    1.0
                );
                while (anisotropicFunction(value, frictionSquared, upper) > 0.0) {
                    upper = 2.0 * upper + std::max(dual, 1.0);
                }
                for (std::uint32_t iteration = 0u;
                     iteration < 100u;
                     ++iteration) {
                    const double middle = 0.5 * (lower + upper);
                    if (anisotropicFunction(
                            value, frictionSquared, middle
                        ) > 0.0) {
                        lower = middle;
                    } else {
                        upper = middle;
                    }
                }
                const double multiplier = 0.5 * (lower + upper);
                const double normal = std::max(value[0] + multiplier, 0.0);
                projected[0] = normal;
                for (std::size_t axis = 0; axis < 2; ++axis) {
                    projected[axis + 1] =
                        frictionSquared[axis] > kMatrixFloor
                        ? value[axis + 1] * frictionSquared[axis] * normal /
                            (frictionSquared[axis] * normal + multiplier)
                        : 0.0;
                }
            }
        }
    }
    if (maximumNormal > 0.0 && projected[0] > maximumNormal) {
        projected[0] = maximumNormal;
        const auto tangent = projectEllipse(
            {{value[1], value[2]}},
            {{friction[0] * maximumNormal, friction[1] * maximumNormal}}
        );
        projected[1] = tangent[0];
        projected[2] = tangent[1];
    }
    return projected;
}

bool conditionedInverse(const Mat3& matrix, Mat3& inverse) {
    double scale = 0.0;
    for (const auto& row : matrix) {
        for (const double value : row) {
            if (!std::isfinite(value)) {
                return false;
            }
            scale = std::max(scale, std::abs(value));
        }
    }
    if (!(scale > kMatrixFloor)) {
        return false;
    }
    Mat3 regularized{};
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            regularized[row][column] =
                0.5 * (matrix[row][column] + matrix[column][row]) / scale +
                (row == column ? kConditioning : 0.0);
        }
    }
    const double c00 = regularized[1][1] * regularized[2][2] -
        regularized[1][2] * regularized[2][1];
    const double c01 = regularized[1][2] * regularized[2][0] -
        regularized[1][0] * regularized[2][2];
    const double c02 = regularized[1][0] * regularized[2][1] -
        regularized[1][1] * regularized[2][0];
    const double determinant = regularized[0][0] * c00 +
        regularized[0][1] * c01 + regularized[0][2] * c02;
    if (!(determinant > kMatrixFloor) || !std::isfinite(determinant)) {
        return false;
    }
    const double reciprocal = 1.0 / (determinant * scale);
    inverse[0] = {{
        c00 * reciprocal,
        (regularized[0][2] * regularized[2][1] -
         regularized[0][1] * regularized[2][2]) * reciprocal,
        (regularized[0][1] * regularized[1][2] -
         regularized[0][2] * regularized[1][1]) * reciprocal,
    }};
    inverse[1] = {{
        c01 * reciprocal,
        (regularized[0][0] * regularized[2][2] -
         regularized[0][2] * regularized[2][0]) * reciprocal,
        (regularized[0][2] * regularized[1][0] -
         regularized[0][0] * regularized[1][2]) * reciprocal,
    }};
    inverse[2] = {{
        c02 * reciprocal,
        (regularized[0][1] * regularized[2][0] -
         regularized[0][0] * regularized[2][1]) * reciprocal,
        (regularized[0][0] * regularized[1][1] -
         regularized[0][1] * regularized[1][0]) * reciprocal,
    }};
    for (const auto& row : inverse) {
        for (const double value : row) {
            if (!std::isfinite(value)) {
                return false;
            }
        }
    }
    return true;
}

Vec3 matrixAction(
    const Batch& batch,
    const std::size_t problem,
    const std::size_t contact,
    const std::span<const Vec3> impulses
) {
    const std::size_t matrixBase = problem * kMatrixElements;
    const std::size_t row = 3u * contact;
    Vec3 result{};
    for (std::size_t source = 0; source < impulses.size(); ++source) {
        for (std::size_t targetAxis = 0; targetAxis < 3; ++targetAxis) {
            for (std::size_t sourceAxis = 0; sourceAxis < 3; ++sourceAxis) {
                result[targetAxis] +=
                    batch.matrices[
                        matrixBase +
                        (row + targetAxis) * kMaxRows +
                        3u * source + sourceAxis
                    ] * impulses[source][sourceAxis];
            }
        }
    }
    return result;
}

double minimumCholeskyPivot(
    const Batch& batch,
    const std::size_t problem
) {
    const std::size_t dimension =
        3u * batch.headers[problem].control.y;
    const std::size_t matrixBase = problem * kMatrixElements;
    std::vector<double> lower(dimension * dimension, 0.0);
    double minimumPivot = std::numeric_limits<double>::infinity();
    for (std::size_t row = 0u; row < dimension; ++row) {
        for (std::size_t column = 0u; column <= row; ++column) {
            const double value = batch.matrices[
                matrixBase + row * kMaxRows + column
            ];
            const double transposeValue = batch.matrices[
                matrixBase + column * kMaxRows + row
            ];
            if (!std::isfinite(value) || value != transposeValue) {
                return -1.0;
            }
            double reduced = value;
            for (std::size_t inner = 0u; inner < column; ++inner) {
                reduced -= lower[row * dimension + inner] *
                    lower[column * dimension + inner];
            }
            if (row == column) {
                if (!(reduced > kMatrixFloor) || !std::isfinite(reduced)) {
                    return -1.0;
                }
                const double pivot = std::sqrt(reduced);
                lower[row * dimension + column] = pivot;
                minimumPivot = std::min(minimumPivot, pivot);
            } else {
                lower[row * dimension + column] = reduced /
                    lower[column * dimension + column];
            }
        }
    }
    return minimumPivot;
}

double coneViolation(
    const Vec3& impulse,
    const double frictionU,
    const double frictionV,
    const double maximumNormal
) {
    const double normal = std::max(impulse[0], 0.0);
    double violation = std::max(-impulse[0], 0.0);
    if (maximumNormal > 0.0) {
        violation = std::max(
            violation,
            std::max(impulse[0] - maximumNormal, 0.0)
        );
    }
    const std::array<double, 2> friction{{frictionU, frictionV}};
    double normalizedSquared = 0.0;
    bool hasActiveTangent = false;
    for (std::size_t axis = 0u; axis < 2u; ++axis) {
        if (friction[axis] > kConeEpsilon) {
            hasActiveTangent = true;
            if (normal > kConeEpsilon) {
                const double scaled = impulse[axis + 1u] /
                    (friction[axis] * normal);
                normalizedSquared += scaled * scaled;
            } else {
                violation = std::max(
                    violation, std::abs(impulse[axis + 1u])
                );
            }
        } else {
            violation = std::max(
                violation, std::abs(impulse[axis + 1u])
            );
        }
    }
    if (hasActiveTangent && normal > kConeEpsilon) {
        violation = std::max(
            violation,
            std::max(std::sqrt(normalizedSquared) - 1.0, 0.0)
        );
    }
    return violation;
}

OracleResult solveOracle(const Batch& batch, const std::size_t problem) {
    constexpr std::uint32_t oracleMaximumIterations = 20000u;
    constexpr double oracleAbsoluteTolerance = 1.0e-11;
    constexpr double oracleRelativeTolerance = 1.0e-10;
    OracleResult result;
    const auto& header = batch.headers[problem];
    const std::size_t contactCount = header.control.y;
    const std::size_t contactBase = problem * kMaxContacts;
    std::vector<Mat3> validationInverses(contactCount);
    std::vector<double> stepScales(contactCount, 0.0);
    result.impulses.resize(contactCount);
    for (std::size_t contact = 0; contact < contactCount; ++contact) {
        const std::size_t row = 3u * contact;
        Mat3 diagonal{};
        for (std::size_t localRow = 0; localRow < 3; ++localRow) {
            for (std::size_t column = 0; column < 3; ++column) {
                diagonal[localRow][column] = batch.matrices[
                    problem * kMatrixElements +
                    (row + localRow) * kMaxRows + row + column
                ];
            }
        }
        if (!conditionedInverse(diagonal, validationInverses[contact])) {
            result.status = NUMI_TEMPORAL_CONE_ISLAND_FACTORIZATION_FAILED;
            return result;
        }
        const auto& source = batch.contacts[contactBase + contact];
        result.impulses[contact] = projectCone(
            {{
                source.warmImpulseAndFrictionV.x,
                source.warmImpulseAndFrictionV.y,
                source.warmImpulseAndFrictionV.z,
            }},
            source.freeVelocityAndFrictionU.w,
            source.warmImpulseAndFrictionV.w,
            source.limits.x
        );
    }

    for (std::size_t contact = 0; contact < contactCount; ++contact) {
        const std::size_t row = 3u * contact;
        double contactRowBound = 0.0;
        for (std::size_t outputAxis = 0; outputAxis < 3; ++outputAxis) {
            double rowSum = 0.0;
            for (std::size_t column = 0;
                 column < 3u * contactCount;
                 ++column) {
                rowSum += std::abs(batch.matrices[
                    problem * kMatrixElements +
                    (row + outputAxis) * kMaxRows + column
                ]);
            }
            contactRowBound = std::max(
                contactRowBound,
                rowSum
            );
        }
        stepScales[contact] = 1.0 /
            std::max(contactRowBound, kMatrixFloor);
    }
    const double effectiveRelaxation = header.tolerances.z;

    std::vector<Vec3> next(contactCount);
    for (std::uint32_t iteration = 0u;
         iteration < oracleMaximumIterations;
         ++iteration) {
        double maximumNatural = 0.0;
        double maximumScale = 1.0;
        for (std::size_t contact = 0; contact < contactCount; ++contact) {
            const auto& source = batch.contacts[contactBase + contact];
            Vec3 residual = matrixAction(batch, problem, contact, result.impulses);
            residual[0] += source.freeVelocityAndFrictionU.x;
            residual[1] += source.freeVelocityAndFrictionU.y;
            residual[2] += source.freeVelocityAndFrictionU.z;
            Vec3 proposed = result.impulses[contact];
            for (std::size_t axis = 0; axis < 3; ++axis) {
                proposed[axis] -= stepScales[contact] * residual[axis];
            }
            const Vec3 projected = projectCone(
                proposed,
                source.freeVelocityAndFrictionU.w,
                source.warmImpulseAndFrictionV.w,
                source.limits.x
            );
            const Vec3 response{{
                residual[0] - source.freeVelocityAndFrictionU.x,
                residual[1] - source.freeVelocityAndFrictionU.y,
                residual[2] - source.freeVelocityAndFrictionU.z,
            }};
            maximumScale = std::max({
                maximumScale,
                std::abs(static_cast<double>(source.freeVelocityAndFrictionU.x)),
                std::abs(static_cast<double>(source.freeVelocityAndFrictionU.y)),
                std::abs(static_cast<double>(source.freeVelocityAndFrictionU.z)),
                std::abs(response[0]),
                std::abs(response[1]),
                std::abs(response[2]),
            });
            for (std::size_t axis = 0; axis < 3; ++axis) {
                const double natural =
                    projected[axis] - result.impulses[contact][axis];
                next[contact][axis] = result.impulses[contact][axis] +
                    effectiveRelaxation * natural;
                maximumNatural = std::max(
                    maximumNatural,
                    std::abs(natural) / stepScales[contact]
                );
            }
        }
        result.iterations = iteration + 1u;
        if (maximumNatural <= oracleAbsoluteTolerance +
                oracleRelativeTolerance * maximumScale) {
            break;
        }
        result.impulses = next;
    }

    double maximumNatural = 0.0;
    double maximumScale = 1.0;
    for (std::size_t contact = 0; contact < contactCount; ++contact) {
        const auto& source = batch.contacts[contactBase + contact];
        const Vec3 action = matrixAction(batch, problem, contact, result.impulses);
        Vec3 residual{{
            action[0] + source.freeVelocityAndFrictionU.x,
            action[1] + source.freeVelocityAndFrictionU.y,
            action[2] + source.freeVelocityAndFrictionU.z,
        }};
        Vec3 proposed = result.impulses[contact];
        for (std::size_t axis = 0; axis < 3; ++axis) {
            proposed[axis] -= stepScales[contact] * residual[axis];
        }
        const Vec3 projected = projectCone(
            proposed,
            source.freeVelocityAndFrictionU.w,
            source.warmImpulseAndFrictionV.w,
            source.limits.x
        );
        maximumScale = std::max({
            maximumScale,
            std::abs(static_cast<double>(source.freeVelocityAndFrictionU.x)),
            std::abs(static_cast<double>(source.freeVelocityAndFrictionU.y)),
            std::abs(static_cast<double>(source.freeVelocityAndFrictionU.z)),
            std::abs(action[0]),
            std::abs(action[1]),
            std::abs(action[2]),
        });
        for (std::size_t axis = 0; axis < 3; ++axis) {
            maximumNatural = std::max(
                maximumNatural,
                std::abs(projected[axis] - result.impulses[contact][axis]) /
                    stepScales[contact]
            );
        }
        result.coneViolation = std::max(
            result.coneViolation,
            coneViolation(
                result.impulses[contact],
                source.freeVelocityAndFrictionU.w,
                source.warmImpulseAndFrictionV.w,
                source.limits.x
            )
        );
        result.objective +=
            0.5 * (
                result.impulses[contact][0] * action[0] +
                result.impulses[contact][1] * action[1] +
                result.impulses[contact][2] * action[2]
            ) +
            result.impulses[contact][0] * source.freeVelocityAndFrictionU.x +
            result.impulses[contact][1] * source.freeVelocityAndFrictionU.y +
            result.impulses[contact][2] * source.freeVelocityAndFrictionU.z;
    }
    result.kktResidual = maximumNatural / maximumScale;
    result.status = maximumNatural <= oracleAbsoluteTolerance +
        oracleRelativeTolerance * maximumScale
        ? NUMI_TEMPORAL_CONE_ISLAND_SUCCESS
        : NUMI_TEMPORAL_CONE_ISLAND_DID_NOT_CONVERGE;
    return result;
}

Batch makeBatch(const std::size_t problemCount) {
    Batch batch;
    batch.headers.resize(problemCount);
    batch.matrices.assign(problemCount * kMatrixElements, 0.0f);
    batch.contacts.resize(problemCount * kMaxContacts);
    constexpr std::array<std::uint32_t, 6> counts{{1u, 2u, 4u, 8u, 16u, 32u}};
    for (std::size_t problem = 0; problem < problemCount; ++problem) {
        const std::uint32_t contactCount = counts[problem % counts.size()];
        batch.headers[problem].control = u4(
            NUMI_TEMPORAL_CONE_ISLAND_ABI_VERSION,
            contactCount,
            4u,
            512u
        );
        batch.headers[problem].tolerances = f4(
            5.0e-7f,
            1.0e-6f,
            problem == 0u ? 0.8f : 1.0f,
            0.0f
        );

        const std::size_t matrixBase = problem * kMatrixElements;
        constexpr std::array<double, 6> couplingStrengths{{
            0.25, 0.5, 1.0, 2.0, 4.0, 8.0,
        }};
        for (std::size_t contact = 0u;
             contact < contactCount;
             ++contact) {
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                const double diagonal =
                    (axis == 0u ? 0.8 : axis == 1u ? 1.0 : 1.2) +
                    0.02 * static_cast<double>((contact + problem) % 7u);
                batch.matrices[
                    matrixBase +
                    (3u * contact + axis) * kMaxRows +
                    3u * contact + axis
                ] = static_cast<float>(diagonal);
            }
        }

        std::vector<std::pair<std::size_t, std::size_t>> edges;
        const auto addEdge = [&](std::size_t first, std::size_t second) {
            if (first == second || first >= contactCount ||
                second >= contactCount) {
                return;
            }
            if (first > second) {
                std::swap(first, second);
            }
            edges.emplace_back(first, second);
        };
        const std::size_t graphMode = (problem / 6u) % 6u;
        if (contactCount > 1u) {
            if (graphMode == 0u) {
                for (std::size_t contact = 1u;
                     contact < contactCount;
                     ++contact) {
                    addEdge(contact - 1u, contact);
                }
            } else if (graphMode == 1u) {
                for (std::size_t contact = 1u;
                     contact < contactCount;
                     ++contact) {
                    addEdge(contact - 1u, contact);
                }
                addEdge(0u, contactCount - 1u);
            } else if (graphMode == 2u) {
                for (std::size_t contact = 1u;
                     contact < contactCount;
                     ++contact) {
                    addEdge(0u, contact);
                }
            } else if (graphMode == 3u) {
                for (std::size_t contact = 0u;
                     contact < contactCount;
                     ++contact) {
                    addEdge(contact, contact + 1u);
                    addEdge(contact, contact + 2u);
                }
            } else if (graphMode == 4u) {
                for (std::size_t group = 0u;
                     group < contactCount;
                     group += 4u) {
                    const std::size_t end = std::min(
                        group + 4u,
                        static_cast<std::size_t>(contactCount)
                    );
                    for (std::size_t first = group; first < end; ++first) {
                        for (std::size_t second = first + 1u;
                             second < end;
                             ++second) {
                            addEdge(first, second);
                        }
                    }
                    addEdge(group, end);
                }
            } else {
                for (std::size_t contact = 1u;
                     contact < contactCount;
                     ++contact) {
                    addEdge(contact - 1u, contact);
                    addEdge(contact / 2u, contact);
                }
            }
        }
        // Retain occasional worst-capacity shared-mode cliques so sparse
        // qualification cannot accidentally depend on bounded row degree.
        if (contactCount == kMaxContacts && problem % 41u == 35u) {
            for (std::size_t first = 0u; first < contactCount; ++first) {
                for (std::size_t second = first + 1u;
                     second < contactCount;
                     ++second) {
                    addEdge(first, second);
                }
            }
        }
        std::sort(edges.begin(), edges.end());
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

        const double couplingStrength = couplingStrengths[
            problem % couplingStrengths.size()
        ];
        const auto addMatrixValue = [&](const std::size_t target,
                                        const std::size_t targetAxis,
                                        const std::size_t source,
                                        const std::size_t sourceAxis,
                                        const double value) {
            batch.matrices[
                matrixBase +
                (3u * target + targetAxis) * kMaxRows +
                3u * source + sourceAxis
            ] += static_cast<float>(value);
        };
        for (std::size_t edgeIndex = 0u;
             edgeIndex < edges.size();
             ++edgeIndex) {
            const auto [first, second] = edges[edgeIndex];
            for (std::size_t mode = 0u; mode < 2u; ++mode) {
                Vec3 firstMode{};
                Vec3 secondMode{};
                for (std::size_t axis = 0u; axis < 3u; ++axis) {
                    const double phase = static_cast<double>(
                        1u + problem + 3u * edgeIndex + 7u * mode + axis
                    );
                    firstMode[axis] = 0.22 * std::sin(0.31 * phase);
                    secondMode[axis] = 0.22 * std::cos(0.27 * phase);
                    if ((problem + edgeIndex + mode) % 2u != 0u) {
                        secondMode[axis] = -secondMode[axis];
                    }
                }
                const double weight = couplingStrength *
                    (mode == 0u ? 1.0 : 0.45);
                for (std::size_t row = 0u; row < 3u; ++row) {
                    for (std::size_t column = 0u; column < 3u; ++column) {
                        const double firstDiagonal =
                            weight * firstMode[row] * firstMode[column];
                        const double secondDiagonal =
                            weight * secondMode[row] * secondMode[column];
                        const double cross =
                            weight * firstMode[row] * secondMode[column];
                        addMatrixValue(
                            first, row, first, column, firstDiagonal
                        );
                        addMatrixValue(
                            second, row, second, column, secondDiagonal
                        );
                        addMatrixValue(first, row, second, column, cross);
                        addMatrixValue(second, column, first, row, cross);
                    }
                }
            }
        }

        const std::size_t contactBase = problem * kMaxContacts;
        for (std::size_t contact = 0; contact < contactCount; ++contact) {
            auto& output = batch.contacts[contactBase + contact];
            const bool separating = (problem + contact) % 9u == 0u;
            const float normalVelocity = separating
                ? 0.05f + 0.01f * static_cast<float>(contact % 5u)
                : -0.15f - 0.015f * static_cast<float>(
                    (problem + 3u * contact) % 17u
                );
            float frictionU =
                0.2f + 0.05f * static_cast<float>((problem + contact) % 9u);
            float frictionV = problem % 3u == 0u
                ? frictionU
                : 0.18f + 0.04f * static_cast<float>(
                    (2u * problem + contact) % 11u
                );
            if ((problem + 3u * contact) % 37u == 11u) {
                frictionU = 0.0f;
            } else if ((2u * problem + contact) % 41u == 17u) {
                frictionV = 0.0f;
            }
            output.freeVelocityAndFrictionU = f4(
                normalVelocity,
                0.25f * std::sin(
                    0.19f * static_cast<float>(problem + contact + 1u)
                ),
                0.25f * std::cos(
                    0.17f * static_cast<float>(2u * problem + contact + 1u)
                ),
                frictionU
            );
            output.warmImpulseAndFrictionV = f4(
                0.01f * static_cast<float>((problem + contact) % 7u),
                0.004f * static_cast<float>(
                    static_cast<int>((problem + contact) % 5u) - 2
                ),
                0.003f * static_cast<float>(
                    static_cast<int>((problem + 2u * contact) % 7u) - 3
                ),
                frictionV
            );
            output.limits = f4(
                (problem + contact) % 23u == 0u ? 1.5f : 0.0f,
                0.0f,
                0.0f,
                0.0f
            );
        }
        if (problem == 1u && contactCount == 2u) {
            // Shared-rigid analytic oracle. Independent contacts would each
            // produce 1/2, while the coupled normal operator [[2,1],[1,2]]
            // has the unique physical solution (1/3, 1/3).
            for (std::size_t row = 0u; row < 6u; ++row) {
                for (std::size_t column = 0u; column < 6u; ++column) {
                    batch.matrices[
                        matrixBase + row * kMaxRows + column
                    ] = 0.0f;
                }
            }
            for (std::size_t contact = 0u; contact < 2u; ++contact) {
                const std::size_t row = 3u * contact;
                batch.matrices[
                    matrixBase + row * kMaxRows + row
                ] = 2.0f;
                batch.matrices[
                    matrixBase + (row + 1u) * kMaxRows + row + 1u
                ] = 1.0f;
                batch.matrices[
                    matrixBase + (row + 2u) * kMaxRows + row + 2u
                ] = 1.0f;
                auto& shared = batch.contacts[contactBase + contact];
                shared.freeVelocityAndFrictionU =
                    f4(-1.0f, 0.0f, 0.0f, 0.0f);
                shared.warmImpulseAndFrictionV =
                    f4(0.0f, 0.0f, 0.0f, 0.0f);
                shared.limits = f4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            batch.matrices[matrixBase + 3u] = 1.0f;
            batch.matrices[matrixBase + 3u * kMaxRows] = 1.0f;
        }
        if (problem == 2u && contactCount > 0u) {
            // Exact degenerate anisotropic cone: tangent U is constrained to
            // zero while tangent V retains one-dimensional Coulomb friction.
            auto& degenerate = batch.contacts[contactBase];
            degenerate.freeVelocityAndFrictionU =
                f4(-1.0f, -0.4f, -0.3f, 0.0f);
            degenerate.warmImpulseAndFrictionV =
                f4(0.0f, 0.0f, 0.0f, 0.6f);
        }
    }
    return batch;
}

StreamBatch makeStreamBatch(const Batch& dense) {
    StreamBatch stream;
    stream.headers.reserve(dense.headers.size());
    for (std::size_t problem = 0u;
         problem < dense.headers.size();
         ++problem) {
        const auto& denseHeader = dense.headers[problem];
        const std::uint32_t contactCount = denseHeader.control.y;
        const std::size_t matrixBase = problem * kMatrixElements;
        const std::size_t denseContactBase = problem * kMaxContacts;
        if (contactCount == 0u || contactCount > kMaxContacts) {
            throw std::runtime_error(
                "cannot stream an invalid dense contact count"
            );
        }
        constexpr std::size_t maximumABIValue =
            std::numeric_limits<std::uint32_t>::max();
        if (stream.contacts.size() >
                maximumABIValue - contactCount ||
            stream.rowOffsets.size() >
                maximumABIValue - (contactCount + 1u) ||
            stream.columnIndices.size() >
                maximumABIValue - NUMI_TEMPORAL_CONE_STREAM_MAX_BLOCKS) {
            throw std::runtime_error("streamed island batch exceeds the ABI");
        }
        NumiTemporalConeStreamHeader header{};
        header.control = u4(
            NUMI_TEMPORAL_CONE_STREAM_ABI_VERSION,
            contactCount,
            denseHeader.control.z,
            denseHeader.control.w
        );
        header.ranges.x = static_cast<std::uint32_t>(
            stream.contacts.size()
        );
        header.ranges.y = static_cast<std::uint32_t>(
            stream.rowOffsets.size()
        );
        header.ranges.z = static_cast<std::uint32_t>(
            stream.columnIndices.size()
        );
        header.tolerances = denseHeader.tolerances;
        stream.contacts.insert(
            stream.contacts.end(),
            dense.contacts.begin() + denseContactBase,
            dense.contacts.begin() + denseContactBase + contactCount
        );

        const std::size_t islandBlockBase = stream.columnIndices.size();
        for (std::size_t target = 0u;
             target < contactCount;
             ++target) {
            stream.rowOffsets.push_back(static_cast<std::uint32_t>(
                stream.columnIndices.size() - islandBlockBase
            ));
            for (std::size_t source = 0u;
                 source < contactCount;
                 ++source) {
                bool nonzero = false;
                std::array<float, 9> block{};
                for (std::size_t targetAxis = 0u;
                     targetAxis < 3u;
                     ++targetAxis) {
                    for (std::size_t sourceAxis = 0u;
                         sourceAxis < 3u;
                         ++sourceAxis) {
                        const float value = dense.matrices[
                            matrixBase +
                            (3u * target + targetAxis) * kMaxRows +
                            3u * source + sourceAxis
                        ];
                        block[3u * targetAxis + sourceAxis] = value;
                        nonzero = nonzero || value != 0.0f;
                    }
                }
                if (!nonzero) {
                    continue;
                }
                stream.columnIndices.push_back(
                    static_cast<std::uint32_t>(source)
                );
                stream.blockValues.insert(
                    stream.blockValues.end(),
                    block.begin(),
                    block.end()
                );
            }
        }
        stream.rowOffsets.push_back(static_cast<std::uint32_t>(
            stream.columnIndices.size() - islandBlockBase
        ));
        const std::size_t islandBlocks =
            stream.columnIndices.size() - islandBlockBase;
        if (islandBlocks > NUMI_TEMPORAL_CONE_STREAM_MAX_BLOCKS) {
            throw std::runtime_error(
                "streamed island exceeds the block capacity"
            );
        }
        header.ranges.w = static_cast<std::uint32_t>(islandBlocks);
        stream.headers.push_back(header);
    }
    return stream;
}

Batch makeFailureBatch() {
    Batch batch = makeBatch(7u);

    // Problem 0: violate the symmetric Delassus contract.
    batch.matrices[1u] += 0.25f;

    // Problem 1: retain a symmetric, present local block whose conditioned
    // determinant is negative, forcing an explicit factorization failure.
    const std::size_t matrixBase = kMatrixElements;
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u; column < 3u; ++column) {
            batch.matrices[
                matrixBase + row * kMaxRows + column
            ] = row == column
                ? (row == 1u ? -1.0f : 1.0f)
                : 0.0f;
        }
    }

    // Problem 2: one iteration cannot certify the coupled four-contact map.
    batch.headers[2].control.z = 1u;
    batch.headers[2].control.w = 1u;

    // Problem 6: finite authored data whose cone norm overflows in FP32 must
    // fail transactionally instead of publishing a nonfinite warm start.
    auto& overflow = batch.contacts[6u * kMaxContacts];
    overflow.warmImpulseAndFrictionV.x = 0.0f;
    overflow.warmImpulseAndFrictionV.y =
        std::numeric_limits<float>::max();
    overflow.warmImpulseAndFrictionV.z =
        std::numeric_limits<float>::max();
    return batch;
}

std::string errorText(NSError* error) {
    return error == nil
        ? std::string("unknown Metal error")
        : std::string(error.localizedDescription.UTF8String);
}

GPUResult runGPU(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> pipeline,
    const Batch& batch
) {
    const std::size_t problemCount = batch.headers.size();
    id<MTLBuffer> headerBuffer = [device
        newBufferWithBytes:batch.headers.data()
                   length:batch.headers.size() *
                       sizeof(NumiTemporalConeIslandHeader)
                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> matrixBuffer = [device
        newBufferWithBytes:batch.matrices.data()
                   length:batch.matrices.size() * sizeof(float)
                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> contactBuffer = [device
        newBufferWithBytes:batch.contacts.data()
                   length:batch.contacts.size() *
                       sizeof(NumiTemporalConeIslandContact)
                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> impulseBuffer = [device
        newBufferWithLength:batch.contacts.size() * sizeof(mr_float4)
                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> statusBuffer = [device
        newBufferWithLength:problemCount *
            sizeof(NumiTemporalConeIslandStatus)
                    options:MTLResourceStorageModeShared];
    if (headerBuffer == nil || matrixBuffer == nil || contactBuffer == nil ||
        impulseBuffer == nil || statusBuffer == nil) {
        throw std::runtime_error("failed to allocate island buffers");
    }
    std::memset(
        impulseBuffer.contents,
        0,
        batch.contacts.size() * sizeof(mr_float4)
    );
    std::memset(
        statusBuffer.contents,
        0,
        problemCount * sizeof(NumiTemporalConeIslandStatus)
    );
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (commandBuffer == nil || encoder == nil) {
        throw std::runtime_error("failed to create island command encoder");
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:headerBuffer offset:0 atIndex:0];
    [encoder setBuffer:matrixBuffer offset:0 atIndex:1];
    [encoder setBuffer:contactBuffer offset:0 atIndex:2];
    [encoder setBuffer:impulseBuffer offset:0 atIndex:3];
    [encoder setBuffer:statusBuffer offset:0 atIndex:4];
    const std::uint32_t count = static_cast<std::uint32_t>(problemCount);
    [encoder setBytes:&count length:sizeof(count) atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
              threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted ||
        commandBuffer.error != nil) {
        throw std::runtime_error(
            "Metal island solve failed: " + errorText(commandBuffer.error)
        );
    }
    GPUResult result;
    const auto* impulses = static_cast<const mr_float4*>(impulseBuffer.contents);
    result.impulses.assign(
        impulses,
        impulses + batch.contacts.size()
    );
    const auto* statuses =
        static_cast<const NumiTemporalConeIslandStatus*>(statusBuffer.contents);
    result.statuses.assign(statuses, statuses + problemCount);
    if (commandBuffer.GPUEndTime >= commandBuffer.GPUStartTime) {
        result.seconds = commandBuffer.GPUEndTime - commandBuffer.GPUStartTime;
    }
    return result;
}

GPUResult runStreamGPU(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> pipeline,
    const StreamBatch& batch
) {
    const std::size_t problemCount = batch.headers.size();
    constexpr std::size_t maximumABIValue =
        std::numeric_limits<std::uint32_t>::max();
    if (problemCount > maximumABIValue ||
        batch.contacts.size() > maximumABIValue ||
        batch.rowOffsets.size() > maximumABIValue ||
        batch.columnIndices.size() > maximumABIValue ||
        batch.blockValues.size() !=
            NUMI_TEMPORAL_CONE_STREAM_BLOCK_ELEMENTS *
                batch.columnIndices.size()) {
        throw std::runtime_error("streamed buffer layout violates the ABI");
    }
    id<MTLBuffer> headerBuffer = [device
        newBufferWithBytes:batch.headers.data()
                   length:batch.headers.size() *
                       sizeof(NumiTemporalConeStreamHeader)
                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> rowOffsetBuffer = [device
        newBufferWithBytes:batch.rowOffsets.data()
                   length:batch.rowOffsets.size() * sizeof(std::uint32_t)
                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> columnBuffer = [device
        newBufferWithBytes:batch.columnIndices.data()
                   length:batch.columnIndices.size() * sizeof(std::uint32_t)
                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> blockBuffer = [device
        newBufferWithBytes:batch.blockValues.data()
                   length:batch.blockValues.size() * sizeof(float)
                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> contactBuffer = [device
        newBufferWithBytes:batch.contacts.data()
                   length:batch.contacts.size() *
                       sizeof(NumiTemporalConeIslandContact)
                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> impulseBuffer = [device
        newBufferWithLength:batch.contacts.size() * sizeof(mr_float4)
                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> statusBuffer = [device
        newBufferWithLength:problemCount *
            sizeof(NumiTemporalConeIslandStatus)
                    options:MTLResourceStorageModeShared];
    if (headerBuffer == nil || rowOffsetBuffer == nil ||
        columnBuffer == nil || blockBuffer == nil ||
        contactBuffer == nil || impulseBuffer == nil ||
        statusBuffer == nil) {
        throw std::runtime_error("failed to allocate streamed island buffers");
    }
    std::memset(
        impulseBuffer.contents,
        0,
        batch.contacts.size() * sizeof(mr_float4)
    );
    std::memset(
        statusBuffer.contents,
        0,
        problemCount * sizeof(NumiTemporalConeIslandStatus)
    );
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (commandBuffer == nil || encoder == nil) {
        throw std::runtime_error(
            "failed to create streamed island command encoder"
        );
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:headerBuffer offset:0 atIndex:0];
    [encoder setBuffer:rowOffsetBuffer offset:0 atIndex:1];
    [encoder setBuffer:columnBuffer offset:0 atIndex:2];
    [encoder setBuffer:blockBuffer offset:0 atIndex:3];
    [encoder setBuffer:contactBuffer offset:0 atIndex:4];
    [encoder setBuffer:impulseBuffer offset:0 atIndex:5];
    [encoder setBuffer:statusBuffer offset:0 atIndex:6];
    const std::uint32_t count = static_cast<std::uint32_t>(problemCount);
    [encoder setBytes:&count length:sizeof(count) atIndex:7];
    const mr_uint4 capacities = u4(
        static_cast<std::uint32_t>(batch.contacts.size()),
        static_cast<std::uint32_t>(batch.rowOffsets.size()),
        static_cast<std::uint32_t>(batch.columnIndices.size()),
        static_cast<std::uint32_t>(batch.contacts.size())
    );
    [encoder setBytes:&capacities length:sizeof(capacities) atIndex:8];
    [encoder dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
              threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted ||
        commandBuffer.error != nil) {
        throw std::runtime_error(
            "Metal streamed island solve failed: " +
            errorText(commandBuffer.error)
        );
    }
    GPUResult result;
    const auto* impulses = static_cast<const mr_float4*>(
        impulseBuffer.contents
    );
    result.impulses.assign(impulses, impulses + batch.contacts.size());
    const auto* statuses =
        static_cast<const NumiTemporalConeIslandStatus*>(
            statusBuffer.contents
        );
    result.statuses.assign(statuses, statuses + problemCount);
    if (commandBuffer.GPUEndTime >= commandBuffer.GPUStartTime) {
        result.seconds = commandBuffer.GPUEndTime - commandBuffer.GPUStartTime;
    }
    return result;
}

int run(const int argc, const char* const* argv) {
    std::size_t problemCount = 256u;
    std::uint32_t replayCount = 3u;
    std::string metallibPath = NUMI_TEMPORAL_CONE_METALLIB;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string_view value(argv[argument]);
        if (value == "--islands" && argument + 1 < argc) {
            problemCount = std::stoull(argv[++argument]);
        } else if (value == "--replays" && argument + 1 < argc) {
            replayCount = static_cast<std::uint32_t>(
                std::stoul(argv[++argument])
            );
        } else if (value == "--metallib" && argument + 1 < argc) {
            metallibPath = argv[++argument];
        } else if (value == "--help") {
            std::cout << "usage: numi-solver-islands [--islands N] "
                         "[--replays N] [--metallib PATH]\n";
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + std::string(value));
        }
    }
    problemCount = std::max<std::size_t>(problemCount, 6u);
    replayCount = std::max<std::uint32_t>(replayCount, 2u);
    if (problemCount > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("island count exceeds the ABI");
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        throw std::runtime_error("no Apple Metal device is available");
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
        throw std::runtime_error("failed to create command queue");
    }
    NSError* error = nil;
    NSString* path = [NSString stringWithUTF8String:metallibPath.c_str()];
    id<MTLLibrary> library = [device
        newLibraryWithURL:[NSURL fileURLWithPath:path]
                    error:&error];
    if (library == nil) {
        throw std::runtime_error("failed to load metallib: " + errorText(error));
    }
    id<MTLFunction> function =
        [library newFunctionWithName:@"numi_temporal_cone_island_solve"];
    if (function == nil) {
        throw std::runtime_error("metallib lacks the island solver kernel");
    }
    id<MTLFunction> streamFunction =
        [library newFunctionWithName:@"numi_temporal_cone_stream_solve"];
    if (streamFunction == nil) {
        throw std::runtime_error(
            "metallib lacks the streamed island solver kernel"
        );
    }
    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:function
                                      error:&error];
    if (pipeline == nil || pipeline.threadExecutionWidth != 32u) {
        throw std::runtime_error(
            "failed to create a SIMD32 island pipeline: " + errorText(error)
        );
    }
    id<MTLComputePipelineState> streamPipeline = [device
        newComputePipelineStateWithFunction:streamFunction
                                      error:&error];
    if (streamPipeline == nil ||
        streamPipeline.threadExecutionWidth != 32u) {
        throw std::runtime_error(
            "failed to create a SIMD32 streamed island pipeline: " +
            errorText(error)
        );
    }

    const Batch batch = makeBatch(problemCount);
    const StreamBatch streamBatch = makeStreamBatch(batch);
    double minimumSPDPivot = std::numeric_limits<double>::infinity();
    bool positiveDefinite = true;
    std::vector<OracleResult> oracle;
    oracle.reserve(problemCount);
    for (std::size_t problem = 0; problem < problemCount; ++problem) {
        const double pivot = minimumCholeskyPivot(batch, problem);
        positiveDefinite = positiveDefinite && pivot > 0.0;
        if (pivot > 0.0) {
            minimumSPDPivot = std::min(minimumSPDPivot, pivot);
        }
        oracle.push_back(solveOracle(batch, problem));
    }
    std::vector<GPUResult> replays;
    replays.reserve(replayCount);
    for (std::uint32_t replay = 0u; replay < replayCount; ++replay) {
        replays.push_back(runGPU(device, queue, pipeline, batch));
    }
    std::vector<GPUResult> streamReplays;
    streamReplays.reserve(replayCount);
    for (std::uint32_t replay = 0u; replay < replayCount; ++replay) {
        streamReplays.push_back(runStreamGPU(
            device,
            queue,
            streamPipeline,
            streamBatch
        ));
    }

    const Batch failureBatch = makeFailureBatch();
    StreamBatch failureStreamBatch = makeStreamBatch(failureBatch);
    failureStreamBatch.headers[3].control.x = 0xffffffffu;
    const auto& malformedHeader = failureStreamBatch.headers[4];
    const std::size_t malformedRowOffset = malformedHeader.ranges.y;
    const std::size_t malformedBlockBase = malformedHeader.ranges.z;
    const std::size_t malformedBegin =
        failureStreamBatch.rowOffsets[malformedRowOffset];
    failureStreamBatch.columnIndices[
        malformedBlockBase + malformedBegin + 1u
    ] = failureStreamBatch.columnIndices[
        malformedBlockBase + malformedBegin
    ];
    failureStreamBatch.headers[5].ranges.w =
        NUMI_TEMPORAL_CONE_STREAM_MAX_BLOCKS + 1u;
    const GPUResult failureFirst = runStreamGPU(
        device,
        queue,
        streamPipeline,
        failureStreamBatch
    );
    const GPUResult failureReplay = runStreamGPU(
        device,
        queue,
        streamPipeline,
        failureStreamBatch
    );
    const std::array<std::uint32_t, 7> expectedFailureCodes{{
        NUMI_TEMPORAL_CONE_ISLAND_INVALID_INPUT,
        NUMI_TEMPORAL_CONE_ISLAND_FACTORIZATION_FAILED,
        NUMI_TEMPORAL_CONE_ISLAND_DID_NOT_CONVERGE,
        NUMI_TEMPORAL_CONE_ISLAND_INVALID_ABI,
        NUMI_TEMPORAL_CONE_ISLAND_INVALID_INPUT,
        NUMI_TEMPORAL_CONE_ISLAND_INVALID_INPUT,
        NUMI_TEMPORAL_CONE_ISLAND_NONFINITE_RESULT,
    }};
    bool typedFailures = true;
    for (std::size_t problem = 0u;
         problem < expectedFailureCodes.size();
         ++problem) {
        const std::uint32_t actualCode =
            failureFirst.statuses[problem].control.x;
        if (actualCode != expectedFailureCodes[problem]) {
            std::cerr
                << "failure_case=" << problem
                << " expected_status=" << expectedFailureCodes[problem]
                << " actual_status=" << actualCode << '\n';
            typedFailures = false;
        }
    }
    const bool deterministicFailures =
        std::memcmp(
            failureFirst.impulses.data(),
            failureReplay.impulses.data(),
            failureFirst.impulses.size() * sizeof(mr_float4)
        ) == 0 &&
        std::memcmp(
            failureFirst.statuses.data(),
            failureReplay.statuses.data(),
            failureFirst.statuses.size() *
                sizeof(NumiTemporalConeIslandStatus)
        ) == 0;
    bool failureRollback = true;
    constexpr std::array<std::size_t, 6> zeroRollbackProblems{{
        0u, 1u, 3u, 4u, 5u, 6u,
    }};
    for (const std::size_t problem : zeroRollbackProblems) {
        const auto& header = failureStreamBatch.headers[problem];
        for (std::size_t contact = 0u;
             contact < header.control.y;
             ++contact) {
            const auto& impulse = failureFirst.impulses[
                header.ranges.x + contact
            ];
            failureRollback = failureRollback &&
                impulse.x == 0.0f &&
                impulse.y == 0.0f &&
                impulse.z == 0.0f &&
                impulse.w == 0.0f;
        }
    }
    const std::size_t rollbackProblem = 2u;
    const std::size_t rollbackDenseContactBase =
        rollbackProblem * kMaxContacts;
    const std::size_t rollbackStreamContactBase =
        failureStreamBatch.headers[rollbackProblem].ranges.x;
    for (std::size_t contact = 0u;
         contact < failureBatch.headers[rollbackProblem].control.y;
         ++contact) {
        const auto& source = failureBatch.contacts[
            rollbackDenseContactBase + contact
        ];
        const Vec3 checkpoint = projectCone(
            {{
                source.warmImpulseAndFrictionV.x,
                source.warmImpulseAndFrictionV.y,
                source.warmImpulseAndFrictionV.z,
            }},
            source.freeVelocityAndFrictionU.w,
            source.warmImpulseAndFrictionV.w,
            source.limits.x
        );
        const auto& actual = failureFirst.impulses[
            rollbackStreamContactBase + contact
        ];
        failureRollback = failureRollback &&
            std::abs(actual.x - checkpoint[0]) <= 1.0e-6 &&
            std::abs(actual.y - checkpoint[1]) <= 1.0e-6 &&
            std::abs(actual.z - checkpoint[2]) <= 1.0e-6;
    }

    bool denseDeterministic = true;
    for (std::size_t replay = 1; replay < replays.size(); ++replay) {
        denseDeterministic = denseDeterministic &&
            std::memcmp(
                replays[0].impulses.data(),
                replays[replay].impulses.data(),
                replays[0].impulses.size() * sizeof(mr_float4)
            ) == 0 &&
            std::memcmp(
                replays[0].statuses.data(),
                replays[replay].statuses.data(),
                replays[0].statuses.size() *
                    sizeof(NumiTemporalConeIslandStatus)
            ) == 0;
    }
    bool deterministic = true;
    for (std::size_t replay = 1u;
         replay < streamReplays.size();
         ++replay) {
        deterministic = deterministic &&
            std::memcmp(
                streamReplays[0].impulses.data(),
                streamReplays[replay].impulses.data(),
                streamReplays[0].impulses.size() * sizeof(mr_float4)
            ) == 0 &&
            std::memcmp(
                streamReplays[0].statuses.data(),
                streamReplays[replay].statuses.data(),
                streamReplays[0].statuses.size() *
                    sizeof(NumiTemporalConeIslandStatus)
            ) == 0;
    }
    bool denseStreamBitwise = true;
    for (std::size_t problem = 0u;
         problem < problemCount;
         ++problem) {
        denseStreamBitwise = denseStreamBitwise &&
            std::memcmp(
                &replays[0].statuses[problem],
                &streamReplays[0].statuses[problem],
                sizeof(NumiTemporalConeIslandStatus)
            ) == 0;
        const std::size_t denseContactBase = problem * kMaxContacts;
        const std::size_t streamContactBase =
            streamBatch.headers[problem].ranges.x;
        for (std::size_t contact = 0u;
             contact < batch.headers[problem].control.y;
             ++contact) {
            denseStreamBitwise = denseStreamBitwise &&
                std::memcmp(
                    &replays[0].impulses[denseContactBase + contact],
                    &streamReplays[0].impulses[
                        streamContactBase + contact
                    ],
                    sizeof(mr_float4)
                ) == 0;
        }
    }

    std::size_t failedIslands = 0u;
    double maximumImpulseError = 0.0;
    double maximumObjectiveError = 0.0;
    double maximumFP64KKTResidual = 0.0;
    double maximumKKTResidual = 0.0;
    double maximumConeViolation = 0.0;
    double maximumPositiveObjective = 0.0;
    std::uint32_t maximumIterations = 0u;
    std::uint32_t maximumAccelerationRestarts = 0u;
    std::size_t acceleratedIslands = 0u;
    std::vector<std::uint32_t> iterationCounts;
    iterationCounts.reserve(problemCount);
    std::uint64_t contactIterations = 0u;
    std::uint64_t contactsSolved = 0u;
    std::uint64_t degenerateConeContacts = 0u;
    double maximumDegenerateInactiveImpulse = 0.0;
    double maximumDegenerateActiveImpulse = 0.0;
    for (std::size_t problem = 0; problem < problemCount; ++problem) {
        const auto& expected = oracle[problem];
        const auto& actual = streamReplays[0].statuses[problem];
        bool valid = expected.status == NUMI_TEMPORAL_CONE_ISLAND_SUCCESS &&
            actual.control.x == NUMI_TEMPORAL_CONE_ISLAND_SUCCESS &&
            actual.control.z == 1u;
        maximumFP64KKTResidual = std::max(
            maximumFP64KKTResidual,
            expected.kktResidual
        );
        maximumKKTResidual = std::max(
            maximumKKTResidual,
            static_cast<double>(actual.residuals.x)
        );
        maximumConeViolation = std::max(
            maximumConeViolation,
            static_cast<double>(actual.residuals.y)
        );
        maximumObjectiveError = std::max(
            maximumObjectiveError,
            std::abs(
                static_cast<double>(actual.residuals.w) -
                expected.objective
            ) / std::max(1.0, std::abs(expected.objective))
        );
        maximumPositiveObjective = std::max(
            maximumPositiveObjective,
            std::max(static_cast<double>(actual.residuals.w), 0.0)
        );
        maximumIterations = std::max(maximumIterations, actual.control.y);
        const auto restartCount = static_cast<std::uint32_t>(
            actual.diagnostics.w
        );
        maximumAccelerationRestarts = std::max(
            maximumAccelerationRestarts,
            restartCount
        );
        acceleratedIslands += restartCount > 0u ? 1u : 0u;
        iterationCounts.push_back(actual.control.y);
        contactsSolved += actual.control.w;
        contactIterations +=
            static_cast<std::uint64_t>(actual.control.w) * actual.control.y;
        const std::size_t streamContactBase =
            streamBatch.headers[problem].ranges.x;
        for (std::size_t contact = 0;
             contact < batch.headers[problem].control.y;
             ++contact) {
            const auto& gpu = streamReplays[0].impulses[
                streamContactBase + contact
            ];
            const auto& cpu = expected.impulses[contact];
            const std::array<double, 3> values{{gpu.x, gpu.y, gpu.z}};
            const auto& source = batch.contacts[
                problem * kMaxContacts + contact
            ];
            const bool activeU =
                source.freeVelocityAndFrictionU.w > kConeEpsilon;
            const bool activeV =
                source.warmImpulseAndFrictionV.w > kConeEpsilon;
            if (activeU != activeV) {
                ++degenerateConeContacts;
                maximumDegenerateInactiveImpulse = std::max(
                    maximumDegenerateInactiveImpulse,
                    std::abs(activeU ? values[2] : values[1])
                );
                maximumDegenerateActiveImpulse = std::max(
                    maximumDegenerateActiveImpulse,
                    std::abs(activeU ? values[1] : values[2])
                );
            }
            for (std::size_t axis = 0; axis < 3; ++axis) {
                maximumImpulseError = std::max(
                    maximumImpulseError,
                    std::abs(values[axis] - cpu[axis]) /
                        std::max(1.0, std::abs(cpu[axis]))
                );
                valid = valid && std::isfinite(values[axis]);
            }
        }
        if (!valid) {
            if (failedIslands < 5u) {
                std::cerr
                    << "failed_island=" << problem
                    << " contacts=" << batch.headers[problem].control.y
                    << " cpu_status=" << expected.status
                    << " gpu_status=" << actual.control.x
                    << " gpu_iterations=" << actual.control.y
                    << " gpu_residual=" << actual.residuals.x
                    << " relaxation=" << actual.diagnostics.y
                    << '\n';
            }
            ++failedIslands;
        }
    }

    double denseTotalSeconds = 0.0;
    for (const auto& replay : replays) {
        denseTotalSeconds += replay.seconds;
    }
    double streamTotalSeconds = 0.0;
    for (const auto& replay : streamReplays) {
        streamTotalSeconds += replay.seconds;
    }
    const double denseAverageSeconds = denseTotalSeconds / replays.size();
    const double averageSeconds =
        streamTotalSeconds / streamReplays.size();
    const double islandsPerSecond = averageSeconds > 0.0
        ? static_cast<double>(problemCount) / averageSeconds
        : 0.0;
    const double contactsPerSecond = averageSeconds > 0.0
        ? static_cast<double>(contactsSolved) / averageSeconds
        : 0.0;
    const double contactIterationsPerSecond = averageSeconds > 0.0
        ? static_cast<double>(contactIterations) / averageSeconds
        : 0.0;
    const std::uint64_t denseBufferBytes =
        batch.headers.size() * sizeof(NumiTemporalConeIslandHeader) +
        batch.matrices.size() * sizeof(float) +
        batch.contacts.size() * sizeof(NumiTemporalConeIslandContact) +
        batch.contacts.size() * sizeof(mr_float4) +
        batch.headers.size() * sizeof(NumiTemporalConeIslandStatus);
    const std::uint64_t streamBufferBytes =
        streamBatch.headers.size() * sizeof(NumiTemporalConeStreamHeader) +
        streamBatch.rowOffsets.size() * sizeof(std::uint32_t) +
        streamBatch.columnIndices.size() * sizeof(std::uint32_t) +
        streamBatch.blockValues.size() * sizeof(float) +
        streamBatch.contacts.size() *
            sizeof(NumiTemporalConeIslandContact) +
        streamBatch.contacts.size() * sizeof(mr_float4) +
        streamBatch.headers.size() * sizeof(NumiTemporalConeIslandStatus);
    const double speedup = averageSeconds > 0.0
        ? denseAverageSeconds / averageSeconds
        : 0.0;
    const double memoryRatio = denseBufferBytes > 0u
        ? static_cast<double>(streamBufferBytes) /
            static_cast<double>(denseBufferBytes)
        : 0.0;
    const std::size_t sharedContactBase =
        streamBatch.headers[1].ranges.x;
    const auto& sharedFirst = streamReplays[0].impulses[sharedContactBase];
    const auto& sharedSecond =
        streamReplays[0].impulses[sharedContactBase + 1u];
    const bool sharedRigidOracle =
        std::abs(static_cast<double>(sharedFirst.x) - 1.0 / 3.0) <= 2.0e-6 &&
        std::abs(static_cast<double>(sharedSecond.x) - 1.0 / 3.0) <= 2.0e-6 &&
        std::abs(sharedFirst.y) <= 1.0e-7 &&
        std::abs(sharedFirst.z) <= 1.0e-7 &&
        std::abs(sharedSecond.y) <= 1.0e-7 &&
        std::abs(sharedSecond.z) <= 1.0e-7;
    const bool underRelaxedPath =
        streamReplays[0].statuses[0].control.x ==
            NUMI_TEMPORAL_CONE_ISLAND_SUCCESS &&
        std::abs(streamReplays[0].statuses[0].diagnostics.y - 0.8f) <=
            std::numeric_limits<float>::epsilon() &&
        streamReplays[0].statuses[0].diagnostics.w == 0.0f;
    std::uint64_t denseActiveBlocks = 0u;
    for (const auto& header : batch.headers) {
        denseActiveBlocks +=
            static_cast<std::uint64_t>(header.control.y) * header.control.y;
    }
    const double blockFill = denseActiveBlocks > 0u
        ? static_cast<double>(streamBatch.columnIndices.size()) /
            static_cast<double>(denseActiveBlocks)
        : 0.0;
    std::sort(iterationCounts.begin(), iterationCounts.end());
    const auto iterationPercentile = [&](const std::size_t numerator) {
        const std::size_t rank = std::max<std::size_t>(
            1u,
            (numerator * iterationCounts.size() + 99u) / 100u
        );
        return iterationCounts[
            std::min(rank, iterationCounts.size()) - 1u
        ];
    };
    const std::uint32_t iterationP50 = iterationPercentile(50u);
    const std::uint32_t iterationP95 = iterationPercentile(95u);
    const std::uint32_t iterationP99 = iterationPercentile(99u);
    std::uint32_t maximumIslandBlocks = 0u;
    std::uint32_t fullCapacityIslands = 0u;
    for (const auto& header : streamBatch.headers) {
        maximumIslandBlocks = std::max(
            maximumIslandBlocks,
            header.ranges.w
        );
        if (header.ranges.w == NUMI_TEMPORAL_CONE_STREAM_MAX_BLOCKS) {
            ++fullCapacityIslands;
        }
    }
    const bool fullCapacityCovered =
        problemCount < 36u ||
        maximumIslandBlocks == NUMI_TEMPORAL_CONE_STREAM_MAX_BLOCKS;
    const bool passed =
        failedIslands == 0u &&
        maximumImpulseError <= 2.0e-5 &&
        maximumObjectiveError <= 2.0e-5 &&
        maximumFP64KKTResidual <= 2.0e-10 &&
        maximumKKTResidual <= 2.0e-6 &&
        maximumConeViolation <= 2.0e-6 &&
        maximumPositiveObjective <= 2.0e-5 &&
        degenerateConeContacts > 0u &&
        maximumDegenerateInactiveImpulse <= 2.0e-6 &&
        maximumDegenerateActiveImpulse > 1.0e-4 &&
        positiveDefinite &&
        sharedRigidOracle &&
        underRelaxedPath &&
        fullCapacityCovered &&
        denseDeterministic &&
        deterministic &&
        denseStreamBitwise &&
        typedFailures &&
        deterministicFailures &&
        failureRollback;

    std::cout << std::fixed << std::setprecision(9)
              << "device=" << device.name.UTF8String << '\n'
              << "islands=" << problemCount
              << " contacts=" << contactsSolved
              << " replays=" << replayCount
              << " failed_islands=" << failedIslands << '\n'
              << "max_fp64_impulse_error=" << maximumImpulseError
              << " max_fp64_objective_error=" << maximumObjectiveError
              << " max_fp64_kkt_residual=" << maximumFP64KKTResidual
              << " max_kkt_residual=" << maximumKKTResidual
              << " max_cone_violation=" << maximumConeViolation
              << " max_positive_objective=" << maximumPositiveObjective
              << " degenerate_cone_contacts=" << degenerateConeContacts
              << " max_degenerate_inactive_impulse="
              << maximumDegenerateInactiveImpulse
              << " max_degenerate_active_impulse="
              << maximumDegenerateActiveImpulse
              << " max_iterations=" << maximumIterations
              << " iteration_p50=" << iterationP50
              << " iteration_p95=" << iterationP95
              << " iteration_p99=" << iterationP99
              << " accelerated_islands=" << acceleratedIslands
              << " max_acceleration_restarts="
              << maximumAccelerationRestarts << '\n'
              << "deterministic_replay="
              << (deterministic ? "true" : "false")
              << " dense_deterministic="
              << (denseDeterministic ? "true" : "false")
              << " dense_stream_bitwise="
              << (denseStreamBitwise ? "true" : "false")
              << " typed_failures=" << (typedFailures ? "true" : "false")
              << " deterministic_failures="
              << (deterministicFailures ? "true" : "false")
              << " failure_rollback="
              << (failureRollback ? "true" : "false")
              << " spd_cholesky="
              << (positiveDefinite ? "true" : "false")
              << " min_spd_pivot=" << minimumSPDPivot
              << " shared_rigid_oracle="
              << (sharedRigidOracle ? "true" : "false")
              << " under_relaxed_path="
              << (underRelaxedPath ? "true" : "false") << '\n'
              << "average_gpu_seconds=" << averageSeconds
              << " islands_per_second=" << islandsPerSecond
              << " contacts_per_second=" << contactsPerSecond
              << " contact_iterations_per_second="
              << contactIterationsPerSecond
              << " streamed_buffer_bytes=" << streamBufferBytes << '\n'
              << "dense_gpu_seconds=" << denseAverageSeconds
              << " dense_to_stream_speedup=" << speedup
              << " dense_buffer_bytes=" << denseBufferBytes
              << " stream_to_dense_memory=" << memoryRatio
              << " streamed_blocks=" << streamBatch.columnIndices.size()
              << " block_fill=" << blockFill
              << " max_island_blocks=" << maximumIslandBlocks
              << " full_capacity_islands=" << fullCapacityIslands << '\n'
              << "stream_threadgroup_memory="
              << streamPipeline.staticThreadgroupMemoryLength
              << " dense_threadgroup_memory="
              << pipeline.staticThreadgroupMemoryLength
              << " stream_max_threads="
              << streamPipeline.maxTotalThreadsPerThreadgroup << '\n'
              << "result=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}

} // namespace

int main(const int argc, const char* const* argv) {
    @autoreleasepool {
        try {
            return run(argc, argv);
        } catch (const std::exception& error) {
            std::cerr << "numi-solver-islands: " << error.what() << '\n';
            return 2;
        }
    }
}
