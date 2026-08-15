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

struct GPUResult {
    std::vector<mr_float4> impulses;
    std::vector<NumiTemporalConeIslandStatus> statuses;
    double seconds = 0.0;
};

struct OracleResult {
    std::vector<Vec3> impulses;
    std::uint32_t status = NUMI_TEMPORAL_CONE_ISLAND_SUCCESS;
    std::uint32_t iterations = 0u;
    double naturalResidual = 0.0;
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
        } else if (std::abs(tangent[axis]) > kConeEpsilon) {
            normalized = std::numeric_limits<double>::infinity();
        }
    }
    if (normalized <= 1.0) {
        return tangent;
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
    if (friction[0] <= kConeEpsilon ||
        friction[1] <= kConeEpsilon) {
        return {{
            maximumNormal > 0.0
                ? std::clamp(value[0], 0.0, maximumNormal)
                : std::max(value[0], 0.0),
            0.0,
            0.0,
        }};
    }

    const double normalized =
        value[1] * value[1] / (friction[0] * friction[0]) +
        value[2] * value[2] / (friction[1] * friction[1]);
    Vec3 projected = value;
    if (!(value[0] >= 0.0 && normalized <= value[0] * value[0])) {
        const double dual = std::hypot(
            friction[0] * value[1],
            friction[1] * value[2]
        );
        if (value[0] + dual <= 0.0) {
            projected = {};
        } else {
            const bool isotropic =
                std::abs(friction[0] - friction[1]) <=
                8.0 * std::numeric_limits<float>::epsilon() *
                    std::max({friction[0], friction[1], 1.0});
            if (isotropic) {
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
                        value[axis + 1] * frictionSquared[axis] * normal /
                        (frictionSquared[axis] * normal + multiplier);
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

double coneViolation(
    const Vec3& impulse,
    const double frictionU,
    const double frictionV
) {
    const double limitU = frictionU * impulse[0];
    const double limitV = frictionV * impulse[0];
    if (limitU > 0.0 && limitV > 0.0) {
        return std::max(
            std::sqrt(
                impulse[1] * impulse[1] / (limitU * limitU) +
                impulse[2] * impulse[2] / (limitV * limitV)
            ) - 1.0,
            0.0
        );
    }
    return std::hypot(impulse[1], impulse[2]);
}

OracleResult solveOracle(const Batch& batch, const std::size_t problem) {
    OracleResult result;
    const auto& header = batch.headers[problem];
    const std::size_t contactCount = header.control.y;
    const std::size_t contactBase = problem * kMaxContacts;
    std::vector<Mat3> inverses(contactCount);
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
        if (!conditionedInverse(diagonal, inverses[contact])) {
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

    double preconditionedRowBound = 0.0;
    for (std::size_t contact = 0; contact < contactCount; ++contact) {
        const std::size_t row = 3u * contact;
        for (std::size_t outputAxis = 0; outputAxis < 3; ++outputAxis) {
            double rowSum = 0.0;
            for (std::size_t column = 0;
                 column < 3u * contactCount;
                 ++column) {
                double value = 0.0;
                for (std::size_t localRow = 0; localRow < 3; ++localRow) {
                    value += inverses[contact][outputAxis][localRow] *
                        batch.matrices[
                            problem * kMatrixElements +
                            (row + localRow) * kMaxRows + column
                        ];
                }
                rowSum += std::abs(value);
            }
            preconditionedRowBound = std::max(
                preconditionedRowBound,
                rowSum
            );
        }
    }
    const double effectiveRelaxation = std::min(
        static_cast<double>(header.tolerances.z),
        1.0 / std::max(preconditionedRowBound, 1.0)
    );

    std::vector<Vec3> next(contactCount);
    for (std::uint32_t iteration = 0u;
         iteration < header.control.w;
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
            for (std::size_t row = 0; row < 3; ++row) {
                for (std::size_t column = 0; column < 3; ++column) {
                    proposed[row] -= inverses[contact][row][column] *
                        residual[column];
                }
            }
            const Vec3 projected = projectCone(
                proposed,
                source.freeVelocityAndFrictionU.w,
                source.warmImpulseAndFrictionV.w,
                source.limits.x
            );
            for (std::size_t axis = 0; axis < 3; ++axis) {
                const double natural =
                    projected[axis] - result.impulses[contact][axis];
                next[contact][axis] = result.impulses[contact][axis] +
                    effectiveRelaxation * natural;
                maximumNatural = std::max(maximumNatural, std::abs(natural));
                maximumScale = std::max(
                    maximumScale,
                    std::abs(projected[axis])
                );
            }
        }
        result.iterations = iteration + 1u;
        if (result.iterations >= header.control.z &&
            maximumNatural <= header.tolerances.x +
                header.tolerances.y * maximumScale) {
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
        for (std::size_t row = 0; row < 3; ++row) {
            for (std::size_t column = 0; column < 3; ++column) {
                proposed[row] -= inverses[contact][row][column] *
                    residual[column];
            }
        }
        const Vec3 projected = projectCone(
            proposed,
            source.freeVelocityAndFrictionU.w,
            source.warmImpulseAndFrictionV.w,
            source.limits.x
        );
        for (std::size_t axis = 0; axis < 3; ++axis) {
            maximumNatural = std::max(
                maximumNatural,
                std::abs(projected[axis] - result.impulses[contact][axis])
            );
            maximumScale = std::max(
                maximumScale,
                std::abs(result.impulses[contact][axis])
            );
        }
        result.coneViolation = std::max(
            result.coneViolation,
            coneViolation(
                result.impulses[contact],
                source.freeVelocityAndFrictionU.w,
                source.warmImpulseAndFrictionV.w
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
    result.naturalResidual = maximumNatural / maximumScale;
    result.status = maximumNatural <= header.tolerances.x +
        header.tolerances.y * maximumScale
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
            256u
        );
        batch.headers[problem].tolerances = f4(
            5.0e-7f,
            1.0e-6f,
            1.0f,
            0.0f
        );

        const std::size_t dimension = 3u * contactCount;
        const std::size_t matrixBase = problem * kMatrixElements;
        std::vector<double> couplingVector(dimension);
        for (std::size_t row = 0; row < dimension; ++row) {
            couplingVector[row] =
                0.25 * std::sin(
                    0.13 * static_cast<double>((problem + 1u) * (row + 3u))
                );
        }
        constexpr std::array<double, 6> couplingStrengths{{
            0.08, 0.25, 0.5, 1.0, 2.0, 4.0,
        }};
        const double rankCoupling =
            couplingStrengths[problem % couplingStrengths.size()];
        for (std::size_t row = 0; row < dimension; ++row) {
            const std::size_t axis = row % 3u;
            const double diagonal =
                (axis == 0u ? 0.8 : axis == 1u ? 1.0 : 1.2) +
                0.02 * static_cast<double>((row / 3u + problem) % 7u);
            for (std::size_t column = 0; column < dimension; ++column) {
                const double value =
                    (row == column ? diagonal : 0.0) +
                    rankCoupling * couplingVector[row] * couplingVector[column];
                batch.matrices[
                    matrixBase + row * kMaxRows + column
                ] = static_cast<float>(value);
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
            const float frictionU =
                0.2f + 0.05f * static_cast<float>((problem + contact) % 9u);
            const float frictionV = problem % 3u == 0u
                ? frictionU
                : 0.18f + 0.04f * static_cast<float>(
                    (2u * problem + contact) % 11u
                );
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
    }
    return batch;
}

Batch makeFailureBatch() {
    Batch batch = makeBatch(3u);

    // Problem 0: violate the symmetric Delassus contract.
    batch.matrices[1u] += 0.25f;

    // Problem 1: retain a symmetric operator but erase the first local block,
    // forcing an explicit block-factorization failure.
    const std::size_t matrixBase = kMatrixElements;
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u; column < 3u; ++column) {
            batch.matrices[
                matrixBase + row * kMaxRows + column
            ] = 0.0f;
        }
    }

    // Problem 2: one iteration cannot certify the coupled four-contact map.
    batch.headers[2].control.z = 1u;
    batch.headers[2].control.w = 1u;
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
    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:function
                                      error:&error];
    if (pipeline == nil || pipeline.threadExecutionWidth != 32u) {
        throw std::runtime_error(
            "failed to create a SIMD32 island pipeline: " + errorText(error)
        );
    }

    const Batch batch = makeBatch(problemCount);
    std::vector<OracleResult> oracle;
    oracle.reserve(problemCount);
    for (std::size_t problem = 0; problem < problemCount; ++problem) {
        oracle.push_back(solveOracle(batch, problem));
    }
    std::vector<GPUResult> replays;
    replays.reserve(replayCount);
    for (std::uint32_t replay = 0u; replay < replayCount; ++replay) {
        replays.push_back(runGPU(device, queue, pipeline, batch));
    }

    const Batch failureBatch = makeFailureBatch();
    const GPUResult failureFirst = runGPU(
        device,
        queue,
        pipeline,
        failureBatch
    );
    const GPUResult failureReplay = runGPU(
        device,
        queue,
        pipeline,
        failureBatch
    );
    const std::array<std::uint32_t, 3> expectedFailureCodes{{
        NUMI_TEMPORAL_CONE_ISLAND_INVALID_INPUT,
        NUMI_TEMPORAL_CONE_ISLAND_FACTORIZATION_FAILED,
        NUMI_TEMPORAL_CONE_ISLAND_DID_NOT_CONVERGE,
    }};
    bool typedFailures = true;
    for (std::size_t problem = 0u;
         problem < expectedFailureCodes.size();
         ++problem) {
        typedFailures = typedFailures &&
            failureFirst.statuses[problem].control.x ==
                expectedFailureCodes[problem];
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
    for (std::size_t problem = 0u; problem < 2u; ++problem) {
        for (std::size_t contact = 0u; contact < kMaxContacts; ++contact) {
            const auto& impulse = failureFirst.impulses[
                problem * kMaxContacts + contact
            ];
            failureRollback = failureRollback &&
                impulse.x == 0.0f &&
                impulse.y == 0.0f &&
                impulse.z == 0.0f &&
                impulse.w == 0.0f;
        }
    }
    const std::size_t rollbackProblem = 2u;
    const std::size_t rollbackContactBase = rollbackProblem * kMaxContacts;
    for (std::size_t contact = 0u;
         contact < failureBatch.headers[rollbackProblem].control.y;
         ++contact) {
        const auto& source = failureBatch.contacts[
            rollbackContactBase + contact
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
            rollbackContactBase + contact
        ];
        failureRollback = failureRollback &&
            std::abs(actual.x - checkpoint[0]) <= 1.0e-6 &&
            std::abs(actual.y - checkpoint[1]) <= 1.0e-6 &&
            std::abs(actual.z - checkpoint[2]) <= 1.0e-6;
    }

    bool deterministic = true;
    for (std::size_t replay = 1; replay < replays.size(); ++replay) {
        deterministic = deterministic &&
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

    std::size_t failedIslands = 0u;
    double maximumImpulseError = 0.0;
    double maximumObjectiveError = 0.0;
    double maximumNaturalResidual = 0.0;
    double maximumConeViolation = 0.0;
    std::uint32_t maximumIterations = 0u;
    std::uint64_t contactIterations = 0u;
    std::uint64_t contactsSolved = 0u;
    for (std::size_t problem = 0; problem < problemCount; ++problem) {
        const auto& expected = oracle[problem];
        const auto& actual = replays[0].statuses[problem];
        bool valid = expected.status == NUMI_TEMPORAL_CONE_ISLAND_SUCCESS &&
            actual.control.x == NUMI_TEMPORAL_CONE_ISLAND_SUCCESS &&
            actual.control.z == 1u;
        maximumNaturalResidual = std::max(
            maximumNaturalResidual,
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
        maximumIterations = std::max(maximumIterations, actual.control.y);
        contactsSolved += actual.control.w;
        contactIterations +=
            static_cast<std::uint64_t>(actual.control.w) * actual.control.y;
        for (std::size_t contact = 0;
             contact < batch.headers[problem].control.y;
             ++contact) {
            const auto& gpu = replays[0].impulses[
                problem * kMaxContacts + contact
            ];
            const auto& cpu = expected.impulses[contact];
            const std::array<double, 3> values{{gpu.x, gpu.y, gpu.z}};
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

    double totalSeconds = 0.0;
    for (const auto& replay : replays) {
        totalSeconds += replay.seconds;
    }
    const double averageSeconds = totalSeconds / replays.size();
    const double islandsPerSecond = averageSeconds > 0.0
        ? static_cast<double>(problemCount) / averageSeconds
        : 0.0;
    const double contactsPerSecond = averageSeconds > 0.0
        ? static_cast<double>(contactsSolved) / averageSeconds
        : 0.0;
    const double contactIterationsPerSecond = averageSeconds > 0.0
        ? static_cast<double>(contactIterations) / averageSeconds
        : 0.0;
    const std::uint64_t bufferBytes =
        batch.headers.size() * sizeof(NumiTemporalConeIslandHeader) +
        batch.matrices.size() * sizeof(float) +
        batch.contacts.size() * sizeof(NumiTemporalConeIslandContact) +
        batch.contacts.size() * sizeof(mr_float4) +
        batch.headers.size() * sizeof(NumiTemporalConeIslandStatus);
    const bool passed =
        failedIslands == 0u &&
        maximumImpulseError <= 2.0e-5 &&
        maximumObjectiveError <= 2.0e-5 &&
        maximumNaturalResidual <= 2.0e-6 &&
        maximumConeViolation <= 2.0e-6 &&
        deterministic &&
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
              << " max_natural_residual=" << maximumNaturalResidual
              << " max_cone_violation=" << maximumConeViolation
              << " max_iterations=" << maximumIterations << '\n'
              << "deterministic_replay="
              << (deterministic ? "true" : "false")
              << " typed_failures=" << (typedFailures ? "true" : "false")
              << " deterministic_failures="
              << (deterministicFailures ? "true" : "false")
              << " failure_rollback="
              << (failureRollback ? "true" : "false") << '\n'
              << "average_gpu_seconds=" << averageSeconds
              << " islands_per_second=" << islandsPerSecond
              << " contacts_per_second=" << contactsPerSecond
              << " contact_iterations_per_second="
              << contactIterationsPerSecond
              << " buffer_bytes=" << bufferBytes << '\n'
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
