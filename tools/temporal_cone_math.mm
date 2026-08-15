#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/temporal_cone_probe.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef NUMI_TEMPORAL_CONE_METALLIB
#error "NUMI_TEMPORAL_CONE_METALLIB must name the built solver metallib"
#endif

namespace {

constexpr double kMatrixFloor = 1.0e-10;
constexpr double kContactMatrixRegularization = 1.0e-2;
constexpr double kConeEpsilon = 1.0e-7;

struct OracleOutput {
    std::array<double, 3> impulse{};
    std::array<std::array<double, 3>, 3> inverse{};
    std::array<double, 3> residual{};
    double maximumDelta = 0.0;
    double coneViolation = 0.0;
    std::uint32_t status = NUMI_TEMPORAL_CONE_PROBE_SUCCESS;
};

struct GPUResult {
    std::vector<NumiTemporalConeProbeOutput> outputs;
    double seconds = 0.0;
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
    const std::uint32_t z = 0u,
    const std::uint32_t w = 0u
) {
    return {x, y, z, w};
}

NumiTemporalConeProbeInput makeInput(
    const std::array<std::array<float, 3>, 3>& response,
    const std::array<float, 3>& freeVelocity,
    const std::array<float, 3>& warmImpulse,
    const float frictionU,
    const float frictionV,
    const float maximumNormalImpulse = 0.0f,
    const std::uint32_t iterations = 8u
) {
    NumiTemporalConeProbeInput input{};
    input.responseRow0 = f4(
        response[0][0], response[0][1], response[0][2]
    );
    input.responseRow1 = f4(
        response[1][0], response[1][1], response[1][2]
    );
    input.responseRow2 = f4(
        response[2][0], response[2][1], response[2][2]
    );
    input.freeVelocityAndFrictionU = f4(
        freeVelocity[0], freeVelocity[1], freeVelocity[2], frictionU
    );
    input.warmImpulseAndFrictionV = f4(
        warmImpulse[0], warmImpulse[1], warmImpulse[2], frictionV
    );
    input.limits = f4(maximumNormalImpulse, 0.0f, 0.0f, 0.0f);
    input.control = u4(
        NUMI_TEMPORAL_CONE_PROBE_ABI_VERSION,
        iterations
    );
    return input;
}

double ellipseProjectionFunction(
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
        value +=
            tangent[axis] * tangent[axis] * radiiSquared[axis] /
            (denominator * denominator);
    }
    return value;
}

std::array<double, 2> projectTangentEllipse(
    const std::array<double, 2>& tangent,
    const std::array<double, 2>& radii
) {
    const std::array<double, 2> radiiSquared{{
        radii[0] * radii[0],
        radii[1] * radii[1],
    }};
    double normalizedSquared = 0.0;
    for (std::size_t axis = 0; axis < 2; ++axis) {
        if (radiiSquared[axis] > kMatrixFloor) {
            normalizedSquared +=
                tangent[axis] * tangent[axis] / radiiSquared[axis];
        } else if (tangent[axis] != 0.0) {
            normalizedSquared = std::numeric_limits<double>::infinity();
        }
    }
    if (normalizedSquared <= 1.0) {
        return tangent;
    }
    const bool activeX = radiiSquared[0] > kMatrixFloor;
    const bool activeY = radiiSquared[1] > kMatrixFloor;
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
    while (ellipseProjectionFunction(tangent, radiiSquared, upper) > 0.0) {
        upper *= 2.0;
    }
    for (std::uint32_t iteration = 0; iteration < 100u; ++iteration) {
        const double middle = 0.5 * (lower + upper);
        if (ellipseProjectionFunction(tangent, radiiSquared, middle) > 0.0) {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    const double multiplier = 0.5 * (lower + upper);
    std::array<double, 2> projected{};
    for (std::size_t axis = 0; axis < 2; ++axis) {
        projected[axis] = radiiSquared[axis] > kMatrixFloor
            ? tangent[axis] * radiiSquared[axis] /
                (radiiSquared[axis] + multiplier)
            : 0.0;
    }
    return projected;
}

double anisotropicConeBoundaryFunction(
    const std::array<double, 3>& impulse,
    const std::array<double, 2>& frictionSquared,
    const double multiplier
) {
    const double normal = impulse[0] + multiplier;
    double value = -1.0;
    for (std::size_t axis = 0; axis < 2; ++axis) {
        if (frictionSquared[axis] <= kMatrixFloor) {
            continue;
        }
        const double denominator =
            frictionSquared[axis] * normal + multiplier;
        if (!(denominator > kMatrixFloor)) {
            return std::numeric_limits<double>::infinity();
        }
        value +=
            impulse[axis + 1] * impulse[axis + 1] *
            frictionSquared[axis] /
            (denominator * denominator);
    }
    return value;
}

std::array<double, 3> projectCone(
    const std::array<double, 3> impulse,
    const double authoredFrictionU,
    const double authoredFrictionV,
    const double maximumNormalImpulse
) {
    const std::array<double, 2> friction{{
        std::max(authoredFrictionU, 0.0),
        std::max(authoredFrictionV, 0.0),
    }};
    const bool frictionless =
        friction[0] <= kConeEpsilon &&
        friction[1] <= kConeEpsilon;
    if (frictionless) {
        return {{
            maximumNormalImpulse > 0.0
                ? std::clamp(impulse[0], 0.0, maximumNormalImpulse)
                : std::max(impulse[0], 0.0),
            0.0,
            0.0,
        }};
    }

    const bool activeU = friction[0] > kConeEpsilon;
    const bool activeV = friction[1] > kConeEpsilon;
    if (activeU != activeV) {
        const double mu = activeU ? friction[0] : friction[1];
        const double tangent = activeU ? impulse[1] : impulse[2];
        std::array<double, 3> projected{{
            impulse[0],
            activeU ? tangent : 0.0,
            activeV ? tangent : 0.0,
        }};
        if (!(impulse[0] >= 0.0 &&
              std::abs(tangent) <= mu * impulse[0])) {
            if (impulse[0] + mu * std::abs(tangent) <= 0.0) {
                projected = {};
            } else {
                const double normal =
                    (impulse[0] + mu * std::abs(tangent)) /
                    (1.0 + mu * mu);
                const double projectedTangent =
                    std::copysign(mu * normal, tangent);
                projected = {{
                    normal,
                    activeU ? projectedTangent : 0.0,
                    activeV ? projectedTangent : 0.0,
                }};
            }
        }
        if (maximumNormalImpulse > 0.0 &&
            projected[0] > maximumNormalImpulse) {
            projected[0] = maximumNormalImpulse;
            const double limitedTangent = std::clamp(
                tangent,
                -mu * maximumNormalImpulse,
                mu * maximumNormalImpulse
            );
            projected[1] = activeU ? limitedTangent : 0.0;
            projected[2] = activeV ? limitedTangent : 0.0;
        }
        return projected;
    }

    double normalizedTangentSquared = 0.0;
    for (std::size_t axis = 0; axis < 2; ++axis) {
        if (friction[axis] > kConeEpsilon) {
            normalizedTangentSquared +=
                impulse[axis + 1] * impulse[axis + 1] /
                (friction[axis] * friction[axis]);
        } else if (impulse[axis + 1] != 0.0) {
            normalizedTangentSquared =
                std::numeric_limits<double>::infinity();
        }
    }
    const bool insideUnbounded =
        impulse[0] >= 0.0 &&
        normalizedTangentSquared <= impulse[0] * impulse[0];
    std::array<double, 3> projected = impulse;
    if (!insideUnbounded) {
        const double weightedDual = std::hypot(
            friction[0] * impulse[1],
            friction[1] * impulse[2]
        );
        if (impulse[0] + weightedDual <= 0.0) {
            projected = {};
        } else {
            const bool isotropic =
                activeU && activeV &&
                std::abs(friction[0] - friction[1]) <=
                    8.0 * std::numeric_limits<float>::epsilon() *
                    std::max({friction[0], friction[1], 1.0});
            if (isotropic) {
                const double tangentNorm = std::hypot(
                    impulse[1], impulse[2]
                );
                const double mu = 0.5 * (friction[0] + friction[1]);
                const double normal =
                    (impulse[0] + mu * tangentNorm) /
                    (1.0 + mu * mu);
                const double tangentScale = tangentNorm > kConeEpsilon
                    ? mu * normal / tangentNorm
                    : 0.0;
                projected = {{
                    normal,
                    tangentScale * impulse[1],
                    tangentScale * impulse[2],
                }};
            } else {
                const std::array<double, 2> frictionSquared{{
                    friction[0] * friction[0],
                    friction[1] * friction[1],
                }};
                double lower = std::max(0.0, -impulse[0]);
                double upper = std::max(
                    lower + std::max(weightedDual, 1.0),
                    1.0
                );
                while (anisotropicConeBoundaryFunction(
                           impulse,
                           frictionSquared,
                           upper
                       ) > 0.0) {
                    upper = 2.0 * upper + std::max(weightedDual, 1.0);
                }
                for (std::uint32_t iteration = 0;
                     iteration < 100u;
                     ++iteration) {
                    const double middle = 0.5 * (lower + upper);
                    if (anisotropicConeBoundaryFunction(
                            impulse,
                            frictionSquared,
                            middle
                        ) > 0.0) {
                        lower = middle;
                    } else {
                        upper = middle;
                    }
                }
                const double multiplier = 0.5 * (lower + upper);
                const double normal = std::max(
                    impulse[0] + multiplier,
                    0.0
                );
                projected[0] = normal;
                for (std::size_t axis = 0; axis < 2; ++axis) {
                    projected[axis + 1] =
                        frictionSquared[axis] > kMatrixFloor
                        ? impulse[axis + 1] * frictionSquared[axis] * normal /
                            (frictionSquared[axis] * normal + multiplier)
                        : 0.0;
                }
            }
        }
    }

    if (maximumNormalImpulse > 0.0 &&
        projected[0] > maximumNormalImpulse) {
        projected[0] = maximumNormalImpulse;
        const auto tangent = projectTangentEllipse(
            {{impulse[1], impulse[2]}},
            {{
                friction[0] * maximumNormalImpulse,
                friction[1] * maximumNormalImpulse,
            }}
        );
        projected[1] = tangent[0];
        projected[2] = tangent[1];
    }
    return projected;
}

double coneViolation(
    const std::array<double, 3>& impulse,
    const double frictionU,
    const double frictionV,
    const double maximumNormalImpulse
) {
    const double normal = std::max(impulse[0], 0.0);
    double violation = std::max(-impulse[0], 0.0);
    if (maximumNormalImpulse > 0.0) {
        violation = std::max(
            violation,
            std::max(impulse[0] - maximumNormalImpulse, 0.0)
        );
    }
    std::array<double, 2> normalizedTangent{};
    bool hasActiveTangent = false;
    const std::array<double, 2> friction{{frictionU, frictionV}};
    for (std::size_t axis = 0u; axis < 2u; ++axis) {
        if (friction[axis] > kConeEpsilon) {
            hasActiveTangent = true;
            normalizedTangent[axis] =
                impulse[axis + 1u] / friction[axis];
        } else {
            violation = std::max(
                violation, std::abs(impulse[axis + 1u])
            );
        }
    }
    if (hasActiveTangent) {
        violation = std::max(
            violation,
            std::max(
                std::hypot(
                    normalizedTangent[0], normalizedTangent[1]
                ) - normal,
                0.0
            )
        );
    }
    return violation;
}

bool conditionedInverse(
    const std::array<std::array<double, 3>, 3>& matrix,
    std::array<std::array<double, 3>, 3>& inverse
) {
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
    const double inverseScale = 1.0 / scale;
    if (!(inverseScale > 0.0) || !std::isfinite(inverseScale)) {
        return false;
    }

    std::array<std::array<double, 3>, 3> regularized{};
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            regularized[row][column] =
                0.5 * matrix[row][column] * inverseScale +
                0.5 * matrix[column][row] * inverseScale;
        }
    }

    const double unshiftedMinor01 =
        regularized[0][0] * regularized[1][1] -
        regularized[0][1] * regularized[1][0];
    const double unshiftedMinor02 =
        regularized[0][0] * regularized[2][2] -
        regularized[0][2] * regularized[2][0];
    const double unshiftedMinor12 =
        regularized[1][1] * regularized[2][2] -
        regularized[1][2] * regularized[2][1];
    const double unshiftedDeterminant =
        regularized[0][0] * (
            regularized[1][1] * regularized[2][2] -
            regularized[1][2] * regularized[2][1]
        ) -
        regularized[0][1] * (
            regularized[1][0] * regularized[2][2] -
            regularized[1][2] * regularized[2][0]
        ) +
        regularized[0][2] * (
            regularized[1][0] * regularized[2][1] -
            regularized[1][1] * regularized[2][0]
        );
    constexpr double psdTolerance =
        64.0 * std::numeric_limits<float>::epsilon();
    if (regularized[0][0] < -psdTolerance ||
        regularized[1][1] < -psdTolerance ||
        regularized[2][2] < -psdTolerance ||
        unshiftedMinor01 < -psdTolerance ||
        unshiftedMinor02 < -psdTolerance ||
        unshiftedMinor12 < -psdTolerance ||
        unshiftedDeterminant < -psdTolerance ||
        !std::isfinite(unshiftedMinor01) ||
        !std::isfinite(unshiftedMinor02) ||
        !std::isfinite(unshiftedMinor12) ||
        !std::isfinite(unshiftedDeterminant)) {
        return false;
    }
    for (std::size_t axis = 0u; axis < 3u; ++axis) {
        regularized[axis][axis] += kContactMatrixRegularization;
    }

    const double c00 =
        regularized[1][1] * regularized[2][2] -
        regularized[1][2] * regularized[2][1];
    const double c01 =
        regularized[1][2] * regularized[2][0] -
        regularized[1][0] * regularized[2][2];
    const double c02 =
        regularized[1][0] * regularized[2][1] -
        regularized[1][1] * regularized[2][0];
    const double determinant =
        regularized[0][0] * c00 +
        regularized[0][1] * c01 +
        regularized[0][2] * c02;
    const double leadingMinor1 = regularized[0][0];
    const double leadingMinor2 =
        regularized[0][0] * regularized[1][1] -
        regularized[0][1] * regularized[1][0];
    if (!(leadingMinor1 > kMatrixFloor) ||
        !(leadingMinor2 > kMatrixFloor) ||
        !(determinant > kMatrixFloor) ||
        !std::isfinite(leadingMinor1) ||
        !std::isfinite(leadingMinor2) ||
        !std::isfinite(determinant)) {
        return false;
    }
    const double reciprocal = inverseScale / determinant;
    if (!(reciprocal > 0.0) || !std::isfinite(reciprocal)) {
        return false;
    }
    inverse[0][0] = c00 * reciprocal;
    inverse[0][1] =
        (regularized[0][2] * regularized[2][1] -
         regularized[0][1] * regularized[2][2]) * reciprocal;
    inverse[0][2] =
        (regularized[0][1] * regularized[1][2] -
         regularized[0][2] * regularized[1][1]) * reciprocal;
    inverse[1][0] = c01 * reciprocal;
    inverse[1][1] =
        (regularized[0][0] * regularized[2][2] -
         regularized[0][2] * regularized[2][0]) * reciprocal;
    inverse[1][2] =
        (regularized[0][2] * regularized[1][0] -
         regularized[0][0] * regularized[1][2]) * reciprocal;
    inverse[2][0] = c02 * reciprocal;
    inverse[2][1] =
        (regularized[0][1] * regularized[2][0] -
         regularized[0][0] * regularized[2][1]) * reciprocal;
    inverse[2][2] =
        (regularized[0][0] * regularized[1][1] -
         regularized[0][1] * regularized[1][0]) * reciprocal;
    for (const auto& row : inverse) {
        for (const double value : row) {
            if (!std::isfinite(value)) {
                return false;
            }
        }
    }
    return true;
}

OracleOutput solveOracle(const NumiTemporalConeProbeInput& input) {
    OracleOutput output;
    if (input.control.x != NUMI_TEMPORAL_CONE_PROBE_ABI_VERSION) {
        output.status = NUMI_TEMPORAL_CONE_PROBE_INVALID_ABI;
        return output;
    }
    if (input.control.y == 0u ||
        input.control.y > NUMI_TEMPORAL_CONE_PROBE_MAX_ITERATIONS) {
        output.status = NUMI_TEMPORAL_CONE_PROBE_INVALID_INPUT;
        return output;
    }

    const std::array<std::array<double, 3>, 3> response{{
        {{input.responseRow0.x, input.responseRow0.y, input.responseRow0.z}},
        {{input.responseRow1.x, input.responseRow1.y, input.responseRow1.z}},
        {{input.responseRow2.x, input.responseRow2.y, input.responseRow2.z}},
    }};
    if (!conditionedInverse(response, output.inverse)) {
        output.status = NUMI_TEMPORAL_CONE_PROBE_FACTORIZATION_FAILED;
        return output;
    }

    const std::array<double, 3> freeVelocity{{
        input.freeVelocityAndFrictionU.x,
        input.freeVelocityAndFrictionU.y,
        input.freeVelocityAndFrictionU.z,
    }};
    const double frictionU = input.freeVelocityAndFrictionU.w;
    const double frictionV = input.warmImpulseAndFrictionV.w;
    const double maximumNormalImpulse = input.limits.x;
    output.impulse = projectCone(
        {{
            input.warmImpulseAndFrictionV.x,
            input.warmImpulseAndFrictionV.y,
            input.warmImpulseAndFrictionV.z,
        }},
        frictionU,
        frictionV,
        maximumNormalImpulse
    );

    for (std::uint32_t iteration = 0u;
         iteration < input.control.y;
         ++iteration) {
        std::array<double, 3> residual{};
        for (std::size_t row = 0; row < 3; ++row) {
            residual[row] = freeVelocity[row];
            for (std::size_t column = 0; column < 3; ++column) {
                residual[row] +=
                    response[row][column] * output.impulse[column];
            }
        }
        std::array<double, 3> proposed = output.impulse;
        for (std::size_t row = 0; row < 3; ++row) {
            for (std::size_t column = 0; column < 3; ++column) {
                proposed[row] -=
                    output.inverse[row][column] * residual[column];
            }
        }
        const auto candidate = projectCone(
            proposed,
            frictionU,
            frictionV,
            maximumNormalImpulse
        );
        output.maximumDelta = 0.0;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            output.maximumDelta = std::max(
                output.maximumDelta,
                std::abs(candidate[axis] - output.impulse[axis])
            );
        }
        output.impulse = candidate;
    }

    for (std::size_t row = 0; row < 3; ++row) {
        output.residual[row] = freeVelocity[row];
        for (std::size_t column = 0; column < 3; ++column) {
            output.residual[row] +=
                response[row][column] * output.impulse[column];
        }
    }
    output.coneViolation = coneViolation(
        output.impulse,
        frictionU,
        frictionV,
        maximumNormalImpulse
    );
    return output;
}

std::vector<NumiTemporalConeProbeInput> makeProblems(
    const std::size_t count,
    const bool isotropicBatch,
    const std::uint32_t iterations
) {
    const std::array<std::array<float, 3>, 3> identity{{
        {{1.0f, 0.0f, 0.0f}},
        {{0.0f, 1.0f, 0.0f}},
        {{0.0f, 0.0f, 1.0f}},
    }};
    std::vector<NumiTemporalConeProbeInput> inputs;
    inputs.reserve(count);
    const auto appendFixedProjection = [&inputs, &identity](
        const std::array<float, 3>& rawImpulse,
        const float frictionU,
        const float frictionV,
        const float maximumNormalImpulse = 0.0f
    ) {
        const auto projected = projectCone(
            {{
                static_cast<double>(rawImpulse[0]),
                static_cast<double>(rawImpulse[1]),
                static_cast<double>(rawImpulse[2]),
            }},
            frictionU,
            frictionV,
            maximumNormalImpulse
        );
        inputs.push_back(makeInput(
            identity,
            {{
                -static_cast<float>(projected[0]),
                -static_cast<float>(projected[1]),
                -static_cast<float>(projected[2]),
            }},
            rawImpulse,
            frictionU,
            frictionV,
            maximumNormalImpulse
        ));
    };
    inputs.push_back(makeInput(
        identity, {{1.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f}},
        0.6f, 0.6f
    ));
    inputs.push_back(makeInput(
        identity, {{-2.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f}},
        0.0f, 0.0f
    ));
    inputs.push_back(makeInput(
        identity, {{-1.0f, -0.2f, 0.1f}}, {{0.1f, 0.0f, 0.0f}},
        0.6f, 0.6f
    ));
    inputs.push_back(makeInput(
        identity, {{-1.0f, -2.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f}},
        0.5f, 0.5f
    ));
    inputs.push_back(makeInput(
        identity, {{-1.0f, -1.0f, 1.0f}}, {{0.0f, 0.0f, 0.0f}},
        0.8f, 0.35f
    ));
    inputs.push_back(makeInput(
        {{{{1.0f, 0.0f, 0.0f}},
          {{0.0f, 1.0e-10f, 0.0f}},
          {{0.0f, 0.0f, 1.0e-10f}}}},
        {{-0.5f, -0.1f, 0.05f}}, {{0.0f, 0.0f, 0.0f}},
        0.7f, 0.4f
    ));
    inputs.push_back(makeInput(
        identity, {{-5.0f, -1.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f}},
        0.5f, 0.5f, 1.25f
    ));
    inputs.push_back(makeInput(
        {{{{1.2f, 0.08f, -0.02f}},
          {{0.07f, 0.9f, 0.04f}},
          {{-0.01f, 0.03f, 0.7f}}}},
        {{-0.8f, -0.4f, 0.2f}}, {{0.05f, 0.01f, -0.01f}},
        0.55f, 0.45f
    ));
    inputs.push_back(makeInput(
        {{{{0.0f, 0.0f, 0.0f}},
          {{0.0f, 0.0f, 0.0f}},
          {{0.0f, 0.0f, 0.0f}}}},
        {{-1.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f}},
        0.5f, 0.5f
    ));
    inputs.push_back(makeInput(
        identity, {{-1.0f, -0.5f, 0.25f}}, {{0.2f, 0.1f, -0.1f}},
        0.5f, 0.0f
    ));
    inputs.push_back(makeInput(
        identity, {{-1.0f, -100.0f, 50.0f}}, {{0.0f, 0.0f, 0.0f}},
        0.01f, 10.0f
    ));
    inputs.push_back(makeInput(
        {{{{2.0e5f, 0.0f, 0.0f}},
          {{0.0f, 1.0e5f, 0.0f}},
          {{0.0f, 0.0f, 3.0e5f}}}},
        {{-1.0e6f, -2.0e6f, 1.0e6f}}, {{0.0f, 0.0f, 0.0f}},
        0.03f, 4.0f, 20.0f
    ));
    inputs.push_back(makeInput(
        identity, {{1.0f, -100.0001f, 0.0f}}, {{0.0f, 0.0f, 0.0f}},
        0.01f, 1.0f
    ));
    inputs.push_back(makeInput(
        identity, {{-1.0f, 0.4f, -0.6f}}, {{0.0f, 0.0f, 0.0f}},
        0.0f, 0.75f
    ));
    inputs.push_back(makeInput(
        identity, {{-5.0f, -2.0f, -2.0f}}, {{0.0f, 0.0f, 0.0f}},
        0.0f, 0.5f, 1.0f
    ));
    inputs.push_back(makeInput(
        identity, {{-1.0f, -5.0e-8f, -0.3f}}, {{0.0f, 0.0f, 0.0f}},
        0.0f, 0.5f
    ));
    inputs.push_back(makeInput(
        {{{{1.0f, 0.0f, 0.0f}},
          {{0.0f, -0.005f, 0.0f}},
          {{0.0f, 0.0f, -0.005f}}}},
        {{-1.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f}},
        0.5f, 0.5f
    ));
    appendFixedProjection(
        {{1.0e20f, 1.0e20f, 0.0f}},
        0.5f,
        0.5f
    );
    appendFixedProjection(
        {{1.0e20f, 1.0e20f, 0.0f}},
        0.5f,
        0.5f,
        1.0e20f
    );
    appendFixedProjection(
        {{4.0e19f, 1.0e20f, -8.0e19f}},
        0.8f,
        0.35f
    );
    appendFixedProjection(
        {{1.0e6f, 1.5e6f, 0.0f}},
        1.0f,
        1.0f
    );
    inputs.back().control.z = 1u;
    appendFixedProjection(
        {{1.0f, 2.0f, 0.1f}},
        0.0f,
        0.5f
    );

    for (std::size_t index = inputs.size(); index < count; ++index) {
        const float a = 0.25f + 0.01f * static_cast<float>(index % 71u);
        const float b = 0.35f + 0.01f * static_cast<float>(index % 53u);
        const float c = 0.45f + 0.01f * static_cast<float>(index % 37u);
        const float d = 0.03f * static_cast<float>(
            static_cast<int>(index % 9u) - 4
        );
        const float e = 0.02f * static_cast<float>(
            static_cast<int>(index % 7u) - 3
        );
        const float f = 0.025f * static_cast<float>(
            static_cast<int>(index % 5u) - 2
        );
        const std::array<std::array<float, 3>, 3> response{{
            {{a * a, a * d, a * e}},
            {{a * d, d * d + b * b, d * e + b * f}},
            {{a * e, d * e + b * f, e * e + f * f + c * c}},
        }};
        const float normalVelocity =
            index % 5u == 0u
            ? 0.1f + 0.01f * static_cast<float>(index % 17u)
            : -0.05f - 0.01f * static_cast<float>(index % 113u);
        const std::array<float, 3> freeVelocity{{
            normalVelocity,
            0.5f * std::sin(static_cast<float>(index) * 0.17f),
            0.5f * std::cos(static_cast<float>(index) * 0.11f),
        }};
        const std::array<float, 3> warm{{
            0.01f * static_cast<float>(index % 13u),
            0.005f * static_cast<float>(
                static_cast<int>(index % 7u) - 3
            ),
            0.004f * static_cast<float>(
                static_cast<int>(index % 9u) - 4
            ),
        }};
        const float frictionU =
            0.15f + 0.05f * static_cast<float>(index % 12u);
        const float frictionV = isotropicBatch
            ? frictionU
            : 0.20f + 0.04f * static_cast<float>(index % 10u);
        inputs.push_back(makeInput(
            response,
            freeVelocity,
            warm,
            frictionU,
            frictionV,
            index % 29u == 0u ? 2.0f : 0.0f,
            8u
        ));
    }
    for (auto& input : inputs) {
        input.control.y = iterations;
    }
    return inputs;
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
    const std::vector<NumiTemporalConeProbeInput>& inputs
) {
    const NSUInteger inputBytes =
        inputs.size() * sizeof(NumiTemporalConeProbeInput);
    const NSUInteger outputBytes =
        inputs.size() * sizeof(NumiTemporalConeProbeOutput);
    id<MTLBuffer> inputBuffer = [device
        newBufferWithBytes:inputs.data()
                   length:inputBytes
                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [device
        newBufferWithLength:outputBytes
                    options:MTLResourceStorageModeShared];
    if (inputBuffer == nil || outputBuffer == nil) {
        throw std::runtime_error("failed to allocate shared Metal buffers");
    }
    std::memset(outputBuffer.contents, 0, outputBytes);

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (commandBuffer == nil || encoder == nil) {
        throw std::runtime_error("failed to create Metal command encoder");
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:inputBuffer offset:0 atIndex:0];
    [encoder setBuffer:outputBuffer offset:0 atIndex:1];
    const std::uint32_t problemCount =
        static_cast<std::uint32_t>(inputs.size());
    [encoder setBytes:&problemCount length:sizeof(problemCount) atIndex:2];
    const NSUInteger width = std::min<NSUInteger>(
        pipeline.maxTotalThreadsPerThreadgroup,
        256u
    );
    [encoder dispatchThreads:MTLSizeMake(inputs.size(), 1u, 1u)
          threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted ||
        commandBuffer.error != nil) {
        throw std::runtime_error(
            "Metal command failed: " + errorText(commandBuffer.error)
        );
    }

    GPUResult result;
    const auto* values = static_cast<const NumiTemporalConeProbeOutput*>(
        outputBuffer.contents
    );
    result.outputs.assign(values, values + inputs.size());
    if (commandBuffer.GPUEndTime >= commandBuffer.GPUStartTime) {
        result.seconds =
            commandBuffer.GPUEndTime - commandBuffer.GPUStartTime;
    }
    return result;
}

double normalizedError(const double actual, const double expected) {
    return std::abs(actual - expected) /
        std::max(1.0, std::abs(expected));
}

int run(const int argc, const char* const* argv) {
    std::size_t problemCount = 65536u;
    std::uint32_t replayCount = 2u;
    std::string metallibPath = NUMI_TEMPORAL_CONE_METALLIB;
    bool isotropicBatch = false;
    std::uint32_t solverIterations = 16u;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string_view value(argv[argument]);
        if (value == "--cases" && argument + 1 < argc) {
            problemCount = std::stoull(argv[++argument]);
        } else if (value == "--replays" && argument + 1 < argc) {
            replayCount = static_cast<std::uint32_t>(
                std::stoul(argv[++argument])
            );
        } else if (value == "--metallib" && argument + 1 < argc) {
            metallibPath = argv[++argument];
        } else if (value == "--isotropic") {
            isotropicBatch = true;
        } else if (value == "--iterations" && argument + 1 < argc) {
            solverIterations = static_cast<std::uint32_t>(
                std::stoul(argv[++argument])
            );
        } else if (value == "--help") {
            std::cout
                << "usage: numi-solver-math [--cases N] [--replays N] "
                   "[--iterations N] [--metallib PATH] [--isotropic]\n";
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + std::string(value));
        }
    }
    problemCount = std::max<std::size_t>(problemCount, 22u);
    replayCount = std::max<std::uint32_t>(replayCount, 2u);
    if (problemCount > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("case count exceeds the probe ABI");
    }
    if (solverIterations == 0u ||
        solverIterations > NUMI_TEMPORAL_CONE_PROBE_MAX_ITERATIONS) {
        throw std::runtime_error("iterations must be in [1, 64]");
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        throw std::runtime_error("no Apple Metal device is available");
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
        throw std::runtime_error("failed to create the Metal command queue");
    }
    NSString* metalLibraryPath = [NSString
        stringWithUTF8String:metallibPath.c_str()];
    NSError* error = nil;
    id<MTLLibrary> library = [device
        newLibraryWithURL:[NSURL fileURLWithPath:metalLibraryPath]
                    error:&error];
    if (library == nil) {
        throw std::runtime_error(
            "failed to load solver metallib: " + errorText(error)
        );
    }
    id<MTLFunction> function =
        [library newFunctionWithName:@"numi_temporal_cone_probe"];
    if (function == nil) {
        throw std::runtime_error("solver metallib lacks the math probe kernel");
    }
    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:function
                                      error:&error];
    if (pipeline == nil) {
        throw std::runtime_error(
            "failed to create the math probe pipeline: " + errorText(error)
        );
    }

    const auto inputs = makeProblems(
        problemCount,
        isotropicBatch,
        solverIterations
    );
    std::vector<GPUResult> replays;
    replays.reserve(replayCount);
    for (std::uint32_t replay = 0u; replay < replayCount; ++replay) {
        replays.push_back(runGPU(device, queue, pipeline, inputs));
    }

    bool deterministic = true;
    for (std::size_t replay = 1; replay < replays.size(); ++replay) {
        deterministic = deterministic &&
            std::memcmp(
                replays[0].outputs.data(),
                replays[replay].outputs.data(),
                replays[0].outputs.size() *
                    sizeof(NumiTemporalConeProbeOutput)
            ) == 0;
    }

    std::size_t failedCases = 0u;
    double maximumRelativeError = 0.0;
    std::size_t maximumRelativeErrorCase = 0u;
    double maximumConeViolation = 0.0;
    double maximumFixedPointResidual = 0.0;
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        const auto oracle = solveOracle(inputs[index]);
        const auto& gpu = replays[0].outputs[index];
        bool valid = gpu.status.x == oracle.status;
        if (gpu.status.x == NUMI_TEMPORAL_CONE_PROBE_SUCCESS) {
            const std::array<double, 3> gpuImpulse{{
                gpu.impulseAndDelta.x,
                gpu.impulseAndDelta.y,
                gpu.impulseAndDelta.z,
            }};
            const std::array<double, 3> gpuResidual{{
                gpu.residualAndConeViolation.x,
                gpu.residualAndConeViolation.y,
                gpu.residualAndConeViolation.z,
            }};
            for (std::size_t axis = 0; axis < 3; ++axis) {
                const double impulseError = normalizedError(
                    gpuImpulse[axis], oracle.impulse[axis]
                );
                const double residualError = normalizedError(
                    gpuResidual[axis], oracle.residual[axis]
                );
                if (impulseError > maximumRelativeError) {
                    maximumRelativeError = impulseError;
                    maximumRelativeErrorCase = index;
                }
                if (residualError > maximumRelativeError) {
                    maximumRelativeError = residualError;
                    maximumRelativeErrorCase = index;
                }
                valid = valid && std::isfinite(gpuImpulse[axis]);
            }
            maximumConeViolation = std::max(
                maximumConeViolation,
                static_cast<double>(gpu.residualAndConeViolation.w)
            );
            const double impulseScale = std::max({
                1.0,
                std::abs(gpuImpulse[0]),
                std::abs(gpuImpulse[1]),
                std::abs(gpuImpulse[2]),
            });
            maximumFixedPointResidual = std::max(
                maximumFixedPointResidual,
                static_cast<double>(gpu.impulseAndDelta.w) / impulseScale
            );
            valid = valid && gpuImpulse[0] >= -1.0e-6;
            if (inputs[index].limits.x > 0.0f) {
                valid = valid &&
                    gpuImpulse[0] <= inputs[index].limits.x + 1.0e-5;
            }
            valid = valid && gpu.residualAndConeViolation.w <= 2.0e-5f;
        }
        if (!valid) {
            ++failedCases;
        }
    }

    const auto& separating = replays[0].outputs[0];
    const bool separatingAcceptedZero =
        std::abs(separating.impulseAndDelta.x) <= 1.0e-6f &&
        std::abs(separating.impulseAndDelta.y) <= 1.0e-6f &&
        std::abs(separating.impulseAndDelta.z) <= 1.0e-6f;
    const auto& sliding = replays[0].outputs[3];
    const double slidingLimit = 0.5 * sliding.impulseAndDelta.x;
    const bool slidingOnCone =
        sliding.impulseAndDelta.x > 0.0f &&
        std::abs(std::abs(sliding.impulseAndDelta.y) - slidingLimit) <=
            2.0e-5;
    const auto& capped = replays[0].outputs[6];
    const bool normalCapRespected =
        capped.impulseAndDelta.x <= 1.25f + 1.0e-5f;
    const auto& zeroV = replays[0].outputs[9];
    const bool zeroVAxisCone =
        std::abs(zeroV.impulseAndDelta.y) > 1.0e-3f &&
        zeroV.impulseAndDelta.z == 0.0f;
    const auto& zeroU = replays[0].outputs[13];
    const bool zeroUAxisCone =
        zeroU.impulseAndDelta.y == 0.0f &&
        std::abs(zeroU.impulseAndDelta.z) > 1.0e-3f;
    const auto& cappedZeroU = replays[0].outputs[14];
    const bool degenerateCap =
        std::abs(cappedZeroU.impulseAndDelta.x - 1.0f) <= 1.0e-6f &&
        cappedZeroU.impulseAndDelta.y == 0.0f &&
        std::abs(std::abs(cappedZeroU.impulseAndDelta.z) - 0.5f) <=
            2.0e-5f;
    const auto& tinyInactive = replays[0].outputs[15];
    const bool exactInactiveAxis =
        tinyInactive.impulseAndDelta.y == 0.0f &&
        std::abs(tinyInactive.impulseAndDelta.z) > 1.0e-3f;
    const bool indefiniteLocalBlockRejected =
        replays[0].outputs[16].status.x ==
            NUMI_TEMPORAL_CONE_PROBE_FACTORIZATION_FAILED;
    const auto extremeProjectionAccepted = [&](const std::size_t index) {
        const auto oracle = solveOracle(inputs[index]);
        const auto& gpu = replays[0].outputs[index];
        if (oracle.status != NUMI_TEMPORAL_CONE_PROBE_SUCCESS ||
            gpu.status.x != NUMI_TEMPORAL_CONE_PROBE_SUCCESS ||
            gpu.residualAndConeViolation.w > 2.0e-6f) {
            return false;
        }
        const std::array<double, 3> actual{{
            gpu.impulseAndDelta.x,
            gpu.impulseAndDelta.y,
            gpu.impulseAndDelta.z,
        }};
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            if (!std::isfinite(actual[axis]) ||
                normalizedError(actual[axis], oracle.impulse[axis]) >
                    1.0e-5) {
                return false;
            }
        }
        return true;
    };
    const bool extremeUnboundedProjection =
        extremeProjectionAccepted(17u);
    const bool extremeCappedProjection =
        extremeProjectionAccepted(18u);
    const bool extremeAnisotropicProjection =
        extremeProjectionAccepted(19u);
    const double dimensionalViolationOracle = coneViolation(
        {{1.0e6, 1.5e6, 0.0}},
        1.0,
        1.0,
        0.0
    );
    const double dimensionalViolationGPU =
        replays[0].outputs[20u].inverseRow0.w;
    const bool dimensionalConeCertificate =
        normalizedError(
            dimensionalViolationGPU,
            dimensionalViolationOracle
        ) <= 1.0e-6 &&
        dimensionalViolationGPU > 1.0e5;
    const auto& oneAxisInterior = replays[0].outputs[21u];
    const bool inactiveAxisDoesNotForceBoundary =
        std::abs(oneAxisInterior.impulseAndDelta.x - 1.0f) <= 1.0e-6f &&
        oneAxisInterior.impulseAndDelta.y == 0.0f &&
        std::abs(oneAxisInterior.impulseAndDelta.z - 0.1f) <= 1.0e-6f;

    double totalGPUSeconds = 0.0;
    for (const auto& replay : replays) {
        totalGPUSeconds += replay.seconds;
    }
    const double averageGPUSeconds =
        totalGPUSeconds / static_cast<double>(replays.size());
    const double casesPerSecond = averageGPUSeconds > 0.0
        ? static_cast<double>(problemCount) / averageGPUSeconds
        : 0.0;

    const bool passed =
        failedCases == 0u &&
        maximumRelativeError <= 1.0e-5 &&
        maximumConeViolation <= 2.0e-6 &&
        maximumFixedPointResidual <= 2.0e-6 &&
        deterministic &&
        separatingAcceptedZero &&
        slidingOnCone &&
        normalCapRespected &&
        zeroVAxisCone &&
        zeroUAxisCone &&
        degenerateCap &&
        exactInactiveAxis &&
        indefiniteLocalBlockRejected &&
        extremeUnboundedProjection &&
        extremeCappedProjection &&
        extremeAnisotropicProjection &&
        dimensionalConeCertificate &&
        inactiveAxisDoesNotForceBoundary;

    if (maximumRelativeError > 1.0e-5) {
        const auto worstOracle = solveOracle(inputs[maximumRelativeErrorCase]);
        const auto& worstGPU = replays[0].outputs[maximumRelativeErrorCase];
        std::cerr
            << "worst_case=" << maximumRelativeErrorCase
            << " gpu_impulse=(" << worstGPU.impulseAndDelta.x
            << ',' << worstGPU.impulseAndDelta.y
            << ',' << worstGPU.impulseAndDelta.z << ')'
            << " fp64_impulse=(" << worstOracle.impulse[0]
            << ',' << worstOracle.impulse[1]
            << ',' << worstOracle.impulse[2] << ")\n";
    }

    std::cout << std::fixed << std::setprecision(9)
              << "device=" << device.name.UTF8String << '\n'
              << "cases=" << problemCount
              << " replays=" << replayCount
              << " friction_batch="
              << (isotropicBatch ? "isotropic" : "anisotropic")
              << " solver_iterations=" << solverIterations
              << " failed_cases=" << failedCases << '\n'
              << "max_fp64_relative_error=" << maximumRelativeError
              << " max_error_case=" << maximumRelativeErrorCase
              << " max_cone_violation=" << maximumConeViolation
              << " max_relative_fixed_point_residual="
              << maximumFixedPointResidual << '\n'
              << "dimensional_violation_probe="
              << dimensionalViolationGPU << '\n'
              << "deterministic_replay=" << (deterministic ? "true" : "false")
              << " separating_zero="
              << (separatingAcceptedZero ? "true" : "false")
              << " sliding_on_cone=" << (slidingOnCone ? "true" : "false")
              << " normal_cap=" << (normalCapRespected ? "true" : "false")
              << " zero_v_axis_cone=" << (zeroVAxisCone ? "true" : "false")
              << " zero_u_axis_cone=" << (zeroUAxisCone ? "true" : "false")
              << " degenerate_cap=" << (degenerateCap ? "true" : "false")
              << " exact_inactive_axis="
              << (exactInactiveAxis ? "true" : "false")
              << " indefinite_local_block_rejected="
              << (indefiniteLocalBlockRejected ? "true" : "false")
              << " extreme_unbounded_projection="
              << (extremeUnboundedProjection ? "true" : "false")
              << " extreme_capped_projection="
              << (extremeCappedProjection ? "true" : "false")
              << " extreme_anisotropic_projection="
              << (extremeAnisotropicProjection ? "true" : "false")
              << " dimensional_cone_certificate="
              << (dimensionalConeCertificate ? "true" : "false")
              << " inactive_axis_interior_projection="
              << (inactiveAxisDoesNotForceBoundary ? "true" : "false")
              << '\n'
              << "average_gpu_seconds=" << averageGPUSeconds
              << " cases_per_second=" << casesPerSecond << '\n'
              << "result=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}

} // namespace

int main(const int argc, const char* const* argv) {
    @autoreleasepool {
        try {
            return run(argc, argv);
        } catch (const std::exception& error) {
            std::cerr << "numi-solver-math: " << error.what() << '\n';
            return 2;
        }
    }
}
