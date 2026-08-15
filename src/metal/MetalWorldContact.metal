#include <metal_stdlib>

#include "metalrobo/engine_types.h"
#include "metalrobo/rod_gpu_shared.h"
#include "metalrobo/unified_quality_shared.h"
#include "numi/temporal_cone_island.h"
#include "numi/temporal_cone_probe.h"

using namespace metal;

namespace {

constant float kQuaternionMinimum = 1.0e-12f;
constant float kMatrixFloor = 1.0e-10f;
constant float kConeEpsilon = 1.0e-7f;
constant float kFloatEpsilon = 1.1920928955078125e-7f;
constant uint kConeProjectionIterations = 28u;
// The coupled point response is scale-normalized below. Keep its smallest
// resolved mode at one percent of the dominant response so redundant or
// nearly rank-deficient articulated contacts cannot turn a bounded target
// velocity into an unbounded impulse correction.
constant float kContactMatrixRegularization = 1.0e-2f;
constant float kConeProjectionSafeMagnitude = 1.0e18f;

struct Mat3 {
    float3 row0;
    float3 row1;
    float3 row2;
};

inline bool finite3(const float3 value) {
    return all(isfinite(value));
}

inline bool finite4(const float4 value) {
    return all(isfinite(value));
}

inline float4 quaternionMultiply(
    const float4 left,
    const float4 right
) {
    return float4(
        left.w * right.x + right.w * left.x +
            left.y * right.z - left.z * right.y,
        left.w * right.y + right.w * left.y +
            left.z * right.x - left.x * right.z,
        left.w * right.z + right.w * left.z +
            left.x * right.y - left.y * right.x,
        left.w * right.w -
            left.x * right.x -
            left.y * right.y -
            left.z * right.z
    );
}

inline bool normalizedQuaternion(
    const float4 input,
    thread float4& output
) {
    if (!finite4(input)) {
        return false;
    }
    const float normSquared = dot(input, input);
    if (!(normSquared > kQuaternionMinimum) ||
        !isfinite(normSquared)) {
        return false;
    }
    output = input * rsqrt(normSquared);
    return finite4(output);
}

inline float eventSegmentDuration(
    device const MRCCDEventStateGPU& state,
    const uint mode
) {
    return max(
        mode == MR_CCD_SEGMENT_SELECTED
        ? state.time.w
        : state.time.y,
        0.0f
    );
}

inline bool integrateQuaternion(
    const float4 source,
    const float3 angularVelocity,
    const float timestep,
    thread float4& output
) {
    const float3 rotationVector =
        timestep * angularVelocity;
    const float angle = length(rotationVector);
    const float halfAngle = 0.5f * angle;
    const float scale = angle > 1.0e-6f
        ? sin(halfAngle) / angle
        : 0.5f - angle * angle / 48.0f;
    return normalizedQuaternion(
        quaternionMultiply(
            float4(
                rotationVector * scale,
                cos(halfAngle)
            ),
            source
        ),
        output
    );
}

inline Mat3 bodyInverseInertia(
    device const MRBodyPropertiesGPU& body
) {
    Mat3 result;
    result.row0 = body.inverseInertiaRow0.xyz;
    result.row1 = body.inverseInertiaRow1.xyz;
    result.row2 = body.inverseInertiaRow2.xyz;
    return result;
}

inline float3 multiply(
    const thread Mat3& matrix,
    const float3 vector
) {
    return float3(
        dot(matrix.row0, vector),
        dot(matrix.row1, vector),
        dot(matrix.row2, vector)
    );
}

inline Mat3 rotationMatrix(const float4 quaternion) {
    const float x = quaternion.x;
    const float y = quaternion.y;
    const float z = quaternion.z;
    const float w = quaternion.w;
    Mat3 result;
    result.row0 = float3(
        1.0f - 2.0f * (y * y + z * z),
        2.0f * (x * y - z * w),
        2.0f * (x * z + y * w)
    );
    result.row1 = float3(
        2.0f * (x * y + z * w),
        1.0f - 2.0f * (x * x + z * z),
        2.0f * (y * z - x * w)
    );
    result.row2 = float3(
        2.0f * (x * z - y * w),
        2.0f * (y * z + x * w),
        1.0f - 2.0f * (x * x + y * y)
    );
    return result;
}

inline Mat3 transpose(const thread Mat3& matrix) {
    Mat3 result;
    result.row0 = float3(
        matrix.row0.x,
        matrix.row1.x,
        matrix.row2.x
    );
    result.row1 = float3(
        matrix.row0.y,
        matrix.row1.y,
        matrix.row2.y
    );
    result.row2 = float3(
        matrix.row0.z,
        matrix.row1.z,
        matrix.row2.z
    );
    return result;
}

inline Mat3 multiply(
    const thread Mat3& left,
    const thread Mat3& right
) {
    const Mat3 rightTranspose = transpose(right);
    Mat3 result;
    result.row0 = float3(
        dot(left.row0, rightTranspose.row0),
        dot(left.row0, rightTranspose.row1),
        dot(left.row0, rightTranspose.row2)
    );
    result.row1 = float3(
        dot(left.row1, rightTranspose.row0),
        dot(left.row1, rightTranspose.row1),
        dot(left.row1, rightTranspose.row2)
    );
    result.row2 = float3(
        dot(left.row2, rightTranspose.row0),
        dot(left.row2, rightTranspose.row1),
        dot(left.row2, rightTranspose.row2)
    );
    return result;
}

inline bool writeWorldInverseInertia(
    thread MRBodyStateGPU& state,
    device const MRBodyPropertiesGPU& body,
    const float4 orientation
) {
    const Mat3 rotation = rotationMatrix(orientation);
    const Mat3 rotated = multiply(
        multiply(rotation, bodyInverseInertia(body)),
        transpose(rotation)
    );
    if (!finite3(rotated.row0) ||
        !finite3(rotated.row1) ||
        !finite3(rotated.row2)) {
        return false;
    }
    state.inverseInertiaWorldRow0 =
        float4(rotated.row0, 0.0f);
    state.inverseInertiaWorldRow1 =
        float4(rotated.row1, 0.0f);
    state.inverseInertiaWorldRow2 =
        float4(rotated.row2, 0.0f);
    return true;
}

inline Mat3 stateInverseInertia(
    device const MRBodyStateGPU& state
) {
    Mat3 result;
    result.row0 = state.inverseInertiaWorldRow0.xyz;
    result.row1 = state.inverseInertiaWorldRow1.xyz;
    result.row2 = state.inverseInertiaWorldRow2.xyz;
    return result;
}

inline bool inverseMatrix(
    const thread Mat3& matrix,
    thread Mat3& inverse
) {
    const float determinant = dot(
        matrix.row0,
        cross(matrix.row1, matrix.row2)
    );
    if (!(abs(determinant) > kMatrixFloor) ||
        !isfinite(determinant)) {
        return false;
    }
    const float reciprocal = 1.0f / determinant;
    const float3 column0 =
        reciprocal * cross(matrix.row1, matrix.row2);
    const float3 column1 =
        reciprocal * cross(matrix.row2, matrix.row0);
    const float3 column2 =
        reciprocal * cross(matrix.row0, matrix.row1);
    inverse.row0 = float3(
        column0.x,
        column1.x,
        column2.x
    );
    inverse.row1 = float3(
        column0.y,
        column1.y,
        column2.y
    );
    inverse.row2 = float3(
        column0.z,
        column1.z,
        column2.z
    );
    return
        finite3(inverse.row0) &&
        finite3(inverse.row1) &&
        finite3(inverse.row2);
}

inline bool validSceneState(
    device const MRBodyStateGPU& state,
    device const MRBodyPropertiesGPU& body,
    const uint globalBody
) {
    float4 normalized;
    return
        finite4(state.position) &&
        normalizedQuaternion(state.orientation, normalized) &&
        finite4(state.linearVelocityAndInverseMass) &&
        finite4(state.angularVelocity) &&
        state.flagsAndIndices[0] == body.motionType &&
        state.flagsAndIndices[1] == MR_INVALID_INDEX &&
        (
            state.flagsAndIndices[2] == globalBody ||
            state.flagsAndIndices[2] == MR_INVALID_INDEX
        );
}

inline uint mapOperatorStatus(const uint code) {
    switch (code) {
    case MR_ARTICULATED_OPERATOR_SUCCESS:
        return MR_STEP_SUCCESS;
    case MR_ARTICULATED_OPERATOR_NONFINITE_INPUT:
        return MR_STEP_NONFINITE_INPUT;
    case MR_ARTICULATED_OPERATOR_FACTORIZATION_FAILED:
        return MR_STEP_FACTORIZATION_FAILED;
    case MR_ARTICULATED_OPERATOR_NONFINITE_RESULT:
    case MR_ARTICULATED_OPERATOR_ACCURACY_FAILED:
        return MR_STEP_NONFINITE_RESULT;
    default:
        return MR_STEP_UNSUPPORTED;
    }
}

inline bool sceneEndpoint(
    device const MRBodyStateGPU& body,
    const uint articulationIndex
) {
    (void)articulationIndex;
    return body.flagsAndIndices[1] == MR_INVALID_INDEX;
}

inline bool dynamicSceneEndpoint(
    device const MRBodyStateGPU& body,
    const uint articulationIndex
) {
    return sceneEndpoint(body, articulationIndex) &&
        body.flagsAndIndices[0] == MR_MOTION_DYNAMIC;
}

inline float3 pointVelocity(
    device const MRBodyStateGPU& body,
    const float3 point
) {
    return
        body.linearVelocityAndInverseMass.xyz +
        cross(
            body.angularVelocity.xyz,
            point - body.position.xyz
        );
}

inline float3 articulatedPointVelocity(
    device const float* pointJacobians,
    const uint pointJacobianBase,
    const uint query,
    const uint nv,
    device const float* velocity
) {
    float3 result = float3(0.0f);
    for (uint dof = 0u; dof < nv; ++dof) {
        result.x +=
            pointJacobians[
                pointJacobianBase +
                (query * 3u + 0u) * nv + dof
            ] * velocity[dof];
        result.y +=
            pointJacobians[
                pointJacobianBase +
                (query * 3u + 1u) * nv + dof
            ] * velocity[dof];
        result.z +=
            pointJacobians[
                pointJacobianBase +
                (query * 3u + 2u) * nv + dof
            ] * velocity[dof];
    }
    return result;
}

inline float3 combinedJacobianColumn(
    device const float* pointJacobians,
    const uint pointJacobianBase,
    const uint localConstraint,
    const uint dof,
    const uint nv,
    const bool articulatedA,
    const bool articulatedB
) {
    float3 result = float3(0.0f);
    const uint queryA = 2u * localConstraint;
    const uint queryB = queryA + 1u;
    if (articulatedA) {
        result -= float3(
            pointJacobians[
                pointJacobianBase +
                (queryA * 3u + 0u) * nv + dof
            ],
            pointJacobians[
                pointJacobianBase +
                (queryA * 3u + 1u) * nv + dof
            ],
            pointJacobians[
                pointJacobianBase +
                (queryA * 3u + 2u) * nv + dof
            ]
        );
    }
    if (articulatedB) {
        result += float3(
            pointJacobians[
                pointJacobianBase +
                (queryB * 3u + 0u) * nv + dof
            ],
            pointJacobians[
                pointJacobianBase +
                (queryB * 3u + 1u) * nv + dof
            ],
            pointJacobians[
                pointJacobianBase +
                (queryB * 3u + 2u) * nv + dof
            ]
        );
    }
    return result;
}

inline bool solveCholesky(
    device const float* factor,
    const uint factorBase,
    const uint nv,
    thread const float* rightHandSide,
    thread float* intermediate,
    thread float* solution
) {
    for (uint row = 0u; row < nv; ++row) {
        float value = rightHandSide[row];
        for (uint column = 0u; column < row; ++column) {
            value -=
                factor[factorBase + row * nv + column] *
                intermediate[column];
        }
        const float diagonal =
            factor[factorBase + row * nv + row];
        if (!(diagonal > 0.0f) || !isfinite(diagonal)) {
            return false;
        }
        intermediate[row] = value / diagonal;
    }
    for (uint reverse = 0u; reverse < nv; ++reverse) {
        const uint row = nv - 1u - reverse;
        float value = intermediate[row];
        for (uint column = row + 1u;
             column < nv;
             ++column) {
            value -=
                factor[factorBase + column * nv + row] *
                solution[column];
        }
        solution[row] =
            value / factor[factorBase + row * nv + row];
        if (!isfinite(solution[row])) {
            return false;
        }
    }
    return true;
}

inline float3 scenePointResponse(
    device const MRBodyStateGPU& body,
    const float3 point,
    const float3 impulse
) {
    if (body.flagsAndIndices[0] != MR_MOTION_DYNAMIC) {
        return float3(0.0f);
    }
    const float3 lever = point - body.position.xyz;
    const float3 angularDelta = multiply(
        stateInverseInertia(body),
        cross(lever, impulse)
    );
    return
        body.linearVelocityAndInverseMass.w * impulse +
        cross(angularDelta, lever);
}

inline void applySceneImpulse(
    device MRBodyStateGPU& body,
    const float3 point,
    const float3 impulse
) {
    if (body.flagsAndIndices[0] != MR_MOTION_DYNAMIC) {
        return;
    }
    const float3 lever = point - body.position.xyz;
    body.linearVelocityAndInverseMass.xyz +=
        body.linearVelocityAndInverseMass.w * impulse;
    body.angularVelocity.xyz += multiply(
        stateInverseInertia(body),
        cross(lever, impulse)
    );
}

inline float ellipseProjectionFunction(
    const float2 tangent,
    const float2 radiiSquared,
    const float multiplier
) {
    float value = -1.0f;
    if (radiiSquared.x > kMatrixFloor) {
        const float denominator = radiiSquared.x + multiplier;
        value +=
            tangent.x * tangent.x * radiiSquared.x /
            (denominator * denominator);
    }
    if (radiiSquared.y > kMatrixFloor) {
        const float denominator = radiiSquared.y + multiplier;
        value +=
            tangent.y * tangent.y * radiiSquared.y /
            (denominator * denominator);
    }
    return value;
}

inline float2 projectTangentEllipse(
    const float2 tangent,
    const float2 radii
) {
    const float2 radiiSquared = radii * radii;
    float normalizedSquared = 0.0f;
    normalizedSquared +=
        radiiSquared.x > kMatrixFloor
        ? tangent.x * tangent.x / radiiSquared.x
        : (tangent.x == 0.0f ? 0.0f : INFINITY);
    normalizedSquared +=
        radiiSquared.y > kMatrixFloor
        ? tangent.y * tangent.y / radiiSquared.y
        : (tangent.y == 0.0f ? 0.0f : INFINITY);
    if (normalizedSquared <= 1.0f) {
        return tangent;
    }
    const bool activeX = radiiSquared.x > kMatrixFloor;
    const bool activeY = radiiSquared.y > kMatrixFloor;
    if (activeX != activeY) {
        return float2(
            activeX ? clamp(tangent.x, -radii.x, radii.x) : 0.0f,
            activeY ? clamp(tangent.y, -radii.y, radii.y) : 0.0f
        );
    }

    float lower = 0.0f;
    float upper = max(
        1.0f,
        max(
            abs(tangent.x) * max(radii.x, 1.0f),
            abs(tangent.y) * max(radii.y, 1.0f)
        )
    );
    for (uint expansion = 0u;
         expansion < 16u &&
             ellipseProjectionFunction(
                 tangent,
                 radiiSquared,
                 upper
             ) > 0.0f;
         ++expansion) {
        upper *= 2.0f;
    }
    for (uint iteration = 0u;
         iteration < kConeProjectionIterations;
         ++iteration) {
        const float middle = 0.5f * (lower + upper);
        if (ellipseProjectionFunction(
                tangent,
                radiiSquared,
                middle
            ) > 0.0f) {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    const float multiplier = 0.5f * (lower + upper);
    float2 projected = float2(
        radiiSquared.x > kMatrixFloor
        ? tangent.x * radiiSquared.x /
            (radiiSquared.x + multiplier)
        : 0.0f,
        radiiSquared.y > kMatrixFloor
        ? tangent.y * radiiSquared.y /
            (radiiSquared.y + multiplier)
        : 0.0f
    );
    const float projectedNormalizedSquared =
        (radiiSquared.x > kMatrixFloor
             ? projected.x * projected.x / radiiSquared.x
             : 0.0f) +
        (radiiSquared.y > kMatrixFloor
             ? projected.y * projected.y / radiiSquared.y
             : 0.0f);
    if (projectedNormalizedSquared > 1.0f) {
        projected *= rsqrt(projectedNormalizedSquared);
    }
    return projected;
}

inline float anisotropicConeBoundaryFunction(
    const float3 impulse,
    const float2 frictionSquared,
    const float multiplier
) {
    const float normal = impulse.x + multiplier;
    float value = -1.0f;
    if (frictionSquared.x > kMatrixFloor) {
        const float denominator =
            frictionSquared.x * normal + multiplier;
        if (!(denominator > kMatrixFloor)) {
            return INFINITY;
        }
        value +=
            impulse.y * impulse.y * frictionSquared.x /
            (denominator * denominator);
    }
    if (frictionSquared.y > kMatrixFloor) {
        const float denominator =
            frictionSquared.y * normal + multiplier;
        if (!(denominator > kMatrixFloor)) {
            return INFINITY;
        }
        value +=
            impulse.z * impulse.z * frictionSquared.y /
            (denominator * denominator);
    }
    return value;
}

// Euclidean closest-point projection onto the authored elliptic Coulomb cone.
// Isotropic friction has a closed form. Anisotropic friction reduces to one
// monotone scalar KKT equation with a fixed bracket/iteration order, retaining
// deterministic FP32 execution. A positive normal cap projects onto the exact
// capped ellipse when the unbounded solution exceeds it.
__attribute__((always_inline)) inline float3
projectFrictionConeValuesUnscaled(
    const float3 impulse,
    const float frictionU,
    const float frictionV,
    const float maximumNormalImpulse
) {
    const float2 friction = max(
        float2(frictionU, frictionV),
        float2(0.0f)
    );
    const float tangentNorm = length(impulse.yz);
    const bool frictionless =
        friction.x <= kConeEpsilon &&
        friction.y <= kConeEpsilon;
    if (frictionless) {
        return float3(
            maximumNormalImpulse > 0.0f
            ? clamp(impulse.x, 0.0f, maximumNormalImpulse)
            : max(impulse.x, 0.0f),
            0.0f,
            0.0f
        );
    }

    const bool activeU = friction.x > kConeEpsilon;
    const bool activeV = friction.y > kConeEpsilon;
    if (activeU != activeV) {
        const float mu = activeU ? friction.x : friction.y;
        const float tangent = activeU ? impulse.y : impulse.z;
        float3 projected = float3(
            impulse.x,
            activeU ? tangent : 0.0f,
            activeV ? tangent : 0.0f
        );
        if (!(impulse.x >= 0.0f &&
              abs(tangent) <= mu * impulse.x)) {
            if (impulse.x + mu * abs(tangent) <= 0.0f) {
                projected = float3(0.0f);
            } else {
                const float normal =
                    (impulse.x + mu * abs(tangent)) /
                    (1.0f + mu * mu);
                const float projectedTangent =
                    copysign(mu * normal, tangent);
                projected = float3(
                    normal,
                    activeU ? projectedTangent : 0.0f,
                    activeV ? projectedTangent : 0.0f
                );
            }
        }
        if (maximumNormalImpulse > 0.0f &&
            projected.x > maximumNormalImpulse) {
            projected.x = maximumNormalImpulse;
            const float limitedTangent = clamp(
                tangent,
                -mu * maximumNormalImpulse,
                mu * maximumNormalImpulse
            );
            projected.y = activeU ? limitedTangent : 0.0f;
            projected.z = activeV ? limitedTangent : 0.0f;
        }
        return projected;
    }

    float normalizedTangentSquared = 0.0f;
    normalizedTangentSquared +=
        friction.x > kConeEpsilon
        ? impulse.y * impulse.y / (friction.x * friction.x)
        : (impulse.y == 0.0f ? 0.0f : INFINITY);
    normalizedTangentSquared +=
        friction.y > kConeEpsilon
        ? impulse.z * impulse.z / (friction.y * friction.y)
        : (impulse.z == 0.0f ? 0.0f : INFINITY);
    const bool insideUnbounded =
        impulse.x >= 0.0f &&
        normalizedTangentSquared <= impulse.x * impulse.x;
    float3 projected = impulse;
    if (!insideUnbounded) {
        const float weightedDual = length(
            friction * impulse.yz
        );
        if (impulse.x + weightedDual <= 0.0f) {
            projected = float3(0.0f);
        } else {
            const float frictionScale = max(
                max(friction.x, friction.y),
                1.0f
            );
            const bool isotropic =
                activeU && activeV &&
                abs(friction.x - friction.y) <=
                    8.0f * kFloatEpsilon * frictionScale;
            if (isotropic) {
                const float mu =
                    0.5f * (friction.x + friction.y);
                const float normal =
                    (impulse.x + mu * tangentNorm) /
                    (1.0f + mu * mu);
                const float tangentScale =
                    tangentNorm > kConeEpsilon
                    ? mu * normal / tangentNorm
                    : 0.0f;
                projected = float3(
                    normal,
                    tangentScale * impulse.y,
                    tangentScale * impulse.z
                );
            } else {
                const float2 frictionSquared =
                    friction * friction;
                float lower = max(0.0f, -impulse.x);
                float upper = max(
                    lower + max(weightedDual, 1.0f),
                    1.0f
                );
                for (uint expansion = 0u;
                     expansion < 16u &&
                         anisotropicConeBoundaryFunction(
                             impulse,
                             frictionSquared,
                             upper
                         ) > 0.0f;
                     ++expansion) {
                    upper =
                        2.0f * upper + max(weightedDual, 1.0f);
                }
                for (uint iteration = 0u;
                     iteration < kConeProjectionIterations;
                     ++iteration) {
                    const float middle = 0.5f * (lower + upper);
                    if (anisotropicConeBoundaryFunction(
                            impulse,
                            frictionSquared,
                            middle
                        ) > 0.0f) {
                        lower = middle;
                    } else {
                        upper = middle;
                    }
                }
                const float multiplier = 0.5f * (lower + upper);
                const float normal = max(
                    impulse.x + multiplier,
                    0.0f
                );
                projected.x = normal;
                projected.y =
                    frictionSquared.x > kMatrixFloor
                    ? impulse.y * frictionSquared.x * normal /
                        (frictionSquared.x * normal + multiplier)
                    : 0.0f;
                projected.z =
                    frictionSquared.y > kMatrixFloor
                    ? impulse.z * frictionSquared.y * normal /
                        (frictionSquared.y * normal + multiplier)
                    : 0.0f;
                float normalizedSquared = 0.0f;
                normalizedSquared +=
                    friction.x > kConeEpsilon
                    ? projected.y * projected.y /
                        (friction.x * friction.x)
                    : 0.0f;
                normalizedSquared +=
                    friction.y > kConeEpsilon
                    ? projected.z * projected.z /
                        (friction.y * friction.y)
                    : 0.0f;
                if (normalizedSquared > normal * normal &&
                    normalizedSquared > kMatrixFloor) {
                    projected.yz *=
                        normal * rsqrt(normalizedSquared);
                }
            }
        }
    }

    if (maximumNormalImpulse > 0.0f &&
        projected.x > maximumNormalImpulse) {
        projected.x = maximumNormalImpulse;
        projected.yz = projectTangentEllipse(
            impulse.yz,
            friction * maximumNormalImpulse
        );
    }
    return projected;
}

// Cone projection is positively homogeneous in the impulse and authored
// normal cap. Normalize only the extreme slow path so squared norms and KKT
// brackets remain finite without taxing ordinary contact iterations.
__attribute__((always_inline)) inline float3 projectFrictionConeValues(
    const float3 impulse,
    const float frictionU,
    const float frictionV,
    const float maximumNormalImpulse
) {
    const float3 magnitude = abs(impulse);
    if (all(magnitude <= float3(kConeProjectionSafeMagnitude)) &&
        maximumNormalImpulse <= kConeProjectionSafeMagnitude) {
        return projectFrictionConeValuesUnscaled(
            impulse,
            frictionU,
            frictionV,
            maximumNormalImpulse
        );
    }
    const float inputScale = max(
        max(magnitude.x, max(magnitude.y, magnitude.z)),
        max(maximumNormalImpulse, 0.0f)
    );
    if (!isfinite(inputScale)) {
        return float3(INFINITY);
    }
    const float3 scaledImpulse = impulse / inputScale;
    const float scaledMaximumNormalImpulse =
        maximumNormalImpulse > 0.0f
        ? maximumNormalImpulse / inputScale
        : 0.0f;
    return inputScale * projectFrictionConeValuesUnscaled(
        scaledImpulse,
        frictionU,
        frictionV,
        scaledMaximumNormalImpulse
    );
}

inline float3 projectFrictionCone(
    const float3 impulse,
    device const MREvaluatedConstraintIRConeGPU& cone
) {
    return projectFrictionConeValues(
        impulse,
        cone.effectiveFrictionU,
        cone.effectiveFrictionV,
        cone.maximumNormalImpulse
    );
}

inline float3 projectFrictionCone(
    const float3 impulse,
    thread const MREvaluatedConstraintIRConeGPU& cone
) {
    return projectFrictionConeValues(
        impulse,
        cone.effectiveFrictionU,
        cone.effectiveFrictionV,
        cone.maximumNormalImpulse
    );
}

inline float scaledLength2(const float2 value) {
    const float scale = max(abs(value.x), abs(value.y));
    if (scale == 0.0f) {
        return 0.0f;
    }
    if (!isfinite(scale)) {
        return INFINITY;
    }
    return scale * length(value / scale);
}

// Feasibility certificate for full, degenerate, frictionless, and capped
// elliptic cones. A zero coefficient constrains only its authored tangent
// axis; it does not erase friction from the orthogonal tangent axis.
inline float frictionConeViolationValues(
    const float3 impulse,
    const float frictionU,
    const float frictionV,
    const float maximumNormalImpulse
) {
    const float normal = max(impulse.x, 0.0f);
    float violation = max(-impulse.x, 0.0f);
    if (maximumNormalImpulse > 0.0f) {
        violation = max(
            violation,
            max(impulse.x - maximumNormalImpulse, 0.0f)
        );
    }
    float2 normalizedTangent = float2(0.0f);
    bool hasActiveTangent = false;
    if (frictionU > kConeEpsilon) {
        hasActiveTangent = true;
        normalizedTangent.x = impulse.y / frictionU;
    } else {
        violation = max(violation, abs(impulse.y));
    }
    if (frictionV > kConeEpsilon) {
        hasActiveTangent = true;
        normalizedTangent.y = impulse.z / frictionV;
    } else {
        violation = max(violation, abs(impulse.z));
    }
    if (hasActiveTangent) {
        const float ellipticRadius = scaledLength2(normalizedTangent);
        violation = max(
            violation,
            max(ellipticRadius - normal, 0.0f)
        );
    }
    return violation;
}

inline bool normalizeSymmetricPSD3x3(
    thread const float matrix[3][3],
    thread float normalized[3][3],
    thread float& inverseScale
) {
    float scale = 0.0f;
    for (uint row = 0u; row < 3u; ++row) {
        for (uint column = 0u; column < 3u; ++column) {
            if (!isfinite(matrix[row][column])) {
                return false;
            }
            scale = max(scale, abs(matrix[row][column]));
        }
    }
    if (!(scale > kMatrixFloor)) {
        return false;
    }
    inverseScale = 1.0f / scale;
    if (!(inverseScale > 0.0f) || !isfinite(inverseScale)) {
        return false;
    }

    for (uint row = 0u; row < 3u; ++row) {
        for (uint column = 0u; column < 3u; ++column) {
            normalized[row][column] =
                0.5f * matrix[row][column] * inverseScale +
                0.5f * matrix[column][row] * inverseScale;
        }
    }

    const float minor01 =
        normalized[0][0] * normalized[1][1] -
        normalized[0][1] * normalized[1][0];
    const float minor02 =
        normalized[0][0] * normalized[2][2] -
        normalized[0][2] * normalized[2][0];
    const float minor12 =
        normalized[1][1] * normalized[2][2] -
        normalized[1][2] * normalized[2][1];
    const float determinant =
        normalized[0][0] * (
            normalized[1][1] * normalized[2][2] -
            normalized[1][2] * normalized[2][1]
        ) -
        normalized[0][1] * (
            normalized[1][0] * normalized[2][2] -
            normalized[1][2] * normalized[2][0]
        ) +
        normalized[0][2] * (
            normalized[1][0] * normalized[2][1] -
            normalized[1][1] * normalized[2][0]
        );
    const float tolerance = 64.0f * kFloatEpsilon;
    return
        normalized[0][0] >= -tolerance &&
        normalized[1][1] >= -tolerance &&
        normalized[2][2] >= -tolerance &&
        minor01 >= -tolerance &&
        minor02 >= -tolerance &&
        minor12 >= -tolerance &&
        determinant >= -tolerance &&
        isfinite(minor01) &&
        isfinite(minor02) &&
        isfinite(minor12) &&
        isfinite(determinant);
}

inline bool positiveSemidefinite3x3(
    thread const float matrix[3][3]
) {
    float normalized[3][3];
    float inverseScale = 0.0f;
    return normalizeSymmetricPSD3x3(
        matrix,
        normalized,
        inverseScale
    );
}

// Every scalar 2x2 principal minor of a PSD Delassus operator obeys the
// Cauchy-Schwarz bound |A_ij| <= sqrt(A_ii) sqrt(A_jj). Form the bound from
// separately rooted diagonals so the diagonal product cannot overflow first.
inline bool temporalConePairCurvatureValid3(
    const float targetDiagonalRoot,
    const float3 coupling,
    const float3 sourceDiagonalRoot
) {
    const float3 targetRoot = float3(targetDiagonalRoot);
    const float3 exactBound = targetRoot * sourceDiagonalRoot;
    const float3 toleranceScale = max(abs(coupling), exactBound);
    const float3 admittedBound =
        exactBound +
        (256.0f * kFloatEpsilon) * toleranceScale;
    return all(abs(coupling) <= admittedBound);
}

inline bool invert3x3(
    thread const float matrix[3][3],
    thread float inverse[3][3]
) {
    // Relative contact motion can be rank deficient for articulated
    // self-contact even though the articulation mass matrix is valid. First
    // certify that the unshifted symmetric response has no negative
    // curvature, then apply a small deterministic CFM floor for inversion.
    float regularized[3][3];
    float inverseScale = 0.0f;
    if (!normalizeSymmetricPSD3x3(
            matrix,
            regularized,
            inverseScale
        )) {
        return false;
    }
    for (uint axis = 0u; axis < 3u; ++axis) {
        regularized[axis][axis] += kContactMatrixRegularization;
    }
    const float c00 =
        regularized[1][1] * regularized[2][2] -
        regularized[1][2] * regularized[2][1];
    const float c01 =
        regularized[1][2] * regularized[2][0] -
        regularized[1][0] * regularized[2][2];
    const float c02 =
        regularized[1][0] * regularized[2][1] -
        regularized[1][1] * regularized[2][0];
    const float determinant =
        regularized[0][0] * c00 +
        regularized[0][1] * c01 +
        regularized[0][2] * c02;
    const float leadingMinor1 = regularized[0][0];
    const float leadingMinor2 =
        regularized[0][0] * regularized[1][1] -
        regularized[0][1] * regularized[1][0];
    // The unshifted PSD gate above owns physical curvature. Sylvester's
    // criterion here certifies that the shifted inverse is safely SPD.
    if (!(leadingMinor1 > kMatrixFloor) ||
        !(leadingMinor2 > kMatrixFloor) ||
        !(determinant > kMatrixFloor) ||
        !isfinite(leadingMinor1) ||
        !isfinite(leadingMinor2) ||
        !isfinite(determinant)) {
        return false;
    }
    const float reciprocal = inverseScale / determinant;
    if (!(reciprocal > 0.0f) || !isfinite(reciprocal)) {
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
    for (uint row = 0u; row < 3u; ++row) {
        for (uint column = 0u; column < 3u; ++column) {
            if (!isfinite(inverse[row][column])) {
                return false;
            }
        }
    }
    return true;
}

inline float3 relativePointVelocity(
    const uint localConstraint,
    device const MRContactConstraintGPU& contact,
    device const MRBodyStateGPU* bodies,
    const uint articulationIndex,
    device const float* pointJacobians,
    const uint pointJacobianBase,
    const uint nv,
    device const float* articulationVelocity
) {
    device const MRBodyStateGPU& bodyA = bodies[contact.bodyA];
    device const MRBodyStateGPU& bodyB = bodies[contact.bodyB];
    const bool articulatedA =
        bodyA.flagsAndIndices[1] != MR_INVALID_INDEX;
    const bool articulatedB =
        bodyB.flagsAndIndices[1] != MR_INVALID_INDEX;
    float3 relative = float3(0.0f);
    if (articulatedA) {
        relative -= articulatedPointVelocity(
            pointJacobians,
            pointJacobianBase,
            2u * localConstraint,
            nv,
            articulationVelocity
        );
    } else {
        relative -= pointVelocity(
            bodyA,
            contact.pointAndSeparation.xyz
        );
    }
    if (articulatedB) {
        relative += articulatedPointVelocity(
            pointJacobians,
            pointJacobianBase,
            2u * localConstraint + 1u,
            nv,
            articulationVelocity
        );
    } else {
        relative += pointVelocity(
            bodyB,
            contact.pointAndSeparation.xyz
        );
    }
    return relative;
}

inline void applyContactDelta(
    const uint localConstraint,
    device const MRContactConstraintGPU& contact,
    const float3 delta,
    device const MREvaluatedConstraintIRRowGPU* rows,
    device float* articulationVelocity,
    device MRBodyStateGPU* bodies,
    device const float* responseColumns,
    const uint responseBase,
    const uint nv,
    const uint articulationIndex
) {
    const float3 impulse =
        rows[0].direction.xyz * delta.x +
        rows[1].direction.xyz * delta.y +
        rows[2].direction.xyz * delta.z;
    for (uint dof = 0u; dof < nv; ++dof) {
        articulationVelocity[dof] +=
            responseColumns[
                responseBase +
                (localConstraint * 3u + 0u) * nv + dof
            ] * delta.x +
            responseColumns[
                responseBase +
                (localConstraint * 3u + 1u) * nv + dof
            ] * delta.y +
            responseColumns[
                responseBase +
                (localConstraint * 3u + 2u) * nv + dof
            ] * delta.z;
    }
    device MRBodyStateGPU& bodyA = bodies[contact.bodyA];
    device MRBodyStateGPU& bodyB = bodies[contact.bodyB];
    if (dynamicSceneEndpoint(bodyA, articulationIndex)) {
        applySceneImpulse(
            bodyA,
            contact.pointAndSeparation.xyz,
            -impulse
        );
    }
    if (dynamicSceneEndpoint(bodyB, articulationIndex)) {
        applySceneImpulse(
            bodyB,
            contact.pointAndSeparation.xyz,
            impulse
        );
    }
}

inline float3 rodSurfaceVelocity(
    device const MRRodNodeStateGPU& nodeA,
    device const MRRodNodeStateGPU& nodeB,
    device const MRRodEdgeStateGPU& edgeState,
    const float weight,
    const float3 radial,
    const float radius
) {
    const float3 edge =
        nodeB.position.xyz - nodeA.position.xyz;
    const float lengthSquared = dot(edge, edge);
    if (!(lengthSquared > 1.0e-20f)) {
        return float3(0.0f);
    }
    const float length = sqrt(lengthSquared);
    const float3 tangent = edge / length;
    const float3 deltaVelocity =
        nodeB.velocity.xyz - nodeA.velocity.xyz;
    const float3 tangentRate =
        (
            deltaVelocity -
            tangent * dot(tangent, deltaVelocity)
        ) / length;
    const float3 angularVelocity =
        cross(tangent, tangentRate) +
        edgeState.twistAndRate.y * tangent;
    return
        mix(
            nodeA.velocity.xyz,
            nodeB.velocity.xyz,
            weight
        ) +
        cross(angularVelocity, radius * radial);
}

inline float3 rodSurfaceResponse(
    device const MRRodNodeStateGPU& sourceA,
    device const MRRodNodeStateGPU& sourceB,
    device const MRRodEdgeStateGPU& sourceEdge,
    const float inverseMassA,
    const float inverseMassB,
    const float inverseTwistInertia,
    const float weight,
    const float3 radial,
    const float radius,
    const float3 impulse
) {
    (void)sourceEdge;
    const float3 edge =
        sourceB.position.xyz - sourceA.position.xyz;
    const float lengthSquared = dot(edge, edge);
    if (!(lengthSquared > 1.0e-20f)) {
        return float3(0.0f);
    }
    const float length = sqrt(lengthSquared);
    const float3 tangent = edge / length;
    const float3 surfaceRadius = radius * radial;
    const float3 angularImpulse =
        cross(surfaceRadius, impulse);
    const float3 bendTranspose =
        (
            cross(angularImpulse, tangent) -
            tangent *
                dot(
                    tangent,
                    cross(angularImpulse, tangent)
                )
        ) / length;
    const float3 deltaA =
        inverseMassA *
        ((1.0f - weight) * impulse - bendTranspose);
    const float3 deltaB =
        inverseMassB *
        (weight * impulse + bendTranspose);
    const float3 deltaDifference = deltaB - deltaA;
    const float3 tangentRate =
        (
            deltaDifference -
            tangent * dot(tangent, deltaDifference)
        ) / length;
    const float twistRate =
        inverseTwistInertia *
        dot(cross(tangent, surfaceRadius), impulse);
    return
        mix(deltaA, deltaB, weight) +
        cross(
            cross(tangent, tangentRate) +
                twistRate * tangent,
            surfaceRadius
        );
}

inline float3 rodToolJacobianColumn(
    device const float* pointJacobians,
    const uint pointJacobianBase,
    const uint localConstraint,
    const uint dof,
    const uint nv
) {
    const uint query = 2u * localConstraint + 1u;
    return float3(
        pointJacobians[
            pointJacobianBase +
            (query * 3u + 0u) * nv + dof
        ],
        pointJacobians[
            pointJacobianBase +
            (query * 3u + 1u) * nv + dof
        ],
        pointJacobians[
            pointJacobianBase +
            (query * 3u + 2u) * nv + dof
        ]
    );
}

inline float3 sceneCrossPointResponse(
    device const MRBodyStateGPU& body,
    const float3 targetPoint,
    const float3 sourcePoint,
    const float3 sourceImpulse
) {
    if (body.flagsAndIndices[0] != MR_MOTION_DYNAMIC) {
        return float3(0.0f);
    }
    const float3 angularDelta = multiply(
        stateInverseInertia(body),
        cross(
            sourcePoint - body.position.xyz,
            sourceImpulse
        )
    );
    return
        body.linearVelocityAndInverseMass.w * sourceImpulse +
        cross(
            angularDelta,
            targetPoint - body.position.xyz
        );
}

inline bool typedRodConstraint(
    device const MRContactConstraintGPU& contact
) {
    return
        (contact.flags & MR_CONSTRAINT_FLAG_ROD_ENDPOINT) !=
        0u;
}

inline float3 typedArticulationJacobianColumn(
    const uint localConstraint,
    device const MRContactConstraintGPU& contact,
    device const MRBodyStateGPU* bodies,
    device const float* pointJacobians,
    const uint pointJacobianBase,
    const uint dof,
    const uint nv,
    device const MRRodToolWitnessGPU* rodWitnesses,
    device const uint* constraintWitnessIndices,
    const uint constraintBase
) {
    if (typedRodConstraint(contact)) {
        const uint witnessIndex =
            constraintWitnessIndices[
                constraintBase + localConstraint
            ];
        if (witnessIndex == MR_INVALID_INDEX) {
            return float3(0.0f);
        }
        const MRRodToolWitnessGPU witness =
            rodWitnesses[witnessIndex];
        device const MRBodyStateGPU& tool =
            bodies[witness.featuresAndFlags.z];
        return
            tool.flagsAndIndices[1] != MR_INVALID_INDEX
            ? rodToolJacobianColumn(
                  pointJacobians,
                  pointJacobianBase,
                  localConstraint,
                  dof,
                  nv
              )
            : float3(0.0f);
    }
    const bool articulatedA =
        bodies[contact.bodyA].flagsAndIndices[1] !=
        MR_INVALID_INDEX;
    const bool articulatedB =
        bodies[contact.bodyB].flagsAndIndices[1] !=
        MR_INVALID_INDEX;
    return combinedJacobianColumn(
        pointJacobians,
        pointJacobianBase,
        localConstraint,
        dof,
        nv,
        articulatedA,
        articulatedB
    );
}

inline float3 typedRelativePointVelocity(
    const uint localConstraint,
    device const MRContactConstraintGPU& contact,
    device const MRBodyStateGPU* bodies,
    const uint articulationIndex,
    device const float* pointJacobians,
    const uint pointJacobianBase,
    const uint nv,
    device const float* articulationVelocity,
    device const MRRodNodeStateGPU* rodNodes,
    device const MRRodEdgeStateGPU* rodEdges,
    device const MRRodColliderGPU* rodColliders,
    device const MRRodToolWitnessGPU* rodWitnesses,
    device const uint* constraintWitnessIndices,
    const uint constraintBase
) {
    if (!typedRodConstraint(contact)) {
        return relativePointVelocity(
            localConstraint,
            contact,
            bodies,
            articulationIndex,
            pointJacobians,
            pointJacobianBase,
            nv,
            articulationVelocity
        );
    }
    const uint witnessIndex =
        constraintWitnessIndices[
            constraintBase + localConstraint
        ];
    if (witnessIndex == MR_INVALID_INDEX) {
        return float3(0.0f);
    }
    const MRRodToolWitnessGPU witness =
        rodWitnesses[witnessIndex];
    const MRRodColliderGPU collider =
        rodColliders[witness.identity.z];
    device const MRBodyStateGPU& tool =
        bodies[witness.featuresAndFlags.z];
    const float3 toolVelocity =
        tool.flagsAndIndices[1] != MR_INVALID_INDEX
        ? articulatedPointVelocity(
              pointJacobians,
              pointJacobianBase,
              2u * localConstraint + 1u,
              nv,
              articulationVelocity
          )
        : pointVelocity(
              tool,
              witness.toolPointAndSeparation.xyz
          );
    const float3 rodVelocity = rodSurfaceVelocity(
        rodNodes[collider.nodeA],
        rodNodes[collider.nodeB],
        rodEdges[collider.edgeIndex],
        witness.rodPointAndWeight.w,
        witness.radialAndTwistJacobianV.xyz,
        collider.radiusAndOffsets.x
    );
    return toolVelocity - rodVelocity;
}

inline float3 rodNodeImpulseForce(
    const MRRodColliderGPU collider,
    const MRRodToolWitnessGPU witness,
    device const MRRodNodeStateGPU* rodNodes,
    const uint nodeIndex,
    const float3 rodImpulse
) {
    if (nodeIndex != collider.nodeA &&
        nodeIndex != collider.nodeB) {
        return float3(0.0f);
    }
    const float3 edge =
        rodNodes[collider.nodeB].position.xyz -
        rodNodes[collider.nodeA].position.xyz;
    const float lengthSquared = dot(edge, edge);
    if (!(lengthSquared > 1.0e-20f)) {
        return float3(0.0f);
    }
    const float length = sqrt(lengthSquared);
    const float3 tangent = edge / length;
    const float3 surfaceRadius =
        collider.radiusAndOffsets.x *
        witness.radialAndTwistJacobianV.xyz;
    const float3 angularImpulse =
        cross(surfaceRadius, rodImpulse);
    const float3 crossed =
        cross(angularImpulse, tangent);
    const float3 bendTranspose =
        (
            crossed - tangent * dot(tangent, crossed)
        ) / length;
    const float weight = witness.rodPointAndWeight.w;
    const float3 force =
        nodeIndex == collider.nodeA
        ? (1.0f - weight) * rodImpulse - bendTranspose
        : weight * rodImpulse + bendTranspose;
    return force;
}

inline float3 rodNodeImpulseVelocityDelta(
    const MRRodColliderGPU collider,
    const MRRodToolWitnessGPU witness,
    device const MRRodNodeStateGPU* rodNodes,
    device const float* inverseRodMasses,
    const uint nodeIndex,
    const float3 rodImpulse
) {
    return inverseRodMasses[nodeIndex] *
        rodNodeImpulseForce(
            collider,
            witness,
            rodNodes,
            nodeIndex,
            rodImpulse
        );
}

inline float rodTwistImpulseTorque(
    const MRRodColliderGPU collider,
    const MRRodToolWitnessGPU witness,
    device const MRRodNodeStateGPU* rodNodes,
    const uint edgeIndex,
    const float3 rodImpulse
) {
    if (edgeIndex != collider.edgeIndex) {
        return 0.0f;
    }
    const float3 edge =
        rodNodes[collider.nodeB].position.xyz -
        rodNodes[collider.nodeA].position.xyz;
    const float lengthSquared = dot(edge, edge);
    if (!(lengthSquared > 1.0e-20f)) {
        return 0.0f;
    }
    const float3 tangent =
        edge * rsqrt(lengthSquared);
    const float3 surfaceRadius =
        collider.radiusAndOffsets.x *
        witness.radialAndTwistJacobianV.xyz;
    return dot(
        cross(tangent, surfaceRadius),
        rodImpulse
    );
}

inline float rodTwistImpulseVelocityDelta(
    const MRRodColliderGPU collider,
    const MRRodToolWitnessGPU witness,
    device const MRRodNodeStateGPU* rodNodes,
    device const float* inverseRodTwistInertias,
    const uint edgeIndex,
    const float3 rodImpulse
) {
    return inverseRodTwistInertias[edgeIndex] *
        rodTwistImpulseTorque(
            collider,
            witness,
            rodNodes,
            edgeIndex,
            rodImpulse
        );
}

// Exact frozen-geometry rod part of J_i A^-1 J_j^T. Only shared nodal and
// twist coordinates contribute, so adjacent procedural capsules couple
// without atomics or a materialized rod Delassus matrix.
inline float rodCrossContactResponse(
    const MRRodColliderGPU targetCollider,
    const MRRodToolWitnessGPU targetWitness,
    const MRRodColliderGPU sourceCollider,
    const MRRodToolWitnessGPU sourceWitness,
    device const MRRodNodeStateGPU* rodNodes,
    device const float* inverseRodMasses,
    device const float* inverseRodTwistInertias,
    const float3 targetDirection,
    const float3 sourceDirection
) {
    const float3 sourceRodImpulse = -sourceDirection;
    float3 targetDeltaA = float3(0.0f);
    float3 targetDeltaB = float3(0.0f);
    const uint sourceNodes[2] = {
        sourceCollider.nodeA,
        sourceCollider.nodeB,
    };
    for (uint sourceNode = 0u;
         sourceNode < 2u;
         ++sourceNode) {
        const uint node = sourceNodes[sourceNode];
        const float3 delta = rodNodeImpulseVelocityDelta(
            sourceCollider,
            sourceWitness,
            rodNodes,
            inverseRodMasses,
            node,
            sourceRodImpulse
        );
        if (node == targetCollider.nodeA) {
            targetDeltaA += delta;
        }
        if (node == targetCollider.nodeB) {
            targetDeltaB += delta;
        }
    }
    const float3 targetEdge =
        rodNodes[targetCollider.nodeB].position.xyz -
        rodNodes[targetCollider.nodeA].position.xyz;
    const float lengthSquared = dot(targetEdge, targetEdge);
    if (!(lengthSquared > 1.0e-20f)) {
        return 0.0f;
    }
    const float length = sqrt(lengthSquared);
    const float3 tangent = targetEdge / length;
    const float3 deltaDifference =
        targetDeltaB - targetDeltaA;
    const float3 tangentRate =
        (
            deltaDifference -
            tangent * dot(tangent, deltaDifference)
        ) / length;
    const float twistRate =
        targetCollider.edgeIndex ==
            sourceCollider.edgeIndex
        ? rodTwistImpulseVelocityDelta(
              sourceCollider,
              sourceWitness,
              rodNodes,
              inverseRodTwistInertias,
              sourceCollider.edgeIndex,
              sourceRodImpulse
          )
        : 0.0f;
    const float3 targetSurfaceRadius =
        targetCollider.radiusAndOffsets.x *
        targetWitness.radialAndTwistJacobianV.xyz;
    const float3 surfaceDelta =
        mix(
            targetDeltaA,
            targetDeltaB,
            targetWitness.rodPointAndWeight.w
        ) +
        cross(
            cross(tangent, tangentRate) +
                twistRate * tangent,
            targetSurfaceRadius
        );
    // Relative velocity is v_tool - v_rod.
    return dot(targetDirection, -surfaceDelta);
}

inline uint rodFactorElementStride(
    device const MRMetalWorldContactDispatchGPU& dispatch
) {
    return
        MR_ROD_FACTOR_TRANSLATION_FLOATS_PER_NODE *
            dispatch.rodNodeCount +
        MR_ROD_FACTOR_TWIST_FLOATS_PER_EDGE *
            dispatch.rodEdgeCount;
}

inline float rodTranslationOperatorEntry(
    device const MRMetalWorldContactDispatchGPU& dispatch,
    constant MRRodGPUDispatch& rod,
    device const MRRodNodeStateGPU* nodes,
    device const float* inverseMasses,
    device const MRRodColliderGPU* colliders,
    device const float* restLengths,
    device const float* stretchStiffness,
    device const float* bendStiffness,
    const uint row,
    const uint column
) {
    const uint rowNode = row / 3u;
    const uint columnNode = column / 3u;
    const uint rowComponent = row - 3u * rowNode;
    const uint columnComponent =
        column - 3u * columnNode;
    const uint globalRowNode = rod.rodNodeBase + rowNode;
    const uint globalColumnNode =
        rod.rodNodeBase + columnNode;
    const float timestep = dispatch.timestepAndBias.x;
    const float timestepSquared = timestep * timestep;
    float value = 0.0f;
    if (row == column) {
        const float inverseMass =
            inverseMasses[globalRowNode];
        if (!(inverseMass > 0.0f)) {
            return NAN;
        }
        value =
            (1.0f +
             timestep *
                 max(rod.dampingDerivativeTolerance.x, 0.0f)) /
            inverseMass;
    }

    for (uint localEdge = 0u;
         localEdge < rod.edgeCount;
         ++localEdge) {
        const uint globalEdge =
            rod.rodEdgeBase + localEdge;
        const MRRodColliderGPU collider =
            colliders[globalEdge];
        const int rowSign =
            globalRowNode == collider.nodeA
            ? -1
            : globalRowNode == collider.nodeB
            ? 1
            : 0;
        const int columnSign =
            globalColumnNode == collider.nodeA
            ? -1
            : globalColumnNode == collider.nodeB
            ? 1
            : 0;
        if (rowSign == 0 || columnSign == 0) {
            continue;
        }
        const float3 edge =
            nodes[collider.nodeB].position.xyz -
            nodes[collider.nodeA].position.xyz;
        const float lengthSquared = dot(edge, edge);
        if (!(lengthSquared > 1.0e-20f)) {
            return NAN;
        }
        const float3 tangent = edge * rsqrt(lengthSquared);
        const float restLength =
            max(restLengths[globalEdge], 1.0e-6f);
        const float stiffness =
            timestepSquared *
            max(stretchStiffness[globalEdge], 0.0f) /
            restLength;
        value +=
            float(rowSign * columnSign) *
            stiffness *
            tangent[rowComponent] *
            tangent[columnComponent];
    }

    for (uint localBend = 0u;
         localBend + 1u < rod.edgeCount;
         ++localBend) {
        const uint firstEdge =
            rod.rodEdgeBase + localBend;
        const MRRodColliderGPU first =
            colliders[firstEdge];
        const MRRodColliderGPU second =
            colliders[firstEdge + 1u];
        if (first.rodIndex != second.rodIndex ||
            first.nodeB != second.nodeA) {
            continue;
        }
        const uint bendNodes[3] = {
            first.nodeA,
            first.nodeB,
            second.nodeB,
        };
        const int coefficients[3] = {1, -2, 1};
        int rowCoefficient = 0;
        int columnCoefficient = 0;
        for (uint local = 0u; local < 3u; ++local) {
            if (globalRowNode == bendNodes[local]) {
                rowCoefficient = coefficients[local];
            }
            if (globalColumnNode == bendNodes[local]) {
                columnCoefficient = coefficients[local];
            }
        }
        if (rowCoefficient == 0 ||
            columnCoefficient == 0 ||
            rowComponent != columnComponent) {
            continue;
        }
        const uint bendIndex = firstEdge - first.rodIndex;
        const float meanRest = max(
            0.5f * (
                restLengths[firstEdge] +
                restLengths[firstEdge + 1u]
            ),
            1.0e-6f
        );
        const float stiffness =
            timestepSquared *
            max(bendStiffness[bendIndex], 0.0f) /
            (meanRest * meanRest * meanRest);
        value +=
            float(rowCoefficient * columnCoefficient) *
            stiffness;
    }
    return value;
}

inline float rodTwistOperatorEntry(
    device const MRMetalWorldContactDispatchGPU& dispatch,
    constant MRRodGPUDispatch& rod,
    device const float* inverseTwistInertias,
    device const MRRodColliderGPU* colliders,
    device const float* restLengths,
    device const float* twistStiffness,
    const uint row,
    const uint column
) {
    const uint globalRow = rod.rodEdgeBase + row;
    const uint globalColumn = rod.rodEdgeBase + column;
    const float timestep = dispatch.timestepAndBias.x;
    float value = 0.0f;
    if (row == column) {
        const float inverseInertia =
            inverseTwistInertias[globalRow];
        if (!(inverseInertia > 0.0f)) {
            return NAN;
        }
        value =
            (1.0f +
             timestep *
                 max(rod.dampingDerivativeTolerance.y, 0.0f)) /
            inverseInertia;
    }
    for (uint localBend = 0u;
         localBend + 1u < rod.edgeCount;
         ++localBend) {
        const uint firstEdge =
            rod.rodEdgeBase + localBend;
        if (colliders[firstEdge].rodIndex !=
            colliders[firstEdge + 1u].rodIndex) {
            continue;
        }
        const int rowSign =
            globalRow == firstEdge
            ? -1
            : globalRow == firstEdge + 1u
            ? 1
            : 0;
        const int columnSign =
            globalColumn == firstEdge
            ? -1
            : globalColumn == firstEdge + 1u
            ? 1
            : 0;
        if (rowSign == 0 || columnSign == 0) {
            continue;
        }
        const uint bendIndex =
            firstEdge - colliders[firstEdge].rodIndex;
        const float meanRest = max(
            0.5f * (
                restLengths[firstEdge] +
                restLengths[firstEdge + 1u]
            ),
            1.0e-6f
        );
        value +=
            float(rowSign * columnSign) *
            timestep * timestep *
            max(twistStiffness[bendIndex], 0.0f) /
            meanRest;
    }
    return value;
}

inline float rodTranslationFactorValue(
    device const float* factors,
    const uint factorBase,
    const uint row,
    const uint column
) {
    if (column > row ||
        row - column >=
            MR_ROD_FACTOR_TRANSLATION_BAND_WIDTH) {
        return 0.0f;
    }
    return factors[
        factorBase +
        row * MR_ROD_FACTOR_TRANSLATION_BAND_WIDTH +
        (row - column)
    ];
}

inline float rodTwistFactorValue(
    device const float* factors,
    const uint factorBase,
    const uint row,
    const uint column
) {
    if (column > row || row - column > 1u) {
        return 0.0f;
    }
    return factors[
        factorBase +
        2u * row +
        (row == column ? 0u : 1u)
    ];
}

inline bool retainedRodTranslationOperatorEntry(
    device const MRMetalWorldContactDispatchGPU& dispatch,
    const uint environment,
    device const MRRodFactorCacheGPU* caches,
    device const float* factors,
    const uint rowNode,
    const uint rowComponent,
    const uint columnNode,
    const uint columnComponent,
    thread float& output
) {
    uint rowOwner = MR_INVALID_INDEX;
    uint columnOwner = MR_INVALID_INDEX;
    MRRodFactorCacheGPU selected = {};
    for (uint rod = 0u; rod < dispatch.rodCount; ++rod) {
        const MRRodFactorCacheGPU cache =
            caches[environment * dispatch.rodCount + rod];
        const bool valid =
            cache.environment == environment &&
            cache.rodIndex == rod &&
            cache.code == MR_ROD_GPU_SUCCESS &&
            (cache.flags & MR_ROD_FACTOR_CACHE_VALID) != 0u &&
            cache.velocityCount != 0u &&
            cache.velocityOffset + cache.velocityCount <=
                dispatch.rodNodeCount &&
            cache.blockCount + cache.blockWidth <=
                dispatch.rodEdgeCount;
        if (!valid) {
            return false;
        }
        const bool containsRow =
            rowNode >= cache.velocityOffset &&
            rowNode <
                cache.velocityOffset + cache.velocityCount;
        const bool containsColumn =
            columnNode >= cache.velocityOffset &&
            columnNode <
                cache.velocityOffset + cache.velocityCount;
        if (containsRow) {
            rowOwner = rod;
        }
        if (containsColumn) {
            columnOwner = rod;
        }
        if (containsRow && containsColumn) {
            selected = cache;
        }
    }
    if (rowOwner == MR_INVALID_INDEX ||
        columnOwner == MR_INVALID_INDEX) {
        return false;
    }
    if (rowOwner != columnOwner) {
        output = 0.0f;
        return true;
    }
    const uint row =
        3u * (rowNode - selected.velocityOffset) +
        rowComponent;
    const uint column =
        3u * (columnNode - selected.velocityOffset) +
        columnComponent;
    const uint maximum = max(row, column);
    const uint first =
        maximum >
            MR_ROD_FACTOR_TRANSLATION_BAND_WIDTH - 1u
        ? maximum -
            (MR_ROD_FACTOR_TRANSLATION_BAND_WIDTH - 1u)
        : 0u;
    const uint last = min(row, column);
    output = 0.0f;
    for (uint inner = first; inner <= last; ++inner) {
        output = fma(
            rodTranslationFactorValue(
                factors,
                selected.firstBlock,
                row,
                inner
            ),
            rodTranslationFactorValue(
                factors,
                selected.firstBlock,
                column,
                inner
            ),
            output
        );
    }
    return isfinite(output);
}

inline bool retainedRodTwistOperatorEntry(
    device const MRMetalWorldContactDispatchGPU& dispatch,
    const uint environment,
    device const MRRodFactorCacheGPU* caches,
    device const float* factors,
    const uint rowEdge,
    const uint columnEdge,
    thread float& output
) {
    uint rowOwner = MR_INVALID_INDEX;
    uint columnOwner = MR_INVALID_INDEX;
    MRRodFactorCacheGPU selected = {};
    for (uint rod = 0u; rod < dispatch.rodCount; ++rod) {
        const MRRodFactorCacheGPU cache =
            caches[environment * dispatch.rodCount + rod];
        const bool valid =
            cache.environment == environment &&
            cache.rodIndex == rod &&
            cache.code == MR_ROD_GPU_SUCCESS &&
            (cache.flags & MR_ROD_FACTOR_CACHE_VALID) != 0u &&
            cache.velocityCount != 0u &&
            cache.blockWidth + 1u == cache.velocityCount &&
            cache.velocityOffset + cache.velocityCount <=
                dispatch.rodNodeCount &&
            cache.blockCount + cache.blockWidth <=
                dispatch.rodEdgeCount;
        if (!valid) {
            return false;
        }
        const bool containsRow =
            rowEdge >= cache.blockCount &&
            rowEdge < cache.blockCount + cache.blockWidth;
        const bool containsColumn =
            columnEdge >= cache.blockCount &&
            columnEdge <
                cache.blockCount + cache.blockWidth;
        if (containsRow) {
            rowOwner = rod;
        }
        if (containsColumn) {
            columnOwner = rod;
        }
        if (containsRow && containsColumn) {
            selected = cache;
        }
    }
    if (rowOwner == MR_INVALID_INDEX ||
        columnOwner == MR_INVALID_INDEX) {
        return false;
    }
    if (rowOwner != columnOwner) {
        output = 0.0f;
        return true;
    }
    const uint row = rowEdge - selected.blockCount;
    const uint column = columnEdge - selected.blockCount;
    const uint maximum = max(row, column);
    const uint first = maximum > 0u ? maximum - 1u : 0u;
    const uint last = min(row, column);
    const uint twistBase =
        selected.firstBlock +
        MR_ROD_FACTOR_TRANSLATION_FLOATS_PER_NODE *
            selected.velocityCount;
    output = 0.0f;
    for (uint inner = first; inner <= last; ++inner) {
        output = fma(
            rodTwistFactorValue(
                factors,
                twistBase,
                row,
                inner
            ),
            rodTwistFactorValue(
                factors,
                twistBase,
                column,
                inner
            ),
            output
        );
    }
    return isfinite(output);
}

inline bool solveRodTranslationFactor(
    device const float* factors,
    const uint factorBase,
    const uint rowCount,
    threadgroup float* vector,
    const uint vectorBase
) {
    for (uint row = 0u; row < rowCount; ++row) {
        float value = vector[vectorBase + row];
        const uint first = row >
                MR_ROD_FACTOR_TRANSLATION_BAND_WIDTH - 1u
            ? row -
                (MR_ROD_FACTOR_TRANSLATION_BAND_WIDTH - 1u)
            : 0u;
        for (uint column = first;
             column < row;
             ++column) {
            value = fma(
                -rodTranslationFactorValue(
                    factors,
                    factorBase,
                    row,
                    column
                ),
                vector[vectorBase + column],
                value
            );
        }
        const float diagonal = rodTranslationFactorValue(
            factors,
            factorBase,
            row,
            row
        );
        if (!(diagonal > 0.0f) || !isfinite(diagonal)) {
            return false;
        }
        vector[vectorBase + row] = value / diagonal;
    }
    for (uint reverse = 0u; reverse < rowCount; ++reverse) {
        const uint row = rowCount - 1u - reverse;
        float value = vector[vectorBase + row];
        const uint last = min(
            rowCount,
            row + MR_ROD_FACTOR_TRANSLATION_BAND_WIDTH
        );
        for (uint source = row + 1u;
             source < last;
             ++source) {
            value = fma(
                -rodTranslationFactorValue(
                    factors,
                    factorBase,
                    source,
                    row
                ),
                vector[vectorBase + source],
                value
            );
        }
        const float diagonal = rodTranslationFactorValue(
            factors,
            factorBase,
            row,
            row
        );
        vector[vectorBase + row] = value / diagonal;
        if (!isfinite(vector[vectorBase + row])) {
            return false;
        }
    }
    return true;
}

inline bool solveRodTwistFactor(
    device const float* factors,
    const uint factorBase,
    const uint rowCount,
    threadgroup float* vector,
    const uint vectorBase
) {
    for (uint row = 0u; row < rowCount; ++row) {
        float value = vector[vectorBase + row];
        if (row != 0u) {
            value = fma(
                -factors[factorBase + 2u * row + 1u],
                vector[vectorBase + row - 1u],
                value
            );
        }
        const float diagonal =
            factors[factorBase + 2u * row];
        if (!(diagonal > 0.0f) || !isfinite(diagonal)) {
            return false;
        }
        vector[vectorBase + row] = value / diagonal;
    }
    for (uint reverse = 0u; reverse < rowCount; ++reverse) {
        const uint row = rowCount - 1u - reverse;
        float value = vector[vectorBase + row];
        if (row + 1u < rowCount) {
            value = fma(
                -factors[
                    factorBase + 2u * (row + 1u) + 1u
                ],
                vector[vectorBase + row + 1u],
                value
            );
        }
        const float diagonal =
            factors[factorBase + 2u * row];
        vector[vectorBase + row] = value / diagonal;
        if (!isfinite(vector[vectorBase + row])) {
            return false;
        }
    }
    return true;
}

// Device-arena variants are used by persistent Wave32 packets. The arena is
// invocation-local, and every connected rod has one island owner, so the
// serialized band solve never races even when packet claims execute in a
// different order.
inline bool solveRodTranslationFactorDevice(
    device const float* factors,
    const uint factorBase,
    const uint rowCount,
    device float* vector,
    const uint vectorBase
) {
    for (uint row = 0u; row < rowCount; ++row) {
        float value = vector[vectorBase + row];
        const uint first = row >
                MR_ROD_FACTOR_TRANSLATION_BAND_WIDTH - 1u
            ? row -
                (MR_ROD_FACTOR_TRANSLATION_BAND_WIDTH - 1u)
            : 0u;
        for (uint column = first;
             column < row;
             ++column) {
            value = fma(
                -rodTranslationFactorValue(
                    factors,
                    factorBase,
                    row,
                    column
                ),
                vector[vectorBase + column],
                value
            );
        }
        const float diagonal = rodTranslationFactorValue(
            factors,
            factorBase,
            row,
            row
        );
        if (!(diagonal > 0.0f) || !isfinite(diagonal)) {
            return false;
        }
        vector[vectorBase + row] = value / diagonal;
    }
    for (uint reverse = 0u; reverse < rowCount; ++reverse) {
        const uint row = rowCount - 1u - reverse;
        float value = vector[vectorBase + row];
        const uint last = min(
            rowCount,
            row + MR_ROD_FACTOR_TRANSLATION_BAND_WIDTH
        );
        for (uint source = row + 1u;
             source < last;
             ++source) {
            value = fma(
                -rodTranslationFactorValue(
                    factors,
                    factorBase,
                    source,
                    row
                ),
                vector[vectorBase + source],
                value
            );
        }
        const float diagonal = rodTranslationFactorValue(
            factors,
            factorBase,
            row,
            row
        );
        vector[vectorBase + row] = value / diagonal;
        if (!isfinite(vector[vectorBase + row])) {
            return false;
        }
    }
    return true;
}

inline bool solveRodTwistFactorDevice(
    device const float* factors,
    const uint factorBase,
    const uint rowCount,
    device float* vector,
    const uint vectorBase
) {
    for (uint row = 0u; row < rowCount; ++row) {
        float value = vector[vectorBase + row];
        if (row != 0u) {
            value = fma(
                -factors[factorBase + 2u * row + 1u],
                vector[vectorBase + row - 1u],
                value
            );
        }
        const float diagonal =
            factors[factorBase + 2u * row];
        if (!(diagonal > 0.0f) || !isfinite(diagonal)) {
            return false;
        }
        vector[vectorBase + row] = value / diagonal;
    }
    for (uint reverse = 0u; reverse < rowCount; ++reverse) {
        const uint row = rowCount - 1u - reverse;
        float value = vector[vectorBase + row];
        if (row + 1u < rowCount) {
            value = fma(
                -factors[
                    factorBase + 2u * (row + 1u) + 1u
                ],
                vector[vectorBase + row + 1u],
                value
            );
        }
        const float diagonal =
            factors[factorBase + 2u * row];
        vector[vectorBase + row] = value / diagonal;
        if (!isfinite(vector[vectorBase + row])) {
            return false;
        }
    }
    return true;
}

} // namespace

// Factors the implicit DER response A = M + hD + h^2K once per rod and
// microstep. Translation uses an exact scalar band for the retained
// stretch/three-node bend stencil; twist uses a bidiagonal factor. Numerical
// factors and later operator workspace share one private lifetime-planned
// arena, while the compact cache is persistent diagnostic state.
kernel void mr_world_factor_rod_operator(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    constant MRRodGPUDispatch& rod [[buffer(1)]],
    device const MRRodNodeStateGPU* candidateRodNodes [[buffer(2)]],
    device const float* inverseMasses [[buffer(3)]],
    device const float* inverseTwistInertias [[buffer(4)]],
    device const MRRodColliderGPU* colliders [[buffer(5)]],
    device const float* restLengths [[buffer(6)]],
    device const float* stretchStiffness [[buffer(7)]],
    device const float* bendStiffness [[buffer(8)]],
    device const float* twistStiffness [[buffer(9)]],
    device MRRodFactorCacheGPU* caches [[buffer(10)]],
    device float* operatorArena [[buffer(11)]],
    constant uint& rodIndex [[buffer(12)]],
    constant MRMetalWorldPassGPU& pass [[buffer(13)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount ||
        rodIndex >= dispatch.rodCount) {
        return;
    }
    const uint cacheIndex =
        environment * dispatch.rodCount + rodIndex;
    MRRodFactorCacheGPU cache = {};
    cache.environment = environment;
    cache.rodIndex = rodIndex;
    cache.velocityOffset = rod.rodNodeBase;
    cache.velocityCount = rod.nodeCount;
    cache.blockCount = rod.rodEdgeBase;
    cache.blockWidth = rod.edgeCount;
    cache.generation =
        pass.physicsSubstep +
        pass.controlStep * max(dispatch.rodCount, 1u);
    cache.code = MR_ROD_GPU_SUCCESS;
    cache.failingElement = MR_INVALID_INDEX;
    cache.diagnostics = float4(
        INFINITY,
        0.0f,
        0.0f,
        0.0f
    );

    const uint factorStride = rodFactorElementStride(dispatch);
    const uint required =
        factorStride +
        3u * dispatch.rodNodeCount +
        dispatch.rodEdgeCount;
    if (dispatch.operatorVelocityCapacity < required ||
        rod.nodeCount == 0u ||
        rod.edgeCount + 1u != rod.nodeCount) {
        cache.code = MR_ROD_GPU_INVALID_DISPATCH;
        caches[cacheIndex] = cache;
        return;
    }
    const uint factorBase =
        environment * dispatch.operatorVelocityCapacity +
        MR_ROD_FACTOR_TRANSLATION_FLOATS_PER_NODE *
            rod.rodNodeBase +
        MR_ROD_FACTOR_TWIST_FLOATS_PER_EDGE *
            rod.rodEdgeBase;
    cache.firstBlock = factorBase;
    cache.flags =
        MR_ROD_FACTOR_CACHE_TRANSLATION_BAND |
        MR_ROD_FACTOR_CACHE_TWIST_BAND;
    const uint nodeBase =
        environment * dispatch.rodNodeCount;
    device const MRRodNodeStateGPU* nodes =
        candidateRodNodes + nodeBase;
    uint projected = 0u;
    float maximumProjection = 0.0f;
    for (uint edge = 0u; edge < rod.edgeCount; ++edge) {
        const uint globalEdge = rod.rodEdgeBase + edge;
        if (stretchStiffness[globalEdge] < 0.0f) {
            ++projected;
            maximumProjection = max(
                maximumProjection,
                -stretchStiffness[globalEdge]
            );
        }
        if (edge + 1u < rod.edgeCount) {
            const uint bendIndex =
                globalEdge - rodIndex;
            if (bendStiffness[bendIndex] < 0.0f) {
                ++projected;
                maximumProjection = max(
                    maximumProjection,
                    -bendStiffness[bendIndex]
                );
            }
            if (twistStiffness[bendIndex] < 0.0f) {
                ++projected;
                maximumProjection = max(
                    maximumProjection,
                    -twistStiffness[bendIndex]
                );
            }
        }
    }

    const uint translationRows = 3u * rod.nodeCount;
    for (uint row = 0u; row < translationRows; ++row) {
        for (uint offset = 0u;
             offset <
                 MR_ROD_FACTOR_TRANSLATION_BAND_WIDTH;
             ++offset) {
            operatorArena[
                factorBase +
                row * MR_ROD_FACTOR_TRANSLATION_BAND_WIDTH +
                offset
            ] = 0.0f;
        }
        const uint first = row >
                MR_ROD_FACTOR_TRANSLATION_BAND_WIDTH - 1u
            ? row -
                (MR_ROD_FACTOR_TRANSLATION_BAND_WIDTH - 1u)
            : 0u;
        for (uint column = first;
             column <= row;
             ++column) {
            float value = rodTranslationOperatorEntry(
                dispatch,
                rod,
                nodes,
                inverseMasses,
                colliders,
                restLengths,
                stretchStiffness,
                bendStiffness,
                row,
                column
            );
            const uint innerFirst = max(
                first,
                column >
                        MR_ROD_FACTOR_TRANSLATION_BAND_WIDTH -
                            1u
                    ? column -
                        (
                            MR_ROD_FACTOR_TRANSLATION_BAND_WIDTH -
                            1u
                        )
                    : 0u
            );
            for (uint inner = innerFirst;
                 inner < column;
                 ++inner) {
                value = fma(
                    -rodTranslationFactorValue(
                        operatorArena,
                        factorBase,
                        row,
                        inner
                    ),
                    rodTranslationFactorValue(
                        operatorArena,
                        factorBase,
                        column,
                        inner
                    ),
                    value
                );
            }
            if (!isfinite(value)) {
                cache.code = MR_ROD_GPU_NONFINITE_RESULT;
                cache.failingElement = row;
                caches[cacheIndex] = cache;
                return;
            }
            if (row == column) {
                const float scale = max(
                    abs(rodTranslationOperatorEntry(
                        dispatch,
                        rod,
                        nodes,
                        inverseMasses,
                        colliders,
                        restLengths,
                        stretchStiffness,
                        bendStiffness,
                        row,
                        row
                    )),
                    1.0f
                );
                const float floor =
                    64.0f * 1.1920928955078125e-7f *
                    scale;
                if (value < floor) {
                    ++projected;
                    maximumProjection = max(
                        maximumProjection,
                        floor - value
                    );
                    value = floor;
                }
                cache.diagnostics.x = min(
                    cache.diagnostics.x,
                    value
                );
                cache.diagnostics.y = max(
                    cache.diagnostics.y,
                    value
                );
                value = sqrt(value);
            } else {
                const float diagonal =
                    rodTranslationFactorValue(
                        operatorArena,
                        factorBase,
                        column,
                        column
                    );
                if (!(diagonal > 0.0f)) {
                    cache.code = MR_ROD_GPU_NONFINITE_RESULT;
                    cache.failingElement = row;
                    caches[cacheIndex] = cache;
                    return;
                }
                value /= diagonal;
            }
            operatorArena[
                factorBase +
                row * MR_ROD_FACTOR_TRANSLATION_BAND_WIDTH +
                (row - column)
            ] = value;
        }
    }

    const uint twistBase =
        factorBase +
        MR_ROD_FACTOR_TRANSLATION_FLOATS_PER_NODE *
            rod.nodeCount;
    for (uint row = 0u; row < rod.edgeCount; ++row) {
        float diagonal = rodTwistOperatorEntry(
            dispatch,
            rod,
            inverseTwistInertias,
            colliders,
            restLengths,
            twistStiffness,
            row,
            row
        );
        float lower = 0.0f;
        if (row != 0u) {
            const float previousDiagonal =
                operatorArena[
                    twistBase + 2u * (row - 1u)
                ];
            lower = rodTwistOperatorEntry(
                dispatch,
                rod,
                inverseTwistInertias,
                colliders,
                restLengths,
                twistStiffness,
                row,
                row - 1u
            ) / previousDiagonal;
            diagonal = fma(-lower, lower, diagonal);
        }
        if (!isfinite(diagonal) || !isfinite(lower)) {
            cache.code = MR_ROD_GPU_NONFINITE_RESULT;
            cache.failingElement = translationRows + row;
            caches[cacheIndex] = cache;
            return;
        }
        const float scale = max(
            abs(rodTwistOperatorEntry(
                dispatch,
                rod,
                inverseTwistInertias,
                colliders,
                restLengths,
                twistStiffness,
                row,
                row
            )),
            1.0f
        );
        const float floor =
            64.0f * 1.1920928955078125e-7f * scale;
        if (diagonal < floor) {
            ++projected;
            maximumProjection = max(
                maximumProjection,
                floor - diagonal
            );
            diagonal = floor;
        }
        cache.diagnostics.x = min(
            cache.diagnostics.x,
            diagonal
        );
        cache.diagnostics.y = max(
            cache.diagnostics.y,
            diagonal
        );
        operatorArena[twistBase + 2u * row] =
            sqrt(diagonal);
        operatorArena[twistBase + 2u * row + 1u] =
            lower;
    }
    cache.projectedCurvatureCount = projected;
    cache.diagnostics.w = maximumProjection;
    cache.flags |= MR_ROD_FACTOR_CACHE_VALID;
    if (projected != 0u) {
        cache.flags |=
            MR_ROD_FACTOR_CACHE_PROJECTED_CURVATURE;
    }
    caches[cacheIndex] = cache;
}

namespace {

inline uint findRoot(thread uint* parents, uint node) {
    uint current = node;
    while (parents[current] != current) {
        current = parents[current];
    }
    while (parents[node] != node) {
        const uint next = parents[node];
        parents[node] = current;
        node = next;
    }
    return current;
}

inline void unionRoots(
    thread uint* parents,
    const uint left,
    const uint right
) {
    const uint leftRoot = findRoot(parents, left);
    const uint rightRoot = findRoot(parents, right);
    if (leftRoot == rightRoot) {
        return;
    }
    const uint minimum = min(leftRoot, rightRoot);
    const uint maximum = max(leftRoot, rightRoot);
    parents[maximum] = minimum;
}

} // namespace

// Selects the exact state observed by a native policy. Task reset and launch
// scheduling run before this pass, while the transactional physics prepare
// runs after policy inference. The unused ping-pong destination is therefore
// the observation-time staging arena and requires no retained buffer.
kernel void mr_world_select_observation_state(
    device const MRMetalWorldDispatchGPU& worldDispatch [[buffer(0)]],
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(1)]],
    constant MRMetalWorldPassGPU& pass [[buffer(2)]],
    device const uint* resetMasks [[buffer(3)]],
    device const float* resetQ [[buffer(4)]],
    device const MRBodyStateGPU* resetScene [[buffer(5)]],
    device const float* sourceQ [[buffer(6)]],
    device const MRBodyStateGPU* sourceScene [[buffer(7)]],
    device float* observationQ [[buffer(8)]],
    device MRBodyStateGPU* observationScene [[buffer(9)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(10)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= worldDispatch.environmentCount ||
        environment >= dispatch.environmentCount ||
        pass.controlStep >= worldDispatch.controlStepCount ||
        pass.physicsSubstep != MR_INVALID_INDEX) {
        return;
    }
    const bool reset =
        (worldDispatch.flags & MR_METAL_WORLD_HAS_RESETS) != 0u &&
        resetMasks[
            pass.controlStep * worldDispatch.resetMaskStepStride +
            environment
        ] != 0u;
    const uint qBase = environment * worldDispatch.qStride;
    for (uint coordinate = 0u;
         coordinate < worldDispatch.nq;
         ++coordinate) {
        observationQ[qBase + coordinate] = reset
            ? resetQ[qBase + coordinate]
            : sourceQ[qBase + coordinate];
    }
    const uint sceneBase = environment * dispatch.sceneBodyStride;
    for (uint body = 0u; body < dispatch.sceneBodyCount; ++body) {
        observationScene[sceneBase + body] = reset
            ? resetScene[sceneBase + body]
            : sourceScene[sceneBase + body];
    }
    MRMetalWorldContactStatusGPU status = {};
    status.code = MR_STEP_SUCCESS;
    status.environment = environment;
    status.controlStep = pass.controlStep;
    status.physicsSubstep = MR_INVALID_INDEX;
    status.firstFailingPair = MR_INVALID_INDEX;
    status.firstFailingConstraint = MR_INVALID_INDEX;
    status.firstFailingEventKeyLow = MR_INVALID_INDEX;
    status.firstFailingEventKeyHigh = MR_INVALID_INDEX;
    statuses[environment] = status;
}

// Applies resets/kinematic targets and checkpoints the complete contact state
// at the start of one control step.
kernel void mr_world_prepare_contact_step(
    device const MRMetalWorldDispatchGPU& worldDispatch [[buffer(0)]],
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(1)]],
    constant MRMetalWorldPassGPU& pass [[buffer(2)]],
    device const uint* resetMasks [[buffer(3)]],
    device const MRBodyStateGPU* resetSceneBodies [[buffer(4)]],
    device const MRBodyStateGPU* kinematicTargets [[buffer(5)]],
    device const MRBodyPropertiesGPU* bodyProperties [[buffer(6)]],
    device const uint* sceneBodyIndices [[buffer(7)]],
    device MRBodyStateGPU* sceneState [[buffer(8)]],
    device MRBodyStateGPU* checkpointSceneState [[buffer(9)]],
    device MRManifoldHeaderGPU* manifoldHeaders [[buffer(10)]],
    device MRManifoldPointGPU* manifoldPoints [[buffer(11)]],
    device uint* manifoldCounts [[buffer(12)]],
    device MRManifoldHeaderGPU* checkpointHeaders [[buffer(13)]],
    device MRManifoldPointGPU* checkpointPoints [[buffer(14)]],
    device uint* checkpointCounts [[buffer(15)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(16)]],
    device MRConvexQueryCacheGPU* convexCaches [[buffer(17)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    MRMetalWorldContactStatusGPU status = {};
    status.code = MR_STEP_SUCCESS;
    status.environment = environment;
    status.controlStep = pass.controlStep;
    status.physicsSubstep = MR_INVALID_INDEX;
    status.firstFailingPair = MR_INVALID_INDEX;
    status.firstFailingConstraint = MR_INVALID_INDEX;
    status.firstFailingEventKeyLow = MR_INVALID_INDEX;
    status.firstFailingEventKeyHigh = MR_INVALID_INDEX;

    const bool hasResets =
        (worldDispatch.flags & MR_METAL_WORLD_HAS_RESETS) != 0u;
    const bool applyReset =
        hasResets &&
        resetMasks[
            pass.controlStep *
                worldDispatch.resetMaskStepStride +
            environment
        ] != 0u;
    const bool hasTargets =
        (dispatch.flags &
         MR_METAL_WORLD_CONTACT_HAS_KINEMATIC_TARGETS) != 0u;
    const uint sceneBase =
        environment * dispatch.sceneBodyStride;
    const uint targetBase =
        (
            pass.controlStep * dispatch.environmentCount +
            environment
        ) * dispatch.sceneBodyStride;
    for (uint localScene = 0u;
         localScene < dispatch.sceneBodyCount;
         ++localScene) {
        const uint globalBody = sceneBodyIndices[localScene];
        MRBodyStateGPU value = applyReset
            ? resetSceneBodies[sceneBase + localScene]
            : sceneState[sceneBase + localScene];
        if (hasTargets &&
            bodyProperties[globalBody].motionType ==
                MR_MOTION_KINEMATIC) {
            value = kinematicTargets[targetBase + localScene];
        }
        sceneState[sceneBase + localScene] = value;
        checkpointSceneState[sceneBase + localScene] = value;
    }

    const uint manifoldBase =
        environment * dispatch.manifoldStride;
    const uint pointBase =
        manifoldBase *
        MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
    const uint oldCount = applyReset
        ? 0u
        : min(
              manifoldCounts[environment],
              dispatch.manifoldCapacity
          );
    checkpointCounts[environment] = oldCount;
    manifoldCounts[environment] = oldCount;
    if (applyReset) {
        const uint cacheBase =
            environment * dispatch.convexCacheStride;
        for (uint pair = 0u;
             pair < dispatch.convexCacheStride;
             ++pair) {
            convexCaches[cacheBase + pair] = {};
        }
    }
    for (uint manifold = 0u;
         manifold < dispatch.manifoldCapacity;
         ++manifold) {
        if (manifold < oldCount) {
            checkpointHeaders[manifoldBase + manifold] =
                manifoldHeaders[manifoldBase + manifold];
            for (uint point = 0u;
                 point <
                     MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
                 ++point) {
                checkpointPoints[
                    pointBase +
                    manifold *
                        MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY +
                    point
                ] = manifoldPoints[
                    pointBase +
                    manifold *
                        MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY +
                    point
                ];
            }
        }
    }
    statuses[environment] = status;
}

// Projects articulation poses and scene states into one global body-state
// array consumed by collision. Inertia for scene bodies is rebuilt from the
// immutable body tensor so callers cannot forge inverse mass.
kernel void mr_world_build_body_states(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRArticulationGPU* articulations [[buffer(1)]],
    device const MRBodyPropertiesGPU* bodyProperties [[buffer(2)]],
    device const uint* sceneBodyIndices [[buffer(3)]],
    device const MRArticulatedBodyPoseGPU* bodyPoses [[buffer(4)]],
    device const MRArticulatedOperatorStatusGPU* operatorStatuses [[buffer(5)]],
    device const MRBodyStateGPU* sceneStates [[buffer(6)]],
    device MRBodyStateGPU* bodyStates [[buffer(7)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(8)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    if (status.code != MR_STEP_SUCCESS) {
        return;
    }
    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    const uint poseBase =
        environment * dispatch.bodyCount;
    for (uint owner = 0u;
         owner < dispatch.articulationCount;
         ++owner) {
        const MRArticulatedOperatorStatusGPU operatorStatus =
            operatorStatuses[
                owner * dispatch.environmentCount +
                environment
            ];
        if (operatorStatus.code !=
                MR_ARTICULATED_OPERATOR_SUCCESS) {
            status.code = mapOperatorStatus(
                operatorStatus.code
            );
            status.firstFailingConstraint =
                operatorStatus.failingIndex;
            status.firstFailingStableKeyLow =
                operatorStatus.code;
            status.firstFailingStableKeyHigh = owner;
            statuses[environment] = status;
            return;
        }
        const MRArticulationGPU articulation =
            articulations[owner];
        for (uint localBody = 0u;
             localBody < articulation.bodyCount;
             ++localBody) {
            const uint globalBody =
                articulation.firstBody + localBody;
            const MRArticulatedBodyPoseGPU pose =
                bodyPoses[poseBase + globalBody];
            MRBodyStateGPU state = {};
            state.position = pose.position;
            state.orientation = pose.orientation;
            state.linearVelocityAndInverseMass.w = 0.0f;
            state.flagsAndIndices[0] =
                bodyProperties[globalBody].motionType;
            state.flagsAndIndices[1] = owner;
            state.flagsAndIndices[2] = globalBody;
            state.flagsAndIndices[3] = 0u;
            bodyStates[bodyBase + globalBody] = state;
        }
    }

    const uint sceneBase =
        environment * dispatch.sceneBodyStride;
    for (uint localScene = 0u;
         localScene < dispatch.sceneBodyCount;
         ++localScene) {
        const uint globalBody = sceneBodyIndices[localScene];
        device const MRBodyPropertiesGPU& properties =
            bodyProperties[globalBody];
        device const MRBodyStateGPU& input =
            sceneStates[sceneBase + localScene];
        if (globalBody >= dispatch.bodyCount ||
            properties.articulationIndex != MR_INVALID_INDEX ||
            !validSceneState(input, properties, globalBody)) {
            status.code = MR_STEP_NONFINITE_INPUT;
            status.firstFailingConstraint = globalBody;
            statuses[environment] = status;
            return;
        }
        float4 orientation;
        if (!normalizedQuaternion(
                input.orientation,
                orientation
            )) {
            status.code = MR_STEP_NONFINITE_INPUT;
            status.firstFailingConstraint = globalBody;
            statuses[environment] = status;
            return;
        }
        MRBodyStateGPU state = input;
        state.position.w = 1.0f;
        state.orientation = orientation;
        const float authoredInverseMass =
            properties.massAndInverseMass.y;
        const bool hasMassOverride =
            properties.motionType == MR_MOTION_DYNAMIC &&
            input.linearVelocityAndInverseMass.w > 0.0f &&
            authoredInverseMass > 0.0f;
        state.linearVelocityAndInverseMass.w =
            properties.motionType == MR_MOTION_DYNAMIC
            ? (hasMassOverride
                ? input.linearVelocityAndInverseMass.w
                : authoredInverseMass)
            : 0.0f;
        state.angularVelocity.w = 0.0f;
        state.flagsAndIndices[0] = properties.motionType;
        state.flagsAndIndices[1] = MR_INVALID_INDEX;
        state.flagsAndIndices[2] = globalBody;
        if (!writeWorldInverseInertia(
                state,
                properties,
                orientation
            )) {
            status.code = MR_STEP_NONFINITE_RESULT;
            status.firstFailingConstraint = globalBody;
            statuses[environment] = status;
            return;
        }
        if (hasMassOverride) {
            const float inverseInertiaScale =
                input.linearVelocityAndInverseMass.w /
                authoredInverseMass;
            state.inverseInertiaWorldRow0.xyz *=
                inverseInertiaScale;
            state.inverseInertiaWorldRow1.xyz *=
                inverseInertiaScale;
            state.inverseInertiaWorldRow2.xyz *=
                inverseInertiaScale;
        }
        bodyStates[bodyBase + globalBody] = state;
    }
    statuses[environment] = status;
}

// Predicts unconstrained scene-body velocities while preserving current
// collision poses. Articulation entries are copied unchanged; ABA owns qdot.
kernel void mr_world_predict_scene(
    device const MRWorldGPU& world [[buffer(0)]],
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(1)]],
    device const MRBodyPropertiesGPU* bodyProperties [[buffer(2)]],
    device const uint* sceneBodyIndices [[buffer(3)]],
    device const MRBodyStateGPU* currentBodies [[buffer(4)]],
    device MRBodyStateGPU* candidateBodies [[buffer(5)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(6)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    if (statuses[environment].code != MR_STEP_SUCCESS) {
        return;
    }
    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    for (uint body = 0u; body < dispatch.bodyCount; ++body) {
        candidateBodies[bodyBase + body] =
            currentBodies[bodyBase + body];
    }
    const float timestep = dispatch.timestepAndBias.x;
    for (uint localScene = 0u;
         localScene < dispatch.sceneBodyCount;
        ++localScene) {
        const uint globalBody = sceneBodyIndices[localScene];
        MRBodyStateGPU state =
            candidateBodies[bodyBase + globalBody];
        device const MRBodyPropertiesGPU& properties =
            bodyProperties[globalBody];
        if (properties.motionType != MR_MOTION_DYNAMIC) {
            continue;
        }
        // Exponential damping is invariant under event-time splitting:
        // exp(-d * h0) * exp(-d * h1) == exp(-d * (h0 + h1)).
        const float linearScale = exp(
            -timestep *
                properties.dampingAndSpeedLimits.x
        );
        const float angularScale = exp(
            -timestep *
                properties.dampingAndSpeedLimits.y
        );
        state.linearVelocityAndInverseMass.xyz =
            linearScale *
                state.linearVelocityAndInverseMass.xyz +
            timestep * world.gravityAndTimestep.xyz;
        state.angularVelocity.xyz *= angularScale;
        if (!finite4(state.linearVelocityAndInverseMass) ||
            !finite4(state.angularVelocity)) {
            MRMetalWorldContactStatusGPU status =
                statuses[environment];
            status.code = MR_STEP_NONFINITE_RESULT;
            status.firstFailingConstraint = globalBody;
            statuses[environment] = status;
            return;
        }
        candidateBodies[bodyBase + globalBody] = state;
    }
}

// Re-materializes ABA's acceleration at the duration owned by the current CCD
// cursor. ABA factorization/acceleration is timestep independent in the
// present explicit-drive path, so this avoids rerunning dynamics merely to
// replace the full-microstep integration produced by the generic ABA kernel.
kernel void mr_world_materialize_event_articulation(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRArticulationGPU* articulations [[buffer(1)]],
    device const MRJointDescriptorGPU* joints [[buffer(2)]],
    device const float* sourceQ [[buffer(3)]],
    device const float* sourceV [[buffer(4)]],
    device const float* acceleration [[buffer(5)]],
    device const MRCCDEventStateGPU* eventStates [[buffer(6)]],
    device float* candidateQ [[buffer(7)]],
    device float* candidateV [[buffer(8)]],
    device MRABAStatusGPU* abaStatuses [[buffer(9)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(10)]],
    constant uint& mode [[buffer(11)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    MRMetalWorldContactStatusGPU contactStatus =
        statuses[environment];
    if (contactStatus.code != MR_STEP_SUCCESS) {
        return;
    }
    const float timestep = eventSegmentDuration(
        eventStates[environment],
        mode
    );
    const uint qBase = environment * dispatch.qStride;
    const uint vBase = environment * dispatch.vStride;
    for (uint coordinate = 0u;
         coordinate < dispatch.nq;
         ++coordinate) {
        candidateQ[qBase + coordinate] =
            sourceQ[qBase + coordinate];
    }
    for (uint owner = 0u;
         owner < dispatch.articulationCount;
         ++owner) {
        const MRArticulationGPU articulation =
            articulations[owner];
        const uint abaStatusIndex =
            owner * dispatch.environmentCount + environment;
        MRABAStatusGPU abaStatus =
            abaStatuses[abaStatusIndex];
        for (uint localV = 0u;
             localV < articulation.nv;
             ++localV) {
            const uint globalV =
                articulation.vOffset + localV;
            const float value =
                sourceV[vBase + globalV] +
                timestep * acceleration[vBase + globalV];
            if (!isfinite(value)) {
                abaStatus.code = MR_ABA_NONFINITE_RESULT;
                abaStatus.failingIndex = globalV;
                contactStatus.code =
                    MR_STEP_NONFINITE_RESULT;
                contactStatus.firstFailingConstraint =
                    globalV;
                abaStatuses[abaStatusIndex] = abaStatus;
                statuses[environment] = contactStatus;
                return;
            }
            candidateV[vBase + globalV] = value;
        }
        const uint articulationQ =
            qBase + articulation.qOffset;
        const uint articulationV =
            vBase + articulation.vOffset;
        if (articulation.rootType == MR_ROOT_FLOATING) {
            candidateQ[articulationQ + 0u] =
                sourceQ[articulationQ + 0u] +
                timestep * candidateV[articulationV + 0u];
            candidateQ[articulationQ + 1u] =
                sourceQ[articulationQ + 1u] +
                timestep * candidateV[articulationV + 1u];
            candidateQ[articulationQ + 2u] =
                sourceQ[articulationQ + 2u] +
                timestep * candidateV[articulationV + 2u];
            float4 orientation;
            if (!integrateQuaternion(
                    float4(
                        sourceQ[articulationQ + 3u],
                        sourceQ[articulationQ + 4u],
                        sourceQ[articulationQ + 5u],
                        sourceQ[articulationQ + 6u]
                    ),
                    float3(
                        candidateV[articulationV + 3u],
                        candidateV[articulationV + 4u],
                        candidateV[articulationV + 5u]
                    ),
                    timestep,
                    orientation
                )) {
                abaStatus.code = MR_ABA_NONFINITE_RESULT;
                abaStatus.failingIndex =
                    articulation.vOffset + 3u;
                contactStatus.code =
                    MR_STEP_NONFINITE_RESULT;
                contactStatus.firstFailingConstraint =
                    articulation.vOffset + 3u;
                abaStatuses[abaStatusIndex] = abaStatus;
                statuses[environment] = contactStatus;
                return;
            }
            candidateQ[articulationQ + 3u] = orientation.x;
            candidateQ[articulationQ + 4u] = orientation.y;
            candidateQ[articulationQ + 5u] = orientation.z;
            candidateQ[articulationQ + 6u] = orientation.w;
        }
        for (uint localJoint = 0u;
             localJoint < articulation.jointCount;
             ++localJoint) {
            const MRJointDescriptorGPU joint =
                joints[
                    articulation.firstJoint + localJoint
                ];
            if (joint.nv != 1u) {
                continue;
            }
            candidateQ[qBase + joint.qOffset] =
                sourceQ[qBase + joint.qOffset] +
                timestep *
                    candidateV[vBase + joint.vOffset];
        }
    }
}

// Predicts free/kinematic scene state over either the remaining CCD interval
// or the selected TOI duration. Selected mode also advances pose so a
// following discrete projection observes the literal event configuration.
kernel void mr_world_predict_scene_event(
    device const MRWorldGPU& world [[buffer(0)]],
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(1)]],
    device const MRBodyPropertiesGPU* bodyProperties [[buffer(2)]],
    device const uint* sceneBodyIndices [[buffer(3)]],
    device const MRBodyStateGPU* currentBodies [[buffer(4)]],
    device const MRCCDEventStateGPU* eventStates [[buffer(5)]],
    device MRBodyStateGPU* candidateBodies [[buffer(6)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(7)]],
    constant uint& mode [[buffer(8)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount ||
        statuses[environment].code != MR_STEP_SUCCESS) {
        return;
    }
    const float timestep = eventSegmentDuration(
        eventStates[environment],
        mode
    );
    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    for (uint body = 0u; body < dispatch.bodyCount; ++body) {
        candidateBodies[bodyBase + body] =
            currentBodies[bodyBase + body];
    }
    for (uint localScene = 0u;
         localScene < dispatch.sceneBodyCount;
         ++localScene) {
        const uint globalBody = sceneBodyIndices[localScene];
        MRBodyStateGPU state =
            candidateBodies[bodyBase + globalBody];
        device const MRBodyPropertiesGPU& properties =
            bodyProperties[globalBody];
        if (properties.motionType == MR_MOTION_STATIC) {
            continue;
        }
        if (properties.motionType == MR_MOTION_DYNAMIC) {
            const float linearScale = exp(
                -timestep *
                    properties.dampingAndSpeedLimits.x
            );
            const float angularScale = exp(
                -timestep *
                    properties.dampingAndSpeedLimits.y
            );
            state.linearVelocityAndInverseMass.xyz =
                linearScale *
                    state.linearVelocityAndInverseMass.xyz +
                timestep * world.gravityAndTimestep.xyz;
            state.angularVelocity.xyz *= angularScale;
        }
        if (mode == MR_CCD_SEGMENT_SELECTED) {
            state.position.xyz +=
                timestep *
                state.linearVelocityAndInverseMass.xyz;
            state.position.w = 1.0f;
            float4 orientation;
            if (!integrateQuaternion(
                    state.orientation,
                    state.angularVelocity.xyz,
                    timestep,
                    orientation
                )) {
                MRMetalWorldContactStatusGPU status =
                    statuses[environment];
                status.code = MR_STEP_NONFINITE_RESULT;
                status.firstFailingConstraint = globalBody;
                statuses[environment] = status;
                return;
            }
            state.orientation = orientation;
            if (!writeWorldInverseInertia(
                    state,
                    properties,
                    orientation
                )) {
                MRMetalWorldContactStatusGPU status =
                    statuses[environment];
                status.code = MR_STEP_NONFINITE_RESULT;
                status.firstFailingConstraint = globalBody;
                statuses[environment] = status;
                return;
            }
        }
        if (!finite4(state.position) ||
            !finite4(state.orientation) ||
            !finite4(state.linearVelocityAndInverseMass) ||
            !finite4(state.angularVelocity)) {
            MRMetalWorldContactStatusGPU status =
                statuses[environment];
            status.code = MR_STEP_NONFINITE_RESULT;
            status.firstFailingConstraint = globalBody;
            statuses[environment] = status;
            return;
        }
        candidateBodies[bodyBase + globalBody] = state;
    }
}

// Overlays articulation poses at a selected event time onto the already
// advanced scene-body array.  Keeping this separate from
// mr_world_build_body_states avoids repacking the compact scene state between
// literal CCD segments.
kernel void mr_world_overlay_event_articulation_bodies(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRArticulationGPU* articulations [[buffer(1)]],
    device const MRBodyPropertiesGPU* bodyProperties [[buffer(2)]],
    device const MRArticulatedBodyPoseGPU* bodyPoses [[buffer(3)]],
    device const MRArticulatedOperatorStatusGPU* operatorStatuses
        [[buffer(4)]],
    device MRBodyStateGPU* eventBodies [[buffer(5)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(6)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    if (status.code != MR_STEP_SUCCESS) {
        return;
    }
    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    const uint poseBase =
        environment * dispatch.bodyCount;
    for (uint owner = 0u;
         owner < dispatch.articulationCount;
         ++owner) {
        const MRArticulatedOperatorStatusGPU operatorStatus =
            operatorStatuses[
                owner * dispatch.environmentCount +
                environment
            ];
        if (operatorStatus.code !=
                MR_ARTICULATED_OPERATOR_SUCCESS) {
            status.code = mapOperatorStatus(
                operatorStatus.code
            );
            status.firstFailingConstraint =
                operatorStatus.failingIndex;
            statuses[environment] = status;
            return;
        }
        const MRArticulationGPU articulation =
            articulations[owner];
        for (uint localBody = 0u;
             localBody < articulation.bodyCount;
             ++localBody) {
            const uint globalBody =
                articulation.firstBody + localBody;
            const MRArticulatedBodyPoseGPU pose =
                bodyPoses[poseBase + globalBody];
            MRBodyStateGPU state = {};
            state.position = pose.position;
            state.orientation = pose.orientation;
            state.linearVelocityAndInverseMass.w = 0.0f;
            state.flagsAndIndices[0] =
                bodyProperties[globalBody].motionType;
            state.flagsAndIndices[1] = owner;
            state.flagsAndIndices[2] = globalBody;
            state.flagsAndIndices[3] = 0u;
            eventBodies[bodyBase + globalBody] = state;
        }
    }
}

// Enforces scalar position/velocity limits on the constrained candidate.
// Before ordinary integration it clips velocity to the admissible one-step
// interval; selected-TOI mode repairs a roundoff-scale boundary crossing and
// removes only the outward velocity component.
kernel void mr_world_project_joint_limits(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRArticulationGPU* articulations [[buffer(1)]],
    device const MRDofPropertiesGPU* dofs [[buffer(2)]],
    device const float* sourceQ [[buffer(3)]],
    device float* candidateQ [[buffer(4)]],
    device float* candidateV [[buffer(5)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(6)]],
    constant uint& stateAlreadyIntegrated [[buffer(7)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount ||
        statuses[environment].code != MR_STEP_SUCCESS) {
        return;
    }
    const uint qBase = environment * dispatch.qStride;
    const uint vBase = environment * dispatch.vStride;
    const float timestep = max(
        dispatch.ccdMode == MR_WORLD_CCD_HYBRID
            ? statuses[environment].eventTimes.w
            : dispatch.timestepAndBias.x,
        1.0e-8f
    );
    for (uint owner = 0u;
         owner < dispatch.articulationCount;
         ++owner) {
        const MRArticulationGPU articulation =
            articulations[owner];
        for (uint localV = 0u;
             localV < articulation.nv;
             ++localV) {
            const uint globalV =
                articulation.vOffset + localV;
            device const MRDofPropertiesGPU& dof =
                dofs[globalV];
            float velocity = candidateV[vBase + globalV];
            if ((dof.flags &
                 MR_DOF_FLAG_VELOCITY_LIMIT) != 0u) {
                velocity = clamp(
                    velocity,
                    -dof.limits.z,
                    dof.limits.z
                );
            }
            if ((dof.flags &
                 MR_DOF_FLAG_POSITION_LIMIT) != 0u &&
                dof.qIndex != MR_INVALID_INDEX &&
                dof.qIndex >= articulation.qOffset &&
                dof.qIndex <
                    articulation.qOffset + articulation.nq) {
                const uint globalQ = dof.qIndex;
                if (stateAlreadyIntegrated != 0u) {
                    float position =
                        candidateQ[qBase + globalQ];
                    if (position < dof.limits.x) {
                        position = dof.limits.x;
                        velocity = max(velocity, 0.0f);
                    } else if (position > dof.limits.y) {
                        position = dof.limits.y;
                        velocity = min(velocity, 0.0f);
                    }
                    candidateQ[qBase + globalQ] = position;
                } else {
                    const float position =
                        sourceQ[qBase + globalQ];
                    const float minimumVelocity =
                        (dof.limits.x - position) / timestep;
                    const float maximumVelocity =
                        (dof.limits.y - position) / timestep;
                    velocity = clamp(
                        velocity,
                        minimumVelocity,
                        maximumVelocity
                    );
                }
            }
            if (!isfinite(velocity)) {
                MRMetalWorldContactStatusGPU status =
                    statuses[environment];
                status.code = MR_STEP_NONFINITE_RESULT;
                status.firstFailingConstraint = globalV;
                statuses[environment] = status;
                return;
            }
            candidateV[vBase + globalV] = velocity;
        }
    }
}

// Reduces device-resident contact counts into the point-query prefix used by
// the articulated factor/Jacobian pass. Fixed strides remain unchanged and
// no count crosses the CPU boundary.
kernel void mr_world_finalize_factor_dispatch(
    device const MRMetalWorldContactDispatchGPU& contactDispatch
        [[buffer(0)]],
    device const MRMetalWorldContactStatusGPU* statuses
        [[buffer(1)]],
    device MRArticulatedOperatorDispatchGPU* operatorDispatch
        [[buffer(2)]],
    device MRIndirectDispatchArgumentsGPU* indirectArguments
        [[buffer(3)]],
    const uint threadIndex [[thread_position_in_grid]]
) {
    if (threadIndex != 0u) {
        return;
    }
    uint maximumConstraints = 0u;
    for (uint environment = 0u;
         environment < contactDispatch.environmentCount;
         ++environment) {
        const MRMetalWorldContactStatusGPU status =
            statuses[environment];
        if (status.code == MR_STEP_SUCCESS) {
            maximumConstraints = max(
                maximumConstraints,
                min(
                    status.requiredConstraints,
                    contactDispatch.constraintCapacity
                )
            );
        }
    }
    const uint activePointCount = min(
        2u * maximumConstraints,
        contactDispatch.pointQueryStride
    );
    for (uint owner = 0u;
         owner < contactDispatch.articulationCount;
         ++owner) {
        operatorDispatch[owner].pointCount =
            activePointCount;
    }
    MRIndirectDispatchArgumentsGPU arguments = {};
    arguments.threadgroupsX =
        activePointCount == 0u
        ? 0u
        : contactDispatch.environmentCount;
    arguments.threadgroupsY = 1u;
    arguments.threadgroupsZ = 1u;
    arguments.activeCount = activePointCount;
    indirectArguments[0] = arguments;
    MRIndirectDispatchArgumentsGPU contactArguments = {};
    contactArguments.threadgroupsX =
        activePointCount == 0u
        ? 0u
        : (
            contactDispatch.environmentCount + 63u
          ) / 64u;
    contactArguments.threadgroupsY = 1u;
    contactArguments.threadgroupsZ = 1u;
    contactArguments.activeCount = maximumConstraints;
    indirectArguments[1] = contactArguments;
}

// Environments can have different active contact counts while the articulated
// operator consumes one batch-wide point prefix. Only the short tail is
// initialized; active queries were emitted by the collision compiler.
kernel void mr_world_fill_point_query_tail(
    device const MRMetalWorldContactDispatchGPU& contactDispatch
        [[buffer(0)]],
    device const MRArticulatedOperatorDispatchGPU& operatorDispatch
        [[buffer(1)]],
    device const MRArticulationGPU* articulations [[buffer(2)]],
    device const MRMetalWorldContactStatusGPU* statuses
        [[buffer(3)]],
    device MRArticulatedPointImpulseGPU* pointQueries
        [[buffer(4)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= contactDispatch.environmentCount ||
        statuses[environment].code != MR_STEP_SUCCESS) {
        return;
    }
    const uint activePoints = min(
        2u * statuses[environment].requiredConstraints,
        operatorDispatch.pointCount
    );
    MRArticulatedPointImpulseGPU dummy = {};
    dummy.bodyIndex =
        articulations[contactDispatch.articulationIndex].rootBody;
    dummy.flags = MR_ARTICULATED_POINT_INACTIVE;
    const uint base =
        environment * contactDispatch.pointQueryStride;
    for (uint point = activePoints;
         point < operatorDispatch.pointCount;
         ++point) {
        pointQueries[base + point] = dummy;
    }
}

// Initializes the articulation-major point-query tensor before canonical
// manifold scatter. Each articulation receives a valid root-body query in
// every fixed-capacity slot; the scatter kernel overwrites only endpoints
// actually owned by that articulation. This lets every articulation operator
// run over one fixed packet shape without host-visible contact counts.
kernel void mr_world_initialize_multi_articulation_queries(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRArticulationGPU* articulations [[buffer(1)]],
    device MRArticulatedPointImpulseGPU* pointQueries [[buffer(2)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(3)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount ||
        statuses[environment].code != MR_STEP_SUCCESS) {
        return;
    }
    for (uint owner = 0u;
         owner < dispatch.articulationCount;
         ++owner) {
        MRArticulatedPointImpulseGPU dummy = {};
        dummy.bodyIndex = articulations[owner].rootBody;
        dummy.flags = MR_ARTICULATED_POINT_INACTIVE;
        const uint base =
            (owner * dispatch.environmentCount + environment) *
            dispatch.pointQueryStride;
        for (uint point = 0u;
             point < dispatch.pointQueryStride;
             ++point) {
            pointQueries[base + point] = dummy;
        }
    }
}

// Composes articulation-local Cholesky factors and analytic point Jacobians
// into the world-global generalized-velocity layout. The factor is block
// diagonal because disconnected articulation trees have no inertial
// cross-terms; constraints create coupling through the shared Jacobian and
// island solve. Staging is articulation-major and compact in each local nv.
kernel void mr_world_compose_multi_articulation_operator(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRArticulationGPU* articulations [[buffer(1)]],
    device const MRArticulatedOperatorDispatchGPU* operatorDispatches
        [[buffer(2)]],
    device const MRArticulatedOperatorStatusGPU* operatorStatuses
        [[buffer(3)]],
    device const float* factorStaging [[buffer(4)]],
    device const float* jacobianStaging [[buffer(5)]],
    device float* factorMatrix [[buffer(6)]],
    device float* pointJacobians [[buffer(7)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(8)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    if (status.code != MR_STEP_SUCCESS) {
        return;
    }
    for (uint owner = 0u;
         owner < dispatch.articulationCount;
         ++owner) {
        const MRArticulatedOperatorStatusGPU localStatus =
            operatorStatuses[
                owner * dispatch.environmentCount + environment
            ];
        if (localStatus.code !=
                MR_ARTICULATED_OPERATOR_SUCCESS) {
            status.code = mapOperatorStatus(localStatus.code);
            status.firstFailingConstraint =
                localStatus.failingIndex;
            statuses[environment] = status;
            return;
        }
    }

    const uint factorBase =
        environment * dispatch.factorStride;
    for (uint entry = 0u;
         entry < dispatch.nv * dispatch.nv;
         ++entry) {
        factorMatrix[factorBase + entry] = 0.0f;
    }
    const uint rowCount =
        dispatch.pointQueryStride * 3u;
    const uint jacobianBase =
        environment * rowCount * dispatch.nv;
    for (uint entry = 0u;
         entry < rowCount * dispatch.nv;
         ++entry) {
        pointJacobians[jacobianBase + entry] = 0.0f;
    }

    ulong factorPrefix = 0ul;
    ulong jacobianPrefix = 0ul;
    for (uint owner = 0u;
         owner < dispatch.articulationCount;
         ++owner) {
        const MRArticulationGPU articulation =
            articulations[owner];
        const MRArticulatedOperatorDispatchGPU localDispatch =
            operatorDispatches[owner];
        const ulong localFactorStride =
            static_cast<ulong>(articulation.nv) *
            static_cast<ulong>(articulation.nv);
        const ulong localJacobianStride =
            static_cast<ulong>(rowCount) *
            static_cast<ulong>(articulation.nv);
        const ulong localFactorBase =
            factorPrefix +
            static_cast<ulong>(environment) *
                localFactorStride;
        const ulong localJacobianBase =
            jacobianPrefix +
            static_cast<ulong>(environment) *
                localJacobianStride;
        if (localDispatch.massMatrixStride !=
                localFactorStride ||
            localDispatch.pointJacobianStride !=
                localJacobianStride ||
            articulation.vOffset + articulation.nv >
                dispatch.nv) {
            status.code = MR_STEP_UNSUPPORTED;
            status.firstFailingConstraint = owner;
            statuses[environment] = status;
            return;
        }
        for (uint row = 0u;
             row < articulation.nv;
             ++row) {
            for (uint column = 0u;
                 column < articulation.nv;
                 ++column) {
                factorMatrix[
                    factorBase +
                    (articulation.vOffset + row) *
                        dispatch.nv +
                    articulation.vOffset + column
                ] = factorStaging[
                    localFactorBase +
                    static_cast<ulong>(row) *
                        articulation.nv +
                    column
                ];
            }
        }
        for (uint row = 0u; row < rowCount; ++row) {
            for (uint column = 0u;
                 column < articulation.nv;
                 ++column) {
                pointJacobians[
                    jacobianBase +
                    row * dispatch.nv +
                    articulation.vOffset + column
                ] = jacobianStaging[
                    localJacobianBase +
                    static_cast<ulong>(row) *
                        articulation.nv +
                    column
                ];
            }
        }
        factorPrefix +=
            static_cast<ulong>(dispatch.environmentCount) *
            localFactorStride;
        jacobianPrefix +=
            static_cast<ulong>(dispatch.environmentCount) *
            localJacobianStride;
    }
    statuses[environment] = status;
}

// Expands the immutable mechanism program into the fixed-capacity runtime
// prefix. The source remains canonical ConstraintIR v2; normalization to two
// endpoint and three row slots happens only in invocation-local storage so
// collision and rod scatter can append without a second offset domain.
kernel void mr_world_seed_authored_constraint_ir(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRConstraintIRBlockGPU* sourceBlocks [[buffer(1)]],
    device const MRConstraintIREndpointGPU* sourceEndpoints [[buffer(2)]],
    device const MRConstraintIRRowGPU* sourceRows [[buffer(3)]],
    device const MRConstraintIRConeGPU* sourceCones [[buffer(4)]],
    device const float* sourceWarmImpulses [[buffer(5)]],
    device const MRWorldDynamicNodeGPU* dynamicNodes [[buffer(6)]],
    device MRContactConstraintGPU* contacts [[buffer(7)]],
    device MRContactPointMetaGPU* metadata [[buffer(8)]],
    device MRConstraintIRBlockGPU* blocks [[buffer(9)]],
    device MRConstraintIREndpointGPU* endpoints [[buffer(10)]],
    device MRConstraintEndpointRuntimeGPU* endpointRuntime [[buffer(11)]],
    device MRConstraintIRRowGPU* rows [[buffer(12)]],
    device MRConstraintIRConeGPU* cones [[buffer(13)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(14)]],
    device const uint* bodyDynamicNodes [[buffer(15)]],
    device const MRBodyStateGPU* candidateBodies [[buffer(16)]],
    device const MRRodNodeStateGPU* candidateRodNodes [[buffer(17)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount ||
        dispatch.authoredConstraintCount == 0u) {
        return;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    if (status.code != MR_STEP_SUCCESS) {
        return;
    }
    if (dispatch.authoredConstraintCount >
            dispatch.constraintCapacity ||
        3ul * dispatch.authoredConstraintCount >
            dispatch.rowCapacity ||
        dispatch.authoredEndpointCount >
            2u * dispatch.authoredConstraintCount) {
        status.code = MR_STEP_CONSTRAINT_CAPACITY_OVERFLOW;
        status.requiredConstraints =
            dispatch.authoredConstraintCount;
        status.requiredRows =
            3u * dispatch.authoredConstraintCount;
        statuses[environment] = status;
        return;
    }

    const uint constraintBase =
        environment * dispatch.constraintStride;
    const uint endpointBase =
        2u * environment * dispatch.constraintStride;
    const uint rowBase =
        environment * dispatch.rowStride;
    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    const uint rodNodeBase =
        environment * dispatch.rodNodeCount;
    for (uint localBlock = 0u;
         localBlock < dispatch.authoredConstraintCount;
         ++localBlock) {
        const MRConstraintIRBlockGPU source =
            sourceBlocks[localBlock];
        if (source.dimension == 0u ||
            source.dimension > 3u ||
            source.endpointCount == 0u ||
            source.endpointCount > 2u ||
            source.endpointOffset + source.endpointCount >
                dispatch.authoredEndpointCount ||
            source.rowOffset + source.dimension >
                dispatch.authoredRowCount ||
            (
                source.coneIndex !=
                    MR_CONSTRAINT_IR_INVALID_INDEX &&
                source.coneIndex >= dispatch.authoredConeCount
            )) {
            status.code = MR_STEP_UNSUPPORTED;
            status.firstFailingConstraint = localBlock;
            status.firstFailingStableKeyLow =
                source.key.words[0];
            status.firstFailingStableKeyHigh =
                source.key.words[1];
            statuses[environment] = status;
            return;
        }

        const uint outputConstraint =
            constraintBase + localBlock;
        MRConstraintIRBlockGPU runtime = source;
        runtime.flags |=
            MR_CONSTRAINT_IR_BLOCK_GENERALIZED;
        runtime.islandIndex = MR_INVALID_INDEX;
        runtime.endpointOffset = 2u * localBlock;
        runtime.endpointCount = 2u;
        runtime.rowOffset = 3u * localBlock;
        runtime.impulseOffset = 3u * localBlock;
        runtime.coneIndex = localBlock;
        blocks[outputConstraint] = runtime;

        MRContactConstraintGPU compatibility = {};
        compatibility.bodyA = MR_INVALID_INDEX;
        compatibility.bodyB = MR_INVALID_INDEX;
        compatibility.flags =
            MR_CONSTRAINT_FLAG_GENERALIZED |
            (
                (source.flags &
                 MR_CONSTRAINT_IR_BLOCK_DISABLED) != 0u
                ? MR_CONSTRAINT_FLAG_DISABLED
                : 0u
            );
        compatibility.islandIndex = MR_INVALID_INDEX;
        compatibility.pairKey =
            (static_cast<ulong>(source.key.words[0]) << 32u) |
            source.key.words[1];
        compatibility.featureKey =
            (static_cast<ulong>(source.key.words[2]) << 32u) |
            source.key.words[3];
        for (uint localRow = 0u;
             localRow < source.dimension;
             ++localRow) {
            compatibility.impulses[localRow] =
                sourceWarmImpulses[
                    source.impulseOffset + localRow
                ];
        }
        contacts[outputConstraint] = compatibility;
        MRContactPointMetaGPU emptyMetadata = {};
        emptyMetadata.colliderA = MR_INVALID_INDEX;
        emptyMetadata.colliderB = MR_INVALID_INDEX;
        emptyMetadata.manifoldIndex = MR_INVALID_INDEX;
        emptyMetadata.pointIndex = MR_INVALID_INDEX;
        metadata[outputConstraint] = emptyMetadata;

        for (uint slot = 0u; slot < 2u; ++slot) {
            const uint destination =
                endpointBase + 2u * localBlock + slot;
            MRConstraintIREndpointGPU endpoint = {};
            endpoint.objectIndex =
                MR_CONSTRAINT_IR_INVALID_INDEX;
            endpoint.articulationIndex =
                MR_CONSTRAINT_IR_INVALID_INDEX;
            endpoint.linkIndex =
                MR_CONSTRAINT_IR_INVALID_INDEX;
            endpoint.role = MR_CONSTRAINT_IR_ENDPOINT_WORLD;
            endpoint.jacobianKind =
                MR_CONSTRAINT_IR_JACOBIAN_WORLD_POINT;
            MRConstraintEndpointRuntimeGPU binding = {};
            binding.dynamicNode =
                MR_CONSTRAINT_IR_INVALID_INDEX;
            binding.ownerKind =
                MR_CONSTRAINT_IR_OWNER_WORLD;
            binding.ownerIndex =
                MR_CONSTRAINT_IR_INVALID_INDEX;
            binding.elementIndex =
                MR_CONSTRAINT_IR_INVALID_INDEX;
            binding.queryIndex =
                MR_CONSTRAINT_IR_INVALID_INDEX;
            binding.secondaryIndex =
                MR_CONSTRAINT_IR_INVALID_INDEX;
            binding.twistIndex =
                MR_CONSTRAINT_IR_INVALID_INDEX;
            if (slot < source.endpointCount) {
                endpoint = sourceEndpoints[
                    source.endpointOffset + slot
                ];
                uint dynamicNode =
                    MR_CONSTRAINT_IR_INVALID_INDEX;
                if (endpoint.jacobianKind ==
                    MR_CONSTRAINT_IR_JACOBIAN_GENERALIZED) {
                    if (endpoint.articulationIndex >=
                            dispatch.articulationCount ||
                        endpoint.objectIndex >= dispatch.nv) {
                        status.code = MR_STEP_UNSUPPORTED;
                        status.firstFailingConstraint =
                            localBlock;
                        statuses[environment] = status;
                        return;
                    }
                    for (uint node = 0u;
                         node < dispatch.dynamicNodeCount;
                         ++node) {
                        const MRWorldDynamicNodeGPU candidate =
                            dynamicNodes[node];
                        if (candidate.kind ==
                                MR_WORLD_DYNAMIC_NODE_ARTICULATION &&
                            candidate.ownerIndex ==
                                endpoint.articulationIndex) {
                            dynamicNode = node;
                            break;
                        }
                    }
                    binding.ownerKind =
                        MR_CONSTRAINT_IR_OWNER_ARTICULATION;
                    binding.ownerIndex =
                        endpoint.articulationIndex;
                    binding.elementIndex =
                        endpoint.objectIndex;
                } else if (
                    endpoint.jacobianKind ==
                    MR_CONSTRAINT_IR_JACOBIAN_ROD_NODE
                ) {
                    if (endpoint.articulationIndex >=
                            dispatch.rodCount ||
                        endpoint.objectIndex >=
                            dispatch.rodNodeCount) {
                        status.code = MR_STEP_UNSUPPORTED;
                        status.firstFailingConstraint =
                            localBlock;
                        statuses[environment] = status;
                        return;
                    }
                    for (uint node = 0u;
                         node < dispatch.dynamicNodeCount;
                         ++node) {
                        const MRWorldDynamicNodeGPU candidate =
                            dynamicNodes[node];
                        if (candidate.kind ==
                                MR_WORLD_DYNAMIC_NODE_ROD &&
                            candidate.ownerIndex ==
                                endpoint.articulationIndex) {
                            dynamicNode = node;
                            break;
                        }
                    }
                    binding.ownerKind =
                        MR_CONSTRAINT_IR_OWNER_ROD_NODE;
                    binding.ownerIndex =
                        endpoint.articulationIndex;
                    binding.elementIndex =
                        endpoint.objectIndex;
                    binding.weights.x = 1.0f;
                } else if (
                    endpoint.jacobianKind ==
                    MR_CONSTRAINT_IR_JACOBIAN_BODY_LOCAL_POINT
                ) {
                    if (endpoint.objectIndex >=
                        dispatch.bodyCount) {
                        status.code = MR_STEP_UNSUPPORTED;
                        status.firstFailingConstraint =
                            localBlock;
                        statuses[environment] = status;
                        return;
                    }
                    dynamicNode =
                        bodyDynamicNodes[endpoint.objectIndex];
                    binding.ownerKind =
                        MR_CONSTRAINT_IR_OWNER_FREE_BODY;
                    binding.ownerIndex =
                        endpoint.objectIndex;
                    binding.elementIndex =
                        endpoint.objectIndex;
                    binding.localAnchorOrRadial =
                        endpoint.anchor;
                } else if (
                    endpoint.role ==
                        MR_CONSTRAINT_IR_ENDPOINT_WORLD &&
                    endpoint.jacobianKind ==
                        MR_CONSTRAINT_IR_JACOBIAN_WORLD_POINT
                ) {
                    binding.localAnchorOrRadial =
                        endpoint.anchor;
                } else {
                    status.code = MR_STEP_UNSUPPORTED;
                    status.firstFailingConstraint =
                        localBlock;
                    statuses[environment] = status;
                    return;
                }
                if (endpoint.role !=
                        MR_CONSTRAINT_IR_ENDPOINT_WORLD &&
                    dynamicNode ==
                        MR_CONSTRAINT_IR_INVALID_INDEX) {
                    status.code = MR_STEP_UNSUPPORTED;
                    status.firstFailingConstraint =
                        localBlock;
                    statuses[environment] = status;
                    return;
                }
                binding.dynamicNode = dynamicNode;
                if (dynamicNode !=
                    MR_CONSTRAINT_IR_INVALID_INDEX) {
                    binding.flags =
                        MR_CONSTRAINT_IR_RUNTIME_DYNAMIC;
                }
            }
            endpoints[destination] = endpoint;
            endpointRuntime[destination] = binding;
        }

        for (uint slot = 0u; slot < 3u; ++slot) {
            MRConstraintIRRowGPU row = {};
            if (slot < source.dimension) {
                row = sourceRows[source.rowOffset + slot];
            }
            if (slot < source.dimension &&
                (source.flags &
                 MR_CONSTRAINT_IR_BLOCK_ROD_ATTACHMENT) !=
                    0u) {
                const MRConstraintIREndpointGPU rodEndpoint =
                    sourceEndpoints[source.endpointOffset];
                const MRConstraintIREndpointGPU targetEndpoint =
                    sourceEndpoints[source.endpointOffset + 1u];
                const float3 rodPosition =
                    candidateRodNodes[
                        rodNodeBase + rodEndpoint.objectIndex
                    ].position.xyz;
                float3 targetPosition =
                    targetEndpoint.anchor.xyz;
                if (targetEndpoint.role !=
                    MR_CONSTRAINT_IR_ENDPOINT_WORLD) {
                    device const MRBodyStateGPU& body =
                        candidateBodies[
                            bodyBase + targetEndpoint.objectIndex
                        ];
                    targetPosition =
                        body.position.xyz +
                        multiply(
                            rotationMatrix(body.orientation),
                            targetEndpoint.anchor.xyz
                        );
                }
                row.positionError = dot(
                    row.direction.xyz,
                    targetPosition - rodPosition
                );
            }
            rows[rowBase + 3u * localBlock + slot] = row;
        }
        MRConstraintIRConeGPU cone = {};
        if (source.coneIndex !=
            MR_CONSTRAINT_IR_INVALID_INDEX) {
            cone = sourceCones[source.coneIndex];
        }
        cones[outputConstraint] = cone;
    }
    statuses[environment] = status;
}

// Evaluates the canonical IR at the current microstep after point Jacobians
// and free velocities are available.
kernel void mr_world_evaluate_constraint_ir(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRContactConstraintGPU* contacts [[buffer(1)]],
    device MRContactConstraintGPU* mutableContacts [[buffer(2)]],
    device const MRConstraintIRBlockGPU* blocks [[buffer(3)]],
    device const MRConstraintIRRowGPU* rows [[buffer(4)]],
    device const MRConstraintIRConeGPU* cones [[buffer(5)]],
    device const MRBodyStateGPU* candidateBodies [[buffer(6)]],
    device const float* candidateV [[buffer(7)]],
    device const float* pointJacobians [[buffer(8)]],
    device const MRArticulatedOperatorStatusGPU* operatorStatuses [[buffer(9)]],
    device MREvaluatedConstraintIRRowGPU* evaluatedRows [[buffer(10)]],
    device MREvaluatedConstraintIRConeGPU* evaluatedCones [[buffer(11)]],
    device MRArticulationFactorCacheGPU* factorCaches [[buffer(12)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(13)]],
    device const MRConstraintIREndpointGPU* endpoints [[buffer(14)]],
    device const MRRodNodeStateGPU* candidateRodNodes [[buffer(15)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    if (status.code != MR_STEP_SUCCESS) {
        return;
    }
    float minimumPivot = INFINITY;
    float maximumPivot = 0.0f;
    float maximumFactorResidual = 0.0f;
    for (uint owner = 0u;
         owner < dispatch.articulationCount;
         ++owner) {
        const uint statusIndex =
            owner * dispatch.environmentCount + environment;
        const MRArticulatedOperatorStatusGPU operatorStatus =
            operatorStatuses[statusIndex];
        MRArticulationFactorCacheGPU cache = {};
        cache.environment = environment;
        cache.articulationIndex = owner;
        cache.nv = operatorStatus.nv;
        cache.generation = status.physicsSubstep;
        cache.code = operatorStatus.code;
        cache.failingIndex = operatorStatus.failingIndex;
        cache.diagnostics = operatorStatus.diagnostics;
        factorCaches[statusIndex] = cache;
        if (operatorStatus.code !=
            MR_ARTICULATED_OPERATOR_SUCCESS) {
            status.code = mapOperatorStatus(
                operatorStatus.code
            );
            status.firstFailingConstraint =
                operatorStatus.failingIndex;
            statuses[environment] = status;
            return;
        }
        minimumPivot = min(
            minimumPivot,
            operatorStatus.diagnostics.x
        );
        maximumPivot = max(
            maximumPivot,
            operatorStatus.diagnostics.y
        );
        maximumFactorResidual = max(
            maximumFactorResidual,
            operatorStatus.diagnostics.z
        );
    }
    status.diagnostics.z =
        dispatch.articulationCount == 0u
        ? 0.0f
        : minimumPivot;
    status.diagnostics.w = maximumPivot;
    status.residuals.w = maximumFactorResidual;

    const uint constraintBase =
        environment * dispatch.constraintStride;
    const uint rowBase =
        environment * dispatch.rowStride;
    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    const uint velocityBase = environment * dispatch.nv;
    const uint pointJacobianBase =
        environment *
        (dispatch.pointQueryStride * 3u * dispatch.nv);
    const float timestep =
        dispatch.ccdMode == MR_WORLD_CCD_HYBRID
        ? max(
              status.eventTimes.w,
              max(
                  dispatch.ccdParameters.y,
                  1.0e-8f
              )
          )
        : dispatch.timestepAndBias.x;
    for (uint localConstraint = 0u;
         localConstraint < status.requiredConstraints;
         ++localConstraint) {
        const uint constraintIndex =
            constraintBase + localConstraint;
        device const MRContactConstraintGPU& contact =
            contacts[constraintIndex];
        const MRConstraintIRBlockGPU block =
            blocks[constraintIndex];
        const bool generalized =
            (block.flags &
             MR_CONSTRAINT_IR_BLOCK_GENERALIZED) != 0u;
        float relativeRows[3] = {
            0.0f,
            0.0f,
            0.0f,
        };
        if (generalized) {
            const uint endpointBase =
                2u * environment *
                    dispatch.constraintStride +
                block.endpointOffset;
            for (uint endpointIndex = 0u;
                 endpointIndex < block.endpointCount;
                 ++endpointIndex) {
                const MRConstraintIREndpointGPU endpoint =
                    endpoints[endpointBase + endpointIndex];
                if (endpoint.role ==
                    MR_CONSTRAINT_IR_ENDPOINT_WORLD) {
                    continue;
                }
                const uint localRow =
                    endpoint.flags &
                    MR_CONSTRAINT_IR_ENDPOINT_ROW_MASK;
                if (localRow >= block.dimension ||
                    localRow >= 3u) {
                    status.code = MR_STEP_UNSUPPORTED;
                    status.firstFailingConstraint =
                        localConstraint;
                    statuses[environment] = status;
                    return;
                }
                if (endpoint.jacobianKind ==
                    MR_CONSTRAINT_IR_JACOBIAN_GENERALIZED) {
                    if (endpoint.objectIndex >= dispatch.nv) {
                        status.code = MR_STEP_UNSUPPORTED;
                        status.firstFailingConstraint =
                            localConstraint;
                        statuses[environment] = status;
                        return;
                    }
                    relativeRows[localRow] = fma(
                        endpoint.axis.x,
                        candidateV[
                            velocityBase + endpoint.objectIndex
                        ],
                        relativeRows[localRow]
                    );
                } else if (
                    endpoint.jacobianKind ==
                    MR_CONSTRAINT_IR_JACOBIAN_ROD_NODE
                ) {
                    if (endpoint.objectIndex >=
                        dispatch.rodNodeCount) {
                        status.code = MR_STEP_UNSUPPORTED;
                        status.firstFailingConstraint =
                            localConstraint;
                        statuses[environment] = status;
                        return;
                    }
                    const float sign =
                        endpoint.role ==
                            MR_CONSTRAINT_IR_ENDPOINT_A
                        ? -1.0f
                        : 1.0f;
                    relativeRows[localRow] +=
                        sign *
                        dot(
                            rows[
                                rowBase +
                                block.rowOffset +
                                localRow
                            ].direction.xyz,
                            candidateRodNodes[
                                environment *
                                    dispatch.rodNodeCount +
                                endpoint.objectIndex
                            ].velocity.xyz
                        );
                } else if (
                    endpoint.jacobianKind ==
                    MR_CONSTRAINT_IR_JACOBIAN_BODY_LOCAL_POINT
                ) {
                    if (endpoint.objectIndex >=
                        dispatch.bodyCount) {
                        status.code = MR_STEP_UNSUPPORTED;
                        status.firstFailingConstraint =
                            localConstraint;
                        statuses[environment] = status;
                        return;
                    }
                    device const MRBodyStateGPU& body =
                        candidateBodies[
                            bodyBase + endpoint.objectIndex
                        ];
                    const float3 lever = multiply(
                        rotationMatrix(body.orientation),
                        endpoint.anchor.xyz
                    );
                    const float3 pointVelocity =
                        body
                            .linearVelocityAndInverseMass
                            .xyz +
                        cross(
                            body.angularVelocity.xyz,
                            lever
                        );
                    const float sign =
                        endpoint.role ==
                            MR_CONSTRAINT_IR_ENDPOINT_A
                        ? -1.0f
                        : 1.0f;
                    relativeRows[localRow] +=
                        sign *
                        dot(
                            rows[
                                rowBase +
                                block.rowOffset +
                                localRow
                            ].direction.xyz,
                            pointVelocity
                        );
                } else {
                    status.code = MR_STEP_UNSUPPORTED;
                    status.firstFailingConstraint =
                        localConstraint;
                    statuses[environment] = status;
                    return;
                }
            }
        } else {
            const float3 relative = relativePointVelocity(
                localConstraint,
                contact,
                candidateBodies + bodyBase,
                dispatch.articulationIndex,
                pointJacobians,
                pointJacobianBase,
                dispatch.nv,
                candidateV + velocityBase
            );
            for (uint localRow = 0u;
                 localRow < 3u;
                 ++localRow) {
                relativeRows[localRow] = dot(
                    rows[
                        rowBase + block.rowOffset + localRow
                    ].direction.xyz,
                    relative
                );
            }
        }
        const MRConstraintIRConeGPU sourceCone =
            cones[constraintIndex];
        const float slip = generalized
            ? 0.0f
            : length(float2(
                  relativeRows[1],
                  relativeRows[2]
              ));
        const bool staticRegion =
            slip <= max(
                1.0e-3f,
                sourceCone.stictionTransitionVelocity
            );
        MREvaluatedConstraintIRConeGPU evaluatedCone = {};
        evaluatedCone.effectiveFrictionU =
            staticRegion
            ? sourceCone.staticFrictionU
            : sourceCone.dynamicFrictionU;
        evaluatedCone.effectiveFrictionV =
            staticRegion
            ? sourceCone.staticFrictionV
            : sourceCone.dynamicFrictionV;
        evaluatedCone.staticFrictionU =
            sourceCone.staticFrictionU;
        evaluatedCone.staticFrictionV =
            sourceCone.staticFrictionV;
        evaluatedCone.dynamicFrictionU =
            sourceCone.dynamicFrictionU;
        evaluatedCone.dynamicFrictionV =
            sourceCone.dynamicFrictionV;
        evaluatedCone.rollingLength = sourceCone.rollingLength;
        evaluatedCone.torsionalLength =
            sourceCone.torsionalLength;
        evaluatedCone.restitutionThreshold =
            sourceCone.restitutionThreshold;
        evaluatedCone.adhesionImpulse =
            sourceCone.adhesionImpulse;
        evaluatedCone.maximumNormalImpulse =
            sourceCone.maximumNormalImpulse;

        float restitutionVelocity = 0.0f;
        for (uint localRow = 0u;
             localRow < 3u;
             ++localRow) {
            const MRConstraintIRRowGPU source =
                rows[
                    rowBase + block.rowOffset + localRow
                ];
            float stabilization = 0.0f;
            if ((source.flags &
                 MR_CONSTRAINT_IR_ROW_POSITION_STABILIZED) !=
                0u) {
                const float tau = max(
                    source.timeConstant,
                    2.0f * timestep
                );
                float positionError = source.positionError;
                if ((source.flags &
                     MR_CONSTRAINT_IR_ROW_CONTACT_NORMAL) !=
                    0u) {
                    positionError = min(
                        positionError +
                            dispatch.timestepAndBias.y,
                        0.0f
                    );
                } else if ((source.flags &
                            MR_CONSTRAINT_IR_ROW_UNILATERAL) !=
                           0u) {
                    positionError = min(positionError, 0.0f);
                }
                const float ratio = timestep / tau;
                const float denominator =
                    1.0f +
                    2.0f * source.dampingRatio * ratio +
                    ratio * ratio;
                stabilization =
                    -timestep * positionError /
                    (tau * tau * denominator);
                stabilization = generalized
                    ? clamp(
                          stabilization,
                          -dispatch.timestepAndBias.z,
                          dispatch.timestepAndBias.z
                      )
                    : clamp(
                          max(stabilization, 0.0f),
                          0.0f,
                          dispatch.timestepAndBias.z
                      );
            }
            if (!generalized &&
                localRow == 0u &&
                (block.flags &
                 MR_CONSTRAINT_IR_BLOCK_NEW_IMPACT) != 0u) {
                const float incoming =
                    relativeRows[0] - source.targetVelocity;
                if (incoming <
                    -sourceCone.restitutionThreshold) {
                    restitutionVelocity =
                        -sourceCone.restitution * incoming;
                    stabilization = max(
                        stabilization,
                        restitutionVelocity
                    );
                }
            }
            MREvaluatedConstraintIRRowGPU evaluated = {};
            evaluated.direction = source.direction;
            evaluated.targetVelocity =
                source.targetVelocity + stabilization;
            const bool exactImpactNormal =
                localRow == 0u &&
                dispatch.ccdMode == MR_WORLD_CCD_HYBRID &&
                block.eventSlot !=
                    MR_CONSTRAINT_IR_INVALID_INDEX &&
                (block.flags &
                 MR_CONSTRAINT_IR_BLOCK_NEW_IMPACT) != 0u;
            evaluated.regularization =
                exactImpactNormal
                ? 0.0f
                : max(
                      source.compliance /
                          (timestep * timestep) +
                          source.dissipation / timestep,
                      0.0f
                  );
            evaluated.impulseLower = source.impulseLower;
            evaluated.impulseUpper = source.impulseUpper;
            evaluated.sourcePositionError =
                source.positionError;
            evaluated.stabilizationVelocity = stabilization;
            evaluated.sourceTargetVelocity =
                source.targetVelocity;
            evaluated.relativeVelocity =
                relativeRows[localRow];
            evaluated.preSolveVelocity =
                relativeRows[localRow];
            if (!finite4(evaluated.direction) ||
                !isfinite(evaluated.targetVelocity) ||
                !isfinite(evaluated.regularization)) {
                status.code = MR_STEP_NONFINITE_RESULT;
                status.firstFailingConstraint =
                    localConstraint;
                statuses[environment] = status;
                return;
            }
            evaluatedRows[
                rowBase + block.rowOffset + localRow
            ] = evaluated;
        }
        evaluatedCone.restitutionVelocity =
            restitutionVelocity;
        evaluatedCones[constraintIndex] = evaluatedCone;
        MRContactConstraintGPU updated = contact;
        updated.targetVelocityAndPreSolveNormal.w =
            relativeRows[0];
        mutableContacts[constraintIndex] = updated;
    }
    statuses[environment] = status;
}

// Deterministic minimum-root union/find over immutable typed dynamic nodes.
// Articulation links map to their tree node, free bodies to individual nodes,
// and every connected rod component to one node. Static and kinematic bodies
// map to MR_INVALID_INDEX and remain boundary endpoints.
kernel void mr_world_build_contact_islands(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRBodyStateGPU* bodies [[buffer(1)]],
    device MRContactConstraintGPU* contacts [[buffer(2)]],
    device MRConstraintIRBlockGPU* blocks [[buffer(3)]],
    device MRContactIslandGPU* islands [[buffer(4)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(5)]],
    device const MRWorldDynamicNodeGPU* dynamicNodes [[buffer(6)]],
    device const uint* bodyDynamicNodes [[buffer(7)]],
    device const MRConstraintEndpointRuntimeGPU* endpointRuntime
        [[buffer(8)]],
    device MRIslandNodeRefGPU* islandNodeReferences [[buffer(9)]],
    device MRIslandConstraintRefGPU*
        islandConstraintReferences [[buffer(10)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    if (status.code != MR_STEP_SUCCESS) {
        return;
    }
    if (dispatch.dynamicNodeCount == 0u ||
        dispatch.dynamicNodeCount >
            MR_WORLD_MAX_DYNAMIC_NODES) {
        status.code = MR_STEP_ISLAND_CAPACITY_OVERFLOW;
        status.requiredIslands = dispatch.dynamicNodeCount;
        statuses[environment] = status;
        return;
    }
    if (status.requiredConstraints == 0u) {
        status.requiredIslands = 0u;
        status.islandCount = 0u;
        statuses[environment] = status;
        return;
    }

    uint parents[MR_WORLD_MAX_DYNAMIC_NODES];
    bool active[MR_WORLD_MAX_DYNAMIC_NODES];
    for (uint node = 0u;
         node < dispatch.dynamicNodeCount;
         ++node) {
        parents[node] = node;
        active[node] = false;
    }
    const uint constraintBase =
        environment * dispatch.constraintStride;
    const uint endpointBase =
        2u * environment * dispatch.constraintStride;
    const uint islandBase =
        environment * dispatch.islandStride;
    for (uint localConstraint = 0u;
         localConstraint < status.requiredConstraints;
         ++localConstraint) {
        const MRConstraintIRBlockGPU block =
            blocks[constraintBase + localConstraint];
        const uint runtimeA =
            endpointBase + block.endpointOffset;
        const uint runtimeB =
            runtimeA + min(block.endpointCount, 2u) - 1u;
        if (block.endpointCount == 0u ||
            block.endpointOffset >=
                2u * dispatch.constraintCapacity ||
            runtimeB >=
                endpointBase +
                    2u * dispatch.constraintCapacity) {
            status.code = MR_STEP_UNSUPPORTED;
            status.firstFailingConstraint = localConstraint;
            statuses[environment] = status;
            return;
        }
        const uint nodeA =
            endpointRuntime[runtimeA].dynamicNode;
        const uint nodeB =
            endpointRuntime[runtimeB].dynamicNode;
        if ((nodeA != MR_INVALID_INDEX &&
             nodeA >= dispatch.dynamicNodeCount) ||
            (nodeB != MR_INVALID_INDEX &&
             nodeB >= dispatch.dynamicNodeCount)) {
            status.code = MR_STEP_UNSUPPORTED;
            status.firstFailingConstraint = localConstraint;
            statuses[environment] = status;
            return;
        }
        if (nodeA == MR_INVALID_INDEX &&
            nodeB == MR_INVALID_INDEX) {
            status.code = MR_STEP_UNSUPPORTED;
            status.firstFailingConstraint =
                localConstraint;
            statuses[environment] = status;
            return;
        }
        if (nodeA != MR_INVALID_INDEX) {
            active[nodeA] = true;
        }
        if (nodeB != MR_INVALID_INDEX) {
            active[nodeB] = true;
        }
        if (nodeA != MR_INVALID_INDEX &&
            nodeB != MR_INVALID_INDEX) {
            unionRoots(parents, nodeA, nodeB);
        }
    }

    uint roots[MR_WORLD_MAX_DYNAMIC_NODES];
    uint rootCount = 0u;
    for (uint node = 0u;
         node < dispatch.dynamicNodeCount;
         ++node) {
        if (!active[node]) {
            continue;
        }
        const uint root = findRoot(parents, node);
        bool found = false;
        for (uint index = 0u; index < rootCount; ++index) {
            found = found || roots[index] == root;
        }
        if (!found) {
            roots[rootCount++] = root;
        }
    }
    status.requiredIslands = rootCount;
    status.islandCount = rootCount;
    if (rootCount > dispatch.islandCapacity) {
        status.code = MR_STEP_ISLAND_CAPACITY_OVERFLOW;
        statuses[environment] = status;
        return;
    }
    for (uint island = 0u; island < rootCount; ++island) {
        MRContactIslandGPU record = {};
        record.environment = environment;
        record.stableRoot = roots[island];
        record.firstConstraint = MR_INVALID_INDEX;
        record.firstNode = MR_INVALID_INDEX;
        record.firstRow = MR_INVALID_INDEX;
        record.generation = 1u;
        islands[islandBase + island] = record;
    }

    const uint nodeReferenceBase =
        environment *
        dispatch.islandNodeReferenceCapacity;
    uint nodeReferenceCursor = 0u;
    uint velocityCursor = 0u;
    for (uint node = 0u;
         node < dispatch.dynamicNodeCount;
         ++node) {
        if (!active[node]) {
            continue;
        }
        const uint root = findRoot(parents, node);
        for (uint island = 0u; island < rootCount; ++island) {
            if (roots[island] == root) {
                device MRContactIslandGPU& record =
                    islands[islandBase + island];
                const MRWorldDynamicNodeGPU dynamic =
                    dynamicNodes[node];
                if ((dynamic.flags &
                     MR_WORLD_DYNAMIC_NODE_VALID) == 0u ||
                    dynamic.stableId != node ||
                    dynamic.velocityCount == 0u ||
                    nodeReferenceCursor >=
                        dispatch.islandNodeReferenceCapacity) {
                    status.code =
                        MR_STEP_ISLAND_CAPACITY_OVERFLOW;
                    status.requiredIslands =
                        max(
                            status.requiredIslands,
                            nodeReferenceCursor + 1u
                        );
                    statuses[environment] = status;
                    return;
                }
                if (record.firstNode == MR_INVALID_INDEX) {
                    record.firstNode = nodeReferenceCursor;
                    record.generalizedVelocityOffset =
                        velocityCursor;
                }
                MRIslandNodeRefGPU reference = {};
                reference.dynamicNode = node;
                reference.localVelocityOffset =
                    record.generalizedVelocityCount;
                reference.velocityCount =
                    dynamic.velocityCount;
                reference.factorIndex = dynamic.factorIndex;
                islandNodeReferences[
                    nodeReferenceBase + nodeReferenceCursor
                ] = reference;
                ++nodeReferenceCursor;
                ++record.dynamicNodeCount;
                record.generalizedVelocityCount +=
                    dynamic.velocityCount;
                velocityCursor += dynamic.velocityCount;
                record.operatorBucket = max(
                    record.operatorBucket,
                    dynamic.operatorBucket
                );
                if (dynamic.kind ==
                    MR_WORLD_DYNAMIC_NODE_ARTICULATION) {
                    ++record.articulationNodeCount;
                } else if (
                    dynamic.kind ==
                    MR_WORLD_DYNAMIC_NODE_FREE_BODY
                ) {
                    ++record.freeBodyNodeCount;
                } else if (
                    dynamic.kind == MR_WORLD_DYNAMIC_NODE_ROD
                ) {
                    ++record.rodNodeCount;
                } else {
                    status.code = MR_STEP_UNSUPPORTED;
                    statuses[environment] = status;
                    return;
                }
                break;
            }
        }
    }

    const uint constraintReferenceBase =
        environment *
        dispatch.islandConstraintReferenceCapacity;
    uint constraintReferenceCursor = 0u;
    for (uint localConstraint = 0u;
         localConstraint < status.requiredConstraints;
         ++localConstraint) {
        device MRContactConstraintGPU& contact =
            contacts[constraintBase + localConstraint];
        const MRConstraintIRBlockGPU block =
            blocks[constraintBase + localConstraint];
        const uint runtimeA =
            endpointBase + block.endpointOffset;
        const uint runtimeB =
            runtimeA + min(block.endpointCount, 2u) - 1u;
        const uint nodeA =
            endpointRuntime[runtimeA].dynamicNode;
        const uint nodeB =
            endpointRuntime[runtimeB].dynamicNode;
        const uint node = nodeA != MR_INVALID_INDEX
            ? nodeA
            : nodeB;
        const uint root = findRoot(parents, node);
        for (uint island = 0u; island < rootCount; ++island) {
            if (roots[island] != root) {
                continue;
            }
            contact.islandIndex = island;
            blocks[constraintBase + localConstraint]
                .islandIndex = island;
            device MRContactIslandGPU& record =
                islands[islandBase + island];
            if (constraintReferenceCursor >=
                dispatch.islandConstraintReferenceCapacity) {
                status.code =
                    MR_STEP_CONSTRAINT_CAPACITY_OVERFLOW;
                status.requiredConstraints =
                    max(
                        status.requiredConstraints,
                        constraintReferenceCursor + 1u
                    );
                statuses[environment] = status;
                return;
            }
            MRIslandConstraintRefGPU reference = {};
            reference.blockIndex = localConstraint;
            reference.rowOffset =
                blocks[
                    constraintBase + localConstraint
                ].rowOffset;
            reference.rowCount =
                blocks[
                    constraintBase + localConstraint
                ].dimension;
            islandConstraintReferences[
                constraintReferenceBase +
                constraintReferenceCursor
            ] = reference;
            record.firstConstraint = min(
                record.firstConstraint,
                localConstraint
            );
            record.firstRow = min(
                record.firstRow,
                reference.rowOffset
            );
            ++record.constraintCount;
            ++constraintReferenceCursor;
            break;
        }
    }
    statuses[environment] = status;
}

// Packs each deterministic island into contiguous 32-contact tiles. The
// contact-index arena is environment-major and preserves canonical constraint
// order even when constraints belonging to different islands were interleaved
// by the manifold compiler.
kernel void mr_world_build_contact_tiles(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRBodyStateGPU* bodies [[buffer(1)]],
    device const MRContactConstraintGPU* contacts [[buffer(2)]],
    device const MRContactIslandGPU* islands [[buffer(3)]],
    device MRIslandWorkGPU* islandWork [[buffer(4)]],
    device MRContactTileGPU* tiles [[buffer(5)]],
    device uint* tileConstraintIndices [[buffer(6)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(7)]],
    device uint* islandWorkFlags [[buffer(8)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    const uint islandBase =
        environment * dispatch.islandStride;
    for (uint islandIndex = 0u;
         islandIndex < dispatch.islandCapacity;
         ++islandIndex) {
        islandWorkFlags[islandBase + islandIndex] = 0u;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    if (status.code != MR_STEP_SUCCESS) {
        return;
    }
    status.requiredSolverTiles = 0u;
    status.requiredSpillRows = 0u;
    status.solverTiles = 0u;
    status.spillRows = 0u;
    if (status.requiredConstraints == 0u) {
        statuses[environment] = status;
        return;
    }

    (void)bodies;
    const uint constraintBase =
        environment * dispatch.constraintStride;
    const uint tileBase =
        environment * dispatch.solverTileCapacity;
    uint tileCursor = 0u;
    uint constraintIndexCursor = 0u;
    uint generalizedConstraintCursor = 0u;
    ulong requiredSpillRows = 0u;
    for (uint islandIndex = 0u;
         islandIndex < status.islandCount;
         ++islandIndex) {
        const MRContactIslandGPU island =
            islands[islandBase + islandIndex];
        const uint islandIndexStart = constraintIndexCursor;
        const bool hasArticulation =
            island.articulationNodeCount != 0u;
        const bool hasRod = island.rodNodeCount != 0u;
        uint generalizedConstraintCount = 0u;
        for (uint localConstraint = 0u;
             localConstraint < status.requiredConstraints;
             ++localConstraint) {
            const MRContactConstraintGPU contact =
                contacts[constraintBase + localConstraint];
            if (contact.islandIndex != islandIndex) {
                continue;
            }
            if ((contact.flags &
                 MR_CONSTRAINT_FLAG_GENERALIZED) != 0u) {
                ++generalizedConstraintCount;
                continue;
            }
            if (constraintIndexCursor <
                dispatch.constraintCapacity) {
                tileConstraintIndices[
                    constraintBase + constraintIndexCursor
                ] = localConstraint;
            }
            ++constraintIndexCursor;
        }
        const uint packedCount =
            constraintIndexCursor - islandIndexStart;
        if (packedCount + generalizedConstraintCount !=
            island.constraintCount) {
            status.code = MR_STEP_NONFINITE_RESULT;
            status.firstFailingConstraint =
                island.firstConstraint;
            statuses[environment] = status;
            return;
        }
        generalizedConstraintCursor +=
            generalizedConstraintCount;
        if (packedCount == 0u) {
            islandWork[islandBase + islandIndex] = {};
            islandWorkFlags[islandBase + islandIndex] = 0u;
            continue;
        }
        const uint tileCount =
            (packedCount + MR_WAVE32_CONTACTS_PER_TILE - 1u) /
            MR_WAVE32_CONTACTS_PER_TILE;
        MRIslandWorkGPU work = {};
        work.environment = environment;
        work.islandIndex = islandIndex;
        work.firstConstraint = island.firstConstraint;
        work.constraintCount = packedCount;
        work.firstTile = tileCursor;
        work.tileCount = tileCount;
        work.dofClass =
            island.operatorBucket <= 8u ? 8u :
            island.operatorBucket <= 16u ? 16u :
            32u;
        work.flags = MR_ISLAND_WORK_VALID |
            (hasArticulation
                 ? MR_ISLAND_WORK_HAS_ARTICULATION
                 : 0u) |
            (hasRod
                 ? MR_ISLAND_WORK_HAS_ROD
                 : 0u) |
            (packedCount > MR_WAVE32_CONTACTS_PER_TILE
                 ? MR_ISLAND_WORK_SPILL
                 : 0u) |
            // The distributed rigid path uses three grid-wide phases.
            // Rod islands instead stream arbitrarily many stable tiles in
            // one owning SIMDgroup so nodal/twist writes retain a unique
            // owner without atomics or a second semantic solver.
            (packedCount > 256u && !hasRod
                 ? MR_ISLAND_WORK_DISTRIBUTED
                 : 0u);
        if (packedCount > 256u && !hasRod) {
            status.queueFlags |=
                MR_ISLAND_WORK_DISTRIBUTED;
        }
        islandWork[islandBase + islandIndex] = work;
        islandWorkFlags[islandBase + islandIndex] = 1u;

        for (uint localTile = 0u;
             localTile < tileCount;
             ++localTile) {
            const uint tileConstraintOffset =
                islandIndexStart +
                localTile * MR_WAVE32_CONTACTS_PER_TILE;
            const uint tileConstraintCount = min(
                MR_WAVE32_CONTACTS_PER_TILE,
                packedCount -
                    localTile * MR_WAVE32_CONTACTS_PER_TILE
            );
            if (tileCursor < dispatch.solverTileCapacity) {
                MRContactTileGPU tile = {};
                tile.environment = environment;
                tile.islandIndex = islandIndex;
                tile.firstConstraint =
                    tileConstraintCount == 0u
                    ? MR_INVALID_INDEX
                    : tileConstraintIndices[
                          constraintBase +
                          tileConstraintOffset
                      ];
                tile.constraintCount = tileConstraintCount;
                tile.nextTile =
                    localTile + 1u < tileCount
                    ? tileCursor + 1u
                    : MR_INVALID_INDEX;
                tile.partialOffset = tileConstraintOffset;
                tile.flags = MR_ISLAND_WORK_VALID;
                tiles[tileBase + tileCursor] = tile;
            }
            ++tileCursor;
        }
        if (packedCount > MR_WAVE32_CONTACTS_PER_TILE) {
            requiredSpillRows +=
                static_cast<ulong>(
                    packedCount -
                    MR_WAVE32_CONTACTS_PER_TILE
                ) * 3u;
        }
    }
    status.requiredSolverTiles = tileCursor;
    status.solverTiles = min(
        tileCursor,
        dispatch.solverTileCapacity
    );
    status.requiredSpillRows = static_cast<uint>(
        min(
            requiredSpillRows,
            static_cast<ulong>(0xffffffffu)
        )
    );
    status.spillRows = min(
        status.requiredSpillRows,
        dispatch.spillRowCapacity
    );
    if (constraintIndexCursor +
            generalizedConstraintCursor !=
            status.requiredConstraints ||
        tileCursor > dispatch.solverTileCapacity ||
        requiredSpillRows > dispatch.spillRowCapacity) {
        status.code = MR_STEP_CONSTRAINT_CAPACITY_OVERFLOW;
        status.firstFailingStableKeyLow =
            status.islandCount == 0u
            ? MR_INVALID_INDEX
            : status.islandCount - 1u;
        status.firstFailingStableKeyHigh = environment;
    }
    statuses[environment] = status;
}

kernel void mr_world_scatter_island_queue(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const uint* flags [[buffer(1)]],
    device const uint* offsets [[buffer(2)]],
    device const MRIslandWorkGPU* denseWork [[buffer(3)]],
    device MRIslandWorkGPU* compactWork [[buffer(4)]],
    device MRWorkQueueHeaderGPU* headers [[buffer(5)]],
    const uint globalIndex [[thread_position_in_grid]]
) {
    const uint total =
        dispatch.environmentCount * dispatch.islandCapacity;
    if (globalIndex == 0u) {
        device MRWorkQueueHeaderGPU& header =
            headers[MR_WORLD_WORK_SOLVER];
        const uint count =
            total == 0u
            ? 0u
            : offsets[total - 1u] +
                (flags[total - 1u] == 1u ? 1u : 0u);
        header = {};
        header.count = min(count, total);
        header.capacity = total;
        header.required = count;
        header.workClass = MR_WORLD_WORK_SOLVER;
        header.overflow = count > total ? 1u : 0u;
        // A following single-SIMDgroup reduction selects a homogeneous
        // 8/16-lane packet width without making any count host-visible.
        header.reserved0 = MR_WAVE32_CONTACTS_PER_TILE;
        header.indirect.threadgroupsX = header.count;
        header.indirect.threadgroupsY = 1u;
        header.indirect.threadgroupsZ = 1u;
        header.indirect.activeCount = header.count;
    }
    if (globalIndex >= total || flags[globalIndex] != 1u) {
        return;
    }
    const uint destination = offsets[globalIndex];
    if (destination < total) {
        compactWork[destination] = denseWork[globalIndex];
    }
}

kernel void mr_world_select_solver_cohort(
    device const MRIslandWorkGPU* compactWork [[buffer(0)]],
    device MRWorkQueueHeaderGPU* headers [[buffer(1)]],
    device MRWaveWorkPacketGPU* packets [[buffer(2)]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    device MRWorkQueueHeaderGPU& header =
        headers[MR_WORLD_WORK_SOLVER];
    const uint count = header.count;
    uint requiredWidth = 8u;
    for (uint workSlot = lane;
         workSlot < count;
         workSlot += MR_WAVE32_CONTACTS_PER_TILE) {
        const MRIslandWorkGPU work = compactWork[workSlot];
        const bool distributed =
            (work.flags & MR_ISLAND_WORK_DISTRIBUTED) != 0u;
        const uint width =
            !distributed &&
            work.dofClass <= 8u &&
            work.constraintCount <= 8u
            ? 8u
            : !distributed &&
              work.dofClass <= 16u &&
              work.constraintCount <= 16u
            ? 16u
            : MR_WAVE32_CONTACTS_PER_TILE;
        requiredWidth = max(requiredWidth, width);
    }
    requiredWidth = simd_max(requiredWidth);
    if (lane == 0u) {
        // The packet solver contains SIMDgroup-wide barriers. Select a compact
        // width only when the entire scan-ordered topology cohort is
        // homogeneous and divides into complete packets; otherwise every
        // island receives a full SIMD32 group. No inactive cohort can then
        // diverge around a barrier.
        const uint cohortWidth =
            count != 0u &&
            requiredWidth <= 8u &&
            (count & 3u) == 0u
            ? 8u
            : count != 0u &&
              requiredWidth <= 16u &&
              (count & 1u) == 0u
            ? 16u
            : MR_WAVE32_CONTACTS_PER_TILE;
        const uint cohortsPerPacket =
            MR_WAVE32_CONTACTS_PER_TILE / cohortWidth;
        const uint packetCount =
            count / cohortsPerPacket;
        header.reserved0 = cohortWidth;
        header.indirect.threadgroupsX = packetCount;
        header.indirect.activeCount = packetCount;
        header.flags &=
            ~(MR_WORLD_QUEUE_COHORT_8 |
              MR_WORLD_QUEUE_COHORT_16);
        header.flags |=
            cohortWidth == 8u
            ? MR_WORLD_QUEUE_COHORT_8
            : cohortWidth == 16u
            ? MR_WORLD_QUEUE_COHORT_16
            : 0u;
    }
    threadgroup_barrier(mem_flags::mem_device);

    const uint cohortWidth = header.reserved0;
    const uint cohortsPerPacket =
        MR_WAVE32_CONTACTS_PER_TILE / cohortWidth;
    const uint packetCount =
        header.indirect.threadgroupsX;
    for (uint packetSlot = lane;
         packetSlot < packetCount;
         packetSlot += MR_WAVE32_CONTACTS_PER_TILE) {
        const uint firstWork =
            packetSlot * cohortsPerPacket;
        MRWaveWorkPacketGPU packet = {};
        packet.islandSlots = uint4(MR_INVALID_INDEX);
        packet.stableKeyLow = uint4(MR_INVALID_INDEX);
        packet.stableKeyHigh = uint4(MR_INVALID_INDEX);
        for (uint cohort = 0u;
             cohort < cohortsPerPacket;
             ++cohort) {
            const uint sourceSlot = firstWork + cohort;
            const MRIslandWorkGPU work =
                compactWork[sourceSlot];
            packet.islandSlots[cohort] = sourceSlot;
            packet.stableKeyLow[cohort] =
                work.islandIndex;
            packet.stableKeyHigh[cohort] =
                work.environment;
        }
        packet.metadata = uint4(
            cohortWidth,
            cohortsPerPacket,
            0u,
            0u
        );
        packets[packetSlot] = packet;
    }
}

kernel void mr_world_flag_distributed_islands(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRIslandWorkGPU* denseWork [[buffer(1)]],
    device uint* flags [[buffer(2)]],
    const uint globalIndex [[thread_position_in_grid]]
) {
    const uint total =
        dispatch.environmentCount * dispatch.islandCapacity;
    if (globalIndex >= total) {
        return;
    }
    const MRIslandWorkGPU work = denseWork[globalIndex];
    flags[globalIndex] =
        (work.flags &
         (
             MR_ISLAND_WORK_VALID |
             MR_ISLAND_WORK_DISTRIBUTED
         )) ==
            (
                MR_ISLAND_WORK_VALID |
                MR_ISLAND_WORK_DISTRIBUTED
            )
        ? 1u
        : 0u;
}

kernel void mr_world_scatter_distributed_island_queue(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const uint* flags [[buffer(1)]],
    device const uint* offsets [[buffer(2)]],
    device const MRIslandWorkGPU* denseWork [[buffer(3)]],
    device MRIslandWorkGPU* compactWork [[buffer(4)]],
    device MRWorkQueueHeaderGPU* headers [[buffer(5)]],
    const uint globalIndex [[thread_position_in_grid]]
) {
    const uint total =
        dispatch.environmentCount * dispatch.islandCapacity;
    if (globalIndex == 0u) {
        device MRWorkQueueHeaderGPU& header =
            headers[MR_WORLD_WORK_SOLVER_DISTRIBUTED];
        const uint count =
            total == 0u
            ? 0u
            : offsets[total - 1u] +
                (flags[total - 1u] == 1u ? 1u : 0u);
        header = {};
        header.count = min(count, total);
        header.capacity = total;
        header.required = count;
        header.workClass =
            MR_WORLD_WORK_SOLVER_DISTRIBUTED;
        header.overflow = count > total ? 1u : 0u;
        header.indirect.threadgroupsX = header.count;
        header.indirect.threadgroupsY = 1u;
        header.indirect.threadgroupsZ = 1u;
        header.indirect.activeCount = header.count;
        header.reserved0 = total;
    }
    if (globalIndex >= total || flags[globalIndex] != 1u) {
        return;
    }
    const uint destination = offsets[globalIndex];
    if (destination < total) {
        compactWork[total + destination] =
            denseWork[globalIndex];
    }
}

kernel void mr_world_flag_distributed_tiles(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(1)]],
    device const MRIslandWorkGPU* denseWork [[buffer(2)]],
    device const MRContactTileGPU* denseTiles [[buffer(3)]],
    device uint* flags [[buffer(4)]],
    const uint globalIndex [[thread_position_in_grid]]
) {
    const uint total =
        dispatch.environmentCount *
        dispatch.solverTileCapacity;
    if (globalIndex >= total) {
        return;
    }
    const uint environment =
        globalIndex / dispatch.solverTileCapacity;
    const uint localTile =
        globalIndex -
        environment * dispatch.solverTileCapacity;
    const MRMetalWorldContactStatusGPU status =
        statuses[environment];
    if (status.code != MR_STEP_SUCCESS ||
        localTile >= status.solverTiles) {
        flags[globalIndex] = 0u;
        return;
    }
    const MRContactTileGPU tile = denseTiles[globalIndex];
    if ((tile.flags & MR_ISLAND_WORK_VALID) == 0u ||
        tile.environment != environment ||
        tile.islandIndex >= status.islandCount) {
        flags[globalIndex] = 0u;
        return;
    }
    const MRIslandWorkGPU work =
        denseWork[
            environment * dispatch.islandCapacity +
            tile.islandIndex
        ];
    flags[globalIndex] =
        (work.flags &
         MR_ISLAND_WORK_DISTRIBUTED) != 0u
        ? 1u
        : 0u;
}

kernel void mr_world_scatter_distributed_tile_queue(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const uint* flags [[buffer(1)]],
    device const uint* offsets [[buffer(2)]],
    device const MRContactTileGPU* denseTiles [[buffer(3)]],
    device MRContactTileGPU* compactTiles [[buffer(4)]],
    device MRWorkQueueHeaderGPU* headers [[buffer(5)]],
    const uint globalIndex [[thread_position_in_grid]]
) {
    const uint total =
        dispatch.environmentCount *
        dispatch.solverTileCapacity;
    if (globalIndex == 0u) {
        device MRWorkQueueHeaderGPU& header =
            headers[MR_WORLD_WORK_SOLVER_SPILL];
        const uint count =
            total == 0u
            ? 0u
            : offsets[total - 1u] +
                (flags[total - 1u] == 1u ? 1u : 0u);
        header = {};
        header.count = min(count, total);
        header.capacity = total;
        header.required = count;
        header.workClass = MR_WORLD_WORK_SOLVER_SPILL;
        header.overflow = count > total ? 1u : 0u;
        header.indirect.threadgroupsX = header.count;
        header.indirect.threadgroupsY = 1u;
        header.indirect.threadgroupsZ = 1u;
        header.indirect.activeCount = header.count;
        header.reserved0 = total;
    }
    if (globalIndex >= total || flags[globalIndex] != 1u) {
        return;
    }
    const uint destination = offsets[globalIndex];
    if (destination < total) {
        compactTiles[total + destination] =
            denseTiles[globalIndex];
    }
}

// Distributed islands use one SIMDgroup per compacted 32-contact tile for
// the expensive factor solves and 3x3 block construction. Per-contact output
// slots are unique; a later single-group segmented reducer applies them in
// canonical tile/contact order, so no atomic velocity accumulation is used.
kernel void mr_world_wave32_distributed_prepare(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const float* factors [[buffer(1)]],
    device const float* pointJacobians [[buffer(2)]],
    device const MRBodyStateGPU* candidateBodies [[buffer(3)]],
    device MRContactConstraintGPU* contacts [[buffer(4)]],
    device const MREvaluatedConstraintIRRowGPU* evaluatedRows [[buffer(5)]],
    device const MREvaluatedConstraintIRConeGPU* evaluatedCones [[buffer(6)]],
    device float* responseColumns [[buffer(7)]],
    device float4* impulseDeltas [[buffer(8)]],
    device MRWave32PreconditionerGPU* preconditioners [[buffer(9)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(10)]],
    device const MRIslandWorkGPU* denseIslandWork [[buffer(11)]],
    device const MRContactTileGPU* compactTiles [[buffer(12)]],
    device const uint* tileConstraintIndices [[buffer(13)]],
    device const MRWorkQueueHeaderGPU* workHeaders [[buffer(14)]],
    const uint3 workGroup [[threadgroup_position_in_grid]],
    const uint3 groupCount [[threadgroups_per_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    const MRWorkQueueHeaderGPU header =
        workHeaders[MR_WORLD_WORK_SOLVER_SPILL];
    const uint stride = max(groupCount.x, 1u);
    for (uint queueSlot = workGroup.x;
         queueSlot < header.count;
         queueSlot += stride) {
        const MRContactTileGPU tile =
            compactTiles[header.reserved0 + queueSlot];
        const uint environment = tile.environment;
        if (environment >= dispatch.environmentCount ||
            lane >= tile.constraintCount) {
            continue;
        }
        const MRMetalWorldContactStatusGPU environmentStatus =
            statuses[environment];
        if (environmentStatus.code != MR_STEP_SUCCESS ||
            tile.islandIndex >= environmentStatus.islandCount) {
            continue;
        }
        const MRIslandWorkGPU work =
            denseIslandWork[
                environment * dispatch.islandCapacity +
                tile.islandIndex
            ];
        if ((work.flags &
             (
                 MR_ISLAND_WORK_VALID |
                 MR_ISLAND_WORK_DISTRIBUTED
             )) !=
            (
                MR_ISLAND_WORK_VALID |
                MR_ISLAND_WORK_DISTRIBUTED
            )) {
            continue;
        }

        const uint constraintBase =
            environment * dispatch.constraintStride;
        const uint rowBase =
            environment * dispatch.rowStride;
        const uint bodyBase =
            environment * dispatch.bodyStateStride;
        const uint factorBase =
            environment * dispatch.factorStride;
        const uint pointJacobianBase =
            environment *
            (dispatch.pointQueryStride * 3u * dispatch.nv);
        const uint responseBase =
            environment *
            (dispatch.constraintStride * 3u * dispatch.nv);
        const uint localConstraint =
            tileConstraintIndices[
                constraintBase + tile.partialOffset + lane
            ];
        device MRContactConstraintGPU& contact =
            contacts[constraintBase + localConstraint];
        device const MRBodyStateGPU* bodies =
            candidateBodies + bodyBase;
        const bool articulatedA =
            bodies[contact.bodyA].flagsAndIndices[1] !=
                MR_INVALID_INDEX;
        const bool articulatedB =
            bodies[contact.bodyB].flagsAndIndices[1] !=
                MR_INVALID_INDEX;

        uint failure = MR_STEP_SUCCESS;
        float rightHandSide[MR_ARTICULATED_ABA_MAX_DOFS];
        float intermediate[MR_ARTICULATED_ABA_MAX_DOFS];
        float solution[MR_ARTICULATED_ABA_MAX_DOFS];
        if ((dispatch.flags &
             MR_METAL_WORLD_CONTACT_STREAMED_RESPONSES) == 0u) {
            for (uint axis = 0u; axis < 3u; ++axis) {
                const float3 direction =
                    evaluatedRows[
                        rowBase + 3u * localConstraint + axis
                    ].direction.xyz;
                for (uint dof = 0u; dof < dispatch.nv; ++dof) {
                    rightHandSide[dof] = dot(
                        direction,
                        combinedJacobianColumn(
                            pointJacobians,
                            pointJacobianBase,
                            localConstraint,
                            dof,
                            dispatch.nv,
                            articulatedA,
                            articulatedB
                        )
                    );
                    intermediate[dof] = 0.0f;
                    solution[dof] = 0.0f;
                }
                if (!solveCholesky(
                        factors,
                        factorBase,
                        dispatch.nv,
                        rightHandSide,
                        intermediate,
                        solution
                    )) {
                    failure = MR_STEP_FACTORIZATION_FAILED;
                    break;
                }
                for (uint dof = 0u;
                     dof < dispatch.nv;
                     ++dof) {
                    responseColumns[
                        responseBase +
                        (localConstraint * 3u + axis) *
                            dispatch.nv +
                        dof
                    ] = solution[dof];
                }
            }
        }

        MRWave32PreconditionerGPU preconditioner = {};
        if (failure == MR_STEP_SUCCESS) {
            device const MREvaluatedConstraintIRRowGPU*
                localRows =
                    evaluatedRows +
                    rowBase + 3u * localConstraint;
            const float3 directions[3] = {
                localRows[0].direction.xyz,
                localRows[1].direction.xyz,
                localRows[2].direction.xyz,
            };
            float effective[3][3];
            for (uint row = 0u; row < 3u; ++row) {
                for (uint column = 0u;
                     column < 3u;
                     ++column) {
                    float value = 0.0f;
                    for (uint dof = 0u;
                         dof < dispatch.nv;
                         ++dof) {
                        const float rowJacobian = dot(
                            directions[row],
                            combinedJacobianColumn(
                                pointJacobians,
                                pointJacobianBase,
                                localConstraint,
                                dof,
                                dispatch.nv,
                                articulatedA,
                                articulatedB
                            )
                        );
                        value += rowJacobian *
                            responseColumns[
                                responseBase +
                                (
                                    localConstraint * 3u +
                                    column
                                ) * dispatch.nv +
                                dof
                            ];
                    }
                    if (dynamicSceneEndpoint(
                            bodies[contact.bodyA],
                            dispatch.articulationIndex
                        )) {
                        value += dot(
                            directions[row],
                            -scenePointResponse(
                                bodies[contact.bodyA],
                                contact.pointAndSeparation.xyz,
                                -directions[column]
                            )
                        );
                    }
                    if (dynamicSceneEndpoint(
                            bodies[contact.bodyB],
                            dispatch.articulationIndex
                        )) {
                        value += dot(
                            directions[row],
                            scenePointResponse(
                                bodies[contact.bodyB],
                                contact.pointAndSeparation.xyz,
                                directions[column]
                            )
                        );
                    }
                    if (row == column) {
                        value +=
                            localRows[row].regularization;
                    }
                    effective[row][column] = value;
                }
            }
            float inverse[3][3];
            if (!invert3x3(effective, inverse)) {
                failure = MR_STEP_FACTORIZATION_FAILED;
            } else {
                preconditioner.row0.xyz = float3(
                    inverse[0][0],
                    inverse[0][1],
                    inverse[0][2]
                );
                preconditioner.row1.xyz = float3(
                    inverse[1][0],
                    inverse[1][1],
                    inverse[1][2]
                );
                preconditioner.row2.xyz = float3(
                    inverse[2][0],
                    inverse[2][1],
                    inverse[2][2]
                );
            }
        }
        preconditioner.row0.w =
            static_cast<float>(failure);
        preconditioner.row1.w =
            static_cast<float>(localConstraint);
        preconditioners[
            constraintBase + localConstraint
        ] = preconditioner;
        if (failure == MR_STEP_SUCCESS) {
            const float3 warm = projectFrictionCone(
                contact.impulses.xyz,
                evaluatedCones[
                    constraintBase + localConstraint
                ]
            );
            contact.impulses.xyz = warm;
            impulseDeltas[
                constraintBase + localConstraint
            ] = float4(warm, 0.0f);
        } else {
            impulseDeltas[
                constraintBase + localConstraint
            ] = float4(
                0.0f,
                0.0f,
                0.0f,
                static_cast<float>(failure)
            );
        }
    }
}

// All contacts in a distributed island evaluate their block-Jacobi cone
// update against the same accepted island velocity. Each compact tile is
// independent until the canonical segmented reducer applies these deltas.
kernel void mr_world_wave32_distributed_delta(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const float* pointJacobians [[buffer(1)]],
    device const float* candidateV [[buffer(2)]],
    device const MRBodyStateGPU* candidateBodies [[buffer(3)]],
    device MRContactConstraintGPU* contacts [[buffer(4)]],
    device const MREvaluatedConstraintIRRowGPU* evaluatedRows [[buffer(5)]],
    device const MREvaluatedConstraintIRConeGPU* evaluatedCones [[buffer(6)]],
    device const MRWave32PreconditionerGPU* preconditioners [[buffer(7)]],
    device float4* impulseDeltas [[buffer(8)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(9)]],
    device const MRContactTileGPU* compactTiles [[buffer(10)]],
    device const uint* tileConstraintIndices [[buffer(11)]],
    device const MRWorkQueueHeaderGPU* workHeaders [[buffer(12)]],
    const uint3 workGroup [[threadgroup_position_in_grid]],
    const uint3 groupCount [[threadgroups_per_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    const MRWorkQueueHeaderGPU header =
        workHeaders[MR_WORLD_WORK_SOLVER_SPILL];
    const uint stride = max(groupCount.x, 1u);
    for (uint queueSlot = workGroup.x;
         queueSlot < header.count;
         queueSlot += stride) {
        const MRContactTileGPU tile =
            compactTiles[header.reserved0 + queueSlot];
        const uint environment = tile.environment;
        if (environment >= dispatch.environmentCount ||
            lane >= tile.constraintCount ||
            statuses[environment].code != MR_STEP_SUCCESS) {
            continue;
        }
        const uint constraintBase =
            environment * dispatch.constraintStride;
        const uint rowBase =
            environment * dispatch.rowStride;
        const uint bodyBase =
            environment * dispatch.bodyStateStride;
        const uint pointJacobianBase =
            environment *
            (dispatch.pointQueryStride * 3u * dispatch.nv);
        const uint localConstraint =
            tileConstraintIndices[
                constraintBase + tile.partialOffset + lane
            ];
        const MRWave32PreconditionerGPU preconditioner =
            preconditioners[
                constraintBase + localConstraint
            ];
        if (static_cast<uint>(
                preconditioner.row0.w
            ) != MR_STEP_SUCCESS) {
            continue;
        }
        device MRContactConstraintGPU& contact =
            contacts[constraintBase + localConstraint];
        device const MREvaluatedConstraintIRRowGPU* localRows =
            evaluatedRows + rowBase + 3u * localConstraint;
        const float3 relative = relativePointVelocity(
            localConstraint,
            contact,
            candidateBodies + bodyBase,
            dispatch.articulationIndex,
            pointJacobians,
            pointJacobianBase,
            dispatch.nv,
            candidateV + environment * dispatch.nv
        );
        const float3 previous = contact.impulses.xyz;
        const float3 rhs = float3(
            localRows[0].targetVelocity -
                dot(localRows[0].direction.xyz, relative) -
                localRows[0].regularization * previous.x,
            localRows[1].targetVelocity -
                dot(localRows[1].direction.xyz, relative) -
                localRows[1].regularization * previous.y,
            localRows[2].targetVelocity -
                dot(localRows[2].direction.xyz, relative) -
                localRows[2].regularization * previous.z
        );
        float3 candidate = previous + float3(
            dot(preconditioner.row0.xyz, rhs),
            dot(preconditioner.row1.xyz, rhs),
            dot(preconditioner.row2.xyz, rhs)
        );
        candidate = projectFrictionCone(
            candidate,
            evaluatedCones[
                constraintBase + localConstraint
            ]
        );
        const float3 delta = candidate - previous;
        if (!finite3(candidate) || !finite3(delta)) {
            impulseDeltas[
                constraintBase + localConstraint
            ] = float4(
                0.0f,
                0.0f,
                0.0f,
                static_cast<float>(MR_STEP_NONFINITE_RESULT)
            );
            continue;
        }
        contact.impulses.xyz = candidate;
        impulseDeltas[
            constraintBase + localConstraint
        ] = float4(delta, 0.0f);
    }
}

// One SIMDgroup deterministically reduces per-contact tile output across a
// distributed island. Lane ownership of articulation DoFs and scene bodies
// prevents write conflicts, and stable tile order makes the reduction
// bitwise replayable.
kernel void mr_world_wave32_distributed_reduce(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const float* pointJacobians [[buffer(1)]],
    device float* candidateV [[buffer(2)]],
    device MRBodyStateGPU* candidateBodies [[buffer(3)]],
    device MRContactConstraintGPU* contacts [[buffer(4)]],
    device const MRContactPointMetaGPU* contactMetadata [[buffer(5)]],
    device const MREvaluatedConstraintIRRowGPU* evaluatedRows [[buffer(6)]],
    device const MREvaluatedConstraintIRConeGPU* evaluatedCones [[buffer(7)]],
    device const float* responseColumns [[buffer(8)]],
    device const float4* impulseDeltas [[buffer(9)]],
    device const MRWave32PreconditionerGPU* preconditioners [[buffer(10)]],
    device MRManifoldPointGPU* candidateManifoldPoints [[buffer(11)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(12)]],
    device const MRIslandWorkGPU* compactIslandWork [[buffer(13)]],
    device const MRContactTileGPU* denseTiles [[buffer(14)]],
    device const uint* tileConstraintIndices [[buffer(15)]],
    device MRWave32IslandStatusGPU* waveStatuses [[buffer(16)]],
    device const MRWorkQueueHeaderGPU* workHeaders [[buffer(17)]],
    constant MRMetalWorldPassGPU& pass [[buffer(18)]],
    const uint3 workGroup [[threadgroup_position_in_grid]],
    const uint3 groupCount [[threadgroups_per_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    const MRWorkQueueHeaderGPU header =
        workHeaders[MR_WORLD_WORK_SOLVER_DISTRIBUTED];
    const uint stride = max(groupCount.x, 1u);
    for (uint queueSlot = workGroup.x;
         queueSlot < header.count;
         queueSlot += stride) {
        const MRIslandWorkGPU work =
            compactIslandWork[header.reserved0 + queueSlot];
        const uint environment = work.environment;
        const uint islandIndex = work.islandIndex;
        if (environment >= dispatch.environmentCount ||
            statuses[environment].code != MR_STEP_SUCCESS ||
            (work.flags &
             MR_ISLAND_WORK_DISTRIBUTED) == 0u) {
            continue;
        }
        const uint constraintBase =
            environment * dispatch.constraintStride;
        const uint rowBase =
            environment * dispatch.rowStride;
        const uint bodyBase =
            environment * dispatch.bodyStateStride;
        const uint pointJacobianBase =
            environment *
            (dispatch.pointQueryStride * 3u * dispatch.nv);
        const uint responseBase =
            environment *
            (dispatch.constraintStride * 3u * dispatch.nv);
        const uint manifoldPointBase =
            environment * dispatch.manifoldStride *
            MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
        const uint tileBase =
            environment * dispatch.solverTileCapacity;
        device float* articulationVelocity =
            candidateV + environment * dispatch.nv;
        device MRBodyStateGPU* bodies =
            candidateBodies + bodyBase;

        uint laneFailure = MR_STEP_SUCCESS;
        uint laneFailureConstraint = MR_INVALID_INDEX;
        float laneMaximumDelta = 0.0f;
        for (uint localTile = 0u;
             localTile < work.tileCount;
             ++localTile) {
            const MRContactTileGPU tile =
                denseTiles[
                    tileBase + work.firstTile + localTile
                ];
            if (lane >= tile.constraintCount) {
                continue;
            }
            const uint localConstraint =
                tileConstraintIndices[
                    constraintBase +
                    tile.partialOffset + lane
                ];
            const uint prepareFailure = static_cast<uint>(
                preconditioners[
                    constraintBase + localConstraint
                ].row0.w
            );
            const uint deltaFailure = static_cast<uint>(
                impulseDeltas[
                    constraintBase + localConstraint
                ].w
            );
            const uint failure = max(
                prepareFailure,
                deltaFailure
            );
            if (failure != MR_STEP_SUCCESS) {
                laneFailure = max(laneFailure, failure);
                laneFailureConstraint = min(
                    laneFailureConstraint,
                    localConstraint
                );
            }
            const float3 delta =
                impulseDeltas[
                    constraintBase + localConstraint
                ].xyz;
            laneMaximumDelta = max(
                laneMaximumDelta,
                max(
                    abs(delta.x),
                    max(abs(delta.y), abs(delta.z))
                )
            );
        }
        const uint maximumFailure = simd_max(laneFailure);
        const uint firstFailureConstraint =
            simd_min(laneFailureConstraint);
        if (maximumFailure != MR_STEP_SUCCESS) {
            if (lane == 0u) {
                MRWave32IslandStatusGPU failed = {};
                failed.code = maximumFailure;
                failed.environment = environment;
                failed.islandIndex = islandIndex;
                failed.residuals.w = static_cast<float>(
                    firstFailureConstraint
                );
                waveStatuses[
                    environment * dispatch.islandStride +
                    islandIndex
                ] = failed;
            }
            continue;
        }

        if ((work.flags &
             MR_ISLAND_WORK_HAS_ARTICULATION) != 0u) {
            for (uint dof = lane;
                 dof < dispatch.nv;
                 dof += MR_WAVE32_CONTACTS_PER_TILE) {
                float velocityDelta = 0.0f;
                for (uint localTile = 0u;
                     localTile < work.tileCount;
                     ++localTile) {
                    const MRContactTileGPU tile =
                        denseTiles[
                            tileBase +
                            work.firstTile + localTile
                        ];
                    for (uint slot = 0u;
                         slot < tile.constraintCount;
                         ++slot) {
                        const uint localConstraint =
                            tileConstraintIndices[
                                constraintBase +
                                tile.partialOffset + slot
                            ];
                        const float3 delta =
                            impulseDeltas[
                                constraintBase +
                                localConstraint
                            ].xyz;
                        velocityDelta +=
                            responseColumns[
                                responseBase +
                                (localConstraint * 3u + 0u) *
                                    dispatch.nv +
                                dof
                            ] * delta.x +
                            responseColumns[
                                responseBase +
                                (localConstraint * 3u + 1u) *
                                    dispatch.nv +
                                dof
                            ] * delta.y +
                            responseColumns[
                                responseBase +
                                (localConstraint * 3u + 2u) *
                                    dispatch.nv +
                                dof
                            ] * delta.z;
                    }
                }
                articulationVelocity[dof] += velocityDelta;
            }
        }
        for (uint bodyIndex = lane;
             bodyIndex < dispatch.bodyCount;
             bodyIndex += MR_WAVE32_CONTACTS_PER_TILE) {
            device MRBodyStateGPU& body = bodies[bodyIndex];
            if (!dynamicSceneEndpoint(
                    body,
                    dispatch.articulationIndex
                )) {
                continue;
            }
            float3 linearImpulse = float3(0.0f);
            float3 angularImpulse = float3(0.0f);
            for (uint localTile = 0u;
                 localTile < work.tileCount;
                 ++localTile) {
                const MRContactTileGPU tile =
                    denseTiles[
                        tileBase + work.firstTile + localTile
                    ];
                for (uint slot = 0u;
                     slot < tile.constraintCount;
                     ++slot) {
                    const uint localConstraint =
                        tileConstraintIndices[
                            constraintBase +
                            tile.partialOffset + slot
                        ];
                    const MRContactConstraintGPU contact =
                        contacts[
                            constraintBase + localConstraint
                        ];
                    const float3 delta =
                        impulseDeltas[
                            constraintBase + localConstraint
                        ].xyz;
                    device const MREvaluatedConstraintIRRowGPU*
                        localRows =
                            evaluatedRows +
                            rowBase + 3u * localConstraint;
                    const float3 impulse =
                        localRows[0].direction.xyz * delta.x +
                        localRows[1].direction.xyz * delta.y +
                        localRows[2].direction.xyz * delta.z;
                    if (contact.bodyA == bodyIndex) {
                        linearImpulse -= impulse;
                        angularImpulse += cross(
                            contact.pointAndSeparation.xyz -
                                body.position.xyz,
                            -impulse
                        );
                    }
                    if (contact.bodyB == bodyIndex) {
                        linearImpulse += impulse;
                        angularImpulse += cross(
                            contact.pointAndSeparation.xyz -
                                body.position.xyz,
                            impulse
                        );
                    }
                }
            }
            body.linearVelocityAndInverseMass.xyz +=
                body.linearVelocityAndInverseMass.w *
                linearImpulse;
            body.angularVelocity.xyz += multiply(
                stateInverseInertia(body),
                angularImpulse
            );
        }
        threadgroup_barrier(mem_flags::mem_device);

        // reserved1==2 marks the last distributed sweep for this
        // microstep. Only then publish manifold impulses and residuals.
        if (pass.reserved1 != 2u) {
            continue;
        }
        float laneNormalResidual = 0.0f;
        float laneConeViolation = 0.0f;
        for (uint localTile = 0u;
             localTile < work.tileCount;
             ++localTile) {
            const MRContactTileGPU tile =
                denseTiles[
                    tileBase + work.firstTile + localTile
                ];
            if (lane >= tile.constraintCount) {
                continue;
            }
            const uint localConstraint =
                tileConstraintIndices[
                    constraintBase +
                    tile.partialOffset + lane
                ];
            device MRContactConstraintGPU& contact =
                contacts[constraintBase + localConstraint];
            device const MREvaluatedConstraintIRRowGPU*
                localRows =
                    evaluatedRows +
                    rowBase + 3u * localConstraint;
            const float3 relative = relativePointVelocity(
                localConstraint,
                contact,
                bodies,
                dispatch.articulationIndex,
                pointJacobians,
                pointJacobianBase,
                dispatch.nv,
                articulationVelocity
            );
            const float normalEquation =
                dot(
                    localRows[0].direction.xyz,
                    relative
                ) -
                localRows[0].targetVelocity +
                localRows[0].regularization *
                    contact.impulses.x;
            laneNormalResidual = max(
                laneNormalResidual,
                contact.impulses.x > kConeEpsilon
                ? abs(normalEquation)
                : max(-normalEquation, 0.0f)
            );
            const MREvaluatedConstraintIRConeGPU cone =
                evaluatedCones[
                    constraintBase + localConstraint
                ];
            const float limitU =
                cone.effectiveFrictionU *
                contact.impulses.x;
            const float limitV =
                cone.effectiveFrictionV *
                contact.impulses.x;
            float coneViolation = 0.0f;
            if (limitU > 0.0f && limitV > 0.0f) {
                coneViolation = max(
                    sqrt(
                        (
                            contact.impulses.y *
                            contact.impulses.y
                        ) / (limitU * limitU) +
                        (
                            contact.impulses.z *
                            contact.impulses.z
                        ) / (limitV * limitV)
                    ) - 1.0f,
                    0.0f
                );
            } else {
                coneViolation =
                    length(contact.impulses.yz);
            }
            laneConeViolation = max(
                laneConeViolation,
                coneViolation
            );
            const MRContactPointMetaGPU metadata =
                contactMetadata[
                    constraintBase + localConstraint
                ];
            if (metadata.manifoldIndex <
                    dispatch.manifoldCapacity &&
                metadata.pointIndex <
                    MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY) {
                candidateManifoldPoints[
                    manifoldPointBase +
                    metadata.manifoldIndex *
                        MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY +
                    metadata.pointIndex
                ].impulses = contact.impulses;
            }
        }
        const float maximumImpulseDelta =
            simd_max(laneMaximumDelta);
        const float maximumNormalResidual =
            simd_max(laneNormalResidual);
        const float maximumConeViolation =
            simd_max(laneConeViolation);
        if (lane == 0u) {
            MRWave32IslandStatusGPU result = {};
            result.code = MR_STEP_SUCCESS;
            result.environment = environment;
            result.islandIndex = islandIndex;
            result.iterations = pass.reserved0;
            result.residuals = float4(
                maximumImpulseDelta,
                maximumNormalResidual,
                maximumConeViolation,
                0.0f
            );
            waveStatuses[
                environment * dispatch.islandStride +
                islandIndex
            ] = result;
        }
    }
}

float waveCohortMaximum(
    float value,
    const uint cohortWidth
) {
    for (uint mask = cohortWidth >> 1u;
         mask != 0u;
         mask >>= 1u) {
        value = max(value, simd_shuffle_xor(value, mask));
    }
    return value;
}

uint waveCohortMaximum(
    uint value,
    const uint cohortWidth
) {
    for (uint mask = cohortWidth >> 1u;
         mask != 0u;
         mask >>= 1u) {
        value = max(value, simd_shuffle_xor(value, mask));
    }
    return value;
}

uint waveCohortMinimum(
    uint value,
    const uint cohortWidth
) {
    for (uint mask = cohortWidth >> 1u;
         mask != 0u;
         mask >>= 1u) {
        value = min(value, simd_shuffle_xor(value, mask));
    }
    return value;
}

inline bool waveIslandContainsBody(
    thread const MRIslandWorkGPU& work,
    const uint bodyIndex,
    device const MRContactConstraintGPU* contacts,
    const uint constraintBase,
    device const MRContactTileGPU* tiles,
    const uint tileBase,
    device const uint* tileConstraintIndices,
    device const MRRodToolWitnessGPU* rodWitnesses,
    device const uint* constraintWitnessIndices
) {
    for (uint localTile = 0u;
         localTile < work.tileCount;
         ++localTile) {
        const MRContactTileGPU tile =
            tiles[tileBase + work.firstTile + localTile];
        for (uint slot = 0u;
             slot < tile.constraintCount;
             ++slot) {
            const uint localConstraint =
                tileConstraintIndices[
                    constraintBase + tile.partialOffset + slot
                ];
            device const MRContactConstraintGPU& contact =
                contacts[constraintBase + localConstraint];
            if (typedRodConstraint(contact)) {
                const uint witnessIndex =
                    constraintWitnessIndices[
                        constraintBase + localConstraint
                    ];
                if (witnessIndex != MR_INVALID_INDEX &&
                    rodWitnesses[
                        witnessIndex
                    ].featuresAndFlags.z == bodyIndex) {
                    return true;
                }
                continue;
            }
            if (contact.bodyA == bodyIndex ||
                contact.bodyB == bodyIndex) {
                return true;
            }
        }
    }
    return false;
}

inline float typedNormalCrossContactResponse(
    const uint targetConstraint,
    const uint sourceConstraint,
    const uint sourceAxis,
    device const MRContactConstraintGPU& target,
    device const MRContactConstraintGPU& source,
    const float3 targetNormal,
    const float3 sourceNormal,
    device const MRBodyStateGPU* bodies,
    device const float* pointJacobians,
    const uint pointJacobianBase,
    device const float* responseColumns,
    const uint responseBase,
    const uint nv,
    const uint articulationIndex,
    device const MRRodNodeStateGPU* rodNodes,
    device const float* inverseRodMasses,
    device const float* inverseRodTwistInertias,
    device const MRRodColliderGPU* rodColliders,
    device const MRRodToolWitnessGPU* rodWitnesses,
    device const uint* constraintWitnessIndices,
    const uint constraintBase
) {
    float response = 0.0f;
    for (uint dof = 0u; dof < nv; ++dof) {
        response += dot(
            targetNormal,
            typedArticulationJacobianColumn(
                targetConstraint,
                target,
                bodies,
                pointJacobians,
                pointJacobianBase,
                dof,
                nv,
                rodWitnesses,
                constraintWitnessIndices,
                constraintBase
            )
        ) * responseColumns[
            responseBase +
            (sourceConstraint * 3u + sourceAxis) * nv +
            dof
        ];
    }

    const bool targetRod = typedRodConstraint(target);
    const bool sourceRod = typedRodConstraint(source);
    MRRodToolWitnessGPU targetWitness = {};
    MRRodToolWitnessGPU sourceWitness = {};
    if (targetRod) {
        targetWitness = rodWitnesses[
            constraintWitnessIndices[
                constraintBase + targetConstraint
            ]
        ];
    }
    if (sourceRod) {
        sourceWitness = rodWitnesses[
            constraintWitnessIndices[
                constraintBase + sourceConstraint
            ]
        ];
    }

    const uint targetEndpointCount = targetRod ? 1u : 2u;
    const uint sourceEndpointCount = sourceRod ? 1u : 2u;
    for (uint targetEndpoint = 0u;
         targetEndpoint < targetEndpointCount;
         ++targetEndpoint) {
        const uint targetBody =
            targetRod
            ? targetWitness.featuresAndFlags.z
            : targetEndpoint == 0u
            ? target.bodyA
            : target.bodyB;
        const float targetSign =
            targetRod
            ? 1.0f
            : targetEndpoint == 0u
            ? -1.0f
            : 1.0f;
        const float3 targetPoint =
            targetRod
            ? targetWitness.toolPointAndSeparation.xyz
            : target.pointAndSeparation.xyz;
        device const MRBodyStateGPU& body =
            bodies[targetBody];
        if (!dynamicSceneEndpoint(body, articulationIndex)) {
            continue;
        }
        for (uint sourceEndpoint = 0u;
             sourceEndpoint < sourceEndpointCount;
             ++sourceEndpoint) {
            const uint sourceBody =
                sourceRod
                ? sourceWitness.featuresAndFlags.z
                : sourceEndpoint == 0u
                ? source.bodyA
                : source.bodyB;
            if (sourceBody != targetBody) {
                continue;
            }
            const float sourceSign =
                sourceRod
                ? 1.0f
                : sourceEndpoint == 0u
                ? -1.0f
                : 1.0f;
            const float3 sourcePoint =
                sourceRod
                ? sourceWitness.toolPointAndSeparation.xyz
                : source.pointAndSeparation.xyz;
            response += targetSign * dot(
                targetNormal,
                sceneCrossPointResponse(
                    body,
                    targetPoint,
                    sourcePoint,
                    sourceSign * sourceNormal
                )
            );
        }
    }

    if (targetRod && sourceRod) {
        const MRRodColliderGPU targetCollider =
            rodColliders[targetWitness.identity.z];
        const MRRodColliderGPU sourceCollider =
            rodColliders[sourceWitness.identity.z];
        if (targetCollider.rodIndex ==
            sourceCollider.rodIndex) {
            response += rodCrossContactResponse(
                targetCollider,
                targetWitness,
                sourceCollider,
                sourceWitness,
                rodNodes,
                inverseRodMasses,
                inverseRodTwistInertias,
                targetNormal,
                sourceNormal
            );
        }
    }
    return response;
}

// Applies one island's complete rod generalized impulse through the retained
// implicit DER factor. A connected rod is represented by exactly one dynamic
// node and therefore has one island owner; lane zero of that packet cohort can
// safely reuse the rod's private workspace without atomics. The band solve is
// matrix-free with respect to contacts and replaces the former diagonal
// inverse-mass approximation in the authoritative velocity update.
inline bool applyFactorizedRodIslandImpulse(
    device const MRMetalWorldContactDispatchGPU& dispatch,
    constant MRMetalWorldPassGPU& pass,
    const uint environment,
    const MRIslandWorkGPU work,
    const uint constraintBase,
    const uint rowBase,
    const uint tileBase,
    device const MRContactConstraintGPU* contacts,
    device const MREvaluatedConstraintIRRowGPU* evaluatedRows,
    device const MRContactTileGPU* tiles,
    device const uint* tileConstraintIndices,
    device const float4* impulseDeltas,
    device MRRodNodeStateGPU* rodNodes,
    device MRRodEdgeStateGPU* rodEdges,
    device const MRRodColliderGPU* rodColliders,
    device const MRRodToolWitnessGPU* rodWitnesses,
    device const uint* constraintWitnessIndices,
    device const MRRodFactorCacheGPU* rodFactorCaches,
    device float* rodOperatorArena,
    thread uint& firstFailingConstraint
) {
    const uint factorStride = rodFactorElementStride(dispatch);
    const uint required =
        factorStride +
        3u * dispatch.rodNodeCount +
        dispatch.rodEdgeCount;
    if (dispatch.operatorVelocityCapacity < required) {
        return false;
    }
    const uint environmentArenaBase =
        environment * dispatch.operatorVelocityCapacity;
    const uint workspaceBase =
        environmentArenaBase + factorStride;
    const uint expectedGeneration =
        pass.physicsSubstep +
        pass.controlStep * max(dispatch.rodCount, 1u);

    for (uint rodIndex = 0u;
         rodIndex < dispatch.rodCount;
         ++rodIndex) {
        bool participates = false;
        uint firstRodConstraint = MR_INVALID_INDEX;
        for (uint localTile = 0u;
             localTile < work.tileCount;
             ++localTile) {
            const MRContactTileGPU tile =
                tiles[tileBase + work.firstTile + localTile];
            for (uint slot = 0u;
                 slot < tile.constraintCount;
                 ++slot) {
                const uint localConstraint =
                    tileConstraintIndices[
                        constraintBase +
                        tile.partialOffset + slot
                    ];
                device const MRContactConstraintGPU& contact =
                    contacts[constraintBase + localConstraint];
                if (!typedRodConstraint(contact)) {
                    continue;
                }
                const uint witnessIndex =
                    constraintWitnessIndices[
                        constraintBase + localConstraint
                    ];
                if (witnessIndex == MR_INVALID_INDEX) {
                    continue;
                }
                const MRRodToolWitnessGPU witness =
                    rodWitnesses[witnessIndex];
                if ((witness.featuresAndFlags.w &
                     MR_ROD_TOOL_WITNESS_VALID) == 0u) {
                    continue;
                }
                const MRRodColliderGPU collider =
                    rodColliders[witness.identity.z];
                if (collider.rodIndex != rodIndex) {
                    continue;
                }
                participates = true;
                firstRodConstraint = min(
                    firstRodConstraint,
                    localConstraint
                );
            }
        }
        if (!participates) {
            continue;
        }

        const MRRodFactorCacheGPU cache =
            rodFactorCaches[
                environment * dispatch.rodCount + rodIndex
            ];
        const uint nodeBase = cache.velocityOffset;
        const uint nodeCount = cache.velocityCount;
        const uint edgeBase = cache.blockCount;
        const uint edgeCount = cache.blockWidth;
        const uint factorEnd =
            cache.firstBlock +
            MR_ROD_FACTOR_TRANSLATION_FLOATS_PER_NODE *
                nodeCount +
            MR_ROD_FACTOR_TWIST_FLOATS_PER_EDGE *
                edgeCount;
        const bool cacheValid =
            cache.environment == environment &&
            cache.rodIndex == rodIndex &&
            cache.generation == expectedGeneration &&
            cache.code == MR_ROD_GPU_SUCCESS &&
            (cache.flags & MR_ROD_FACTOR_CACHE_VALID) != 0u &&
            nodeCount != 0u &&
            edgeCount + 1u == nodeCount &&
            nodeBase + nodeCount <= dispatch.rodNodeCount &&
            edgeBase + edgeCount <= dispatch.rodEdgeCount &&
            cache.firstBlock >= environmentArenaBase &&
            factorEnd <= environmentArenaBase + factorStride;
        if (!cacheValid) {
            firstFailingConstraint = min(
                firstFailingConstraint,
                firstRodConstraint
            );
            return false;
        }

        const uint translationWorkspace =
            workspaceBase + 3u * nodeBase;
        const uint twistWorkspace =
            workspaceBase +
            3u * dispatch.rodNodeCount +
            edgeBase;
        for (uint localNode = 0u;
             localNode < nodeCount;
             ++localNode) {
            const uint nodeIndex = nodeBase + localNode;
            float3 force = float3(0.0f);
            for (uint localTile = 0u;
                 localTile < work.tileCount;
                 ++localTile) {
                const MRContactTileGPU tile =
                    tiles[tileBase + work.firstTile + localTile];
                for (uint slot = 0u;
                     slot < tile.constraintCount;
                     ++slot) {
                    const uint localConstraint =
                        tileConstraintIndices[
                            constraintBase +
                            tile.partialOffset + slot
                        ];
                    device const MRContactConstraintGPU& contact =
                        contacts[
                            constraintBase + localConstraint
                        ];
                    if (!typedRodConstraint(contact)) {
                        continue;
                    }
                    const uint witnessIndex =
                        constraintWitnessIndices[
                            constraintBase + localConstraint
                        ];
                    if (witnessIndex == MR_INVALID_INDEX) {
                        continue;
                    }
                    const MRRodToolWitnessGPU witness =
                        rodWitnesses[witnessIndex];
                    const MRRodColliderGPU collider =
                        rodColliders[witness.identity.z];
                    if (collider.rodIndex != rodIndex ||
                        (nodeIndex != collider.nodeA &&
                         nodeIndex != collider.nodeB)) {
                        continue;
                    }
                    const float3 delta =
                        impulseDeltas[
                            constraintBase + localConstraint
                        ].xyz;
                    device const MREvaluatedConstraintIRRowGPU*
                        localRows =
                            evaluatedRows +
                            rowBase + 3u * localConstraint;
                    const float3 worldImpulse =
                        localRows[0].direction.xyz * delta.x +
                        localRows[1].direction.xyz * delta.y +
                        localRows[2].direction.xyz * delta.z;
                    force += rodNodeImpulseForce(
                        collider,
                        witness,
                        rodNodes,
                        nodeIndex,
                        -worldImpulse
                    );
                }
            }
            rodOperatorArena[
                translationWorkspace + 3u * localNode + 0u
            ] = force.x;
            rodOperatorArena[
                translationWorkspace + 3u * localNode + 1u
            ] = force.y;
            rodOperatorArena[
                translationWorkspace + 3u * localNode + 2u
            ] = force.z;
        }
        for (uint localEdge = 0u;
             localEdge < edgeCount;
             ++localEdge) {
            const uint edgeIndex = edgeBase + localEdge;
            float torque = 0.0f;
            for (uint localTile = 0u;
                 localTile < work.tileCount;
                 ++localTile) {
                const MRContactTileGPU tile =
                    tiles[tileBase + work.firstTile + localTile];
                for (uint slot = 0u;
                     slot < tile.constraintCount;
                     ++slot) {
                    const uint localConstraint =
                        tileConstraintIndices[
                            constraintBase +
                            tile.partialOffset + slot
                        ];
                    device const MRContactConstraintGPU& contact =
                        contacts[
                            constraintBase + localConstraint
                        ];
                    if (!typedRodConstraint(contact)) {
                        continue;
                    }
                    const uint witnessIndex =
                        constraintWitnessIndices[
                            constraintBase + localConstraint
                        ];
                    if (witnessIndex == MR_INVALID_INDEX) {
                        continue;
                    }
                    const MRRodToolWitnessGPU witness =
                        rodWitnesses[witnessIndex];
                    const MRRodColliderGPU collider =
                        rodColliders[witness.identity.z];
                    if (collider.rodIndex != rodIndex ||
                        edgeIndex != collider.edgeIndex) {
                        continue;
                    }
                    const float3 delta =
                        impulseDeltas[
                            constraintBase + localConstraint
                        ].xyz;
                    device const MREvaluatedConstraintIRRowGPU*
                        localRows =
                            evaluatedRows +
                            rowBase + 3u * localConstraint;
                    const float3 worldImpulse =
                        localRows[0].direction.xyz * delta.x +
                        localRows[1].direction.xyz * delta.y +
                        localRows[2].direction.xyz * delta.z;
                    torque += rodTwistImpulseTorque(
                        collider,
                        witness,
                        rodNodes,
                        edgeIndex,
                        -worldImpulse
                    );
                }
            }
            rodOperatorArena[twistWorkspace + localEdge] =
                torque;
        }

        bool valid = solveRodTranslationFactorDevice(
            rodOperatorArena,
            cache.firstBlock,
            3u * nodeCount,
            rodOperatorArena,
            translationWorkspace
        );
        if (valid) {
            valid = solveRodTwistFactorDevice(
                rodOperatorArena,
                cache.firstBlock +
                    MR_ROD_FACTOR_TRANSLATION_FLOATS_PER_NODE *
                        nodeCount,
                edgeCount,
                rodOperatorArena,
                twistWorkspace
            );
        }
        if (!valid) {
            firstFailingConstraint = min(
                firstFailingConstraint,
                firstRodConstraint
            );
            return false;
        }
        for (uint localNode = 0u;
             localNode < nodeCount;
             ++localNode) {
            const uint nodeIndex = nodeBase + localNode;
            const float3 velocityDelta = float3(
                rodOperatorArena[
                    translationWorkspace +
                    3u * localNode + 0u
                ],
                rodOperatorArena[
                    translationWorkspace +
                    3u * localNode + 1u
                ],
                rodOperatorArena[
                    translationWorkspace +
                    3u * localNode + 2u
                ]
            );
            if (!finite3(velocityDelta)) {
                firstFailingConstraint = min(
                    firstFailingConstraint,
                    firstRodConstraint
                );
                return false;
            }
            rodNodes[nodeIndex].velocity.xyz += velocityDelta;
        }
        for (uint localEdge = 0u;
             localEdge < edgeCount;
             ++localEdge) {
            const float twistDelta =
                rodOperatorArena[twistWorkspace + localEdge];
            if (!isfinite(twistDelta)) {
                firstFailingConstraint = min(
                    firstFailingConstraint,
                    firstRodConstraint
                );
                return false;
            }
            rodEdges[edgeBase + localEdge].twistAndRate.y +=
                twistDelta;
        }
    }
    return true;
}

// Shared packet body. Standalone Metal obtains packet slots through indirect
// dispatch while MLX uses a fixed worker grid that repeatedly claims slots
// from the invocation-local queue cursor.
inline void mrWorldWave32SolvePacket(
    device const MRMetalWorldContactDispatchGPU& dispatch,
    device const float* factors,
    device const float* pointJacobians,
    device float* candidateV,
    device MRBodyStateGPU* candidateBodies,
    device MRContactConstraintGPU* contacts,
    device const MRContactPointMetaGPU* contactMetadata,
    device const MREvaluatedConstraintIRRowGPU* evaluatedRows,
    device const MREvaluatedConstraintIRConeGPU* evaluatedCones,
    device float* responseColumns,
    device MRManifoldPointGPU* candidateManifoldPoints,
    device const MRMetalWorldContactStatusGPU* statuses,
    device const MRIslandWorkGPU* islandWork,
    device const MRContactTileGPU* tiles,
    device const uint* tileConstraintIndices,
    device float4* impulseDeltas,
    device MRWave32PreconditionerGPU* preconditioners,
    device MRWave32IslandStatusGPU* waveStatuses,
    device const MRWorkQueueHeaderGPU* workHeaders,
    device const MRWaveWorkPacketGPU* workPackets,
    device MRRodNodeStateGPU* candidateRodNodes,
    device MRRodEdgeStateGPU* candidateRodEdges,
    device const float* inverseRodMasses,
    device const float* inverseRodTwistInertias,
    device const MRRodColliderGPU* rodColliders,
    device MRRodToolWitnessGPU* rodWitnesses,
    device const uint* constraintWitnessIndices,
    device const MRRodFactorCacheGPU* rodFactorCaches,
    device float* rodOperatorArena,
    constant MRMetalWorldPassGPU& pass,
    const uint packetSlot,
    const uint lane,
    threadgroup uint* failureCodes,
    threadgroup uint* failureConstraints,
    threadgroup uint* sharedFailure
) {
    const MRWorkQueueHeaderGPU workHeader =
        workHeaders[MR_WORLD_WORK_SOLVER];
    const MRWaveWorkPacketGPU packet =
        workPackets[packetSlot];
    const uint cohortWidth =
        packet.metadata.x == 8u ||
        packet.metadata.x == 16u
        ? packet.metadata.x
        : MR_WAVE32_CONTACTS_PER_TILE;
    const uint cohortIndex = lane / cohortWidth;
    const uint localLane =
        lane - cohortIndex * cohortWidth;
    if (cohortIndex >= packet.metadata.y) {
        return;
    }
    const uint workSlot = packet.islandSlots[cohortIndex];
    if (workSlot >= workHeader.count) {
        return;
    }
    const MRIslandWorkGPU work = islandWork[workSlot];
    const uint environment = work.environment;
    const uint islandIndex = work.islandIndex;
    if (environment >= dispatch.environmentCount) {
        return;
    }
    const uint islandBase =
        environment * dispatch.islandStride;
    const MRMetalWorldContactStatusGPU environmentStatus =
        statuses[environment];
    if (environmentStatus.code != MR_STEP_SUCCESS ||
        islandIndex >= environmentStatus.islandCount) {
        return;
    }
    if ((work.flags & MR_ISLAND_WORK_VALID) == 0u ||
        work.environment != environment ||
        work.islandIndex != islandIndex ||
        packet.stableKeyLow[cohortIndex] != islandIndex ||
        packet.stableKeyHigh[cohortIndex] != environment) {
        return;
    }
    if ((work.flags & MR_ISLAND_WORK_DISTRIBUTED) != 0u) {
        return;
    }

    const uint constraintBase =
        environment * dispatch.constraintStride;
    const uint rowBase =
        environment * dispatch.rowStride;
    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    const uint velocityBase = environment * dispatch.nv;
    const uint factorBase =
        environment * dispatch.factorStride;
    const uint pointJacobianBase =
        environment *
        (dispatch.pointQueryStride * 3u * dispatch.nv);
    const uint responseBase =
        environment *
        (dispatch.constraintStride * 3u * dispatch.nv);
    const uint manifoldPointBase =
        environment * dispatch.manifoldStride *
        MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
    const uint tileBase =
        environment * dispatch.solverTileCapacity;
    device float* articulationVelocity =
        candidateV + velocityBase;
    device MRBodyStateGPU* bodies =
        candidateBodies + bodyBase;
    device MRRodNodeStateGPU* rodNodes =
        candidateRodNodes +
        environment * dispatch.rodNodeCount;
    device MRRodEdgeStateGPU* rodEdges =
        candidateRodEdges +
        environment * dispatch.rodEdgeCount;

    uint localFailure = MR_STEP_SUCCESS;
    uint localFailureConstraint = MR_INVALID_INDEX;
    float laneMaximumDelta = 0.0f;
    float laneFailureRhs = 0.0f;
    float laneFailureInverse = 0.0f;
    float laneFailureRelaxation = 0.0f;
    if (localLane == 0u) {
        sharedFailure[cohortIndex] = MR_STEP_SUCCESS;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SIMD-saturated response/preconditioner construction and warm-start
    // projection. Each lane solves its own three factorized RHS values.
    for (uint localTile = 0u;
         localTile < work.tileCount;
         ++localTile) {
        const MRContactTileGPU tile =
            tiles[tileBase + work.firstTile + localTile];
        if (localLane >= tile.constraintCount) {
            continue;
        }
        const uint localConstraint =
            tileConstraintIndices[
                constraintBase + tile.partialOffset + localLane
            ];
        device MRContactConstraintGPU& contact =
            contacts[constraintBase + localConstraint];
        float rightHandSide[MR_ARTICULATED_ABA_MAX_DOFS];
        float intermediate[MR_ARTICULATED_ABA_MAX_DOFS];
        float solution[MR_ARTICULATED_ABA_MAX_DOFS];
        if ((dispatch.flags &
             MR_METAL_WORLD_CONTACT_STREAMED_RESPONSES) == 0u) {
            for (uint axis = 0u; axis < 3u; ++axis) {
                const float3 direction =
                    evaluatedRows[
                        rowBase + 3u * localConstraint + axis
                    ].direction.xyz;
                for (uint dof = 0u; dof < dispatch.nv; ++dof) {
                    rightHandSide[dof] = dot(
                        direction,
                        typedArticulationJacobianColumn(
                            localConstraint,
                            contact,
                            bodies,
                            pointJacobians,
                            pointJacobianBase,
                            dof,
                            dispatch.nv,
                            rodWitnesses,
                            constraintWitnessIndices,
                            constraintBase
                        )
                    );
                    intermediate[dof] = 0.0f;
                    solution[dof] = 0.0f;
                }
                if (!solveCholesky(
                        factors,
                        factorBase,
                        dispatch.nv,
                        rightHandSide,
                        intermediate,
                        solution
                    )) {
                    localFailure = MR_STEP_FACTORIZATION_FAILED;
                    localFailureConstraint = min(
                        localFailureConstraint,
                        localConstraint
                    );
                    break;
                }
                for (uint dof = 0u;
                     dof < dispatch.nv;
                     ++dof) {
                    responseColumns[
                        responseBase +
                        (localConstraint * 3u + axis) *
                            dispatch.nv +
                        dof
                    ] = solution[dof];
                }
            }
        }
        if (localFailure != MR_STEP_SUCCESS) {
            continue;
        }

        device const MREvaluatedConstraintIRRowGPU* localRows =
            evaluatedRows + rowBase + 3u * localConstraint;
        const float3 directions[3] = {
            localRows[0].direction.xyz,
            localRows[1].direction.xyz,
            localRows[2].direction.xyz,
        };
        float effective[3][3];
        for (uint row = 0u; row < 3u; ++row) {
            for (uint column = 0u; column < 3u; ++column) {
                float value = typedNormalCrossContactResponse(
                    localConstraint,
                    localConstraint,
                    column,
                    contact,
                    contact,
                    directions[row],
                    directions[column],
                    bodies,
                    pointJacobians,
                    pointJacobianBase,
                    responseColumns,
                    responseBase,
                    dispatch.nv,
                    dispatch.articulationIndex,
                    rodNodes,
                    inverseRodMasses,
                    inverseRodTwistInertias,
                    rodColliders,
                    rodWitnesses,
                    constraintWitnessIndices,
                    constraintBase
                );
                if (row == column) {
                    value += localRows[row].regularization;
                }
                effective[row][column] = value;
            }
        }
        float inverse[3][3];
        if (!invert3x3(effective, inverse)) {
            localFailure = MR_STEP_FACTORIZATION_FAILED;
            localFailureConstraint = min(
                localFailureConstraint,
                localConstraint
            );
            continue;
        }
        MRWave32PreconditionerGPU preconditioner = {};
        preconditioner.row0.xyz = float3(
            inverse[0][0],
            inverse[0][1],
            inverse[0][2]
        );
        preconditioner.row1.xyz = float3(
            inverse[1][0],
            inverse[1][1],
            inverse[1][2]
        );
        preconditioner.row2.xyz = float3(
            inverse[2][0],
            inverse[2][1],
            inverse[2][2]
        );
        preconditioners[
            constraintBase + localConstraint
        ] = preconditioner;
        const float3 warm = projectFrictionCone(
            contact.impulses.xyz,
            evaluatedCones[constraintBase + localConstraint]
        );
        contact.impulses.xyz = warm;
        impulseDeltas[
            constraintBase + localConstraint
        ] = float4(warm, 0.0f);
    }

    failureCodes[lane] = localFailure;
    failureConstraints[lane] = localFailureConstraint;
    threadgroup_barrier(mem_flags::mem_device |
                        mem_flags::mem_threadgroup);
    if (localLane == 0u) {
        sharedFailure[cohortIndex] = MR_STEP_SUCCESS;
        uint firstConstraint = MR_INVALID_INDEX;
        const uint firstCohortLane =
            cohortIndex * cohortWidth;
        for (uint sourceLane = firstCohortLane;
             sourceLane < firstCohortLane + cohortWidth;
             ++sourceLane) {
            if (failureCodes[sourceLane] == MR_STEP_SUCCESS) {
                continue;
            }
            sharedFailure[cohortIndex] =
                failureCodes[sourceLane];
            firstConstraint = min(
                firstConstraint,
                failureConstraints[sourceLane]
            );
        }
        if (sharedFailure[cohortIndex] != MR_STEP_SUCCESS) {
            MRWave32IslandStatusGPU failed = {};
            failed.code = sharedFailure[cohortIndex];
            failed.environment = environment;
            failed.islandIndex = islandIndex;
            failed.residuals.w =
                static_cast<float>(firstConstraint);
            waveStatuses[islandBase + islandIndex] = failed;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint groupFailure = simd_max(localFailure);
    if (groupFailure != MR_STEP_SUCCESS) {
        if (localLane == 0u &&
            sharedFailure[cohortIndex] == MR_STEP_SUCCESS) {
            // A sibling cohort failed its factor solve. Route this healthy
            // island through the deterministic ordered replay so every lane
            // exits the packet uniformly without sacrificing per-environment
            // transactional isolation.
            MRWave32IslandStatusGPU replay = {};
            replay.code = MR_STEP_SUCCESS;
            replay.environment = environment;
            replay.islandIndex = islandIndex;
            replay.residuals.w = 1.0f;
            waveStatuses[islandBase + islandIndex] = replay;
        }
        return;
    }

    // Apply every warm-start impulse exactly once. Articulation DoFs and free
    // bodies have unique lane owners, eliminating conflicting atomic writes.
    if ((work.flags & MR_ISLAND_WORK_HAS_ARTICULATION) != 0u) {
        for (uint dof = localLane;
             dof < dispatch.nv;
             dof += cohortWidth) {
            float velocityDelta = 0.0f;
            bool participates = false;
            for (uint localTile = 0u;
                 localTile < work.tileCount;
                 ++localTile) {
                const MRContactTileGPU tile =
                    tiles[tileBase + work.firstTile + localTile];
                for (uint slot = 0u;
                     slot < tile.constraintCount;
                     ++slot) {
                    const uint localConstraint =
                        tileConstraintIndices[
                            constraintBase +
                            tile.partialOffset + slot
                        ];
                    const float3 delta =
                        impulseDeltas[
                            constraintBase + localConstraint
                        ].xyz;
                    const float response0 =
                        responseColumns[
                            responseBase +
                                (localConstraint * 3u + 0u) *
                                    dispatch.nv +
                                dof
                        ];
                    const float response1 =
                        responseColumns[
                            responseBase +
                                (localConstraint * 3u + 1u) *
                                    dispatch.nv +
                                dof
                        ];
                    const float response2 =
                        responseColumns[
                            responseBase +
                                (localConstraint * 3u + 2u) *
                                    dispatch.nv +
                                dof
                        ];
                    participates =
                        participates ||
                        response0 != 0.0f ||
                        response1 != 0.0f ||
                        response2 != 0.0f;
                    velocityDelta +=
                        response0 * delta.x +
                        response1 * delta.y +
                        response2 * delta.z;
                }
            }
            if (participates) {
                articulationVelocity[dof] += velocityDelta;
            }
        }
    }
    for (uint bodyIndex = localLane;
         bodyIndex < dispatch.bodyCount;
         bodyIndex += cohortWidth) {
        device MRBodyStateGPU& body = bodies[bodyIndex];
        if (!dynamicSceneEndpoint(
                body,
                dispatch.articulationIndex
            ) ||
            !waveIslandContainsBody(
                work,
                bodyIndex,
                contacts,
                constraintBase,
                tiles,
                tileBase,
                tileConstraintIndices,
                rodWitnesses,
                constraintWitnessIndices
            )) {
            continue;
        }
        float3 linearImpulse = float3(0.0f);
        float3 angularImpulse = float3(0.0f);
        for (uint localTile = 0u;
             localTile < work.tileCount;
             ++localTile) {
            const MRContactTileGPU tile =
                tiles[tileBase + work.firstTile + localTile];
            for (uint slot = 0u;
                 slot < tile.constraintCount;
                 ++slot) {
                const uint localConstraint =
                    tileConstraintIndices[
                        constraintBase +
                        tile.partialOffset + slot
                    ];
                device const MRContactConstraintGPU& contact =
                    contacts[constraintBase + localConstraint];
                const float3 delta =
                    impulseDeltas[
                        constraintBase + localConstraint
                    ].xyz;
                device const MREvaluatedConstraintIRRowGPU*
                    localRows =
                        evaluatedRows +
                        rowBase + 3u * localConstraint;
                const float3 impulse =
                    localRows[0].direction.xyz * delta.x +
                    localRows[1].direction.xyz * delta.y +
                    localRows[2].direction.xyz * delta.z;
                if (typedRodConstraint(contact)) {
                    const uint witnessIndex =
                        constraintWitnessIndices[
                            constraintBase + localConstraint
                        ];
                    if (witnessIndex != MR_INVALID_INDEX) {
                        const MRRodToolWitnessGPU witness =
                            rodWitnesses[witnessIndex];
                        if (witness.featuresAndFlags.z ==
                            bodyIndex) {
                            linearImpulse += impulse;
                            angularImpulse += cross(
                                witness
                                    .toolPointAndSeparation.xyz -
                                    body.position.xyz,
                                impulse
                            );
                        }
                    }
                    continue;
                }
                if (contact.bodyA == bodyIndex) {
                    linearImpulse -= impulse;
                    angularImpulse += cross(
                        contact.pointAndSeparation.xyz -
                            body.position.xyz,
                        -impulse
                    );
                }
                if (contact.bodyB == bodyIndex) {
                    linearImpulse += impulse;
                    angularImpulse += cross(
                        contact.pointAndSeparation.xyz -
                            body.position.xyz,
                        impulse
                    );
                }
            }
        }
        body.linearVelocityAndInverseMass.xyz +=
            body.linearVelocityAndInverseMass.w *
            linearImpulse;
        body.angularVelocity.xyz += multiply(
            stateInverseInertia(body),
            angularImpulse
        );
    }
    if ((work.flags & MR_ISLAND_WORK_HAS_ROD) != 0u &&
        localLane == 0u &&
        !applyFactorizedRodIslandImpulse(
            dispatch,
            pass,
            environment,
            work,
            constraintBase,
            rowBase,
            tileBase,
            contacts,
            evaluatedRows,
            tiles,
            tileConstraintIndices,
            impulseDeltas,
            rodNodes,
            rodEdges,
            rodColliders,
            rodWitnesses,
            constraintWitnessIndices,
            rodFactorCaches,
            rodOperatorArena,
            localFailureConstraint
        )) {
        localFailure = MR_STEP_FACTORIZATION_FAILED;
    }
    threadgroup_barrier(mem_flags::mem_device);

    const uint solverIterations =
        dispatch.velocityIterations +
        (pass.reserved0 != 0u
             ? dispatch.finalVelocityIterations
             : 0u);
    for (uint iteration = 0u;
         iteration < solverIterations;
         ++iteration) {
        for (uint localTile = 0u;
             localTile < work.tileCount;
             ++localTile) {
            const MRContactTileGPU tile =
                tiles[tileBase + work.firstTile + localTile];
            if (localLane >= tile.constraintCount) {
                continue;
            }
            const uint localConstraint =
                tileConstraintIndices[
                    constraintBase +
                    tile.partialOffset +
                    localLane
                ];
            device MRContactConstraintGPU& contact =
                contacts[constraintBase + localConstraint];
            device const MREvaluatedConstraintIRRowGPU*
                localRows =
                    evaluatedRows +
                    rowBase + 3u * localConstraint;
            const float3 relative = typedRelativePointVelocity(
                localConstraint,
                contact,
                bodies,
                dispatch.articulationIndex,
                pointJacobians,
                pointJacobianBase,
                dispatch.nv,
                articulationVelocity,
                rodNodes,
                rodEdges,
                rodColliders,
                rodWitnesses,
                constraintWitnessIndices,
                constraintBase
            );
            const float3 previous = contact.impulses.xyz;
            const float3 rhs = float3(
                localRows[0].targetVelocity -
                    dot(localRows[0].direction.xyz, relative) -
                    localRows[0].regularization * previous.x,
                localRows[1].targetVelocity -
                    dot(localRows[1].direction.xyz, relative) -
                    localRows[1].regularization * previous.y,
                localRows[2].targetVelocity -
                    dot(localRows[2].direction.xyz, relative) -
                    localRows[2].regularization * previous.z
            );
            const MRWave32PreconditionerGPU preconditioner =
                preconditioners[
                    constraintBase + localConstraint
                ];
            float3 proposed = previous + float3(
                dot(preconditioner.row0.xyz, rhs),
                dot(preconditioner.row1.xyz, rhs),
                dot(preconditioner.row2.xyz, rhs)
            );
            proposed = projectFrictionCone(
                proposed,
                evaluatedCones[
                    constraintBase + localConstraint
                ]
            );
            float absoluteNormalRowSum = 0.0f;
            float diagonalNormalResponse = 0.0f;
            for (uint sourceTile = 0u;
                 sourceTile < work.tileCount;
                 ++sourceTile) {
                const MRContactTileGPU source =
                    tiles[
                        tileBase + work.firstTile + sourceTile
                    ];
                for (uint sourceSlot = 0u;
                     sourceSlot < source.constraintCount;
                     ++sourceSlot) {
                    const uint sourceConstraint =
                        tileConstraintIndices[
                            constraintBase +
                            source.partialOffset +
                            sourceSlot
                        ];
                    device const MRContactConstraintGPU&
                        sourceContact =
                            contacts[
                                constraintBase +
                                sourceConstraint
                            ];
                    const float3 sourceNormal =
                        evaluatedRows[
                            rowBase +
                            3u * sourceConstraint
                        ].direction.xyz;
                    const float crossResponse =
                        typedNormalCrossContactResponse(
                            localConstraint,
                            sourceConstraint,
                            0u,
                            contact,
                            sourceContact,
                            localRows[0].direction.xyz,
                            sourceNormal,
                            bodies,
                            pointJacobians,
                            pointJacobianBase,
                            responseColumns,
                            responseBase,
                            dispatch.nv,
                            dispatch.articulationIndex,
                            rodNodes,
                            inverseRodMasses,
                            inverseRodTwistInertias,
                            rodColliders,
                            rodWitnesses,
                            constraintWitnessIndices,
                            constraintBase
                        );
                    absoluteNormalRowSum += abs(crossResponse);
                    if (sourceConstraint == localConstraint) {
                        diagonalNormalResponse =
                            abs(crossResponse) +
                            localRows[0].regularization;
                    }
                }
            }
            const float relaxation =
                absoluteNormalRowSum >
                    max(diagonalNormalResponse, kMatrixFloor)
                ? clamp(
                      diagonalNormalResponse /
                          absoluteNormalRowSum,
                      1.0f / 32.0f,
                      1.0f
                  )
                : 1.0f;
            const float3 delta =
                relaxation * (proposed - previous);
            const float3 candidate = previous + delta;
            if (!finite3(candidate) || !finite3(delta)) {
                laneFailureRhs = max(
                    abs(rhs.x),
                    max(abs(rhs.y), abs(rhs.z))
                );
                laneFailureInverse = max(
                    max(
                        abs(preconditioner.row0.x),
                        max(
                            abs(preconditioner.row0.y),
                            abs(preconditioner.row0.z)
                        )
                    ),
                    max(
                        max(
                            abs(preconditioner.row1.x),
                            max(
                                abs(preconditioner.row1.y),
                                abs(preconditioner.row1.z)
                            )
                        ),
                        max(
                            abs(preconditioner.row2.x),
                            max(
                                abs(preconditioner.row2.y),
                                abs(preconditioner.row2.z)
                            )
                        )
                    )
                );
                laneFailureRelaxation = relaxation;
                localFailure = MR_STEP_NONFINITE_RESULT;
                localFailureConstraint = min(
                    localFailureConstraint,
                    localConstraint
                );
                impulseDeltas[
                    constraintBase + localConstraint
                ] = float4(0.0f);
                continue;
            }
            contact.impulses.xyz = candidate;
            impulseDeltas[
                constraintBase + localConstraint
            ] = float4(delta, 0.0f);
            laneMaximumDelta = max(
                laneMaximumDelta,
                max(
                    abs(delta.x),
                    max(abs(delta.y), abs(delta.z))
                )
            );
        }
        threadgroup_barrier(mem_flags::mem_device);

        if ((work.flags &
             MR_ISLAND_WORK_HAS_ARTICULATION) != 0u) {
            for (uint dof = localLane;
                 dof < dispatch.nv;
                 dof += cohortWidth) {
                float velocityDelta = 0.0f;
                bool participates = false;
                for (uint localTile = 0u;
                     localTile < work.tileCount;
                     ++localTile) {
                    const MRContactTileGPU tile =
                        tiles[
                            tileBase +
                            work.firstTile + localTile
                        ];
                    for (uint slot = 0u;
                         slot < tile.constraintCount;
                         ++slot) {
                        const uint localConstraint =
                            tileConstraintIndices[
                                constraintBase +
                                tile.partialOffset + slot
                            ];
                        const float3 delta =
                            impulseDeltas[
                                constraintBase +
                                localConstraint
                            ].xyz;
                        const float response0 =
                            responseColumns[
                                responseBase +
                                    (localConstraint * 3u + 0u) *
                                        dispatch.nv +
                                    dof
                            ];
                        const float response1 =
                            responseColumns[
                                responseBase +
                                    (localConstraint * 3u + 1u) *
                                        dispatch.nv +
                                    dof
                            ];
                        const float response2 =
                            responseColumns[
                                responseBase +
                                    (localConstraint * 3u + 2u) *
                                        dispatch.nv +
                                    dof
                            ];
                        participates =
                            participates ||
                            response0 != 0.0f ||
                            response1 != 0.0f ||
                            response2 != 0.0f;
                        velocityDelta +=
                            response0 * delta.x +
                            response1 * delta.y +
                            response2 * delta.z;
                    }
                }
                if (participates) {
                    articulationVelocity[dof] +=
                        velocityDelta;
                }
            }
        }
        for (uint bodyIndex = localLane;
             bodyIndex < dispatch.bodyCount;
             bodyIndex += cohortWidth) {
            device MRBodyStateGPU& body = bodies[bodyIndex];
            if (!dynamicSceneEndpoint(
                    body,
                    dispatch.articulationIndex
                ) ||
                !waveIslandContainsBody(
                    work,
                    bodyIndex,
                    contacts,
                    constraintBase,
                    tiles,
                    tileBase,
                    tileConstraintIndices,
                    rodWitnesses,
                    constraintWitnessIndices
                )) {
                continue;
            }
            float3 linearImpulse = float3(0.0f);
            float3 angularImpulse = float3(0.0f);
            for (uint localTile = 0u;
                 localTile < work.tileCount;
                 ++localTile) {
                const MRContactTileGPU tile =
                    tiles[
                        tileBase + work.firstTile + localTile
                    ];
                for (uint slot = 0u;
                     slot < tile.constraintCount;
                     ++slot) {
                    const uint localConstraint =
                        tileConstraintIndices[
                            constraintBase +
                            tile.partialOffset + slot
                        ];
                    device const MRContactConstraintGPU& contact =
                        contacts[
                            constraintBase + localConstraint
                        ];
                    const float3 delta =
                        impulseDeltas[
                            constraintBase + localConstraint
                        ].xyz;
                    device const MREvaluatedConstraintIRRowGPU*
                        localRows =
                            evaluatedRows +
                            rowBase + 3u * localConstraint;
                    const float3 impulse =
                        localRows[0].direction.xyz * delta.x +
                        localRows[1].direction.xyz * delta.y +
                        localRows[2].direction.xyz * delta.z;
                    if (typedRodConstraint(contact)) {
                        const uint witnessIndex =
                            constraintWitnessIndices[
                                constraintBase +
                                localConstraint
                            ];
                        if (witnessIndex != MR_INVALID_INDEX) {
                            const MRRodToolWitnessGPU witness =
                                rodWitnesses[witnessIndex];
                            if (witness.featuresAndFlags.z ==
                                bodyIndex) {
                                linearImpulse += impulse;
                                angularImpulse += cross(
                                    witness
                                        .toolPointAndSeparation
                                        .xyz -
                                        body.position.xyz,
                                    impulse
                                );
                            }
                        }
                        continue;
                    }
                    if (contact.bodyA == bodyIndex) {
                        linearImpulse -= impulse;
                        angularImpulse += cross(
                            contact.pointAndSeparation.xyz -
                                body.position.xyz,
                            -impulse
                        );
                    }
                    if (contact.bodyB == bodyIndex) {
                        linearImpulse += impulse;
                        angularImpulse += cross(
                            contact.pointAndSeparation.xyz -
                                body.position.xyz,
                            impulse
                        );
                    }
                }
            }
            body.linearVelocityAndInverseMass.xyz +=
                body.linearVelocityAndInverseMass.w *
                linearImpulse;
            body.angularVelocity.xyz += multiply(
                stateInverseInertia(body),
                angularImpulse
            );
        }
        if ((work.flags & MR_ISLAND_WORK_HAS_ROD) != 0u &&
            localLane == 0u &&
            !applyFactorizedRodIslandImpulse(
                dispatch,
                pass,
                environment,
                work,
                constraintBase,
                rowBase,
                tileBase,
                contacts,
                evaluatedRows,
                tiles,
                tileConstraintIndices,
                impulseDeltas,
                rodNodes,
                rodEdges,
                rodColliders,
                rodWitnesses,
                constraintWitnessIndices,
                rodFactorCaches,
                rodOperatorArena,
                localFailureConstraint
            )) {
            localFailure = MR_STEP_FACTORIZATION_FAILED;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }

    float laneNormalResidual = 0.0f;
    float laneConeViolation = 0.0f;
    for (uint localTile = 0u;
         localTile < work.tileCount;
         ++localTile) {
        const MRContactTileGPU tile =
            tiles[tileBase + work.firstTile + localTile];
        if (localLane >= tile.constraintCount) {
            continue;
        }
        const uint localConstraint =
            tileConstraintIndices[
                constraintBase +
                tile.partialOffset +
                localLane
            ];
        device MRContactConstraintGPU& contact =
            contacts[constraintBase + localConstraint];
        device const MREvaluatedConstraintIRRowGPU* localRows =
            evaluatedRows + rowBase + 3u * localConstraint;
        const float3 relative = typedRelativePointVelocity(
            localConstraint,
            contact,
            bodies,
            dispatch.articulationIndex,
            pointJacobians,
            pointJacobianBase,
            dispatch.nv,
            articulationVelocity,
            rodNodes,
            rodEdges,
            rodColliders,
            rodWitnesses,
            constraintWitnessIndices,
            constraintBase
        );
        const float normalEquation =
            dot(localRows[0].direction.xyz, relative) -
            localRows[0].targetVelocity +
            localRows[0].regularization *
                contact.impulses.x;
        laneNormalResidual = max(
            laneNormalResidual,
            contact.impulses.x > kConeEpsilon
            ? abs(normalEquation)
            : max(-normalEquation, 0.0f)
        );
        const MREvaluatedConstraintIRConeGPU cone =
            evaluatedCones[
                constraintBase + localConstraint
            ];
        const float limitU =
            cone.effectiveFrictionU * contact.impulses.x;
        const float limitV =
            cone.effectiveFrictionV * contact.impulses.x;
        float coneViolation = 0.0f;
        if (limitU > 0.0f && limitV > 0.0f) {
            coneViolation = max(
                sqrt(
                    (contact.impulses.y * contact.impulses.y) /
                        (limitU * limitU) +
                    (contact.impulses.z * contact.impulses.z) /
                        (limitV * limitV)
                ) - 1.0f,
                0.0f
            );
        } else {
            coneViolation = length(contact.impulses.yz);
        }
        laneConeViolation = max(
            laneConeViolation,
            coneViolation
        );

        const MRContactPointMetaGPU metadata =
            contactMetadata[constraintBase + localConstraint];
        if (typedRodConstraint(contact)) {
            const uint witnessIndex =
                constraintWitnessIndices[
                    constraintBase + localConstraint
                ];
            if (witnessIndex != MR_INVALID_INDEX) {
                rodWitnesses[witnessIndex].impulses =
                    contact.impulses;
            }
        }
        if (metadata.manifoldIndex <
                dispatch.manifoldCapacity &&
            metadata.pointIndex <
                MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY) {
            candidateManifoldPoints[
                manifoldPointBase +
                metadata.manifoldIndex *
                    MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY +
                metadata.pointIndex
            ].impulses = contact.impulses;
        }
    }
    const float maximumImpulseDelta = waveCohortMaximum(
        laneMaximumDelta,
        cohortWidth
    );
    const float maximumNormalResidual = waveCohortMaximum(
        laneNormalResidual,
        cohortWidth
    );
    const float maximumConeViolation = waveCohortMaximum(
        laneConeViolation,
        cohortWidth
    );
    const uint maximumFailureCode = waveCohortMaximum(
        localFailure,
        cohortWidth
    );
    const uint firstFailureConstraint =
        waveCohortMinimum(
            localFailureConstraint,
            cohortWidth
        );
    const float maximumFailureRhs = waveCohortMaximum(
        laneFailureRhs,
        cohortWidth
    );
    const float maximumFailureInverse = waveCohortMaximum(
        laneFailureInverse,
        cohortWidth
    );
    const float maximumFailureRelaxation = waveCohortMaximum(
        laneFailureRelaxation,
        cohortWidth
    );
    if (localLane == 0u) {
        MRWave32IslandStatusGPU result = {};
        result.code =
            maximumFailureCode;
        result.environment = environment;
        result.islandIndex = islandIndex;
        result.iterations = solverIterations;
        if (maximumFailureCode != MR_STEP_SUCCESS) {
            result.residuals = float4(
                maximumFailureRhs,
                maximumFailureInverse,
                maximumFailureRelaxation,
                static_cast<float>(firstFailureConstraint)
            );
        } else {
            result.residuals = float4(
                maximumImpulseDelta,
                maximumNormalResidual,
                maximumConeViolation,
                0.0f
            );
        }
        waveStatuses[islandBase + islandIndex] = result;
    }
}

// One SIMD32 group owns one full mixed-contact island or a deterministic
// packet of two/four homogeneous compact islands. Within a packet, contiguous
// 16/8-lane cohorts own independent islands. A cohort lane owns one coupled
// normal/tangent block in each tile, then cooperatively assembles a single
// articulation/free-body velocity update without atomics.
kernel void mr_world_wave32_solve(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const float* factors [[buffer(1)]],
    device const float* pointJacobians [[buffer(2)]],
    device float* candidateV [[buffer(3)]],
    device MRBodyStateGPU* candidateBodies [[buffer(4)]],
    device MRContactConstraintGPU* contacts [[buffer(5)]],
    device const MRContactPointMetaGPU* contactMetadata [[buffer(6)]],
    device const MREvaluatedConstraintIRRowGPU* evaluatedRows [[buffer(7)]],
    device const MREvaluatedConstraintIRConeGPU* evaluatedCones [[buffer(8)]],
    device float* responseColumns [[buffer(9)]],
    device MRManifoldPointGPU* candidateManifoldPoints [[buffer(10)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(11)]],
    device const MRIslandWorkGPU* islandWork [[buffer(12)]],
    device const MRContactTileGPU* tiles [[buffer(13)]],
    device const uint* tileConstraintIndices [[buffer(14)]],
    device float4* impulseDeltas [[buffer(15)]],
    device MRWave32PreconditionerGPU* preconditioners [[buffer(16)]],
    device MRWave32IslandStatusGPU* waveStatuses [[buffer(17)]],
    device const MRWorkQueueHeaderGPU* workHeaders [[buffer(18)]],
    constant MRMetalWorldPassGPU& pass [[buffer(19)]],
    device const MRWaveWorkPacketGPU* workPackets [[buffer(20)]],
    device MRRodNodeStateGPU* candidateRodNodes [[buffer(21)]],
    device MRRodEdgeStateGPU* candidateRodEdges [[buffer(22)]],
    device const float* inverseRodMasses [[buffer(23)]],
    device const float* inverseRodTwistInertias [[buffer(24)]],
    device const MRRodColliderGPU* rodColliders [[buffer(25)]],
    device MRRodToolWitnessGPU* rodWitnesses [[buffer(26)]],
    device const uint* constraintWitnessIndices [[buffer(27)]],
    device const MRRodFactorCacheGPU* rodFactorCaches [[buffer(28)]],
    device float* rodOperatorArena [[buffer(29)]],
    const uint packetSlot [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup uint failureCodes[
        MR_WAVE32_CONTACTS_PER_TILE
    ];
    threadgroup uint failureConstraints[
        MR_WAVE32_CONTACTS_PER_TILE
    ];
    threadgroup uint sharedFailure[4u];
    mrWorldWave32SolvePacket(
        dispatch,
        factors,
        pointJacobians,
        candidateV,
        candidateBodies,
        contacts,
        contactMetadata,
        evaluatedRows,
        evaluatedCones,
        responseColumns,
        candidateManifoldPoints,
        statuses,
        islandWork,
        tiles,
        tileConstraintIndices,
        impulseDeltas,
        preconditioners,
        waveStatuses,
        workHeaders,
        workPackets,
        candidateRodNodes,
        candidateRodEdges,
        inverseRodMasses,
        inverseRodTwistInertias,
        rodColliders,
        rodWitnesses,
        constraintWitnessIndices,
        rodFactorCaches,
        rodOperatorArena,
        pass,
        packetSlot,
        lane,
        failureCodes,
        failureConstraints,
        sharedFailure
    );
}

// MLX cannot issue an indirect dispatch through its active encoder. A bounded
// occupancy-sized grid therefore remains resident and claims deterministic
// packet slots through one relaxed atomic per packet. Packet writes are
// disjoint, so claim order cannot affect physical ordering or replay hashes.
kernel void mr_world_wave32_solve_persistent(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const float* factors [[buffer(1)]],
    device const float* pointJacobians [[buffer(2)]],
    device float* candidateV [[buffer(3)]],
    device MRBodyStateGPU* candidateBodies [[buffer(4)]],
    device MRContactConstraintGPU* contacts [[buffer(5)]],
    device const MRContactPointMetaGPU* contactMetadata [[buffer(6)]],
    device const MREvaluatedConstraintIRRowGPU* evaluatedRows [[buffer(7)]],
    device const MREvaluatedConstraintIRConeGPU* evaluatedCones [[buffer(8)]],
    device float* responseColumns [[buffer(9)]],
    device MRManifoldPointGPU* candidateManifoldPoints [[buffer(10)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(11)]],
    device const MRIslandWorkGPU* islandWork [[buffer(12)]],
    device const MRContactTileGPU* tiles [[buffer(13)]],
    device const uint* tileConstraintIndices [[buffer(14)]],
    device float4* impulseDeltas [[buffer(15)]],
    device MRWave32PreconditionerGPU* preconditioners [[buffer(16)]],
    device MRWave32IslandStatusGPU* waveStatuses [[buffer(17)]],
    device MRWorkQueueHeaderGPU* workHeaders [[buffer(18)]],
    constant MRMetalWorldPassGPU& pass [[buffer(19)]],
    device const MRWaveWorkPacketGPU* workPackets [[buffer(20)]],
    device MRRodNodeStateGPU* candidateRodNodes [[buffer(21)]],
    device MRRodEdgeStateGPU* candidateRodEdges [[buffer(22)]],
    device const float* inverseRodMasses [[buffer(23)]],
    device const float* inverseRodTwistInertias [[buffer(24)]],
    device const MRRodColliderGPU* rodColliders [[buffer(25)]],
    device MRRodToolWitnessGPU* rodWitnesses [[buffer(26)]],
    device const uint* constraintWitnessIndices [[buffer(27)]],
    device const MRRodFactorCacheGPU* rodFactorCaches [[buffer(28)]],
    device float* rodOperatorArena [[buffer(29)]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint kPacketClaimBatch = 4u;
    threadgroup uint claimedPacketBase;
    threadgroup uint failureCodes[
        MR_WAVE32_CONTACTS_PER_TILE
    ];
    threadgroup uint failureConstraints[
        MR_WAVE32_CONTACTS_PER_TILE
    ];
    threadgroup uint sharedFailure[4u];
    device MRWorkQueueHeaderGPU& header =
        workHeaders[MR_WORLD_WORK_SOLVER];
    device atomic_uint* cursor =
        reinterpret_cast<device atomic_uint*>(
            &header.workerCursor
        );
    if (lane == 0u) {
        device atomic_uint* flags =
            reinterpret_cast<device atomic_uint*>(
                &header.flags
            );
        atomic_fetch_or_explicit(
            flags,
            static_cast<uint>(
                MR_WORLD_QUEUE_PERSISTENT_WORKER
            ),
            memory_order_relaxed
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    while (true) {
        if (lane == 0u) {
            claimedPacketBase = atomic_fetch_add_explicit(
                cursor,
                kPacketClaimBatch,
                memory_order_relaxed
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (claimedPacketBase >=
            header.indirect.threadgroupsX) {
            break;
        }
        for (uint localPacket = 0u;
             localPacket < kPacketClaimBatch;
             ++localPacket) {
            const uint packetSlot =
                claimedPacketBase + localPacket;
            if (packetSlot >=
                header.indirect.threadgroupsX) {
                break;
            }
            mrWorldWave32SolvePacket(
                dispatch,
                factors,
                pointJacobians,
                candidateV,
                candidateBodies,
                contacts,
                contactMetadata,
                evaluatedRows,
                evaluatedCones,
                responseColumns,
                candidateManifoldPoints,
                statuses,
                islandWork,
                tiles,
                tileConstraintIndices,
                impulseDeltas,
                preconditioners,
                waveStatuses,
                workHeaders,
                workPackets,
                candidateRodNodes,
                candidateRodEdges,
                inverseRodMasses,
                inverseRodTwistInertias,
                rodColliders,
                rodWitnesses,
                constraintWitnessIndices,
                rodFactorCaches,
                rodOperatorArena,
                pass,
                packetSlot,
                lane,
                failureCodes,
                failureConstraints,
                sharedFailure
            );
            threadgroup_barrier(
                mem_flags::mem_device |
                mem_flags::mem_threadgroup
            );
        }
    }
}

// Environment-level reduction is deliberately separate from the island
// kernels: islands never contend on status publication and failed
// environments remain transactionally isolated.
kernel void mr_world_reduce_wave32_status(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRWave32IslandStatusGPU* waveStatuses [[buffer(1)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(2)]],
    device const MRWorkQueueHeaderGPU* workHeaders [[buffer(3)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    if (status.code != MR_STEP_SUCCESS) {
        return;
    }
    const uint islandBase =
        environment * dispatch.islandStride;
    float4 residuals = float4(0.0f);
    uint iterations = 0u;
    for (uint island = 0u;
         island < status.islandCount;
         ++island) {
        const MRWave32IslandStatusGPU source =
            waveStatuses[islandBase + island];
        if (source.code != MR_STEP_SUCCESS) {
            status.code = source.code;
            if (source.iterations == MR_INVALID_INDEX) {
                status.firstFailingConstraint =
                    MR_INVALID_INDEX;
                status.firstFailingStableKeyLow =
                    source.islandIndex;
                status.firstFailingStableKeyHigh =
                    source.environment;
            } else {
                status.firstFailingConstraint =
                    static_cast<uint>(source.residuals.w);
                status.firstFailingStableKeyLow = 0u;
                status.firstFailingStableKeyHigh = 0u;
                residuals = source.residuals;
            }
            break;
        }
        residuals = max(residuals, source.residuals);
        iterations = max(iterations, source.iterations);
    }
    status.solverIterations = iterations;
    status.residuals = residuals;
    const MRWorkQueueHeaderGPU solverHeader =
        workHeaders[MR_WORLD_WORK_SOLVER];
    status.queueFlags |=
        solverHeader.flags &
        (
            MR_WORLD_QUEUE_COHORT_8 |
            MR_WORLD_QUEUE_COHORT_16
        );
    if ((solverHeader.flags &
         MR_WORLD_QUEUE_PERSISTENT_WORKER) != 0u) {
        status.queueFlags |=
            MR_WORLD_QUEUE_PERSISTENT_WORKER;
        status.workerPackets =
            solverHeader.indirect.threadgroupsX;
        status.workerHighWater = max(
            status.workerHighWater,
            solverHeader.indirect.threadgroupsX
        );
        const uint roundedPacketClaims =
            (
                solverHeader.indirect.threadgroupsX + 3u
            ) & ~3u;
        status.workerEmptyPulls =
            solverHeader.workerCursor >
                    roundedPacketClaims
            ? (
                  solverHeader.workerCursor -
                  roundedPacketClaims
              ) / 4u
            : 0u;
    }
    if (residuals.w > 0.0f) {
        status.queueFlags |=
            MR_ISLAND_WORK_STIFF_REPLAY;
    }
    statuses[environment] = status;
}

// Ordered sparse generalized blocks share the composed block-diagonal
// articulation factor and canonical ConstraintIR stream. Contact/rod blocks
// are handled by Wave32; this pass closes limits/equalities/gears/tendons
// without routing through the compatibility multi-articulation solver.
kernel void mr_world_solve_generalized_constraints(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const float* factors [[buffer(1)]],
    device float* candidateV [[buffer(2)]],
    device MRContactConstraintGPU* contacts [[buffer(3)]],
    device const MRConstraintIRBlockGPU* blocks [[buffer(4)]],
    device const MRConstraintIREndpointGPU* endpoints [[buffer(5)]],
    device const MREvaluatedConstraintIRRowGPU* evaluatedRows [[buffer(6)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(7)]],
    constant MRMetalWorldPassGPU& pass [[buffer(8)]],
    device MRBodyStateGPU* candidateBodies [[buffer(9)]],
    device MRRodNodeStateGPU* candidateRodNodes [[buffer(10)]],
    device const float* inverseRodMasses [[buffer(11)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount ||
        dispatch.authoredConstraintCount == 0u) {
        return;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    if (status.code != MR_STEP_SUCCESS ||
        dispatch.nv > MR_ARTICULATED_ABA_MAX_DOFS) {
        return;
    }
    const uint constraintBase =
        environment * dispatch.constraintStride;
    const uint endpointBase =
        2u * environment * dispatch.constraintStride;
    const uint rowBase =
        environment * dispatch.rowStride;
    const uint velocityBase = environment * dispatch.nv;
    const uint factorBase =
        environment * dispatch.factorStride;
    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    const uint rodNodeBase =
        environment * dispatch.rodNodeCount;
    device float* velocity =
        candidateV + velocityBase;
    float rightHandSide[MR_ARTICULATED_ABA_MAX_DOFS];
    float intermediate[MR_ARTICULATED_ABA_MAX_DOFS];
    float solution[MR_ARTICULATED_ABA_MAX_DOFS];
    float maximumResidual = 0.0f;
    const uint iterations = max(
        dispatch.velocityIterations +
            (pass.reserved0 != 0u
                 ? dispatch.finalVelocityIterations
                 : 0u),
        1u
    );

    for (uint iteration = 0u;
         iteration < iterations;
         ++iteration) {
        for (uint localConstraint = 0u;
             localConstraint <
                 min(
                     status.requiredConstraints,
                     dispatch.authoredConstraintCount
                 );
             ++localConstraint) {
            const uint constraintIndex =
                constraintBase + localConstraint;
            device MRContactConstraintGPU& compatibility =
                contacts[constraintIndex];
            const MRConstraintIRBlockGPU block =
                blocks[constraintIndex];
            if ((block.flags &
                 MR_CONSTRAINT_IR_BLOCK_GENERALIZED) == 0u ||
                (block.flags &
                 MR_CONSTRAINT_IR_BLOCK_DISABLED) != 0u) {
                continue;
            }
            for (uint localRow = 0u;
                 localRow < block.dimension;
                 ++localRow) {
                for (uint dof = 0u;
                     dof < dispatch.nv;
                     ++dof) {
                    rightHandSide[dof] = 0.0f;
                    intermediate[dof] = 0.0f;
                    solution[dof] = 0.0f;
                }
                const MREvaluatedConstraintIRRowGPU row =
                    evaluatedRows[
                        rowBase + block.rowOffset + localRow
                    ];
                const float3 direction =
                    row.direction.xyz;
                float response = row.regularization;
                bool hasArticulation = false;
                for (uint endpointIndex = 0u;
                     endpointIndex < block.endpointCount;
                     ++endpointIndex) {
                    const MRConstraintIREndpointGPU endpoint =
                        endpoints[
                            endpointBase +
                            block.endpointOffset +
                            endpointIndex
                        ];
                    if (endpoint.role ==
                        MR_CONSTRAINT_IR_ENDPOINT_WORLD) {
                        continue;
                    }
                    const uint endpointRow =
                        endpoint.flags &
                        MR_CONSTRAINT_IR_ENDPOINT_ROW_MASK;
                    if (endpointRow >= block.dimension) {
                        status.code = MR_STEP_UNSUPPORTED;
                        status.firstFailingConstraint =
                            localConstraint;
                        statuses[environment] = status;
                        return;
                    }
                    if (endpointRow != localRow) {
                        continue;
                    }
                    const float sign =
                        endpoint.role ==
                            MR_CONSTRAINT_IR_ENDPOINT_A
                        ? -1.0f
                        : 1.0f;
                    if (endpoint.jacobianKind ==
                        MR_CONSTRAINT_IR_JACOBIAN_GENERALIZED) {
                        if (endpoint.objectIndex >= dispatch.nv) {
                            status.code = MR_STEP_UNSUPPORTED;
                            status.firstFailingConstraint =
                                localConstraint;
                            statuses[environment] = status;
                            return;
                        }
                        rightHandSide[
                            endpoint.objectIndex
                        ] += endpoint.axis.x;
                        hasArticulation = true;
                    } else if (
                        endpoint.jacobianKind ==
                        MR_CONSTRAINT_IR_JACOBIAN_ROD_NODE
                    ) {
                        if (endpoint.objectIndex >=
                            dispatch.rodNodeCount) {
                            status.code = MR_STEP_UNSUPPORTED;
                            status.firstFailingConstraint =
                                localConstraint;
                            statuses[environment] = status;
                            return;
                        }
                        response +=
                            inverseRodMasses[
                                endpoint.objectIndex
                            ] * dot(direction, direction);
                    } else if (
                        endpoint.jacobianKind ==
                        MR_CONSTRAINT_IR_JACOBIAN_BODY_LOCAL_POINT
                    ) {
                        if (endpoint.objectIndex >=
                            dispatch.bodyCount) {
                            status.code = MR_STEP_UNSUPPORTED;
                            status.firstFailingConstraint =
                                localConstraint;
                            statuses[environment] = status;
                            return;
                        }
                        device const MRBodyStateGPU& body =
                            candidateBodies[
                                bodyBase + endpoint.objectIndex
                            ];
                        const float3 point =
                            body.position.xyz +
                            multiply(
                                rotationMatrix(
                                    body.orientation
                                ),
                                endpoint.anchor.xyz
                            );
                        const float3 endpointImpulse =
                            sign * direction;
                        response += dot(
                            endpointImpulse,
                            scenePointResponse(
                                body,
                                point,
                                endpointImpulse
                            )
                        );
                    } else {
                        status.code = MR_STEP_UNSUPPORTED;
                        status.firstFailingConstraint =
                            localConstraint;
                        statuses[environment] = status;
                        return;
                    }
                }
                if (hasArticulation &&
                    !solveCholesky(
                        factors,
                        factorBase,
                        dispatch.nv,
                        rightHandSide,
                        intermediate,
                        solution
                    )) {
                    status.code =
                        MR_STEP_FACTORIZATION_FAILED;
                    status.firstFailingConstraint =
                        localConstraint;
                    statuses[environment] = status;
                    return;
                }
                for (uint dof = 0u;
                     dof < dispatch.nv;
                     ++dof) {
                    response = fma(
                        rightHandSide[dof],
                        solution[dof],
                        response
                    );
                }
                const float relative = row.relativeVelocity;
                if (!(response > 1.0e-12f) ||
                    !isfinite(response) ||
                    !isfinite(relative)) {
                    status.code =
                        MR_STEP_FACTORIZATION_FAILED;
                    status.firstFailingConstraint =
                        localConstraint;
                    statuses[environment] = status;
                    return;
                }
                const float previous =
                    compatibility.impulses[localRow];
                const float residual =
                    relative - row.targetVelocity +
                    row.regularization * previous;
                const float proposed = clamp(
                    previous - residual / response,
                    row.impulseLower,
                    row.impulseUpper
                );
                const float delta = proposed - previous;
                if (!isfinite(proposed) ||
                    !isfinite(delta)) {
                    status.code = MR_STEP_NONFINITE_RESULT;
                    status.firstFailingConstraint =
                        localConstraint;
                    statuses[environment] = status;
                    return;
                }
                compatibility.impulses[localRow] = proposed;
                for (uint dof = 0u;
                     dof < dispatch.nv;
                     ++dof) {
                    velocity[dof] = fma(
                        solution[dof],
                        delta,
                        velocity[dof]
                    );
                }
                for (uint endpointIndex = 0u;
                     endpointIndex < block.endpointCount;
                     ++endpointIndex) {
                    const MRConstraintIREndpointGPU endpoint =
                        endpoints[
                            endpointBase +
                            block.endpointOffset +
                            endpointIndex
                        ];
                    if (endpoint.role ==
                        MR_CONSTRAINT_IR_ENDPOINT_WORLD) {
                        continue;
                    }
                    const uint endpointRow =
                        endpoint.flags &
                        MR_CONSTRAINT_IR_ENDPOINT_ROW_MASK;
                    if (endpointRow != localRow) {
                        continue;
                    }
                    const float sign =
                        endpoint.role ==
                            MR_CONSTRAINT_IR_ENDPOINT_A
                        ? -1.0f
                        : 1.0f;
                    if (endpoint.jacobianKind ==
                        MR_CONSTRAINT_IR_JACOBIAN_ROD_NODE) {
                        candidateRodNodes[
                            rodNodeBase +
                            endpoint.objectIndex
                        ].velocity.xyz +=
                            inverseRodMasses[
                                endpoint.objectIndex
                            ] *
                            sign * direction * delta;
                    } else if (
                        endpoint.jacobianKind ==
                        MR_CONSTRAINT_IR_JACOBIAN_BODY_LOCAL_POINT
                    ) {
                        device MRBodyStateGPU& body =
                            candidateBodies[
                                bodyBase + endpoint.objectIndex
                            ];
                        const float3 point =
                            body.position.xyz +
                            multiply(
                                rotationMatrix(
                                    body.orientation
                                ),
                                endpoint.anchor.xyz
                            );
                        applySceneImpulse(
                            body,
                            point,
                            sign * direction * delta
                        );
                    }
                }
                maximumResidual = max(
                    maximumResidual,
                    abs(residual)
                );
            }
        }
    }
    status.solverIterations = max(
        status.solverIterations,
        iterations
    );
    status.residuals.y = max(
        status.residuals.y,
        maximumResidual
    );
    statuses[environment] = status;
}

// Coupled cone velocity solve for one articulation plus arbitrary
// free rigid bodies. The articulation Cholesky factor is reused for all three
// RHS of every contact; no dense inverse is formed.
kernel void mr_world_solve_contact_islands(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const float* factors [[buffer(1)]],
    device const float* pointJacobians [[buffer(2)]],
    device float* candidateV [[buffer(3)]],
    device MRBodyStateGPU* candidateBodies [[buffer(4)]],
    device MRContactConstraintGPU* contacts [[buffer(5)]],
    device const MRContactPointMetaGPU* contactMetadata [[buffer(6)]],
    device const MREvaluatedConstraintIRRowGPU* evaluatedRows [[buffer(7)]],
    device const MREvaluatedConstraintIRConeGPU* evaluatedCones [[buffer(8)]],
    device float* responseColumns [[buffer(9)]],
    device MRManifoldPointGPU* candidateManifoldPoints [[buffer(10)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(11)]],
    constant MRMetalWorldPassGPU& pass [[buffer(12)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    if (status.code != MR_STEP_SUCCESS) {
        return;
    }
    const bool stiffReplay = pass.reserved1 != 0u;
    if (stiffReplay &&
        (status.queueFlags &
         MR_ISLAND_WORK_STIFF_REPLAY) == 0u) {
        return;
    }
    if (status.requiredConstraints == 0u) {
        status.solverIterations = 0u;
        status.residuals = float4(0.0f);
        statuses[environment] = status;
        return;
    }
    if (dispatch.nv > MR_ARTICULATED_ABA_MAX_DOFS ||
        pass.reserved0 > 1u ||
        pass.reserved1 > 1u) {
        status.code = MR_STEP_UNSUPPORTED;
        statuses[environment] = status;
        return;
    }

    const uint constraintBase =
        environment * dispatch.constraintStride;
    const uint rowBase =
        environment * dispatch.rowStride;
    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    const uint velocityBase = environment * dispatch.nv;
    const uint factorBase =
        environment * dispatch.factorStride;
    const uint pointJacobianBase =
        environment *
        (dispatch.pointQueryStride * 3u * dispatch.nv);
    const uint responseBase =
        environment *
        (dispatch.constraintStride * 3u * dispatch.nv);
    const uint manifoldPointBase =
        environment * dispatch.manifoldStride *
        MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;

    float rightHandSide[MR_ARTICULATED_ABA_MAX_DOFS];
    float intermediate[MR_ARTICULATED_ABA_MAX_DOFS];
    float solution[MR_ARTICULATED_ABA_MAX_DOFS];
    device float* articulationVelocity =
        candidateV + velocityBase;
    device MRBodyStateGPU* bodies =
        candidateBodies + bodyBase;

    // Cache M^-1 J^T for normal/u/v.
    for (uint localConstraint = 0u;
         localConstraint < status.requiredConstraints;
         ++localConstraint) {
        device const MRContactConstraintGPU& contact =
            contacts[constraintBase + localConstraint];
        if ((contact.flags &
             (
                 MR_CONSTRAINT_FLAG_ROD_ENDPOINT |
                 MR_CONSTRAINT_FLAG_GENERALIZED
             )) != 0u) {
            continue;
        }
        const bool articulatedA =
            bodies[contact.bodyA].flagsAndIndices[1] !=
                MR_INVALID_INDEX;
        const bool articulatedB =
            bodies[contact.bodyB].flagsAndIndices[1] !=
                MR_INVALID_INDEX;
        for (uint axis = 0u; axis < 3u; ++axis) {
            const float3 direction =
                evaluatedRows[
                    rowBase + 3u * localConstraint + axis
                ].direction.xyz;
            for (uint dof = 0u; dof < dispatch.nv; ++dof) {
                rightHandSide[dof] = dot(
                    direction,
                    combinedJacobianColumn(
                        pointJacobians,
                        pointJacobianBase,
                        localConstraint,
                        dof,
                        dispatch.nv,
                        articulatedA,
                        articulatedB
                    )
                );
                intermediate[dof] = 0.0f;
                solution[dof] = 0.0f;
            }
            if (!solveCholesky(
                    factors,
                    factorBase,
                    dispatch.nv,
                    rightHandSide,
                    intermediate,
                    solution
                )) {
                status.code = MR_STEP_FACTORIZATION_FAILED;
                status.firstFailingConstraint =
                    localConstraint;
                status.firstFailingStableKeyLow =
                    uint(contact.pairKey);
                status.firstFailingStableKeyHigh =
                    uint(contact.pairKey >> 32u);
                statuses[environment] = status;
                return;
            }
            for (uint dof = 0u; dof < dispatch.nv; ++dof) {
                responseColumns[
                    responseBase +
                    (localConstraint * 3u + axis) *
                        dispatch.nv +
                    dof
                ] = solution[dof];
            }
        }
    }

    // Initial ordered solve warm-starts from persistent manifolds. A stiff
    // replay starts from the velocity/impulse state already accepted by the
    // Wave32 sweep, so applying the full warm impulse again would double it.
    if (!stiffReplay) {
        for (uint localConstraint = 0u;
             localConstraint < status.requiredConstraints;
             ++localConstraint) {
            device MRContactConstraintGPU& contact =
                contacts[constraintBase + localConstraint];
            if ((contact.flags &
                 (
                     MR_CONSTRAINT_FLAG_ROD_ENDPOINT |
                     MR_CONSTRAINT_FLAG_GENERALIZED
                 )) != 0u) {
                continue;
            }
            const float3 warm = projectFrictionCone(
                contact.impulses.xyz,
                evaluatedCones[
                    constraintBase + localConstraint
                ]
            );
            contact.impulses.xyz = warm;
            applyContactDelta(
                localConstraint,
                contact,
                warm,
                evaluatedRows +
                    rowBase + 3u * localConstraint,
                articulationVelocity,
                bodies,
                responseColumns,
                responseBase,
                dispatch.nv,
                dispatch.articulationIndex
            );
        }
    }

    float maximumImpulseDelta = 0.0f;
    const uint solverIterations =
        dispatch.velocityIterations +
        (pass.reserved0 != 0u
             ? dispatch.finalVelocityIterations
             : 0u);
    for (uint iteration = 0u;
         iteration < solverIterations;
         ++iteration) {
        for (uint localConstraint = 0u;
             localConstraint < status.requiredConstraints;
             ++localConstraint) {
            device MRContactConstraintGPU& contact =
                contacts[constraintBase + localConstraint];
            if ((contact.flags &
                 (
                     MR_CONSTRAINT_FLAG_ROD_ENDPOINT |
                     MR_CONSTRAINT_FLAG_GENERALIZED
                 )) != 0u) {
                continue;
            }
            device const MREvaluatedConstraintIRRowGPU* localRows =
                evaluatedRows +
                rowBase + 3u * localConstraint;
            const float3 directions[3] = {
                localRows[0].direction.xyz,
                localRows[1].direction.xyz,
                localRows[2].direction.xyz,
            };
            device const MRBodyStateGPU& bodyA =
                bodies[contact.bodyA];
            device const MRBodyStateGPU& bodyB =
                bodies[contact.bodyB];
            const bool articulatedA =
                bodyA.flagsAndIndices[1] !=
                    MR_INVALID_INDEX;
            const bool articulatedB =
                bodyB.flagsAndIndices[1] !=
                    MR_INVALID_INDEX;

            float effective[3][3];
            for (uint row = 0u; row < 3u; ++row) {
                for (uint column = 0u;
                     column < 3u;
                     ++column) {
                    float value = 0.0f;
                    for (uint dof = 0u;
                         dof < dispatch.nv;
                         ++dof) {
                        const float rowJacobian = dot(
                            directions[row],
                            combinedJacobianColumn(
                                pointJacobians,
                                pointJacobianBase,
                                localConstraint,
                                dof,
                                dispatch.nv,
                                articulatedA,
                                articulatedB
                            )
                        );
                        value +=
                            rowJacobian *
                            responseColumns[
                                responseBase +
                                (
                                    localConstraint * 3u +
                                    column
                                ) * dispatch.nv +
                                dof
                            ];
                    }
                    if (dynamicSceneEndpoint(
                            bodyA,
                            dispatch.articulationIndex
                        )) {
                        value += dot(
                            directions[row],
                            -scenePointResponse(
                                bodyA,
                                contact.pointAndSeparation.xyz,
                                -directions[column]
                            )
                        );
                    }
                    if (dynamicSceneEndpoint(
                            bodyB,
                            dispatch.articulationIndex
                        )) {
                        value += dot(
                            directions[row],
                            scenePointResponse(
                                bodyB,
                                contact.pointAndSeparation.xyz,
                                directions[column]
                            )
                        );
                    }
                    if (row == column) {
                        value += localRows[row].regularization;
                    }
                    effective[row][column] = value;
                }
            }
            float inverse[3][3];
            if (!invert3x3(effective, inverse)) {
                status.code = MR_STEP_FACTORIZATION_FAILED;
                status.firstFailingConstraint =
                    localConstraint;
                status.firstFailingStableKeyLow =
                    uint(contact.pairKey);
                status.firstFailingStableKeyHigh =
                    uint(contact.pairKey >> 32u);
                statuses[environment] = status;
                return;
            }

            const float3 relative = relativePointVelocity(
                localConstraint,
                contact,
                bodies,
                dispatch.articulationIndex,
                pointJacobians,
                pointJacobianBase,
                dispatch.nv,
                articulationVelocity
            );
            const float3 previous = contact.impulses.xyz;
            float rhs[3];
            for (uint row = 0u; row < 3u; ++row) {
                rhs[row] =
                    localRows[row].targetVelocity -
                    dot(directions[row], relative) -
                    localRows[row].regularization *
                        previous[row];
            }
            float3 candidate = previous;
            for (uint row = 0u; row < 3u; ++row) {
                for (uint column = 0u;
                     column < 3u;
                     ++column) {
                    candidate[row] +=
                        inverse[row][column] * rhs[column];
                }
            }
            candidate = projectFrictionCone(
                candidate,
                evaluatedCones[
                    constraintBase + localConstraint
                ]
            );
            const float3 delta = candidate - previous;
            if (!finite3(candidate) || !finite3(delta)) {
                status.code = MR_STEP_NONFINITE_RESULT;
                status.firstFailingConstraint =
                    localConstraint;
                status.firstFailingStableKeyLow =
                    uint(contact.pairKey);
                status.firstFailingStableKeyHigh =
                    uint(contact.pairKey >> 32u);
                statuses[environment] = status;
                return;
            }
            maximumImpulseDelta = max(
                maximumImpulseDelta,
                max(
                    abs(delta.x),
                    max(abs(delta.y), abs(delta.z))
                )
            );
            contact.impulses.xyz = candidate;
            applyContactDelta(
                localConstraint,
                contact,
                delta,
                localRows,
                articulationVelocity,
                bodies,
                responseColumns,
                responseBase,
                dispatch.nv,
                dispatch.articulationIndex
            );
        }
    }

    float maximumNormalResidual = 0.0f;
    float maximumConeViolation = 0.0f;
    for (uint localConstraint = 0u;
         localConstraint < status.requiredConstraints;
         ++localConstraint) {
        device MRContactConstraintGPU& contact =
            contacts[constraintBase + localConstraint];
        if ((contact.flags &
             (
                 MR_CONSTRAINT_FLAG_ROD_ENDPOINT |
                 MR_CONSTRAINT_FLAG_GENERALIZED
             )) != 0u) {
            continue;
        }
        device const MREvaluatedConstraintIRRowGPU* localRows =
            evaluatedRows +
            rowBase + 3u * localConstraint;
        const float3 relative = relativePointVelocity(
            localConstraint,
            contact,
            bodies,
            dispatch.articulationIndex,
            pointJacobians,
            pointJacobianBase,
            dispatch.nv,
            articulationVelocity
        );
        const float normalEquation =
            dot(localRows[0].direction.xyz, relative) -
            localRows[0].targetVelocity +
            localRows[0].regularization *
                contact.impulses.x;
        const float normalResidual =
            contact.impulses.x > kConeEpsilon
            ? abs(normalEquation)
            : max(-normalEquation, 0.0f);
        maximumNormalResidual = max(
            maximumNormalResidual,
            normalResidual
        );
        device const MREvaluatedConstraintIRConeGPU& cone =
            evaluatedCones[constraintBase + localConstraint];
        const float limitU =
            cone.effectiveFrictionU * contact.impulses.x;
        const float limitV =
            cone.effectiveFrictionV * contact.impulses.x;
        float coneViolation = 0.0f;
        if (limitU > 0.0f && limitV > 0.0f) {
            coneViolation = max(
                sqrt(
                    (contact.impulses.y *
                     contact.impulses.y) /
                        (limitU * limitU) +
                    (contact.impulses.z *
                     contact.impulses.z) /
                        (limitV * limitV)
                ) - 1.0f,
                0.0f
            );
        } else {
            coneViolation = length(contact.impulses.yz);
        }
        maximumConeViolation = max(
            maximumConeViolation,
            coneViolation
        );

        const MRContactPointMetaGPU metadata =
            contactMetadata[constraintBase + localConstraint];
        if (metadata.manifoldIndex <
                dispatch.manifoldCapacity &&
            metadata.pointIndex <
                MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY) {
            candidateManifoldPoints[
                manifoldPointBase +
                metadata.manifoldIndex *
                    MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY +
                metadata.pointIndex
            ].impulses = contact.impulses;
        }
    }
    if (!isfinite(maximumImpulseDelta) ||
        !isfinite(maximumNormalResidual) ||
        !isfinite(maximumConeViolation)) {
        status.code = MR_STEP_NONFINITE_RESULT;
        statuses[environment] = status;
        return;
    }
    status.solverIterations = solverIterations;
    status.residuals.x = maximumImpulseDelta;
    status.residuals.y = maximumNormalResidual;
    status.residuals.z = maximumConeViolation;
    if (stiffReplay) {
        status.queueFlags &=
            ~MR_ISLAND_WORK_STIFF_REPLAY;
    }
    statuses[environment] = status;
}

inline void rodSurfaceImpulseTranspose(
    device MRRodNodeStateGPU& nodeA,
    device MRRodNodeStateGPU& nodeB,
    device MRRodEdgeStateGPU& edgeState,
    const float inverseMassA,
    const float inverseMassB,
    const float inverseTwistInertia,
    const float weight,
    const float3 radial,
    const float radius,
    const float3 impulse
) {
    const float3 edge =
        nodeB.position.xyz - nodeA.position.xyz;
    const float lengthSquared = dot(edge, edge);
    if (!(lengthSquared > 1.0e-20f)) {
        return;
    }
    const float length = sqrt(lengthSquared);
    const float3 tangent = edge / length;
    const float3 surfaceRadius = radius * radial;
    const float3 angularImpulse =
        cross(surfaceRadius, impulse);
    const float3 bendTranspose =
        (
            cross(angularImpulse, tangent) -
            tangent *
                dot(
                    tangent,
                    cross(angularImpulse, tangent)
                )
        ) / length;
    const float3 forceA =
        (1.0f - weight) * impulse - bendTranspose;
    const float3 forceB =
        weight * impulse + bendTranspose;
    nodeA.velocity.xyz += inverseMassA * forceA;
    nodeB.velocity.xyz += inverseMassB * forceB;
    edgeState.twistAndRate.y +=
        inverseTwistInertia *
        dot(cross(tangent, surfaceRadius), impulse);
}

// Ordered rod blocks consume the same IR, rows, cones, factor cache and
// articulation Jacobians as Wave32. The current rod solveA is the positive
// diagonal of the implicit DER operator; the endpoint Jacobian and transpose
// are exact for the frozen microstep, including bending-frame and twist
// surface velocity. Replacing solveA with the retained band factor therefore
// does not change contact semantics or island composition.
kernel void mr_world_solve_rod_contact_constraints(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const float* factors [[buffer(1)]],
    device const float* pointJacobians [[buffer(2)]],
    device float* candidateV [[buffer(3)]],
    device MRBodyStateGPU* candidateBodies [[buffer(4)]],
    device MRRodNodeStateGPU* candidateRodNodes [[buffer(5)]],
    device MRRodEdgeStateGPU* candidateRodEdges [[buffer(6)]],
    device const float* inverseRodMasses [[buffer(7)]],
    device const float* inverseRodTwistInertias [[buffer(8)]],
    device const MRRodColliderGPU* rodColliders [[buffer(9)]],
    device MRContactConstraintGPU* contacts [[buffer(10)]],
    device const MRConstraintIRBlockGPU* blocks [[buffer(11)]],
    device const MREvaluatedConstraintIRRowGPU* evaluatedRows [[buffer(12)]],
    device const MREvaluatedConstraintIRConeGPU* evaluatedCones [[buffer(13)]],
    device float* responseColumns [[buffer(14)]],
    device MRRodToolWitnessGPU* witnesses [[buffer(15)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(16)]],
    constant MRMetalWorldPassGPU& pass [[buffer(17)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    if (status.code != MR_STEP_SUCCESS ||
        dispatch.rodToolPairCount == 0u ||
        dispatch.nv > MR_ARTICULATED_ABA_MAX_DOFS) {
        return;
    }
    const uint constraintBase =
        environment * dispatch.constraintStride;
    const uint rowBase =
        environment * dispatch.rowStride;
    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    const uint nodeBase =
        environment * dispatch.rodNodeCount;
    const uint edgeBase =
        environment * dispatch.rodEdgeCount;
    const uint velocityBase = environment * dispatch.nv;
    const uint factorBase =
        environment * dispatch.factorStride;
    const uint pointJacobianBase =
        environment *
        (dispatch.pointQueryStride * 3u * dispatch.nv);
    const uint responseBase =
        environment *
        (dispatch.constraintStride * 3u * dispatch.nv);
    float rightHandSide[MR_ARTICULATED_ABA_MAX_DOFS];
    float intermediate[MR_ARTICULATED_ABA_MAX_DOFS];
    float solution[MR_ARTICULATED_ABA_MAX_DOFS];
    device float* articulationVelocity =
        candidateV + velocityBase;
    device MRBodyStateGPU* bodies =
        candidateBodies + bodyBase;

    // Cache articulation response columns for the tool endpoint. Rod and
    // maximal-body responses remain direct matrix-free operations.
    for (uint localConstraint = 0u;
         localConstraint < status.requiredConstraints;
         ++localConstraint) {
        device const MRContactConstraintGPU& contact =
            contacts[constraintBase + localConstraint];
        if ((contact.flags &
             MR_CONSTRAINT_FLAG_ROD_ENDPOINT) == 0u) {
            continue;
        }
        const MRConstraintIRBlockGPU block =
            blocks[constraintBase + localConstraint];
        const MRRodToolWitnessGPU witness =
            witnesses[block.reserved0];
        device const MRBodyStateGPU& tool =
            bodies[witness.featuresAndFlags.z];
        const bool articulated =
            tool.flagsAndIndices[1] != MR_INVALID_INDEX;
        for (uint axis = 0u; axis < 3u; ++axis) {
            const float3 direction =
                evaluatedRows[
                    rowBase + block.rowOffset + axis
                ].direction.xyz;
            for (uint dof = 0u;
                 dof < dispatch.nv;
                 ++dof) {
                rightHandSide[dof] =
                    articulated
                    ? dot(
                          direction,
                          rodToolJacobianColumn(
                              pointJacobians,
                              pointJacobianBase,
                              localConstraint,
                              dof,
                              dispatch.nv
                          )
                      )
                    : 0.0f;
                intermediate[dof] = 0.0f;
                solution[dof] = 0.0f;
            }
            if (articulated &&
                !solveCholesky(
                    factors,
                    factorBase,
                    dispatch.nv,
                    rightHandSide,
                    intermediate,
                    solution
                )) {
                status.code = MR_STEP_FACTORIZATION_FAILED;
                status.firstFailingConstraint =
                    localConstraint;
                status.firstFailingStableKeyLow =
                    uint(contact.pairKey);
                status.firstFailingStableKeyHigh =
                    uint(contact.pairKey >> 32u);
                statuses[environment] = status;
                return;
            }
            for (uint dof = 0u;
                 dof < dispatch.nv;
                 ++dof) {
                responseColumns[
                    responseBase +
                    (localConstraint * 3u + axis) *
                        dispatch.nv +
                    dof
                ] = solution[dof];
            }
        }
    }

    float maximumImpulseDelta = 0.0f;
    const uint outerIterations = max(
        dispatch.rodContactOuterIterations,
        1u
    );
    for (uint iteration = 0u;
         iteration < outerIterations;
         ++iteration) {
        for (uint localConstraint = 0u;
             localConstraint < status.requiredConstraints;
             ++localConstraint) {
            device MRContactConstraintGPU& contact =
                contacts[constraintBase + localConstraint];
            if ((contact.flags &
                 MR_CONSTRAINT_FLAG_ROD_ENDPOINT) == 0u) {
                continue;
            }
            const MRConstraintIRBlockGPU block =
                blocks[constraintBase + localConstraint];
            device MRRodToolWitnessGPU& witness =
                witnesses[block.reserved0];
            const uint rodColliderIndex =
                witness.identity.z;
            device const MRRodColliderGPU& collider =
                rodColliders[rodColliderIndex];
            device MRRodNodeStateGPU& nodeA =
                candidateRodNodes[
                    nodeBase + collider.nodeA
                ];
            device MRRodNodeStateGPU& nodeB =
                candidateRodNodes[
                    nodeBase + collider.nodeB
                ];
            device MRRodEdgeStateGPU& rodEdge =
                candidateRodEdges[
                    edgeBase + collider.edgeIndex
                ];
            device MRBodyStateGPU& tool =
                bodies[witness.featuresAndFlags.z];
            const float weight =
                witness.rodPointAndWeight.w;
            const float3 radial =
                witness.radialAndTwistJacobianV.xyz;
            const float radius =
                collider.radiusAndOffsets.x;
            const bool articulated =
                tool.flagsAndIndices[1] != MR_INVALID_INDEX;
            const float3 directions[3] = {
                evaluatedRows[
                    rowBase + block.rowOffset + 0u
                ].direction.xyz,
                evaluatedRows[
                    rowBase + block.rowOffset + 1u
                ].direction.xyz,
                evaluatedRows[
                    rowBase + block.rowOffset + 2u
                ].direction.xyz,
            };
            if (iteration == 0u) {
                MREvaluatedConstraintIRConeGPU warmCone =
                    evaluatedCones[
                        constraintBase + localConstraint
                    ];
                const float3 warm =
                    projectFrictionCone(
                        contact.impulses.xyz,
                        warmCone
                    );
                const float3 warmImpulse =
                    directions[0] * warm.x +
                    directions[1] * warm.y +
                    directions[2] * warm.z;
                rodSurfaceImpulseTranspose(
                    nodeA,
                    nodeB,
                    rodEdge,
                    inverseRodMasses[collider.nodeA],
                    inverseRodMasses[collider.nodeB],
                    inverseRodTwistInertias[
                        collider.edgeIndex
                    ],
                    weight,
                    radial,
                    radius,
                    -warmImpulse
                );
                if (articulated) {
                    for (uint dof = 0u;
                         dof < dispatch.nv;
                         ++dof) {
                        articulationVelocity[dof] +=
                            responseColumns[
                                responseBase +
                                (
                                    localConstraint * 3u +
                                    0u
                                ) * dispatch.nv +
                                dof
                            ] * warm.x +
                            responseColumns[
                                responseBase +
                                (
                                    localConstraint * 3u +
                                    1u
                                ) * dispatch.nv +
                                dof
                            ] * warm.y +
                            responseColumns[
                                responseBase +
                                (
                                    localConstraint * 3u +
                                    2u
                                ) * dispatch.nv +
                                dof
                            ] * warm.z;
                    }
                } else {
                    applySceneImpulse(
                        tool,
                        witness.toolPointAndSeparation.xyz,
                        warmImpulse
                    );
                }
                contact.impulses.xyz = warm;
            }
            const float3 rodVelocity =
                rodSurfaceVelocity(
                    nodeA,
                    nodeB,
                    rodEdge,
                    weight,
                    radial,
                    radius
                );
            const float3 toolVelocity =
                articulated
                ? articulatedPointVelocity(
                      pointJacobians,
                      pointJacobianBase,
                      2u * localConstraint + 1u,
                      dispatch.nv,
                      articulationVelocity
                  )
                : pointVelocity(
                      tool,
                      witness.toolPointAndSeparation.xyz
                  );
            const float3 relative =
                toolVelocity - rodVelocity;
            const float relativeRows[3] = {
                dot(directions[0], relative),
                dot(directions[1], relative),
                dot(directions[2], relative),
            };

            float effective[3][3];
            for (uint row = 0u; row < 3u; ++row) {
                for (uint column = 0u;
                     column < 3u;
                     ++column) {
                    const float3 rodResponse =
                        rodSurfaceResponse(
                            nodeA,
                            nodeB,
                            rodEdge,
                            inverseRodMasses[
                                collider.nodeA
                            ],
                            inverseRodMasses[
                                collider.nodeB
                            ],
                            inverseRodTwistInertias[
                                collider.edgeIndex
                            ],
                            weight,
                            radial,
                            radius,
                            directions[column]
                        );
                    float3 toolResponse =
                        float3(0.0f);
                    if (articulated) {
                        for (uint dof = 0u;
                             dof < dispatch.nv;
                             ++dof) {
                            toolResponse +=
                                rodToolJacobianColumn(
                                    pointJacobians,
                                    pointJacobianBase,
                                    localConstraint,
                                    dof,
                                    dispatch.nv
                                ) *
                                responseColumns[
                                    responseBase +
                                    (
                                        localConstraint * 3u +
                                        column
                                    ) * dispatch.nv +
                                    dof
                                ];
                        }
                    } else {
                        toolResponse = scenePointResponse(
                            tool,
                            witness.toolPointAndSeparation.xyz,
                            directions[column]
                        );
                    }
                    float value = dot(
                        directions[row],
                        toolResponse + rodResponse
                    );
                    if (row == column) {
                        value +=
                            evaluatedRows[
                                rowBase +
                                block.rowOffset + row
                            ].regularization;
                    }
                    effective[row][column] = value;
                }
            }
            float inverse[3][3];
            if (!invert3x3(effective, inverse)) {
                status.code = MR_STEP_FACTORIZATION_FAILED;
                status.firstFailingConstraint =
                    localConstraint;
                status.firstFailingStableKeyLow =
                    uint(contact.pairKey);
                status.firstFailingStableKeyHigh =
                    uint(contact.pairKey >> 32u);
                statuses[environment] = status;
                return;
            }
            const float3 previous = contact.impulses.xyz;
            float target[3];
            for (uint row = 0u; row < 3u; ++row) {
                const MREvaluatedConstraintIRRowGPU evaluated =
                    evaluatedRows[
                        rowBase + block.rowOffset + row
                    ];
                target[row] = evaluated.targetVelocity;
                if (row == 0u &&
                    iteration == 0u &&
                    (contact.flags &
                     MR_CONSTRAINT_FLAG_NEW_IMPACT) != 0u &&
                    witness
                            .normalAndPreSolveVelocity.w <
                        -evaluatedCones[
                             constraintBase +
                             localConstraint
                         ].restitutionThreshold) {
                    target[row] = max(
                        target[row],
                        -contact.response.x *
                            witness
                                .normalAndPreSolveVelocity.w
                    );
                }
            }
            float rhs[3];
            for (uint row = 0u; row < 3u; ++row) {
                rhs[row] =
                    target[row] -
                    relativeRows[row] -
                    evaluatedRows[
                        rowBase + block.rowOffset + row
                    ].regularization *
                        previous[row];
            }
            float3 proposed = previous;
            for (uint row = 0u; row < 3u; ++row) {
                for (uint column = 0u;
                     column < 3u;
                     ++column) {
                    proposed[row] +=
                        inverse[row][column] * rhs[column];
                }
            }
            MREvaluatedConstraintIRConeGPU cone =
                evaluatedCones[
                    constraintBase + localConstraint
                ];
            const float slip =
                length(float2(relativeRows[1], relativeRows[2]));
            cone.effectiveFrictionU =
                slip <= 1.0e-3f
                ? cone.staticFrictionU
                : cone.dynamicFrictionU;
            cone.effectiveFrictionV =
                slip <= 1.0e-3f
                ? cone.staticFrictionV
                : cone.dynamicFrictionV;
            proposed = projectFrictionCone(proposed, cone);
            const float3 delta = proposed - previous;
            if (!finite3(proposed) || !finite3(delta)) {
                status.code = MR_STEP_NONFINITE_RESULT;
                status.firstFailingConstraint =
                    localConstraint;
                statuses[environment] = status;
                return;
            }
            const float3 worldImpulse =
                directions[0] * delta.x +
                directions[1] * delta.y +
                directions[2] * delta.z;
            rodSurfaceImpulseTranspose(
                nodeA,
                nodeB,
                rodEdge,
                inverseRodMasses[collider.nodeA],
                inverseRodMasses[collider.nodeB],
                inverseRodTwistInertias[
                    collider.edgeIndex
                ],
                weight,
                radial,
                radius,
                -worldImpulse
            );
            if (articulated) {
                for (uint dof = 0u;
                     dof < dispatch.nv;
                     ++dof) {
                    articulationVelocity[dof] +=
                        responseColumns[
                            responseBase +
                            (localConstraint * 3u + 0u) *
                                dispatch.nv +
                            dof
                        ] * delta.x +
                        responseColumns[
                            responseBase +
                            (localConstraint * 3u + 1u) *
                                dispatch.nv +
                            dof
                        ] * delta.y +
                        responseColumns[
                            responseBase +
                            (localConstraint * 3u + 2u) *
                                dispatch.nv +
                            dof
                        ] * delta.z;
                }
            } else {
                applySceneImpulse(
                    tool,
                    witness.toolPointAndSeparation.xyz,
                    worldImpulse
                );
            }
            contact.impulses.xyz = proposed;
            witness.impulses.xyz = proposed;
            maximumImpulseDelta = max(
                maximumImpulseDelta,
                max(
                    abs(delta.x),
                    max(abs(delta.y), abs(delta.z))
                )
            );
        }
    }
    status.residuals.x = max(
        status.residuals.x,
        maximumImpulseDelta
    );
    status.solverIterations = max(
        status.solverIterations,
        outerIterations
    );
    (void)pass;
    statuses[environment] = status;
}

// Integrates only after constrained velocities are available.
kernel void mr_world_integrate_contact_state(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRArticulationGPU* articulations [[buffer(1)]],
    device const MRJointDescriptorGPU* joints [[buffer(2)]],
    device const MRBodyPropertiesGPU* bodyProperties [[buffer(3)]],
    device const uint* sceneBodyIndices [[buffer(4)]],
    device const float* sourceQ [[buffer(5)]],
    device const float* candidateV [[buffer(6)]],
    device float* candidateQ [[buffer(7)]],
    device MRBodyStateGPU* candidateBodies [[buffer(8)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(9)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    if (status.code != MR_STEP_SUCCESS) {
        return;
    }
    const float timestep = dispatch.timestepAndBias.x;
    const uint qBase = environment * dispatch.qStride;
    const uint vBase = environment * dispatch.vStride;
    for (uint q = 0u; q < dispatch.nq; ++q) {
        candidateQ[qBase + q] = sourceQ[qBase + q];
    }
    for (uint owner = 0u;
         owner < dispatch.articulationCount;
         ++owner) {
        const MRArticulationGPU articulation =
            articulations[owner];
        const uint articulationQ =
            qBase + articulation.qOffset;
        const uint articulationV =
            vBase + articulation.vOffset;
        if (articulation.rootType == MR_ROOT_FLOATING) {
            candidateQ[articulationQ + 0u] =
                sourceQ[articulationQ + 0u] +
                timestep * candidateV[articulationV + 0u];
            candidateQ[articulationQ + 1u] =
                sourceQ[articulationQ + 1u] +
                timestep * candidateV[articulationV + 1u];
            candidateQ[articulationQ + 2u] =
                sourceQ[articulationQ + 2u] +
                timestep * candidateV[articulationV + 2u];
            const float3 rotationVector =
                timestep * float3(
                    candidateV[articulationV + 3u],
                    candidateV[articulationV + 4u],
                    candidateV[articulationV + 5u]
                );
            const float angle = length(rotationVector);
            const float halfAngle = 0.5f * angle;
            const float scale = angle > 1.0e-6f
                ? sin(halfAngle) / angle
                : 0.5f - angle * angle / 48.0f;
            const float4 increment = float4(
                rotationVector * scale,
                cos(halfAngle)
            );
            float4 orientation;
            if (!normalizedQuaternion(
                    quaternionMultiply(
                        increment,
                        float4(
                            sourceQ[articulationQ + 3u],
                            sourceQ[articulationQ + 4u],
                            sourceQ[articulationQ + 5u],
                            sourceQ[articulationQ + 6u]
                        )
                    ),
                    orientation
                )) {
                status.code = MR_STEP_NONFINITE_RESULT;
                status.firstFailingConstraint =
                    articulation.qOffset + 3u;
                statuses[environment] = status;
                return;
            }
            candidateQ[articulationQ + 3u] = orientation.x;
            candidateQ[articulationQ + 4u] = orientation.y;
            candidateQ[articulationQ + 5u] = orientation.z;
            candidateQ[articulationQ + 6u] = orientation.w;
        }
        for (uint localJoint = 0u;
             localJoint < articulation.jointCount;
             ++localJoint) {
            const MRJointDescriptorGPU joint =
                joints[
                    articulation.firstJoint + localJoint
                ];
            if (joint.nv == 1u) {
                candidateQ[qBase + joint.qOffset] =
                    sourceQ[qBase + joint.qOffset] +
                    timestep *
                        candidateV[vBase + joint.vOffset];
            }
        }
    }

    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    for (uint localScene = 0u;
         localScene < dispatch.sceneBodyCount;
         ++localScene) {
        const uint globalBody = sceneBodyIndices[localScene];
        device MRBodyStateGPU& state =
            candidateBodies[bodyBase + globalBody];
        device const MRBodyPropertiesGPU& properties =
            bodyProperties[globalBody];
        if (properties.motionType == MR_MOTION_STATIC) {
            continue;
        }
        const float maxLinear =
            properties.dampingAndSpeedLimits.z;
        const float maxAngular =
            properties.dampingAndSpeedLimits.w;
        float3 linear = state.linearVelocityAndInverseMass.xyz;
        float3 angular = state.angularVelocity.xyz;
        const float linearLength = length(linear);
        const float angularLength = length(angular);
        if (maxLinear > 0.0f && linearLength > maxLinear) {
            linear *= maxLinear / linearLength;
        }
        if (maxAngular > 0.0f && angularLength > maxAngular) {
            angular *= maxAngular / angularLength;
        }
        state.linearVelocityAndInverseMass.xyz = linear;
        state.angularVelocity.xyz = angular;
        state.position.xyz += timestep * linear;
        state.position.w = 1.0f;
        const float3 rotationVector = timestep * angular;
        const float angle = length(rotationVector);
        const float halfAngle = 0.5f * angle;
        const float scale = angle > 1.0e-6f
            ? sin(halfAngle) / angle
            : 0.5f - angle * angle / 48.0f;
        const float4 increment = float4(
            rotationVector * scale,
            cos(halfAngle)
        );
        float4 orientation;
        if (!normalizedQuaternion(
                quaternionMultiply(
                    increment,
                    state.orientation
                ),
                orientation
            ) ||
            !finite4(state.position)) {
            status.code = MR_STEP_NONFINITE_RESULT;
            status.firstFailingConstraint = globalBody;
            statuses[environment] = status;
            return;
        }
        state.orientation = orientation;
        MRBodyStateGPU updatedState = state;
        if (!writeWorldInverseInertia(
                updatedState,
                properties,
                orientation
            )) {
            status.code = MR_STEP_NONFINITE_RESULT;
            status.firstFailingConstraint = globalBody;
            statuses[environment] = status;
            return;
        }
        state = updatedState;
    }
    status.eventTimes.x = timestep;
    status.eventTimes.y = 0.0f;
    statuses[environment] = status;
}

// Converts a contact-stage failure into the existing world transaction latch.
kernel void mr_world_latch_contact_status(
    device const MRMetalWorldDispatchGPU& worldDispatch [[buffer(0)]],
    constant MRMetalWorldPassGPU& pass [[buffer(1)]],
    device const MRMetalWorldContactStatusGPU* contactStatuses [[buffer(2)]],
    device MRMetalWorldStatusGPU* worldStatuses [[buffer(3)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= worldDispatch.environmentCount) {
        return;
    }
    MRMetalWorldStatusGPU status = worldStatuses[environment];
    const MRMetalWorldContactStatusGPU contact =
        contactStatuses[environment];
    if (status.code == MR_STEP_SUCCESS &&
        (
            contact.code != MR_STEP_SUCCESS ||
            contact.environment != environment ||
            contact.controlStep != pass.controlStep ||
            contact.physicsSubstep != pass.physicsSubstep
        )) {
        status.code =
            contact.code == MR_STEP_SUCCESS
            ? MR_STEP_UNSUPPORTED
            : contact.code;
        status.failingSubstep = pass.physicsSubstep;
        status.failingIndex =
            contact.firstFailingConstraint != MR_INVALID_INDEX
            ? contact.firstFailingConstraint
            : contact.firstFailingPair;
        worldStatuses[environment] = status;
    }
}

// Finished environments are masked with MR_STEP_FIXED_BUDGET_COMPLETE while
// later statically encoded event passes run. Restore their source state into
// the ordinary candidate buffers so the final transactional commit remains
// branch-free and never publishes stale scratch.
kernel void mr_world_restore_inactive_event_candidate(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRArticulationGPU* articulations [[buffer(1)]],
    device const uint* sceneBodyIndices [[buffer(2)]],
    device const float* sourceQ [[buffer(3)]],
    device const float* sourceV [[buffer(4)]],
    device const MRBodyStateGPU* sourceScene [[buffer(5)]],
    device const MRManifoldHeaderGPU* sourceHeaders [[buffer(6)]],
    device const MRManifoldPointGPU* sourcePoints [[buffer(7)]],
    device const uint* sourceCounts [[buffer(8)]],
    device float* candidateQ [[buffer(9)]],
    device float* candidateV [[buffer(10)]],
    device MRBodyStateGPU* candidateBodies [[buffer(11)]],
    device MRManifoldHeaderGPU* candidateHeaders [[buffer(12)]],
    device MRManifoldPointGPU* candidatePoints [[buffer(13)]],
    device uint* candidateCounts [[buffer(14)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(15)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount ||
        statuses[environment].code !=
            MR_STEP_FIXED_BUDGET_COMPLETE) {
        return;
    }
    (void)articulations;
    const uint qBase = environment * dispatch.qStride;
    const uint vBase = environment * dispatch.vStride;
    for (uint coordinate = 0u;
         coordinate < dispatch.nq;
         ++coordinate) {
        candidateQ[qBase + coordinate] =
            sourceQ[qBase + coordinate];
    }
    for (uint coordinate = 0u;
         coordinate < dispatch.nv;
         ++coordinate) {
        candidateV[vBase + coordinate] =
            sourceV[vBase + coordinate];
    }
    const uint sceneBase =
        environment * dispatch.sceneBodyStride;
    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    for (uint localScene = 0u;
         localScene < dispatch.sceneBodyCount;
         ++localScene) {
        candidateBodies[
            bodyBase + sceneBodyIndices[localScene]
        ] = sourceScene[sceneBase + localScene];
    }
    const uint manifoldBase =
        environment * dispatch.manifoldStride;
    const uint pointBase =
        manifoldBase *
        MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
    candidateCounts[environment] =
        sourceCounts[environment];
    for (uint manifold = 0u;
         manifold < dispatch.manifoldCapacity;
         ++manifold) {
        candidateHeaders[manifoldBase + manifold] =
            sourceHeaders[manifoldBase + manifold];
        for (uint point = 0u;
             point <
                 MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
             ++point) {
            const uint index =
                pointBase +
                manifold *
                    MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY +
                point;
            candidatePoints[index] = sourcePoints[index];
        }
    }
}

// Advances the private event-state ping-pong between statically encoded TOI
// passes. Failed candidates copy their last accepted input, preserving the
// control-step checkpoint for the final public rollback.
kernel void mr_world_publish_event_segment(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRArticulationGPU* articulations [[buffer(1)]],
    device const uint* sceneBodyIndices [[buffer(2)]],
    device const float* sourceQ [[buffer(3)]],
    device const float* sourceV [[buffer(4)]],
    device const MRBodyStateGPU* sourceScene [[buffer(5)]],
    device const MRManifoldHeaderGPU* sourceHeaders [[buffer(6)]],
    device const MRManifoldPointGPU* sourcePoints [[buffer(7)]],
    device const uint* sourceCounts [[buffer(8)]],
    device const float* candidateQ [[buffer(9)]],
    device const float* candidateV [[buffer(10)]],
    device const MRBodyStateGPU* candidateBodies [[buffer(11)]],
    device const MRManifoldHeaderGPU* candidateHeaders [[buffer(12)]],
    device const MRManifoldPointGPU* candidatePoints [[buffer(13)]],
    device const uint* candidateCounts [[buffer(14)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(15)]],
    device float* destinationQ [[buffer(16)]],
    device float* destinationV [[buffer(17)]],
    device MRBodyStateGPU* destinationScene [[buffer(18)]],
    device MRManifoldHeaderGPU* destinationHeaders [[buffer(19)]],
    device MRManifoldPointGPU* destinationPoints [[buffer(20)]],
    device uint* destinationCounts [[buffer(21)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    const bool publish =
        statuses[environment].code == MR_STEP_SUCCESS;
    (void)articulations;
    const uint qBase = environment * dispatch.qStride;
    const uint vBase = environment * dispatch.vStride;
    for (uint coordinate = 0u;
         coordinate < dispatch.nq;
         ++coordinate) {
        destinationQ[qBase + coordinate] =
            publish
            ? candidateQ[qBase + coordinate]
            : sourceQ[qBase + coordinate];
    }
    for (uint coordinate = 0u;
         coordinate < dispatch.nv;
         ++coordinate) {
        destinationV[vBase + coordinate] =
            publish
            ? candidateV[vBase + coordinate]
            : sourceV[vBase + coordinate];
    }
    const uint sceneBase =
        environment * dispatch.sceneBodyStride;
    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    for (uint localScene = 0u;
         localScene < dispatch.sceneBodyCount;
         ++localScene) {
        destinationScene[sceneBase + localScene] =
            publish
            ? candidateBodies[
                  bodyBase + sceneBodyIndices[localScene]
              ]
            : sourceScene[sceneBase + localScene];
    }
    const uint manifoldBase =
        environment * dispatch.manifoldStride;
    const uint pointBase =
        manifoldBase *
        MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
    destinationCounts[environment] =
        publish
        ? candidateCounts[environment]
        : sourceCounts[environment];
    for (uint manifold = 0u;
         manifold < dispatch.manifoldCapacity;
         ++manifold) {
        destinationHeaders[manifoldBase + manifold] =
            publish
            ? candidateHeaders[manifoldBase + manifold]
            : sourceHeaders[manifoldBase + manifold];
        for (uint point = 0u;
             point <
                 MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
             ++point) {
            const uint index =
                pointBase +
                manifold *
                    MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY +
                point;
            destinationPoints[index] =
                publish
                ? candidatePoints[index]
                : sourcePoints[index];
        }
    }
}

// Publishes or rolls back free-body and manifold state after the q/v commit.
kernel void mr_world_commit_contact_state(
    device const MRMetalWorldDispatchGPU& worldDispatch [[buffer(0)]],
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(1)]],
    constant MRMetalWorldPassGPU& pass [[buffer(2)]],
    device const MRMetalWorldStatusGPU* worldStatuses [[buffer(3)]],
    device const uint* sceneBodyIndices [[buffer(4)]],
    device const MRBodyStateGPU* candidateBodies [[buffer(5)]],
    device const MRBodyStateGPU* checkpointScene [[buffer(6)]],
    device MRBodyStateGPU* destinationScene [[buffer(7)]],
    device const MRManifoldHeaderGPU* candidateHeaders [[buffer(8)]],
    device const MRManifoldPointGPU* candidatePoints [[buffer(9)]],
    device const uint* candidateCounts [[buffer(10)]],
    device const MRManifoldHeaderGPU* checkpointHeaders [[buffer(11)]],
    device const MRManifoldPointGPU* checkpointPoints [[buffer(12)]],
    device const uint* checkpointCounts [[buffer(13)]],
    device MRManifoldHeaderGPU* destinationHeaders [[buffer(14)]],
    device MRManifoldPointGPU* destinationPoints [[buffer(15)]],
    device uint* destinationCounts [[buffer(16)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    const bool publish =
        pass.physicsSubstep < worldDispatch.physicsSubsteps &&
        worldStatuses[environment].code == MR_STEP_SUCCESS;
    const uint sceneBase =
        environment * dispatch.sceneBodyStride;
    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    for (uint localScene = 0u;
         localScene < dispatch.sceneBodyCount;
         ++localScene) {
        destinationScene[sceneBase + localScene] =
            publish
            ? candidateBodies[
                  bodyBase + sceneBodyIndices[localScene]
              ]
            : checkpointScene[sceneBase + localScene];
    }
    const uint manifoldBase =
        environment * dispatch.manifoldStride;
    const uint pointBase =
        manifoldBase *
        MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
    destinationCounts[environment] = publish
        ? candidateCounts[environment]
        : checkpointCounts[environment];
    for (uint manifold = 0u;
         manifold < dispatch.manifoldCapacity;
         ++manifold) {
        destinationHeaders[manifoldBase + manifold] =
            publish
            ? candidateHeaders[manifoldBase + manifold]
            : checkpointHeaders[manifoldBase + manifold];
        for (uint point = 0u;
             point <
                 MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
             ++point) {
            const uint index =
                pointBase +
                manifold *
                    MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY +
                point;
            destinationPoints[index] = publish
                ? candidatePoints[index]
                : checkpointPoints[index];
        }
    }
}

// Convex/simplex caches are performance-semantic state: they influence query
// iteration order and therefore must obey the same transaction as physics.
// Only the final successful microstep publishes active convex/mesh entries.
kernel void mr_world_publish_convex_cache(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRCompiledCollisionPairGPU* eligiblePairs [[buffer(1)]],
    device const uint* overlapFlags [[buffer(2)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(3)]],
    device const MRConvexQueryCacheGPU* candidateCaches [[buffer(4)]],
    device MRConvexQueryCacheGPU* publishedCaches [[buffer(5)]],
    const uint globalIndex [[thread_position_in_grid]]
) {
    const uint total =
        dispatch.environmentCount * dispatch.eligiblePairCount;
    if (globalIndex >= total) {
        return;
    }
    const uint environment =
        globalIndex / dispatch.eligiblePairCount;
    const uint compiledPair =
        globalIndex -
        environment * dispatch.eligiblePairCount;
    const uint pairClass =
        eligiblePairs[compiledPair].pairClass;
    if (statuses[environment].code == MR_STEP_SUCCESS &&
        overlapFlags[globalIndex] == 1u &&
        (pairClass == MR_COLLISION_PAIR_CONVEX ||
         pairClass == MR_COLLISION_PAIR_MESH)) {
        const uint cacheIndex =
            environment * dispatch.convexCacheStride +
            eligiblePairs[compiledPair].convexCacheSlot;
        publishedCaches[cacheIndex] =
            candidateCaches[cacheIndex];
    }
}

// Appends scene-body pose/velocity observations and captures typed contact
// evidence for the completed control step.
kernel void mr_world_capture_contact(
    device const MRMetalWorldDispatchGPU& worldDispatch [[buffer(0)]],
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(1)]],
    constant MRMetalWorldPassGPU& pass [[buffer(2)]],
    device const MRBodyStateGPU* sceneState [[buffer(3)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(4)]],
    device float* observations [[buffer(5)]],
    device MRMetalWorldContactStatusGPU* publicStatuses [[buffer(6)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    const uint sceneBase =
        environment * dispatch.sceneBodyStride;
    const uint observationBase =
        pass.controlStep * worldDispatch.observationStepStride +
        environment *
            worldDispatch.observationEnvironmentStride +
        worldDispatch.nq + worldDispatch.nv;
    for (uint localScene = 0u;
         localScene < dispatch.sceneBodyCount;
         ++localScene) {
        const MRBodyStateGPU state =
            sceneState[sceneBase + localScene];
        const uint output =
            observationBase + 13u * localScene;
        observations[output + 0u] = state.position.x;
        observations[output + 1u] = state.position.y;
        observations[output + 2u] = state.position.z;
        observations[output + 3u] = state.orientation.x;
        observations[output + 4u] = state.orientation.y;
        observations[output + 5u] = state.orientation.z;
        observations[output + 6u] = state.orientation.w;
        observations[output + 7u] =
            state.linearVelocityAndInverseMass.x;
        observations[output + 8u] =
            state.linearVelocityAndInverseMass.y;
        observations[output + 9u] =
            state.linearVelocityAndInverseMass.z;
        observations[output + 10u] =
            state.angularVelocity.x;
        observations[output + 11u] =
            state.angularVelocity.y;
        observations[output + 12u] =
            state.angularVelocity.z;
    }
    publicStatuses[
        pass.controlStep * dispatch.environmentCount +
        environment
    ] = statuses[environment];
}

inline bool qualityContactResponseDiagonal(
    device const MRMetalWorldContactDispatchGPU& dispatch,
    const uint localConstraint,
    const MRContactConstraintGPU contact,
    const float3 direction,
    device const MRBodyStateGPU* bodies,
    device const float* factorMatrices,
    const uint factorBase,
    device const float* pointJacobians,
    const uint pointJacobianBase,
    thread float& diagonal
) {
    float rightHandSide[MR_ARTICULATED_ABA_MAX_DOFS];
    float intermediate[MR_ARTICULATED_ABA_MAX_DOFS];
    float solution[MR_ARTICULATED_ABA_MAX_DOFS];
    const bool articulatedA =
        bodies[contact.bodyA].flagsAndIndices[1] !=
            MR_INVALID_INDEX;
    const bool articulatedB =
        bodies[contact.bodyB].flagsAndIndices[1] !=
            MR_INVALID_INDEX;
    for (uint dof = 0u; dof < dispatch.nv; ++dof) {
        rightHandSide[dof] = dot(
            direction,
            combinedJacobianColumn(
                pointJacobians,
                pointJacobianBase,
                localConstraint,
                dof,
                dispatch.nv,
                articulatedA,
                articulatedB
            )
        );
        intermediate[dof] = 0.0f;
        solution[dof] = 0.0f;
    }
    if (!solveCholesky(
            factorMatrices,
            factorBase,
            dispatch.nv,
            rightHandSide,
            intermediate,
            solution
        )) {
        return false;
    }
    float response = 0.0f;
    for (uint dof = 0u; dof < dispatch.nv; ++dof) {
        response = fma(
            rightHandSide[dof],
            solution[dof],
            response
        );
    }
    if (dynamicSceneEndpoint(
            bodies[contact.bodyA],
            dispatch.articulationIndex
        )) {
        response += dot(
            direction,
            scenePointResponse(
                bodies[contact.bodyA],
                contact.pointAndSeparation.xyz,
                direction
            )
        );
    }
    if (contact.bodyB != contact.bodyA &&
        dynamicSceneEndpoint(
            bodies[contact.bodyB],
            dispatch.articulationIndex
        )) {
        response += dot(
            direction,
            scenePointResponse(
                bodies[contact.bodyB],
                contact.pointAndSeparation.xyz,
                direction
            )
        );
    }
    diagonal = max(response, 0.0f);
    return isfinite(diagonal);
}

inline bool qualityRodContactResponseDiagonal(
    device const MRMetalWorldContactDispatchGPU& dispatch,
    const uint localConstraint,
    const float3 direction,
    device const MRBodyStateGPU* bodies,
    device const float* factorMatrices,
    const uint factorBase,
    device const float* pointJacobians,
    const uint pointJacobianBase,
    device const MRRodNodeStateGPU* rodNodes,
    device const MRRodEdgeStateGPU* rodEdges,
    device const float* inverseRodMasses,
    device const float* inverseRodTwistInertias,
    device const MRRodColliderGPU* rodColliders,
    const MRRodToolWitnessGPU witness,
    thread float& diagonal
) {
    const MRRodColliderGPU collider =
        rodColliders[witness.identity.z];
    device const MRRodNodeStateGPU& nodeA =
        rodNodes[collider.nodeA];
    device const MRRodNodeStateGPU& nodeB =
        rodNodes[collider.nodeB];
    device const MRRodEdgeStateGPU& edge =
        rodEdges[collider.edgeIndex];
    const float3 rodResponse = rodSurfaceResponse(
        nodeA,
        nodeB,
        edge,
        inverseRodMasses[collider.nodeA],
        inverseRodMasses[collider.nodeB],
        inverseRodTwistInertias[collider.edgeIndex],
        witness.rodPointAndWeight.w,
        witness.radialAndTwistJacobianV.xyz,
        collider.radiusAndOffsets.x,
        direction
    );
    float response = dot(direction, rodResponse);

    device const MRBodyStateGPU& tool =
        bodies[witness.featuresAndFlags.z];
    if (tool.flagsAndIndices[1] != MR_INVALID_INDEX) {
        float rightHandSide[MR_ARTICULATED_ABA_MAX_DOFS];
        float intermediate[MR_ARTICULATED_ABA_MAX_DOFS];
        float solution[MR_ARTICULATED_ABA_MAX_DOFS];
        for (uint dof = 0u; dof < dispatch.nv; ++dof) {
            rightHandSide[dof] = dot(
                direction,
                rodToolJacobianColumn(
                    pointJacobians,
                    pointJacobianBase,
                    localConstraint,
                    dof,
                    dispatch.nv
                )
            );
            intermediate[dof] = 0.0f;
            solution[dof] = 0.0f;
        }
        if (!solveCholesky(
                factorMatrices,
                factorBase,
                dispatch.nv,
                rightHandSide,
                intermediate,
                solution
            )) {
            return false;
        }
        for (uint dof = 0u; dof < dispatch.nv; ++dof) {
            response = fma(
                rightHandSide[dof],
                solution[dof],
                response
            );
        }
    } else if (dynamicSceneEndpoint(
                   tool,
                   dispatch.articulationIndex
               )) {
        response += dot(
            direction,
            scenePointResponse(
                tool,
                witness.toolPointAndSeparation.xyz,
                direction
            )
        );
    }
    diagonal = max(response, 0.0f);
    return isfinite(diagonal);
}

inline bool qualityGeneralizedResponseDiagonal(
    device const MRMetalWorldContactDispatchGPU& dispatch,
    const MRConstraintIRBlockGPU block,
    device const MRConstraintIREndpointGPU* endpoints,
    const uint endpointBase,
    device const float* factorMatrices,
    const uint factorBase,
    device const MRBodyStateGPU* candidateBodies,
    const uint bodyBase,
    device const float* inverseRodMasses,
    const float3 direction,
    thread float& diagonal
) {
    if (block.dimension != 1u ||
        dispatch.nv > MR_ARTICULATED_ABA_MAX_DOFS) {
        return false;
    }
    float rightHandSide[MR_ARTICULATED_ABA_MAX_DOFS];
    float intermediate[MR_ARTICULATED_ABA_MAX_DOFS];
    float solution[MR_ARTICULATED_ABA_MAX_DOFS];
    for (uint dof = 0u; dof < dispatch.nv; ++dof) {
        rightHandSide[dof] = 0.0f;
        intermediate[dof] = 0.0f;
        solution[dof] = 0.0f;
    }
    for (uint endpointIndex = 0u;
         endpointIndex < block.endpointCount;
         ++endpointIndex) {
        const MRConstraintIREndpointGPU endpoint =
            endpoints[
                endpointBase +
                block.endpointOffset +
                endpointIndex
            ];
        if (endpoint.role ==
            MR_CONSTRAINT_IR_ENDPOINT_WORLD) {
            continue;
        }
        const uint endpointRow =
            endpoint.flags &
            MR_CONSTRAINT_IR_ENDPOINT_ROW_MASK;
        if (endpointRow != 0u) {
            return false;
        }
        const float sign =
            endpoint.role == MR_CONSTRAINT_IR_ENDPOINT_A
            ? -1.0f
            : 1.0f;
        if (endpoint.jacobianKind ==
            MR_CONSTRAINT_IR_JACOBIAN_GENERALIZED) {
            if (endpoint.objectIndex >= dispatch.nv) {
                return false;
            }
            rightHandSide[endpoint.objectIndex] +=
                endpoint.axis.x;
        } else if (
            endpoint.jacobianKind ==
            MR_CONSTRAINT_IR_JACOBIAN_ROD_NODE
        ) {
            if (endpoint.objectIndex >=
                dispatch.rodNodeCount) {
                return false;
            }
            diagonal +=
                inverseRodMasses[endpoint.objectIndex] *
                dot(direction, direction);
        } else if (
            endpoint.jacobianKind ==
            MR_CONSTRAINT_IR_JACOBIAN_BODY_LOCAL_POINT
        ) {
            if (endpoint.objectIndex >= dispatch.bodyCount) {
                return false;
            }
            device const MRBodyStateGPU& body =
                candidateBodies[
                    bodyBase + endpoint.objectIndex
                ];
            const float3 point =
                body.position.xyz +
                multiply(
                    rotationMatrix(body.orientation),
                    endpoint.anchor.xyz
                );
            const float3 impulse = sign * direction;
            diagonal += dot(
                impulse,
                scenePointResponse(body, point, impulse)
            );
        } else {
            return false;
        }
    }
    if (!solveCholesky(
            factorMatrices,
            factorBase,
            dispatch.nv,
            rightHandSide,
            intermediate,
            solution
        )) {
        return false;
    }
    for (uint dof = 0u; dof < dispatch.nv; ++dof) {
        diagonal = fma(
            rightHandSide[dof],
            solution[dof],
            diagonal
        );
    }
    return diagonal >= 0.0f && isfinite(diagonal);
}

// Adapts evaluated ConstraintIR blocks to the common primal product
// cone operator. The generalized coordinate order is articulation velocity
// followed by six world-frame coordinates for every compiled scene body,
// three translational coordinates per rod node, and one twist coordinate per
// rod edge.
// Static/kinematic scene slots receive identity dynamics and zero Jacobians,
// keeping the fixed MLX shape SPD without allowing a boundary body to move.
kernel void mr_world_prepare_unified_quality(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRUnifiedQualityDispatchGPU& qualityDispatch
        [[buffer(1)]],
    device const uint* sceneBodyIndices [[buffer(2)]],
    device const float* factorMatrices [[buffer(3)]],
    device const float* pointJacobians [[buffer(4)]],
    device const float* candidateV [[buffer(5)]],
    device const MRBodyStateGPU* candidateBodies [[buffer(6)]],
    device const MRContactConstraintGPU* contacts [[buffer(7)]],
    device const MRConstraintIREndpointGPU* irEndpoints
        [[buffer(8)]],
    device const MRConstraintIRBlockGPU* irBlocks [[buffer(9)]],
    device const MREvaluatedConstraintIRRowGPU* evaluatedRows
        [[buffer(10)]],
    device const MREvaluatedConstraintIRConeGPU* evaluatedCones
        [[buffer(11)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(12)]],
    device MRUnifiedQualityBlockGPU* qualityBlocks [[buffer(13)]],
    device float* dynamicsMatrices [[buffer(14)]],
    device float* jacobianMatrices [[buffer(15)]],
    device float* biasVectors [[buffer(16)]],
    device float* freeVelocities [[buffer(17)]],
    device float* warmVelocities [[buffer(18)]],
    device float* warmImpulses [[buffer(19)]],
    device const MRRodNodeStateGPU* candidateRodNodes [[buffer(20)]],
    device const MRRodEdgeStateGPU* candidateRodEdges [[buffer(21)]],
    device const float* inverseRodMasses [[buffer(22)]],
    device const float* inverseRodTwistInertias [[buffer(23)]],
    device const MRRodColliderGPU* rodColliders [[buffer(24)]],
    device const MRRodToolWitnessGPU* rodWitnesses [[buffer(25)]],
    device const uint* rodConstraintWitnessIndices [[buffer(26)]],
    device const MRRodFactorCacheGPU* rodFactorCaches [[buffer(27)]],
    device const float* rodFactorArena [[buffer(28)]],
    const uint environment [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    if (environment >= dispatch.environmentCount ||
        lane >= 32u) {
        return;
    }
    MRMetalWorldContactStatusGPU contactStatus =
        statuses[environment];
    const uint sceneNv = 6u * dispatch.sceneBodyCount;
    const uint rodNodeVelocityBase = dispatch.nv + sceneNv;
    const uint rodTwistVelocityBase =
        rodNodeVelocityBase + 3u * dispatch.rodNodeCount;
    const uint totalNv =
        rodTwistVelocityBase + dispatch.rodEdgeCount;
    const bool validDispatch =
        contactStatus.code == MR_STEP_SUCCESS &&
        qualityDispatch.abiVersion ==
            MR_UNIFIED_QUALITY_ABI_VERSION &&
        qualityDispatch.problemCount ==
            dispatch.environmentCount &&
        qualityDispatch.generalizedVelocityCount == totalNv &&
        qualityDispatch.rowCount ==
            3u * qualityDispatch.blockCount &&
        qualityDispatch.blockCount <=
            dispatch.constraintCapacity &&
        contactStatus.requiredConstraints <=
            qualityDispatch.blockCount &&
        qualityDispatch.blockStride >=
            qualityDispatch.blockCount;
    if (!validDispatch) {
        if (lane == 0u &&
            contactStatus.code == MR_STEP_SUCCESS) {
            contactStatus.code =
                contactStatus.requiredConstraints >
                        qualityDispatch.blockCount
                ? MR_STEP_CONSTRAINT_CAPACITY_OVERFLOW
                : MR_STEP_UNSUPPORTED;
            statuses[environment] = contactStatus;
        }
        return;
    }

    const uint vectorBase =
        environment * qualityDispatch.vectorStride;
    const uint dynamicsBase =
        environment * qualityDispatch.dynamicsStride;
    const uint jacobianBase =
        environment * qualityDispatch.jacobianStride;
    const uint qualityBlockBase =
        environment * qualityDispatch.blockStride;
    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    const uint velocityBase = environment * dispatch.nv;
    const uint factorBase =
        environment * dispatch.factorStride;
    const uint pointJacobianBase =
        environment *
        dispatch.pointQueryStride * 3u * dispatch.nv;
    const uint constraintBase =
        environment * dispatch.constraintStride;
    const uint rowBase = environment * dispatch.rowStride;

    bool localFailure = false;
    for (uint dof = lane; dof < totalNv; dof += 32u) {
        float value = 0.0f;
        if (dof < dispatch.nv) {
            value = candidateV[velocityBase + dof];
        } else if (dof < rodNodeVelocityBase) {
            const uint sceneCoordinate = dof - dispatch.nv;
            const uint localScene = sceneCoordinate / 6u;
            const uint component =
                sceneCoordinate - 6u * localScene;
            const uint globalBody =
                sceneBodyIndices[localScene];
            const MRBodyStateGPU body =
                candidateBodies[bodyBase + globalBody];
            value =
                component < 3u
                ? body.linearVelocityAndInverseMass[component]
                : body.angularVelocity[component - 3u];
        } else if (dof < rodTwistVelocityBase) {
            const uint rodCoordinate =
                dof - rodNodeVelocityBase;
            const uint node = rodCoordinate / 3u;
            const uint component =
                rodCoordinate - 3u * node;
            const MRRodNodeStateGPU state =
                candidateRodNodes[
                    environment * dispatch.rodNodeCount + node
                ];
            value = state.velocity[component];
        } else {
            const uint edge = dof - rodTwistVelocityBase;
            const MRRodEdgeStateGPU state =
                candidateRodEdges[
                    environment * dispatch.rodEdgeCount + edge
                ];
            value = state.twistAndRate.y;
        }
        freeVelocities[vectorBase + dof] = value;
        warmVelocities[vectorBase + dof] = value;
        localFailure =
            localFailure || !isfinite(value);
    }

    for (uint entry = lane;
         entry < totalNv * totalNv;
         entry += 32u) {
        const uint row = entry / totalNv;
        const uint column = entry - row * totalNv;
        float value = 0.0f;
        if (row < dispatch.nv && column < dispatch.nv) {
            const uint maximumInner = min(row, column);
            for (uint inner = 0u;
                 inner <= maximumInner;
                 ++inner) {
                value = fma(
                    factorMatrices[
                        factorBase + row * dispatch.nv + inner
                    ],
                    factorMatrices[
                        factorBase +
                        column * dispatch.nv +
                        inner
                    ],
                    value
                );
            }
        } else if (
            row >= dispatch.nv &&
            column >= dispatch.nv &&
            row < rodNodeVelocityBase &&
            column < rodNodeVelocityBase
        ) {
            const uint rowSceneCoordinate =
                row - dispatch.nv;
            const uint columnSceneCoordinate =
                column - dispatch.nv;
            const uint rowScene = rowSceneCoordinate / 6u;
            const uint columnScene =
                columnSceneCoordinate / 6u;
            const uint rowComponent =
                rowSceneCoordinate - 6u * rowScene;
            const uint columnComponent =
                columnSceneCoordinate -
                6u * columnScene;
            if (rowScene == columnScene) {
                const uint globalBody =
                    sceneBodyIndices[rowScene];
                device const MRBodyStateGPU& body =
                    candidateBodies[bodyBase + globalBody];
                if (body.flagsAndIndices[0] ==
                    MR_MOTION_DYNAMIC) {
                    if (rowComponent < 3u &&
                        columnComponent < 3u) {
                        value =
                            rowComponent == columnComponent
                            ? 1.0f /
                                body
                                    .linearVelocityAndInverseMass
                                    .w
                            : 0.0f;
                    } else if (
                        rowComponent >= 3u &&
                        columnComponent >= 3u
                    ) {
                        Mat3 inertia;
                        if (!inverseMatrix(
                                stateInverseInertia(body),
                                inertia
                            )) {
                            localFailure = true;
                        } else {
                            const uint inertiaRow =
                                rowComponent - 3u;
                            const uint inertiaColumn =
                                columnComponent - 3u;
                            value =
                                inertiaRow == 0u
                                ? inertia.row0[inertiaColumn]
                                : inertiaRow == 1u
                                ? inertia.row1[inertiaColumn]
                                : inertia.row2[inertiaColumn];
                        }
                    }
                } else {
                    value =
                        rowComponent == columnComponent
                        ? 1.0f
                        : 0.0f;
                }
            }
        } else if (
            row >= rodNodeVelocityBase &&
            column >= rodNodeVelocityBase &&
            row < rodTwistVelocityBase &&
            column < rodTwistVelocityBase
        ) {
            const uint rowCoordinate =
                row - rodNodeVelocityBase;
            const uint columnCoordinate =
                column - rodNodeVelocityBase;
            const uint rowNode = rowCoordinate / 3u;
            const uint columnNode = columnCoordinate / 3u;
            const uint rowComponent =
                rowCoordinate - 3u * rowNode;
            const uint columnComponent =
                columnCoordinate - 3u * columnNode;
            if (!retainedRodTranslationOperatorEntry(
                    dispatch,
                    environment,
                    rodFactorCaches,
                    rodFactorArena,
                    rowNode,
                    rowComponent,
                    columnNode,
                    columnComponent,
                    value
                )) {
                localFailure = true;
            }
        } else if (
            row >= rodTwistVelocityBase &&
            column >= rodTwistVelocityBase
        ) {
            const uint rowEdge = row - rodTwistVelocityBase;
            const uint columnEdge =
                column - rodTwistVelocityBase;
            if (!retainedRodTwistOperatorEntry(
                    dispatch,
                    environment,
                    rodFactorCaches,
                    rodFactorArena,
                    rowEdge,
                    columnEdge,
                    value
                )) {
                localFailure = true;
            }
        }
        dynamicsMatrices[dynamicsBase + entry] = value;
        localFailure =
            localFailure || !isfinite(value);
    }

    for (uint localConstraint = lane;
         localConstraint < qualityDispatch.blockCount;
         localConstraint += 32u) {
        bool active =
            localConstraint <
                contactStatus.requiredConstraints;
        MRUnifiedQualityBlockGPU qualityBlock = {};
        qualityBlock.layout = uint4(
            3u * localConstraint,
            3u,
            MR_UNIFIED_QUALITY_ELLIPTIC_CONE,
            0u
        );
        qualityBlock.stableKey = uint4(
            localConstraint,
            environment,
            0u,
            0u
        );
        qualityBlock.scale0 =
            float4(1.0f, 0.5f, 0.5f, 1.0f);
        qualityBlock.scale1 = float4(1.0f);
        qualityBlock.regularization0 =
            float4(1.0f);
        qualityBlock.regularization1 =
            float4(1.0f);
        qualityBlock.boundsAndShift = float4(0.0f);
        if (active) {
            const MRConstraintIRBlockGPU sourceBlock =
                irBlocks[constraintBase + localConstraint];
            const MREvaluatedConstraintIRConeGPU cone =
                evaluatedCones[
                    constraintBase + localConstraint
                ];
            qualityBlock.stableKey = uint4(
                sourceBlock.key.words[0],
                sourceBlock.key.words[1],
                sourceBlock.key.words[2],
                sourceBlock.key.words[3]
            );
            const MRContactConstraintGPU contact =
                contacts[constraintBase + localConstraint];
            const bool generalized =
                (sourceBlock.flags &
                 MR_CONSTRAINT_IR_BLOCK_GENERALIZED) != 0u ||
                (contact.flags &
                 MR_CONSTRAINT_FLAG_GENERALIZED) != 0u;
            if (generalized) {
                float responseDiagonal = 0.0f;
                const bool responseValid =
                    qualityGeneralizedResponseDiagonal(
                        dispatch,
                        sourceBlock,
                        irEndpoints,
                        2u * environment *
                            dispatch.constraintStride,
                        factorMatrices,
                        factorBase,
                        candidateBodies,
                        bodyBase,
                        inverseRodMasses,
                        evaluatedRows[
                            rowBase + 3u * localConstraint
                        ].direction.xyz,
                        responseDiagonal
                    );
                const MREvaluatedConstraintIRRowGPU row =
                    evaluatedRows[
                        rowBase + 3u * localConstraint
                    ];
                if (!responseValid ||
                    sourceBlock.dimension != 1u ||
                    !isfinite(row.impulseLower) ||
                    !isfinite(row.impulseUpper) ||
                    row.impulseLower > row.impulseUpper) {
                    localFailure = true;
                }
                const float numericalFloor = max(
                    qualityDispatch.numerics.y,
                    qualityDispatch.numerics.x *
                        1.1920928955078125e-7f *
                        max(
                            responseDiagonal,
                            qualityDispatch.numerics.y
                        )
                );
                uint scalarFlags =
                    MR_UNIFIED_QUALITY_BLOCK_REPORT_FLOOR;
                if (row.impulseLower <=
                        -0.5f * MR_CONSTRAINT_IR_UNBOUNDED &&
                    row.impulseUpper >=
                        0.5f * MR_CONSTRAINT_IR_UNBOUNDED) {
                    scalarFlags |=
                        MR_UNIFIED_QUALITY_BLOCK_HARD_EQUALITY;
                }
                qualityBlock.layout = uint4(
                    3u * localConstraint,
                    1u,
                    MR_UNIFIED_QUALITY_SCALAR_INTERVAL,
                    scalarFlags
                );
                qualityBlock.scale0 = float4(1.0f);
                qualityBlock.regularization0 = float4(
                    max(row.regularization, numericalFloor),
                    1.0f,
                    1.0f,
                    1.0f
                );
                qualityBlock.boundsAndShift = float4(
                    row.impulseLower,
                    row.impulseUpper,
                    0.0f,
                    0.0f
                );
                biasVectors[
                    vectorBase + 3u * localConstraint
                ] = -row.targetVelocity;
                biasVectors[
                    vectorBase + 3u * localConstraint + 1u
                ] = 0.0f;
                biasVectors[
                    vectorBase + 3u * localConstraint + 2u
                ] = 0.0f;
                warmImpulses[
                    vectorBase + 3u * localConstraint
                ] = contact.impulses.x;
                warmImpulses[
                    vectorBase + 3u * localConstraint + 1u
                ] = 0.0f;
                warmImpulses[
                    vectorBase + 3u * localConstraint + 2u
                ] = 0.0f;
            } else {
            if (!(cone.effectiveFrictionU > 0.0f) ||
                !(cone.effectiveFrictionV > 0.0f) ||
                !isfinite(cone.effectiveFrictionU) ||
                !isfinite(cone.effectiveFrictionV)) {
                localFailure = true;
            }
            qualityBlock.scale0 = float4(
                1.0f,
                cone.effectiveFrictionU,
                cone.effectiveFrictionV,
                1.0f
            );
            float responseDiagonal[3] = {
                0.0f,
                0.0f,
                0.0f,
            };
            for (uint axis = 0u; axis < 3u; ++axis) {
                const float3 direction =
                    evaluatedRows[
                        rowBase +
                        3u * localConstraint +
                        axis
                    ].direction.xyz;
                const bool rodContact =
                    (contact.flags &
                     MR_CONSTRAINT_FLAG_ROD_ENDPOINT) != 0u;
                const bool responseValid =
                    rodContact
                    ? qualityRodContactResponseDiagonal(
                          dispatch,
                          localConstraint,
                          direction,
                          candidateBodies + bodyBase,
                          factorMatrices,
                          factorBase,
                          pointJacobians,
                          pointJacobianBase,
                          candidateRodNodes +
                              environment *
                                  dispatch.rodNodeCount,
                          candidateRodEdges +
                              environment *
                                  dispatch.rodEdgeCount,
                          inverseRodMasses,
                          inverseRodTwistInertias,
                          rodColliders,
                          rodWitnesses[
                              rodConstraintWitnessIndices[
                                  constraintBase +
                                  localConstraint
                              ]
                          ],
                          responseDiagonal[axis]
                      )
                    : qualityContactResponseDiagonal(
                          dispatch,
                          localConstraint,
                          contact,
                          direction,
                          candidateBodies + bodyBase,
                          factorMatrices,
                          factorBase,
                          pointJacobians,
                          pointJacobianBase,
                          responseDiagonal[axis]
                      );
                if (!responseValid) {
                    localFailure = true;
                }
            }
            const float positiveBlockMean = max(
                (
                    responseDiagonal[0] +
                    responseDiagonal[1] +
                    responseDiagonal[2]
                ) / 3.0f,
                qualityDispatch.numerics.y
            );
            for (uint axis = 0u; axis < 3u; ++axis) {
                const MREvaluatedConstraintIRRowGPU row =
                    evaluatedRows[
                        rowBase +
                        3u * localConstraint +
                        axis
                    ];
                qualityBlock.regularization0[axis] = max(
                    row.regularization,
                    max(
                        max(
                            qualityDispatch.numerics.y,
                            qualityDispatch.numerics.x *
                                1.1920928955078125e-7f *
                                max(
                                    responseDiagonal[axis],
                                    positiveBlockMean
                                )
                        ),
                        // A pure 64-epsilon equality floor is too stiff for
                        // a repeatedly projected FP32 cone. Keep its exact
                        // floor, then bound the contact metric condition
                        // number with a declared scale-aware 5e-4 response
                        // floor. This is contact regularization, not hidden
                        // equality compliance, and is reported by the
                        // quality status.
                        5.0e-4f *
                            positiveBlockMean
                    )
                );
                biasVectors[
                    vectorBase +
                    3u * localConstraint +
                    axis
                ] = -row.targetVelocity;
            }
            qualityBlock.boundsAndShift = float4(
                cone.adhesionImpulse,
                cone.maximumNormalImpulse,
                0.0f,
                0.0f
            );
            warmImpulses[
                vectorBase + 3u * localConstraint + 0u
            ] = contact.impulses.x;
            warmImpulses[
                vectorBase + 3u * localConstraint + 1u
            ] = contact.impulses.y;
            warmImpulses[
                vectorBase + 3u * localConstraint + 2u
            ] = contact.impulses.z;
            }
        } else {
            for (uint axis = 0u; axis < 3u; ++axis) {
                biasVectors[
                    vectorBase +
                    3u * localConstraint +
                    axis
                ] = 0.0f;
                warmImpulses[
                    vectorBase +
                    3u * localConstraint +
                    axis
                ] = 0.0f;
            }
        }
        qualityBlocks[
            qualityBlockBase + localConstraint
        ] = qualityBlock;
    }

    for (uint entry = lane;
         entry < qualityDispatch.rowCount * totalNv;
         entry += 32u) {
        const uint row = entry / totalNv;
        const uint dof = entry - row * totalNv;
        const uint localConstraint = row / 3u;
        const uint axis = row - 3u * localConstraint;
        float coefficient = 0.0f;
        if (localConstraint <
                contactStatus.requiredConstraints) {
            const MRContactConstraintGPU contact =
                contacts[constraintBase + localConstraint];
            const MRConstraintIRBlockGPU sourceBlock =
                irBlocks[constraintBase + localConstraint];
            const float3 direction =
                evaluatedRows[rowBase + row].direction.xyz;
            const bool rodContact =
                (contact.flags &
                 MR_CONSTRAINT_FLAG_ROD_ENDPOINT) != 0u;
            const bool generalized =
                (sourceBlock.flags &
                 MR_CONSTRAINT_IR_BLOCK_GENERALIZED) != 0u ||
                (contact.flags &
                 MR_CONSTRAINT_FLAG_GENERALIZED) != 0u;
            if (generalized) {
                if (axis == 0u) {
                    const uint endpointBase =
                        2u * environment *
                        dispatch.constraintStride;
                    for (uint endpointIndex = 0u;
                         endpointIndex <
                             sourceBlock.endpointCount;
                         ++endpointIndex) {
                        const MRConstraintIREndpointGPU endpoint =
                            irEndpoints[
                                endpointBase +
                                sourceBlock.endpointOffset +
                                endpointIndex
                            ];
                        const uint endpointRow =
                            endpoint.flags &
                            MR_CONSTRAINT_IR_ENDPOINT_ROW_MASK;
                        if (endpointRow != 0u ||
                            endpoint.role ==
                                MR_CONSTRAINT_IR_ENDPOINT_WORLD) {
                            continue;
                        }
                        const float sign =
                            endpoint.role ==
                                MR_CONSTRAINT_IR_ENDPOINT_A
                            ? -1.0f
                            : 1.0f;
                        if (endpoint.jacobianKind ==
                            MR_CONSTRAINT_IR_JACOBIAN_GENERALIZED) {
                            if (dof < dispatch.nv &&
                                endpoint.objectIndex == dof) {
                                coefficient += endpoint.axis.x;
                            }
                        } else if (
                            endpoint.jacobianKind ==
                            MR_CONSTRAINT_IR_JACOBIAN_ROD_NODE
                        ) {
                            if (dof >= rodNodeVelocityBase &&
                                dof < rodTwistVelocityBase) {
                                const uint coordinate =
                                    dof -
                                    rodNodeVelocityBase;
                                const uint node =
                                    coordinate / 3u;
                                const uint component =
                                    coordinate - 3u * node;
                                if (node ==
                                    endpoint.objectIndex) {
                                    coefficient +=
                                        sign *
                                        direction[component];
                                }
                            }
                        } else if (
                            endpoint.jacobianKind ==
                            MR_CONSTRAINT_IR_JACOBIAN_BODY_LOCAL_POINT
                        ) {
                            if (dof >= dispatch.nv &&
                                dof < rodNodeVelocityBase) {
                                const uint sceneCoordinate =
                                    dof - dispatch.nv;
                                const uint localScene =
                                    sceneCoordinate / 6u;
                                const uint component =
                                    sceneCoordinate -
                                    6u * localScene;
                                const uint globalBody =
                                    sceneBodyIndices[
                                        localScene
                                    ];
                                if (globalBody ==
                                    endpoint.objectIndex) {
                                    if (component < 3u) {
                                        coefficient +=
                                            sign *
                                            direction[component];
                                    } else {
                                        device const MRBodyStateGPU&
                                            body =
                                                candidateBodies[
                                                    bodyBase +
                                                    globalBody
                                                ];
                                        const float3 lever =
                                            multiply(
                                                rotationMatrix(
                                                    body.orientation
                                                ),
                                                endpoint.anchor.xyz
                                            );
                                        coefficient +=
                                            sign *
                                            cross(
                                                lever,
                                                direction
                                            )[component - 3u];
                                    }
                                }
                            }
                        }
                    }
                }
            } else if (rodContact) {
                const uint flatWitness =
                    rodConstraintWitnessIndices[
                        constraintBase + localConstraint
                    ];
                const MRRodToolWitnessGPU witness =
                    rodWitnesses[flatWitness];
                const MRRodColliderGPU collider =
                    rodColliders[witness.identity.z];
                if (dof < dispatch.nv) {
                    coefficient = dot(
                        direction,
                        rodToolJacobianColumn(
                            pointJacobians,
                            pointJacobianBase,
                            localConstraint,
                            dof,
                            dispatch.nv
                        )
                    );
                } else if (dof < rodNodeVelocityBase) {
                    const uint sceneCoordinate =
                        dof - dispatch.nv;
                    const uint localScene =
                        sceneCoordinate / 6u;
                    const uint component =
                        sceneCoordinate -
                        6u * localScene;
                    const uint globalBody =
                        sceneBodyIndices[localScene];
                    device const MRBodyStateGPU& body =
                        candidateBodies[
                            bodyBase + globalBody
                        ];
                    if (globalBody ==
                            witness.featuresAndFlags.z &&
                        body.flagsAndIndices[0] ==
                            MR_MOTION_DYNAMIC) {
                        if (component < 3u) {
                            coefficient =
                                direction[component];
                        } else {
                            const float3 lever =
                                witness
                                    .toolPointAndSeparation
                                    .xyz -
                                body.position.xyz;
                            coefficient =
                                cross(lever, direction)[
                                    component - 3u
                                ];
                        }
                    }
                } else if (dof < rodTwistVelocityBase) {
                    const uint rodCoordinate =
                        dof - rodNodeVelocityBase;
                    const uint node = rodCoordinate / 3u;
                    const uint component =
                        rodCoordinate - 3u * node;
                    const float weight =
                        witness.rodPointAndWeight.w;
                    const float3 edgeVector =
                        candidateRodNodes[
                            environment *
                                dispatch.rodNodeCount +
                            collider.nodeB
                        ].position.xyz -
                        candidateRodNodes[
                            environment *
                                dispatch.rodNodeCount +
                            collider.nodeA
                        ].position.xyz;
                    const float lengthSquared =
                        dot(edgeVector, edgeVector);
                    if (lengthSquared > 1.0e-20f) {
                        const float length =
                            sqrt(lengthSquared);
                        const float3 tangent =
                            edgeVector / length;
                        const float3 surfaceRadius =
                            collider.radiusAndOffsets.x *
                            witness
                                .radialAndTwistJacobianV
                                .xyz;
                        const float3 angularImpulse =
                            cross(surfaceRadius, direction);
                        const float3 bendTranspose =
                            (
                                cross(
                                    angularImpulse,
                                    tangent
                                ) -
                                tangent *
                                    dot(
                                        tangent,
                                        cross(
                                            angularImpulse,
                                            tangent
                                        )
                                    )
                            ) / length;
                        const float3 nodeForce =
                            node == collider.nodeA
                            ? (1.0f - weight) *
                                  direction -
                                  bendTranspose
                            : node == collider.nodeB
                            ? weight * direction +
                                  bendTranspose
                            : float3(0.0f);
                        coefficient = -nodeForce[component];
                    }
                } else {
                    const uint edge =
                        dof - rodTwistVelocityBase;
                    if (edge == collider.edgeIndex) {
                        const float3 edgeVector =
                            candidateRodNodes[
                                environment *
                                    dispatch.rodNodeCount +
                                collider.nodeB
                            ].position.xyz -
                            candidateRodNodes[
                                environment *
                                    dispatch.rodNodeCount +
                                collider.nodeA
                            ].position.xyz;
                        const float lengthSquared =
                            dot(edgeVector, edgeVector);
                        if (lengthSquared > 1.0e-20f) {
                            const float3 tangent =
                                edgeVector *
                                rsqrt(lengthSquared);
                            const float3 surfaceRadius =
                                collider.radiusAndOffsets.x *
                                witness
                                    .radialAndTwistJacobianV
                                    .xyz;
                            coefficient = -dot(
                                cross(
                                    tangent,
                                    surfaceRadius
                                ),
                                direction
                            );
                        }
                    }
                }
            } else if (dof < dispatch.nv) {
                const bool articulatedA =
                    candidateBodies[
                        bodyBase + contact.bodyA
                    ].flagsAndIndices[1] !=
                        MR_INVALID_INDEX;
                const bool articulatedB =
                    candidateBodies[
                        bodyBase + contact.bodyB
                    ].flagsAndIndices[1] !=
                        MR_INVALID_INDEX;
                coefficient = dot(
                    direction,
                    combinedJacobianColumn(
                        pointJacobians,
                        pointJacobianBase,
                        localConstraint,
                        dof,
                        dispatch.nv,
                        articulatedA,
                        articulatedB
                    )
                );
            } else if (dof < rodNodeVelocityBase) {
                const uint sceneCoordinate =
                    dof - dispatch.nv;
                const uint localScene =
                    sceneCoordinate / 6u;
                const uint component =
                    sceneCoordinate - 6u * localScene;
                const uint globalBody =
                    sceneBodyIndices[localScene];
                device const MRBodyStateGPU& body =
                    candidateBodies[bodyBase + globalBody];
                if (body.flagsAndIndices[0] ==
                    MR_MOTION_DYNAMIC) {
                    float sign = 0.0f;
                    if (contact.bodyA == globalBody) {
                        sign -= 1.0f;
                    }
                    if (contact.bodyB == globalBody) {
                        sign += 1.0f;
                    }
                    if (component < 3u) {
                        coefficient =
                            sign * direction[component];
                    } else {
                        const float3 lever =
                            contact.pointAndSeparation.xyz -
                            body.position.xyz;
                        coefficient =
                            sign *
                            cross(lever, direction)[
                                component - 3u
                            ];
                    }
                }
            }
        }
        jacobianMatrices[jacobianBase + entry] =
            coefficient;
        localFailure =
            localFailure || !isfinite(coefficient);
        static_cast<void>(axis);
    }

    if (simd_any(localFailure) && lane == 0u) {
        contactStatus.code = MR_STEP_NONFINITE_RESULT;
        contactStatus.firstFailingConstraint = 0u;
        statuses[environment] = contactStatus;
    }
}

// Reconstructs the second primal starting point from the persistent contact
// impulses:
//
//     v_warm = v_free + A^-1 J^T lambda_cached.
//
// The preparation kernel writes A and J in parallel, so this is deliberately
// a separate graph stage. Each generalized coordinate accumulates rows in
// canonical order; worker scheduling therefore cannot perturb the sum.
kernel void mr_world_reconstruct_unified_quality_warm_start(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRUnifiedQualityDispatchGPU& qualityDispatch
        [[buffer(1)]],
    device const float* factorMatrices [[buffer(2)]],
    device const float* dynamicsMatrices [[buffer(3)]],
    device const float* jacobianMatrices [[buffer(4)]],
    device const float* freeVelocities [[buffer(5)]],
    device const float* warmImpulses [[buffer(6)]],
    device float* warmVelocities [[buffer(7)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(8)]],
    device const MRRodFactorCacheGPU* rodFactorCaches [[buffer(9)]],
    device const float* rodFactorArena [[buffer(10)]],
    const uint environment [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    if (environment >= dispatch.environmentCount ||
        lane >= MR_SIMD_WIDTH) {
        return;
    }
    MRMetalWorldContactStatusGPU contactStatus =
        statuses[environment];
    const uint rodVelocityBase =
        dispatch.nv + 6u * dispatch.sceneBodyCount;
    const uint totalNv =
        rodVelocityBase +
        3u * dispatch.rodNodeCount +
        dispatch.rodEdgeCount;
    const bool validDispatch =
        contactStatus.code == MR_STEP_SUCCESS &&
        dispatch.nv <= MR_ARTICULATED_ABA_MAX_DOFS &&
        totalNv ==
            qualityDispatch.generalizedVelocityCount &&
        totalNv <=
            MR_UNIFIED_QUALITY_MAX_GENERALIZED_VELOCITIES &&
        qualityDispatch.rowCount <=
            MR_UNIFIED_QUALITY_MAX_ROWS;
    if (!validDispatch) {
        if (lane == 0u &&
            contactStatus.code == MR_STEP_SUCCESS) {
            contactStatus.code = MR_STEP_UNSUPPORTED;
            statuses[environment] = contactStatus;
        }
        return;
    }

    const uint vectorBase =
        environment * qualityDispatch.vectorStride;
    const uint jacobianBase =
        environment * qualityDispatch.jacobianStride;
    threadgroup float generalizedImpulse[
        MR_UNIFIED_QUALITY_MAX_GENERALIZED_VELOCITIES
    ];
    for (uint dof = lane; dof < totalNv; dof += MR_SIMD_WIDTH) {
        float value = 0.0f;
        for (uint row = 0u;
             row < qualityDispatch.rowCount;
             ++row) {
            value = fma(
                jacobianMatrices[
                    jacobianBase + row * totalNv + dof
                ],
                warmImpulses[vectorBase + row],
                value
            );
        }
        generalizedImpulse[dof] = value;
    }
    threadgroup_barrier(
        mem_flags::mem_threadgroup |
        mem_flags::mem_device
    );

    if (lane != 0u) {
        return;
    }
    bool valid = true;
    float rightHandSide[MR_ARTICULATED_ABA_MAX_DOFS];
    float intermediate[MR_ARTICULATED_ABA_MAX_DOFS];
    float solution[MR_ARTICULATED_ABA_MAX_DOFS];
    for (uint dof = 0u; dof < dispatch.nv; ++dof) {
        rightHandSide[dof] = generalizedImpulse[dof];
        intermediate[dof] = 0.0f;
        solution[dof] = 0.0f;
    }
    if (dispatch.nv != 0u) {
        valid = solveCholesky(
            factorMatrices,
            environment * dispatch.factorStride,
            dispatch.nv,
            rightHandSide,
            intermediate,
            solution
        );
    }
    for (uint dof = 0u; valid && dof < dispatch.nv; ++dof) {
        const float value =
            freeVelocities[vectorBase + dof] + solution[dof];
        warmVelocities[vectorBase + dof] = value;
        valid = valid && isfinite(value);
    }

    // Scene-body dynamics are independent 6x6 SPD blocks. Factor each block
    // in stable body/component order rather than materializing a global
    // inverse or relying on atomic wrench accumulation.
    const uint dynamicsBase =
        environment * qualityDispatch.dynamicsStride;
    for (uint localScene = 0u;
         valid && localScene < dispatch.sceneBodyCount;
         ++localScene) {
        float factor[36];
        float bodyIntermediate[6];
        float bodySolution[6];
        const uint coordinateBase =
            dispatch.nv + 6u * localScene;
        for (uint row = 0u; row < 6u; ++row) {
            bodyIntermediate[row] = 0.0f;
            bodySolution[row] = 0.0f;
            for (uint column = 0u; column < 6u; ++column) {
                factor[row * 6u + column] = 0.0f;
            }
        }
        for (uint row = 0u; valid && row < 6u; ++row) {
            for (uint column = 0u; column <= row; ++column) {
                float value = dynamicsMatrices[
                    dynamicsBase +
                    (coordinateBase + row) * totalNv +
                    coordinateBase + column
                ];
                for (uint inner = 0u;
                     inner < column;
                     ++inner) {
                    value = fma(
                        -factor[row * 6u + inner],
                        factor[column * 6u + inner],
                        value
                    );
                }
                if (row == column) {
                    if (!(value >
                          qualityDispatch.numerics.y) ||
                        !isfinite(value)) {
                        valid = false;
                        break;
                    }
                    factor[row * 6u + column] = sqrt(value);
                } else {
                    factor[row * 6u + column] =
                        value /
                        factor[column * 6u + column];
                }
            }
        }
        for (uint row = 0u; valid && row < 6u; ++row) {
            float value =
                generalizedImpulse[coordinateBase + row];
            for (uint column = 0u;
                 column < row;
                 ++column) {
                value = fma(
                    -factor[row * 6u + column],
                    bodyIntermediate[column],
                    value
                );
            }
            bodyIntermediate[row] =
                value / factor[row * 6u + row];
        }
        for (uint reverse = 0u;
             valid && reverse < 6u;
             ++reverse) {
            const uint row = 5u - reverse;
            float value = bodyIntermediate[row];
            for (uint column = row + 1u;
                 column < 6u;
                 ++column) {
                value = fma(
                    -factor[column * 6u + row],
                    bodySolution[column],
                    value
                );
            }
            bodySolution[row] =
                value / factor[row * 6u + row];
            valid = isfinite(bodySolution[row]);
        }
        for (uint component = 0u;
             valid && component < 6u;
             ++component) {
            const uint dof = coordinateBase + component;
            const float value =
                freeVelocities[vectorBase + dof] +
                bodySolution[component];
            warmVelocities[vectorBase + dof] = value;
            valid = valid && isfinite(value);
        }
    }
    // Rod coordinates consume the retained banded factor of the same
    // M + hD + h^2K operator materialized above. This removes the old
    // diagonal-only warm-start approximation without constructing an inverse.
    for (uint rodIndex = 0u;
         valid && rodIndex < dispatch.rodCount;
         ++rodIndex) {
        const MRRodFactorCacheGPU cache =
            rodFactorCaches[
                environment * dispatch.rodCount + rodIndex
            ];
        const uint nodeBase = cache.velocityOffset;
        const uint nodeCount = cache.velocityCount;
        const uint edgeBase = cache.blockCount;
        const uint edgeCount = cache.blockWidth;
        const bool cacheValid =
            cache.environment == environment &&
            cache.rodIndex == rodIndex &&
            cache.code == MR_ROD_GPU_SUCCESS &&
            (cache.flags & MR_ROD_FACTOR_CACHE_VALID) != 0u &&
            nodeBase + nodeCount <= dispatch.rodNodeCount &&
            edgeBase + edgeCount <= dispatch.rodEdgeCount;
        if (!cacheValid) {
            valid = false;
            break;
        }
        const uint translationBase =
            rodVelocityBase + 3u * nodeBase;
        const uint twistBase =
            rodVelocityBase +
            3u * dispatch.rodNodeCount +
            edgeBase;
        valid = solveRodTranslationFactor(
            rodFactorArena,
            cache.firstBlock,
            3u * nodeCount,
            generalizedImpulse,
            translationBase
        );
        if (valid) {
            valid = solveRodTwistFactor(
                rodFactorArena,
                cache.firstBlock +
                    MR_ROD_FACTOR_TRANSLATION_FLOATS_PER_NODE *
                        nodeCount,
                edgeCount,
                generalizedImpulse,
                twistBase
            );
        }
        for (uint coordinate = 0u;
             valid && coordinate < 3u * nodeCount;
             ++coordinate) {
            const uint dof = translationBase + coordinate;
            const float value =
                freeVelocities[vectorBase + dof] +
                generalizedImpulse[dof];
            warmVelocities[vectorBase + dof] = value;
            valid = isfinite(value);
        }
        for (uint edge = 0u;
             valid && edge < edgeCount;
             ++edge) {
            const uint dof = twistBase + edge;
            const float value =
                freeVelocities[vectorBase + dof] +
                generalizedImpulse[dof];
            warmVelocities[vectorBase + dof] = value;
            valid = isfinite(value);
        }
    }
    if (!valid) {
        // The persistent-impulse reconstruction is only the second of two
        // primal starting points. A singular or badly scaled cached start
        // must not reject a valid free-velocity problem; deterministically
        // fall back to v* and let the quality solver evaluate that objective.
        for (uint dof = 0u; dof < totalNv; ++dof) {
            warmVelocities[vectorBase + dof] =
                freeVelocities[vectorBase + dof];
        }
    }
}

// Stable active-problem compaction for the MLX persistent quality executor.
// One SIMD32 group scans environments in canonical order. The cursor is reset
// invocation-locally and atomics are reserved exclusively for later worker
// claims, never packet ordering.
kernel void mr_world_build_unified_quality_queue(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRUnifiedQualityDispatchGPU& qualityDispatch
        [[buffer(1)]],
    device const MRMetalWorldContactStatusGPU* statuses [[buffer(2)]],
    device MRUnifiedQualityWorkQueueGPU& queue [[buffer(3)]],
    device MRUnifiedQualityWorkPacketGPU* packets [[buffer(4)]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    if (lane >= MR_SIMD_WIDTH) {
        return;
    }
    const bool valid =
        qualityDispatch.abiVersion ==
            MR_UNIFIED_QUALITY_ABI_VERSION &&
        qualityDispatch.problemCount ==
            dispatch.environmentCount;
    uint total = 0u;
    if (valid) {
        for (uint tile = 0u;
             tile < dispatch.environmentCount;
             tile += MR_SIMD_WIDTH) {
            const uint environment = tile + lane;
            const bool active =
                environment < dispatch.environmentCount &&
                statuses[environment].code == MR_STEP_SUCCESS;
            const uint prefix =
                simd_prefix_exclusive_sum(active ? 1u : 0u);
            const uint tileCount =
                simd_sum(active ? 1u : 0u);
            if (active) {
                MRUnifiedQualityWorkPacketGPU packet = {};
                packet.identity = uint4(
                    environment,
                    0u,
                    environment,
                    0u
                );
                packet.scheduling = uint4(
                    statuses[environment].eventGeneration,
                    0u,
                    0u,
                    0u
                );
                packets[total + prefix] = packet;
            }
            total += tileCount;
        }
    }
    if (lane == 0u) {
        MRUnifiedQualityWorkQueueGPU next = {};
        next.count = total;
        next.capacity = qualityDispatch.problemCount;
        next.required = total;
        next.cursor = 0u;
        next.workerGroups = max(
            dispatch.waveWorkerGroupCount,
            1u
        );
        next.flags =
            MR_UNIFIED_QUALITY_QUEUE_PERSISTENT_WORKER;
        next.firstFailingStableKey =
            uint4(MR_UNIFIED_QUALITY_INVALID_INDEX);
        next.indirect = uint4(total, 1u, 1u, total);
        queue = next;
    }
}

// Transactionally transfers a converged generalized-velocity solution back
// to the candidate world and persistent manifold impulses. A failed quality
// island only marks its environment; the publication kernels preserve the
// previous accepted state.
kernel void mr_world_apply_unified_quality(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRUnifiedQualityDispatchGPU& qualityDispatch
        [[buffer(1)]],
    device const uint* sceneBodyIndices [[buffer(2)]],
    device const MRUnifiedQualityStatusGPU* qualityStatuses
        [[buffer(3)]],
    device const float* qualityVelocities [[buffer(4)]],
    device const float* qualityImpulses [[buffer(5)]],
    device float* candidateV [[buffer(6)]],
    device MRBodyStateGPU* candidateBodies [[buffer(7)]],
    device MRContactConstraintGPU* contacts [[buffer(8)]],
    device const MRContactPointMetaGPU* contactMetadata [[buffer(9)]],
    device MRManifoldPointGPU* candidateManifoldPoints
        [[buffer(10)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(11)]],
    device MRRodNodeStateGPU* candidateRodNodes [[buffer(12)]],
    device MRRodEdgeStateGPU* candidateRodEdges [[buffer(13)]],
    device MRRodToolWitnessGPU* rodWitnesses [[buffer(14)]],
    device const uint* rodConstraintWitnessIndices [[buffer(15)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    MRMetalWorldContactStatusGPU contactStatus =
        statuses[environment];
    if (contactStatus.code != MR_STEP_SUCCESS) {
        return;
    }
    const MRUnifiedQualityStatusGPU quality =
        qualityStatuses[environment];
    if (quality.code != MR_UNIFIED_QUALITY_SUCCESS) {
        contactStatus.code =
            quality.code ==
                    MR_UNIFIED_QUALITY_FACTORIZATION_FAILED
            ? MR_STEP_FACTORIZATION_FAILED
            : quality.code ==
                      MR_UNIFIED_QUALITY_DID_NOT_CONVERGE ||
                  quality.code ==
                      MR_UNIFIED_QUALITY_PCG_FAILED ||
                  quality.code ==
                      MR_UNIFIED_QUALITY_LINE_SEARCH_FAILED
            ? MR_STEP_DID_NOT_CONVERGE
            : MR_STEP_NONFINITE_RESULT;
        contactStatus.firstFailingConstraint =
            quality.failingBlock;
        contactStatus.firstFailingStableKeyLow =
            quality.firstFailingStableKey.x;
        contactStatus.firstFailingStableKeyHigh =
            quality.firstFailingStableKey.y;
        contactStatus.solverIterations =
            quality.newtonIterations;
        contactStatus.qualityNewtonIterations =
            quality.newtonIterations;
        contactStatus.qualityPCGIterations =
            quality.pcgIterations;
        contactStatus.qualityLineSearchBacktracks =
            quality.lineSearchBacktracks;
        contactStatus.qualitySolvePath = quality.solvePath;
        contactStatus.residuals = float4(
            quality.certificates1.z,
            quality.certificates0.x,
            quality.certificates0.y,
            quality.certificates1.x
        );
        contactStatus.qualityCertificates =
            quality.certificates0;
        contactStatus.qualityDiagnostics = float4(
            quality.certificates1.x,
            quality.certificates1.y,
            quality.certificates1.z,
            max(quality.numerics.x, quality.numerics.y)
        );
        statuses[environment] = contactStatus;
        return;
    }

    const uint vectorBase =
        environment * qualityDispatch.vectorStride;
    const uint velocityBase = environment * dispatch.nv;
    for (uint dof = 0u; dof < dispatch.nv; ++dof) {
        candidateV[velocityBase + dof] =
            qualityVelocities[vectorBase + dof];
    }
    const uint bodyBase =
        environment * dispatch.bodyStateStride;
    for (uint localScene = 0u;
         localScene < dispatch.sceneBodyCount;
         ++localScene) {
        const uint globalBody = sceneBodyIndices[localScene];
        device MRBodyStateGPU& body =
            candidateBodies[bodyBase + globalBody];
        if (body.flagsAndIndices[0] != MR_MOTION_DYNAMIC) {
            continue;
        }
        const uint offset =
            vectorBase + dispatch.nv + 6u * localScene;
        body.linearVelocityAndInverseMass.xyz = float3(
            qualityVelocities[offset + 0u],
            qualityVelocities[offset + 1u],
            qualityVelocities[offset + 2u]
        );
        body.angularVelocity.xyz = float3(
            qualityVelocities[offset + 3u],
            qualityVelocities[offset + 4u],
            qualityVelocities[offset + 5u]
        );
    }
    const uint rodVelocityBase =
        vectorBase +
        dispatch.nv +
        6u * dispatch.sceneBodyCount;
    const uint rodNodeBase =
        environment * dispatch.rodNodeCount;
    for (uint node = 0u;
         node < dispatch.rodNodeCount;
         ++node) {
        const uint offset = rodVelocityBase + 3u * node;
        candidateRodNodes[rodNodeBase + node].velocity.xyz =
            float3(
                qualityVelocities[offset + 0u],
                qualityVelocities[offset + 1u],
                qualityVelocities[offset + 2u]
            );
    }
    const uint rodEdgeBase =
        environment * dispatch.rodEdgeCount;
    const uint rodTwistBase =
        rodVelocityBase + 3u * dispatch.rodNodeCount;
    for (uint edge = 0u;
         edge < dispatch.rodEdgeCount;
         ++edge) {
        candidateRodEdges[
            rodEdgeBase + edge
        ].twistAndRate.y =
            qualityVelocities[rodTwistBase + edge];
    }
    const uint constraintBase =
        environment * dispatch.constraintStride;
    const uint manifoldPointBase =
        environment * dispatch.manifoldStride *
        MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY;
    for (uint localConstraint = 0u;
         localConstraint < contactStatus.requiredConstraints;
         ++localConstraint) {
        const float3 impulse = float3(
            qualityImpulses[
                vectorBase + 3u * localConstraint + 0u
            ],
            qualityImpulses[
                vectorBase + 3u * localConstraint + 1u
            ],
            qualityImpulses[
                vectorBase + 3u * localConstraint + 2u
            ]
        );
        device MRContactConstraintGPU& contact =
            contacts[constraintBase + localConstraint];
        contact.impulses.xyz = impulse;
        if ((contact.flags &
             MR_CONSTRAINT_FLAG_GENERALIZED) != 0u) {
            continue;
        } else if ((contact.flags &
             MR_CONSTRAINT_FLAG_ROD_ENDPOINT) != 0u) {
            rodWitnesses[
                rodConstraintWitnessIndices[
                    constraintBase + localConstraint
                ]
            ].impulses.xyz = impulse;
        } else {
            const MRContactPointMetaGPU metadata =
                contactMetadata[
                    constraintBase + localConstraint
                ];
            const uint point =
                manifoldPointBase +
                metadata.manifoldIndex *
                    MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY +
                metadata.pointIndex;
            candidateManifoldPoints[point].impulses.xyz =
                impulse;
        }
    }
    contactStatus.solverIterations =
        quality.newtonIterations;
    contactStatus.qualityNewtonIterations =
        quality.newtonIterations;
    contactStatus.qualityPCGIterations =
        quality.pcgIterations;
    contactStatus.qualityLineSearchBacktracks =
        quality.lineSearchBacktracks;
    contactStatus.qualitySolvePath = quality.solvePath;
    contactStatus.residuals = float4(
        abs(quality.certificates1.z),
        quality.certificates0.x,
        quality.certificates0.y,
        quality.certificates1.x
    );
    contactStatus.qualityCertificates =
        quality.certificates0;
    contactStatus.qualityDiagnostics = float4(
        quality.certificates1.x,
        quality.certificates1.y,
        quality.certificates1.z,
        max(quality.numerics.x, quality.numerics.y)
    );
    contactStatus.diagnostics.z = quality.numerics.z;
    contactStatus.diagnostics.w = quality.numerics.w;
    statuses[environment] = contactStatus;
}

// Publishes invocation-level persistent-worker evidence after every queued
// problem has completed. This stage is deliberately separate from the solver:
// no worker contends on per-environment status, and queue diagnostics cannot
// perturb the primal state.
kernel void mr_world_publish_unified_quality_queue_status(
    device const MRMetalWorldContactDispatchGPU& dispatch [[buffer(0)]],
    device const MRUnifiedQualityWorkQueueGPU& queue [[buffer(1)]],
    device MRMetalWorldContactStatusGPU* statuses [[buffer(2)]],
    const uint environment [[thread_position_in_grid]]
) {
    if (environment >= dispatch.environmentCount) {
        return;
    }
    MRMetalWorldContactStatusGPU status = statuses[environment];
    status.workerPackets = queue.packetsProcessed;
    status.workerHighWater = max(
        status.workerHighWater,
        queue.count
    );
    status.workerEmptyPulls = queue.emptyPulls;
    status.queueFlags |= MR_WORLD_QUEUE_PERSISTENT_WORKER;
    if (status.code == MR_STEP_SUCCESS &&
        queue.packetsProcessed != queue.count) {
        status.code = MR_STEP_DID_NOT_CONVERGE;
        status.firstFailingConstraint =
            MR_UNIFIED_QUALITY_INVALID_INDEX;
    }
    statuses[environment] = status;
}

// Standalone mathematical boundary for the exact Temporal Cone local block.
// It deliberately calls the same conditioned inverse and friction projection
// helpers as the production island solver instead of maintaining test-only
// copies of the Metal mathematics.
kernel void numi_temporal_cone_probe(
    device const NumiTemporalConeProbeInput* inputs [[buffer(0)]],
    device NumiTemporalConeProbeOutput* outputs [[buffer(1)]],
    constant uint& problemCount [[buffer(2)]],
    const uint problem [[thread_position_in_grid]]
) {
    if (problem >= problemCount) {
        return;
    }

    const NumiTemporalConeProbeInput input = inputs[problem];
    NumiTemporalConeProbeOutput output = {};
    output.status.x = NUMI_TEMPORAL_CONE_PROBE_INVALID_INPUT;
    if (input.control.x != NUMI_TEMPORAL_CONE_PROBE_ABI_VERSION) {
        output.status.x = NUMI_TEMPORAL_CONE_PROBE_INVALID_ABI;
        outputs[problem] = output;
        return;
    }
    if (input.control.y == 0u ||
        input.control.y > NUMI_TEMPORAL_CONE_PROBE_MAX_ITERATIONS ||
        input.control.z > 1u ||
        !finite4(input.responseRow0) ||
        !finite4(input.responseRow1) ||
        !finite4(input.responseRow2) ||
        !finite4(input.freeVelocityAndFrictionU) ||
        !finite4(input.warmImpulseAndFrictionV) ||
        !finite4(input.limits) ||
        input.freeVelocityAndFrictionU.w < 0.0f ||
        input.warmImpulseAndFrictionV.w < 0.0f ||
        input.limits.x < 0.0f) {
        outputs[problem] = output;
        return;
    }

    float response[3][3] = {
        {
            input.responseRow0.x,
            input.responseRow0.y,
            input.responseRow0.z,
        },
        {
            input.responseRow1.x,
            input.responseRow1.y,
            input.responseRow1.z,
        },
        {
            input.responseRow2.x,
            input.responseRow2.y,
            input.responseRow2.z,
        },
    };
    float inverse[3][3];
    if (!invert3x3(response, inverse)) {
        output.status.x =
            NUMI_TEMPORAL_CONE_PROBE_FACTORIZATION_FAILED;
        outputs[problem] = output;
        return;
    }

    MREvaluatedConstraintIRConeGPU cone = {};
    cone.effectiveFrictionU = input.freeVelocityAndFrictionU.w;
    cone.effectiveFrictionV = input.warmImpulseAndFrictionV.w;
    cone.maximumNormalImpulse = input.limits.x;
    const float authoredConeViolation = input.control.z != 0u
        ? frictionConeViolationValues(
              input.warmImpulseAndFrictionV.xyz,
              cone.effectiveFrictionU,
              cone.effectiveFrictionV,
              cone.maximumNormalImpulse
          )
        : 0.0f;

    float3 impulse = projectFrictionCone(
        input.warmImpulseAndFrictionV.xyz,
        cone
    );
    float maximumDelta = 0.0f;
    for (uint iteration = 0u;
         iteration < input.control.y;
         ++iteration) {
        const float3 responseImpulse = float3(
            dot(input.responseRow0.xyz, impulse),
            dot(input.responseRow1.xyz, impulse),
            dot(input.responseRow2.xyz, impulse)
        );
        const float3 rhs =
            -(input.freeVelocityAndFrictionU.xyz + responseImpulse);
        const float3 proposed = impulse + float3(
            inverse[0][0] * rhs.x +
                inverse[0][1] * rhs.y +
                inverse[0][2] * rhs.z,
            inverse[1][0] * rhs.x +
                inverse[1][1] * rhs.y +
                inverse[1][2] * rhs.z,
            inverse[2][0] * rhs.x +
                inverse[2][1] * rhs.y +
                inverse[2][2] * rhs.z
        );
        const float3 candidate = projectFrictionCone(proposed, cone);
        const float3 delta = candidate - impulse;
        maximumDelta = max(
            abs(delta.x),
            max(abs(delta.y), abs(delta.z))
        );
        impulse = candidate;
        if (!finite3(impulse) || !isfinite(maximumDelta)) {
            output.status.x =
                NUMI_TEMPORAL_CONE_PROBE_NONFINITE_RESULT;
            outputs[problem] = output;
            return;
        }
    }

    const float3 residual =
        input.freeVelocityAndFrictionU.xyz + float3(
            dot(input.responseRow0.xyz, impulse),
            dot(input.responseRow1.xyz, impulse),
            dot(input.responseRow2.xyz, impulse)
        );
    const float coneViolation = frictionConeViolationValues(
        impulse,
        cone.effectiveFrictionU,
        cone.effectiveFrictionV,
        cone.maximumNormalImpulse
    );

    output.impulseAndDelta = float4(impulse, maximumDelta);
    output.inverseRow0 = float4(
        inverse[0][0], inverse[0][1], inverse[0][2],
        authoredConeViolation
    );
    output.inverseRow1 = float4(
        inverse[1][0], inverse[1][1], inverse[1][2], 0.0f
    );
    output.inverseRow2 = float4(
        inverse[2][0], inverse[2][1], inverse[2][2], 0.0f
    );
    output.residualAndConeViolation = float4(
        residual,
        coneViolation
    );
    output.status = uint4(
        NUMI_TEMPORAL_CONE_PROBE_SUCCESS,
        input.control.y,
        0u,
        0u
    );
    outputs[problem] = output;
}

inline float3 temporalConeIslandResidual(
    device const float* matrix,
    const uint matrixBase,
    const uint row,
    const uint contactCount,
    const float3 freeVelocity,
    threadgroup const float4* impulses
) {
    float3 residual = freeVelocity;
    for (uint sourceContact = 0u;
         sourceContact < contactCount;
         ++sourceContact) {
        const float3 source = impulses[sourceContact].xyz;
        const uint column = 3u * sourceContact;
        residual.x +=
            matrix[
                matrixBase +
                (row + 0u) * NUMI_TEMPORAL_CONE_ISLAND_MAX_ROWS +
                column + 0u
            ] * source.x +
            matrix[
                matrixBase +
                (row + 0u) * NUMI_TEMPORAL_CONE_ISLAND_MAX_ROWS +
                column + 1u
            ] * source.y +
            matrix[
                matrixBase +
                (row + 0u) * NUMI_TEMPORAL_CONE_ISLAND_MAX_ROWS +
                column + 2u
            ] * source.z;
        residual.y +=
            matrix[
                matrixBase +
                (row + 1u) * NUMI_TEMPORAL_CONE_ISLAND_MAX_ROWS +
                column + 0u
            ] * source.x +
            matrix[
                matrixBase +
                (row + 1u) * NUMI_TEMPORAL_CONE_ISLAND_MAX_ROWS +
                column + 1u
            ] * source.y +
            matrix[
                matrixBase +
                (row + 1u) * NUMI_TEMPORAL_CONE_ISLAND_MAX_ROWS +
                column + 2u
            ] * source.z;
        residual.z +=
            matrix[
                matrixBase +
                (row + 2u) * NUMI_TEMPORAL_CONE_ISLAND_MAX_ROWS +
                column + 0u
            ] * source.x +
            matrix[
                matrixBase +
                (row + 2u) * NUMI_TEMPORAL_CONE_ISLAND_MAX_ROWS +
                column + 1u
            ] * source.y +
            matrix[
                matrixBase +
                (row + 2u) * NUMI_TEMPORAL_CONE_ISLAND_MAX_ROWS +
                column + 2u
            ] * source.z;
    }
    return residual;
}

inline float temporalConeViolation(
    const float3 impulse,
    const float frictionU,
    const float frictionV,
    const float maximumNormalImpulse
) {
    return frictionConeViolationValues(
        impulse,
        frictionU,
        frictionV,
        maximumNormalImpulse
    );
}

// Exact first-order variational-inequality gap for the authored elliptic
// Coulomb set. For an uncapped cone, dual feasibility is
// r_n >= ||(mu_u r_u, mu_v r_v)|| and complementarity is lambda.r = 0.
// For a capped cone, minimizing dot(r, z) over the complete feasible set gives
// cap * min(r_n - ||(mu_u r_u, mu_v r_v)||, 0). Divide the work gap by a
// three-axis impulse support scale so this certificate has velocity units and
// can share the KKT tolerance without becoming small through the step size.
inline float temporalConeComplementarityResidual(
    const float3 impulse,
    const float3 residual,
    const float frictionU,
    const float frictionV,
    const float maximumNormalImpulse
) {
    const float2 friction = max(
        float2(frictionU, frictionV),
        float2(0.0f)
    );
    const float dualTangent = scaledLength2(
        friction * residual.yz
    );
    const float dualSlope = residual.x - dualTangent;
    const float work = dot(impulse, residual);
    float supportScale = max(
        abs(impulse.x),
        max(abs(impulse.y), abs(impulse.z))
    );
    float dualViolation = max(-dualSlope, 0.0f);
    float gap = work;
    if (maximumNormalImpulse > 0.0f) {
        supportScale = max(
            supportScale,
            maximumNormalImpulse * max(
                1.0f,
                max(friction.x, friction.y)
            )
        );
        dualViolation = 0.0f;
        gap -= maximumNormalImpulse * min(dualSlope, 0.0f);
    }
    return max(
        dualViolation,
        abs(gap) / (3.0f * max(1.0f, supportScale))
    );
}

// One SIMD32 group owns one complete dense contact-space island. Every lane
// owns one three-row cone block. Matrix actions visit source contacts in
// canonical order; no floating-point atomics or unordered reductions enter
// the physical update. The iteration is block-Jacobi so all contacts consume
// one immutable impulse generation and publish the next generation together.
kernel void numi_temporal_cone_island_solve(
    device const NumiTemporalConeIslandHeader* headers [[buffer(0)]],
    device const float* matrices [[buffer(1)]],
    device const NumiTemporalConeIslandContact* contacts [[buffer(2)]],
    device float4* outputImpulses [[buffer(3)]],
    device NumiTemporalConeIslandStatus* outputStatuses [[buffer(4)]],
    constant uint& problemCount [[buffer(5)]],
    const uint problem [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    if (problem >= problemCount) {
        return;
    }
    const NumiTemporalConeIslandHeader header = headers[problem];
    const uint contactCount = header.control.y;
    const uint contactBase =
        problem * NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS;
    const uint matrixBase =
        problem * NUMI_TEMPORAL_CONE_ISLAND_MATRIX_ELEMENTS;
    const bool active = lane < contactCount;
    if (lane < NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS) {
        outputImpulses[contactBase + lane] = float4(0.0f);
    }

    uint localFailure = NUMI_TEMPORAL_CONE_ISLAND_SUCCESS;
    if (header.control.x != NUMI_TEMPORAL_CONE_ISLAND_ABI_VERSION) {
        localFailure = NUMI_TEMPORAL_CONE_ISLAND_INVALID_ABI;
    } else if (
        contactCount == 0u ||
        contactCount > NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS ||
        header.control.z == 0u ||
        header.control.z > header.control.w ||
        header.control.w > NUMI_TEMPORAL_CONE_ISLAND_MAX_ITERATIONS ||
        !finite4(header.tolerances) ||
        !(header.tolerances.x > 0.0f) ||
        header.tolerances.y < 0.0f ||
        !(header.tolerances.z > 0.0f) ||
        header.tolerances.z > 1.0f
    ) {
        localFailure = NUMI_TEMPORAL_CONE_ISLAND_INVALID_INPUT;
    }

    NumiTemporalConeIslandContact contact = {};
    MREvaluatedConstraintIRConeGPU cone = {};
    float diagonal[3][3] = {};
    if (active && localFailure == NUMI_TEMPORAL_CONE_ISLAND_SUCCESS) {
        contact = contacts[contactBase + lane];
        if (!finite4(contact.freeVelocityAndFrictionU) ||
            !finite4(contact.warmImpulseAndFrictionV) ||
            !finite4(contact.limits) ||
            contact.freeVelocityAndFrictionU.w < 0.0f ||
            contact.warmImpulseAndFrictionV.w < 0.0f ||
            contact.limits.x < 0.0f) {
            localFailure = NUMI_TEMPORAL_CONE_ISLAND_INVALID_INPUT;
        }

        const uint row = 3u * lane;
        const uint dimension = 3u * contactCount;
        for (uint localRow = 0u;
             localRow < 3u &&
                 localFailure == NUMI_TEMPORAL_CONE_ISLAND_SUCCESS;
             ++localRow) {
            for (uint column = 0u;
                 column < dimension;
                 ++column) {
                const float value = matrices[
                    matrixBase +
                    (row + localRow) *
                        NUMI_TEMPORAL_CONE_ISLAND_MAX_ROWS +
                    column
                ];
                const float transposeValue = matrices[
                    matrixBase +
                    column * NUMI_TEMPORAL_CONE_ISLAND_MAX_ROWS +
                    row + localRow
                ];
                const float symmetryScale = max(
                    1.0f,
                    max(abs(value), abs(transposeValue))
                );
                if (!isfinite(value) ||
                    !isfinite(transposeValue) ||
                    abs(value - transposeValue) >
                        64.0f * kFloatEpsilon * symmetryScale) {
                    localFailure =
                        NUMI_TEMPORAL_CONE_ISLAND_INVALID_INPUT;
                    break;
                }
            }
        }
        if (localFailure == NUMI_TEMPORAL_CONE_ISLAND_SUCCESS) {
            for (uint localRow = 0u; localRow < 3u; ++localRow) {
                for (uint localColumn = 0u;
                     localColumn < 3u;
                     ++localColumn) {
                    diagonal[localRow][localColumn] = matrices[
                        matrixBase +
                        (row + localRow) *
                            NUMI_TEMPORAL_CONE_ISLAND_MAX_ROWS +
                        row + localColumn
                    ];
                }
            }
            if (!positiveSemidefinite3x3(diagonal)) {
                localFailure =
                    NUMI_TEMPORAL_CONE_ISLAND_FACTORIZATION_FAILED;
            }
        }
        if (localFailure == NUMI_TEMPORAL_CONE_ISLAND_SUCCESS) {
            const float3 diagonalRoot = sqrt(max(
                float3(
                    diagonal[0][0],
                    diagonal[1][1],
                    diagonal[2][2]
                ),
                float3(0.0f)
            ));
            for (uint source = lane + 1u;
                 source < contactCount &&
                     localFailure ==
                         NUMI_TEMPORAL_CONE_ISLAND_SUCCESS;
                 ++source) {
                const uint sourceRow = 3u * source;
                const float3 sourceDiagonalRoot = sqrt(max(
                    float3(
                        matrices[
                            matrixBase +
                            (sourceRow + 0u) *
                                NUMI_TEMPORAL_CONE_ISLAND_MAX_ROWS +
                            sourceRow + 0u
                        ],
                        matrices[
                            matrixBase +
                            (sourceRow + 1u) *
                                NUMI_TEMPORAL_CONE_ISLAND_MAX_ROWS +
                            sourceRow + 1u
                        ],
                        matrices[
                            matrixBase +
                            (sourceRow + 2u) *
                                NUMI_TEMPORAL_CONE_ISLAND_MAX_ROWS +
                            sourceRow + 2u
                        ]
                    ),
                    float3(0.0f)
                ));
                for (uint localRow = 0u; localRow < 3u; ++localRow) {
                    const float3 coupling = float3(
                        matrices[
                            matrixBase +
                            (row + localRow) *
                                NUMI_TEMPORAL_CONE_ISLAND_MAX_ROWS +
                            sourceRow + 0u
                        ],
                        matrices[
                            matrixBase +
                            (row + localRow) *
                                NUMI_TEMPORAL_CONE_ISLAND_MAX_ROWS +
                            sourceRow + 1u
                        ],
                        matrices[
                            matrixBase +
                            (row + localRow) *
                                NUMI_TEMPORAL_CONE_ISLAND_MAX_ROWS +
                            sourceRow + 2u
                        ]
                    );
                    if (!temporalConePairCurvatureValid3(
                            diagonalRoot[localRow],
                            coupling,
                            sourceDiagonalRoot
                        )) {
                        localFailure =
                            NUMI_TEMPORAL_CONE_ISLAND_FACTORIZATION_FAILED;
                        break;
                    }
                }
            }
        }
        cone.effectiveFrictionU = contact.freeVelocityAndFrictionU.w;
        cone.effectiveFrictionV = contact.warmImpulseAndFrictionV.w;
        cone.maximumNormalImpulse = contact.limits.x;
    }

    const uint initialFailureKey = simd_min(
        localFailure == NUMI_TEMPORAL_CONE_ISLAND_SUCCESS
        ? 0xffffffffu
        : localFailure
    );
    const uint initialFailure = initialFailureKey == 0xffffffffu
        ? NUMI_TEMPORAL_CONE_ISLAND_SUCCESS
        : initialFailureKey;
    if (initialFailure != NUMI_TEMPORAL_CONE_ISLAND_SUCCESS) {
        if (lane == 0u) {
            NumiTemporalConeIslandStatus status = {};
            status.control = uint4(
                initialFailure,
                0u,
                0u,
                contactCount
            );
            outputStatuses[problem] = status;
        }
        return;
    }

    // One positive scalar metric owns all three axes of a contact cone. This
    // preserves the exact Euclidean projected-gradient/KKT fixed point. The
    // scalar is the reciprocal of the largest absolute operator-row sum in
    // the contact block. For symmetric A, the repeated block scalars form a
    // diagonal majorizer D with D - A positive semidefinite, so the parallel
    // step is stable without changing the solved contact problem.
    float laneAbsoluteRowBound = 0.0f;
    if (active) {
        const uint row = 3u * lane;
        const uint dimension = 3u * contactCount;
        for (uint outputAxis = 0u; outputAxis < 3u; ++outputAxis) {
            float rowSum = 0.0f;
            for (uint column = 0u; column < dimension; ++column) {
                rowSum += abs(matrices[
                    matrixBase +
                    (row + outputAxis) *
                        NUMI_TEMPORAL_CONE_ISLAND_MAX_ROWS +
                    column
                ]);
            }
            laneAbsoluteRowBound = max(
                laneAbsoluteRowBound,
                rowSum
            );
        }
    }
    float stepScale = 0.0f;
    if (active) {
        if (!isfinite(laneAbsoluteRowBound) ||
            !(laneAbsoluteRowBound > 0.0f)) {
            localFailure =
                NUMI_TEMPORAL_CONE_ISLAND_NONFINITE_RESULT;
        } else {
            stepScale = 1.0f / max(
                laneAbsoluteRowBound,
                kMatrixFloor
            );
            if (!isfinite(stepScale) || !(stepScale > 0.0f)) {
                localFailure =
                    NUMI_TEMPORAL_CONE_ISLAND_NONFINITE_RESULT;
            }
        }
    }
    const float effectiveRelaxation = header.tolerances.z;

    threadgroup float4 current[
        NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS
    ];
    threadgroup float4 next[
        NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS
    ];
    threadgroup float4 previous[
        NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS
    ];
    threadgroup float4 search[
        NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS
    ];
    threadgroup float4 checkpoint[
        NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS
    ];
    if (active) {
        const float3 projectedWarmStart = projectFrictionCone(
            contact.warmImpulseAndFrictionV.xyz,
            cone
        );
        if (!finite3(projectedWarmStart)) {
            localFailure = NUMI_TEMPORAL_CONE_ISLAND_NONFINITE_RESULT;
            current[lane] = float4(0.0f);
        } else {
            current[lane] = float4(projectedWarmStart, 0.0f);
        }
    } else {
        current[lane] = float4(0.0f);
    }
    checkpoint[lane] = current[lane];
    previous[lane] = current[lane];
    search[lane] = current[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint checkpointFailure = simd_max(localFailure);
    if (checkpointFailure != NUMI_TEMPORAL_CONE_ISLAND_SUCCESS) {
        if (active) {
            outputImpulses[contactBase + lane] = float4(0.0f);
        }
        if (lane == 0u) {
            NumiTemporalConeIslandStatus status = {};
            status.control = uint4(
                checkpointFailure,
                0u,
                0u,
                contactCount
            );
            outputStatuses[problem] = status;
        }
        return;
    }

    uint completedIterations = 0u;
    const bool accelerationEnabled = effectiveRelaxation == 1.0f;
    float accelerationClock = 1.0f;
    float momentumScale = 0.0f;
    uint accelerationRestarts = 0u;
    for (uint iteration = 0u;
         iteration < header.control.w;
         ++iteration) {
        float laneNaturalDelta = 0.0f;
        float laneScale = 1.0f;
        float laneRestartMeasure = 0.0f;
        localFailure = NUMI_TEMPORAL_CONE_ISLAND_SUCCESS;
        const bool extrapolated = momentumScale != 0.0f;
        if (active && extrapolated) {
            search[lane] = current[lane] + momentumScale * (
                current[lane] - previous[lane]
            );
        }
        if (extrapolated) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (active) {
            const uint row = 3u * lane;
            const float3 impulse = extrapolated
                ? search[lane].xyz
                : current[lane].xyz;
            const float3 residual = extrapolated
                ? temporalConeIslandResidual(
                    matrices,
                    matrixBase,
                    row,
                    contactCount,
                    contact.freeVelocityAndFrictionU.xyz,
                    search
                )
                : temporalConeIslandResidual(
                    matrices,
                    matrixBase,
                    row,
                    contactCount,
                    contact.freeVelocityAndFrictionU.xyz,
                    current
                );
            const float3 proposed = impulse - stepScale * residual;
            const float3 projected = projectFrictionCone(proposed, cone);
            const float3 naturalDelta = projected - impulse;
            const float3 candidate =
                impulse + effectiveRelaxation * naturalDelta;
            if (!finite3(residual) ||
                !finite3(projected) ||
                !finite3(candidate)) {
                localFailure =
                    NUMI_TEMPORAL_CONE_ISLAND_NONFINITE_RESULT;
                next[lane] = current[lane];
            } else {
                next[lane] = float4(candidate, 0.0f);
                if (extrapolated) {
                    laneRestartMeasure = dot(
                        impulse - candidate,
                        candidate - current[lane].xyz
                    );
                }
                laneNaturalDelta = max(
                    abs(naturalDelta.x),
                    max(abs(naturalDelta.y), abs(naturalDelta.z))
                ) / stepScale;
                const float3 response =
                    residual - contact.freeVelocityAndFrictionU.xyz;
                laneScale = max(
                    1.0f,
                    max(
                        max(
                            abs(contact.freeVelocityAndFrictionU.x),
                            max(
                                abs(contact.freeVelocityAndFrictionU.y),
                                abs(contact.freeVelocityAndFrictionU.z)
                            )
                        ),
                        max(
                            abs(response.x),
                            max(abs(response.y), abs(response.z))
                        )
                    )
                );
                if (!isfinite(laneNaturalDelta) ||
                    !isfinite(laneScale) ||
                    !isfinite(laneRestartMeasure)) {
                    localFailure =
                        NUMI_TEMPORAL_CONE_ISLAND_NONFINITE_RESULT;
                    next[lane] = current[lane];
                    laneNaturalDelta = 0.0f;
                    laneScale = 1.0f;
                    laneRestartMeasure = 0.0f;
                }
            }
        } else {
            next[lane] = float4(0.0f);
        }
        const uint iterationFailure = simd_max(localFailure);
        const float maximumNaturalDelta = simd_max(laneNaturalDelta);
        const float maximumScale = simd_max(laneScale);
        const float restartMeasure = simd_sum(laneRestartMeasure);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        completedIterations = iteration + 1u;
        if (iterationFailure != NUMI_TEMPORAL_CONE_ISLAND_SUCCESS) {
            localFailure = iterationFailure;
            break;
        }
        const float tolerance =
            header.tolerances.x +
            header.tolerances.y * maximumScale;
        const bool candidateReady =
            completedIterations >= header.control.z &&
            maximumNaturalDelta <= tolerance;
        float laneComplementarityDelta = 0.0f;
        if (active && candidateReady) {
            const uint row = 3u * lane;
            const float3 impulse = extrapolated
                ? search[lane].xyz
                : current[lane].xyz;
            const float3 residual = extrapolated
                ? temporalConeIslandResidual(
                    matrices,
                    matrixBase,
                    row,
                    contactCount,
                    contact.freeVelocityAndFrictionU.xyz,
                    search
                )
                : temporalConeIslandResidual(
                    matrices,
                    matrixBase,
                    row,
                    contactCount,
                    contact.freeVelocityAndFrictionU.xyz,
                    current
                );
            laneComplementarityDelta =
                temporalConeComplementarityResidual(
                    impulse,
                    residual,
                    cone.effectiveFrictionU,
                    cone.effectiveFrictionV,
                    cone.maximumNormalImpulse
                );
            if (!isfinite(laneComplementarityDelta)) {
                laneComplementarityDelta = INFINITY;
            }
        }
        const float maximumComplementarityDelta = simd_max(
            laneComplementarityDelta
        );
        const bool certifiedCandidateReady =
            candidateReady &&
            maximumComplementarityDelta <= tolerance;
        const bool iterationConverged =
            momentumScale == 0.0f &&
            certifiedCandidateReady;
        if (iterationConverged) {
            break;
        }
        previous[lane] = current[lane];
        current[lane] = next[lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (accelerationEnabled) {
            // Restart on misaligned proximal progress, provisional tolerance,
            // or the fixed bound. Only beta=0 can pass the exact KKT gate.
            const bool restartAcceleration =
                certifiedCandidateReady ||
                restartMeasure > 0.0f ||
                completedIterations < 16u ||
                completedIterations % 64u == 0u;
            if (restartAcceleration) {
                if (extrapolated) {
                    ++accelerationRestarts;
                }
                accelerationClock = 1.0f;
                momentumScale = 0.0f;
            } else {
                const float nextClock = 0.5f * (
                    1.0f + sqrt(fma(
                        4.0f * accelerationClock,
                        accelerationClock,
                        1.0f
                    ))
                );
                momentumScale =
                    (accelerationClock - 1.0f) / nextClock;
                accelerationClock = nextClock;
            }
        }
    }

    float laneNaturalResidual = 0.0f;
    float laneConeViolation = 0.0f;
    float laneKKTScale = 1.0f;
    float laneImpulseScale = 0.0f;
    float laneRawResidual = 0.0f;
    float laneObjective = 0.0f;
    float laneComplementarityResidual = 0.0f;
    if (active &&
        localFailure == NUMI_TEMPORAL_CONE_ISLAND_SUCCESS) {
        const uint row = 3u * lane;
        const float3 impulse = current[lane].xyz;
        const float3 residual = temporalConeIslandResidual(
            matrices,
            matrixBase,
            row,
            contactCount,
            contact.freeVelocityAndFrictionU.xyz,
            current
        );
        const float3 proposed = impulse - stepScale * residual;
        const float3 projected = projectFrictionCone(proposed, cone);
        const float3 naturalDelta = projected - impulse;
        laneNaturalResidual = max(
            abs(naturalDelta.x),
            max(abs(naturalDelta.y), abs(naturalDelta.z))
        ) / stepScale;
        laneConeViolation = temporalConeViolation(
            impulse,
            cone.effectiveFrictionU,
            cone.effectiveFrictionV,
            cone.maximumNormalImpulse
        );
        laneImpulseScale = max(
            abs(impulse.x),
            max(abs(impulse.y), abs(impulse.z))
        );
        laneRawResidual = max(
            abs(residual.x),
            max(abs(residual.y), abs(residual.z))
        );
        const float3 response =
            residual - contact.freeVelocityAndFrictionU.xyz;
        laneKKTScale = max(
            1.0f,
            max(
                max(
                    abs(contact.freeVelocityAndFrictionU.x),
                    max(
                        abs(contact.freeVelocityAndFrictionU.y),
                        abs(contact.freeVelocityAndFrictionU.z)
                    )
                ),
                max(
                    abs(response.x),
                    max(abs(response.y), abs(response.z))
                )
            )
        );
        laneObjective =
            0.5f * dot(impulse, response) +
            dot(impulse, contact.freeVelocityAndFrictionU.xyz);
        laneComplementarityResidual =
            temporalConeComplementarityResidual(
                impulse,
                residual,
                cone.effectiveFrictionU,
                cone.effectiveFrictionV,
                cone.maximumNormalImpulse
            );
        if (!finite3(residual) ||
            !finite3(projected) ||
            !finite3(naturalDelta) ||
            !isfinite(laneNaturalResidual) ||
            !isfinite(laneConeViolation) ||
            !isfinite(laneImpulseScale) ||
            !isfinite(laneRawResidual) ||
            !isfinite(laneKKTScale) ||
            !isfinite(laneObjective) ||
            !isfinite(laneComplementarityResidual)) {
            localFailure =
                NUMI_TEMPORAL_CONE_ISLAND_NONFINITE_RESULT;
            laneNaturalResidual = 0.0f;
            laneConeViolation = 0.0f;
            laneImpulseScale = 0.0f;
            laneRawResidual = 0.0f;
            laneKKTScale = 1.0f;
            laneObjective = 0.0f;
            laneComplementarityResidual = 0.0f;
        }
    }

    const uint diagnosticFailure = simd_max(localFailure);
    const float maximumNaturalResidual = simd_max(laneNaturalResidual);
    const float maximumConeViolation = simd_max(laneConeViolation);
    const float maximumKKTScale = simd_max(laneKKTScale);
    const float maximumImpulseScale = simd_max(laneImpulseScale);
    const float maximumRawResidual = simd_max(laneRawResidual);
    const float objective = simd_sum(laneObjective);
    const float maximumComplementarityResidual = simd_max(
        laneComplementarityResidual
    );
    const float kktTolerance =
        header.tolerances.x +
        header.tolerances.y * maximumKKTScale;
    const float coneTolerance =
        header.tolerances.x +
        header.tolerances.y * max(1.0f, maximumImpulseScale);
    const float objectiveTolerance =
        3.0f * float(contactCount) *
        kktTolerance * maximumImpulseScale;
    const bool finiteCertificate =
        isfinite(maximumNaturalResidual) &&
        isfinite(maximumConeViolation) &&
        isfinite(maximumKKTScale) &&
        maximumKKTScale > 0.0f &&
        isfinite(maximumImpulseScale) &&
        isfinite(maximumRawResidual) &&
        isfinite(objective) &&
        isfinite(maximumComplementarityResidual) &&
        isfinite(kktTolerance) &&
        isfinite(coneTolerance) &&
        isfinite(objectiveTolerance);
    const uint finalFailure =
        diagnosticFailure != NUMI_TEMPORAL_CONE_ISLAND_SUCCESS
        ? diagnosticFailure
        : finiteCertificate
        ? NUMI_TEMPORAL_CONE_ISLAND_SUCCESS
        : NUMI_TEMPORAL_CONE_ISLAND_NONFINITE_RESULT;
    const float normalizedNaturalResidual = finiteCertificate
        ? maximumNaturalResidual / maximumKKTScale
        : 0.0f;
    const float normalizedComplementarityResidual = finiteCertificate
        ? maximumComplementarityResidual / maximumKKTScale
        : 0.0f;
    const bool finalConverged =
        finalFailure == NUMI_TEMPORAL_CONE_ISLAND_SUCCESS &&
        maximumNaturalResidual <= kktTolerance &&
        maximumComplementarityResidual <= kktTolerance &&
        maximumConeViolation <= coneTolerance &&
        objective <= objectiveTolerance;
    if (active) {
        outputImpulses[contactBase + lane] = finalConverged
            ? current[lane]
            : checkpoint[lane];
    }
    if (lane == 0u) {
        NumiTemporalConeIslandStatus status = {};
        const uint code =
            finalFailure != NUMI_TEMPORAL_CONE_ISLAND_SUCCESS
            ? finalFailure
            : finalConverged
            ? NUMI_TEMPORAL_CONE_ISLAND_SUCCESS
            : NUMI_TEMPORAL_CONE_ISLAND_DID_NOT_CONVERGE;
        status.control = uint4(
            code,
            completedIterations,
            finalConverged ? 1u : 0u,
            contactCount
        );
        status.residuals = finiteCertificate
            ? float4(
                normalizedNaturalResidual,
                maximumConeViolation,
                maximumImpulseScale,
                objective
            )
            : float4(0.0f);
        status.diagnostics = finiteCertificate
            ? float4(
                maximumRawResidual,
                effectiveRelaxation,
                normalizedComplementarityResidual,
                float(accelerationRestarts)
            )
            : float4(0.0f);
        outputStatuses[problem] = status;
    }
}

inline float3 temporalConeStreamResidual(
    device const uint* columnIndices,
    device const float* blockValues,
    const uint blockBase,
    const uint rowBegin,
    const uint rowEnd,
    const float3 freeVelocity,
    threadgroup const float4* impulses
) {
    float3 residual = freeVelocity;
    for (uint relativeBlock = rowBegin;
         relativeBlock < rowEnd;
         ++relativeBlock) {
        const uint block = blockBase + relativeBlock;
        const uint sourceContact = columnIndices[block];
        const float3 source = impulses[sourceContact].xyz;
        const uint valueBase =
            block * NUMI_TEMPORAL_CONE_STREAM_BLOCK_ELEMENTS;
        residual.x +=
            blockValues[valueBase + 0u] * source.x +
            blockValues[valueBase + 1u] * source.y +
            blockValues[valueBase + 2u] * source.z;
        residual.y +=
            blockValues[valueBase + 3u] * source.x +
            blockValues[valueBase + 4u] * source.y +
            blockValues[valueBase + 5u] * source.z;
        residual.z +=
            blockValues[valueBase + 6u] * source.x +
            blockValues[valueBase + 7u] * source.y +
            blockValues[valueBase + 8u] * source.z;
    }
    return residual;
}

inline uint temporalConeFindRelativeBlock(
    device const uint* rowOffsets,
    device const uint* columnIndices,
    const uint rowOffsetBase,
    const uint blockBase,
    const uint row,
    const uint source
) {
    const uint rowBegin = rowOffsets[rowOffsetBase + row];
    const uint rowEnd = rowOffsets[rowOffsetBase + row + 1u];
    uint lower = rowBegin;
    uint upper = rowEnd;
    while (lower < upper) {
        const uint middle = lower + (upper - lower) / 2u;
        const uint candidateSource = columnIndices[
            blockBase + middle
        ];
        if (candidateSource < source) {
            lower = middle + 1u;
        } else {
            upper = middle;
        }
    }
    return lower < rowEnd &&
        columnIndices[blockBase + lower] == source
        ? lower
        : 0xffffffffu;
}

// One SIMD32 group owns one packed block-CSR contact island. Rows and source
// contacts are both canonical and immutable throughout the solve. Missing
// blocks are exact zeros, so the streamed matrix action is mathematically
// identical to the dense Delassus action without materializing those zeros.
kernel void numi_temporal_cone_stream_solve(
    device const NumiTemporalConeStreamHeader* headers [[buffer(0)]],
    device const uint* rowOffsets [[buffer(1)]],
    device const uint* columnIndices [[buffer(2)]],
    device const float* blockValues [[buffer(3)]],
    device const NumiTemporalConeIslandContact* contacts [[buffer(4)]],
    device float4* outputImpulses [[buffer(5)]],
    device NumiTemporalConeIslandStatus* outputStatuses [[buffer(6)]],
    constant uint& problemCount [[buffer(7)]],
    // x contacts, y row offsets, z blocks, w output impulses.
    constant uint4& capacities [[buffer(8)]],
    const uint problem [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    if (problem >= problemCount) {
        return;
    }
    const NumiTemporalConeStreamHeader header = headers[problem];
    const uint contactCount = header.control.y;
    const uint contactBase = header.ranges.x;
    const uint rowOffsetBase = header.ranges.y;
    const uint blockBase = header.ranges.z;
    const uint blockCount = header.ranges.w;
    const bool active = lane < contactCount;
    const bool contactRangeValid =
        contactBase <= capacities.x &&
        contactCount <= capacities.x - contactBase &&
        contactBase <= capacities.w &&
        contactCount <= capacities.w - contactBase;
    const bool rowRangeValid =
        rowOffsetBase <= capacities.y &&
        contactCount < capacities.y - rowOffsetBase;
    const bool blockRangeValid =
        blockBase <= capacities.z &&
        blockCount <= capacities.z - blockBase;

    uint localFailure = NUMI_TEMPORAL_CONE_ISLAND_SUCCESS;
    if (header.control.x != NUMI_TEMPORAL_CONE_STREAM_ABI_VERSION) {
        localFailure = NUMI_TEMPORAL_CONE_ISLAND_INVALID_ABI;
    } else if (
        contactCount == 0u ||
        contactCount > NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS ||
        blockCount < contactCount ||
        blockCount > NUMI_TEMPORAL_CONE_STREAM_MAX_BLOCKS ||
        !contactRangeValid ||
        !rowRangeValid ||
        !blockRangeValid ||
        header.control.z == 0u ||
        header.control.z > header.control.w ||
        header.control.w > NUMI_TEMPORAL_CONE_ISLAND_MAX_ITERATIONS ||
        !finite4(header.tolerances) ||
        !(header.tolerances.x > 0.0f) ||
        header.tolerances.y < 0.0f ||
        !(header.tolerances.z > 0.0f) ||
        header.tolerances.z > 1.0f
    ) {
        localFailure = NUMI_TEMPORAL_CONE_ISLAND_INVALID_INPUT;
    }
    const uint headerFailureKey = simd_min(
        localFailure == NUMI_TEMPORAL_CONE_ISLAND_SUCCESS
        ? 0xffffffffu
        : localFailure
    );
    const uint headerFailure = headerFailureKey == 0xffffffffu
        ? NUMI_TEMPORAL_CONE_ISLAND_SUCCESS
        : headerFailureKey;
    if (headerFailure != NUMI_TEMPORAL_CONE_ISLAND_SUCCESS) {
        if (active && contactRangeValid) {
            outputImpulses[contactBase + lane] = float4(0.0f);
        }
        if (lane == 0u) {
            NumiTemporalConeIslandStatus status = {};
            status.control = uint4(
                headerFailure,
                0u,
                0u,
                contactCount
            );
            outputStatuses[problem] = status;
        }
        return;
    }

    uint rowBegin = 0u;
    uint rowEnd = 0u;
    if (active) {
        rowBegin = rowOffsets[rowOffsetBase + lane];
        rowEnd = rowOffsets[rowOffsetBase + lane + 1u];
        if (rowBegin >= rowEnd ||
            rowBegin > blockCount ||
            rowEnd > blockCount) {
            localFailure = NUMI_TEMPORAL_CONE_ISLAND_INVALID_INPUT;
        }
        outputImpulses[contactBase + lane] = float4(0.0f);
    }
    const uint rowFailure = simd_max(localFailure);
    if (rowFailure != NUMI_TEMPORAL_CONE_ISLAND_SUCCESS) {
        if (lane == 0u) {
            NumiTemporalConeIslandStatus status = {};
            status.control = uint4(
                rowFailure,
                0u,
                0u,
                contactCount
            );
            outputStatuses[problem] = status;
        }
        return;
    }

    NumiTemporalConeIslandContact contact = {};
    MREvaluatedConstraintIRConeGPU cone = {};
    float diagonal[3][3] = {};
    if (active) {
        contact = contacts[contactBase + lane];
        if (!finite4(contact.freeVelocityAndFrictionU) ||
            !finite4(contact.warmImpulseAndFrictionV) ||
            !finite4(contact.limits) ||
            contact.freeVelocityAndFrictionU.w < 0.0f ||
            contact.warmImpulseAndFrictionV.w < 0.0f ||
            contact.limits.x < 0.0f) {
            localFailure = NUMI_TEMPORAL_CONE_ISLAND_INVALID_INPUT;
        }
        bool diagonalSeen = false;
        uint previousSource = 0u;
        bool firstSource = true;
        for (uint relativeBlock = rowBegin;
             relativeBlock < rowEnd;
             ++relativeBlock) {
            const uint block = blockBase + relativeBlock;
            const uint source = columnIndices[block];
            if (source >= contactCount ||
                (!firstSource && source <= previousSource)) {
                localFailure = NUMI_TEMPORAL_CONE_ISLAND_INVALID_INPUT;
                break;
            }
            firstSource = false;
            previousSource = source;
            const uint valueBase =
                block * NUMI_TEMPORAL_CONE_STREAM_BLOCK_ELEMENTS;
            for (uint element = 0u;
                 element < NUMI_TEMPORAL_CONE_STREAM_BLOCK_ELEMENTS;
                 ++element) {
                if (!isfinite(blockValues[valueBase + element])) {
                    localFailure =
                        NUMI_TEMPORAL_CONE_ISLAND_INVALID_INPUT;
                    break;
                }
            }
            if (source == lane) {
                diagonalSeen = true;
                for (uint row = 0u; row < 3u; ++row) {
                    for (uint column = 0u; column < 3u; ++column) {
                        diagonal[row][column] = blockValues[
                            valueBase + 3u * row + column
                        ];
                    }
                }
            }
        }
        if (!diagonalSeen) {
            localFailure = NUMI_TEMPORAL_CONE_ISLAND_INVALID_INPUT;
        }
    }
    const uint structureFailure = simd_max(localFailure);
    if (structureFailure != NUMI_TEMPORAL_CONE_ISLAND_SUCCESS) {
        if (lane == 0u) {
            NumiTemporalConeIslandStatus status = {};
            status.control = uint4(
                structureFailure,
                0u,
                0u,
                contactCount
            );
            outputStatuses[problem] = status;
        }
        return;
    }

    // Reuse the rollback checkpoint arena as immutable scalar diagonals during
    // preflight; every lane overwrites it with the projected warm start before
    // the first physical iteration.
    threadgroup float4 checkpoint[
        NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS
    ];
    checkpoint[lane] = active
        ? float4(
              sqrt(max(diagonal[0][0], 0.0f)),
              sqrt(max(diagonal[1][1], 0.0f)),
              sqrt(max(diagonal[2][2], 0.0f)),
              0.0f
          )
        : float4(0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active) {
        // Enforce A_ij = transpose(A_ji) directly on the packed operator.
        for (uint relativeBlock = rowBegin;
             relativeBlock < rowEnd &&
                 localFailure == NUMI_TEMPORAL_CONE_ISLAND_SUCCESS;
             ++relativeBlock) {
            const uint block = blockBase + relativeBlock;
            const uint source = columnIndices[block];
            const uint reverseRelative = temporalConeFindRelativeBlock(
                rowOffsets,
                columnIndices,
                rowOffsetBase,
                blockBase,
                source,
                lane
            );
            if (reverseRelative == 0xffffffffu) {
                localFailure = NUMI_TEMPORAL_CONE_ISLAND_INVALID_INPUT;
                break;
            }
            const uint reverseBlock = blockBase + reverseRelative;
            const uint valueBase =
                block * NUMI_TEMPORAL_CONE_STREAM_BLOCK_ELEMENTS;
            const uint reverseValueBase =
                reverseBlock * NUMI_TEMPORAL_CONE_STREAM_BLOCK_ELEMENTS;
            for (uint row = 0u; row < 3u; ++row) {
                for (uint column = 0u; column < 3u; ++column) {
                    const float value = blockValues[
                        valueBase + 3u * row + column
                    ];
                    const float reverseValue = blockValues[
                        reverseValueBase + 3u * column + row
                    ];
                    const float symmetryScale = max(
                        1.0f,
                        max(abs(value), abs(reverseValue))
                    );
                    if (abs(value - reverseValue) >
                        64.0f * kFloatEpsilon * symmetryScale) {
                        localFailure =
                            NUMI_TEMPORAL_CONE_ISLAND_INVALID_INPUT;
                        break;
                    }
                }
            }
            if (lane < source &&
                localFailure == NUMI_TEMPORAL_CONE_ISLAND_SUCCESS) {
                for (uint row = 0u; row < 3u; ++row) {
                    const float3 coupling = float3(
                        blockValues[valueBase + 3u * row + 0u],
                        blockValues[valueBase + 3u * row + 1u],
                        blockValues[valueBase + 3u * row + 2u]
                    );
                    if (!temporalConePairCurvatureValid3(
                            checkpoint[lane][row],
                            coupling,
                            checkpoint[source].xyz
                        )) {
                        localFailure =
                            NUMI_TEMPORAL_CONE_ISLAND_FACTORIZATION_FAILED;
                        break;
                    }
                }
            }
        }
        if (localFailure == NUMI_TEMPORAL_CONE_ISLAND_SUCCESS &&
            !positiveSemidefinite3x3(diagonal)) {
            localFailure =
                NUMI_TEMPORAL_CONE_ISLAND_FACTORIZATION_FAILED;
        }
        cone.effectiveFrictionU = contact.freeVelocityAndFrictionU.w;
        cone.effectiveFrictionV = contact.warmImpulseAndFrictionV.w;
        cone.maximumNormalImpulse = contact.limits.x;
    }
    const uint initialFailureKey = simd_min(
        localFailure == NUMI_TEMPORAL_CONE_ISLAND_SUCCESS
        ? 0xffffffffu
        : localFailure
    );
    const uint initialFailure = initialFailureKey == 0xffffffffu
        ? NUMI_TEMPORAL_CONE_ISLAND_SUCCESS
        : initialFailureKey;
    if (initialFailure != NUMI_TEMPORAL_CONE_ISLAND_SUCCESS) {
        if (lane == 0u) {
            NumiTemporalConeIslandStatus status = {};
            status.control = uint4(
                initialFailure,
                0u,
                0u,
                contactCount
            );
            outputStatuses[problem] = status;
        }
        return;
    }

    float laneAbsoluteRowBound = 0.0f;
    if (active) {
        for (uint outputAxis = 0u; outputAxis < 3u; ++outputAxis) {
            float rowSum = 0.0f;
            for (uint relativeBlock = rowBegin;
                 relativeBlock < rowEnd;
                 ++relativeBlock) {
                const uint block = blockBase + relativeBlock;
                const uint valueBase =
                    block * NUMI_TEMPORAL_CONE_STREAM_BLOCK_ELEMENTS;
                for (uint sourceAxis = 0u;
                     sourceAxis < 3u;
                     ++sourceAxis) {
                    rowSum += abs(blockValues[
                        valueBase + 3u * outputAxis + sourceAxis
                    ]);
                }
            }
            laneAbsoluteRowBound = max(
                laneAbsoluteRowBound,
                rowSum
            );
        }
    }
    float stepScale = 0.0f;
    if (active) {
        if (!isfinite(laneAbsoluteRowBound) ||
            !(laneAbsoluteRowBound > 0.0f)) {
            localFailure =
                NUMI_TEMPORAL_CONE_ISLAND_NONFINITE_RESULT;
        } else {
            stepScale = 1.0f / max(
                laneAbsoluteRowBound,
                kMatrixFloor
            );
            if (!isfinite(stepScale) || !(stepScale > 0.0f)) {
                localFailure =
                    NUMI_TEMPORAL_CONE_ISLAND_NONFINITE_RESULT;
            }
        }
    }
    const float effectiveRelaxation = header.tolerances.z;

    threadgroup float4 current[
        NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS
    ];
    threadgroup float4 next[
        NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS
    ];
    threadgroup float4 previous[
        NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS
    ];
    threadgroup float4 search[
        NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS
    ];
    if (active) {
        const float3 projectedWarmStart = projectFrictionCone(
            contact.warmImpulseAndFrictionV.xyz,
            cone
        );
        if (!finite3(projectedWarmStart)) {
            localFailure = NUMI_TEMPORAL_CONE_ISLAND_NONFINITE_RESULT;
            current[lane] = float4(0.0f);
        } else {
            current[lane] = float4(projectedWarmStart, 0.0f);
        }
    } else {
        current[lane] = float4(0.0f);
    }
    checkpoint[lane] = current[lane];
    previous[lane] = current[lane];
    search[lane] = current[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint checkpointFailure = simd_max(localFailure);
    if (checkpointFailure != NUMI_TEMPORAL_CONE_ISLAND_SUCCESS) {
        if (active) {
            outputImpulses[contactBase + lane] = float4(0.0f);
        }
        if (lane == 0u) {
            NumiTemporalConeIslandStatus status = {};
            status.control = uint4(
                checkpointFailure,
                0u,
                0u,
                contactCount
            );
            outputStatuses[problem] = status;
        }
        return;
    }

    uint completedIterations = 0u;
    const bool accelerationEnabled = effectiveRelaxation == 1.0f;
    float accelerationClock = 1.0f;
    float momentumScale = 0.0f;
    uint accelerationRestarts = 0u;
    for (uint iteration = 0u;
         iteration < header.control.w;
         ++iteration) {
        float laneNaturalDelta = 0.0f;
        float laneScale = 1.0f;
        float laneRestartMeasure = 0.0f;
        localFailure = NUMI_TEMPORAL_CONE_ISLAND_SUCCESS;
        const bool extrapolated = momentumScale != 0.0f;
        if (active && extrapolated) {
            search[lane] = current[lane] + momentumScale * (
                current[lane] - previous[lane]
            );
        }
        if (extrapolated) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (active) {
            const float3 impulse = extrapolated
                ? search[lane].xyz
                : current[lane].xyz;
            const float3 residual = extrapolated
                ? temporalConeStreamResidual(
                    columnIndices,
                    blockValues,
                    blockBase,
                    rowBegin,
                    rowEnd,
                    contact.freeVelocityAndFrictionU.xyz,
                    search
                )
                : temporalConeStreamResidual(
                    columnIndices,
                    blockValues,
                    blockBase,
                    rowBegin,
                    rowEnd,
                    contact.freeVelocityAndFrictionU.xyz,
                    current
                );
            const float3 proposed = impulse - stepScale * residual;
            const float3 projected = projectFrictionCone(proposed, cone);
            const float3 naturalDelta = projected - impulse;
            const float3 candidate =
                impulse + effectiveRelaxation * naturalDelta;
            if (!finite3(residual) ||
                !finite3(projected) ||
                !finite3(candidate)) {
                localFailure =
                    NUMI_TEMPORAL_CONE_ISLAND_NONFINITE_RESULT;
                next[lane] = current[lane];
            } else {
                next[lane] = float4(candidate, 0.0f);
                if (extrapolated) {
                    laneRestartMeasure = dot(
                        impulse - candidate,
                        candidate - current[lane].xyz
                    );
                }
                laneNaturalDelta = max(
                    abs(naturalDelta.x),
                    max(abs(naturalDelta.y), abs(naturalDelta.z))
                ) / stepScale;
                const float3 response =
                    residual - contact.freeVelocityAndFrictionU.xyz;
                laneScale = max(
                    1.0f,
                    max(
                        max(
                            abs(contact.freeVelocityAndFrictionU.x),
                            max(
                                abs(contact.freeVelocityAndFrictionU.y),
                                abs(contact.freeVelocityAndFrictionU.z)
                            )
                        ),
                        max(
                            abs(response.x),
                            max(abs(response.y), abs(response.z))
                        )
                    )
                );
                if (!isfinite(laneNaturalDelta) ||
                    !isfinite(laneScale) ||
                    !isfinite(laneRestartMeasure)) {
                    localFailure =
                        NUMI_TEMPORAL_CONE_ISLAND_NONFINITE_RESULT;
                    next[lane] = current[lane];
                    laneNaturalDelta = 0.0f;
                    laneScale = 1.0f;
                    laneRestartMeasure = 0.0f;
                }
            }
        } else {
            next[lane] = float4(0.0f);
        }
        const uint iterationFailure = simd_max(localFailure);
        const float maximumNaturalDelta = simd_max(laneNaturalDelta);
        const float maximumScale = simd_max(laneScale);
        const float restartMeasure = simd_sum(laneRestartMeasure);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        completedIterations = iteration + 1u;
        if (iterationFailure != NUMI_TEMPORAL_CONE_ISLAND_SUCCESS) {
            localFailure = iterationFailure;
            break;
        }
        const float tolerance =
            header.tolerances.x +
            header.tolerances.y * maximumScale;
        const bool candidateReady =
            completedIterations >= header.control.z &&
            maximumNaturalDelta <= tolerance;
        float laneComplementarityDelta = 0.0f;
        if (active && candidateReady) {
            const float3 impulse = extrapolated
                ? search[lane].xyz
                : current[lane].xyz;
            const float3 residual = extrapolated
                ? temporalConeStreamResidual(
                    columnIndices,
                    blockValues,
                    blockBase,
                    rowBegin,
                    rowEnd,
                    contact.freeVelocityAndFrictionU.xyz,
                    search
                )
                : temporalConeStreamResidual(
                    columnIndices,
                    blockValues,
                    blockBase,
                    rowBegin,
                    rowEnd,
                    contact.freeVelocityAndFrictionU.xyz,
                    current
                );
            laneComplementarityDelta =
                temporalConeComplementarityResidual(
                    impulse,
                    residual,
                    cone.effectiveFrictionU,
                    cone.effectiveFrictionV,
                    cone.maximumNormalImpulse
                );
            if (!isfinite(laneComplementarityDelta)) {
                laneComplementarityDelta = INFINITY;
            }
        }
        const float maximumComplementarityDelta = simd_max(
            laneComplementarityDelta
        );
        const bool certifiedCandidateReady =
            candidateReady &&
            maximumComplementarityDelta <= tolerance;
        const bool iterationConverged =
            momentumScale == 0.0f &&
            certifiedCandidateReady;
        if (iterationConverged) {
            break;
        }
        previous[lane] = current[lane];
        current[lane] = next[lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (accelerationEnabled) {
            // Match the dense qualification path exactly.
            const bool restartAcceleration =
                certifiedCandidateReady ||
                restartMeasure > 0.0f ||
                completedIterations < 16u ||
                completedIterations % 64u == 0u;
            if (restartAcceleration) {
                if (extrapolated) {
                    ++accelerationRestarts;
                }
                accelerationClock = 1.0f;
                momentumScale = 0.0f;
            } else {
                const float nextClock = 0.5f * (
                    1.0f + sqrt(fma(
                        4.0f * accelerationClock,
                        accelerationClock,
                        1.0f
                    ))
                );
                momentumScale =
                    (accelerationClock - 1.0f) / nextClock;
                accelerationClock = nextClock;
            }
        }
    }

    float laneNaturalResidual = 0.0f;
    float laneConeViolation = 0.0f;
    float laneKKTScale = 1.0f;
    float laneImpulseScale = 0.0f;
    float laneRawResidual = 0.0f;
    float laneObjective = 0.0f;
    float laneComplementarityResidual = 0.0f;
    if (active &&
        localFailure == NUMI_TEMPORAL_CONE_ISLAND_SUCCESS) {
        const float3 impulse = current[lane].xyz;
        const float3 residual = temporalConeStreamResidual(
            columnIndices,
            blockValues,
            blockBase,
            rowBegin,
            rowEnd,
            contact.freeVelocityAndFrictionU.xyz,
            current
        );
        const float3 proposed = impulse - stepScale * residual;
        const float3 projected = projectFrictionCone(proposed, cone);
        const float3 naturalDelta = projected - impulse;
        laneNaturalResidual = max(
            abs(naturalDelta.x),
            max(abs(naturalDelta.y), abs(naturalDelta.z))
        ) / stepScale;
        laneConeViolation = temporalConeViolation(
            impulse,
            cone.effectiveFrictionU,
            cone.effectiveFrictionV,
            cone.maximumNormalImpulse
        );
        laneImpulseScale = max(
            abs(impulse.x),
            max(abs(impulse.y), abs(impulse.z))
        );
        laneRawResidual = max(
            abs(residual.x),
            max(abs(residual.y), abs(residual.z))
        );
        const float3 response =
            residual - contact.freeVelocityAndFrictionU.xyz;
        laneKKTScale = max(
            1.0f,
            max(
                max(
                    abs(contact.freeVelocityAndFrictionU.x),
                    max(
                        abs(contact.freeVelocityAndFrictionU.y),
                        abs(contact.freeVelocityAndFrictionU.z)
                    )
                ),
                max(
                    abs(response.x),
                    max(abs(response.y), abs(response.z))
                )
            )
        );
        laneObjective =
            0.5f * dot(impulse, response) +
            dot(impulse, contact.freeVelocityAndFrictionU.xyz);
        laneComplementarityResidual =
            temporalConeComplementarityResidual(
                impulse,
                residual,
                cone.effectiveFrictionU,
                cone.effectiveFrictionV,
                cone.maximumNormalImpulse
            );
        if (!finite3(residual) ||
            !finite3(projected) ||
            !finite3(naturalDelta) ||
            !isfinite(laneNaturalResidual) ||
            !isfinite(laneConeViolation) ||
            !isfinite(laneImpulseScale) ||
            !isfinite(laneRawResidual) ||
            !isfinite(laneKKTScale) ||
            !isfinite(laneObjective) ||
            !isfinite(laneComplementarityResidual)) {
            localFailure =
                NUMI_TEMPORAL_CONE_ISLAND_NONFINITE_RESULT;
            laneNaturalResidual = 0.0f;
            laneConeViolation = 0.0f;
            laneImpulseScale = 0.0f;
            laneRawResidual = 0.0f;
            laneKKTScale = 1.0f;
            laneObjective = 0.0f;
            laneComplementarityResidual = 0.0f;
        }
    }

    const uint diagnosticFailure = simd_max(localFailure);
    const float maximumNaturalResidual = simd_max(laneNaturalResidual);
    const float maximumConeViolation = simd_max(laneConeViolation);
    const float maximumKKTScale = simd_max(laneKKTScale);
    const float maximumImpulseScale = simd_max(laneImpulseScale);
    const float maximumRawResidual = simd_max(laneRawResidual);
    const float objective = simd_sum(laneObjective);
    const float maximumComplementarityResidual = simd_max(
        laneComplementarityResidual
    );
    const float kktTolerance =
        header.tolerances.x +
        header.tolerances.y * maximumKKTScale;
    const float coneTolerance =
        header.tolerances.x +
        header.tolerances.y * max(1.0f, maximumImpulseScale);
    const float objectiveTolerance =
        3.0f * float(contactCount) *
        kktTolerance * maximumImpulseScale;
    const bool finiteCertificate =
        isfinite(maximumNaturalResidual) &&
        isfinite(maximumConeViolation) &&
        isfinite(maximumKKTScale) &&
        maximumKKTScale > 0.0f &&
        isfinite(maximumImpulseScale) &&
        isfinite(maximumRawResidual) &&
        isfinite(objective) &&
        isfinite(maximumComplementarityResidual) &&
        isfinite(kktTolerance) &&
        isfinite(coneTolerance) &&
        isfinite(objectiveTolerance);
    const uint finalFailure =
        diagnosticFailure != NUMI_TEMPORAL_CONE_ISLAND_SUCCESS
        ? diagnosticFailure
        : finiteCertificate
        ? NUMI_TEMPORAL_CONE_ISLAND_SUCCESS
        : NUMI_TEMPORAL_CONE_ISLAND_NONFINITE_RESULT;
    const float normalizedNaturalResidual = finiteCertificate
        ? maximumNaturalResidual / maximumKKTScale
        : 0.0f;
    const float normalizedComplementarityResidual = finiteCertificate
        ? maximumComplementarityResidual / maximumKKTScale
        : 0.0f;
    const bool finalConverged =
        finalFailure == NUMI_TEMPORAL_CONE_ISLAND_SUCCESS &&
        maximumNaturalResidual <= kktTolerance &&
        maximumComplementarityResidual <= kktTolerance &&
        maximumConeViolation <= coneTolerance &&
        objective <= objectiveTolerance;
    if (active) {
        outputImpulses[contactBase + lane] = finalConverged
            ? current[lane]
            : checkpoint[lane];
    }
    if (lane == 0u) {
        NumiTemporalConeIslandStatus status = {};
        const uint code =
            finalFailure != NUMI_TEMPORAL_CONE_ISLAND_SUCCESS
            ? finalFailure
            : finalConverged
            ? NUMI_TEMPORAL_CONE_ISLAND_SUCCESS
            : NUMI_TEMPORAL_CONE_ISLAND_DID_NOT_CONVERGE;
        status.control = uint4(
            code,
            completedIterations,
            finalConverged ? 1u : 0u,
            contactCount
        );
        status.residuals = finiteCertificate
            ? float4(
                normalizedNaturalResidual,
                maximumConeViolation,
                maximumImpulseScale,
                objective
            )
            : float4(0.0f);
        status.diagnostics = finiteCertificate
            ? float4(
                maximumRawResidual,
                effectiveRelaxation,
                normalizedComplementarityResidual,
                float(accelerationRestarts)
            )
            : float4(0.0f);
        outputStatuses[problem] = status;
    }
}

inline bool temporalConeRegularizationPSD(
    device const float* values,
    const uint base
) {
    float scale = 0.0f;
    for (uint element = 0u; element < 9u; ++element) {
        const float value = values[base + element];
        if (!isfinite(value)) {
            return false;
        }
        scale = max(scale, abs(value));
    }
    if (scale == 0.0f) {
        return true;
    }
    const float inverseScale = 1.0f / scale;
    if (!(inverseScale > 0.0f) || !isfinite(inverseScale)) {
        return false;
    }

    float matrix[3][3];
    const float symmetryTolerance =
        64.0f * kFloatEpsilon * max(1.0f, scale);
    for (uint row = 0u; row < 3u; ++row) {
        for (uint column = 0u; column < 3u; ++column) {
            const float value = values[base + 3u * row + column];
            const float transpose = values[
                base + 3u * column + row
            ];
            if (abs(value - transpose) > symmetryTolerance) {
                return false;
            }
            matrix[row][column] =
                0.5f * value * inverseScale +
                0.5f * transpose * inverseScale;
        }
    }

    const float minor01 =
        matrix[0][0] * matrix[1][1] -
        matrix[0][1] * matrix[1][0];
    const float minor02 =
        matrix[0][0] * matrix[2][2] -
        matrix[0][2] * matrix[2][0];
    const float minor12 =
        matrix[1][1] * matrix[2][2] -
        matrix[1][2] * matrix[2][1];
    const float determinant =
        matrix[0][0] * (
            matrix[1][1] * matrix[2][2] -
            matrix[1][2] * matrix[2][1]
        ) -
        matrix[0][1] * (
            matrix[1][0] * matrix[2][2] -
            matrix[1][2] * matrix[2][0]
        ) +
        matrix[0][2] * (
            matrix[1][0] * matrix[2][1] -
            matrix[1][1] * matrix[2][0]
        );
    const float tolerance = 64.0f * kFloatEpsilon;
    return
        matrix[0][0] >= -tolerance &&
        matrix[1][1] >= -tolerance &&
        matrix[2][2] >= -tolerance &&
        minor01 >= -tolerance &&
        minor02 >= -tolerance &&
        minor12 >= -tolerance &&
        determinant >= -tolerance &&
        isfinite(minor01) &&
        isfinite(minor02) &&
        isfinite(minor12) &&
        isfinite(determinant);
}

inline uint temporalConePackedLowerIndex(
    const uint row,
    const uint column
) {
    return (row * (row + 1u)) / 2u + column;
}

// Deterministically assembles a complete block-CSR Delassus operator from
// contact Jacobians J_i and already-computed response columns M^-1 J_i^T.
// Topology is supplied separately and is accepted only when it contains every
// and only diagonal/shared-owner block. A successful output header is the
// transaction commit consumed by numi_temporal_cone_stream_solve.
kernel void numi_temporal_cone_stream_assemble(
    device const NumiTemporalConeAssemblyHeader* headers [[buffer(0)]],
    device const NumiTemporalConeAssemblyContactSpan* spans [[buffer(1)]],
    device const NumiTemporalConeAssemblyTerm* terms [[buffer(2)]],
    device const float* jacobianValues [[buffer(3)]],
    device const float* responseValues [[buffer(4)]],
    device const float* regularizationValues [[buffer(5)]],
    device const uint* rowOffsets [[buffer(6)]],
    device const uint* columnIndices [[buffer(7)]],
    device float* outputBlockValues [[buffer(8)]],
    device NumiTemporalConeStreamHeader* outputHeaders [[buffer(9)]],
    device NumiTemporalConeAssemblyStatus* outputStatuses [[buffer(10)]],
    constant uint& problemCount [[buffer(11)]],
    // x spans, y terms, z Jacobian floats, w response floats.
    constant uint4& inputCapacities [[buffer(12)]],
    // x row offsets, y column indices, z regularization floats,
    // w output block-value floats.
    constant uint4& outputCapacities [[buffer(13)]],
    const uint problem [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    // The production assembly path certifies the complete scalar operator,
    // not only its diagonal blocks and 2x2 principal minors. Packed storage
    // keeps the 96-row maximum below 19 KiB and is confined to assembly, so
    // the iterative solver's threadgroup footprint does not change.
    threadgroup float packedLower[
        NUMI_TEMPORAL_CONE_PACKED_LOWER_ELEMENTS
    ];

    if (problem >= problemCount) {
        return;
    }
    if (lane == 0u) {
        outputHeaders[problem] = {};
        outputStatuses[problem] = {};
    }

    const NumiTemporalConeAssemblyHeader header = headers[problem];
    const uint contactCount = header.control.y;
    const uint contactBase = header.outputRanges.x;
    const uint rowOffsetBase = header.outputRanges.y;
    const uint blockBase = header.outputRanges.z;
    const uint blockCount = header.outputRanges.w;
    const uint spanBase = header.inputRanges.x;
    const uint regularizationBase = header.inputRanges.y;
    const bool active = lane < contactCount;

    uint localFailure = NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS;
    if (header.control.x != NUMI_TEMPORAL_CONE_ASSEMBLY_ABI_VERSION) {
        localFailure = NUMI_TEMPORAL_CONE_ASSEMBLY_INVALID_ABI;
    } else if (
        contactCount == 0u ||
        contactCount > NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS ||
        blockCount < contactCount ||
        blockCount > NUMI_TEMPORAL_CONE_STREAM_MAX_BLOCKS ||
        header.control.z == 0u ||
        header.control.z > header.control.w ||
        header.control.w > NUMI_TEMPORAL_CONE_ISLAND_MAX_ITERATIONS ||
        !finite4(header.tolerances) ||
        !(header.tolerances.x > 0.0f) ||
        header.tolerances.y < 0.0f ||
        !(header.tolerances.z > 0.0f) ||
        header.tolerances.z > 1.0f
    ) {
        localFailure = NUMI_TEMPORAL_CONE_ASSEMBLY_INVALID_INPUT;
    } else if (
        spanBase > inputCapacities.x ||
        contactCount > inputCapacities.x - spanBase ||
        rowOffsetBase > outputCapacities.x ||
        contactCount >= outputCapacities.x - rowOffsetBase ||
        blockBase > outputCapacities.y ||
        blockCount > outputCapacities.y - blockBase ||
        regularizationBase > outputCapacities.z ||
        contactCount * NUMI_TEMPORAL_CONE_STREAM_BLOCK_ELEMENTS >
            outputCapacities.z - regularizationBase ||
        blockBase >
            outputCapacities.w /
                NUMI_TEMPORAL_CONE_STREAM_BLOCK_ELEMENTS ||
        blockCount >
            outputCapacities.w /
                NUMI_TEMPORAL_CONE_STREAM_BLOCK_ELEMENTS - blockBase
    ) {
        localFailure = NUMI_TEMPORAL_CONE_ASSEMBLY_CAPACITY_EXCEEDED;
    }
    const uint headerFailureKey = simd_min(
        localFailure == NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS
        ? 0xffffffffu
        : localFailure
    );
    const uint headerFailure = headerFailureKey == 0xffffffffu
        ? NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS
        : headerFailureKey;
    if (headerFailure != NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS) {
        if (lane == 0u) {
            NumiTemporalConeAssemblyStatus status = {};
            status.control = uint4(
                headerFailure,
                contactCount,
                blockCount,
                0u
            );
            outputStatuses[problem] = status;
        }
        return;
    }

    uint rowBegin = 0u;
    uint rowEnd = 0u;
    NumiTemporalConeAssemblyContactSpan span = {};
    uint maximumTerms = 0u;
    if (active) {
        rowBegin = rowOffsets[rowOffsetBase + lane];
        rowEnd = rowOffsets[rowOffsetBase + lane + 1u];
        span = spans[spanBase + lane];
        maximumTerms = span.ranges.y;
        if (rowBegin >= rowEnd ||
            rowBegin > blockCount ||
            rowEnd > blockCount ||
            span.ranges.y == 0u ||
            span.ranges.y >
                NUMI_TEMPORAL_CONE_ASSEMBLY_MAX_TERMS_PER_CONTACT ||
            span.ranges.x > inputCapacities.y ||
            span.ranges.y > inputCapacities.y - span.ranges.x) {
            localFailure = NUMI_TEMPORAL_CONE_ASSEMBLY_INVALID_INPUT;
        }
        uint previousSource = 0u;
        bool firstSource = true;
        bool diagonalSeen = false;
        for (uint relativeBlock = rowBegin;
             relativeBlock < rowEnd &&
                 localFailure == NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS;
             ++relativeBlock) {
            const uint source = columnIndices[blockBase + relativeBlock];
            if (source >= contactCount ||
                (!firstSource && source <= previousSource)) {
                localFailure = NUMI_TEMPORAL_CONE_ASSEMBLY_INVALID_INPUT;
                break;
            }
            firstSource = false;
            previousSource = source;
            diagonalSeen = diagonalSeen || source == lane;
        }
        if (!diagonalSeen) {
            localFailure = NUMI_TEMPORAL_CONE_ASSEMBLY_INVALID_INPUT;
        }
    }
    const uint groupMaximumTerms = simd_max(maximumTerms);
    uint structuralFailure = simd_max(localFailure);
    if (structuralFailure != NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS) {
        if (lane == 0u) {
            NumiTemporalConeAssemblyStatus status = {};
            status.control = uint4(
                structuralFailure,
                contactCount,
                blockCount,
                groupMaximumTerms
            );
            outputStatuses[problem] = status;
        }
        return;
    }

    if (active) {
        uint previousOwner = 0u;
        bool firstOwner = true;
        for (uint localTerm = 0u;
             localTerm < span.ranges.y;
             ++localTerm) {
            const NumiTemporalConeAssemblyTerm term = terms[
                span.ranges.x + localTerm
            ];
            const uint owner = term.control.x;
            const uint dofCount = term.control.y;
            const uint valueCount = 3u * dofCount;
            if (owner == 0xffffffffu ||
                (!firstOwner && owner <= previousOwner) ||
                dofCount == 0u ||
                dofCount >
                    NUMI_TEMPORAL_CONE_ASSEMBLY_MAX_DOF_PER_TERM ||
                term.control.z > inputCapacities.z ||
                valueCount > inputCapacities.z - term.control.z ||
                term.control.w > inputCapacities.w ||
                valueCount > inputCapacities.w - term.control.w) {
                localFailure = NUMI_TEMPORAL_CONE_ASSEMBLY_INVALID_INPUT;
                break;
            }
            firstOwner = false;
            previousOwner = owner;
            for (uint value = 0u; value < valueCount; ++value) {
                if (!isfinite(jacobianValues[term.control.z + value]) ||
                    !isfinite(responseValues[term.control.w + value])) {
                    localFailure =
                        NUMI_TEMPORAL_CONE_ASSEMBLY_INVALID_INPUT;
                    break;
                }
            }
        }
        const uint regularizationContactBase =
            regularizationBase +
            lane * NUMI_TEMPORAL_CONE_STREAM_BLOCK_ELEMENTS;
        for (uint value = 0u;
             value < NUMI_TEMPORAL_CONE_STREAM_BLOCK_ELEMENTS;
             ++value) {
            if (!isfinite(
                    regularizationValues[regularizationContactBase + value]
                )) {
                localFailure = NUMI_TEMPORAL_CONE_ASSEMBLY_INVALID_INPUT;
            }
        }
        if (localFailure == NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS &&
            !temporalConeRegularizationPSD(
                regularizationValues,
                regularizationContactBase
            )) {
            localFailure =
                NUMI_TEMPORAL_CONE_ASSEMBLY_NON_PSD_REGULARIZATION;
        }
    }
    structuralFailure = simd_max(localFailure);
    if (structuralFailure != NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS) {
        if (lane == 0u) {
            NumiTemporalConeAssemblyStatus status = {};
            status.control = uint4(
                structuralFailure,
                contactCount,
                blockCount,
                groupMaximumTerms
            );
            outputStatuses[problem] = status;
        }
        return;
    }

    float laneMaximumCoefficient = 0.0f;
    float laneMinimumDiagonal = INFINITY;
    uint laneTopologyErrors = 0u;
    if (active) {
        uint topologyCursor = rowBegin;
        for (uint source = 0u; source < contactCount; ++source) {
            const NumiTemporalConeAssemblyContactSpan sourceSpan = spans[
                spanBase + source
            ];
            uint foundRelativeBlock = 0xffffffffu;
            if (topologyCursor < rowEnd &&
                columnIndices[blockBase + topologyCursor] == source) {
                foundRelativeBlock = topologyCursor;
                ++topologyCursor;
            }
            const bool foundBlock = foundRelativeBlock != 0xffffffffu;
            float coefficients[
                NUMI_TEMPORAL_CONE_STREAM_BLOCK_ELEMENTS
            ];
            for (uint element = 0u;
                 element < NUMI_TEMPORAL_CONE_STREAM_BLOCK_ELEMENTS;
                 ++element) {
                coefficients[element] = source == lane && foundBlock
                    ? regularizationValues[
                        regularizationBase +
                        lane * NUMI_TEMPORAL_CONE_STREAM_BLOCK_ELEMENTS +
                        element
                    ]
                    : 0.0f;
            }
            uint targetTermIndex = 0u;
            uint sourceTermIndex = 0u;
            bool sharesOwner = false;
            while (targetTermIndex < span.ranges.y &&
                   sourceTermIndex < sourceSpan.ranges.y) {
                const NumiTemporalConeAssemblyTerm targetTerm = terms[
                    span.ranges.x + targetTermIndex
                ];
                const NumiTemporalConeAssemblyTerm sourceTerm = terms[
                    sourceSpan.ranges.x + sourceTermIndex
                ];
                if (targetTerm.control.x < sourceTerm.control.x) {
                    ++targetTermIndex;
                    continue;
                }
                if (sourceTerm.control.x < targetTerm.control.x) {
                    ++sourceTermIndex;
                    continue;
                }
                sharesOwner = true;
                if (targetTerm.control.y != sourceTerm.control.y) {
                    localFailure =
                        NUMI_TEMPORAL_CONE_ASSEMBLY_INVALID_INPUT;
                    ++targetTermIndex;
                    ++sourceTermIndex;
                    continue;
                }
                if (foundBlock) {
                    const uint dofCount = targetTerm.control.y;
                    for (uint dof = 0u; dof < dofCount; ++dof) {
                        const float response0 = responseValues[
                            sourceTerm.control.w + 3u * dof + 0u
                        ];
                        const float response1 = responseValues[
                            sourceTerm.control.w + 3u * dof + 1u
                        ];
                        const float response2 = responseValues[
                            sourceTerm.control.w + 3u * dof + 2u
                        ];
                        for (uint targetAxis = 0u;
                             targetAxis < 3u;
                             ++targetAxis) {
                            const float jacobian = jacobianValues[
                                targetTerm.control.z +
                                targetAxis * dofCount + dof
                            ];
                            coefficients[3u * targetAxis + 0u] = fma(
                                jacobian,
                                response0,
                                coefficients[3u * targetAxis + 0u]
                            );
                            coefficients[3u * targetAxis + 1u] = fma(
                                jacobian,
                                response1,
                                coefficients[3u * targetAxis + 1u]
                            );
                            coefficients[3u * targetAxis + 2u] = fma(
                                jacobian,
                                response2,
                                coefficients[3u * targetAxis + 2u]
                            );
                        }
                    }
                }
                ++targetTermIndex;
                ++sourceTermIndex;
            }
            const bool expectedBlock = source == lane || sharesOwner;
            if (expectedBlock != foundBlock) {
                ++laneTopologyErrors;
                localFailure =
                    NUMI_TEMPORAL_CONE_ASSEMBLY_MISSING_COUPLING;
                continue;
            }
            if (!foundBlock) {
                continue;
            }

            const uint outputValueBase =
                (blockBase + foundRelativeBlock) *
                NUMI_TEMPORAL_CONE_STREAM_BLOCK_ELEMENTS;
            for (uint element = 0u;
                 element < NUMI_TEMPORAL_CONE_STREAM_BLOCK_ELEMENTS;
                 ++element) {
                float coefficient = coefficients[element];
                if (!isfinite(coefficient)) {
                    localFailure =
                        NUMI_TEMPORAL_CONE_ASSEMBLY_NONFINITE_RESULT;
                    coefficient = 0.0f;
                }
                outputBlockValues[outputValueBase + element] = coefficient;
                laneMaximumCoefficient = max(
                    laneMaximumCoefficient,
                    abs(coefficient)
                );
                if (source == lane && element % 4u == 0u) {
                    laneMinimumDiagonal = min(
                        laneMinimumDiagonal,
                        coefficient
                    );
                }
            }
        }
    }
    const uint assemblyFailureKey = simd_min(
        localFailure == NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS
        ? 0xffffffffu
        : localFailure
    );
    const uint assemblyFailure = assemblyFailureKey == 0xffffffffu
        ? NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS
        : assemblyFailureKey;
    const uint topologyErrors = simd_sum(laneTopologyErrors);
    threadgroup_barrier(mem_flags::mem_device);
    if (assemblyFailure != NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS) {
        if (lane == 0u) {
            NumiTemporalConeAssemblyStatus status = {};
            status.control = uint4(
                assemblyFailure,
                contactCount,
                blockCount,
                groupMaximumTerms
            );
            status.diagnostics.w = float(topologyErrors);
            outputStatuses[problem] = status;
        }
        return;
    }

    float laneMaximumSymmetryError = 0.0f;
    if (active) {
        for (uint relativeBlock = rowBegin;
             relativeBlock < rowEnd;
             ++relativeBlock) {
            const uint block = blockBase + relativeBlock;
            const uint source = columnIndices[block];
            const uint reverseBegin = rowOffsets[rowOffsetBase + source];
            const uint reverseEnd = rowOffsets[
                rowOffsetBase + source + 1u
            ];
            uint reverseBlock = 0xffffffffu;
            uint lower = reverseBegin;
            uint upper = reverseEnd;
            while (lower < upper) {
                const uint middle = lower + (upper - lower) / 2u;
                const uint candidateSource = columnIndices[
                    blockBase + middle
                ];
                if (candidateSource < lane) {
                    lower = middle + 1u;
                } else {
                    upper = middle;
                }
            }
            if (lower < reverseEnd &&
                columnIndices[blockBase + lower] == lane) {
                reverseBlock = blockBase + lower;
            }
            if (reverseBlock == 0xffffffffu) {
                localFailure =
                    NUMI_TEMPORAL_CONE_ASSEMBLY_MISSING_COUPLING;
                continue;
            }
            const uint valueBase =
                block * NUMI_TEMPORAL_CONE_STREAM_BLOCK_ELEMENTS;
            const uint reverseValueBase =
                reverseBlock * NUMI_TEMPORAL_CONE_STREAM_BLOCK_ELEMENTS;
            for (uint row = 0u; row < 3u; ++row) {
                for (uint column = 0u; column < 3u; ++column) {
                    const float value = outputBlockValues[
                        valueBase + 3u * row + column
                    ];
                    const float reverseValue = outputBlockValues[
                        reverseValueBase + 3u * column + row
                    ];
                    const float error = abs(value - reverseValue);
                    laneMaximumSymmetryError = max(
                        laneMaximumSymmetryError,
                        error
                    );
                    const float scale = max(
                        1.0f,
                        max(abs(value), abs(reverseValue))
                    );
                    if (error > 64.0f * kFloatEpsilon * scale) {
                        localFailure =
                            NUMI_TEMPORAL_CONE_ASSEMBLY_ASYMMETRIC_RESPONSE;
                    }
                }
            }
        }
    }
    const uint symmetryFailureKey = simd_min(
        localFailure == NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS
        ? 0xffffffffu
        : localFailure
    );
    const uint symmetryFailure = symmetryFailureKey == 0xffffffffu
        ? NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS
        : symmetryFailureKey;
    const float maximumSymmetryError = simd_max(
        laneMaximumSymmetryError
    );
    const float maximumCoefficient = simd_max(laneMaximumCoefficient);
    const float minimumDiagonal = simd_min(laneMinimumDiagonal);
    uint certificateFailure = symmetryFailure;
    if (certificateFailure == NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS) {
        const uint dimension = 3u * contactCount;
        const uint packedCount = dimension * (dimension + 1u) / 2u;
        const float inverseScale = maximumCoefficient > 0.0f
            ? 1.0f / maximumCoefficient
            : 1.0f;
        for (uint index = lane; index < packedCount; index += 32u) {
            packedLower[index] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            for (uint relativeBlock = rowBegin;
                 relativeBlock < rowEnd;
                 ++relativeBlock) {
                const uint source = columnIndices[
                    blockBase + relativeBlock
                ];
                if (source > lane) {
                    continue;
                }
                const uint valueBase =
                    (blockBase + relativeBlock) *
                    NUMI_TEMPORAL_CONE_STREAM_BLOCK_ELEMENTS;
                for (uint targetAxis = 0u;
                     targetAxis < 3u;
                     ++targetAxis) {
                    const uint scalarRow = 3u * lane + targetAxis;
                    for (uint sourceAxis = 0u;
                         sourceAxis < 3u;
                         ++sourceAxis) {
                        const uint scalarColumn =
                            3u * source + sourceAxis;
                        if (scalarColumn <= scalarRow) {
                            packedLower[temporalConePackedLowerIndex(
                                scalarRow,
                                scalarColumn
                            )] = outputBlockValues[
                                valueBase + 3u * targetAxis + sourceAxis
                            ] * inverseScale;
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Block the fixed-order scalar factorization by contact. Lane zero
        // factors one 3x3 diagonal block while every other lane solves one
        // trailing 3x3 block row. This is algebraically the same unpivoted
        // semidefinite Cholesky order, but needs one barrier per contact rather
        // than one per scalar row. A numerically zero pivot is valid only when
        // the corresponding Schur-complement column is also zero.
        const float factorTolerance =
            64.0f * kFloatEpsilon * float(dimension);
        for (uint pivotContact = 0u;
             pivotContact < contactCount;
             ++pivotContact) {
            const uint pivotBase = 3u * pivotContact;
            float laneL00 = 0.0f;
            float laneL10 = 0.0f;
            float laneL20 = 0.0f;
            float laneL11 = 0.0f;
            float laneL21 = 0.0f;
            float laneL22 = 0.0f;
            uint lanePivotFailure =
                NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS;
            if (lane == 0u) {
                float reduced00 = packedLower[
                    temporalConePackedLowerIndex(
                        pivotBase,
                        pivotBase
                    )
                ];
                float reduced10 = packedLower[
                    temporalConePackedLowerIndex(
                        pivotBase + 1u,
                        pivotBase
                    )
                ];
                float reduced20 = packedLower[
                    temporalConePackedLowerIndex(
                        pivotBase + 2u,
                        pivotBase
                    )
                ];
                float reduced11 = packedLower[
                    temporalConePackedLowerIndex(
                        pivotBase + 1u,
                        pivotBase + 1u
                    )
                ];
                float reduced21 = packedLower[
                    temporalConePackedLowerIndex(
                        pivotBase + 2u,
                        pivotBase + 1u
                    )
                ];
                float reduced22 = packedLower[
                    temporalConePackedLowerIndex(
                        pivotBase + 2u,
                        pivotBase + 2u
                    )
                ];
                for (uint inner = 0u; inner < pivotBase; ++inner) {
                    const float value0 = packedLower[
                        temporalConePackedLowerIndex(pivotBase, inner)
                    ];
                    const float value1 = packedLower[
                        temporalConePackedLowerIndex(
                            pivotBase + 1u,
                            inner
                        )
                    ];
                    const float value2 = packedLower[
                        temporalConePackedLowerIndex(
                            pivotBase + 2u,
                            inner
                        )
                    ];
                    reduced00 = fma(-value0, value0, reduced00);
                    reduced10 = fma(-value1, value0, reduced10);
                    reduced20 = fma(-value2, value0, reduced20);
                    reduced11 = fma(-value1, value1, reduced11);
                    reduced21 = fma(-value2, value1, reduced21);
                    reduced22 = fma(-value2, value2, reduced22);
                }
                if (!isfinite(reduced00) ||
                    reduced00 < -factorTolerance) {
                    lanePivotFailure =
                        NUMI_TEMPORAL_CONE_ASSEMBLY_INDEFINITE_OPERATOR;
                } else {
                    laneL00 = reduced00 > factorTolerance
                        ? sqrt(reduced00)
                        : 0.0f;
                }
                if (lanePivotFailure ==
                    NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS) {
                    if (laneL00 > 0.0f) {
                        laneL10 = reduced10 / laneL00;
                        laneL20 = reduced20 / laneL00;
                    } else if (abs(reduced10) > factorTolerance ||
                               abs(reduced20) > factorTolerance) {
                        lanePivotFailure =
                            NUMI_TEMPORAL_CONE_ASSEMBLY_INDEFINITE_OPERATOR;
                    }
                    if (!isfinite(laneL10) || !isfinite(laneL20)) {
                        lanePivotFailure =
                            NUMI_TEMPORAL_CONE_ASSEMBLY_INDEFINITE_OPERATOR;
                    }
                }
                reduced11 = fma(-laneL10, laneL10, reduced11);
                if (lanePivotFailure ==
                        NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS &&
                    (!isfinite(reduced11) ||
                     reduced11 < -factorTolerance)) {
                    lanePivotFailure =
                        NUMI_TEMPORAL_CONE_ASSEMBLY_INDEFINITE_OPERATOR;
                } else {
                    laneL11 = reduced11 > factorTolerance
                        ? sqrt(reduced11)
                        : 0.0f;
                }
                reduced21 = fma(-laneL20, laneL10, reduced21);
                if (lanePivotFailure ==
                    NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS) {
                    if (laneL11 > 0.0f) {
                        laneL21 = reduced21 / laneL11;
                    } else if (abs(reduced21) > factorTolerance) {
                        lanePivotFailure =
                            NUMI_TEMPORAL_CONE_ASSEMBLY_INDEFINITE_OPERATOR;
                    }
                    if (!isfinite(laneL21)) {
                        lanePivotFailure =
                            NUMI_TEMPORAL_CONE_ASSEMBLY_INDEFINITE_OPERATOR;
                    }
                }
                reduced22 = fma(-laneL20, laneL20, reduced22);
                reduced22 = fma(-laneL21, laneL21, reduced22);
                if (lanePivotFailure ==
                        NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS &&
                    (!isfinite(reduced22) ||
                     reduced22 < -factorTolerance)) {
                    lanePivotFailure =
                        NUMI_TEMPORAL_CONE_ASSEMBLY_INDEFINITE_OPERATOR;
                } else {
                    laneL22 = reduced22 > factorTolerance
                        ? sqrt(reduced22)
                        : 0.0f;
                }
                if (lanePivotFailure ==
                    NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS) {
                    packedLower[temporalConePackedLowerIndex(
                        pivotBase,
                        pivotBase
                    )] = laneL00;
                    packedLower[temporalConePackedLowerIndex(
                        pivotBase + 1u,
                        pivotBase
                    )] = laneL10;
                    packedLower[temporalConePackedLowerIndex(
                        pivotBase + 2u,
                        pivotBase
                    )] = laneL20;
                    packedLower[temporalConePackedLowerIndex(
                        pivotBase + 1u,
                        pivotBase + 1u
                    )] = laneL11;
                    packedLower[temporalConePackedLowerIndex(
                        pivotBase + 2u,
                        pivotBase + 1u
                    )] = laneL21;
                    packedLower[temporalConePackedLowerIndex(
                        pivotBase + 2u,
                        pivotBase + 2u
                    )] = laneL22;
                }
            }
            const uint pivotFailure = simd_max(lanePivotFailure);
            const float l00 = simd_broadcast(laneL00, 0u);
            const float l10 = simd_broadcast(laneL10, 0u);
            const float l20 = simd_broadcast(laneL20, 0u);
            const float l11 = simd_broadcast(laneL11, 0u);
            const float l21 = simd_broadcast(laneL21, 0u);
            const float l22 = simd_broadcast(laneL22, 0u);
            if (pivotFailure !=
                NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS) {
                certificateFailure = pivotFailure;
                break;
            }

            uint laneFactorFailure =
                NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS;
            const uint rowContact = pivotContact + 1u + lane;
            if (rowContact < contactCount) {
                for (uint targetAxis = 0u;
                     targetAxis < 3u;
                     ++targetAxis) {
                    const uint row = 3u * rowContact + targetAxis;
                    float reduced0 = packedLower[
                        temporalConePackedLowerIndex(row, pivotBase)
                    ];
                    float reduced1 = packedLower[
                        temporalConePackedLowerIndex(row, pivotBase + 1u)
                    ];
                    float reduced2 = packedLower[
                        temporalConePackedLowerIndex(row, pivotBase + 2u)
                    ];
                    for (uint inner = 0u; inner < pivotBase; ++inner) {
                        const float rowValue = packedLower[
                            temporalConePackedLowerIndex(row, inner)
                        ];
                        reduced0 = fma(
                            -rowValue,
                            packedLower[temporalConePackedLowerIndex(
                                pivotBase,
                                inner
                            )],
                            reduced0
                        );
                        reduced1 = fma(
                            -rowValue,
                            packedLower[temporalConePackedLowerIndex(
                                pivotBase + 1u,
                                inner
                            )],
                            reduced1
                        );
                        reduced2 = fma(
                            -rowValue,
                            packedLower[temporalConePackedLowerIndex(
                                pivotBase + 2u,
                                inner
                            )],
                            reduced2
                        );
                    }
                    float value0 = 0.0f;
                    float value1 = 0.0f;
                    float value2 = 0.0f;
                    if (l00 > 0.0f) {
                        value0 = reduced0 / l00;
                    } else if (abs(reduced0) > factorTolerance) {
                        laneFactorFailure =
                            NUMI_TEMPORAL_CONE_ASSEMBLY_INDEFINITE_OPERATOR;
                    }
                    reduced1 = fma(-value0, l10, reduced1);
                    if (l11 > 0.0f) {
                        value1 = reduced1 / l11;
                    } else if (abs(reduced1) > factorTolerance) {
                        laneFactorFailure =
                            NUMI_TEMPORAL_CONE_ASSEMBLY_INDEFINITE_OPERATOR;
                    }
                    reduced2 = fma(-value0, l20, reduced2);
                    reduced2 = fma(-value1, l21, reduced2);
                    if (l22 > 0.0f) {
                        value2 = reduced2 / l22;
                    } else if (abs(reduced2) > factorTolerance) {
                        laneFactorFailure =
                            NUMI_TEMPORAL_CONE_ASSEMBLY_INDEFINITE_OPERATOR;
                    }
                    if (!isfinite(value0) || !isfinite(value1) ||
                        !isfinite(value2) || !isfinite(reduced0) ||
                        !isfinite(reduced1) || !isfinite(reduced2)) {
                        laneFactorFailure =
                            NUMI_TEMPORAL_CONE_ASSEMBLY_INDEFINITE_OPERATOR;
                    }
                    packedLower[temporalConePackedLowerIndex(
                        row,
                        pivotBase
                    )] = value0;
                    packedLower[temporalConePackedLowerIndex(
                        row,
                        pivotBase + 1u
                    )] = value1;
                    packedLower[temporalConePackedLowerIndex(
                        row,
                        pivotBase + 2u
                    )] = value2;
                }
            }
            const uint columnFailure = simd_max(laneFactorFailure);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (columnFailure !=
                NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS) {
                certificateFailure = columnFailure;
                break;
            }
        }
    }
    if (lane == 0u) {
        NumiTemporalConeAssemblyStatus status = {};
        status.control = uint4(
            certificateFailure,
            contactCount,
            blockCount,
            groupMaximumTerms
        );
        status.diagnostics = float4(
            maximumSymmetryError,
            maximumCoefficient,
            minimumDiagonal,
            float(topologyErrors)
        );
        outputStatuses[problem] = status;
        if (certificateFailure == NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS) {
            NumiTemporalConeStreamHeader output = {};
            output.control = uint4(
                NUMI_TEMPORAL_CONE_STREAM_ABI_VERSION,
                contactCount,
                header.control.z,
                header.control.w
            );
            output.ranges = uint4(
                contactBase,
                rowOffsetBase,
                blockBase,
                blockCount
            );
            output.tolerances = header.tolerances;
            outputHeaders[problem] = output;
        }
    }
}

inline bool temporalConeRigidInertiaValid(
    thread const NumiTemporalConeRigidBody& body,
    thread float& minorProxy
) {
    const float3 row0 = body.inverseInertiaRow0.xyz;
    const float3 row1 = body.inverseInertiaRow1.xyz;
    const float3 row2 = body.inverseInertiaRow2.xyz;
    const float scale = max(
        1.0f,
        max(
            max(max(abs(row0.x), abs(row0.y)), abs(row0.z)),
            max(
                max(max(abs(row1.x), abs(row1.y)), abs(row1.z)),
                max(max(abs(row2.x), abs(row2.y)), abs(row2.z))
            )
        )
    );
    const float symmetryTolerance = 64.0f * kFloatEpsilon * scale;
    const float minor1 = row0.x;
    const float minor2 = row0.x * row1.y - row0.y * row1.x;
    const float determinant =
        row0.x * (row1.y * row2.z - row1.z * row2.y) -
        row0.y * (row1.x * row2.z - row1.z * row2.x) +
        row0.z * (row1.x * row2.y - row1.y * row2.x);
    minorProxy = min(minor1, min(minor2, determinant));
    return finite4(body.linearVelocityAndInverseMass) &&
        finite4(body.angularVelocity) &&
        finite4(body.inverseInertiaRow0) &&
        finite4(body.inverseInertiaRow1) &&
        finite4(body.inverseInertiaRow2) &&
        body.linearVelocityAndInverseMass.w > 0.0f &&
        abs(row0.y - row1.x) <= symmetryTolerance &&
        abs(row0.z - row2.x) <= symmetryTolerance &&
        abs(row1.z - row2.y) <= symmetryTolerance &&
        minor1 > kMatrixFloor && minor2 > kMatrixFloor &&
        determinant > kMatrixFloor;
}

inline float3 temporalConeRigidInertiaMultiply(
    thread const NumiTemporalConeRigidBody& body,
    const float3 value
) {
    return float3(
        dot(body.inverseInertiaRow0.xyz, value),
        dot(body.inverseInertiaRow1.xyz, value),
        dot(body.inverseInertiaRow2.xyz, value)
    );
}

// Converts rigid contact geometry and body mass properties into the exact
// packed J and M^-1 J^T contract consumed by the generic sparse assembler.
// One SIMD32 group owns one island and each contact lane commits its span only
// after all bodies, frames, ranges, and material values pass validation.
kernel void numi_temporal_cone_rigid_response(
    device const NumiTemporalConeRigidHeader* headers [[buffer(0)]],
    device const NumiTemporalConeRigidBody* bodies [[buffer(1)]],
    device const NumiTemporalConeRigidContact* rigidContacts [[buffer(2)]],
    device const NumiTemporalConeRigidLaw* laws [[buffer(3)]],
    device NumiTemporalConeAssemblyContactSpan* outputSpans [[buffer(4)]],
    device NumiTemporalConeAssemblyTerm* outputTerms [[buffer(5)]],
    device float* outputJacobians [[buffer(6)]],
    device float* outputResponses [[buffer(7)]],
    device NumiTemporalConeIslandContact* outputContacts [[buffer(8)]],
    device float* outputRegularization [[buffer(9)]],
    device NumiTemporalConeRigidStatus* outputStatuses [[buffer(10)]],
    constant uint& problemCount [[buffer(11)]],
    // x bodies, y rigid contacts, z laws, w spans.
    constant uint4& structuralCapacities [[buffer(12)]],
    // x terms, y Jacobian floats, z response floats, w solver contacts.
    constant uint4& valueCapacities [[buffer(13)]],
    constant uint& regularizationCapacity [[buffer(14)]],
    const uint problem [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    if (problem >= problemCount) {
        return;
    }
    const NumiTemporalConeRigidHeader header = headers[problem];
    const uint bodyCount = header.control.y;
    const uint contactCount = header.control.z;
    const uint bodyBase = header.inputRanges.x;
    const uint rigidContactBase = header.inputRanges.y;
    const uint spanBase = header.responseRanges.x;
    const uint termBase = header.responseRanges.y;
    const uint jacobianBase = header.responseRanges.z;
    const uint responseBase = header.responseRanges.w;
    const uint solverContactBase = header.solverRanges.x;
    const uint lawBase = header.solverRanges.z;
    const uint regularizationBase = header.solverRanges.w;
    const bool activeBody = lane < bodyCount;
    const bool activeContact = lane < contactCount;

    if (activeContact && spanBase <= structuralCapacities.w &&
        lane < structuralCapacities.w - spanBase) {
        outputSpans[spanBase + lane] = {};
    }
    if (lane == 0u) {
        outputStatuses[problem] = {};
    }

    uint localFailure = NUMI_TEMPORAL_CONE_RIGID_SUCCESS;
    if (header.control.x != NUMI_TEMPORAL_CONE_RIGID_ABI_VERSION) {
        localFailure = NUMI_TEMPORAL_CONE_RIGID_INVALID_ABI;
    } else if (bodyCount == 0u ||
        bodyCount > NUMI_TEMPORAL_CONE_RIGID_MAX_BODIES ||
        contactCount == 0u ||
        contactCount > NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS ||
        bodyBase > structuralCapacities.x ||
        bodyCount > structuralCapacities.x - bodyBase ||
        rigidContactBase > structuralCapacities.y ||
        contactCount > structuralCapacities.y - rigidContactBase ||
        lawBase > structuralCapacities.z ||
        contactCount > structuralCapacities.z - lawBase ||
        spanBase > structuralCapacities.w ||
        contactCount > structuralCapacities.w - spanBase ||
        termBase > valueCapacities.x ||
        2u * contactCount > valueCapacities.x - termBase ||
        jacobianBase > valueCapacities.y ||
        2u * contactCount * NUMI_TEMPORAL_CONE_RIGID_VALUES_PER_TERM >
            valueCapacities.y - jacobianBase ||
        responseBase > valueCapacities.z ||
        2u * contactCount * NUMI_TEMPORAL_CONE_RIGID_VALUES_PER_TERM >
            valueCapacities.z - responseBase ||
        solverContactBase > valueCapacities.w ||
        contactCount > valueCapacities.w - solverContactBase ||
        regularizationBase > regularizationCapacity ||
        9u * contactCount > regularizationCapacity - regularizationBase) {
        localFailure = NUMI_TEMPORAL_CONE_RIGID_INVALID_INPUT;
    }

    float laneMinimumInverseMass = INFINITY;
    float laneMinimumInertiaProxy = INFINITY;
    if (activeBody && localFailure == NUMI_TEMPORAL_CONE_RIGID_SUCCESS) {
        float minorProxy = 0.0f;
        const NumiTemporalConeRigidBody body = bodies[bodyBase + lane];
        if (!temporalConeRigidInertiaValid(body, minorProxy)) {
            localFailure = NUMI_TEMPORAL_CONE_RIGID_INVALID_INPUT;
        }
        laneMinimumInverseMass = body.linearVelocityAndInverseMass.w;
        laneMinimumInertiaProxy = minorProxy;
    }

    NumiTemporalConeRigidContact contact = {};
    NumiTemporalConeRigidLaw law = {};
    float laneFrameError = 0.0f;
    uint laneTermCount = 0u;
    if (activeContact &&
        localFailure == NUMI_TEMPORAL_CONE_RIGID_SUCCESS) {
        contact = rigidContacts[rigidContactBase + lane];
        law = laws[lawBase + lane];
        const uint bodyA = contact.bodies.x;
        const uint bodyB = contact.bodies.y;
        const float3 normal = contact.normalAndFrictionU.xyz;
        const float3 tangentU = contact.tangentUAndFrictionV.xyz;
        const float3 tangentV = contact.tangentVAndMaximumNormal.xyz;
        laneFrameError = max(
            max(
                max(abs(dot(normal, normal) - 1.0f),
                    abs(dot(tangentU, tangentU) - 1.0f)),
                abs(dot(tangentV, tangentV) - 1.0f)
            ),
            max(
                max(abs(dot(normal, tangentU)), abs(dot(normal, tangentV))),
                max(abs(dot(tangentU, tangentV)),
                    abs(dot(cross(normal, tangentU), tangentV) - 1.0f))
            )
        );
        const bool bodyAValid = bodyA == NUMI_TEMPORAL_CONE_RIGID_STATIC_BODY ||
            bodyA < bodyCount;
        const bool bodyBValid = bodyB == NUMI_TEMPORAL_CONE_RIGID_STATIC_BODY ||
            bodyB < bodyCount;
        if (!bodyAValid || !bodyBValid ||
            bodyA == bodyB ||
            (bodyA == NUMI_TEMPORAL_CONE_RIGID_STATIC_BODY &&
             bodyB == NUMI_TEMPORAL_CONE_RIGID_STATIC_BODY) ||
            !finite4(contact.offsetA) || !finite4(contact.offsetB) ||
            !finite4(contact.normalAndFrictionU) ||
            !finite4(contact.tangentUAndFrictionV) ||
            !finite4(contact.tangentVAndMaximumNormal) ||
            !finite4(contact.bias) || !finite4(contact.warmImpulse) ||
            !finite4(law.stiffnessAndRestitution) ||
            !finite4(law.dampingAndImpactThreshold) ||
            !finite4(law.stabilization) ||
            contact.normalAndFrictionU.w < 0.0f ||
            contact.tangentUAndFrictionV.w < 0.0f ||
            contact.tangentVAndMaximumNormal.w < 0.0f ||
            any(law.stiffnessAndRestitution.xyz < 0.0f) ||
            law.stiffnessAndRestitution.w < 0.0f ||
            law.stiffnessAndRestitution.w > 1.0f ||
            any(law.dampingAndImpactThreshold.xyz < 0.0f) ||
            law.dampingAndImpactThreshold.w < 0.0f ||
            law.stabilization.y < 0.0f ||
            law.stabilization.z < 0.0f ||
            !(law.stabilization.w > 0.0f) ||
            any(law.dampingAndImpactThreshold.xyz +
                law.stabilization.w * law.stiffnessAndRestitution.xyz <=
                0.0f) ||
            laneFrameError > 2.0e-4f) {
            localFailure = NUMI_TEMPORAL_CONE_RIGID_INVALID_INPUT;
        }
        laneTermCount =
            (bodyA == NUMI_TEMPORAL_CONE_RIGID_STATIC_BODY ? 0u : 1u) +
            (bodyB == NUMI_TEMPORAL_CONE_RIGID_STATIC_BODY ? 0u : 1u);
    }
    const uint failure = simd_max(localFailure);
    const float maximumFrameError = simd_max(laneFrameError);
    const float minimumInverseMass = simd_min(laneMinimumInverseMass);
    const float minimumInertiaProxy = simd_min(laneMinimumInertiaProxy);
    const uint generatedTerms = simd_sum(laneTermCount);
    if (failure != NUMI_TEMPORAL_CONE_RIGID_SUCCESS) {
        if (lane == 0u) {
            NumiTemporalConeRigidStatus status = {};
            status.control = uint4(failure, bodyCount, contactCount, 0u);
            status.diagnostics = float4(
                maximumFrameError,
                minimumInverseMass,
                minimumInertiaProxy,
                0.0f
            );
            outputStatuses[problem] = status;
        }
        return;
    }

    if (activeContact) {
        const uint bodyA = contact.bodies.x;
        const uint bodyB = contact.bodies.y;
        const float3 axes[3] = {
            contact.normalAndFrictionU.xyz,
            contact.tangentUAndFrictionV.xyz,
            contact.tangentVAndMaximumNormal.xyz
        };
        const uint firstBody = min(bodyA, bodyB);
        const uint secondBody = max(bodyA, bodyB);
        const uint orderedBodies[2] = {firstBody, secondBody};
        const uint contactTermBase = termBase + 2u * lane;
        for (uint slot = 0u; slot < laneTermCount; ++slot) {
            const uint bodyIndex = orderedBodies[slot];
            const bool isBodyA = bodyIndex == bodyA;
            const float sign = isBodyA ? -1.0f : 1.0f;
            const float3 offset = isBodyA
                ? contact.offsetA.xyz
                : contact.offsetB.xyz;
            const NumiTemporalConeRigidBody body = bodies[
                bodyBase + bodyIndex
            ];
            const uint valueOffset =
                (2u * lane + slot) *
                NUMI_TEMPORAL_CONE_RIGID_VALUES_PER_TERM;
            NumiTemporalConeAssemblyTerm term = {};
            term.control = uint4(
                bodyBase + bodyIndex,
                NUMI_TEMPORAL_CONE_RIGID_DOF,
                jacobianBase + valueOffset,
                responseBase + valueOffset
            );
            outputTerms[contactTermBase + slot] = term;
            for (uint axis = 0u; axis < 3u; ++axis) {
                const float3 linear = sign * axes[axis];
                const float3 angular = sign * cross(offset, axes[axis]);
                const uint jacobianAxisBase =
                    jacobianBase + valueOffset +
                    axis * NUMI_TEMPORAL_CONE_RIGID_DOF;
                outputJacobians[jacobianAxisBase + 0u] = linear.x;
                outputJacobians[jacobianAxisBase + 1u] = linear.y;
                outputJacobians[jacobianAxisBase + 2u] = linear.z;
                outputJacobians[jacobianAxisBase + 3u] = angular.x;
                outputJacobians[jacobianAxisBase + 4u] = angular.y;
                outputJacobians[jacobianAxisBase + 5u] = angular.z;
                const float3 angularResponse =
                    temporalConeRigidInertiaMultiply(body, angular);
                const uint responseTermBase = responseBase + valueOffset;
                outputResponses[responseTermBase + 0u * 3u + axis] =
                    body.linearVelocityAndInverseMass.w * linear.x;
                outputResponses[responseTermBase + 1u * 3u + axis] =
                    body.linearVelocityAndInverseMass.w * linear.y;
                outputResponses[responseTermBase + 2u * 3u + axis] =
                    body.linearVelocityAndInverseMass.w * linear.z;
                outputResponses[responseTermBase + 3u * 3u + axis] =
                    angularResponse.x;
                outputResponses[responseTermBase + 4u * 3u + axis] =
                    angularResponse.y;
                outputResponses[responseTermBase + 5u * 3u + axis] =
                    angularResponse.z;
            }
        }

        float3 velocityA = float3(0.0f);
        float3 velocityB = float3(0.0f);
        if (bodyA != NUMI_TEMPORAL_CONE_RIGID_STATIC_BODY) {
            const NumiTemporalConeRigidBody body = bodies[bodyBase + bodyA];
            velocityA = body.linearVelocityAndInverseMass.xyz +
                cross(body.angularVelocity.xyz, contact.offsetA.xyz);
        }
        if (bodyB != NUMI_TEMPORAL_CONE_RIGID_STATIC_BODY) {
            const NumiTemporalConeRigidBody body = bodies[bodyBase + bodyB];
            velocityB = body.linearVelocityAndInverseMass.xyz +
                cross(body.angularVelocity.xyz, contact.offsetB.xyz);
        }
        const float3 relativeVelocity = velocityB - velocityA;
        const float3 rawVelocity = float3(
            dot(axes[0], relativeVelocity),
            dot(axes[1], relativeVelocity),
            dot(axes[2], relativeVelocity)
        );
        const float timestep = law.stabilization.w;
        const float3 denominator =
            law.dampingAndImpactThreshold.xyz +
            timestep * law.stiffnessAndRestitution.xyz;
        const float3 gamma = 1.0f / (timestep * denominator);
        const float penetration = min(
            law.stabilization.x + law.stabilization.y,
            0.0f
        );
        const float uncappedRecovery =
            -law.stiffnessAndRestitution.x * penetration /
            denominator.x;
        const float recoveryTarget = law.stabilization.z > 0.0f
            ? min(law.stabilization.z, uncappedRecovery)
            : uncappedRecovery;
        const float restitutionTarget =
            rawVelocity.x < -law.dampingAndImpactThreshold.w
            ? -law.stiffnessAndRestitution.w * rawVelocity.x
            : 0.0f;
        const float normalTarget = max(recoveryTarget, restitutionTarget);
        const float3 freeVelocity = rawVelocity + contact.bias.xyz -
            float3(normalTarget, 0.0f, 0.0f);
        if (!finite3(freeVelocity) || !finite3(gamma) ||
            any(gamma <= 0.0f)) {
            // Body and frame inputs are already finite; this guards overflow.
            outputSpans[spanBase + lane] = {};
        } else {
            NumiTemporalConeIslandContact solverContact = {};
            solverContact.freeVelocityAndFrictionU = float4(
                freeVelocity,
                contact.normalAndFrictionU.w
            );
            solverContact.warmImpulseAndFrictionV = float4(
                contact.warmImpulse.xyz,
                contact.tangentUAndFrictionV.w
            );
            solverContact.limits.x =
                contact.tangentVAndMaximumNormal.w;
            outputContacts[solverContactBase + lane] = solverContact;
            const uint regularizationContactBase =
                regularizationBase + 9u * lane;
            for (uint element = 0u; element < 9u; ++element) {
                outputRegularization[regularizationContactBase + element] =
                    element % 4u == 0u ? gamma[element / 4u] : 0.0f;
            }
            NumiTemporalConeAssemblyContactSpan span = {};
            span.ranges = uint4(contactTermBase, laneTermCount, 0u, 0u);
            outputSpans[spanBase + lane] = span;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    uint nonfiniteOutput = 0u;
    if (activeContact && outputSpans[spanBase + lane].ranges.y == 0u) {
        nonfiniteOutput = NUMI_TEMPORAL_CONE_RIGID_NONFINITE_RESULT;
    }
    const uint outputFailure = simd_max(nonfiniteOutput);
    if (outputFailure != NUMI_TEMPORAL_CONE_RIGID_SUCCESS && activeContact) {
        outputSpans[spanBase + lane] = {};
    }
    if (lane == 0u) {
        NumiTemporalConeRigidStatus status = {};
        status.control = uint4(
            outputFailure,
            bodyCount,
            contactCount,
            outputFailure == NUMI_TEMPORAL_CONE_RIGID_SUCCESS
                ? generatedTerms
                : 0u
        );
        status.diagnostics = float4(
            maximumFrameError,
            minimumInverseMass,
            minimumInertiaProxy,
            0.0f
        );
        outputStatuses[problem] = status;
    }
}

// Rotates analytic world-point Jacobians into contact coordinates without
// requiring a dense mass factor. This is the preparation boundary for the
// streamed inverse-ABA response path. It publishes only complete contact rows
// after ABI, upstream-kinematics, capacity, and frame validation succeed.
kernel void numi_temporal_cone_articulated_prepare_jacobians(
    device const NumiTemporalConeArticulatedHeader* headers [[buffer(0)]],
    device const MRArticulatedOperatorStatusGPU* operatorStatuses
        [[buffer(1)]],
    device const float* pointJacobians [[buffer(2)]],
    device const NumiTemporalConeArticulatedContact* contacts [[buffer(3)]],
    device float* outputJacobians [[buffer(4)]],
    device NumiTemporalConeArticulatedStatus* outputStatuses [[buffer(5)]],
    device MRMetalWorldContactStatusGPU* inverseGates [[buffer(6)]],
    constant uint& problemCount [[buffer(7)]],
    // x point-Jacobian floats, y contacts, z operator statuses, w reserved.
    constant uint4& inputCapacities [[buffer(8)]],
    constant uint& outputCapacity [[buffer(9)]],
    const uint problem [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    if (problem >= problemCount) {
        return;
    }
    const NumiTemporalConeArticulatedHeader header = headers[problem];
    const uint dofCount = header.control.y;
    const uint contactCount = header.control.z;
    const uint pointJacobianBase = header.inputRanges.y;
    const uint contactBase = header.inputRanges.w;
    const uint jacobianBase = header.responseRanges.z;
    const uint operatorStatusIndex = header.operatorRanges.x;
    const bool activeContact = lane < contactCount;
    const uint valuesPerContact = 3u * dofCount;
    if (lane == 0u) {
        outputStatuses[problem] = {};
        MRMetalWorldContactStatusGPU gate = {};
        gate.code = MR_STEP_UNSUPPORTED;
        gate.environment = problem;
        inverseGates[problem] = gate;
    }

    uint localFailure = NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS;
    if (header.control.x != NUMI_TEMPORAL_CONE_ARTICULATED_ABI_VERSION) {
        localFailure = NUMI_TEMPORAL_CONE_ARTICULATED_INVALID_ABI;
    } else if (dofCount == 0u ||
        dofCount > NUMI_TEMPORAL_CONE_ARTICULATED_MAX_DOF ||
        contactCount == 0u ||
        contactCount > NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS ||
        header.operatorRanges.y !=
            NUMI_TEMPORAL_CONE_ARTICULATED_RESPONSE_INVERSE_ABA ||
        any(header.operatorRanges.zw != uint2(0u)) ||
        contactBase > inputCapacities.y ||
        contactCount > inputCapacities.y - contactBase ||
        operatorStatusIndex >= inputCapacities.z ||
        jacobianBase > outputCapacity ||
        contactCount * valuesPerContact > outputCapacity - jacobianBase) {
        localFailure = NUMI_TEMPORAL_CONE_ARTICULATED_INVALID_INPUT;
    }

    MRArticulatedOperatorStatusGPU operatorStatus = {};
    if (localFailure == NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS) {
        operatorStatus = operatorStatuses[operatorStatusIndex];
        if (operatorStatus.code != MR_ARTICULATED_OPERATOR_SUCCESS ||
            operatorStatus.nv != dofCount ||
            !finite4(operatorStatus.diagnostics)) {
            localFailure = NUMI_TEMPORAL_CONE_ARTICULATED_UPSTREAM_FAILURE;
        }
    }

    NumiTemporalConeArticulatedContact contact = {};
    float3 axes[3] = {
        float3(0.0f), float3(0.0f), float3(0.0f)
    };
    uint pointIndex = 0u;
    float laneFrameError = 0.0f;
    if (activeContact &&
        localFailure == NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS) {
        contact = contacts[contactBase + lane];
        pointIndex = contact.control.x;
        axes[0] = contact.normalAndFrictionU.xyz;
        axes[1] = contact.tangentUAndFrictionV.xyz;
        axes[2] = contact.tangentVAndMaximumNormal.xyz;
        laneFrameError = max(
            max(
                max(abs(dot(axes[0], axes[0]) - 1.0f),
                    abs(dot(axes[1], axes[1]) - 1.0f)),
                abs(dot(axes[2], axes[2]) - 1.0f)
            ),
            max(
                max(abs(dot(axes[0], axes[1])),
                    abs(dot(axes[0], axes[2]))),
                max(abs(dot(axes[1], axes[2])),
                    abs(dot(cross(axes[0], axes[1]), axes[2]) - 1.0f))
            )
        );
        const ulong pointEnd =
            static_cast<ulong>(pointJacobianBase) +
            (static_cast<ulong>(pointIndex) + 1ul) * 3ul * dofCount;
        if (any(contact.control.yzw != uint3(0u)) ||
            pointEnd > inputCapacities.x ||
            !finite4(contact.normalAndFrictionU) ||
            !finite4(contact.tangentUAndFrictionV) ||
            !finite4(contact.tangentVAndMaximumNormal) ||
            laneFrameError > 2.0e-4f) {
            localFailure = NUMI_TEMPORAL_CONE_ARTICULATED_INVALID_INPUT;
        }
    }

    const uint failure = simd_max(localFailure);
    const float maximumFrameError = simd_max(laneFrameError);
    if (failure != NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS) {
        if (lane == 0u) {
            NumiTemporalConeArticulatedStatus status = {};
            status.control = uint4(failure, dofCount, contactCount, 0u);
            status.diagnostics.x = maximumFrameError;
            outputStatuses[problem] = status;
            MRMetalWorldContactStatusGPU gate = {};
            gate.code = MR_STEP_UNSUPPORTED;
            gate.environment = problem;
            inverseGates[problem] = gate;
        }
        return;
    }

    if (activeContact) {
        const uint worldPointBase = pointJacobianBase +
            pointIndex * 3u * dofCount;
        const uint contactValueBase =
            jacobianBase + lane * valuesPerContact;
        for (uint dof = 0u; dof < dofCount; ++dof) {
            const float3 worldColumn = float3(
                pointJacobians[worldPointBase + 0u * dofCount + dof],
                pointJacobians[worldPointBase + 1u * dofCount + dof],
                pointJacobians[worldPointBase + 2u * dofCount + dof]
            );
            if (!finite3(worldColumn)) {
                localFailure = NUMI_TEMPORAL_CONE_ARTICULATED_INVALID_INPUT;
            }
            for (uint axis = 0u; axis < 3u; ++axis) {
                outputJacobians[
                    contactValueBase + axis * dofCount + dof
                ] = dot(axes[axis], worldColumn);
            }
        }
    }
    const uint publicationFailure = simd_max(localFailure);
    if (lane == 0u) {
        NumiTemporalConeArticulatedStatus status = {};
        status.control = uint4(
            publicationFailure,
            dofCount,
            contactCount,
            publicationFailure == NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS
                ? contactCount
                : 0u
        );
        status.diagnostics.x = maximumFrameError;
        outputStatuses[problem] = status;
        MRMetalWorldContactStatusGPU gate = {};
        gate.code = publicationFailure ==
                NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS
            ? MR_STEP_SUCCESS
            : MR_STEP_UNSUPPORTED;
        gate.environment = problem;
        gate.requiredConstraints = publicationFailure ==
                NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS
            ? contactCount
            : 0u;
        inverseGates[problem] = gate;
    }
}

// Finalizes streamed inverse-ABA columns into the generic assembly layout.
// Prepared Jacobians and inverse responses are both [contact][axis][DoF]; the
// assembler consumes Jacobians in that layout and responses transposed to
// [contact][DoF][axis]. Spans are published only after every response value,
// contact law, and upstream status passes validation.
kernel void numi_temporal_cone_articulated_finalize_inverse(
    device const NumiTemporalConeArticulatedHeader* headers [[buffer(0)]],
    device const float* inputVelocities [[buffer(1)]],
    device const NumiTemporalConeArticulatedContact* contacts [[buffer(2)]],
    device const NumiTemporalConeRigidLaw* laws [[buffer(3)]],
    device const float* preparedJacobians [[buffer(4)]],
    device const float* inverseResponses [[buffer(5)]],
    device const NumiTemporalConeArticulatedStatus* preparationStatuses
        [[buffer(6)]],
    device const MRInverseMassStatusGPU* inverseStatuses [[buffer(7)]],
    device NumiTemporalConeAssemblyContactSpan* outputSpans [[buffer(8)]],
    device NumiTemporalConeAssemblyTerm* outputTerms [[buffer(9)]],
    device float* outputResponses [[buffer(10)]],
    device NumiTemporalConeIslandContact* outputContacts [[buffer(11)]],
    device float* outputRegularization [[buffer(12)]],
    device NumiTemporalConeArticulatedStatus* outputStatuses [[buffer(13)]],
    constant uint& problemCount [[buffer(14)]],
    // x velocities, y articulated contacts, z laws, w prepared Jacobians.
    constant uint4& inputCapacities [[buffer(15)]],
    // x inverse responses, y preparation statuses, z inverse statuses,
    // w reserved.
    constant uint4& inverseCapacities [[buffer(16)]],
    // x spans, y terms, z transposed responses, w solver contacts.
    constant uint4& outputCapacities [[buffer(17)]],
    constant uint& regularizationCapacity [[buffer(18)]],
    // x conservative trace upper, y minimum authored armature.
    device const float2* inverseConditionBounds [[buffer(19)]],
    const uint problem [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    if (problem >= problemCount) {
        return;
    }
    const NumiTemporalConeArticulatedHeader header = headers[problem];
    const uint dofCount = header.control.y;
    const uint contactCount = header.control.z;
    const uint velocityBase = header.inputRanges.z;
    const uint contactBase = header.inputRanges.w;
    const uint spanBase = header.responseRanges.x;
    const uint termBase = header.responseRanges.y;
    const uint jacobianBase = header.responseRanges.z;
    const uint responseBase = header.responseRanges.w;
    const uint solverContactBase = header.solverRanges.x;
    const uint lawBase = header.solverRanges.z;
    const uint regularizationBase = header.solverRanges.w;
    const uint valuesPerContact = 3u * dofCount;
    const bool activeContact = lane < contactCount;
    if (activeContact && spanBase <= outputCapacities.x &&
        lane < outputCapacities.x - spanBase) {
        outputSpans[spanBase + lane] = {};
    }
    if (lane == 0u) {
        outputStatuses[problem] = {};
    }

    uint localFailure = NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS;
    if (header.control.x != NUMI_TEMPORAL_CONE_ARTICULATED_ABI_VERSION) {
        localFailure = NUMI_TEMPORAL_CONE_ARTICULATED_INVALID_ABI;
    } else if (dofCount == 0u ||
        dofCount > NUMI_TEMPORAL_CONE_ARTICULATED_MAX_DOF ||
        contactCount == 0u ||
        contactCount > NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS ||
        header.operatorRanges.y !=
            NUMI_TEMPORAL_CONE_ARTICULATED_RESPONSE_INVERSE_ABA ||
        any(header.operatorRanges.zw != uint2(0u)) ||
        velocityBase > inputCapacities.x ||
        dofCount > inputCapacities.x - velocityBase ||
        contactBase > inputCapacities.y ||
        contactCount > inputCapacities.y - contactBase ||
        lawBase > inputCapacities.z ||
        contactCount > inputCapacities.z - lawBase ||
        jacobianBase > inputCapacities.w ||
        contactCount * valuesPerContact >
            inputCapacities.w - jacobianBase ||
        jacobianBase > inverseCapacities.x ||
        contactCount * valuesPerContact >
            inverseCapacities.x - jacobianBase ||
        problem >= inverseCapacities.y ||
        problem >= inverseCapacities.z ||
        inverseCapacities.w != 0u ||
        spanBase > outputCapacities.x ||
        contactCount > outputCapacities.x - spanBase ||
        termBase > outputCapacities.y ||
        contactCount > outputCapacities.y - termBase ||
        responseBase > outputCapacities.z ||
        contactCount * valuesPerContact >
            outputCapacities.z - responseBase ||
        solverContactBase > outputCapacities.w ||
        contactCount > outputCapacities.w - solverContactBase ||
        regularizationBase > regularizationCapacity ||
        contactCount * 9u >
            regularizationCapacity - regularizationBase) {
        localFailure = NUMI_TEMPORAL_CONE_ARTICULATED_INVALID_INPUT;
    }

    NumiTemporalConeArticulatedStatus preparationStatus = {};
    MRInverseMassStatusGPU inverseStatus = {};
    float inverseConditionUpper = INFINITY;
    if (localFailure == NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS) {
        preparationStatus = preparationStatuses[problem];
        inverseStatus = inverseStatuses[problem];
        if (preparationStatus.control.x !=
                NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS ||
            preparationStatus.control.y != dofCount ||
            preparationStatus.control.z != contactCount ||
            preparationStatus.control.w != contactCount) {
            localFailure = NUMI_TEMPORAL_CONE_ARTICULATED_UPSTREAM_FAILURE;
        } else if (inverseStatus.code != MR_INVERSE_MASS_SUCCESS ||
            inverseStatus.environment != problem ||
            inverseStatus.nv != dofCount ||
            inverseStatus.rhsCount != 3u * contactCount ||
            !finite4(inverseStatus.diagnostics) ||
            !(inverseStatus.diagnostics.x > 0.0f) ||
            inverseStatus.diagnostics.y < inverseStatus.diagnostics.x) {
            localFailure =
                NUMI_TEMPORAL_CONE_ARTICULATED_INVERSE_MASS_FAILED;
        } else {
            const float2 conditionBounds = inverseConditionBounds[problem];
            inverseConditionUpper = conditionBounds.x / conditionBounds.y;
            if (!all(isfinite(conditionBounds)) ||
                !(conditionBounds.x > 0.0f) ||
                !(conditionBounds.y > 0.0f) ||
                !isfinite(inverseConditionUpper) ||
                inverseConditionUpper >
                    NUMI_TEMPORAL_CONE_ARTICULATED_MAX_INVERSE_CONDITION_UPPER) {
                localFailure =
                    NUMI_TEMPORAL_CONE_ARTICULATED_CONDITIONING_FAILED;
            }
        }
    }

    NumiTemporalConeArticulatedContact contact = {};
    NumiTemporalConeRigidLaw law = {};
    if (activeContact &&
        localFailure == NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS) {
        contact = contacts[contactBase + lane];
        law = laws[lawBase + lane];
        if (!finite4(contact.normalAndFrictionU) ||
            !finite4(contact.tangentUAndFrictionV) ||
            !finite4(contact.tangentVAndMaximumNormal) ||
            !finite4(contact.bias) || !finite4(contact.warmImpulse) ||
            contact.normalAndFrictionU.w < 0.0f ||
            contact.tangentUAndFrictionV.w < 0.0f ||
            contact.tangentVAndMaximumNormal.w < 0.0f ||
            contact.bias.w != 0.0f || contact.warmImpulse.w != 0.0f ||
            !finite4(law.stiffnessAndRestitution) ||
            !finite4(law.dampingAndImpactThreshold) ||
            !finite4(law.stabilization) ||
            any(law.stiffnessAndRestitution.xyz < 0.0f) ||
            law.stiffnessAndRestitution.w < 0.0f ||
            law.stiffnessAndRestitution.w > 1.0f ||
            any(law.dampingAndImpactThreshold < 0.0f) ||
            law.stabilization.y < 0.0f ||
            law.stabilization.z < 0.0f ||
            !(law.stabilization.w > 0.0f) ||
            any(law.dampingAndImpactThreshold.xyz +
                law.stabilization.w * law.stiffnessAndRestitution.xyz <=
                0.0f)) {
            localFailure = NUMI_TEMPORAL_CONE_ARTICULATED_INVALID_INPUT;
        }
    }
    uint failure = simd_max(localFailure);
    if (failure != NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS) {
        if (lane == 0u) {
            NumiTemporalConeArticulatedStatus status = preparationStatus;
            status.control = uint4(failure, dofCount, contactCount, 0u);
            status.conditioning = float4(
                inverseStatus.diagnostics.x,
                inverseStatus.diagnostics.y,
                inverseConditionUpper,
                0.0f
            );
            status.diagnostics.w = float(inverseStatus.code);
            outputStatuses[problem] = status;
        }
        return;
    }

    if (activeContact) {
        const uint contactValueBase = lane * valuesPerContact;
        float3 rawVelocity = float3(0.0f);
        for (uint axis = 0u; axis < 3u; ++axis) {
            for (uint dof = 0u; dof < dofCount; ++dof) {
                const float jacobian = preparedJacobians[
                    jacobianBase + contactValueBase + axis * dofCount + dof
                ];
                const float response = inverseResponses[
                    jacobianBase + contactValueBase + axis * dofCount + dof
                ];
                if (!isfinite(jacobian) || !isfinite(response)) {
                    localFailure =
                        NUMI_TEMPORAL_CONE_ARTICULATED_NONFINITE_RESULT;
                }
                outputResponses[
                    responseBase + contactValueBase + dof * 3u + axis
                ] = response;
                rawVelocity[axis] = fma(
                    jacobian,
                    inputVelocities[velocityBase + dof],
                    rawVelocity[axis]
                );
            }
        }
        const float timestep = law.stabilization.w;
        const float3 contactDenominator =
            law.dampingAndImpactThreshold.xyz +
            timestep * law.stiffnessAndRestitution.xyz;
        const float3 gamma = 1.0f / (timestep * contactDenominator);
        const float penetration = min(
            law.stabilization.x + law.stabilization.y,
            0.0f
        );
        const float uncappedRecovery =
            -law.stiffnessAndRestitution.x * penetration /
            contactDenominator.x;
        const float recoveryTarget = law.stabilization.z > 0.0f
            ? min(law.stabilization.z, uncappedRecovery)
            : uncappedRecovery;
        const float restitutionTarget =
            rawVelocity.x < -law.dampingAndImpactThreshold.w
            ? -law.stiffnessAndRestitution.w * rawVelocity.x
            : 0.0f;
        const float normalTarget = max(recoveryTarget, restitutionTarget);
        const float3 freeVelocity = rawVelocity + contact.bias.xyz -
            float3(normalTarget, 0.0f, 0.0f);
        if (!finite3(freeVelocity) || !finite3(gamma) ||
            any(gamma <= 0.0f)) {
            localFailure =
                NUMI_TEMPORAL_CONE_ARTICULATED_NONFINITE_RESULT;
        } else {
            NumiTemporalConeAssemblyTerm term = {};
            term.control = uint4(
                header.control.w,
                dofCount,
                jacobianBase + contactValueBase,
                responseBase + contactValueBase
            );
            outputTerms[termBase + lane] = term;
            NumiTemporalConeIslandContact solverContact = {};
            solverContact.freeVelocityAndFrictionU = float4(
                freeVelocity,
                contact.normalAndFrictionU.w
            );
            solverContact.warmImpulseAndFrictionV = float4(
                contact.warmImpulse.xyz,
                contact.tangentUAndFrictionV.w
            );
            solverContact.limits.x =
                contact.tangentVAndMaximumNormal.w;
            outputContacts[solverContactBase + lane] = solverContact;
            const uint contactRegularizationBase =
                regularizationBase + 9u * lane;
            for (uint element = 0u; element < 9u; ++element) {
                outputRegularization[
                    contactRegularizationBase + element
                ] = element % 4u == 0u
                    ? gamma[element / 4u]
                    : 0.0f;
            }
        }
    }
    failure = simd_max(localFailure);
    if (failure == NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS &&
        activeContact) {
        NumiTemporalConeAssemblyContactSpan span = {};
        span.ranges = uint4(termBase + lane, 1u, 0u, 0u);
        outputSpans[spanBase + lane] = span;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (failure != NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS &&
        activeContact) {
        outputSpans[spanBase + lane] = {};
    }
    if (lane == 0u) {
        NumiTemporalConeArticulatedStatus status = preparationStatus;
        status.control = uint4(
            failure,
            dofCount,
            contactCount,
            failure == NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS
                ? contactCount
                : 0u
        );
        status.conditioning = float4(
            inverseStatus.diagnostics.x,
            inverseStatus.diagnostics.y,
            inverseConditionUpper,
            0.0f
        );
        status.diagnostics = float4(
            preparationStatus.diagnostics.x,
            0.0f,
            0.0f,
            float(inverseStatus.code)
        );
        outputStatuses[problem] = status;
    }
}

// Converts the canonical articulated operator's lower Cholesky factor and
// analytic world-point Jacobians into the exact packed J / M^-1 J^T contract
// consumed by the generic sparse assembler. Each active contact lane solves
// three checked triangular systems. No explicit inverse is formed.
kernel void numi_temporal_cone_articulated_response(
    device const NumiTemporalConeArticulatedHeader* headers [[buffer(0)]],
    device const MRArticulatedOperatorStatusGPU* operatorStatuses
        [[buffer(1)]],
    device const float* factors [[buffer(2)]],
    device const float* pointJacobians [[buffer(3)]],
    device const float* inputVelocities [[buffer(4)]],
    device const NumiTemporalConeArticulatedContact* contacts [[buffer(5)]],
    device const NumiTemporalConeRigidLaw* laws [[buffer(6)]],
    device NumiTemporalConeAssemblyContactSpan* outputSpans [[buffer(7)]],
    device NumiTemporalConeAssemblyTerm* outputTerms [[buffer(8)]],
    device float* outputJacobians [[buffer(9)]],
    device float* outputResponses [[buffer(10)]],
    device NumiTemporalConeIslandContact* outputContacts [[buffer(11)]],
    device float* outputRegularization [[buffer(12)]],
    device NumiTemporalConeArticulatedStatus* outputStatuses [[buffer(13)]],
    constant uint& problemCount [[buffer(14)]],
    // x factor floats, y point-Jacobian floats, z input velocities,
    // w articulated contacts.
    constant uint4& inputCapacities [[buffer(15)]],
    // x spans, y terms, z output Jacobian floats, w response floats.
    constant uint4& responseCapacities [[buffer(16)]],
    // x solver contacts, y laws, z regularization floats, w operator statuses.
    constant uint4& solverCapacities [[buffer(17)]],
    const uint problem [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    if (problem >= problemCount) {
        return;
    }
    const NumiTemporalConeArticulatedHeader header = headers[problem];
    const uint dofCount = header.control.y;
    const uint contactCount = header.control.z;
    const uint factorBase = header.inputRanges.x;
    const uint pointJacobianBase = header.inputRanges.y;
    const uint velocityBase = header.inputRanges.z;
    const uint contactBase = header.inputRanges.w;
    const uint spanBase = header.responseRanges.x;
    const uint termBase = header.responseRanges.y;
    const uint jacobianBase = header.responseRanges.z;
    const uint responseBase = header.responseRanges.w;
    const uint solverContactBase = header.solverRanges.x;
    const uint lawBase = header.solverRanges.z;
    const uint regularizationBase = header.solverRanges.w;
    const uint operatorStatusIndex = header.operatorRanges.x;
    const bool activeDof = lane < dofCount;
    const bool activeContact = lane < contactCount;
    const uint valuesPerContact = 3u * dofCount;

    if (activeContact && spanBase <= responseCapacities.x &&
        lane < responseCapacities.x - spanBase) {
        outputSpans[spanBase + lane] = {};
    }
    if (lane == 0u) {
        outputStatuses[problem] = {};
    }

    uint localFailure = NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS;
    if (header.control.x != NUMI_TEMPORAL_CONE_ARTICULATED_ABI_VERSION) {
        localFailure = NUMI_TEMPORAL_CONE_ARTICULATED_INVALID_ABI;
    } else if (dofCount == 0u ||
        dofCount > NUMI_TEMPORAL_CONE_ARTICULATED_MAX_DOF ||
        contactCount == 0u ||
        contactCount > NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS ||
        header.operatorRanges.y !=
            NUMI_TEMPORAL_CONE_ARTICULATED_RESPONSE_DENSE_FACTOR ||
        any(header.operatorRanges.zw != uint2(0u)) ||
        factorBase > inputCapacities.x ||
        dofCount * dofCount > inputCapacities.x - factorBase ||
        velocityBase > inputCapacities.z ||
        dofCount > inputCapacities.z - velocityBase ||
        contactBase > inputCapacities.w ||
        contactCount > inputCapacities.w - contactBase ||
        spanBase > responseCapacities.x ||
        contactCount > responseCapacities.x - spanBase ||
        termBase > responseCapacities.y ||
        contactCount > responseCapacities.y - termBase ||
        jacobianBase > responseCapacities.z ||
        contactCount * valuesPerContact >
            responseCapacities.z - jacobianBase ||
        responseBase > responseCapacities.w ||
        contactCount * valuesPerContact >
            responseCapacities.w - responseBase ||
        solverContactBase > solverCapacities.x ||
        contactCount > solverCapacities.x - solverContactBase ||
        lawBase > solverCapacities.y ||
        contactCount > solverCapacities.y - lawBase ||
        regularizationBase > solverCapacities.z ||
        9u * contactCount > solverCapacities.z - regularizationBase ||
        operatorStatusIndex >= solverCapacities.w) {
        localFailure = NUMI_TEMPORAL_CONE_ARTICULATED_INVALID_INPUT;
    }

    MRArticulatedOperatorStatusGPU operatorStatus = {};
    if (localFailure == NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS) {
        operatorStatus = operatorStatuses[operatorStatusIndex];
        if (operatorStatus.code != MR_ARTICULATED_OPERATOR_SUCCESS ||
            operatorStatus.nv != dofCount ||
            !finite4(operatorStatus.diagnostics)) {
            localFailure = NUMI_TEMPORAL_CONE_ARTICULATED_UPSTREAM_FAILURE;
        }
    }

    float laneMinimumPivot = INFINITY;
    float laneMaximumPivot = 0.0f;
    float laneMaximumFactor = 0.0f;
    if (activeDof &&
        localFailure == NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS) {
        for (uint column = 0u; column < dofCount; ++column) {
            const float value = factors[
                factorBase + lane * dofCount + column
            ];
            if (!isfinite(value) ||
                (column > lane && value != 0.0f)) {
                localFailure =
                    NUMI_TEMPORAL_CONE_ARTICULATED_INVALID_INPUT;
            }
            laneMaximumFactor = max(laneMaximumFactor, abs(value));
        }
        const float pivot = factors[
            factorBase + lane * dofCount + lane
        ];
        if (!(pivot > 0.0f) || !isfinite(pivot)) {
            localFailure =
                NUMI_TEMPORAL_CONE_ARTICULATED_FACTORIZATION_FAILED;
        }
        laneMinimumPivot = pivot;
        laneMaximumPivot = pivot;
        if (!isfinite(inputVelocities[velocityBase + lane])) {
            localFailure = NUMI_TEMPORAL_CONE_ARTICULATED_INVALID_INPUT;
        }
    }

    NumiTemporalConeArticulatedContact contact = {};
    NumiTemporalConeRigidLaw law = {};
    float3 axes[3] = {
        float3(0.0f), float3(0.0f), float3(0.0f)
    };
    uint pointIndex = 0u;
    float laneFrameError = 0.0f;
    if (activeContact &&
        localFailure == NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS) {
        contact = contacts[contactBase + lane];
        law = laws[lawBase + lane];
        pointIndex = contact.control.x;
        axes[0] = contact.normalAndFrictionU.xyz;
        axes[1] = contact.tangentUAndFrictionV.xyz;
        axes[2] = contact.tangentVAndMaximumNormal.xyz;
        laneFrameError = max(
            max(
                max(abs(dot(axes[0], axes[0]) - 1.0f),
                    abs(dot(axes[1], axes[1]) - 1.0f)),
                abs(dot(axes[2], axes[2]) - 1.0f)
            ),
            max(
                max(abs(dot(axes[0], axes[1])),
                    abs(dot(axes[0], axes[2]))),
                max(abs(dot(axes[1], axes[2])),
                    abs(dot(cross(axes[0], axes[1]), axes[2]) - 1.0f))
            )
        );
        const ulong pointEnd =
            static_cast<ulong>(pointJacobianBase) +
            (static_cast<ulong>(pointIndex) + 1ul) * 3ul * dofCount;
        if (any(contact.control.yzw != uint3(0u)) ||
            pointEnd > inputCapacities.y ||
            !finite4(contact.normalAndFrictionU) ||
            !finite4(contact.tangentUAndFrictionV) ||
            !finite4(contact.tangentVAndMaximumNormal) ||
            !finite4(contact.bias) || !finite4(contact.warmImpulse) ||
            contact.normalAndFrictionU.w < 0.0f ||
            contact.tangentUAndFrictionV.w < 0.0f ||
            contact.tangentVAndMaximumNormal.w < 0.0f ||
            contact.bias.w != 0.0f || contact.warmImpulse.w != 0.0f ||
            !finite4(law.stiffnessAndRestitution) ||
            !finite4(law.dampingAndImpactThreshold) ||
            !finite4(law.stabilization) ||
            any(law.stiffnessAndRestitution.xyz < 0.0f) ||
            law.stiffnessAndRestitution.w < 0.0f ||
            law.stiffnessAndRestitution.w > 1.0f ||
            any(law.dampingAndImpactThreshold < 0.0f) ||
            law.stabilization.y < 0.0f ||
            law.stabilization.z < 0.0f ||
            !(law.stabilization.w > 0.0f) ||
            any(law.dampingAndImpactThreshold.xyz +
                law.stabilization.w * law.stiffnessAndRestitution.xyz <=
                0.0f) ||
            laneFrameError > 2.0e-4f) {
            localFailure = NUMI_TEMPORAL_CONE_ARTICULATED_INVALID_INPUT;
        }
    }

    uint failure = simd_max(localFailure);
    const float minimumPivot = simd_min(laneMinimumPivot);
    const float maximumPivot = simd_max(laneMaximumPivot);
    const float maximumFactor = simd_max(laneMaximumFactor);
    const float maximumFrameError = simd_max(laneFrameError);
    if (failure != NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS) {
        if (lane == 0u) {
            NumiTemporalConeArticulatedStatus status = {};
            status.control = uint4(failure, dofCount, contactCount, 0u);
            status.conditioning = float4(
                minimumPivot,
                maximumPivot,
                0.0f,
                operatorStatus.diagnostics.z
            );
            status.diagnostics.x = maximumFrameError;
            outputStatuses[problem] = status;
        }
        return;
    }

    // Compute the exact FP32 infinity-norm condition estimate of the factor
    // operator: ||L L^T||_inf * ||(L L^T)^-1||_inf. Symmetry means solving
    // one unit RHS per DoF lane yields both an inverse column and its matching
    // row absolute sum. This is more informative than a pivot-ratio proxy and
    // remains deterministic for the fixed factor order.
    float laneMatrixRowNorm = 0.0f;
    float laneInverseRowNorm = 0.0f;
    if (activeDof) {
        for (uint column = 0u; column < dofCount; ++column) {
            float coefficient = 0.0f;
            const uint innerCount = min(lane, column) + 1u;
            for (uint inner = 0u; inner < innerCount; ++inner) {
                coefficient = fma(
                    factors[factorBase + lane * dofCount + inner],
                    factors[factorBase + column * dofCount + inner],
                    coefficient
                );
            }
            laneMatrixRowNorm += abs(coefficient);
        }
        float intermediate[NUMI_TEMPORAL_CONE_ARTICULATED_MAX_DOF];
        float solution[NUMI_TEMPORAL_CONE_ARTICULATED_MAX_DOF];
        for (uint row = 0u; row < dofCount; ++row) {
            float value = row == lane ? 1.0f : 0.0f;
            for (uint column = 0u; column < row; ++column) {
                value -= factors[
                    factorBase + row * dofCount + column
                ] * intermediate[column];
            }
            intermediate[row] = value / factors[
                factorBase + row * dofCount + row
            ];
        }
        for (uint reverse = 0u; reverse < dofCount; ++reverse) {
            const uint row = dofCount - 1u - reverse;
            float value = intermediate[row];
            for (uint column = row + 1u; column < dofCount; ++column) {
                value -= factors[
                    factorBase + column * dofCount + row
                ] * solution[column];
            }
            solution[row] = value / factors[
                factorBase + row * dofCount + row
            ];
            laneInverseRowNorm += abs(solution[row]);
        }
        if (!isfinite(laneMatrixRowNorm) ||
            !isfinite(laneInverseRowNorm)) {
            localFailure =
                NUMI_TEMPORAL_CONE_ARTICULATED_NONFINITE_RESULT;
        }
    }
    const float matrixInfinityNorm = simd_max(laneMatrixRowNorm);
    const float inverseInfinityNorm = simd_max(laneInverseRowNorm);
    const float conditionInfinity =
        matrixInfinityNorm * inverseInfinityNorm;
    if (lane == 0u) {
        if (!isfinite(conditionInfinity)) {
            localFailure =
                NUMI_TEMPORAL_CONE_ARTICULATED_NONFINITE_RESULT;
        } else if (conditionInfinity >
                NUMI_TEMPORAL_CONE_ARTICULATED_MAX_CONDITION_INFINITY) {
            localFailure =
                NUMI_TEMPORAL_CONE_ARTICULATED_CONDITIONING_FAILED;
        }
    }
    failure = simd_max(localFailure);
    if (failure != NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS) {
        if (lane == 0u) {
            NumiTemporalConeArticulatedStatus status = {};
            status.control = uint4(failure, dofCount, contactCount, 0u);
            status.conditioning = float4(
                minimumPivot,
                maximumPivot,
                conditionInfinity,
                operatorStatus.diagnostics.z
            );
            status.diagnostics.x = maximumFrameError;
            outputStatuses[problem] = status;
        }
        return;
    }

    float laneMaximumBackwardError = 0.0f;
    if (activeContact) {
        const uint worldPointBase = pointJacobianBase +
            pointIndex * 3u * dofCount;
        const uint contactValueBase = lane * valuesPerContact;
        for (uint dof = 0u; dof < dofCount; ++dof) {
            const float3 worldColumn = float3(
                pointJacobians[worldPointBase + 0u * dofCount + dof],
                pointJacobians[worldPointBase + 1u * dofCount + dof],
                pointJacobians[worldPointBase + 2u * dofCount + dof]
            );
            if (!finite3(worldColumn)) {
                localFailure = NUMI_TEMPORAL_CONE_ARTICULATED_INVALID_INPUT;
            }
            for (uint axis = 0u; axis < 3u; ++axis) {
                outputJacobians[
                    jacobianBase + contactValueBase + axis * dofCount + dof
                ] = dot(axes[axis], worldColumn);
            }
        }

        for (uint axis = 0u; axis < 3u; ++axis) {
            float intermediate[NUMI_TEMPORAL_CONE_ARTICULATED_MAX_DOF];
            float solution[NUMI_TEMPORAL_CONE_ARTICULATED_MAX_DOF];
            for (uint row = 0u; row < dofCount; ++row) {
                float value = outputJacobians[
                    jacobianBase + contactValueBase + axis * dofCount + row
                ];
                for (uint column = 0u; column < row; ++column) {
                    value -= factors[
                        factorBase + row * dofCount + column
                    ] * intermediate[column];
                }
                intermediate[row] = value / factors[
                    factorBase + row * dofCount + row
                ];
            }
            for (uint reverse = 0u; reverse < dofCount; ++reverse) {
                const uint row = dofCount - 1u - reverse;
                float value = intermediate[row];
                for (uint column = row + 1u;
                     column < dofCount;
                     ++column) {
                    value -= factors[
                        factorBase + column * dofCount + row
                    ] * solution[column];
                }
                solution[row] = value / factors[
                    factorBase + row * dofCount + row
                ];
                outputResponses[
                    responseBase + contactValueBase + row * 3u + axis
                ] = solution[row];
                if (!isfinite(solution[row])) {
                    localFailure =
                        NUMI_TEMPORAL_CONE_ARTICULATED_NONFINITE_RESULT;
                }
            }
            float maximumRightHandSide = 0.0f;
            float maximumSolution = 0.0f;
            for (uint row = 0u; row < dofCount; ++row) {
                float value = 0.0f;
                for (uint column = row; column < dofCount; ++column) {
                    value += factors[
                        factorBase + column * dofCount + row
                    ] * solution[column];
                }
                intermediate[row] = value;
                maximumRightHandSide = max(
                    maximumRightHandSide,
                    abs(outputJacobians[
                        jacobianBase + contactValueBase +
                            axis * dofCount + row
                    ])
                );
                maximumSolution = max(maximumSolution, abs(solution[row]));
            }
            float maximumResidual = 0.0f;
            for (uint row = 0u; row < dofCount; ++row) {
                float action = 0.0f;
                for (uint column = 0u; column <= row; ++column) {
                    action += factors[
                        factorBase + row * dofCount + column
                    ] * intermediate[column];
                }
                maximumResidual = max(
                    maximumResidual,
                    abs(action - outputJacobians[
                        jacobianBase + contactValueBase +
                            axis * dofCount + row
                    ])
                );
            }
            const float denominator = maximumRightHandSide +
                maximumFactor * maximumFactor * float(dofCount) *
                    float(dofCount) * maximumSolution + 1.0e-30f;
            const float backwardError = maximumResidual / denominator;
            laneMaximumBackwardError = max(
                laneMaximumBackwardError,
                backwardError
            );
            if (!isfinite(backwardError) || backwardError >
                    MR_ARTICULATED_OPERATOR_MAX_RELATIVE_RESIDUAL) {
                localFailure = isfinite(backwardError)
                    ? NUMI_TEMPORAL_CONE_ARTICULATED_ACCURACY_FAILED
                    : NUMI_TEMPORAL_CONE_ARTICULATED_NONFINITE_RESULT;
            }
        }
    }

    failure = simd_max(localFailure);
    const float maximumBackwardError = simd_max(
        laneMaximumBackwardError
    );
    if (failure != NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS) {
        if (lane == 0u) {
            NumiTemporalConeArticulatedStatus status = {};
            status.control = uint4(failure, dofCount, contactCount, 0u);
            status.conditioning = float4(
                minimumPivot,
                maximumPivot,
                conditionInfinity,
                operatorStatus.diagnostics.z
            );
            status.diagnostics = float4(
                maximumFrameError,
                maximumBackwardError,
                0.0f,
                0.0f
            );
            outputStatuses[problem] = status;
        }
        return;
    }

    if (activeContact) {
        const uint contactValueBase = lane * valuesPerContact;
        NumiTemporalConeAssemblyTerm term = {};
        term.control = uint4(
            header.control.w,
            dofCount,
            jacobianBase + contactValueBase,
            responseBase + contactValueBase
        );
        outputTerms[termBase + lane] = term;
        float3 rawVelocity = float3(0.0f);
        for (uint axis = 0u; axis < 3u; ++axis) {
            for (uint dof = 0u; dof < dofCount; ++dof) {
                rawVelocity[axis] = fma(
                    outputJacobians[
                        jacobianBase + contactValueBase +
                            axis * dofCount + dof
                    ],
                    inputVelocities[velocityBase + dof],
                    rawVelocity[axis]
                );
            }
        }
        const float timestep = law.stabilization.w;
        const float3 contactDenominator =
            law.dampingAndImpactThreshold.xyz +
            timestep * law.stiffnessAndRestitution.xyz;
        const float3 gamma = 1.0f / (timestep * contactDenominator);
        const float penetration = min(
            law.stabilization.x + law.stabilization.y,
            0.0f
        );
        const float uncappedRecovery =
            -law.stiffnessAndRestitution.x * penetration /
            contactDenominator.x;
        const float recoveryTarget = law.stabilization.z > 0.0f
            ? min(law.stabilization.z, uncappedRecovery)
            : uncappedRecovery;
        const float restitutionTarget =
            rawVelocity.x < -law.dampingAndImpactThreshold.w
            ? -law.stiffnessAndRestitution.w * rawVelocity.x
            : 0.0f;
        const float normalTarget = max(recoveryTarget, restitutionTarget);
        const float3 freeVelocity = rawVelocity + contact.bias.xyz -
            float3(normalTarget, 0.0f, 0.0f);
        if (!finite3(freeVelocity) || !finite3(gamma) || any(gamma <= 0.0f)) {
            localFailure = NUMI_TEMPORAL_CONE_ARTICULATED_NONFINITE_RESULT;
        } else {
            NumiTemporalConeIslandContact solverContact = {};
            solverContact.freeVelocityAndFrictionU = float4(
                freeVelocity,
                contact.normalAndFrictionU.w
            );
            solverContact.warmImpulseAndFrictionV = float4(
                contact.warmImpulse.xyz,
                contact.tangentUAndFrictionV.w
            );
            solverContact.limits.x =
                contact.tangentVAndMaximumNormal.w;
            outputContacts[solverContactBase + lane] = solverContact;
            const uint contactRegularizationBase =
                regularizationBase + 9u * lane;
            for (uint element = 0u; element < 9u; ++element) {
                outputRegularization[contactRegularizationBase + element] =
                    element % 4u == 0u ? gamma[element / 4u] : 0.0f;
            }
        }
    }
    failure = simd_max(localFailure);
    if (failure == NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS && activeContact) {
        NumiTemporalConeAssemblyContactSpan span = {};
        span.ranges = uint4(termBase + lane, 1u, 0u, 0u);
        outputSpans[spanBase + lane] = span;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (failure != NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS && activeContact) {
        outputSpans[spanBase + lane] = {};
    }
    if (lane == 0u) {
        NumiTemporalConeArticulatedStatus status = {};
        status.control = uint4(
            failure,
            dofCount,
            contactCount,
            failure == NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS
                ? contactCount
                : 0u
        );
        status.conditioning = float4(
            minimumPivot,
            maximumPivot,
            conditionInfinity,
            operatorStatus.diagnostics.z
        );
        status.diagnostics = float4(
            maximumFrameError,
            maximumBackwardError,
            0.0f,
            0.0f
        );
        outputStatuses[problem] = status;
    }
}

// Applies solved contact impulses in generalized coordinates. One DoF lane
// scans contacts in canonical order; any upstream or nonfinite failure
// republishes the exact input velocity vector for the whole articulation.
kernel void numi_temporal_cone_articulated_publish(
    device const NumiTemporalConeArticulatedHeader* headers [[buffer(0)]],
    device const float* inputVelocities [[buffer(1)]],
    device const float* responseValues [[buffer(2)]],
    device const float4* impulses [[buffer(3)]],
    device const NumiTemporalConeArticulatedStatus* responseStatuses
        [[buffer(4)]],
    device const NumiTemporalConeIslandStatus* solverStatuses [[buffer(5)]],
    device float* outputVelocities [[buffer(6)]],
    device NumiTemporalConeArticulatedStatus* outputStatuses [[buffer(7)]],
    constant uint& problemCount [[buffer(8)]],
    // x input velocities, y response floats, z impulses, w output velocities.
    constant uint4& capacities [[buffer(9)]],
    const uint problem [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    if (problem >= problemCount) {
        return;
    }
    const NumiTemporalConeArticulatedHeader header = headers[problem];
    const uint dofCount = header.control.y;
    const uint contactCount = header.control.z;
    const uint inputBase = header.inputRanges.z;
    const uint responseBase = header.responseRanges.w;
    const uint impulseBase = header.solverRanges.x;
    const uint outputBase = header.solverRanges.y;
    const uint valuesPerContact = 3u * dofCount;
    const bool active = lane < dofCount;
    uint localFailure = NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS;
    if (header.control.x != NUMI_TEMPORAL_CONE_ARTICULATED_ABI_VERSION) {
        localFailure = NUMI_TEMPORAL_CONE_ARTICULATED_INVALID_ABI;
    } else if (dofCount == 0u ||
        dofCount > NUMI_TEMPORAL_CONE_ARTICULATED_MAX_DOF ||
        contactCount == 0u ||
        contactCount > NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS ||
        inputBase > capacities.x || dofCount > capacities.x - inputBase ||
        responseBase > capacities.y ||
        contactCount * valuesPerContact > capacities.y - responseBase ||
        impulseBase > capacities.z ||
        contactCount > capacities.z - impulseBase ||
        outputBase > capacities.w || dofCount > capacities.w - outputBase) {
        localFailure = NUMI_TEMPORAL_CONE_ARTICULATED_INVALID_INPUT;
    } else if (responseStatuses[problem].control.x !=
            NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS ||
        solverStatuses[problem].control.x !=
            NUMI_TEMPORAL_CONE_ISLAND_SUCCESS ||
        solverStatuses[problem].control.z != 1u ||
        solverStatuses[problem].control.w != contactCount) {
        localFailure = NUMI_TEMPORAL_CONE_ARTICULATED_UPSTREAM_FAILURE;
    }
    uint failure = simd_max(localFailure);
    float laneDelta = 0.0f;
    if (active && inputBase + lane < capacities.x &&
        outputBase + lane < capacities.w) {
        const float input = inputVelocities[inputBase + lane];
        float output = input;
        if (failure == NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS) {
            float delta = 0.0f;
            for (uint contact = 0u; contact < contactCount; ++contact) {
                const float3 lambda = impulses[impulseBase + contact].xyz;
                const uint valueBase = responseBase +
                    contact * valuesPerContact + lane * 3u;
                delta = fma(
                    responseValues[valueBase + 0u], lambda.x, delta
                );
                delta = fma(
                    responseValues[valueBase + 1u], lambda.y, delta
                );
                delta = fma(
                    responseValues[valueBase + 2u], lambda.z, delta
                );
            }
            output += delta;
            laneDelta = abs(delta);
            if (!isfinite(input) || !isfinite(output)) {
                localFailure =
                    NUMI_TEMPORAL_CONE_ARTICULATED_NONFINITE_RESULT;
            }
        }
        outputVelocities[outputBase + lane] = output;
    }
    failure = simd_max(localFailure);
    if (failure != NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS && active &&
        inputBase + lane < capacities.x && outputBase + lane < capacities.w) {
        outputVelocities[outputBase + lane] = inputVelocities[inputBase + lane];
    }
    const float maximumDelta = simd_max(laneDelta);
    if (lane == 0u) {
        NumiTemporalConeArticulatedStatus status = responseStatuses[problem];
        status.control.x = failure;
        status.diagnostics.z =
            failure == NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS
            ? maximumDelta
            : 0.0f;
        outputStatuses[problem] = status;
    }
}

// Applies solved contact impulses to rigid velocities without atomics. Each
// body lane scans contacts in canonical order, making accumulation order and
// rollback deterministic. Any upstream failure republishes the exact input.
kernel void numi_temporal_cone_rigid_publish(
    device const NumiTemporalConeRigidHeader* headers [[buffer(0)]],
    device const NumiTemporalConeRigidBody* inputBodies [[buffer(1)]],
    device const NumiTemporalConeRigidContact* contacts [[buffer(2)]],
    device const float4* impulses [[buffer(3)]],
    device const NumiTemporalConeRigidStatus* responseStatuses [[buffer(4)]],
    device const NumiTemporalConeIslandStatus* solverStatuses [[buffer(5)]],
    device NumiTemporalConeRigidBody* outputBodies [[buffer(6)]],
    device NumiTemporalConeRigidStatus* outputStatuses [[buffer(7)]],
    constant uint& problemCount [[buffer(8)]],
    // x input bodies, y contacts, z impulses, w output bodies.
    constant uint4& capacities [[buffer(9)]],
    const uint problem [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    if (problem >= problemCount) {
        return;
    }
    const NumiTemporalConeRigidHeader header = headers[problem];
    const uint bodyCount = header.control.y;
    const uint contactCount = header.control.z;
    const uint bodyBase = header.inputRanges.x;
    const uint contactBase = header.inputRanges.y;
    const uint impulseBase = header.solverRanges.x;
    const uint outputBodyBase = header.solverRanges.y;
    const bool active = lane < bodyCount;
    uint localFailure = NUMI_TEMPORAL_CONE_RIGID_SUCCESS;
    if (header.control.x != NUMI_TEMPORAL_CONE_RIGID_ABI_VERSION) {
        localFailure = NUMI_TEMPORAL_CONE_RIGID_INVALID_ABI;
    } else if (bodyCount == 0u ||
        bodyCount > NUMI_TEMPORAL_CONE_RIGID_MAX_BODIES ||
        contactCount == 0u ||
        contactCount > NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS ||
        bodyBase > capacities.x || bodyCount > capacities.x - bodyBase ||
        contactBase > capacities.y || contactCount > capacities.y - contactBase ||
        impulseBase > capacities.z || contactCount > capacities.z - impulseBase ||
        outputBodyBase > capacities.w || bodyCount > capacities.w - outputBodyBase) {
        localFailure = NUMI_TEMPORAL_CONE_RIGID_INVALID_INPUT;
    } else if (responseStatuses[problem].control.x !=
            NUMI_TEMPORAL_CONE_RIGID_SUCCESS ||
        solverStatuses[problem].control.x !=
            NUMI_TEMPORAL_CONE_ISLAND_SUCCESS ||
        solverStatuses[problem].control.z != 1u ||
        solverStatuses[problem].control.w != contactCount) {
        localFailure = NUMI_TEMPORAL_CONE_RIGID_UPSTREAM_FAILURE;
    }
    const uint failure = simd_max(localFailure);
    float laneMaximumDelta = 0.0f;
    if (active && bodyBase + lane < capacities.x &&
        outputBodyBase + lane < capacities.w) {
        const NumiTemporalConeRigidBody input = inputBodies[bodyBase + lane];
        NumiTemporalConeRigidBody output = input;
        if (failure == NUMI_TEMPORAL_CONE_RIGID_SUCCESS) {
            float3 linearDelta = float3(0.0f);
            float3 angularDelta = float3(0.0f);
            for (uint contactIndex = 0u;
                 contactIndex < contactCount;
                 ++contactIndex) {
                const NumiTemporalConeRigidContact contact = contacts[
                    contactBase + contactIndex
                ];
                float sign = 0.0f;
                float3 offset = float3(0.0f);
                if (contact.bodies.x == lane) {
                    sign = -1.0f;
                    offset = contact.offsetA.xyz;
                } else if (contact.bodies.y == lane) {
                    sign = 1.0f;
                    offset = contact.offsetB.xyz;
                }
                if (sign != 0.0f) {
                    const float3 lambda = impulses[
                        impulseBase + contactIndex
                    ].xyz;
                    const float3 worldImpulse = sign * (
                        contact.normalAndFrictionU.xyz * lambda.x +
                        contact.tangentUAndFrictionV.xyz * lambda.y +
                        contact.tangentVAndMaximumNormal.xyz * lambda.z
                    );
                    linearDelta +=
                        input.linearVelocityAndInverseMass.w * worldImpulse;
                    angularDelta += temporalConeRigidInertiaMultiply(
                        input,
                        cross(offset, worldImpulse)
                    );
                }
            }
            output.linearVelocityAndInverseMass.xyz += linearDelta;
            output.angularVelocity.xyz += angularDelta;
            if (!finite4(output.linearVelocityAndInverseMass) ||
                !finite4(output.angularVelocity)) {
                output = input;
                localFailure = NUMI_TEMPORAL_CONE_RIGID_NONFINITE_RESULT;
            } else {
                laneMaximumDelta = max(length(linearDelta), length(angularDelta));
            }
        }
        outputBodies[outputBodyBase + lane] = output;
    }
    const uint outputFailure = simd_max(localFailure);
    if (outputFailure == NUMI_TEMPORAL_CONE_RIGID_NONFINITE_RESULT && active) {
        outputBodies[outputBodyBase + lane] = inputBodies[bodyBase + lane];
    }
    const float maximumDelta = simd_max(laneMaximumDelta);
    if (lane == 0u) {
        NumiTemporalConeRigidStatus status = responseStatuses[problem];
        status.control.x = outputFailure;
        status.diagnostics.w = outputFailure == NUMI_TEMPORAL_CONE_RIGID_SUCCESS
            ? maximumDelta
            : 0.0f;
        outputStatuses[problem] = status;
    }
}

// Advances post-contact rigid velocities with symplectic translation and an
// exponential-map quaternion update. A whole island publishes atomically at
// the status level: invalid pose/velocity data or upstream failure preserves
// every input pose exactly.
kernel void numi_temporal_cone_rigid_integrate(
    device const NumiTemporalConeIntegrationHeader* headers [[buffer(0)]],
    device const NumiTemporalConeRigidPose* inputPoses [[buffer(1)]],
    device const NumiTemporalConeRigidBody* velocities [[buffer(2)]],
    device const NumiTemporalConeRigidStatus* publishStatuses [[buffer(3)]],
    device NumiTemporalConeRigidPose* outputPoses [[buffer(4)]],
    device NumiTemporalConeIntegrationStatus* outputStatuses [[buffer(5)]],
    constant uint& problemCount [[buffer(6)]],
    // x input poses, y velocity bodies, z output poses, w reserved.
    constant uint4& capacities [[buffer(7)]],
    const uint problem [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    if (problem >= problemCount) {
        return;
    }
    const NumiTemporalConeIntegrationHeader header = headers[problem];
    const uint bodyCount = header.control.y;
    const uint inputBase = header.ranges.x;
    const uint velocityBase = header.ranges.y;
    const uint outputBase = header.ranges.z;
    const bool active = lane < bodyCount;
    uint localFailure = NUMI_TEMPORAL_CONE_INTEGRATION_SUCCESS;
    if (header.control.x != NUMI_TEMPORAL_CONE_INTEGRATION_ABI_VERSION) {
        localFailure = NUMI_TEMPORAL_CONE_INTEGRATION_INVALID_ABI;
    } else if (bodyCount == 0u ||
        bodyCount > NUMI_TEMPORAL_CONE_RIGID_MAX_BODIES ||
        inputBase > capacities.x || bodyCount > capacities.x - inputBase ||
        velocityBase > capacities.y ||
        bodyCount > capacities.y - velocityBase ||
        outputBase > capacities.z || bodyCount > capacities.z - outputBase ||
        !finite4(header.timestep) || !(header.timestep.x > 0.0f)) {
        localFailure = NUMI_TEMPORAL_CONE_INTEGRATION_INVALID_INPUT;
    } else if (publishStatuses[problem].control.x !=
            NUMI_TEMPORAL_CONE_RIGID_SUCCESS ||
        publishStatuses[problem].control.y != bodyCount) {
        localFailure = NUMI_TEMPORAL_CONE_INTEGRATION_UPSTREAM_FAILURE;
    }

    NumiTemporalConeRigidPose input = {};
    NumiTemporalConeRigidPose candidate = {};
    float laneInputNormError = 0.0f;
    if (active && inputBase + lane < capacities.x &&
        outputBase + lane < capacities.z) {
        input = inputPoses[inputBase + lane];
        candidate = input;
        outputPoses[outputBase + lane] = input;
    }
    if (active && localFailure == NUMI_TEMPORAL_CONE_INTEGRATION_SUCCESS) {
        const NumiTemporalConeRigidBody velocity = velocities[
            velocityBase + lane
        ];
        const float normSquared = dot(input.orientation, input.orientation);
        laneInputNormError = abs(normSquared - 1.0f);
        if (!finite4(input.position) || !finite4(input.orientation) ||
            !finite4(velocity.linearVelocityAndInverseMass) ||
            !finite4(velocity.angularVelocity) ||
            laneInputNormError > 2.0e-4f) {
            localFailure = NUMI_TEMPORAL_CONE_INTEGRATION_INVALID_INPUT;
        }
    }
    uint failure = simd_max(localFailure);
    const float maximumInputNormError = simd_max(laneInputNormError);
    float laneOutputNormError = 0.0f;
    float laneLinearDisplacement = 0.0f;
    float laneAngularStep = 0.0f;
    if (active && failure == NUMI_TEMPORAL_CONE_INTEGRATION_SUCCESS) {
        const NumiTemporalConeRigidBody velocity = velocities[
            velocityBase + lane
        ];
        const float timestep = header.timestep.x;
        candidate.position.xyz = fma(
            float3(timestep),
            velocity.linearVelocityAndInverseMass.xyz,
            input.position.xyz
        );
        float4 orientation = input.orientation;
        if (!integrateQuaternion(
                input.orientation,
                velocity.angularVelocity.xyz,
                timestep,
                orientation
            )) {
            localFailure = NUMI_TEMPORAL_CONE_INTEGRATION_NONFINITE_RESULT;
        }
        candidate.orientation = orientation;
        laneOutputNormError = abs(dot(orientation, orientation) - 1.0f);
        laneLinearDisplacement =
            timestep * length(velocity.linearVelocityAndInverseMass.xyz);
        laneAngularStep = timestep * length(velocity.angularVelocity.xyz);
        if (!finite4(candidate.position) ||
            !finite4(candidate.orientation) ||
            laneOutputNormError > 64.0f * kFloatEpsilon) {
            localFailure = NUMI_TEMPORAL_CONE_INTEGRATION_NONFINITE_RESULT;
        }
    }
    failure = simd_max(localFailure);
    if (active) {
        outputPoses[outputBase + lane] =
            failure == NUMI_TEMPORAL_CONE_INTEGRATION_SUCCESS
            ? candidate
            : input;
    }
    const float maximumOutputNormError = simd_max(laneOutputNormError);
    const float maximumLinearDisplacement = simd_max(laneLinearDisplacement);
    const float maximumAngularStep = simd_max(laneAngularStep);
    if (lane == 0u) {
        NumiTemporalConeIntegrationStatus status = {};
        status.control = uint4(failure, bodyCount, 0u, 0u);
        status.diagnostics = float4(
            maximumInputNormError,
            failure == NUMI_TEMPORAL_CONE_INTEGRATION_SUCCESS
                ? maximumOutputNormError
                : 0.0f,
            failure == NUMI_TEMPORAL_CONE_INTEGRATION_SUCCESS
                ? maximumLinearDisplacement
                : 0.0f,
            failure == NUMI_TEMPORAL_CONE_INTEGRATION_SUCCESS
                ? maximumAngularStep
                : 0.0f
        );
        outputStatuses[problem] = status;
    }
}
