#include <metal_stdlib>
#include "numi/deformable_tet.h"
using namespace metal;

inline float3x3 tetRows(thread const mr_float4* rows) {
    return transpose(float3x3(rows[0].xyz, rows[1].xyz, rows[2].xyz));
}

inline NumiDeformableTetOutput evaluateTet(NumiDeformableTetInput input) {
    NumiDeformableTetOutput output{};
    uint failure = input.control.x == NUMI_DEFORMABLE_TET_ABI_VERSION
        ? 0u : NUMI_DEFORMABLE_TET_FAILURE_ABI;
    for (uint node = 0; node < 4; ++node)
        if (!all(isfinite(input.positions[node]))) failure |= NUMI_DEFORMABLE_TET_FAILURE_NONFINITE;
    for (uint row = 0; row < 3; ++row)
        if (!all(isfinite(input.inverseRestRows[row]))) failure |= NUMI_DEFORMABLE_TET_FAILURE_NONFINITE;
    if (!all(isfinite(input.material))) failure |= NUMI_DEFORMABLE_TET_FAILURE_NONFINITE;
    float mu = input.material.x, lambda = input.material.y;
    if (!(mu > 0) || !(lambda >= 0)) failure |= NUMI_DEFORMABLE_TET_FAILURE_MATERIAL;
    float volume = input.inverseRestRows[0].w;
    float3x3 inverseRest = tetRows(input.inverseRestRows);
    float inverseDet = determinant(inverseRest);
    if (!(volume > 0) || !(inverseDet > 0) ||
        abs(6.0f * volume * inverseDet - 1.0f) > 2e-4f)
        failure |= NUMI_DEFORMABLE_TET_FAILURE_REST_GEOMETRY;
    if (failure != 0) {
        output.control.x = failure; return output;
    }
    float3x3 edges(input.positions[1].xyz - input.positions[0].xyz,
                  input.positions[2].xyz - input.positions[0].xyz,
                  input.positions[3].xyz - input.positions[0].xyz);
    float3x3 F = edges * inverseRest;
    float J = determinant(F);
    if (!isfinite(J) || !(J > 1e-6f)) {
        output.control.x = NUMI_DEFORMABLE_TET_FAILURE_INVERSION;
        return output;
    }
    float3x3 cofactor(cross(F[1], F[2]), cross(F[2], F[0]), cross(F[0], F[1]));
    float logJ = log(J);
    float3x3 P = mu * F + ((lambda * logJ - mu) / J) * cofactor;
    float invariant = dot(F[0], F[0]) + dot(F[1], F[1]) + dot(F[2], F[2]);
    float energy = volume * (0.5f * mu * (invariant - 3.0f) - mu * logJ +
                             0.5f * lambda * logJ * logJ);
    float3x3 H = -volume * P * transpose(inverseRest);
    output.forces[0] = float4(-(H[0] + H[1] + H[2]), 0);
    for (uint node = 1; node < 4; ++node) output.forces[node] = float4(H[node - 1], 0);
    float3x3 rows = transpose(P);
    for (uint row = 0; row < 3; ++row) output.firstPiolaRows[row] = float4(rows[row], 0);
    output.energyAndVolumeRatio = float4(energy, J, 0, 0);
    if (!isfinite(energy) || !all(isfinite(P[0])) || !all(isfinite(P[1])) ||
        !all(isfinite(P[2])) || !all(isfinite(H[0])) || !all(isfinite(H[1])) || !all(isfinite(H[2])) ||
        !all(isfinite(output.forces[0]))) {
        output = {}; output.control.x = NUMI_DEFORMABLE_TET_FAILURE_NONFINITE;
    }
    return output;
}

kernel void numi_deformable_tet_evaluate(
    device const NumiDeformableTetInput* inputs [[buffer(0)]],
    device NumiDeformableTetOutput* outputs [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index < count) outputs[index] = evaluateTet(inputs[index]);
}

kernel void numi_deformable_tet_advance(
    device NumiDeformableTetInput* states [[buffer(0)]],
    device NumiDeformableTetOutput* outputs [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    // Four nodal velocities per element: xyz m/s, w positive mass in kg.
    device float4* velocityAndMass [[buffer(3)]],
    constant float4& gravityAndTimestep [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    uint accumulatedFailure = outputs[index].control.y;
    NumiDeformableTetInput state = states[index];
    auto force = evaluateTet(state);
    if (force.control.x != 0) {
        force.control.y = accumulatedFailure | force.control.x;
        outputs[index] = force; return;
    }
    if (!all(isfinite(gravityAndTimestep)) || !(gravityAndTimestep.w > 0)) {
        force = {};
        force.control.x = NUMI_DEFORMABLE_TET_FAILURE_NONFINITE;
        force.control.y = accumulatedFailure | force.control.x;
        outputs[index] = force; return;
    }
    float4 velocities[4];
    for (uint node = 0; node < 4; ++node) {
        float4 v = velocityAndMass[4 * index + node];
        if (!all(isfinite(v)) || !(v.w > 0)) {
            force = {};
            force.control.x = NUMI_DEFORMABLE_TET_FAILURE_MATERIAL;
            force.control.y = accumulatedFailure | force.control.x;
            outputs[index] = force; return;
        }
        v.xyz += (gravityAndTimestep.xyz + force.forces[node].xyz / v.w) * gravityAndTimestep.w;
        state.positions[node].xyz = fma(v.xyz, float3(gravityAndTimestep.w), state.positions[node].xyz);
        velocities[node] = v;
    }
    auto candidate = evaluateTet(state);
    if (candidate.control.x == 0) {
        states[index] = state;
        for (uint node = 0; node < 4; ++node)
            velocityAndMass[4 * index + node] = velocities[node];
    }
    // A rejected candidate leaves the complete accepted state untouched.
    candidate.control.y = accumulatedFailure | candidate.control.x;
    outputs[index] = candidate;
}
