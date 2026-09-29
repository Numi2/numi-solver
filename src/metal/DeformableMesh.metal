#include "DeformableTet.metal"
#include "numi/deformable_mesh.h"

// Shared nodes have one owning position, velocity and lumped mass. Every
// incidence entry is (element, local corner, owning node, reserved), sorted
// by element/corner. Gathering avoids nondeterministic floating-point atomics.
kernel void numi_deformable_mesh_evaluate(
    constant NumiDeformableMeshConfig& config [[buffer(0)]],
    device const NumiDeformableMeshElement* elements [[buffer(1)]],
    device const float4* positions [[buffer(2)]],
    device NumiDeformableTetOutput* outputs [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= config.counts.y) return;
    auto element = elements[index];
    NumiDeformableTetInput input{};
    if (any(element.nodes >= config.counts.x)) {
        NumiDeformableTetOutput invalid{};
        invalid.control.x = NUMI_DEFORMABLE_MESH_FAILURE_TOPOLOGY;
        outputs[index] = invalid; return;
    }
    for (uint n = 0; n < 4; ++n) input.positions[n] = positions[element.nodes[n]];
    for (uint r = 0; r < 3; ++r) input.inverseRestRows[r] = element.inverseRestRows[r];
    input.material = element.material; input.control = element.control;
    outputs[index] = evaluateTet(input);
}

kernel void numi_deformable_mesh_predict(
    constant NumiDeformableMeshConfig& config [[buffer(0)]],
    device const NumiDeformableMeshElement* elements [[buffer(1)]],
    device const float4* positions [[buffer(2)]],
    device const float4* velocityAndMass [[buffer(3)]],
    device const NumiDeformableTetOutput* forces [[buffer(4)]],
    device const uint* offsets [[buffer(5)]],
    device const uint4* incidence [[buffer(6)]],
    device float4* candidatePositions [[buffer(7)]],
    device float4* candidateVelocityAndMass [[buffer(8)]],
    device float4* contactImpulseAndKineticLoss [[buffer(9)]],
    uint node [[thread_position_in_grid]]
) {
    if (node >= config.counts.x) return;
    float4 v = velocityAndMass[node], p = positions[node];
    float4 contact = 0;
    bool valid = config.counts.w == NUMI_DEFORMABLE_MESH_ABI_VERSION &&
        all(isfinite(config.gravityAndTimestep)) && config.gravityAndTimestep.w > 0 &&
        all(isfinite(config.plane)) && (config.plane.y == 0 || config.plane.y == 1) &&
        all(isfinite(p)) && all(isfinite(v)) && v.w > 0;
    uint first = offsets[node], end = offsets[node + 1];
    valid = valid && first < end && end <= config.counts.z;
    float3 force = 0;
    if (valid) for (uint i = first; i < end; ++i) {
        uint4 entry = incidence[i];
        if (entry.x >= config.counts.y || entry.y >= 4 || entry.z != node) {
            valid = false; break;
        }
        if (elements[entry.x].nodes[entry.y] != node || forces[entry.x].control.x != 0) {
            valid = false; break;
        }
        force += forces[entry.x].forces[entry.y].xyz;
    }
    if (valid) {
        v.xyz += (force / v.w + config.gravityAndTimestep.xyz) * config.gravityAndTimestep.w;
        p.xyz = fma(v.xyz, float3(config.gravityAndTimestep.w), p.xyz);
        if (config.plane.y == 1 && p.z < config.plane.x) {
            p.z = config.plane.x;
            if (v.z < 0) {
                contact.z = -v.w * v.z;
                contact.w = 0.5f * v.w * v.z * v.z;
                v.z = 0;
            }
        }
    } else {
        // Zero mass is an invalid candidate marker; accepted state is untouched.
        v.w = 0;
    }
    candidatePositions[node] = p;
    candidateVelocityAndMass[node] = v;
    contactImpulseAndKineticLoss[node] = contact;
}

kernel void numi_deformable_mesh_validate(
    constant NumiDeformableMeshConfig& config [[buffer(0)]],
    device const NumiDeformableMeshElement* elements [[buffer(1)]],
    device const float4* positions [[buffer(2)]],
    device const NumiDeformableTetOutput* previousOutputs [[buffer(4)]],
    device const uint* offsets [[buffer(5)]],
    device const uint4* incidence [[buffer(6)]],
    device const float4* candidatePositions [[buffer(7)]],
    device const float4* candidateVelocityAndMass [[buffer(8)]],
    device const float4* contactImpulseAndKineticLoss [[buffer(9)]],
    device NumiDeformableMeshStatus& status [[buffer(10)]],
    device const NumiDeformableTetOutput* candidateOutputs [[buffer(11)]],
    uint index [[thread_position_in_grid]]
) {
    if (index != 0) return;
    uint failure = config.counts.w == NUMI_DEFORMABLE_MESH_ABI_VERSION
        ? 0 : NUMI_DEFORMABLE_TET_FAILURE_ABI;
    if (config.counts.z != 4 * config.counts.y || offsets[0] != 0 ||
        offsets[config.counts.x] != config.counts.z)
        failure |= NUMI_DEFORMABLE_MESH_FAILURE_TOPOLOGY;
    float minJ = status.ledger.y;
    float3 netForce = 0;
    float impulse = 0, loss = 0;
    for (uint e = 0; e < config.counts.y; ++e) {
        failure |= previousOutputs[e].control.x | candidateOutputs[e].control.x;
        minJ = min(minJ, candidateOutputs[e].energyAndVolumeRatio.y);
        for (uint n = 0; n < 4; ++n) netForce += previousOutputs[e].forces[n].xyz;
    }
    for (uint n = 0; n < config.counts.x; ++n) {
        uint first = offsets[n], end = offsets[n + 1];
        if (first >= end || end > config.counts.z) {
            failure |= NUMI_DEFORMABLE_MESH_FAILURE_TOPOLOGY;
        } else for (uint i = first; i < end; ++i) {
            uint4 entry = incidence[i];
            if (entry.x >= config.counts.y || entry.y >= 4 || entry.z != n) {
                failure |= NUMI_DEFORMABLE_MESH_FAILURE_TOPOLOGY; continue;
            }
            if (elements[entry.x].nodes[entry.y] != n ||
                (i > first && 4 * entry.x + entry.y <=
                    4 * incidence[i - 1].x + incidence[i - 1].y))
                failure |= NUMI_DEFORMABLE_MESH_FAILURE_TOPOLOGY;
        }
        float4 p = candidatePositions[n], v = candidateVelocityAndMass[n];
        float4 contact = contactImpulseAndKineticLoss[n];
        if (!all(isfinite(p)) || !all(isfinite(v)) || !(v.w > 0) ||
            !all(isfinite(contact)) || contact.z < 0 || contact.w < 0 ||
            (config.plane.y == 1 && (p.z < config.plane.x || positions[n].z < config.plane.x)))
            failure |= NUMI_DEFORMABLE_MESH_FAILURE_STATE;
        impulse += contact.z; loss += contact.w;
    }
    if (!all(isfinite(netForce)) || !isfinite(impulse) || !isfinite(loss))
        failure |= NUMI_DEFORMABLE_MESH_FAILURE_STATE;
    status.control.x = failure;
    status.control.y |= failure;
    if (failure == 0) {
        ++status.control.w;
        status.ledger.x += impulse;
        status.ledger.y = minJ;
        status.ledger.z = max(status.ledger.z, length(netForce));
        status.ledger.w += loss;
    } else ++status.control.z;
}

kernel void numi_deformable_mesh_commit(
    constant NumiDeformableMeshConfig& config [[buffer(0)]],
    device float4* positions [[buffer(2)]],
    device float4* velocityAndMass [[buffer(3)]],
    device const float4* candidatePositions [[buffer(7)]],
    device const float4* candidateVelocityAndMass [[buffer(8)]],
    device const NumiDeformableMeshStatus& status [[buffer(10)]],
    uint node [[thread_position_in_grid]]
) {
    if (node < config.counts.x && status.control.x == 0) {
        positions[node] = candidatePositions[node];
        velocityAndMass[node] = candidateVelocityAndMass[node];
    }
}
