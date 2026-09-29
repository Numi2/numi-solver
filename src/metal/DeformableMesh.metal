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
    device float4* freePositions [[buffer(12)]],
    device float4* freeVelocityAndMass [[buffer(13)]],
    uint node [[thread_position_in_grid]]
) {
    if (node >= config.counts.x) return;
    float4 v = velocityAndMass[node], p = positions[node];
    float4 contact = 0;
    bool valid = config.counts.w == NUMI_DEFORMABLE_MESH_ABI_VERSION &&
        all(isfinite(config.gravityAndTimestep)) && config.gravityAndTimestep.w > 0 &&
        all(isfinite(config.plane)) && (config.plane.y == 0 || config.plane.y == 1) &&
        config.integration.x <= NUMI_DEFORMABLE_MESH_INTEGRATOR_SUPPORT_VERLET &&
        all(config.integration.yzw == 0) &&
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
        float3 acceleration = force / v.w + config.gravityAndTimestep.xyz;
        float kick = config.gravityAndTimestep.w;
        if (config.integration.x == NUMI_DEFORMABLE_MESH_INTEGRATOR_SUPPORT_VERLET) {
            kick *= 0.5f;
            // A resting contact supplies its reaction before velocity/position
            // integration. It does zero work and removes no incoming energy.
            if (config.plane.y == 1 && p.z == config.plane.x && v.z == 0 && acceleration.z < 0) {
                contact.z = -v.w * acceleration.z * kick;
                acceleration.z = 0;
            }
        }
        v.xyz += acceleration * kick;
        p.xyz = fma(v.xyz, float3(config.gravityAndTimestep.w), p.xyz);
        // Retain the exact pre-projection prediction. Its elastic energy must
        // be evaluated separately; kinetic removal alone omits position work.
        freePositions[node] = p;
        freeVelocityAndMass[node] = v;
        if (config.plane.y == 1 && p.z < config.plane.x) {
            p.z = config.plane.x;
            if (v.z < 0) {
                contact.z += -v.w * v.z;
                contact.w = 0.5f * v.w * v.z * v.z;
                v.z = 0;
            }
        }
    } else {
        // Zero mass is an invalid candidate marker; accepted state is untouched.
        v.w = 0;
        freePositions[node] = p;
        freeVelocityAndMass[node] = v;
    }
    candidatePositions[node] = p;
    candidateVelocityAndMass[node] = v;
    contactImpulseAndKineticLoss[node] = contact;
}

// The second kick uses each branch's own evaluated elastic forces. Candidate
// and pre-impact predictions share the old-position half kick, but a newly
// contacted plane changes the second force evaluation. Account for that
// kinetic difference instead of treating it as normal impact dissipation.
kernel void numi_deformable_mesh_finish_verlet(
    constant NumiDeformableMeshConfig& config [[buffer(0)]],
    device const NumiDeformableMeshElement* elements [[buffer(1)]],
    device const uint* offsets [[buffer(5)]],
    device const uint4* incidence [[buffer(6)]],
    device const float4* candidatePositions [[buffer(7)]],
    device float4* candidateVelocityAndMass [[buffer(8)]],
    device float4* contactImpulseAndKineticLoss [[buffer(9)]],
    device const NumiDeformableTetOutput* candidateOutputs [[buffer(11)]],
    device const float4* freePositions [[buffer(12)]],
    device float4* freeVelocityAndMass [[buffer(13)]],
    device const NumiDeformableTetOutput* freeOutputs [[buffer(14)]],
    uint node [[thread_position_in_grid]]
) {
    if (node >= config.counts.x || config.integration.x != NUMI_DEFORMABLE_MESH_INTEGRATOR_SUPPORT_VERLET) return;
    float4 v = candidateVelocityAndMass[node], fv = freeVelocityAndMass[node];
    uint first = offsets[node], end = offsets[node + 1];
    bool valid = v.w > 0 && fv.w > 0 && first < end && end <= config.counts.z;
    float3 force = 0, freeForce = 0;
    if (valid) for (uint i = first; i < end; ++i) {
        uint4 entry = incidence[i];
        if (entry.x >= config.counts.y || entry.y >= 4 || entry.z != node) { valid = false; break; }
        if (elements[entry.x].nodes[entry.y] != node || candidateOutputs[entry.x].control.x != 0 ||
            freeOutputs[entry.x].control.x != 0) { valid = false; break; }
        force += candidateOutputs[entry.x].forces[entry.y].xyz;
        freeForce += freeOutputs[entry.x].forces[entry.y].xyz;
    }
    if (valid) {
        float kick = 0.5f * config.gravityAndTimestep.w;
        float3 acceleration = force / v.w + config.gravityAndTimestep.xyz;
        float3 freeAcceleration = freeForce / fv.w + config.gravityAndTimestep.xyz;
        float4 contact = contactImpulseAndKineticLoss[node];
        if (config.plane.y == 1 && candidatePositions[node].z == config.plane.x && v.z == 0 && acceleration.z < 0) {
            contact.z += -v.w * acceleration.z * kick;
            acceleration.z = 0;
        }
        if (config.plane.y == 1 && freePositions[node].z == config.plane.x && fv.z == 0 && freeAcceleration.z < 0)
            freeAcceleration.z = 0;
        v.xyz += acceleration * kick;
        fv.xyz += freeAcceleration * kick;
        if (config.plane.y == 1 && candidatePositions[node].z == config.plane.x && v.z < 0) {
            contact.z += -v.w * v.z;
            contact.w += 0.5f * v.w * v.z * v.z;
            v.z = 0;
        }
        contactImpulseAndKineticLoss[node] = contact;
    } else { v.w = 0; fv.w = 0; }
    candidateVelocityAndMass[node] = v;
    freeVelocityAndMass[node] = fv;
}

kernel void numi_deformable_mesh_validate(
    constant NumiDeformableMeshConfig& config [[buffer(0)]],
    device const NumiDeformableMeshElement* elements [[buffer(1)]],
    device const float4* positions [[buffer(2)]],
    device const float4* velocityAndMass [[buffer(3)]],
    device const NumiDeformableTetOutput* previousOutputs [[buffer(4)]],
    device const uint* offsets [[buffer(5)]],
    device const uint4* incidence [[buffer(6)]],
    device const float4* candidatePositions [[buffer(7)]],
    device const float4* candidateVelocityAndMass [[buffer(8)]],
    device const float4* contactImpulseAndKineticLoss [[buffer(9)]],
    device NumiDeformableMeshStatus& status [[buffer(10)]],
    device const NumiDeformableTetOutput* candidateOutputs [[buffer(11)]],
    device const float4* freePositions [[buffer(12)]],
    device const float4* freeVelocityAndMass [[buffer(13)]],
    device const NumiDeformableTetOutput* freeOutputs [[buffer(14)]],
    uint index [[thread_position_in_grid]]
) {
    if (index != 0) return;
    uint failure = config.counts.w == NUMI_DEFORMABLE_MESH_ABI_VERSION
        ? 0 : NUMI_DEFORMABLE_TET_FAILURE_ABI;
    if (config.integration.x > NUMI_DEFORMABLE_MESH_INTEGRATOR_SUPPORT_VERLET || any(config.integration.yzw != 0))
        failure |= NUMI_DEFORMABLE_MESH_FAILURE_STATE;
    if (config.counts.z != 4 * config.counts.y || offsets[0] != 0 ||
        offsets[config.counts.x] != config.counts.z)
        failure |= NUMI_DEFORMABLE_MESH_FAILURE_TOPOLOGY;
    float minJ = status.ledger.y;
    float3 netForce = 0;
    float impulse = 0, loss = 0;
    float elasticProjection = 0, gravityProjection = 0;
    float freeEnergyChange = 0, elasticForceWork = 0;
    float kineticProjection = 0;
    for (uint e = 0; e < config.counts.y; ++e) {
        failure |= previousOutputs[e].control.x | candidateOutputs[e].control.x | freeOutputs[e].control.x;
        elasticProjection += candidateOutputs[e].energyAndVolumeRatio.x - freeOutputs[e].energyAndVolumeRatio.x;
        freeEnergyChange += freeOutputs[e].energyAndVolumeRatio.x - previousOutputs[e].energyAndVolumeRatio.x;
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
            else elasticForceWork += dot(previousOutputs[entry.x].forces[entry.y].xyz,
                freePositions[n].xyz - positions[n].xyz);
        }
        float4 p = candidatePositions[n], v = candidateVelocityAndMass[n];
        float4 contact = contactImpulseAndKineticLoss[n];
        float4 fp = freePositions[n], fv = freeVelocityAndMass[n], oldV = velocityAndMass[n];
        if (!all(isfinite(p)) || !all(isfinite(v)) || !(v.w > 0) ||
            !all(isfinite(contact)) || !all(isfinite(fp)) || !all(isfinite(fv)) ||
            !all(isfinite(oldV)) || contact.z < 0 || contact.w < 0 ||
            (config.plane.y == 1 && (p.z < config.plane.x || positions[n].z < config.plane.x)))
            failure |= NUMI_DEFORMABLE_MESH_FAILURE_STATE;
        impulse += contact.z; loss += contact.w;
        gravityProjection -= v.w * dot(config.gravityAndTimestep.xyz, p.xyz - fp.xyz);
        kineticProjection += 0.5f * v.w * dot(v.xyz - fv.xyz, v.xyz + fv.xyz);
        freeEnergyChange += 0.5f * fv.w * dot(fv.xyz - oldV.xyz, fv.xyz + oldV.xyz)
            - fv.w * dot(config.gravityAndTimestep.xyz, fp.xyz - positions[n].xyz);
    }
    // Retain the exact legacy decomposition for Euler; Verlet's two force
    // branches require a separately measured projected kinetic change.
    if (config.integration.x == NUMI_DEFORMABLE_MESH_INTEGRATOR_EULER) kineticProjection = -loss;
    float forceResponse = kineticProjection + loss;
    float4 nextKinetic = status.contactKinetic + float4(kineticProjection, forceResponse, abs(forceResponse), 0);
    nextKinetic.w = max(status.contactKinetic.w, abs(forceResponse));
    float contactEnergyChange = elasticProjection + gravityProjection + kineticProjection;
    float4 nextContact = status.contactWork + float4(elasticProjection, gravityProjection, contactEnergyChange, 0);
    nextContact.w = max(status.contactWork.w, contactEnergyChange);
    float4 nextIntegration = status.integration + float4(freeEnergyChange, abs(freeEnergyChange), 0, elasticForceWork);
    nextIntegration.z = max(status.integration.z, abs(freeEnergyChange));
    float positionWorkBound = abs(elasticProjection) + abs(gravityProjection);
    float4 nextBounds = status.projectionBounds;
    nextBounds.x += positionWorkBound;
    nextBounds.y = max(nextBounds.y, positionWorkBound);
    nextBounds.z = max(nextBounds.z, abs(contactEnergyChange));
    nextBounds.w += max(0.0f, contactEnergyChange);
    if (!all(isfinite(netForce)) || !isfinite(impulse) || !isfinite(loss) ||
        !all(isfinite(nextContact)) || !all(isfinite(nextIntegration)) || !all(isfinite(nextBounds)) ||
        !all(isfinite(nextKinetic)))
        failure |= NUMI_DEFORMABLE_MESH_FAILURE_STATE;
    status.control.x = failure;
    status.control.y |= failure;
    if (failure == 0) {
        ++status.control.w;
        status.ledger.x += impulse;
        status.ledger.y = minJ;
        status.ledger.z = max(status.ledger.z, length(netForce));
        status.ledger.w += loss;
        status.contactWork = nextContact;
        status.integration = nextIntegration;
        status.projectionBounds = nextBounds;
        status.contactKinetic = nextKinetic;
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
