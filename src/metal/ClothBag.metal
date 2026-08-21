#include <metal_stdlib>

#include "numi/cloth_bag_gpu.h"

using namespace metal;

namespace {

inline void recordFailure(
    device atomic_uint* failure,
    const uint flag
) {
    atomic_fetch_or_explicit(failure, flag, memory_order_relaxed);
}

inline bool validConfig(
    constant NumiClothBagGPUConfig& config,
    device atomic_uint* failure
) {
    if (config.control.x != NUMI_CLOTH_BAG_GPU_ABI_VERSION) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_ABI);
        return false;
    }
    if (!(config.gravityAndTimestep.w > 0.0f) ||
        !isfinite(config.gravityAndTimestep.w) ||
        !all(isfinite(config.gravityAndTimestep.xyz)) ||
        !all(isfinite(config.gripTargetAndActive.xyz))) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
        return false;
    }
    return true;
}

} // namespace

kernel void numi_cloth_bag_begin_substep(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device NumiClothBagGPUParticle* particles [[buffer(1)]],
    device NumiClothBagGPUDistance* distances [[buffer(2)]],
    device NumiClothBagGPUGrip* grips [[buffer(3)]],
    device atomic_uint* failure [[buffer(4)]],
    const uint index [[thread_position_in_grid]]
) {
    if (!validConfig(config, failure)) {
        return;
    }
    if (index < config.control.y) {
        NumiClothBagGPUParticle particle = particles[index];
        const float3 position = particle.positionAndInverseMass.xyz;
        const float inverseMass = particle.positionAndInverseMass.w;
        const float3 velocity = particle.velocity.xyz;
        if (!all(isfinite(position)) || !all(isfinite(velocity)) ||
            !isfinite(inverseMass) || inverseMass < 0.0f) {
            recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
            return;
        }
        particle.previousAndMass.xyz = position;
        if (inverseMass > 0.0f) {
            const float timestep = config.gravityAndTimestep.w;
            particle.velocity.xyz = fma(
                config.gravityAndTimestep.xyz,
                float3(timestep),
                velocity
            );
            particle.positionAndInverseMass.xyz = fma(
                particle.velocity.xyz,
                float3(timestep),
                position
            );
        }
        particles[index] = particle;
    }
    if (index < config.control.z) {
        distances[index].material.z = 0.0f;
    }
    if (index < config.control.w) {
        grips[index].lambda = float4(0.0f);
    }
}

kernel void numi_cloth_bag_solve_distance(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device NumiClothBagGPUParticle* particles [[buffer(1)]],
    device NumiClothBagGPUDistance* distances [[buffer(2)]],
    constant NumiClothBagGPUBatch& batch [[buffer(3)]],
    device atomic_uint* failure [[buffer(4)]],
    const uint localIndex [[thread_position_in_grid]]
) {
    if (!validConfig(config, failure) || localIndex >= batch.control.y) {
        return;
    }
    const uint constraintIndex = batch.control.x + localIndex;
    if (constraintIndex >= config.control.z) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
        return;
    }
    NumiClothBagGPUDistance constraint = distances[constraintIndex];
    const uint firstIndex = constraint.particlesAndColor.x;
    const uint secondIndex = constraint.particlesAndColor.y;
    if (firstIndex >= config.control.y || secondIndex >= config.control.y ||
        firstIndex == secondIndex) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
        return;
    }
    if (constraint.particlesAndColor.z != batch.control.z) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_BATCH);
        return;
    }
    NumiClothBagGPUParticle first = particles[firstIndex];
    NumiClothBagGPUParticle second = particles[secondIndex];
    const float3 difference =
        second.positionAndInverseMass.xyz -
        first.positionAndInverseMass.xyz;
    const float currentLength = length(difference);
    const float timestep = config.gravityAndTimestep.w;
    const float alpha = constraint.material.y / (timestep * timestep);
    const float denominator = first.positionAndInverseMass.w +
        second.positionAndInverseMass.w + alpha;
    if (!isfinite(currentLength) || !isfinite(alpha) ||
        !isfinite(denominator) || !isfinite(constraint.material.x) ||
        !isfinite(constraint.material.z)) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
        return;
    }
    if (!(currentLength > 1.0e-12f) || !(denominator > 0.0f)) {
        return;
    }
    const float value = currentLength - constraint.material.x;
    const float deltaLambda =
        (-value - alpha * constraint.material.z) / denominator;
    const float3 correction = difference * (deltaLambda / currentLength);
    constraint.material.z += deltaLambda;
    first.positionAndInverseMass.xyz -=
        correction * first.positionAndInverseMass.w;
    second.positionAndInverseMass.xyz +=
        correction * second.positionAndInverseMass.w;
    if (!all(isfinite(first.positionAndInverseMass.xyz)) ||
        !all(isfinite(second.positionAndInverseMass.xyz)) ||
        !isfinite(constraint.material.z)) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
        return;
    }
    particles[firstIndex] = first;
    particles[secondIndex] = second;
    distances[constraintIndex] = constraint;
}

kernel void numi_cloth_bag_solve_grip(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device NumiClothBagGPUParticle* particles [[buffer(1)]],
    device NumiClothBagGPUGrip* grips [[buffer(2)]],
    device atomic_uint* failure [[buffer(3)]],
    const uint index [[thread_position_in_grid]]
) {
    if (!validConfig(config, failure) || index >= config.control.w ||
        !(config.gripTargetAndActive.w > 0.0f)) {
        return;
    }
    NumiClothBagGPUGrip grip = grips[index];
    const uint particleIndex = grip.particle.x;
    if (particleIndex >= config.control.y) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
        return;
    }
    NumiClothBagGPUParticle particle = particles[particleIndex];
    const float timestep = config.gravityAndTimestep.w;
    const float alpha = grip.targetOffsetAndCompliance.w /
        (timestep * timestep);
    const float denominator = particle.positionAndInverseMass.w + alpha;
    if (!isfinite(alpha) || !isfinite(denominator) ||
        !all(isfinite(grip.targetOffsetAndCompliance)) ||
        !all(isfinite(grip.lambda))) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
        return;
    }
    if (!(denominator > 0.0f)) {
        return;
    }
    const float3 target = config.gripTargetAndActive.xyz +
        grip.targetOffsetAndCompliance.xyz;
    const float3 value = particle.positionAndInverseMass.xyz - target;
    const float3 deltaLambda =
        (-value - alpha * grip.lambda.xyz) / denominator;
    grip.lambda.xyz += deltaLambda;
    particle.positionAndInverseMass.xyz +=
        deltaLambda * particle.positionAndInverseMass.w;
    if (!all(isfinite(particle.positionAndInverseMass.xyz)) ||
        !all(isfinite(grip.lambda.xyz))) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
        return;
    }
    particles[particleIndex] = particle;
    grips[index] = grip;
}

kernel void numi_cloth_bag_limit_strain(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device NumiClothBagGPUParticle* particles [[buffer(1)]],
    device const NumiClothBagGPUDistance* distances [[buffer(2)]],
    constant NumiClothBagGPUBatch& batch [[buffer(3)]],
    device atomic_uint* failure [[buffer(4)]],
    const uint localIndex [[thread_position_in_grid]]
) {
    if (!validConfig(config, failure) || localIndex >= batch.control.y) {
        return;
    }
    const uint constraintIndex = batch.control.x + localIndex;
    if (constraintIndex >= config.control.z) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
        return;
    }
    const NumiClothBagGPUDistance constraint = distances[constraintIndex];
    const uint firstIndex = constraint.particlesAndColor.x;
    const uint secondIndex = constraint.particlesAndColor.y;
    if (firstIndex >= config.control.y || secondIndex >= config.control.y ||
        firstIndex == secondIndex) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
        return;
    }
    if (constraint.particlesAndColor.z != batch.control.z) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_BATCH);
        return;
    }
    NumiClothBagGPUParticle first = particles[firstIndex];
    NumiClothBagGPUParticle second = particles[secondIndex];
    const float3 difference =
        second.positionAndInverseMass.xyz -
        first.positionAndInverseMass.xyz;
    const float currentLength = length(difference);
    const float maximumLength = constraint.material.x *
        (1.0f + constraint.material.w);
    if (!isfinite(currentLength) || !isfinite(maximumLength)) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
        return;
    }
    if (!(currentLength > maximumLength) ||
        !(currentLength > 1.0e-12f)) {
        return;
    }
    const float denominator = first.positionAndInverseMass.w +
        second.positionAndInverseMass.w;
    if (!(denominator > 0.0f) || !isfinite(denominator)) {
        return;
    }
    const float correctionMagnitude = currentLength - maximumLength;
    const float3 direction = difference / currentLength;
    first.positionAndInverseMass.xyz += direction *
        (first.positionAndInverseMass.w * correctionMagnitude / denominator);
    second.positionAndInverseMass.xyz -= direction *
        (second.positionAndInverseMass.w * correctionMagnitude / denominator);
    particles[firstIndex] = first;
    particles[secondIndex] = second;
}

kernel void numi_cloth_bag_finalize_substep(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device NumiClothBagGPUParticle* particles [[buffer(1)]],
    device atomic_uint* failure [[buffer(2)]],
    const uint index [[thread_position_in_grid]]
) {
    if (!validConfig(config, failure) || index >= config.control.y) {
        return;
    }
    NumiClothBagGPUParticle particle = particles[index];
    const float inverseTimestep = 1.0f / config.gravityAndTimestep.w;
    particle.velocity.xyz =
        (particle.positionAndInverseMass.xyz - particle.previousAndMass.xyz) *
        inverseTimestep;
    if (!all(isfinite(particle.velocity.xyz))) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
        return;
    }
    particles[index] = particle;
}
