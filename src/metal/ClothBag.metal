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
        !all(isfinite(config.gripTargetAndActive.xyz)) ||
        !isfinite(config.clothMaterial.x) || config.clothMaterial.x < 0.0f) {
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
    device NumiClothBagGPUKnot* knots [[buffer(4)]],
    device NumiClothBagGPUBend* bends [[buffer(5)]],
    device atomic_uint* failure [[buffer(6)]],
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
    if (index < config.constraintCounts.x) {
        knots[index].material.z = 0.0f;
    }
    if (index < config.constraintCounts.y) {
        bends[index].material.w = 0.0f;
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

kernel void numi_cloth_bag_solve_knot(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device NumiClothBagGPUParticle* particles [[buffer(1)]],
    device NumiClothBagGPUKnot* knots [[buffer(2)]],
    constant NumiClothBagGPUBatch& batch [[buffer(3)]],
    device atomic_uint* failure [[buffer(4)]],
    const uint localIndex [[thread_position_in_grid]]
) {
    if (!validConfig(config, failure) || localIndex >= batch.control.y) {
        return;
    }
    const uint constraintIndex = batch.control.x + localIndex;
    if (constraintIndex >= config.constraintCounts.x) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
        return;
    }
    NumiClothBagGPUKnot constraint = knots[constraintIndex];
    if (constraint.control.x != batch.control.z) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_BATCH);
        return;
    }
    const uint4 indices = constraint.particles;
    if (any(indices >= uint4(config.control.y)) ||
        indices.x == indices.y || indices.x == indices.z ||
        indices.x == indices.w || indices.y == indices.z ||
        indices.y == indices.w || indices.z == indices.w) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
        return;
    }
    NumiClothBagGPUParticle warpFirst = particles[indices.x];
    NumiClothBagGPUParticle warpSecond = particles[indices.y];
    NumiClothBagGPUParticle weftFirst = particles[indices.z];
    NumiClothBagGPUParticle weftSecond = particles[indices.w];
    const float3 warpVector =
        warpSecond.positionAndInverseMass.xyz -
        warpFirst.positionAndInverseMass.xyz;
    const float3 weftVector =
        weftSecond.positionAndInverseMass.xyz -
        weftFirst.positionAndInverseMass.xyz;
    const float warpLength = length(warpVector);
    const float weftLength = length(weftVector);
    if (!isfinite(warpLength) || !isfinite(weftLength) ||
        !all(isfinite(constraint.material))) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
        return;
    }
    if (!(warpLength > 1.0e-12f) || !(weftLength > 1.0e-12f)) {
        return;
    }
    const float3 warp = warpVector / warpLength;
    const float3 weft = weftVector / weftLength;
    const float cosine = clamp(dot(warp, weft), -1.0f, 1.0f);
    const float value = cosine - constraint.material.x;
    const float3 warpGradient = (weft - warp * cosine) / warpLength;
    const float3 weftGradient = (warp - weft * cosine) / weftLength;
    const float3 gradients[4] = {
        -warpGradient, warpGradient, -weftGradient, weftGradient
    };
    const float inverseMasses[4] = {
        warpFirst.positionAndInverseMass.w,
        warpSecond.positionAndInverseMass.w,
        weftFirst.positionAndInverseMass.w,
        weftSecond.positionAndInverseMass.w,
    };
    const float timestep = config.gravityAndTimestep.w;
    const float alpha = constraint.material.y / (timestep * timestep);
    float denominator = alpha;
    for (uint participant = 0u; participant < 4u; ++participant) {
        denominator += inverseMasses[participant] *
            dot(gradients[participant], gradients[participant]);
    }
    if (!(denominator > 0.0f) || !isfinite(denominator)) {
        return;
    }
    const float deltaLambda =
        (-value - alpha * constraint.material.z) / denominator;
    constraint.material.z += deltaLambda;
    warpFirst.positionAndInverseMass.xyz +=
        gradients[0] * (inverseMasses[0] * deltaLambda);
    warpSecond.positionAndInverseMass.xyz +=
        gradients[1] * (inverseMasses[1] * deltaLambda);
    weftFirst.positionAndInverseMass.xyz +=
        gradients[2] * (inverseMasses[2] * deltaLambda);
    weftSecond.positionAndInverseMass.xyz +=
        gradients[3] * (inverseMasses[3] * deltaLambda);
    if (!all(isfinite(warpFirst.positionAndInverseMass.xyz)) ||
        !all(isfinite(warpSecond.positionAndInverseMass.xyz)) ||
        !all(isfinite(weftFirst.positionAndInverseMass.xyz)) ||
        !all(isfinite(weftSecond.positionAndInverseMass.xyz)) ||
        !isfinite(constraint.material.z)) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
        return;
    }
    particles[indices.x] = warpFirst;
    particles[indices.y] = warpSecond;
    particles[indices.z] = weftFirst;
    particles[indices.w] = weftSecond;
    knots[constraintIndex] = constraint;
}

kernel void numi_cloth_bag_solve_bend(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device NumiClothBagGPUParticle* particles [[buffer(1)]],
    device NumiClothBagGPUBend* bends [[buffer(2)]],
    constant NumiClothBagGPUBatch& batch [[buffer(3)]],
    device atomic_uint* failure [[buffer(4)]],
    const uint localIndex [[thread_position_in_grid]]
) {
    if (!validConfig(config, failure) || localIndex >= batch.control.y) {
        return;
    }
    const uint constraintIndex = batch.control.x + localIndex;
    if (constraintIndex >= config.constraintCounts.y) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
        return;
    }
    NumiClothBagGPUBend constraint = bends[constraintIndex];
    if (constraint.particlesAndColor.w != batch.control.z) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_BATCH);
        return;
    }
    const uint firstIndex = constraint.particlesAndColor.x;
    const uint middleIndex = constraint.particlesAndColor.y;
    const uint thirdIndex = constraint.particlesAndColor.z;
    if (firstIndex >= config.control.y || middleIndex >= config.control.y ||
        thirdIndex >= config.control.y || firstIndex == thirdIndex) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
        return;
    }
    NumiClothBagGPUParticle first = particles[firstIndex];
    NumiClothBagGPUParticle third = particles[thirdIndex];
    const float3 difference =
        third.positionAndInverseMass.xyz -
        first.positionAndInverseMass.xyz;
    const float currentChord = length(difference);
    if (!isfinite(currentChord) || !all(isfinite(constraint.material))) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
        return;
    }
    if (!(currentChord > 1.0e-12f)) {
        return;
    }
    const float value = currentChord - constraint.material.x;
    const float3 direction = difference / currentChord;
    const float3 gradients[2] = {-direction, direction};
    const float inverseMasses[2] = {
        first.positionAndInverseMass.w,
        third.positionAndInverseMass.w,
    };
    const float timestep = config.gravityAndTimestep.w;
    const float alpha = constraint.material.z / (timestep * timestep);
    float freeDenominator = alpha;
    for (uint participant = 0u; participant < 2u; ++participant) {
        freeDenominator += inverseMasses[participant] *
            dot(gradients[participant], gradients[participant]);
    }
    if (!(freeDenominator >= 1.0e-16f) || !isfinite(freeDenominator)) {
        return;
    }
    const float numerator = -value - alpha * constraint.material.w;
    const float freeDeltaLambda = numerator / freeDenominator;
    const bool groundEnabled = config.constraintCounts.z != 0u;
    const float groundHeight = config.clothMaterial.x;
    bool groundActive[2] = {false, false};
    float denominator = alpha;
    const float3 positions[2] = {
        first.positionAndInverseMass.xyz,
        third.positionAndInverseMass.xyz,
    };
    for (uint participant = 0u; participant < 2u; ++participant) {
        groundActive[participant] = groundEnabled &&
            positions[participant].z <= groundHeight + 1.0e-9f &&
            gradients[participant].z * freeDeltaLambda < 0.0f;
        denominator += inverseMasses[participant] * (
            dot(gradients[participant], gradients[participant]) -
            (groundActive[participant]
                ? gradients[participant].z * gradients[participant].z
                : 0.0f)
        );
    }
    if (!(denominator >= 1.0e-16f) || !isfinite(denominator)) {
        return;
    }
    const float deltaLambda = numerator / denominator;
    float fraction = 1.0f;
    if (groundEnabled) {
        for (uint participant = 0u; participant < 2u; ++participant) {
            if (groundActive[participant]) {
                continue;
            }
            const float verticalCorrection = gradients[participant].z *
                inverseMasses[participant] * deltaLambda;
            if (verticalCorrection < 0.0f) {
                fraction = min(
                    fraction,
                    max(0.0f, positions[participant].z - groundHeight) /
                        -verticalCorrection
                );
            }
        }
    }
    const float appliedLambda = deltaLambda * fraction;
    constraint.material.w += appliedLambda;
    float3 firstCorrection =
        gradients[0] * (inverseMasses[0] * appliedLambda);
    float3 thirdCorrection =
        gradients[1] * (inverseMasses[1] * appliedLambda);
    if (groundActive[0]) {
        firstCorrection.z = 0.0f;
    }
    if (groundActive[1]) {
        thirdCorrection.z = 0.0f;
    }
    first.positionAndInverseMass.xyz += firstCorrection;
    third.positionAndInverseMass.xyz += thirdCorrection;
    if (groundEnabled) {
        first.positionAndInverseMass.z = max(
            first.positionAndInverseMass.z, groundHeight
        );
        third.positionAndInverseMass.z = max(
            third.positionAndInverseMass.z, groundHeight
        );
    }
    if (!all(isfinite(first.positionAndInverseMass.xyz)) ||
        !all(isfinite(third.positionAndInverseMass.xyz)) ||
        !isfinite(constraint.material.w)) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
        return;
    }
    particles[firstIndex] = first;
    particles[thirdIndex] = third;
    bends[constraintIndex] = constraint;
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
