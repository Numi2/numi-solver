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
        !all(isfinite(config.clothMaterial)) || config.clothMaterial.x < 0.0f ||
        !all(isfinite(config.fruitMaterial))) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
        return false;
    }
    return true;
}

struct PointSegmentSample {
    float3 closest;
    float weight;
    float distance;
};

inline float3 safeNormalized(const float3 value) {
    const float magnitude = length(value);
    return magnitude > 1.0e-14f
        ? value / magnitude
        : float3(1.0f, 0.0f, 0.0f);
}

inline PointSegmentSample samplePointSegment(
    const float3 point,
    const float3 first,
    const float3 second
) {
    const float3 direction = second - first;
    const float lengthSquared = dot(direction, direction);
    const float weight = lengthSquared > 1.0e-20f
        ? clamp(dot(direction, point - first) / lengthSquared, 0.0f, 1.0f)
        : 0.0f;
    const float3 closest = fma(direction, float3(weight), first);
    return {closest, weight, length(closest - point)};
}

inline PointSegmentSample sampleSweptPointSegment(
    const NumiClothBagGPUFruit fruit,
    const NumiClothBagGPUParticle first,
    const NumiClothBagGPUParticle second,
    const float time
) {
    const float3 ballPosition = mix(
        fruit.previousAndRadius.xyz,
        fruit.positionAndInverseMass.xyz,
        time
    );
    const float3 firstPosition = mix(
        first.previousAndMass.xyz,
        first.positionAndInverseMass.xyz,
        time
    );
    const float3 secondPosition = mix(
        second.previousAndMass.xyz,
        second.positionAndInverseMass.xyz,
        time
    );
    return samplePointSegment(ballPosition, firstPosition, secondPosition);
}

inline float applyYarnCorrection(
    constant NumiClothBagGPUConfig& config,
    device NumiClothBagGPUParticle* particles,
    const uint firstIndex,
    const uint secondIndex,
    thread NumiClothBagGPUFruit& fruit,
    const float firstWeight,
    const float secondWeight,
    const float3 normal,
    const float correctionDistance,
    device atomic_uint* failure
) {
    NumiClothBagGPUParticle first = particles[firstIndex];
    NumiClothBagGPUParticle second = particles[secondIndex];
    const bool groundEnabled = config.constraintCounts.z != 0u;
    const float groundHeight = config.clothMaterial.x;
    bool firstGroundActive = groundEnabled && normal.z < 0.0f &&
        firstWeight > 0.0f &&
        first.positionAndInverseMass.z <= groundHeight + 1.0e-9f;
    bool secondGroundActive = groundEnabled && normal.z < 0.0f &&
        secondWeight > 0.0f &&
        second.positionAndInverseMass.z <= groundHeight + 1.0e-9f;
    if (firstGroundActive) {
        first.positionAndInverseMass.z = groundHeight;
    }
    if (secondGroundActive) {
        second.positionAndInverseMass.z = groundHeight;
    }

    float remaining = correctionDistance;
    float accumulatedLambda = 0.0f;
    for (uint activeSetIteration = 0u;
         activeSetIteration < 3u && remaining > 1.0e-14f;
         ++activeSetIteration) {
        float3 firstResponse = normal * first.positionAndInverseMass.w;
        float3 secondResponse = normal * second.positionAndInverseMass.w;
        if (firstGroundActive) {
            firstResponse.z = 0.0f;
        }
        if (secondGroundActive) {
            secondResponse.z = 0.0f;
        }
        const float denominator = fruit.positionAndInverseMass.w +
            dot(normal, firstResponse) * firstWeight * firstWeight +
            dot(normal, secondResponse) * secondWeight * secondWeight;
        if (!(denominator > 0.0f) || !isfinite(denominator)) {
            break;
        }
        const float unconstrainedLambda = remaining / denominator;
        float stepLambda = unconstrainedLambda;
        if (groundEnabled && normal.z < 0.0f) {
            const float firstVertical = firstResponse.z * firstWeight;
            if (!firstGroundActive && firstVertical < 0.0f) {
                stepLambda = min(
                    stepLambda,
                    max(
                        0.0f,
                        first.positionAndInverseMass.z - groundHeight
                    ) / -firstVertical
                );
            }
            const float secondVertical = secondResponse.z * secondWeight;
            if (!secondGroundActive && secondVertical < 0.0f) {
                stepLambda = min(
                    stepLambda,
                    max(
                        0.0f,
                        second.positionAndInverseMass.z - groundHeight
                    ) / -secondVertical
                );
            }
        }
        fruit.positionAndInverseMass.xyz -= normal *
            (fruit.positionAndInverseMass.w * stepLambda);
        first.positionAndInverseMass.xyz += firstResponse *
            (firstWeight * stepLambda);
        second.positionAndInverseMass.xyz += secondResponse *
            (secondWeight * stepLambda);
        accumulatedLambda += stepLambda;
        remaining = max(0.0f, remaining - denominator * stepLambda);
        if (stepLambda >= unconstrainedLambda - 1.0e-14f) {
            break;
        }
        bool activated = false;
        if (!firstGroundActive && firstWeight > 0.0f &&
            first.positionAndInverseMass.z <= groundHeight + 1.0e-9f) {
            first.positionAndInverseMass.z = groundHeight;
            firstGroundActive = true;
            activated = true;
        }
        if (!secondGroundActive && secondWeight > 0.0f &&
            second.positionAndInverseMass.z <= groundHeight + 1.0e-9f) {
            second.positionAndInverseMass.z = groundHeight;
            secondGroundActive = true;
            activated = true;
        }
        if (!activated) {
            break;
        }
    }
    if (!all(isfinite(fruit.positionAndInverseMass.xyz)) ||
        !all(isfinite(first.positionAndInverseMass.xyz)) ||
        !all(isfinite(second.positionAndInverseMass.xyz)) ||
        !isfinite(accumulatedLambda)) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
        return 0.0f;
    }
    particles[firstIndex] = first;
    particles[secondIndex] = second;
    return accumulatedLambda;
}

} // namespace

kernel void numi_cloth_bag_begin_substep(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device NumiClothBagGPUParticle* particles [[buffer(1)]],
    device NumiClothBagGPUDistance* distances [[buffer(2)]],
    device NumiClothBagGPUGrip* grips [[buffer(3)]],
    device NumiClothBagGPUKnot* knots [[buffer(4)]],
    device NumiClothBagGPUBend* bends [[buffer(5)]],
    device NumiClothBagGPUFruit* fruits [[buffer(6)]],
    device NumiClothBagGPUFruitPair* fruitPairs [[buffer(7)]],
    device atomic_uint* failure [[buffer(8)]],
    device NumiClothBagGPUYarnContact* yarnContacts [[buffer(9)]],
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
    if (index < config.constraintCounts.w) {
        NumiClothBagGPUFruit fruit = fruits[index];
        const float3 position = fruit.positionAndInverseMass.xyz;
        const float inverseMass = fruit.positionAndInverseMass.w;
        const float radius = fruit.previousAndRadius.w;
        const float3 velocity = fruit.velocityAndGroundImpulse.xyz;
        if (!all(isfinite(position)) || !all(isfinite(velocity)) ||
            !all(isfinite(fruit.angularVelocity)) ||
            !all(isfinite(fruit.orientation)) || !isfinite(inverseMass) ||
            !isfinite(radius) || !(inverseMass > 0.0f) || !(radius > 0.0f)) {
            recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
            return;
        }
        const float timestep = config.gravityAndTimestep.w;
        fruit.previousAndRadius.xyz = position;
        fruit.velocityAndGroundImpulse.xyz = fma(
            config.gravityAndTimestep.xyz,
            float3(timestep),
            velocity
        );
        fruit.velocityAndGroundImpulse.w = 0.0f;
        fruit.positionAndInverseMass.xyz = fma(
            fruit.velocityAndGroundImpulse.xyz,
            float3(timestep),
            position
        );
        fruits[index] = fruit;
    }
    if (index < config.contactCounts.x) {
        fruitPairs[index].contact = float4(0.0f);
    }
    if (index < config.contactCounts.y) {
        NumiClothBagGPUYarnContact contact = yarnContacts[index];
        contact.control.w = 0u;
        contact.fruitNormalAndImpulse = float4(0.0f);
        contact.segmentImpulse = float4(0.0f);
        yarnContacts[index] = contact;
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

kernel void numi_cloth_bag_solve_fruit_pair(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device NumiClothBagGPUFruit* fruits [[buffer(1)]],
    device NumiClothBagGPUFruitPair* pairs [[buffer(2)]],
    constant NumiClothBagGPUBatch& batch [[buffer(3)]],
    device atomic_uint* failure [[buffer(4)]],
    const uint localIndex [[thread_position_in_grid]]
) {
    if (!validConfig(config, failure) || localIndex >= batch.control.y) {
        return;
    }
    const uint pairIndex = batch.control.x + localIndex;
    if (pairIndex >= config.contactCounts.x) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
        return;
    }
    NumiClothBagGPUFruitPair pair = pairs[pairIndex];
    if (pair.fruitsAndColor.z != batch.control.z) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_BATCH);
        return;
    }
    const uint firstIndex = pair.fruitsAndColor.x;
    const uint secondIndex = pair.fruitsAndColor.y;
    if (firstIndex >= config.constraintCounts.w ||
        secondIndex >= config.constraintCounts.w ||
        firstIndex == secondIndex) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
        return;
    }
    NumiClothBagGPUFruit first = fruits[firstIndex];
    NumiClothBagGPUFruit second = fruits[secondIndex];
    const float3 difference =
        second.positionAndInverseMass.xyz -
        first.positionAndInverseMass.xyz;
    const float currentLength = length(difference);
    const float target =
        first.previousAndRadius.w + second.previousAndRadius.w;
    if (!isfinite(currentLength) || !isfinite(target) ||
        !all(isfinite(pair.contact))) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
        return;
    }
    if (!(currentLength < target) || !(currentLength > 1.0e-12f)) {
        return;
    }
    const float denominator =
        first.positionAndInverseMass.w + second.positionAndInverseMass.w;
    if (!(denominator > 0.0f) || !isfinite(denominator)) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
        return;
    }
    const float lambda = (target - currentLength) / denominator;
    const float3 normal = difference / currentLength;
    const float3 correction = normal * lambda;
    first.positionAndInverseMass.xyz -=
        correction * first.positionAndInverseMass.w;
    second.positionAndInverseMass.xyz +=
        correction * second.positionAndInverseMass.w;
    const float impulseMagnitude = lambda / config.gravityAndTimestep.w;
    pair.contact.xyz += normal * impulseMagnitude;
    pair.contact.w += impulseMagnitude;
    if (!all(isfinite(first.positionAndInverseMass.xyz)) ||
        !all(isfinite(second.positionAndInverseMass.xyz)) ||
        !all(isfinite(pair.contact))) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
        return;
    }
    fruits[firstIndex] = first;
    fruits[secondIndex] = second;
    pairs[pairIndex] = pair;
}

kernel void numi_cloth_bag_solve_ground(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device NumiClothBagGPUParticle* particles [[buffer(1)]],
    device NumiClothBagGPUFruit* fruits [[buffer(2)]],
    device atomic_uint* failure [[buffer(3)]],
    const uint index [[thread_position_in_grid]]
) {
    if (!validConfig(config, failure) || config.constraintCounts.z == 0u) {
        return;
    }
    if (index < config.control.y) {
        NumiClothBagGPUParticle particle = particles[index];
        if (!isfinite(particle.positionAndInverseMass.z)) {
            recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
            return;
        }
        particle.positionAndInverseMass.z = max(
            particle.positionAndInverseMass.z,
            config.clothMaterial.x
        );
        particles[index] = particle;
    }
    if (index < config.constraintCounts.w) {
        NumiClothBagGPUFruit fruit = fruits[index];
        const float penetration =
            fruit.previousAndRadius.w - fruit.positionAndInverseMass.z;
        if (!isfinite(penetration)) {
            recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
            return;
        }
        if (penetration > 0.0f) {
            const float impulse = penetration /
                (fruit.positionAndInverseMass.w *
                 config.gravityAndTimestep.w);
            fruit.velocityAndGroundImpulse.w += impulse;
            fruit.positionAndInverseMass.z = fruit.previousAndRadius.w;
        }
        fruits[index] = fruit;
    }
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

kernel void numi_cloth_bag_solve_yarn_batch(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device NumiClothBagGPUParticle* particles [[buffer(1)]],
    device const NumiClothBagGPUDistance* distances [[buffer(2)]],
    device NumiClothBagGPUFruit* fruits [[buffer(3)]],
    device NumiClothBagGPUYarnContact* contacts [[buffer(4)]],
    constant NumiClothBagGPUBatch& batch [[buffer(5)]],
    device atomic_uint* failure [[buffer(6)]],
    const uint fruitIndex [[thread_position_in_threadgroup]]
) {
    if (!validConfig(config, failure) ||
        config.contactCounts.y !=
            config.control.z * config.constraintCounts.w ||
        batch.control.x + batch.control.y > config.control.z ||
        batch.control.y == 0u || batch.control.w > 1u ||
        fruitIndex >= config.constraintCounts.w) {
        if (fruitIndex < config.constraintCounts.w) {
            recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
        }
        return;
    }
    NumiClothBagGPUFruit fruit = fruits[fruitIndex];
    for (uint phase = 0u; phase < batch.control.y; ++phase) {
        const uint localSegment =
            (phase + fruitIndex) % batch.control.y;
        const uint segmentIndex = batch.control.x + localSegment;
        const NumiClothBagGPUDistance segment = distances[segmentIndex];
        const uint firstIndex = segment.particlesAndColor.x;
        const uint secondIndex = segment.particlesAndColor.y;
        const uint contactIndex =
            fruitIndex * config.control.z + segmentIndex;
        if (segment.particlesAndColor.z != batch.control.z ||
            firstIndex >= config.control.y ||
            secondIndex >= config.control.y || firstIndex == secondIndex ||
            contactIndex >= config.contactCounts.y) {
            recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
            threadgroup_barrier(mem_flags::mem_device);
            continue;
        }
        NumiClothBagGPUYarnContact contact = contacts[contactIndex];
        float3 normal = float3(0.0f);
        float firstWeight = 0.0f;
        float secondWeight = 0.0f;
        float correction = 0.0f;
        if (batch.control.w == 1u) {
            if (contact.control.y != 0u) {
                normal = contact.sweptNormalAndAdvance.xyz;
                secondWeight = contact.weightsAndTime.y;
                firstWeight = 1.0f - secondWeight;
                correction = contact.sweptNormalAndAdvance.w;
            }
        } else {
            const NumiClothBagGPUParticle first = particles[firstIndex];
            const NumiClothBagGPUParticle second = particles[secondIndex];
            const PointSegmentSample current = samplePointSegment(
                fruit.positionAndInverseMass.xyz,
                first.positionAndInverseMass.xyz,
                second.positionAndInverseMass.xyz
            );
            const float target =
                fruit.previousAndRadius.w + config.clothMaterial.x;
            if (current.distance < target) {
                if (current.distance < 1.0e-12f) {
                    const float3 segmentDirection = safeNormalized(
                        second.positionAndInverseMass.xyz -
                        first.positionAndInverseMass.xyz
                    );
                    const float3 axis = abs(segmentDirection.z) < 0.9f
                        ? float3(0.0f, 0.0f, 1.0f)
                        : float3(1.0f, 0.0f, 0.0f);
                    normal = safeNormalized(cross(segmentDirection, axis));
                } else {
                    normal = (
                        current.closest - fruit.positionAndInverseMass.xyz
                    ) / current.distance;
                }
                secondWeight = current.weight;
                firstWeight = 1.0f - secondWeight;
                correction = target - current.distance;
            }
        }
        if (correction > 0.0f) {
            const float lambda = applyYarnCorrection(
                config,
                particles,
                firstIndex,
                secondIndex,
                fruit,
                firstWeight,
                secondWeight,
                normal,
                correction,
                failure
            );
            if (lambda > 0.0f) {
                const float impulse = lambda / config.gravityAndTimestep.w;
                contact.fruitNormalAndImpulse.xyz -= normal * impulse;
                contact.fruitNormalAndImpulse.w += impulse;
                contact.segmentImpulse.x += firstWeight * impulse;
                contact.segmentImpulse.y += secondWeight * impulse;
                if (batch.control.w == 1u) {
                    contact.segmentImpulse.z = contact.weightsAndTime.z;
                    contact.segmentImpulse.w += correction;
                }
                contact.control.w += 1u;
            }
        }
        contacts[contactIndex] = contact;
        threadgroup_barrier(mem_flags::mem_device);
    }
    fruits[fruitIndex] = fruit;
}

kernel void numi_cloth_bag_build_yarn_contacts(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device const NumiClothBagGPUParticle* particles [[buffer(1)]],
    device const NumiClothBagGPUDistance* distances [[buffer(2)]],
    device const NumiClothBagGPUFruit* fruits [[buffer(3)]],
    device NumiClothBagGPUYarnContact* contacts [[buffer(4)]],
    device atomic_uint* failure [[buffer(5)]],
    const uint index [[thread_position_in_grid]]
) {
    if (!validConfig(config, failure) || index >= config.contactCounts.y) {
        return;
    }
    const uint distanceCount = config.control.z;
    const uint fruitCount = config.constraintCounts.w;
    if (distanceCount == 0u || fruitCount == 0u ||
        config.contactCounts.y != distanceCount * fruitCount) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
        return;
    }
    const uint fruitIndex = index / distanceCount;
    const uint segmentIndex = index - fruitIndex * distanceCount;
    const NumiClothBagGPUDistance segment = distances[segmentIndex];
    const uint firstIndex = segment.particlesAndColor.x;
    const uint secondIndex = segment.particlesAndColor.y;
    if (fruitIndex >= fruitCount || firstIndex >= config.control.y ||
        secondIndex >= config.control.y || firstIndex == secondIndex) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
        return;
    }

    const NumiClothBagGPUFruit fruit = fruits[fruitIndex];
    const NumiClothBagGPUParticle first = particles[firstIndex];
    const NumiClothBagGPUParticle second = particles[secondIndex];
    const float target = fruit.previousAndRadius.w + config.clothMaterial.x;
    const PointSegmentSample current = samplePointSegment(
        fruit.positionAndInverseMass.xyz,
        first.positionAndInverseMass.xyz,
        second.positionAndInverseMass.xyz
    );
    float3 currentNormal;
    const bool degenerateCurrent = current.distance < 1.0e-12f;
    if (degenerateCurrent) {
        const float3 segmentDirection = safeNormalized(
            second.positionAndInverseMass.xyz -
            first.positionAndInverseMass.xyz
        );
        const float3 axis = abs(segmentDirection.z) < 0.9f
            ? float3(0.0f, 0.0f, 1.0f)
            : float3(1.0f, 0.0f, 0.0f);
        currentNormal = safeNormalized(cross(segmentDirection, axis));
    } else {
        currentNormal = (
            current.closest - fruit.positionAndInverseMass.xyz
        ) / current.distance;
    }

    float impactTime = 0.0f;
    float3 sweptNormal = float3(0.0f);
    float sweptWeight = 0.0f;
    float removedAdvance = 0.0f;
    bool sweptImpact = false;
    const float motionBound =
        length(
            fruit.positionAndInverseMass.xyz - fruit.previousAndRadius.xyz
        ) +
        length(
            first.positionAndInverseMass.xyz - first.previousAndMass.xyz
        ) +
        length(
            second.positionAndInverseMass.xyz - second.previousAndMass.xyz
        );
    if (motionBound >= 1.0e-14f) {
        constexpr float distanceTolerance = 1.0e-9f;
        PointSegmentSample impact = sampleSweptPointSegment(
            fruit, first, second, 0.0f
        );
        bool found = impact.distance <= target + distanceTolerance;
        if (!found) {
            for (uint iteration = 0u; iteration < 80u; ++iteration) {
                impact = sampleSweptPointSegment(
                    fruit, first, second, impactTime
                );
                const float gap = impact.distance - target;
                if (gap <= distanceTolerance) {
                    found = true;
                    break;
                }
                const float advance = 0.9f * gap / motionBound;
                if (!isfinite(advance) || !(advance > 0.0f) ||
                    impactTime + advance >= 1.0f) {
                    break;
                }
                impactTime += max(advance, 1.0e-10f);
            }
        }
        if (found && impact.distance >= 1.0e-12f) {
            const float3 ballAtImpact = mix(
                fruit.previousAndRadius.xyz,
                fruit.positionAndInverseMass.xyz,
                impactTime
            );
            const float3 firstAtImpact = mix(
                first.previousAndMass.xyz,
                first.positionAndInverseMass.xyz,
                impactTime
            );
            const float3 secondAtImpact = mix(
                second.previousAndMass.xyz,
                second.positionAndInverseMass.xyz,
                impactTime
            );
            sweptNormal = (impact.closest - ballAtImpact) / impact.distance;
            sweptWeight = impact.weight;
            const float3 segmentRemaining =
                (first.positionAndInverseMass.xyz - firstAtImpact) *
                    (1.0f - sweptWeight) +
                (second.positionAndInverseMass.xyz - secondAtImpact) *
                    sweptWeight;
            const float3 ballRemaining =
                fruit.positionAndInverseMass.xyz - ballAtImpact;
            removedAdvance = dot(
                ballRemaining - segmentRemaining, sweptNormal
            );
            sweptImpact = removedAdvance > 0.0f;
            if (!sweptImpact) {
                removedAdvance = 0.0f;
            }
        }
    }

    NumiClothBagGPUYarnContact contact = contacts[index];
    contact.identity = uint4(
        fruitIndex, segmentIndex, firstIndex, secondIndex
    );
    contact.currentNormalAndSeparation = float4(
        currentNormal, current.distance - target
    );
    contact.sweptNormalAndAdvance = float4(
        sweptNormal, removedAdvance
    );
    contact.weightsAndTime = float4(
        current.weight, sweptWeight, impactTime, target
    );
    contact.control.xyz = uint3(
        current.distance < target,
        sweptImpact,
        degenerateCurrent
    );
    if (!all(isfinite(contact.currentNormalAndSeparation)) ||
        !all(isfinite(contact.sweptNormalAndAdvance)) ||
        !all(isfinite(contact.weightsAndTime))) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
        return;
    }
    contacts[index] = contact;
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

kernel void numi_cloth_bag_finalize_fruit(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device NumiClothBagGPUFruit* fruits [[buffer(1)]],
    device atomic_uint* failure [[buffer(2)]],
    const uint index [[thread_position_in_grid]]
) {
    if (!validConfig(config, failure) ||
        index >= config.constraintCounts.w) {
        return;
    }
    NumiClothBagGPUFruit fruit = fruits[index];
    fruit.velocityAndGroundImpulse.xyz =
        (fruit.positionAndInverseMass.xyz - fruit.previousAndRadius.xyz) /
        config.gravityAndTimestep.w;
    if (!all(isfinite(fruit.velocityAndGroundImpulse))) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
        return;
    }
    fruits[index] = fruit;
}
