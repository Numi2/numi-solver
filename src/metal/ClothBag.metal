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

struct SegmentSegmentSample {
    float3 firstPoint;
    float3 secondPoint;
    float firstWeight;
    float secondWeight;
    float distance;
};

inline float3 safeNormalized(const float3 value) {
    const float magnitude = length(value);
    return magnitude > 1.0e-14f
        ? value / magnitude
        : float3(1.0f, 0.0f, 0.0f);
}

inline float fruitInverseInertia(const NumiClothBagGPUFruit fruit) {
    const float radius = fruit.previousAndRadius.w;
    return 2.5f * fruit.positionAndInverseMass.w / (radius * radius);
}

inline void applyFruitImpulse(
    thread NumiClothBagGPUFruit& fruit,
    const float3 impulse,
    const float3 contactOffset
) {
    fruit.velocityAndGroundImpulse.xyz +=
        impulse * fruit.positionAndInverseMass.w;
    fruit.angularVelocity.xyz +=
        cross(contactOffset, impulse) * fruitInverseInertia(fruit);
}

inline void recordFrictionStatus(
    device atomic_uint* status,
    const uint counterIndex,
    const float tangentialImpulse,
    const float frictionLimit
) {
    atomic_fetch_add_explicit(
        status + counterIndex, 1u, memory_order_relaxed
    );
    if (frictionLimit > 0.0f) {
        atomic_fetch_max_explicit(
            status + 4u,
            as_type<uint>(tangentialImpulse / frictionLimit),
            memory_order_relaxed
        );
    }
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

inline SegmentSegmentSample sampleSegments(
    const float3 firstStart,
    const float3 firstEnd,
    const float3 secondStart,
    const float3 secondEnd
) {
    const float3 firstDirection = firstEnd - firstStart;
    const float3 secondDirection = secondEnd - secondStart;
    const float3 offset = firstStart - secondStart;
    const float firstLengthSquared = dot(firstDirection, firstDirection);
    const float secondLengthSquared = dot(secondDirection, secondDirection);
    const float secondProjection = dot(secondDirection, offset);
    float firstWeight = 0.0f;
    float secondWeight = 0.0f;
    if (firstLengthSquared <= 1.0e-20f &&
        secondLengthSquared <= 1.0e-20f) {
        return {
            firstStart,
            secondStart,
            0.0f,
            0.0f,
            length(secondStart - firstStart),
        };
    }
    if (firstLengthSquared <= 1.0e-20f) {
        secondWeight = clamp(
            secondProjection / secondLengthSquared, 0.0f, 1.0f
        );
    } else {
        const float firstProjection = dot(firstDirection, offset);
        if (secondLengthSquared <= 1.0e-20f) {
            firstWeight = clamp(
                -firstProjection / firstLengthSquared, 0.0f, 1.0f
            );
        } else {
            const float crossProjection = dot(
                firstDirection, secondDirection
            );
            const float denominator =
                firstLengthSquared * secondLengthSquared -
                crossProjection * crossProjection;
            if (denominator > 1.0e-20f) {
                firstWeight = clamp(
                    (crossProjection * secondProjection -
                     firstProjection * secondLengthSquared) / denominator,
                    0.0f,
                    1.0f
                );
            }
            secondWeight = (
                crossProjection * firstWeight + secondProjection
            ) / secondLengthSquared;
            if (secondWeight < 0.0f) {
                secondWeight = 0.0f;
                firstWeight = clamp(
                    -firstProjection / firstLengthSquared, 0.0f, 1.0f
                );
            } else if (secondWeight > 1.0f) {
                secondWeight = 1.0f;
                firstWeight = clamp(
                    (crossProjection - firstProjection) /
                        firstLengthSquared,
                    0.0f,
                    1.0f
                );
            }
        }
    }
    const float3 firstPoint = fma(
        firstDirection, float3(firstWeight), firstStart
    );
    const float3 secondPoint = fma(
        secondDirection, float3(secondWeight), secondStart
    );
    return {
        firstPoint,
        secondPoint,
        firstWeight,
        secondWeight,
        length(secondPoint - firstPoint),
    };
}

inline float3 selfContactNormal(
    const float3 separation,
    const float3 firstDirection,
    const float3 secondDirection,
    const float3 previousOffset
) {
    const float separationLength = length(separation);
    if (separationLength > 1.0e-12f) {
        return separation / separationLength;
    }
    float3 normal = safeNormalized(cross(firstDirection, secondDirection));
    if (dot(previousOffset, normal) > 0.0f) {
        normal *= -1.0f;
    }
    return normal;
}

inline bool edgeBoundsOverlap(
    const NumiClothBagGPUParticle firstStart,
    const NumiClothBagGPUParticle firstEnd,
    const NumiClothBagGPUParticle secondStart,
    const NumiClothBagGPUParticle secondEnd,
    const bool swept,
    const float expansion
) {
    float3 firstMinimum = min(
        firstStart.positionAndInverseMass.xyz,
        firstEnd.positionAndInverseMass.xyz
    );
    float3 firstMaximum = max(
        firstStart.positionAndInverseMass.xyz,
        firstEnd.positionAndInverseMass.xyz
    );
    float3 secondMinimum = min(
        secondStart.positionAndInverseMass.xyz,
        secondEnd.positionAndInverseMass.xyz
    );
    float3 secondMaximum = max(
        secondStart.positionAndInverseMass.xyz,
        secondEnd.positionAndInverseMass.xyz
    );
    if (swept) {
        firstMinimum = min(
            firstMinimum,
            min(
                firstStart.previousAndMass.xyz,
                firstEnd.previousAndMass.xyz
            )
        );
        firstMaximum = max(
            firstMaximum,
            max(
                firstStart.previousAndMass.xyz,
                firstEnd.previousAndMass.xyz
            )
        );
        secondMinimum = min(
            secondMinimum,
            min(
                secondStart.previousAndMass.xyz,
                secondEnd.previousAndMass.xyz
            )
        );
        secondMaximum = max(
            secondMaximum,
            max(
                secondStart.previousAndMass.xyz,
                secondEnd.previousAndMass.xyz
            )
        );
    }
    firstMinimum -= expansion;
    firstMaximum += expansion;
    secondMinimum -= expansion;
    secondMaximum += expansion;
    return all(firstMinimum <= secondMaximum) &&
        all(firstMaximum >= secondMinimum);
}

inline SegmentSegmentSample sampleSweptSegments(
    const NumiClothBagGPUParticle firstStart,
    const NumiClothBagGPUParticle firstEnd,
    const NumiClothBagGPUParticle secondStart,
    const NumiClothBagGPUParticle secondEnd,
    const float time
) {
    return sampleSegments(
        mix(
            firstStart.previousAndMass.xyz,
            firstStart.positionAndInverseMass.xyz,
            time
        ),
        mix(
            firstEnd.previousAndMass.xyz,
            firstEnd.positionAndInverseMass.xyz,
            time
        ),
        mix(
            secondStart.previousAndMass.xyz,
            secondStart.positionAndInverseMass.xyz,
            time
        ),
        mix(
            secondEnd.previousAndMass.xyz,
            secondEnd.positionAndInverseMass.xyz,
            time
        )
    );
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
        particle.velocity.w = 0.0f;
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
            particle.velocity.w = particle.velocity.z;
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

kernel void numi_cloth_bag_build_self_cells(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device const NumiClothBagGPUParticle* particles [[buffer(1)]],
    device const NumiClothBagGPUDistance* distances [[buffer(2)]],
    device ulong* entries [[buffer(3)]],
    device atomic_uint* failure [[buffer(4)]],
    const uint index [[thread_position_in_grid]]
) {
    constexpr uint entryCapacity = 4096u;
    if (!validConfig(config, failure) || index >= entryCapacity) {
        return;
    }
    if (index >= config.control.z) {
        entries[index] = 0xfffffffffffffffful;
        return;
    }
    const NumiClothBagGPUDistance segment = distances[index];
    const uint firstIndex = segment.particlesAndColor.x;
    const uint secondIndex = segment.particlesAndColor.y;
    const float cellSize = config.clothMaterial.y;
    if (firstIndex >= config.control.y || secondIndex >= config.control.y ||
        !(cellSize > 2.0f * config.clothMaterial.x)) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
        entries[index] = 0xfffffffffffffffful;
        return;
    }
    const float3 first = particles[firstIndex].positionAndInverseMass.xyz;
    const float3 second = particles[secondIndex].positionAndInverseMass.xyz;
    if (length(second - first) >
        cellSize - 2.0f * config.clothMaterial.x) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
        entries[index] = 0xfffffffffffffffful;
        return;
    }
    const int3 coordinate = int3(floor((0.5f * (first + second)) / cellSize));
    if (any(coordinate < int3(-512)) || any(coordinate > int3(511))) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
        entries[index] = 0xfffffffffffffffful;
        return;
    }
    const uint3 encoded = uint3(coordinate + int3(512));
    const uint key = encoded.x | (encoded.y << 10u) | (encoded.z << 20u);
    entries[index] = (static_cast<ulong>(key) << 32u) |
        static_cast<ulong>(index);
}

kernel void numi_cloth_bag_sort_self_cells(
    device ulong* entries [[buffer(0)]],
    const uint localIndex [[thread_position_in_threadgroup]],
    const uint groupIndex [[threadgroup_position_in_grid]]
) {
    constexpr uint entryCapacity = 4096u;
    constexpr uint threadCount = 256u;
    threadgroup ulong sortedEntries[entryCapacity];
    if (groupIndex != 0u) {
        return;
    }
    for (uint index = localIndex;
         index < entryCapacity;
         index += threadCount) {
        sortedEntries[index] = entries[index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint sequence = 2u;
         sequence <= entryCapacity;
         sequence <<= 1u) {
        for (uint stride = sequence >> 1u;
             stride > 0u;
             stride >>= 1u) {
            for (uint index = localIndex;
                 index < entryCapacity;
                 index += threadCount) {
                const uint partner = index ^ stride;
                if (partner > index) {
                    const bool ascending = (index & sequence) == 0u;
                    const ulong first = sortedEntries[index];
                    const ulong second = sortedEntries[partner];
                    if ((ascending && first > second) ||
                        (!ascending && first < second)) {
                        sortedEntries[index] = second;
                        sortedEntries[partner] = first;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint index = localIndex;
         index < entryCapacity;
         index += threadCount) {
        entries[index] = sortedEntries[index];
    }
}

kernel void numi_cloth_bag_detect_self_spatial(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device const NumiClothBagGPUParticle* particles [[buffer(1)]],
    device const NumiClothBagGPUDistance* distances [[buffer(2)]],
    device const ulong* sortedEntries [[buffer(3)]],
    device const uint* pairLookup [[buffer(4)]],
    device uint* activeEpochs [[buffer(5)]],
    device atomic_uint* failure [[buffer(6)]],
    constant uint& epoch [[buffer(7)]],
    const uint firstSegment [[thread_position_in_grid]]
) {
    if (!validConfig(config, failure) || firstSegment >= config.control.z ||
        epoch <= 1u) {
        return;
    }
    const NumiClothBagGPUDistance firstDistance = distances[firstSegment];
    const uint firstStartIndex = firstDistance.particlesAndColor.x;
    const uint firstEndIndex = firstDistance.particlesAndColor.y;
    const NumiClothBagGPUParticle firstStart = particles[firstStartIndex];
    const NumiClothBagGPUParticle firstEnd = particles[firstEndIndex];
    const float cellSize = config.clothMaterial.y;
    const int3 coordinate = int3(floor(
        (0.5f * (
            firstStart.positionAndInverseMass.xyz +
            firstEnd.positionAndInverseMass.xyz
        )) / cellSize
    ));
    for (int zOffset = -1; zOffset <= 1; ++zOffset) {
        for (int yOffset = -1; yOffset <= 1; ++yOffset) {
            for (int xOffset = -1; xOffset <= 1; ++xOffset) {
                const int3 neighbor = coordinate + int3(
                    xOffset, yOffset, zOffset
                );
                if (any(neighbor < int3(-512)) ||
                    any(neighbor > int3(511))) {
                    continue;
                }
                const uint3 encoded = uint3(neighbor + int3(512));
                const uint key = encoded.x |
                    (encoded.y << 10u) | (encoded.z << 20u);
                uint lower = 0u;
                uint upper = config.control.z;
                while (lower < upper) {
                    const uint middle = lower + (upper - lower) / 2u;
                    const uint middleKey = static_cast<uint>(
                        sortedEntries[middle] >> 32u
                    );
                    if (middleKey < key) {
                        lower = middle + 1u;
                    } else {
                        upper = middle;
                    }
                }
                for (uint sortedIndex = lower;
                     sortedIndex < config.control.z;
                     ++sortedIndex) {
                    const ulong entry = sortedEntries[sortedIndex];
                    if (static_cast<uint>(entry >> 32u) != key) {
                        break;
                    }
                    const uint secondSegment = static_cast<uint>(entry);
                    if (secondSegment <= firstSegment) {
                        continue;
                    }
                    const uint pairLookupIndex =
                        firstSegment *
                            (2u * config.control.z - firstSegment - 1u) /
                            2u +
                        (secondSegment - firstSegment - 1u);
                    const uint pairIndex = pairLookup[pairLookupIndex];
                    if (pairIndex == NUMI_CLOTH_BAG_GPU_INVALID_PARTICLE) {
                        continue;
                    }
                    const NumiClothBagGPUDistance secondDistance =
                        distances[secondSegment];
                    if (edgeBoundsOverlap(
                        firstStart,
                        firstEnd,
                        particles[secondDistance.particlesAndColor.x],
                        particles[secondDistance.particlesAndColor.y],
                        false,
                        config.clothMaterial.x
                    )) {
                        activeEpochs[pairIndex] = epoch;
                    }
                }
            }
        }
    }
}

kernel void numi_cloth_bag_count_self_batches(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device const NumiClothBagGPUBatch* batches [[buffer(1)]],
    device const uint* activeEpochs [[buffer(2)]],
    device uint* activeBatchCounts [[buffer(3)]],
    device atomic_uint* failure [[buffer(4)]],
    constant uint& epoch [[buffer(5)]],
    const uint batchIndex [[thread_position_in_grid]]
) {
    if (!validConfig(config, failure) ||
        batchIndex >= config.contactCounts.w) {
        return;
    }
    const NumiClothBagGPUBatch batch = batches[batchIndex];
    if (batch.control.x + batch.control.y > config.contactCounts.z) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
        activeBatchCounts[batchIndex] = 0u;
        return;
    }
    uint count = 0u;
    for (uint local = 0u; local < batch.control.y; ++local) {
        count += activeEpochs[batch.control.x + local] == epoch;
    }
    activeBatchCounts[batchIndex] = count;
}

kernel void numi_cloth_bag_compact_self_batches(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device const uint* activeBatchCounts [[buffer(1)]],
    device uint* activeBatchIndices [[buffer(2)]],
    device uint* activeBatchCount [[buffer(3)]],
    const uint localIndex [[thread_position_in_threadgroup]],
    const uint groupIndex [[threadgroup_position_in_grid]]
) {
    constexpr uint threadCount = 256u;
    threadgroup uint prefix[threadCount];
    threadgroup uint base;
    if (groupIndex != 0u) {
        return;
    }
    if (localIndex == 0u) {
        base = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint chunkCount =
        (config.contactCounts.w + threadCount - 1u) / threadCount;
    for (uint chunk = 0u; chunk < chunkCount; ++chunk) {
        const uint batchIndex = chunk * threadCount + localIndex;
        const uint active = batchIndex < config.contactCounts.w &&
            activeBatchCounts[batchIndex] != 0u;
        prefix[localIndex] = active;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = 1u; offset < threadCount; offset <<= 1u) {
            const uint addition = localIndex >= offset
                ? prefix[localIndex - offset]
                : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            prefix[localIndex] += addition;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (active != 0u) {
            activeBatchIndices[base + prefix[localIndex] - 1u] = batchIndex;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (localIndex == 0u) {
            base += prefix[threadCount - 1u];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (localIndex == 0u) {
        *activeBatchCount = base;
    }
}

kernel void numi_cloth_bag_detect_self_contact(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device const NumiClothBagGPUParticle* particles [[buffer(1)]],
    device const NumiClothBagGPUSelfPair* pairs [[buffer(2)]],
    device const NumiClothBagGPUBatch* batches [[buffer(3)]],
    device uint* activeFlags [[buffer(4)]],
    device uint* activeBatchCounts [[buffer(5)]],
    device atomic_uint* failure [[buffer(6)]],
    constant uint& mode [[buffer(7)]],
    device const NumiClothBagGPUDistance* distances [[buffer(8)]],
    const uint localIndex [[thread_position_in_threadgroup]],
    const uint batchIndex [[threadgroup_position_in_grid]]
) {
    threadgroup atomic_uint groupCount;
    if (localIndex == 0u) {
        atomic_store_explicit(&groupCount, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!validConfig(config, failure) || mode > 1u ||
        batchIndex >= config.contactCounts.w) {
        if (localIndex == 0u) {
            recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
        }
        return;
    }
    const NumiClothBagGPUBatch batch = batches[batchIndex];
    if (batch.control.x + batch.control.y > config.contactCounts.z) {
        if (localIndex == 0u) {
            recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
            activeBatchCounts[batchIndex] = 0u;
        }
        return;
    }
    if (localIndex < batch.control.y) {
        const uint pairIndex = batch.control.x + localIndex;
        const NumiClothBagGPUSelfPair pair = pairs[pairIndex];
        bool active = false;
        if (pair.firstSegment >= config.control.z ||
            pair.secondSegment >= config.control.z ||
            pair.firstSegment == pair.secondSegment) {
            recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
        } else {
            const uint4 first =
                distances[pair.firstSegment].particlesAndColor;
            const uint4 second =
                distances[pair.secondSegment].particlesAndColor;
            const uint4 indices = uint4(
                first.x, first.y, second.x, second.y
            );
            if (any(indices >= uint4(config.control.y))) {
                recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
            } else {
                active = edgeBoundsOverlap(
                    particles[indices.x],
                    particles[indices.y],
                    particles[indices.z],
                    particles[indices.w],
                    mode == 1u,
                    config.clothMaterial.x
                );
            }
        }
        activeFlags[pairIndex] = active;
        if (active) {
            atomic_fetch_add_explicit(
                &groupCount, 1u, memory_order_relaxed
            );
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (localIndex == 0u) {
        activeBatchCounts[batchIndex] = atomic_load_explicit(
            &groupCount, memory_order_relaxed
        );
    }
}

kernel void numi_cloth_bag_solve_self_contact(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device NumiClothBagGPUParticle* particles [[buffer(1)]],
    device const NumiClothBagGPUSelfPair* pairs [[buffer(2)]],
    device const NumiClothBagGPUBatch* batches [[buffer(3)]],
    device const uint* activeEpochs [[buffer(4)]],
    device const uint* activeBatchIndices [[buffer(5)]],
    device const uint* activeBatchCount [[buffer(6)]],
    device atomic_uint* status [[buffer(7)]],
    device atomic_uint* failure [[buffer(8)]],
    constant uint& mode [[buffer(9)]],
    constant uint& epoch [[buffer(10)]],
    device const NumiClothBagGPUDistance* distances [[buffer(11)]],
    device NumiClothBagGPUSelfImpulse* impulseRecords [[buffer(12)]],
    device ulong* impulseKeys [[buffer(13)]],
    device atomic_uint* impulseCount [[buffer(14)]],
    const uint localIndex [[thread_position_in_threadgroup]],
    const uint groupWidth [[threads_per_threadgroup]],
    const uint groupIndex [[threadgroup_position_in_grid]]
) {
    threadgroup uint responsePrefix[256];
    threadgroup uint responseBase;
    if (!validConfig(config, failure) || mode > 1u || groupIndex != 0u) {
        if (localIndex == 0u) {
            recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
        }
        return;
    }
    constexpr float distanceTolerance = 1.0e-9f;
    const float target = 2.0f * config.clothMaterial.x;
    for (uint activeBatch = 0u;
         activeBatch < *activeBatchCount;
         ++activeBatch) {
        bool acceptedResponse = false;
        NumiClothBagGPUSelfImpulse response = {};
        const uint batchIndex = activeBatchIndices[activeBatch];
        const NumiClothBagGPUBatch batch = batches[batchIndex];
        if (batch.control.x + batch.control.y > config.contactCounts.z) {
            if (localIndex == 0u) {
                recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
            }
            continue;
        }
        const uint pairIndex = batch.control.x + localIndex;
        if (localIndex < batch.control.y &&
            activeEpochs[pairIndex] == epoch) {
            const NumiClothBagGPUSelfPair pair = pairs[pairIndex];
            if (pair.firstSegment >= config.control.z ||
                pair.secondSegment >= config.control.z ||
                pair.firstSegment == pair.secondSegment) {
                recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
            } else {
                const uint4 first =
                    distances[pair.firstSegment].particlesAndColor;
                const uint4 second =
                    distances[pair.secondSegment].particlesAndColor;
                const uint4 indices = uint4(
                    first.x, first.y, second.x, second.y
                );
                if (any(indices >= uint4(config.control.y)) ||
                    indices.x == indices.y || indices.x == indices.z ||
                    indices.x == indices.w || indices.y == indices.z ||
                    indices.y == indices.w || indices.z == indices.w) {
                    recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
                } else {
                NumiClothBagGPUParticle firstStart = particles[indices.x];
                NumiClothBagGPUParticle firstEnd = particles[indices.y];
                NumiClothBagGPUParticle secondStart = particles[indices.z];
                NumiClothBagGPUParticle secondEnd = particles[indices.w];
                if (edgeBoundsOverlap(
                    firstStart,
                    firstEnd,
                    secondStart,
                    secondEnd,
                    mode == 1u,
                    config.clothMaterial.x
                )) {
                    float firstWeight = 0.0f;
                    float secondWeight = 0.0f;
                    float3 normal = float3(0.0f);
                    float correction = 0.0f;
                    if (mode == 0u) {
                        const SegmentSegmentSample closest = sampleSegments(
                            firstStart.positionAndInverseMass.xyz,
                            firstEnd.positionAndInverseMass.xyz,
                            secondStart.positionAndInverseMass.xyz,
                            secondEnd.positionAndInverseMass.xyz
                        );
                        if (closest.distance < target) {
                            firstWeight = closest.firstWeight;
                            secondWeight = closest.secondWeight;
                            const float3 firstPrevious = mix(
                                firstStart.previousAndMass.xyz,
                                firstEnd.previousAndMass.xyz,
                                firstWeight
                            );
                            const float3 secondPrevious = mix(
                                secondStart.previousAndMass.xyz,
                                secondEnd.previousAndMass.xyz,
                                secondWeight
                            );
                            normal = selfContactNormal(
                                closest.secondPoint - closest.firstPoint,
                                firstEnd.positionAndInverseMass.xyz -
                                    firstStart.positionAndInverseMass.xyz,
                                secondEnd.positionAndInverseMass.xyz -
                                    secondStart.positionAndInverseMass.xyz,
                                firstPrevious - secondPrevious
                            );
                            correction = target - closest.distance;
                        }
                    } else {
                        const float motionBound =
                            length(
                                firstStart.positionAndInverseMass.xyz -
                                firstStart.previousAndMass.xyz
                            ) +
                            length(
                                firstEnd.positionAndInverseMass.xyz -
                                firstEnd.previousAndMass.xyz
                            ) +
                            length(
                                secondStart.positionAndInverseMass.xyz -
                                secondStart.previousAndMass.xyz
                            ) +
                            length(
                                secondEnd.positionAndInverseMass.xyz -
                                secondEnd.previousAndMass.xyz
                            );
                        if (motionBound >= 1.0e-14f) {
                            float time = 0.0f;
                            SegmentSegmentSample impact =
                                sampleSweptSegments(
                                    firstStart,
                                    firstEnd,
                                    secondStart,
                                    secondEnd,
                                    0.0f
                                );
                            bool found = impact.distance <=
                                target + distanceTolerance;
                            if (!found) {
                                for (uint iteration = 0u;
                                     iteration < 80u;
                                     ++iteration) {
                                    impact = sampleSweptSegments(
                                        firstStart,
                                        firstEnd,
                                        secondStart,
                                        secondEnd,
                                        time
                                    );
                                    const float gap = impact.distance - target;
                                    if (gap <= distanceTolerance) {
                                        found = true;
                                        break;
                                    }
                                    const float advance =
                                        0.9f * gap / motionBound;
                                    if (!isfinite(advance) ||
                                        !(advance > 0.0f) ||
                                        time + advance >= 1.0f) {
                                        break;
                                    }
                                    time += max(advance, 1.0e-10f);
                                }
                            }
                            if (found) {
                                firstWeight = impact.firstWeight;
                                secondWeight = impact.secondWeight;
                                const float3 firstPrevious = mix(
                                    firstStart.previousAndMass.xyz,
                                    firstEnd.previousAndMass.xyz,
                                    firstWeight
                                );
                                const float3 secondPrevious = mix(
                                    secondStart.previousAndMass.xyz,
                                    secondEnd.previousAndMass.xyz,
                                    secondWeight
                                );
                                const float3 firstImpactStart = mix(
                                    firstStart.previousAndMass.xyz,
                                    firstStart.positionAndInverseMass.xyz,
                                    time
                                );
                                const float3 firstImpactEnd = mix(
                                    firstEnd.previousAndMass.xyz,
                                    firstEnd.positionAndInverseMass.xyz,
                                    time
                                );
                                const float3 secondImpactStart = mix(
                                    secondStart.previousAndMass.xyz,
                                    secondStart.positionAndInverseMass.xyz,
                                    time
                                );
                                const float3 secondImpactEnd = mix(
                                    secondEnd.previousAndMass.xyz,
                                    secondEnd.positionAndInverseMass.xyz,
                                    time
                                );
                                normal = selfContactNormal(
                                    impact.secondPoint - impact.firstPoint,
                                    firstImpactEnd - firstImpactStart,
                                    secondImpactEnd - secondImpactStart,
                                    firstPrevious - secondPrevious
                                );
                                const float3 firstRemaining =
                                    (firstStart.positionAndInverseMass.xyz -
                                     firstImpactStart) *
                                        (1.0f - firstWeight) +
                                    (firstEnd.positionAndInverseMass.xyz -
                                     firstImpactEnd) * firstWeight;
                                const float3 secondRemaining =
                                    (secondStart.positionAndInverseMass.xyz -
                                     secondImpactStart) *
                                        (1.0f - secondWeight) +
                                    (secondEnd.positionAndInverseMass.xyz -
                                     secondImpactEnd) * secondWeight;
                                correction = dot(
                                    firstRemaining - secondRemaining,
                                    normal
                                );
                                if (!(correction > 0.0f)) {
                                    correction = 0.0f;
                                }
                            }
                        }
                    }
                    if (correction > 0.0f) {
                        const float firstStartWeight = 1.0f - firstWeight;
                        const float secondStartWeight = 1.0f - secondWeight;
                        const float denominator =
                            firstStart.positionAndInverseMass.w *
                                firstStartWeight * firstStartWeight +
                            firstEnd.positionAndInverseMass.w *
                                firstWeight * firstWeight +
                            secondStart.positionAndInverseMass.w *
                                secondStartWeight * secondStartWeight +
                            secondEnd.positionAndInverseMass.w *
                                secondWeight * secondWeight;
                        if (denominator > 0.0f && isfinite(denominator)) {
                            const float lambda = correction / denominator;
                            firstStart.positionAndInverseMass.xyz -= normal *
                                (firstStart.positionAndInverseMass.w *
                                 firstStartWeight * lambda);
                            firstEnd.positionAndInverseMass.xyz -= normal *
                                (firstEnd.positionAndInverseMass.w *
                                 firstWeight * lambda);
                            secondStart.positionAndInverseMass.xyz += normal *
                                (secondStart.positionAndInverseMass.w *
                                 secondStartWeight * lambda);
                            secondEnd.positionAndInverseMass.xyz += normal *
                                (secondEnd.positionAndInverseMass.w *
                                 secondWeight * lambda);
                            particles[indices.x] = firstStart;
                            particles[indices.y] = firstEnd;
                            particles[indices.z] = secondStart;
                            particles[indices.w] = secondEnd;
                            atomic_fetch_add_explicit(
                                status + mode, 1u, memory_order_relaxed
                            );
                            atomic_fetch_max_explicit(
                                status + 2u,
                                as_type<uint>(correction),
                                memory_order_relaxed
                            );
                            const float impulse = lambda /
                                config.gravityAndTimestep.w;
                            response.identity = uint4(
                                pairIndex,
                                pair.firstSegment,
                                pair.secondSegment,
                                epoch
                            );
                            response.normalAndImpulse = float4(
                                -normal * impulse, impulse
                            );
                            response.endpointImpulses = float4(
                                firstStartWeight * impulse,
                                firstWeight * impulse,
                                secondStartWeight * impulse,
                                secondWeight * impulse
                            );
                            acceptedResponse = true;
                        }
                    }
                }
                }
            }
        }
        responsePrefix[localIndex] = acceptedResponse ? 1u : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        for (uint offset = 1u; offset < groupWidth; offset <<= 1u) {
            const uint addition = localIndex >= offset
                ? responsePrefix[localIndex - offset]
                : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            responsePrefix[localIndex] += addition;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (localIndex == 0u) {
            const uint acceptedCount = responsePrefix[groupWidth - 1u];
            responseBase = atomic_fetch_add_explicit(
                impulseCount, acceptedCount, memory_order_relaxed
            );
            if (responseBase + acceptedCount >
                NUMI_CLOTH_BAG_GPU_SELF_IMPULSE_CAPACITY) {
                recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (acceptedResponse) {
            const uint recordIndex =
                responseBase + responsePrefix[localIndex] - 1u;
            if (recordIndex < NUMI_CLOTH_BAG_GPU_SELF_IMPULSE_CAPACITY) {
                impulseRecords[recordIndex] = response;
                impulseKeys[recordIndex] =
                    (static_cast<ulong>(response.identity.x) << 32u) |
                    static_cast<ulong>(recordIndex);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
}

kernel void numi_cloth_bag_sort_self_impulses(
    device ulong* keys [[buffer(0)]],
    const uint localIndex [[thread_position_in_threadgroup]],
    const uint groupIndex [[threadgroup_position_in_grid]]
) {
    constexpr uint capacity = NUMI_CLOTH_BAG_GPU_SELF_IMPULSE_CAPACITY;
    constexpr uint threadCount = 256u;
    threadgroup ulong sortedKeys[capacity];
    if (groupIndex != 0u) {
        return;
    }
    for (uint index = localIndex; index < capacity; index += threadCount) {
        sortedKeys[index] = keys[index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint sequence = 2u; sequence <= capacity; sequence <<= 1u) {
        for (uint stride = sequence >> 1u; stride > 0u; stride >>= 1u) {
            for (uint index = localIndex;
                 index < capacity;
                 index += threadCount) {
                const uint partner = index ^ stride;
                if (partner > index) {
                    const bool ascending = (index & sequence) == 0u;
                    const ulong first = sortedKeys[index];
                    const ulong second = sortedKeys[partner];
                    if ((ascending && first > second) ||
                        (!ascending && first < second)) {
                        sortedKeys[index] = second;
                        sortedKeys[partner] = first;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint index = localIndex; index < capacity; index += threadCount) {
        keys[index] = sortedKeys[index];
    }
}

kernel void numi_cloth_bag_clear_self_impulse_map(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device uint* pairToAggregate [[buffer(1)]],
    device atomic_uint* failure [[buffer(2)]],
    const uint pairIndex [[thread_position_in_grid]]
) {
    if (!validConfig(config, failure) ||
        pairIndex >= config.contactCounts.z) {
        return;
    }
    pairToAggregate[pairIndex] = 0u;
}

kernel void numi_cloth_bag_aggregate_self_impulses(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device const ulong* sortedKeys [[buffer(1)]],
    device const NumiClothBagGPUSelfImpulse* records [[buffer(2)]],
    device const atomic_uint* recordCount [[buffer(3)]],
    device NumiClothBagGPUSelfImpulse* aggregates [[buffer(4)]],
    device uint* pairToAggregate [[buffer(5)]],
    device atomic_uint* failure [[buffer(6)]],
    const uint sortedIndex [[thread_position_in_grid]]
) {
    if (!validConfig(config, failure) ||
        sortedIndex >= NUMI_CLOTH_BAG_GPU_SELF_IMPULSE_CAPACITY) {
        return;
    }
    const uint count = min(
        atomic_load_explicit(recordCount, memory_order_relaxed),
        NUMI_CLOTH_BAG_GPU_SELF_IMPULSE_CAPACITY
    );
    if (sortedIndex >= count) {
        return;
    }
    const uint pairIndex = static_cast<uint>(sortedKeys[sortedIndex] >> 32u);
    if (pairIndex >= config.contactCounts.z) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
        return;
    }
    if (sortedIndex > 0u &&
        static_cast<uint>(sortedKeys[sortedIndex - 1u] >> 32u) == pairIndex) {
        return;
    }
    NumiClothBagGPUSelfImpulse aggregate = {};
    aggregate.identity.x = pairIndex;
    for (uint cursor = sortedIndex; cursor < count; ++cursor) {
        const ulong key = sortedKeys[cursor];
        if (static_cast<uint>(key >> 32u) != pairIndex) {
            break;
        }
        const uint recordIndex = static_cast<uint>(key);
        if (recordIndex >= count) {
            recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
            return;
        }
        const NumiClothBagGPUSelfImpulse source = records[recordIndex];
        aggregate.identity.yz = source.identity.yz;
        aggregate.identity.w += 1u;
        aggregate.normalAndImpulse += source.normalAndImpulse;
        aggregate.endpointImpulses += source.endpointImpulses;
    }
    aggregates[sortedIndex] = aggregate;
    pairToAggregate[pairIndex] = sortedIndex + 1u;
}

kernel void numi_cloth_bag_count_self_friction_batches(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device const NumiClothBagGPUBatch* batches [[buffer(1)]],
    device const uint* pairToAggregate [[buffer(2)]],
    device uint* activeBatchCounts [[buffer(3)]],
    device atomic_uint* failure [[buffer(4)]],
    const uint batchIndex [[thread_position_in_grid]]
) {
    if (!validConfig(config, failure) ||
        batchIndex >= config.contactCounts.w) {
        return;
    }
    const NumiClothBagGPUBatch batch = batches[batchIndex];
    if (batch.control.x + batch.control.y > config.contactCounts.z) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
        activeBatchCounts[batchIndex] = 0u;
        return;
    }
    uint active = 0u;
    for (uint local = 0u; local < batch.control.y; ++local) {
        active += pairToAggregate[batch.control.x + local] != 0u;
    }
    activeBatchCounts[batchIndex] = active;
}

kernel void numi_cloth_bag_apply_self_friction(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device NumiClothBagGPUParticle* particles [[buffer(1)]],
    device const NumiClothBagGPUDistance* distances [[buffer(2)]],
    device const NumiClothBagGPUSelfPair* pairs [[buffer(3)]],
    device const NumiClothBagGPUBatch* batches [[buffer(4)]],
    device const uint* pairToAggregate [[buffer(5)]],
    device const NumiClothBagGPUSelfImpulse* aggregates [[buffer(6)]],
    device const uint* activeBatchIndices [[buffer(7)]],
    device const uint* activeBatchCount [[buffer(8)]],
    device atomic_uint* status [[buffer(9)]],
    device atomic_uint* failure [[buffer(10)]],
    const uint localIndex [[thread_position_in_threadgroup]],
    const uint groupIndex [[threadgroup_position_in_grid]]
) {
    if (!validConfig(config, failure) || groupIndex != 0u) {
        if (localIndex == 0u) {
            recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
        }
        return;
    }
    const float friction = config.clothMaterial.w;
    for (uint activeBatch = 0u;
         activeBatch < *activeBatchCount;
         ++activeBatch) {
        const uint batchIndex = activeBatchIndices[activeBatch];
        const NumiClothBagGPUBatch batch = batches[batchIndex];
        const uint pairIndex = batch.control.x + localIndex;
        if (localIndex < batch.control.y) {
            const uint aggregateSlot = pairToAggregate[pairIndex];
            if (aggregateSlot != 0u) {
                const NumiClothBagGPUSelfPair pair = pairs[pairIndex];
                const NumiClothBagGPUSelfImpulse contact =
                    aggregates[aggregateSlot - 1u];
                if (pair.firstSegment >= config.control.z ||
                    pair.secondSegment >= config.control.z ||
                    contact.identity.x != pairIndex) {
                    recordFailure(
                        failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE
                    );
                } else {
                    const uint4 first = distances[
                        pair.firstSegment
                    ].particlesAndColor;
                    const uint4 second = distances[
                        pair.secondSegment
                    ].particlesAndColor;
                    const float normalImpulse =
                        contact.normalAndImpulse.w;
                    const float normalLength = length(
                        contact.normalAndImpulse.xyz
                    );
                    const float firstSum =
                        contact.endpointImpulses.x +
                        contact.endpointImpulses.y;
                    const float secondSum =
                        contact.endpointImpulses.z +
                        contact.endpointImpulses.w;
                    if (normalImpulse > 0.0f && normalLength > 1.0e-10f &&
                        firstSum > 1.0e-12f && secondSum > 1.0e-12f) {
                        const float3 normal =
                            contact.normalAndImpulse.xyz / normalLength;
                        const float2 firstWeights =
                            contact.endpointImpulses.xy / firstSum;
                        const float2 secondWeights =
                            contact.endpointImpulses.zw / secondSum;
                        NumiClothBagGPUParticle firstStart =
                            particles[first.x];
                        NumiClothBagGPUParticle firstEnd =
                            particles[first.y];
                        NumiClothBagGPUParticle secondStart =
                            particles[second.x];
                        NumiClothBagGPUParticle secondEnd =
                            particles[second.y];
                        const float3 firstVelocity =
                            firstStart.velocity.xyz * firstWeights.x +
                            firstEnd.velocity.xyz * firstWeights.y;
                        const float3 secondVelocity =
                            secondStart.velocity.xyz * secondWeights.x +
                            secondEnd.velocity.xyz * secondWeights.y;
                        const float3 relativeVelocity =
                            firstVelocity - secondVelocity;
                        const float3 tangentVelocity = relativeVelocity -
                            normal * dot(relativeVelocity, normal);
                        const float slipSpeed = length(tangentVelocity);
                        if (slipSpeed > 1.0e-10f) {
                            const float3 tangent =
                                tangentVelocity / slipSpeed;
                            const float denominator =
                                firstStart.positionAndInverseMass.w *
                                    firstWeights.x * firstWeights.x +
                                firstEnd.positionAndInverseMass.w *
                                    firstWeights.y * firstWeights.y +
                                secondStart.positionAndInverseMass.w *
                                    secondWeights.x * secondWeights.x +
                                secondEnd.positionAndInverseMass.w *
                                    secondWeights.y * secondWeights.y;
                            if (denominator > 0.0f) {
                                const float frictionLimit =
                                    friction * normalImpulse;
                                const float tangentialImpulse = min(
                                    slipSpeed / denominator,
                                    frictionLimit
                                );
                                if (tangentialImpulse > 0.0f) {
                                    const float3 impulseOnFirst =
                                        tangent * -tangentialImpulse;
                                    firstStart.velocity.xyz +=
                                        impulseOnFirst *
                                        (firstStart.positionAndInverseMass.w *
                                         firstWeights.x);
                                    firstEnd.velocity.xyz +=
                                        impulseOnFirst *
                                        (firstEnd.positionAndInverseMass.w *
                                         firstWeights.y);
                                    secondStart.velocity.xyz -=
                                        impulseOnFirst *
                                        (secondStart.positionAndInverseMass.w *
                                         secondWeights.x);
                                    secondEnd.velocity.xyz -=
                                        impulseOnFirst *
                                        (secondEnd.positionAndInverseMass.w *
                                         secondWeights.y);
                                    particles[first.x] = firstStart;
                                    particles[first.y] = firstEnd;
                                    particles[second.x] = secondStart;
                                    particles[second.y] = secondEnd;
                                    atomic_fetch_add_explicit(
                                        status + 7u,
                                        1u,
                                        memory_order_relaxed
                                    );
                                    if (frictionLimit > 0.0f) {
                                        atomic_fetch_max_explicit(
                                            status + 4u,
                                            as_type<uint>(
                                                tangentialImpulse /
                                                frictionLimit
                                            ),
                                            memory_order_relaxed
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
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
    const float predictedVerticalVelocity = particle.velocity.w;
    particle.velocity.xyz =
        (particle.positionAndInverseMass.xyz - particle.previousAndMass.xyz) *
        inverseTimestep;
    particle.velocity.w = 0.0f;
    if (config.constraintCounts.z != 0u &&
        particle.positionAndInverseMass.w > 0.0f &&
        particle.positionAndInverseMass.z <=
            config.clothMaterial.x + 1.0e-6f) {
        particle.velocity.w = max(
            0.0f,
            (particle.velocity.z - predictedVerticalVelocity) /
                particle.positionAndInverseMass.w
        );
    }
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

kernel void numi_cloth_bag_integrate_fruit_orientation(
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
    const float3 angularVelocity = fruit.angularVelocity.xyz;
    const float4 orientation = fruit.orientation;
    const float4 derivative = float4(
        orientation.w * angularVelocity +
            cross(angularVelocity, orientation.xyz),
        -dot(angularVelocity, orientation.xyz)
    );
    const float4 advanced = fma(
        derivative,
        float4(0.5f * config.gravityAndTimestep.w),
        orientation
    );
    const float magnitudeSquared = dot(advanced, advanced);
    if (!all(isfinite(advanced)) || !(magnitudeSquared > 1.0e-20f)) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
        return;
    }
    fruit.orientation = advanced * rsqrt(magnitudeSquared);
    fruits[index] = fruit;
}

kernel void numi_cloth_bag_apply_cloth_ground_friction(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device NumiClothBagGPUParticle* particles [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    device atomic_uint* failure [[buffer(3)]],
    const uint index [[thread_position_in_grid]]
) {
    if (!validConfig(config, failure) || index >= config.control.y ||
        config.constraintCounts.z == 0u) {
        return;
    }
    NumiClothBagGPUParticle particle = particles[index];
    const float inverseMass = particle.positionAndInverseMass.w;
    const float normalImpulse = particle.velocity.w;
    const float friction = config.clothMaterial.z;
    if (!(friction >= 0.0f) || !isfinite(friction)) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
        return;
    }
    if (!(normalImpulse > 0.0f) || !(inverseMass > 0.0f)) {
        return;
    }
    const float3 tangentVelocity = float3(
        particle.velocity.x, particle.velocity.y, 0.0f
    );
    const float slipSpeed = length(tangentVelocity);
    if (!(slipSpeed > 1.0e-10f)) {
        return;
    }
    const float frictionLimit = friction * normalImpulse;
    const float tangentialImpulse = min(
        slipSpeed / inverseMass, frictionLimit
    );
    if (!(tangentialImpulse > 0.0f)) {
        return;
    }
    particle.velocity.xyz -= tangentVelocity *
        (tangentialImpulse * inverseMass / slipSpeed);
    if (!all(isfinite(particle.velocity))) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
        return;
    }
    particles[index] = particle;
    recordFrictionStatus(
        status, 2u, tangentialImpulse, frictionLimit
    );
}

kernel void numi_cloth_bag_apply_yarn_friction(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device NumiClothBagGPUParticle* particles [[buffer(1)]],
    device const NumiClothBagGPUDistance* distances [[buffer(2)]],
    device NumiClothBagGPUFruit* fruits [[buffer(3)]],
    device const NumiClothBagGPUYarnContact* contacts [[buffer(4)]],
    constant NumiClothBagGPUBatch& batch [[buffer(5)]],
    device atomic_uint* status [[buffer(6)]],
    device atomic_uint* failure [[buffer(7)]],
    const uint fruitIndex [[thread_position_in_threadgroup]]
) {
    if (!validConfig(config, failure) ||
        config.contactCounts.y !=
            config.control.z * config.constraintCounts.w ||
        batch.control.x + batch.control.y > config.control.z ||
        batch.control.y == 0u ||
        fruitIndex >= config.constraintCounts.w) {
        if (fruitIndex < config.constraintCounts.w) {
            recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_RANGE);
        }
        return;
    }
    const float friction = config.fruitMaterial.w;
    if (!(friction >= 0.0f) || !isfinite(friction)) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
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
        const NumiClothBagGPUYarnContact contact = contacts[contactIndex];
        const float normalImpulse = contact.fruitNormalAndImpulse.w;
        const float normalLength = length(
            contact.fruitNormalAndImpulse.xyz
        );
        if (normalImpulse > 0.0f && normalLength > 1.0e-10f) {
            const float3 normal =
                contact.fruitNormalAndImpulse.xyz / normalLength;
            float2 weights = contact.segmentImpulse.xy / normalImpulse;
            const float weightSum = weights.x + weights.y;
            if (weightSum > 1.0e-12f) {
                weights /= weightSum;
                NumiClothBagGPUParticle first = particles[firstIndex];
                NumiClothBagGPUParticle second = particles[secondIndex];
                const float3 yarnVelocity =
                    first.velocity.xyz * weights.x +
                    second.velocity.xyz * weights.y;
                const float3 ballOffset =
                    normal * -fruit.previousAndRadius.w;
                const float3 ballContactVelocity =
                    fruit.velocityAndGroundImpulse.xyz +
                    cross(fruit.angularVelocity.xyz, ballOffset);
                const float3 relativeVelocity =
                    ballContactVelocity - yarnVelocity;
                const float3 tangentVelocity = relativeVelocity -
                    normal * dot(relativeVelocity, normal);
                const float slipSpeed = length(tangentVelocity);
                if (slipSpeed > 1.0e-10f) {
                    const float3 tangent = tangentVelocity / slipSpeed;
                    const float3 ballLever = cross(ballOffset, tangent);
                    float denominator = fruit.positionAndInverseMass.w +
                        fruitInverseInertia(fruit) *
                            dot(ballLever, ballLever);
                    float3 firstResponse =
                        tangent * first.positionAndInverseMass.w;
                    float3 secondResponse =
                        tangent * second.positionAndInverseMass.w;
                    if (config.constraintCounts.z != 0u) {
                        if (first.positionAndInverseMass.z <=
                                config.clothMaterial.x + 1.0e-6f &&
                            firstResponse.z < 0.0f) {
                            firstResponse.z = 0.0f;
                        }
                        if (second.positionAndInverseMass.z <=
                                config.clothMaterial.x + 1.0e-6f &&
                            secondResponse.z < 0.0f) {
                            secondResponse.z = 0.0f;
                        }
                    }
                    denominator +=
                        dot(tangent, firstResponse) * weights.x * weights.x +
                        dot(tangent, secondResponse) * weights.y * weights.y;
                    if (denominator > 0.0f && isfinite(denominator)) {
                        const float frictionLimit =
                            friction * normalImpulse;
                        const float tangentialImpulse = min(
                            slipSpeed / denominator, frictionLimit
                        );
                        if (tangentialImpulse > 0.0f) {
                            applyFruitImpulse(
                                fruit,
                                tangent * -tangentialImpulse,
                                ballOffset
                            );
                            first.velocity.xyz += firstResponse *
                                (tangentialImpulse * weights.x);
                            second.velocity.xyz += secondResponse *
                                (tangentialImpulse * weights.y);
                            particles[firstIndex] = first;
                            particles[secondIndex] = second;
                            recordFrictionStatus(
                                status,
                                1u,
                                tangentialImpulse,
                                frictionLimit
                            );
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
    if (!all(isfinite(fruit.velocityAndGroundImpulse)) ||
        !all(isfinite(fruit.angularVelocity))) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
        return;
    }
    fruits[fruitIndex] = fruit;
}

kernel void numi_cloth_bag_apply_fruit_pair_friction(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device NumiClothBagGPUFruit* fruits [[buffer(1)]],
    device const NumiClothBagGPUFruitPair* pairs [[buffer(2)]],
    constant NumiClothBagGPUBatch& batch [[buffer(3)]],
    device atomic_uint* status [[buffer(4)]],
    device atomic_uint* failure [[buffer(5)]],
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
    const NumiClothBagGPUFruitPair pair = pairs[pairIndex];
    if (pair.fruitsAndColor.z != batch.control.z ||
        pair.fruitsAndColor.x >= config.constraintCounts.w ||
        pair.fruitsAndColor.y >= config.constraintCounts.w ||
        pair.fruitsAndColor.x == pair.fruitsAndColor.y) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_BATCH);
        return;
    }
    const float friction = config.fruitMaterial.x;
    const float normalImpulse = pair.contact.w;
    const float normalLength = length(pair.contact.xyz);
    if (!(friction >= 0.0f) || !isfinite(friction)) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
        return;
    }
    if (!(normalImpulse > 0.0f) || !(normalLength > 1.0e-10f)) {
        return;
    }
    NumiClothBagGPUFruit first = fruits[pair.fruitsAndColor.x];
    NumiClothBagGPUFruit second = fruits[pair.fruitsAndColor.y];
    const float3 normal = pair.contact.xyz / normalLength;
    const float3 firstOffset = normal * first.previousAndRadius.w;
    const float3 secondOffset = normal * -second.previousAndRadius.w;
    const float3 firstContactVelocity =
        first.velocityAndGroundImpulse.xyz +
        cross(first.angularVelocity.xyz, firstOffset);
    const float3 secondContactVelocity =
        second.velocityAndGroundImpulse.xyz +
        cross(second.angularVelocity.xyz, secondOffset);
    const float3 relativeVelocity =
        secondContactVelocity - firstContactVelocity;
    const float3 tangentVelocity = relativeVelocity -
        normal * dot(relativeVelocity, normal);
    const float slipSpeed = length(tangentVelocity);
    if (!(slipSpeed > 1.0e-10f)) {
        return;
    }
    const float3 tangent = tangentVelocity / slipSpeed;
    const float3 firstLever = cross(firstOffset, tangent);
    const float3 secondLever = cross(secondOffset, tangent);
    const float denominator =
        first.positionAndInverseMass.w + second.positionAndInverseMass.w +
        fruitInverseInertia(first) *
            dot(firstLever, firstLever) +
        fruitInverseInertia(second) *
            dot(secondLever, secondLever);
    if (!(denominator > 0.0f) || !isfinite(denominator)) {
        return;
    }
    const float frictionLimit = friction * normalImpulse;
    const float tangentialImpulse = min(
        slipSpeed / denominator, frictionLimit
    );
    if (!(tangentialImpulse > 0.0f)) {
        return;
    }
    const float3 impulseOnSecond = tangent * -tangentialImpulse;
    applyFruitImpulse(second, impulseOnSecond, secondOffset);
    applyFruitImpulse(first, -impulseOnSecond, firstOffset);
    if (!all(isfinite(first.velocityAndGroundImpulse)) ||
        !all(isfinite(second.velocityAndGroundImpulse)) ||
        !all(isfinite(first.angularVelocity)) ||
        !all(isfinite(second.angularVelocity))) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
        return;
    }
    fruits[pair.fruitsAndColor.x] = first;
    fruits[pair.fruitsAndColor.y] = second;
    recordFrictionStatus(
        status, 0u, tangentialImpulse, frictionLimit
    );
}

kernel void numi_cloth_bag_apply_fruit_ground_friction(
    constant NumiClothBagGPUConfig& config [[buffer(0)]],
    device NumiClothBagGPUFruit* fruits [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    device atomic_uint* failure [[buffer(3)]],
    const uint index [[thread_position_in_grid]]
) {
    if (!validConfig(config, failure) ||
        index >= config.constraintCounts.w ||
        config.constraintCounts.z == 0u) {
        return;
    }
    NumiClothBagGPUFruit fruit = fruits[index];
    const float normalImpulse = fruit.velocityAndGroundImpulse.w;
    const float friction = config.fruitMaterial.y;
    const float rollingResistance = config.fruitMaterial.z;
    if (!(friction >= 0.0f) || !(rollingResistance >= 0.0f) ||
        !isfinite(friction) || !isfinite(rollingResistance)) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
        return;
    }
    if (!(normalImpulse > 0.0f)) {
        return;
    }
    const float3 normal = float3(0.0f, 0.0f, 1.0f);
    const float3 contactOffset =
        float3(0.0f, 0.0f, -fruit.previousAndRadius.w);
    const float3 contactVelocity = fruit.velocityAndGroundImpulse.xyz +
        cross(fruit.angularVelocity.xyz, contactOffset);
    const float3 tangentVelocity = contactVelocity -
        normal * dot(contactVelocity, normal);
    const float slipSpeed = length(tangentVelocity);
    if (slipSpeed > 1.0e-10f) {
        const float3 tangent = tangentVelocity / slipSpeed;
        const float3 lever = cross(contactOffset, tangent);
        const float denominator = fruit.positionAndInverseMass.w +
            fruitInverseInertia(fruit) *
                dot(lever, lever);
        const float frictionLimit = friction * normalImpulse;
        const float tangentialImpulse = min(
            slipSpeed / denominator, frictionLimit
        );
        if (tangentialImpulse > 0.0f) {
            applyFruitImpulse(
                fruit,
                tangent * -tangentialImpulse,
                contactOffset
            );
            recordFrictionStatus(
                status, 3u, tangentialImpulse, frictionLimit
            );
        }
    }
    const float3 rollingAngularVelocity = float3(
        fruit.angularVelocity.x, fruit.angularVelocity.y, 0.0f
    );
    const float rollingSpeed = length(rollingAngularVelocity);
    if (rollingSpeed > 1.0e-12f && rollingResistance > 0.0f) {
        const float inverseInertia = fruitInverseInertia(fruit);
        const float requiredAngularImpulse =
            rollingSpeed / inverseInertia;
        const float rollingImpulseLimit = rollingResistance *
            fruit.previousAndRadius.w * normalImpulse;
        const float angularImpulse = min(
            requiredAngularImpulse, rollingImpulseLimit
        );
        fruit.angularVelocity.xyz -= rollingAngularVelocity *
            (angularImpulse * inverseInertia / rollingSpeed);
        atomic_fetch_add_explicit(
            status + 6u, 1u, memory_order_relaxed
        );
        if (rollingImpulseLimit > 0.0f) {
            atomic_fetch_max_explicit(
                status + 5u,
                as_type<uint>(angularImpulse / rollingImpulseLimit),
                memory_order_relaxed
            );
        }
    }
    if (!all(isfinite(fruit.velocityAndGroundImpulse)) ||
        !all(isfinite(fruit.angularVelocity))) {
        recordFailure(failure, NUMI_CLOTH_BAG_GPU_FAILURE_NONFINITE);
        return;
    }
    fruits[index] = fruit;
}
