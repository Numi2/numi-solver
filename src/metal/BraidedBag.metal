#include <metal_stdlib>

#include "numi/braided_bag.h"
#include "numi/temporal_cone_island.h"

using namespace metal;

namespace {

inline bool bagFinite3(const float3 value) {
    return all(isfinite(value));
}

inline float3 bagContactAxis(
    threadgroup const NumiBraidedBagContact& contact,
    const uint axis
) {
    return axis == 0u
        ? contact.normalAndSeparation.xyz
        : axis == 1u
        ? contact.tangentU.xyz
        : contact.tangentV.xyz;
}

inline float bagParticleInverseMass(
    device const NumiBraidedBagNode* nodes,
    device const NumiBraidedBagBall* balls,
    const uint environment,
    const uint particle
) {
    if (particle < NUMI_BRAIDED_BAG_NODE_COUNT) {
        return nodes[
            environment * NUMI_BRAIDED_BAG_NODE_COUNT + particle
        ].positionAndInverseMass.w;
    }
    const uint ball = particle - NUMI_BRAIDED_BAG_NODE_COUNT;
    return ball < NUMI_BRAIDED_BAG_BALL_COUNT
        ? balls[
            environment * NUMI_BRAIDED_BAG_BALL_COUNT + ball
        ].positionAndInverseMass.w
        : 0.0f;
}

inline float3 bagParticleCandidateVelocity(
    device const float4* nodeVelocities,
    device const float4* ballVelocities,
    const uint environment,
    const uint particle
) {
    if (particle < NUMI_BRAIDED_BAG_NODE_COUNT) {
        return nodeVelocities[
            environment * NUMI_BRAIDED_BAG_NODE_COUNT + particle
        ].xyz;
    }
    const uint ball = particle - NUMI_BRAIDED_BAG_NODE_COUNT;
    return ball < NUMI_BRAIDED_BAG_BALL_COUNT
        ? ballVelocities[
            environment * NUMI_BRAIDED_BAG_BALL_COUNT + ball
        ].xyz
        : float3(0.0f);
}

inline float3 bagRelativeVelocity(
    threadgroup const NumiBraidedBagContact& contact,
    device const float4* nodeVelocities,
    device const float4* ballVelocities,
    const uint environment
) {
    float3 velocity = float3(0.0f);
    const uint count = uint(contact.weights.w);
    for (uint participant = 0u; participant < count; ++participant) {
        velocity += contact.weights[participant] *
            bagParticleCandidateVelocity(
                nodeVelocities,
                ballVelocities,
                environment,
                contact.particles[participant]
            );
    }
    return velocity;
}

inline void bagContactFrame(
    const float3 authoredNormal,
    thread float3& normal,
    thread float3& tangentU,
    thread float3& tangentV
) {
    const float magnitude = length(authoredNormal);
    normal = magnitude > 1.0e-8f && isfinite(magnitude)
        ? authoredNormal / magnitude
        : float3(0.0f, 0.0f, 1.0f);
    const float3 reference = abs(normal.z) < 0.875f
        ? float3(0.0f, 0.0f, 1.0f)
        : float3(1.0f, 0.0f, 0.0f);
    tangentU = normalize(cross(reference, normal));
    tangentV = cross(normal, tangentU);
}

inline float bagContactCoefficient(
    threadgroup const NumiBraidedBagContact& target,
    threadgroup const NumiBraidedBagContact& source,
    device const NumiBraidedBagNode* nodes,
    device const NumiBraidedBagBall* balls,
    const uint environment,
    const uint targetAxis,
    const uint sourceAxis
) {
    const float3 targetDirection = bagContactAxis(target, targetAxis);
    const float3 sourceDirection = bagContactAxis(source, sourceAxis);
    float coefficient = 0.0f;
    const uint targetCount = uint(target.weights.w);
    const uint sourceCount = uint(source.weights.w);
    for (uint targetParticipant = 0u;
         targetParticipant < targetCount;
         ++targetParticipant) {
        const uint particle = target.particles[targetParticipant];
        for (uint sourceParticipant = 0u;
             sourceParticipant < sourceCount;
             ++sourceParticipant) {
            if (particle != source.particles[sourceParticipant]) {
                continue;
            }
            coefficient = fma(
                target.weights[targetParticipant] *
                    source.weights[sourceParticipant] *
                    bagParticleInverseMass(
                        nodes,
                        balls,
                        environment,
                        particle
                    ),
                dot(targetDirection, sourceDirection),
                coefficient
            );
        }
    }
    return coefficient;
}

inline float3 bagImpulseWorld(
    threadgroup const NumiBraidedBagContact& contact,
    const float3 impulse
) {
    return contact.normalAndSeparation.xyz * impulse.x +
        contact.tangentU.xyz * impulse.y +
        contact.tangentV.xyz * impulse.z;
}

inline void bagPairBalls(
    const uint pair,
    thread uint& first,
    thread uint& second
) {
    uint cursor = 0u;
    first = 0u;
    second = 1u;
    for (uint candidateFirst = 0u;
         candidateFirst < NUMI_BRAIDED_BAG_BALL_COUNT;
         ++candidateFirst) {
        for (uint candidateSecond = candidateFirst + 1u;
             candidateSecond < NUMI_BRAIDED_BAG_BALL_COUNT;
             ++candidateSecond) {
            if (cursor == pair) {
                first = candidateFirst;
                second = candidateSecond;
                return;
            }
            ++cursor;
        }
    }
}

inline bool bagContactsShareParticle(
    threadgroup const NumiBraidedBagContact& target,
    threadgroup const NumiBraidedBagContact& source
) {
    const uint targetCount = uint(target.weights.w);
    const uint sourceCount = uint(source.weights.w);
    for (uint targetParticipant = 0u;
         targetParticipant < targetCount;
         ++targetParticipant) {
        for (uint sourceParticipant = 0u;
             sourceParticipant < sourceCount;
             ++sourceParticipant) {
            if (target.particles[targetParticipant] ==
                source.particles[sourceParticipant]) {
                return true;
            }
        }
    }
    return false;
}

} // namespace

kernel void numi_braided_bag_free_motion(
    constant NumiBraidedBagConfig& config [[buffer(0)]],
    device const NumiBraidedBagNode* nodes [[buffer(1)]],
    device const NumiBraidedBagBall* balls [[buffer(2)]],
    device const NumiBraidedBagEdge* edges [[buffer(3)]],
    device float4* candidateNodeVelocities [[buffer(4)]],
    device float4* candidateBallVelocities [[buffer(5)]],
    const uint index [[thread_position_in_grid]]
) {
    const uint totalParticles =
        config.control.y * NUMI_BRAIDED_BAG_PARTICLE_COUNT;
    if (index >= totalParticles) {
        return;
    }
    const uint environment = index / NUMI_BRAIDED_BAG_PARTICLE_COUNT;
    const uint particle = index -
        environment * NUMI_BRAIDED_BAG_PARTICLE_COUNT;
    const float timestep = config.timing.x;
    if (particle < NUMI_BRAIDED_BAG_NODE_COUNT) {
        const uint nodeBase =
            environment * NUMI_BRAIDED_BAG_NODE_COUNT;
        const NumiBraidedBagNode state = nodes[nodeBase + particle];
        const float inverseMass = state.positionAndInverseMass.w;
        if (!(inverseMass > 0.0f)) {
            candidateNodeVelocities[nodeBase + particle] = float4(0.0f);
            return;
        }
        float3 force = float3(0.0f);
        for (uint edge = 0u;
             edge < NUMI_BRAIDED_BAG_EDGE_COUNT;
             ++edge) {
            const NumiBraidedBagEdge link = edges[edge];
            uint other = NUMI_BRAIDED_BAG_INVALID_PARTICLE;
            if (link.nodes.x == particle) {
                other = link.nodes.y;
            } else if (link.nodes.y == particle) {
                other = link.nodes.x;
            }
            if (other == NUMI_BRAIDED_BAG_INVALID_PARTICLE) {
                continue;
            }
            const NumiBraidedBagNode otherState = nodes[nodeBase + other];
            const float3 delta =
                otherState.positionAndInverseMass.xyz -
                state.positionAndInverseMass.xyz;
            const float distance = length(delta);
            if (!(distance > 1.0e-8f) || !isfinite(distance)) {
                continue;
            }
            const float3 direction = delta / distance;
            const float relativeSpeed = dot(
                otherState.velocity.xyz - state.velocity.xyz,
                direction
            );
            const float magnitude =
                config.braidMaterial.x * (distance - link.rest.x) +
                config.braidMaterial.y * relativeSpeed;
            force = fma(direction, magnitude, force);
        }
        const float3 acceleration = float3(
            0.0f,
            0.0f,
            config.timing.y
        ) + inverseMass * force - config.timing.z * state.velocity.xyz;
        candidateNodeVelocities[nodeBase + particle] = float4(
            state.velocity.xyz + timestep * acceleration,
            0.0f
        );
        return;
    }

    const uint ball = particle - NUMI_BRAIDED_BAG_NODE_COUNT;
    const uint ballIndex =
        environment * NUMI_BRAIDED_BAG_BALL_COUNT + ball;
    const NumiBraidedBagBall state = balls[ballIndex];
    const float3 acceleration = float3(
        0.0f,
        0.0f,
        config.timing.y
    ) - config.timing.w * state.velocityAndRadius.xyz;
    candidateBallVelocities[ballIndex] = float4(
        state.velocityAndRadius.xyz + timestep * acceleration,
        0.0f
    );
}

kernel void numi_braided_bag_build_contacts(
    constant NumiBraidedBagConfig& config [[buffer(0)]],
    device const NumiBraidedBagNode* nodes [[buffer(1)]],
    device const NumiBraidedBagBall* balls [[buffer(2)]],
    device const NumiBraidedBagEdge* edges [[buffer(3)]],
    device const float4* candidateNodeVelocities [[buffer(4)]],
    device const float4* candidateBallVelocities [[buffer(5)]],
    device const float4* warmImpulses [[buffer(6)]],
    device const uint* warmIdentities [[buffer(7)]],
    device NumiBraidedBagContact* outputGeometry [[buffer(8)]],
    device NumiTemporalConeIslandContact* outputContacts [[buffer(9)]],
    device NumiTemporalConeIslandHeader* outputDenseHeaders [[buffer(10)]],
    device float* outputDenseMatrices [[buffer(11)]],
    device NumiTemporalConeStreamHeader* outputStreamHeaders [[buffer(12)]],
    device float* outputStreamBlocks [[buffer(13)]],
    constant uint& path [[buffer(14)]],
    device uint* streamRowOffsets [[buffer(15)]],
    device float* outputMaximumRowSum [[buffer(16)]],
    device uint* outputStreamColumnIndices [[buffer(17)]],
    device uint* outputActiveBlockCounts [[buffer(18)]],
    const uint environment [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    if (environment >= config.control.y) {
        return;
    }
    threadgroup NumiBraidedBagContact candidates[
        NUMI_BRAIDED_BAG_CONTACT_COUNT
    ];
    threadgroup NumiBraidedBagContact contacts[
        NUMI_BRAIDED_BAG_CONTACT_COUNT
    ];
    const uint nodeBase =
        environment * NUMI_BRAIDED_BAG_NODE_COUNT;
    const uint ballBase =
        environment * NUMI_BRAIDED_BAG_BALL_COUNT;
    const uint compactContactBase =
        environment * NUMI_BRAIDED_BAG_CONTACT_COUNT;
    const uint solverContactBase =
        environment * NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS;
    if (lane < NUMI_BRAIDED_BAG_CONTACT_COUNT) {
        NumiBraidedBagContact contact = {};
        contact.particles = uint4(NUMI_BRAIDED_BAG_INVALID_PARTICLE);
        float3 authoredNormal = float3(0.0f, 0.0f, 1.0f);
        float separation = 0.0f;
        if (lane < NUMI_BRAIDED_BAG_BALL_COUNT) {
            const uint ball = lane;
            const NumiBraidedBagBall ballState = balls[ballBase + ball];
            float minimumDistanceSquared = INFINITY;
            uint closestEdge = 0u;
            float closestWeight = 0.0f;
            float3 closestPoint = float3(0.0f);
            for (uint edge = 0u;
                 edge < NUMI_BRAIDED_BAG_EDGE_COUNT;
                 ++edge) {
                const NumiBraidedBagEdge link = edges[edge];
                const float3 first = nodes[
                    nodeBase + link.nodes.x
                ].positionAndInverseMass.xyz;
                const float3 second = nodes[
                    nodeBase + link.nodes.y
                ].positionAndInverseMass.xyz;
                const float3 edgeVector = second - first;
                const float denominator = dot(edgeVector, edgeVector);
                const float weight = denominator > 1.0e-12f
                    ? clamp(
                        dot(
                            ballState.positionAndInverseMass.xyz - first,
                            edgeVector
                        ) / denominator,
                        0.0f,
                        1.0f
                    )
                    : 0.0f;
                const float3 point = fma(edgeVector, weight, first);
                const float3 offset =
                    ballState.positionAndInverseMass.xyz - point;
                const float distanceSquared = dot(offset, offset);
                if (distanceSquared < minimumDistanceSquared) {
                    minimumDistanceSquared = distanceSquared;
                    closestEdge = edge;
                    closestWeight = weight;
                    closestPoint = point;
                }
            }
            const NumiBraidedBagEdge link = edges[closestEdge];
            const float3 offset =
                ballState.positionAndInverseMass.xyz - closestPoint;
            const float distance = sqrt(max(minimumDistanceSquared, 0.0f));
            authoredNormal = distance > 1.0e-8f
                ? offset / distance
                : float3(0.0f, 0.0f, 1.0f);
            separation = distance - (
                ballState.velocityAndRadius.w + config.braidMaterial.z
            );
            contact.particles = uint4(
                NUMI_BRAIDED_BAG_NODE_COUNT + ball,
                link.nodes.x,
                link.nodes.y,
                NUMI_BRAIDED_BAG_INVALID_PARTICLE
            );
            contact.weights = float4(
                1.0f,
                -(1.0f - closestWeight),
                -closestWeight,
                3.0f
            );
            contact.identity = uint4(
                0u,
                ball * NUMI_BRAIDED_BAG_EDGE_COUNT + closestEdge,
                0u,
                0u
            );
        } else if (lane <
            NUMI_BRAIDED_BAG_BALL_COUNT +
                NUMI_BRAIDED_BAG_BALL_PAIR_COUNT) {
            const uint pair = lane - NUMI_BRAIDED_BAG_BALL_COUNT;
            uint firstBall;
            uint secondBall;
            bagPairBalls(pair, firstBall, secondBall);
            const NumiBraidedBagBall first = balls[ballBase + firstBall];
            const NumiBraidedBagBall second = balls[ballBase + secondBall];
            const float3 offset =
                first.positionAndInverseMass.xyz -
                second.positionAndInverseMass.xyz;
            const float distance = length(offset);
            authoredNormal = distance > 1.0e-8f
                ? offset / distance
                : float3(1.0f, 0.0f, 0.0f);
            separation = distance -
                first.velocityAndRadius.w - second.velocityAndRadius.w;
            contact.particles = uint4(
                NUMI_BRAIDED_BAG_NODE_COUNT + firstBall,
                NUMI_BRAIDED_BAG_NODE_COUNT + secondBall,
                NUMI_BRAIDED_BAG_INVALID_PARTICLE,
                NUMI_BRAIDED_BAG_INVALID_PARTICLE
            );
            contact.weights = float4(1.0f, -1.0f, 0.0f, 2.0f);
            contact.identity = uint4(1u, pair, 0u, 0u);
        } else {
            const uint ball = lane -
                NUMI_BRAIDED_BAG_BALL_COUNT -
                NUMI_BRAIDED_BAG_BALL_PAIR_COUNT;
            const NumiBraidedBagBall state = balls[ballBase + ball];
            authoredNormal = float3(0.0f, 0.0f, 1.0f);
            separation = state.positionAndInverseMass.z -
                state.velocityAndRadius.w - config.contact.z;
            contact.particles = uint4(
                NUMI_BRAIDED_BAG_NODE_COUNT + ball,
                NUMI_BRAIDED_BAG_INVALID_PARTICLE,
                NUMI_BRAIDED_BAG_INVALID_PARTICLE,
                NUMI_BRAIDED_BAG_INVALID_PARTICLE
            );
            contact.weights = float4(1.0f, 0.0f, 0.0f, 1.0f);
            contact.identity = uint4(2u, ball, 0u, 0u);
        }
        float3 normal;
        float3 tangentU;
        float3 tangentV;
        bagContactFrame(authoredNormal, normal, tangentU, tangentV);
        contact.normalAndSeparation = float4(normal, separation);
        contact.tangentU = float4(tangentU, 0.0f);
        contact.tangentV = float4(tangentV, 0.0f);
        candidates[lane] = contact;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool candidateActive = false;
    float3 relativeVelocity = float3(0.0f);
    if (lane < NUMI_BRAIDED_BAG_CONTACT_COUNT) {
        relativeVelocity = bagRelativeVelocity(
            candidates[lane],
            candidateNodeVelocities,
            candidateBallVelocities,
            environment
        );
        const float predictedSeparation =
            candidates[lane].normalAndSeparation.w +
            config.timing.x * min(
                dot(
                    relativeVelocity,
                    candidates[lane].normalAndSeparation.xyz
                ),
                0.0f
            );
        candidateActive = predictedSeparation <= 0.01f;
    }
    const uint compactIndex = simd_prefix_exclusive_sum(
        candidateActive ? 1u : 0u
    );
    const uint contactCount = simd_sum(candidateActive ? 1u : 0u);
    if (candidateActive) {
        const NumiBraidedBagContact contact = candidates[lane];
        contacts[compactIndex] = contact;
        outputGeometry[compactContactBase + compactIndex] = contact;
        const float inverseTimestep = 1.0f / config.timing.x;
        const float separation = contact.normalAndSeparation.w;
        const float bias = separation >= 0.0f
            ? separation * inverseTimestep
            : max(
                config.contact.x * separation * inverseTimestep,
                -config.contact.w
            );
        const uint identity =
            (contact.identity.x << 16u) | contact.identity.y;
        float3 warmImpulse = float3(0.0f);
        for (uint previous = 0u;
             previous < NUMI_BRAIDED_BAG_CONTACT_COUNT;
             ++previous) {
            if (warmIdentities[compactContactBase + previous] == identity) {
                warmImpulse = warmImpulses[
                    compactContactBase + previous
                ].xyz;
                break;
            }
        }
        NumiTemporalConeIslandContact solverContact = {};
        solverContact.freeVelocityAndFrictionU = float4(
            dot(relativeVelocity, contact.normalAndSeparation.xyz) + bias,
            dot(relativeVelocity, contact.tangentU.xyz),
            dot(relativeVelocity, contact.tangentV.xyz),
            config.braidMaterial.w
        );
        solverContact.warmImpulseAndFrictionV = float4(
            warmImpulse,
            config.braidMaterial.w
        );
        solverContact.limits = float4(0.0f);
        outputContacts[solverContactBase + compactIndex] = solverContact;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint rowBlockCount = 0u;
    if (lane < contactCount) {
        for (uint source = 0u; source < contactCount; ++source) {
            if (lane == source ||
                bagContactsShareParticle(contacts[lane], contacts[source])) {
                ++rowBlockCount;
            }
        }
    }
    const uint rowBlockBegin = simd_prefix_exclusive_sum(rowBlockCount);
    const uint blockCount = simd_sum(rowBlockCount);
    const uint rowOffsetBase =
        environment * (NUMI_BRAIDED_BAG_CONTACT_COUNT + 1u);
    const uint blockBase =
        environment * NUMI_BRAIDED_BAG_STREAM_BLOCK_COUNT;
    if (lane < contactCount) {
        streamRowOffsets[rowOffsetBase + lane] = rowBlockBegin;
        uint rowBlock = rowBlockBegin;
        for (uint source = 0u; source < contactCount; ++source) {
            if (lane == source ||
                bagContactsShareParticle(contacts[lane], contacts[source])) {
                outputStreamColumnIndices[blockBase + rowBlock] = source;
                ++rowBlock;
            }
        }
    }
    if (lane == 0u) {
        streamRowOffsets[rowOffsetBase + contactCount] = blockCount;
        outputActiveBlockCounts[environment] = blockCount;
        NumiTemporalConeIslandHeader denseHeader = {};
        denseHeader.control = uint4(
            NUMI_TEMPORAL_CONE_ISLAND_ABI_VERSION,
            contactCount,
            config.control.z,
            config.control.w
        );
        denseHeader.tolerances = config.solver;
        outputDenseHeaders[environment] = denseHeader;

        NumiTemporalConeStreamHeader streamHeader = {};
        streamHeader.control = uint4(
            NUMI_TEMPORAL_CONE_STREAM_ABI_VERSION,
            contactCount,
            config.control.z,
            config.control.w
        );
        streamHeader.ranges = uint4(
            solverContactBase,
            rowOffsetBase,
            blockBase,
            blockCount
        );
        streamHeader.tolerances = config.solver;
        outputStreamHeaders[environment] = streamHeader;
    }

    float laneMaximumRowSum = 0.0f;
    if (path == NUMI_BRAIDED_BAG_PATH_DENSE) {
        const uint matrixBase =
            environment * NUMI_TEMPORAL_CONE_ISLAND_MATRIX_ELEMENTS;
        for (uint element = lane;
             element < NUMI_TEMPORAL_CONE_ISLAND_MATRIX_ELEMENTS;
             element += 32u) {
            outputDenseMatrices[matrixBase + element] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_device);
        if (lane < contactCount) {
            float3 rowSums = float3(0.0f);
            for (uint source = 0u; source < contactCount; ++source) {
                for (uint targetAxis = 0u;
                     targetAxis < 3u;
                     ++targetAxis) {
                    for (uint sourceAxis = 0u;
                         sourceAxis < 3u;
                         ++sourceAxis) {
                        float coefficient = bagContactCoefficient(
                            contacts[lane],
                            contacts[source],
                            nodes,
                            balls,
                            environment,
                            targetAxis,
                            sourceAxis
                        );
                        if (lane == source && targetAxis == sourceAxis) {
                            coefficient += config.contact.y;
                        }
                        rowSums[targetAxis] += abs(coefficient);
                        outputDenseMatrices[
                            matrixBase +
                            (3u * lane + targetAxis) *
                                NUMI_TEMPORAL_CONE_ISLAND_MAX_ROWS +
                            3u * source + sourceAxis
                        ] = coefficient;
                    }
                }
            }
            laneMaximumRowSum = max(
                rowSums.x,
                max(rowSums.y, rowSums.z)
            );
        }
    } else if (lane < contactCount) {
        const uint rowBegin = streamRowOffsets[rowOffsetBase + lane];
        const uint rowEnd = streamRowOffsets[rowOffsetBase + lane + 1u];
        float3 rowSums = float3(0.0f);
        for (uint rowBlock = rowBegin; rowBlock < rowEnd; ++rowBlock) {
            const uint source = outputStreamColumnIndices[
                blockBase + rowBlock
            ];
            const uint valueBase =
                (blockBase + rowBlock) * 9u;
            for (uint targetAxis = 0u;
                 targetAxis < 3u;
                 ++targetAxis) {
                for (uint sourceAxis = 0u;
                     sourceAxis < 3u;
                     ++sourceAxis) {
                    float coefficient = bagContactCoefficient(
                        contacts[lane],
                        contacts[source],
                        nodes,
                        balls,
                        environment,
                        targetAxis,
                        sourceAxis
                    );
                    if (lane == source && targetAxis == sourceAxis) {
                        coefficient += config.contact.y;
                    }
                    rowSums[targetAxis] += abs(coefficient);
                    outputStreamBlocks[
                        valueBase + 3u * targetAxis + sourceAxis
                    ] = coefficient;
                }
            }
        }
        laneMaximumRowSum = max(
            rowSums.x,
            max(rowSums.y, rowSums.z)
        );
    }
    const float maximumRowSum = simd_max(laneMaximumRowSum);
    if (lane == 0u) {
        outputMaximumRowSum[environment] = maximumRowSum;
    }
}

kernel void numi_braided_bag_apply(
    constant NumiBraidedBagConfig& config [[buffer(0)]],
    device NumiBraidedBagNode* nodes [[buffer(1)]],
    device NumiBraidedBagBall* balls [[buffer(2)]],
    device const NumiBraidedBagEdge* edges [[buffer(3)]],
    device const float4* candidateNodeVelocities [[buffer(4)]],
    device const float4* candidateBallVelocities [[buffer(5)]],
    device const NumiBraidedBagContact* geometry [[buffer(6)]],
    device const float4* solvedImpulses [[buffer(7)]],
    device const NumiTemporalConeIslandStatus* solverStatuses [[buffer(8)]],
    device float4* warmImpulses [[buffer(9)]],
    device uint* warmIdentities [[buffer(10)]],
    device NumiBraidedBagStatus* statuses [[buffer(11)]],
    device const float* maximumRowSums [[buffer(12)]],
    device const uint* activeBlockCounts [[buffer(13)]],
    const uint environment [[threadgroup_position_in_grid]],
    const uint threadIndex [[thread_index_in_threadgroup]],
    const uint simdLane [[thread_index_in_simdgroup]],
    const uint simdGroup [[simdgroup_index_in_threadgroup]]
) {
    if (environment >= config.control.y) {
        return;
    }
    threadgroup NumiBraidedBagContact contacts[
        NUMI_BRAIDED_BAG_CONTACT_COUNT
    ];
    threadgroup float maximumNodeRadiusGroups[2];
    threadgroup float maximumNodeSpeedGroups[2];
    threadgroup float maximumStretchGroups[2];
    threadgroup float maximumPenetrationGroups[2];
    threadgroup float maximumBallRadiusGroups[2];
    threadgroup float minimumBallHeightGroups[2];
    threadgroup float maximumBallSpeedGroups[2];
    threadgroup float kineticDeltaGroups[2];
    threadgroup float kineticScaleGroups[2];
    threadgroup float allowedRecoveryWorkGroups[2];
    threadgroup float maximumConeUtilizationGroups[2];
    threadgroup float maximumNormalImpulseGroups[2];
    threadgroup float maximumTangentImpulseGroups[2];
    threadgroup uint impulseBearingGroups[2];
    threadgroup uint stickingGroups[2];
    threadgroup uint slidingGroups[2];
    threadgroup uint zeroImpulseGroups[2];
    threadgroup uint escapeMaskGroups[2];
    const uint compactContactBase =
        environment * NUMI_BRAIDED_BAG_CONTACT_COUNT;
    const uint solverContactBase =
        environment * NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS;
    const uint contactCount = solverStatuses[environment].control.w;
    if (threadIndex < contactCount) {
        contacts[threadIndex] = geometry[
            compactContactBase + threadIndex
        ];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool accepted = solverStatuses[environment].control.x ==
        NUMI_TEMPORAL_CONE_ISLAND_SUCCESS;
    if (threadIndex < NUMI_BRAIDED_BAG_PARTICLE_COUNT && accepted) {
        const uint particle = threadIndex;
        const float inverseMass = bagParticleInverseMass(
            nodes,
            balls,
            environment,
            particle
        );
        float3 velocity = bagParticleCandidateVelocity(
            candidateNodeVelocities,
            candidateBallVelocities,
            environment,
            particle
        );
        if (inverseMass > 0.0f) {
            for (uint contactIndex = 0u;
                 contactIndex < contactCount;
                 ++contactIndex) {
                const NumiBraidedBagContact contact = contacts[contactIndex];
                const uint count = uint(contact.weights.w);
                for (uint participant = 0u;
                     participant < count;
                     ++participant) {
                    if (contact.particles[participant] == particle) {
                        velocity += inverseMass *
                            contact.weights[participant] *
                            bagImpulseWorld(
                                contacts[contactIndex],
                                solvedImpulses[
                                    solverContactBase + contactIndex
                                ].xyz
                            );
                    }
                }
            }
        }
        if (particle < NUMI_BRAIDED_BAG_NODE_COUNT) {
            const uint index =
                environment * NUMI_BRAIDED_BAG_NODE_COUNT + particle;
            NumiBraidedBagNode state = nodes[index];
            if (inverseMass > 0.0f && bagFinite3(velocity)) {
                state.velocity = float4(velocity, 0.0f);
                state.positionAndInverseMass.xyz += config.timing.x * velocity;
                nodes[index] = state;
            }
        } else {
            const uint ball = particle - NUMI_BRAIDED_BAG_NODE_COUNT;
            const uint index =
                environment * NUMI_BRAIDED_BAG_BALL_COUNT + ball;
            NumiBraidedBagBall state = balls[index];
            if (bagFinite3(velocity)) {
                state.velocityAndRadius.xyz = velocity;
                state.positionAndInverseMass.xyz += config.timing.x * velocity;
                balls[index] = state;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    float lanePenetration = 0.0f;
    float laneAllowedRecoveryWork = 0.0f;
    float laneConeUtilization = 0.0f;
    float laneNormalImpulse = 0.0f;
    float laneTangentImpulse = 0.0f;
    uint laneImpulseBearing = 0u;
    uint laneSticking = 0u;
    uint laneSliding = 0u;
    uint laneZeroImpulse = 0u;
    if (threadIndex < NUMI_BRAIDED_BAG_CONTACT_COUNT) {
        if (accepted && threadIndex < contactCount) {
            const NumiBraidedBagContact contact = contacts[threadIndex];
            const float3 impulse = solvedImpulses[
                solverContactBase + threadIndex
            ].xyz;
            warmImpulses[compactContactBase + threadIndex] = float4(
                impulse,
                0.0f
            );
            warmIdentities[compactContactBase + threadIndex] =
                (contact.identity.x << 16u) | contact.identity.y;
            lanePenetration = max(
                -contact.normalAndSeparation.w,
                0.0f
            );
            const float inverseTimestep = 1.0f / config.timing.x;
            const float separation = contact.normalAndSeparation.w;
            const float bias = separation >= 0.0f
                ? separation * inverseTimestep
                : max(
                    config.contact.x * separation * inverseTimestep,
                    -config.contact.w
                );
            laneAllowedRecoveryWork = max(-impulse.x * bias, 0.0f);
            laneNormalImpulse = max(impulse.x, 0.0f);
            laneTangentImpulse = length(impulse.yz);
            if (laneNormalImpulse > 1.0e-6f) {
                laneImpulseBearing = 1u;
                const float coneRadius =
                    config.braidMaterial.w * laneNormalImpulse;
                laneConeUtilization = coneRadius > 0.0f
                    ? laneTangentImpulse / coneRadius
                    : 0.0f;
                if (laneConeUtilization >= 0.99f) {
                    laneSliding = 1u;
                } else {
                    laneSticking = 1u;
                }
            } else {
                laneZeroImpulse = 1u;
            }
        } else {
            warmImpulses[compactContactBase + threadIndex] = float4(0.0f);
            warmIdentities[compactContactBase + threadIndex] =
                NUMI_BRAIDED_BAG_INVALID_PARTICLE;
        }
    }

    const uint nodeBase = environment * NUMI_BRAIDED_BAG_NODE_COUNT;
    float laneNodeRadius = 0.0f;
    float laneNodeSpeed = 0.0f;
    if (threadIndex < NUMI_BRAIDED_BAG_NODE_COUNT) {
        const NumiBraidedBagNode state = nodes[nodeBase + threadIndex];
        laneNodeRadius = length(state.positionAndInverseMass.xy);
        laneNodeSpeed = length(state.velocity.xyz);
    }

    float laneStretch = 0.0f;
    for (uint edge = threadIndex;
         edge < NUMI_BRAIDED_BAG_EDGE_COUNT;
         edge += 64u) {
        const NumiBraidedBagEdge link = edges[edge];
        const float lengthNow = distance(
            nodes[nodeBase + link.nodes.x].positionAndInverseMass.xyz,
            nodes[nodeBase + link.nodes.y].positionAndInverseMass.xyz
        );
        laneStretch = max(
            laneStretch,
            abs(lengthNow - link.rest.x) / link.rest.x
        );
    }

    const uint ballBase = environment * NUMI_BRAIDED_BAG_BALL_COUNT;
    float laneBallRadius = 0.0f;
    float laneBallHeight = INFINITY;
    float laneBallSpeed = 0.0f;
    float laneBallPhysicalRadius = 0.0f;
    if (threadIndex < NUMI_BRAIDED_BAG_BALL_COUNT) {
        const NumiBraidedBagBall state = balls[ballBase + threadIndex];
        laneBallRadius = length(state.positionAndInverseMass.xy);
        laneBallHeight = state.positionAndInverseMass.z;
        laneBallSpeed = length(state.velocityAndRadius.xyz);
        laneBallPhysicalRadius = state.velocityAndRadius.w;
    }

    float laneKineticDelta = 0.0f;
    float laneKineticScale = 0.0f;
    if (accepted && threadIndex < NUMI_BRAIDED_BAG_PARTICLE_COUNT) {
        const float inverseMass = bagParticleInverseMass(
            nodes,
            balls,
            environment,
            threadIndex
        );
        if (inverseMass > 0.0f) {
            const float3 before = bagParticleCandidateVelocity(
                candidateNodeVelocities,
                candidateBallVelocities,
                environment,
                threadIndex
            );
            const float3 after = threadIndex < NUMI_BRAIDED_BAG_NODE_COUNT
                ? nodes[nodeBase + threadIndex].velocity.xyz
                : balls[
                    ballBase +
                    threadIndex - NUMI_BRAIDED_BAG_NODE_COUNT
                ].velocityAndRadius.xyz;
            const float kineticBefore =
                0.5f * dot(before, before) / inverseMass;
            const float kineticAfter =
                0.5f * dot(after, after) / inverseMass;
            laneKineticDelta = kineticAfter - kineticBefore;
            laneKineticScale = max(kineticBefore, kineticAfter);
        }
    }

    const float groupMaximumNodeRadius = simd_max(laneNodeRadius);
    const float groupMaximumNodeSpeed = simd_max(laneNodeSpeed);
    const float groupMaximumStretch = simd_max(laneStretch);
    const float groupMaximumPenetration = simd_max(lanePenetration);
    const float groupMaximumBallRadius = simd_max(laneBallRadius);
    const float groupMinimumBallHeight = simd_min(laneBallHeight);
    const float groupMaximumBallSpeed = simd_max(laneBallSpeed);
    const float groupKineticDelta = simd_sum(laneKineticDelta);
    const float groupKineticScale = simd_sum(laneKineticScale);
    const float groupAllowedRecoveryWork = simd_sum(
        laneAllowedRecoveryWork
    );
    const float groupMaximumConeUtilization = simd_max(
        laneConeUtilization
    );
    const float groupMaximumNormalImpulse = simd_max(laneNormalImpulse);
    const float groupMaximumTangentImpulse = simd_max(laneTangentImpulse);
    const uint groupImpulseBearing = simd_sum(laneImpulseBearing);
    const uint groupSticking = simd_sum(laneSticking);
    const uint groupSliding = simd_sum(laneSliding);
    const uint groupZeroImpulse = simd_sum(laneZeroImpulse);
    if (simdLane == 0u) {
        maximumNodeRadiusGroups[simdGroup] = groupMaximumNodeRadius;
        maximumNodeSpeedGroups[simdGroup] = groupMaximumNodeSpeed;
        maximumStretchGroups[simdGroup] = groupMaximumStretch;
        maximumPenetrationGroups[simdGroup] = groupMaximumPenetration;
        maximumBallRadiusGroups[simdGroup] = groupMaximumBallRadius;
        minimumBallHeightGroups[simdGroup] = groupMinimumBallHeight;
        maximumBallSpeedGroups[simdGroup] = groupMaximumBallSpeed;
        kineticDeltaGroups[simdGroup] = groupKineticDelta;
        kineticScaleGroups[simdGroup] = groupKineticScale;
        allowedRecoveryWorkGroups[simdGroup] =
            groupAllowedRecoveryWork;
        maximumConeUtilizationGroups[simdGroup] =
            groupMaximumConeUtilization;
        maximumNormalImpulseGroups[simdGroup] =
            groupMaximumNormalImpulse;
        maximumTangentImpulseGroups[simdGroup] =
            groupMaximumTangentImpulse;
        impulseBearingGroups[simdGroup] = groupImpulseBearing;
        stickingGroups[simdGroup] = groupSticking;
        slidingGroups[simdGroup] = groupSliding;
        zeroImpulseGroups[simdGroup] = groupZeroImpulse;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float maximumNodeRadius = max(
        maximumNodeRadiusGroups[0],
        maximumNodeRadiusGroups[1]
    );
    uint laneEscapeMask = 0u;
    if (threadIndex < NUMI_BRAIDED_BAG_BALL_COUNT) {
        const bool radialEscape =
            laneBallRadius - laneBallPhysicalRadius >
            maximumNodeRadius + config.braidMaterial.z + config.bounds.w;
        const bool bottomEscape =
            laneBallHeight < config.bounds.y - config.bounds.w;
        const bool mouthEscape =
            laneBallHeight - laneBallPhysicalRadius >
            config.bounds.z + config.bounds.w;
        if (radialEscape || bottomEscape || mouthEscape) {
            laneEscapeMask = 1u << threadIndex;
        }
    }
    const uint groupEscapeMask = simd_sum(laneEscapeMask);
    if (simdLane == 0u) {
        escapeMaskGroups[simdGroup] = groupEscapeMask;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (threadIndex == 0u) {
        NumiBraidedBagStatus status = statuses[environment];
        status.control.y += 1u;
        status.control.z = max(
            status.control.z,
            solverStatuses[environment].control.y
        );
        status.solverMetrics.z = max(
            status.solverMetrics.z,
            solverStatuses[environment].residuals.x
        );
        status.solverMetrics.w = max(
            status.solverMetrics.w,
            solverStatuses[environment].residuals.y
        );
        status.certificateMetrics.x = max(
            status.certificateMetrics.x,
            solverStatuses[environment].diagnostics.z
        );
        status.certificateMetrics.y = max(
            status.certificateMetrics.y,
            max(solverStatuses[environment].residuals.w, 0.0f)
        );
        const float maximumRowSum = maximumRowSums[environment];
        const float conditionBound = config.contact.y > 0.0f
            ? maximumRowSum / config.contact.y
            : INFINITY;
        status.certificateMetrics.z = max(
            status.certificateMetrics.z,
            maximumRowSum
        );
        status.certificateMetrics.w = max(
            status.certificateMetrics.w,
            conditionBound
        );
        status.topologyMetrics.x = min(
            status.topologyMetrics.x,
            contactCount
        );
        status.topologyMetrics.y = max(
            status.topologyMetrics.y,
            contactCount
        );
        status.topologyMetrics.z += contactCount;
        status.topologyMetrics.w = max(
            status.topologyMetrics.w,
            activeBlockCounts[environment]
        );
        if (!accepted) {
            status.control.x += 1u;
        }
        status.solverMetrics.x = max(
            status.solverMetrics.x,
            max(
                maximumPenetrationGroups[0],
                maximumPenetrationGroups[1]
            )
        );
        status.solverMetrics.y = max(
            status.solverMetrics.y,
            max(maximumStretchGroups[0], maximumStretchGroups[1])
        );
        status.physicalMetrics.x = max(
            status.physicalMetrics.x,
            max(maximumBallRadiusGroups[0], maximumBallRadiusGroups[1])
        );
        status.physicalMetrics.y = min(
            status.physicalMetrics.y,
            min(minimumBallHeightGroups[0], minimumBallHeightGroups[1])
        );
        status.physicalMetrics.z = max(
            status.physicalMetrics.z,
            max(maximumBallSpeedGroups[0], maximumBallSpeedGroups[1])
        );
        status.physicalMetrics.w = max(
            status.physicalMetrics.w,
            max(maximumNodeSpeedGroups[0], maximumNodeSpeedGroups[1])
        );
        status.control.w |= escapeMaskGroups[0] + escapeMaskGroups[1];

        const float kineticDelta =
            kineticDeltaGroups[0] + kineticDeltaGroups[1];
        const float kineticScale =
            kineticScaleGroups[0] + kineticScaleGroups[1];
        const float allowedRecoveryWork =
            allowedRecoveryWorkGroups[0] + allowedRecoveryWorkGroups[1];
        const float energyExcess = max(
            kineticDelta - allowedRecoveryWork,
            0.0f
        );
        const float normalizedEnergyExcess = energyExcess / max(
            1.0f,
            max(kineticScale, allowedRecoveryWork)
        );
        status.energyMetrics.x = max(
            status.energyMetrics.x,
            max(kineticDelta, 0.0f)
        );
        status.energyMetrics.y = max(
            status.energyMetrics.y,
            allowedRecoveryWork
        );
        status.energyMetrics.z = max(
            status.energyMetrics.z,
            energyExcess
        );
        status.energyMetrics.w = max(
            status.energyMetrics.w,
            normalizedEnergyExcess
        );
        status.frictionMetrics.x = max(
            status.frictionMetrics.x,
            max(
                maximumConeUtilizationGroups[0],
                maximumConeUtilizationGroups[1]
            )
        );
        status.frictionMetrics.y = max(
            status.frictionMetrics.y,
            max(
                maximumNormalImpulseGroups[0],
                maximumNormalImpulseGroups[1]
            )
        );
        status.frictionMetrics.z = max(
            status.frictionMetrics.z,
            max(
                maximumTangentImpulseGroups[0],
                maximumTangentImpulseGroups[1]
            )
        );
        status.contactRegimes.x +=
            impulseBearingGroups[0] + impulseBearingGroups[1];
        status.contactRegimes.y += stickingGroups[0] + stickingGroups[1];
        status.contactRegimes.z += slidingGroups[0] + slidingGroups[1];
        status.contactRegimes.w +=
            zeroImpulseGroups[0] + zeroImpulseGroups[1];
        statuses[environment] = status;
    }
}
