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

inline uint bagContactBallMask(const uint contact) {
    if (contact < NUMI_BRAIDED_BAG_BALL_COUNT) {
        return 1u << contact;
    }
    const uint pairBase = NUMI_BRAIDED_BAG_BALL_COUNT;
    if (contact < pairBase + NUMI_BRAIDED_BAG_BALL_PAIR_COUNT) {
        uint first;
        uint second;
        bagPairBalls(contact - pairBase, first, second);
        return (1u << first) | (1u << second);
    }
    const uint ball = contact -
        pairBase - NUMI_BRAIDED_BAG_BALL_PAIR_COUNT;
    return 1u << ball;
}

inline bool bagContactsMayShare(
    const uint target,
    const uint source
) {
    if (target == source) {
        return true;
    }
    if (target < NUMI_BRAIDED_BAG_BALL_COUNT &&
        source < NUMI_BRAIDED_BAG_BALL_COUNT) {
        return true;
    }
    return (bagContactBallMask(target) & bagContactBallMask(source)) != 0u;
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
    device const uint* streamRowOffsets [[buffer(15)]],
    const uint environment [[threadgroup_position_in_grid]],
    const uint lane [[thread_index_in_simdgroup]]
) {
    if (environment >= config.control.y) {
        return;
    }
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
        uint identity = 0u;
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
            contact.identity = uint4(0u, closestEdge, 0u, 0u);
            identity = closestEdge;
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
            identity = 0x10000u | pair;
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
            identity = 0x20000u | ball;
        }
        float3 normal;
        float3 tangentU;
        float3 tangentV;
        bagContactFrame(authoredNormal, normal, tangentU, tangentV);
        contact.normalAndSeparation = float4(normal, separation);
        contact.tangentU = float4(tangentU, 0.0f);
        contact.tangentV = float4(tangentV, 0.0f);
        contacts[lane] = contact;
        outputGeometry[compactContactBase + lane] = contact;
        const float3 relativeVelocity = bagRelativeVelocity(
            contacts[lane],
            candidateNodeVelocities,
            candidateBallVelocities,
            environment
        );
        const float inverseTimestep = 1.0f / config.timing.x;
        const float bias = separation >= 0.0f
            ? separation * inverseTimestep
            : max(
                config.contact.x * separation * inverseTimestep,
                -config.contact.w
            );
        NumiTemporalConeIslandContact solverContact = {};
        solverContact.freeVelocityAndFrictionU = float4(
            dot(relativeVelocity, normal) + bias,
            dot(relativeVelocity, tangentU),
            dot(relativeVelocity, tangentV),
            config.braidMaterial.w
        );
        solverContact.warmImpulseAndFrictionV = float4(
            warmIdentities[compactContactBase + lane] == identity
                ? warmImpulses[compactContactBase + lane].xyz
                : float3(0.0f),
            config.braidMaterial.w
        );
        solverContact.limits = float4(0.0f);
        outputContacts[solverContactBase + lane] = solverContact;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0u) {
        NumiTemporalConeIslandHeader denseHeader = {};
        denseHeader.control = uint4(
            NUMI_TEMPORAL_CONE_ISLAND_ABI_VERSION,
            NUMI_BRAIDED_BAG_CONTACT_COUNT,
            config.control.z,
            config.control.w
        );
        denseHeader.tolerances = config.solver;
        outputDenseHeaders[environment] = denseHeader;

        NumiTemporalConeStreamHeader streamHeader = {};
        streamHeader.control = uint4(
            NUMI_TEMPORAL_CONE_STREAM_ABI_VERSION,
            NUMI_BRAIDED_BAG_CONTACT_COUNT,
            config.control.z,
            config.control.w
        );
        streamHeader.ranges = uint4(
            solverContactBase,
            environment * (NUMI_BRAIDED_BAG_CONTACT_COUNT + 1u),
            environment * NUMI_BRAIDED_BAG_STREAM_BLOCK_COUNT,
            NUMI_BRAIDED_BAG_STREAM_BLOCK_COUNT
        );
        streamHeader.tolerances = config.solver;
        outputStreamHeaders[environment] = streamHeader;
    }

    if (path == NUMI_BRAIDED_BAG_PATH_DENSE) {
        const uint matrixBase =
            environment * NUMI_TEMPORAL_CONE_ISLAND_MATRIX_ELEMENTS;
        for (uint element = lane;
             element < NUMI_TEMPORAL_CONE_ISLAND_MATRIX_ELEMENTS;
             element += 32u) {
            outputDenseMatrices[matrixBase + element] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_device);
        if (lane < NUMI_BRAIDED_BAG_CONTACT_COUNT) {
            for (uint source = 0u;
                 source < NUMI_BRAIDED_BAG_CONTACT_COUNT;
                 ++source) {
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
                        outputDenseMatrices[
                            matrixBase +
                            (3u * lane + targetAxis) *
                                NUMI_TEMPORAL_CONE_ISLAND_MAX_ROWS +
                            3u * source + sourceAxis
                        ] = coefficient;
                    }
                }
            }
        }
    } else if (lane < NUMI_BRAIDED_BAG_CONTACT_COUNT) {
        const uint blockBase =
            environment * NUMI_BRAIDED_BAG_STREAM_BLOCK_COUNT;
        uint rowBlock = streamRowOffsets[
            environment * (NUMI_BRAIDED_BAG_CONTACT_COUNT + 1u) + lane
        ];
        for (uint source = 0u;
             source < NUMI_BRAIDED_BAG_CONTACT_COUNT;
             ++source) {
            if (!bagContactsMayShare(lane, source)) {
                continue;
            }
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
                    outputStreamBlocks[
                        valueBase + 3u * targetAxis + sourceAxis
                    ] = coefficient;
                }
            }
            ++rowBlock;
        }
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
    const uint environment [[threadgroup_position_in_grid]],
    const uint threadIndex [[thread_index_in_threadgroup]]
) {
    if (environment >= config.control.y) {
        return;
    }
    threadgroup NumiBraidedBagContact contacts[
        NUMI_BRAIDED_BAG_CONTACT_COUNT
    ];
    const uint compactContactBase =
        environment * NUMI_BRAIDED_BAG_CONTACT_COUNT;
    const uint solverContactBase =
        environment * NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS;
    if (threadIndex < NUMI_BRAIDED_BAG_CONTACT_COUNT) {
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
                 contactIndex < NUMI_BRAIDED_BAG_CONTACT_COUNT;
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
        if (!accepted) {
            status.control.x += 1u;
        }
        for (uint contactIndex = 0u;
             contactIndex < NUMI_BRAIDED_BAG_CONTACT_COUNT;
             ++contactIndex) {
            if (accepted) {
                warmImpulses[compactContactBase + contactIndex] =
                    solvedImpulses[solverContactBase + contactIndex];
                const NumiBraidedBagContact contact = contacts[contactIndex];
                warmIdentities[compactContactBase + contactIndex] =
                    (contact.identity.x << 16u) | contact.identity.y;
            } else {
                warmImpulses[compactContactBase + contactIndex] =
                    float4(0.0f);
                warmIdentities[compactContactBase + contactIndex] =
                    NUMI_BRAIDED_BAG_INVALID_PARTICLE;
            }
            status.solverMetrics.x = max(
                status.solverMetrics.x,
                max(-contacts[contactIndex].normalAndSeparation.w, 0.0f)
            );
        }
        const uint nodeBase =
            environment * NUMI_BRAIDED_BAG_NODE_COUNT;
        float maximumNodeRadius = 0.0f;
        for (uint node = 0u;
             node < NUMI_BRAIDED_BAG_NODE_COUNT;
             ++node) {
            const NumiBraidedBagNode state = nodes[nodeBase + node];
            maximumNodeRadius = max(
                maximumNodeRadius,
                length(state.positionAndInverseMass.xy)
            );
            status.physicalMetrics.w = max(
                status.physicalMetrics.w,
                length(state.velocity.xyz)
            );
        }
        for (uint edge = 0u;
             edge < NUMI_BRAIDED_BAG_EDGE_COUNT;
             ++edge) {
            const NumiBraidedBagEdge link = edges[edge];
            const float lengthNow = distance(
                nodes[nodeBase + link.nodes.x].positionAndInverseMass.xyz,
                nodes[nodeBase + link.nodes.y].positionAndInverseMass.xyz
            );
            status.solverMetrics.y = max(
                status.solverMetrics.y,
                abs(lengthNow - link.rest.x) / link.rest.x
            );
        }
        const uint ballBase =
            environment * NUMI_BRAIDED_BAG_BALL_COUNT;
        for (uint ball = 0u;
             ball < NUMI_BRAIDED_BAG_BALL_COUNT;
             ++ball) {
            const NumiBraidedBagBall state = balls[ballBase + ball];
            const float radial = length(state.positionAndInverseMass.xy);
            status.physicalMetrics.x = max(
                status.physicalMetrics.x,
                radial
            );
            status.physicalMetrics.y = min(
                status.physicalMetrics.y,
                state.positionAndInverseMass.z
            );
            status.physicalMetrics.z = max(
                status.physicalMetrics.z,
                length(state.velocityAndRadius.xyz)
            );
            const bool radialEscape =
                radial - state.velocityAndRadius.w >
                maximumNodeRadius + config.braidMaterial.z + config.bounds.w;
            const bool bottomEscape =
                state.positionAndInverseMass.z <
                config.bounds.y - config.bounds.w;
            const bool mouthEscape =
                state.positionAndInverseMass.z -
                    state.velocityAndRadius.w >
                config.bounds.z + config.bounds.w;
            if (radialEscape || bottomEscape || mouthEscape) {
                status.control.w |= 1u << ball;
            }
        }
        statuses[environment] = status;
    }
}
