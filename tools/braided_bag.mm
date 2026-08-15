#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/braided_bag.h"
#include "numi/temporal_cone_island.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef NUMI_TEMPORAL_CONE_METALLIB
#error "NUMI_TEMPORAL_CONE_METALLIB must name the built solver metallib"
#endif

namespace {

constexpr float kPi = 3.14159265358979323846f;

enum class BagPreset {
    baseline,
    stiffBraid,
    lowCFM,
    highFriction,
};

std::string_view presetName(const BagPreset preset) {
    switch (preset) {
    case BagPreset::baseline:
        return "baseline";
    case BagPreset::stiffBraid:
        return "stiff-braid";
    case BagPreset::lowCFM:
        return "low-cfm";
    case BagPreset::highFriction:
        return "high-friction";
    }
    return "unknown";
}

BagPreset parsePreset(const std::string_view value) {
    if (value == "baseline") {
        return BagPreset::baseline;
    }
    if (value == "stiff-braid") {
        return BagPreset::stiffBraid;
    }
    if (value == "low-cfm") {
        return BagPreset::lowCFM;
    }
    if (value == "high-friction") {
        return BagPreset::highFriction;
    }
    throw std::runtime_error("unknown braided-bag preset: " + std::string(value));
}

mr_float4 f4(
    const float x,
    const float y,
    const float z,
    const float w = 0.0f
) {
    return {x, y, z, w};
}

mr_uint4 u4(
    const std::uint32_t x,
    const std::uint32_t y,
    const std::uint32_t z,
    const std::uint32_t w
) {
    return {x, y, z, w};
}

std::string errorText(NSError* error) {
    return error == nil
        ? std::string("unknown Metal error")
        : std::string(error.localizedDescription.UTF8String);
}

struct InitialState {
    NumiBraidedBagConfig config{};
    std::vector<NumiBraidedBagNode> nodes;
    std::vector<NumiBraidedBagBall> balls;
    std::vector<NumiBraidedBagEdge> edges;
    std::vector<std::uint32_t> rowOffsets;
    std::vector<std::uint32_t> columnIndices;
    std::vector<NumiBraidedBagStatus> statuses;
};

struct RunResult {
    std::vector<NumiBraidedBagNode> nodes;
    std::vector<NumiBraidedBagBall> balls;
    std::vector<NumiBraidedBagStatus> statuses;
    std::vector<NumiTemporalConeIslandStatus> solverStatuses;
    double seconds = 0.0;
};

std::array<float, 3> nodePosition(
    const std::size_t level,
    const std::size_t ring
) {
    const float radius = level == 0u
        ? 0.08f
        : level == 1u
        ? 0.32f
        : 0.52f;
    const float angle =
        2.0f * kPi * static_cast<float>(ring) /
            static_cast<float>(NUMI_BRAIDED_BAG_RING_SIZE) +
        0.20f * static_cast<float>(level);
    return {{
        radius * std::cos(angle),
        radius * std::sin(angle),
        0.18f * static_cast<float>(level),
    }};
}

InitialState makeInitialState(
    const std::size_t environmentCount,
    const BagPreset preset,
    const float timestep
) {
    InitialState state;
    state.config.control = u4(
        NUMI_BRAIDED_BAG_ABI_VERSION,
        static_cast<std::uint32_t>(environmentCount),
        4u,
        768u
    );
    state.config.timing = f4(
        timestep,
        -9.81f,
        0.08f,
        0.01f
    );
    state.config.braidMaterial = f4(
        80.0f,
        0.30f,
        0.025f,
        0.60f
    );
    state.config.contact = f4(0.25f, 0.05f, 0.0f, 3.0f);
    state.config.solver = f4(1.0e-6f, 1.0e-6f, 1.0f, 0.0f);
    state.config.bounds = f4(0.52f, 0.0f, 1.08f, 0.02f);
    switch (preset) {
    case BagPreset::baseline:
        break;
    case BagPreset::stiffBraid:
        state.config.braidMaterial.x = 320.0f;
        state.config.braidMaterial.y = 0.75f;
        state.config.timing.z = 0.12f;
        state.config.control.w = 768u;
        break;
    case BagPreset::lowCFM:
        state.config.contact.y = 0.02f;
        state.config.control.w = 1024u;
        break;
    case BagPreset::highFriction:
        state.config.braidMaterial.w = 1.20f;
        state.config.control.w = 768u;
        break;
    }

    state.edges.reserve(NUMI_BRAIDED_BAG_EDGE_COUNT);
    for (std::size_t level = 0u;
         level + 1u < NUMI_BRAIDED_BAG_LEVEL_COUNT;
         ++level) {
        for (std::size_t ring = 0u;
             ring < NUMI_BRAIDED_BAG_RING_SIZE;
             ++ring) {
            for (const int direction : {-1, 1}) {
                const std::size_t nextRing = static_cast<std::size_t>(
                    (static_cast<int>(ring) + direction +
                     static_cast<int>(NUMI_BRAIDED_BAG_RING_SIZE)) %
                    static_cast<int>(NUMI_BRAIDED_BAG_RING_SIZE)
                );
                const std::size_t first =
                    level * NUMI_BRAIDED_BAG_RING_SIZE + ring;
                const std::size_t second =
                    (level + 1u) * NUMI_BRAIDED_BAG_RING_SIZE + nextRing;
                const auto firstPosition = nodePosition(level, ring);
                const auto secondPosition = nodePosition(level + 1u, nextRing);
                const double dx = secondPosition[0] - firstPosition[0];
                const double dy = secondPosition[1] - firstPosition[1];
                const double dz = secondPosition[2] - firstPosition[2];
                NumiBraidedBagEdge edge{};
                edge.nodes = u4(
                    static_cast<std::uint32_t>(first),
                    static_cast<std::uint32_t>(second),
                    0u,
                    0u
                );
                edge.rest = f4(static_cast<float>(std::sqrt(
                    dx * dx + dy * dy + dz * dz
                )), 0.0f, 0.0f, 0.0f);
                state.edges.push_back(edge);
            }
        }
    }
    for (std::size_t ring = 0u;
         ring < NUMI_BRAIDED_BAG_RING_SIZE / 2u;
         ++ring) {
        const std::size_t opposite =
            ring + NUMI_BRAIDED_BAG_RING_SIZE / 2u;
        const auto firstPosition = nodePosition(0u, ring);
        const auto secondPosition = nodePosition(0u, opposite);
        const double dx = secondPosition[0] - firstPosition[0];
        const double dy = secondPosition[1] - firstPosition[1];
        const double dz = secondPosition[2] - firstPosition[2];
        NumiBraidedBagEdge edge{};
        edge.nodes = u4(
            static_cast<std::uint32_t>(ring),
            static_cast<std::uint32_t>(opposite),
            0u,
            0u
        );
        edge.rest = f4(static_cast<float>(std::sqrt(
            dx * dx + dy * dy + dz * dz
        )), 0.0f, 0.0f, 0.0f);
        state.edges.push_back(edge);
    }
    if (state.edges.size() != NUMI_BRAIDED_BAG_EDGE_COUNT) {
        throw std::runtime_error("braid edge count does not match ABI");
    }

    state.nodes.resize(environmentCount * NUMI_BRAIDED_BAG_NODE_COUNT);
    state.balls.resize(environmentCount * NUMI_BRAIDED_BAG_BALL_COUNT);
    state.statuses.resize(environmentCount);
    constexpr std::array<std::array<float, 3>, 6> authoredBalls{{
        {{0.180000f, 0.000000f, 0.82f}},
        {{-0.090000f, 0.155885f, 0.82f}},
        {{-0.090000f, -0.155885f, 0.82f}},
        {{0.085000f, 0.147224f, 0.48f}},
        {{-0.170000f, 0.000000f, 0.48f}},
        {{0.085000f, -0.147224f, 0.48f}},
    }};
    for (std::size_t environment = 0u;
         environment < environmentCount;
         ++environment) {
        for (std::size_t level = 0u;
             level < NUMI_BRAIDED_BAG_LEVEL_COUNT;
             ++level) {
            for (std::size_t ring = 0u;
                 ring < NUMI_BRAIDED_BAG_RING_SIZE;
                 ++ring) {
                const std::size_t node =
                    level * NUMI_BRAIDED_BAG_RING_SIZE + ring;
                const auto position = nodePosition(level, ring);
                auto& output = state.nodes[
                    environment * NUMI_BRAIDED_BAG_NODE_COUNT + node
                ];
                output.positionAndInverseMass = f4(
                    position[0],
                    position[1],
                    position[2],
                    level + 1u == NUMI_BRAIDED_BAG_LEVEL_COUNT
                        ? 0.0f
                        : 40.0f
                );
                output.velocity = f4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        const float phase =
            2.0f * kPi * static_cast<float>(environment % 17u) / 17.0f;
        const float cosine = std::cos(phase);
        const float sine = std::sin(phase);
        for (std::size_t ball = 0u;
             ball < NUMI_BRAIDED_BAG_BALL_COUNT;
             ++ball) {
            const auto& authored = authoredBalls[ball];
            auto& output = state.balls[
                environment * NUMI_BRAIDED_BAG_BALL_COUNT + ball
            ];
            output.positionAndInverseMass = f4(
                cosine * authored[0] - sine * authored[1],
                sine * authored[0] + cosine * authored[1],
                authored[2],
                4.0f
            );
            output.velocityAndRadius = f4(0.0f, 0.0f, 0.0f, 0.14f);
        }
        state.statuses[environment].physicalMetrics.y =
            std::numeric_limits<float>::infinity();
        state.statuses[environment].topologyMetrics.x =
            std::numeric_limits<std::uint32_t>::max();
    }

    state.rowOffsets.resize(
        environmentCount * (NUMI_BRAIDED_BAG_CONTACT_COUNT + 1u),
        0u
    );
    state.columnIndices.resize(
        environmentCount * NUMI_BRAIDED_BAG_STREAM_BLOCK_COUNT,
        0u
    );
    return state;
}

id<MTLComputePipelineState> makePipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString* name
) {
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) {
        throw std::runtime_error(
            "missing Metal function: " + std::string(name.UTF8String)
        );
    }
    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:function
                                      error:&error];
    if (pipeline == nil) {
        throw std::runtime_error(
            "failed to create pipeline " + std::string(name.UTF8String) +
            ": " + errorText(error)
        );
    }
    return pipeline;
}

struct Pipelines {
    id<MTLComputePipelineState> freeMotion;
    id<MTLComputePipelineState> buildContacts;
    id<MTLComputePipelineState> denseSolver;
    id<MTLComputePipelineState> streamedSolver;
    id<MTLComputePipelineState> apply;
};

RunResult runGPU(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    const Pipelines& pipelines,
    const InitialState& initial,
    const std::uint32_t stepCount,
    const std::uint32_t path
) {
    const std::size_t environmentCount = initial.config.control.y;
    const auto makeBytes = [&](const auto& values) -> id<MTLBuffer> {
        return [device
            newBufferWithBytes:values.data()
                       length:values.size() * sizeof(values.front())
                      options:MTLResourceStorageModeShared];
    };
    const auto makeLength = [&](const std::size_t bytes) -> id<MTLBuffer> {
        return [device
            newBufferWithLength:std::max<std::size_t>(bytes, 16u)
                        options:MTLResourceStorageModeShared];
    };

    id<MTLBuffer> configBuffer = [device
        newBufferWithBytes:&initial.config
                   length:sizeof(initial.config)
                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> nodeBuffer = makeBytes(initial.nodes);
    id<MTLBuffer> ballBuffer = makeBytes(initial.balls);
    id<MTLBuffer> edgeBuffer = makeBytes(initial.edges);
    id<MTLBuffer> candidateNodeBuffer = makeLength(
        initial.nodes.size() * sizeof(mr_float4)
    );
    id<MTLBuffer> candidateBallBuffer = makeLength(
        initial.balls.size() * sizeof(mr_float4)
    );
    id<MTLBuffer> warmImpulseBuffer = makeLength(
        environmentCount * NUMI_BRAIDED_BAG_CONTACT_COUNT *
            sizeof(mr_float4)
    );
    id<MTLBuffer> warmIdentityBuffer = makeLength(
        environmentCount * NUMI_BRAIDED_BAG_CONTACT_COUNT *
            sizeof(std::uint32_t)
    );
    id<MTLBuffer> geometryBuffer = makeLength(
        environmentCount * NUMI_BRAIDED_BAG_CONTACT_COUNT *
            sizeof(NumiBraidedBagContact)
    );
    id<MTLBuffer> contactBuffer = makeLength(
        environmentCount * NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS *
            sizeof(NumiTemporalConeIslandContact)
    );
    id<MTLBuffer> denseHeaderBuffer = makeLength(
        environmentCount * sizeof(NumiTemporalConeIslandHeader)
    );
    id<MTLBuffer> denseMatrixBuffer = makeLength(
        path == NUMI_BRAIDED_BAG_PATH_DENSE
            ? environmentCount * NUMI_TEMPORAL_CONE_ISLAND_MATRIX_ELEMENTS *
                sizeof(float)
            : 16u
    );
    id<MTLBuffer> streamHeaderBuffer = makeLength(
        environmentCount * sizeof(NumiTemporalConeStreamHeader)
    );
    id<MTLBuffer> streamBlockBuffer = makeLength(
        path == NUMI_BRAIDED_BAG_PATH_STREAMED
            ? environmentCount * NUMI_BRAIDED_BAG_STREAM_BLOCK_COUNT * 9u *
                sizeof(float)
            : 16u
    );
    id<MTLBuffer> rowOffsetBuffer = makeBytes(initial.rowOffsets);
    id<MTLBuffer> columnIndexBuffer = makeBytes(initial.columnIndices);
    id<MTLBuffer> impulseBuffer = makeLength(
        environmentCount * NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS *
            sizeof(mr_float4)
    );
    id<MTLBuffer> solverStatusBuffer = makeLength(
        environmentCount * sizeof(NumiTemporalConeIslandStatus)
    );
    id<MTLBuffer> bagStatusBuffer = makeBytes(initial.statuses);
    id<MTLBuffer> maximumRowSumBuffer = makeLength(
        environmentCount * sizeof(float)
    );
    id<MTLBuffer> activeBlockCountBuffer = makeLength(
        environmentCount * sizeof(std::uint32_t)
    );
    const std::array buffers{
        configBuffer, nodeBuffer, ballBuffer, edgeBuffer,
        candidateNodeBuffer, candidateBallBuffer, warmImpulseBuffer,
        warmIdentityBuffer, geometryBuffer, contactBuffer,
        denseHeaderBuffer, denseMatrixBuffer, streamHeaderBuffer,
        streamBlockBuffer, rowOffsetBuffer, columnIndexBuffer,
        impulseBuffer, solverStatusBuffer, bagStatusBuffer,
        maximumRowSumBuffer,
        activeBlockCountBuffer,
    };
    if (std::any_of(buffers.begin(), buffers.end(), [](id<MTLBuffer> value) {
            return value == nil;
        })) {
        throw std::runtime_error("failed to allocate braided-bag buffers");
    }
    std::memset(warmImpulseBuffer.contents, 0, warmImpulseBuffer.length);
    std::fill_n(
        static_cast<std::uint32_t*>(warmIdentityBuffer.contents),
        environmentCount * NUMI_BRAIDED_BAG_CONTACT_COUNT,
        NUMI_BRAIDED_BAG_INVALID_PARTICLE
    );
    std::memset(contactBuffer.contents, 0, contactBuffer.length);
    std::memset(impulseBuffer.contents, 0, impulseBuffer.length);
    std::memset(solverStatusBuffer.contents, 0, solverStatusBuffer.length);
    std::memset(maximumRowSumBuffer.contents, 0, maximumRowSumBuffer.length);
    std::memset(
        activeBlockCountBuffer.contents,
        0,
        activeBlockCountBuffer.length
    );

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (commandBuffer == nil || encoder == nil) {
        throw std::runtime_error("failed to create braided-bag command encoder");
    }
    const auto barrier = [&]() {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    };
    const MTLSize environmentGroups = MTLSizeMake(environmentCount, 1u, 1u);
    const MTLSize simd32 = MTLSizeMake(32u, 1u, 1u);
    const MTLSize applyThreads = MTLSizeMake(64u, 1u, 1u);
    const std::uint32_t problemCount = static_cast<std::uint32_t>(
        environmentCount
    );
    const mr_uint4 streamCapacities = u4(
        static_cast<std::uint32_t>(
            environmentCount * NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS
        ),
        static_cast<std::uint32_t>(initial.rowOffsets.size()),
        static_cast<std::uint32_t>(
            environmentCount * NUMI_BRAIDED_BAG_STREAM_BLOCK_COUNT
        ),
        static_cast<std::uint32_t>(
            environmentCount * NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS
        )
    );
    for (std::uint32_t step = 0u; step < stepCount; ++step) {
        [encoder setComputePipelineState:pipelines.freeMotion];
        [encoder setBuffer:configBuffer offset:0 atIndex:0];
        [encoder setBuffer:nodeBuffer offset:0 atIndex:1];
        [encoder setBuffer:ballBuffer offset:0 atIndex:2];
        [encoder setBuffer:edgeBuffer offset:0 atIndex:3];
        [encoder setBuffer:candidateNodeBuffer offset:0 atIndex:4];
        [encoder setBuffer:candidateBallBuffer offset:0 atIndex:5];
        const std::size_t particleCount =
            environmentCount * NUMI_BRAIDED_BAG_PARTICLE_COUNT;
        [encoder dispatchThreads:MTLSizeMake(particleCount, 1u, 1u)
              threadsPerThreadgroup:MTLSizeMake(128u, 1u, 1u)];
        barrier();

        [encoder setComputePipelineState:pipelines.buildContacts];
        [encoder setBuffer:configBuffer offset:0 atIndex:0];
        [encoder setBuffer:nodeBuffer offset:0 atIndex:1];
        [encoder setBuffer:ballBuffer offset:0 atIndex:2];
        [encoder setBuffer:edgeBuffer offset:0 atIndex:3];
        [encoder setBuffer:candidateNodeBuffer offset:0 atIndex:4];
        [encoder setBuffer:candidateBallBuffer offset:0 atIndex:5];
        [encoder setBuffer:warmImpulseBuffer offset:0 atIndex:6];
        [encoder setBuffer:warmIdentityBuffer offset:0 atIndex:7];
        [encoder setBuffer:geometryBuffer offset:0 atIndex:8];
        [encoder setBuffer:contactBuffer offset:0 atIndex:9];
        [encoder setBuffer:denseHeaderBuffer offset:0 atIndex:10];
        [encoder setBuffer:denseMatrixBuffer offset:0 atIndex:11];
        [encoder setBuffer:streamHeaderBuffer offset:0 atIndex:12];
        [encoder setBuffer:streamBlockBuffer offset:0 atIndex:13];
        [encoder setBytes:&path length:sizeof(path) atIndex:14];
        [encoder setBuffer:rowOffsetBuffer offset:0 atIndex:15];
        [encoder setBuffer:maximumRowSumBuffer offset:0 atIndex:16];
        [encoder setBuffer:columnIndexBuffer offset:0 atIndex:17];
        [encoder setBuffer:activeBlockCountBuffer offset:0 atIndex:18];
        [encoder dispatchThreadgroups:environmentGroups
                 threadsPerThreadgroup:simd32];
        barrier();

        if (path == NUMI_BRAIDED_BAG_PATH_DENSE) {
            [encoder setComputePipelineState:pipelines.denseSolver];
            [encoder setBuffer:denseHeaderBuffer offset:0 atIndex:0];
            [encoder setBuffer:denseMatrixBuffer offset:0 atIndex:1];
            [encoder setBuffer:contactBuffer offset:0 atIndex:2];
            [encoder setBuffer:impulseBuffer offset:0 atIndex:3];
            [encoder setBuffer:solverStatusBuffer offset:0 atIndex:4];
            [encoder setBytes:&problemCount
                       length:sizeof(problemCount)
                      atIndex:5];
        } else {
            [encoder setComputePipelineState:pipelines.streamedSolver];
            [encoder setBuffer:streamHeaderBuffer offset:0 atIndex:0];
            [encoder setBuffer:rowOffsetBuffer offset:0 atIndex:1];
            [encoder setBuffer:columnIndexBuffer offset:0 atIndex:2];
            [encoder setBuffer:streamBlockBuffer offset:0 atIndex:3];
            [encoder setBuffer:contactBuffer offset:0 atIndex:4];
            [encoder setBuffer:impulseBuffer offset:0 atIndex:5];
            [encoder setBuffer:solverStatusBuffer offset:0 atIndex:6];
            [encoder setBytes:&problemCount
                       length:sizeof(problemCount)
                      atIndex:7];
            [encoder setBytes:&streamCapacities
                       length:sizeof(streamCapacities)
                      atIndex:8];
        }
        [encoder dispatchThreadgroups:environmentGroups
                 threadsPerThreadgroup:simd32];
        barrier();

        [encoder setComputePipelineState:pipelines.apply];
        [encoder setBuffer:configBuffer offset:0 atIndex:0];
        [encoder setBuffer:nodeBuffer offset:0 atIndex:1];
        [encoder setBuffer:ballBuffer offset:0 atIndex:2];
        [encoder setBuffer:edgeBuffer offset:0 atIndex:3];
        [encoder setBuffer:candidateNodeBuffer offset:0 atIndex:4];
        [encoder setBuffer:candidateBallBuffer offset:0 atIndex:5];
        [encoder setBuffer:geometryBuffer offset:0 atIndex:6];
        [encoder setBuffer:impulseBuffer offset:0 atIndex:7];
        [encoder setBuffer:solverStatusBuffer offset:0 atIndex:8];
        [encoder setBuffer:warmImpulseBuffer offset:0 atIndex:9];
        [encoder setBuffer:warmIdentityBuffer offset:0 atIndex:10];
        [encoder setBuffer:bagStatusBuffer offset:0 atIndex:11];
        [encoder setBuffer:maximumRowSumBuffer offset:0 atIndex:12];
        [encoder setBuffer:activeBlockCountBuffer offset:0 atIndex:13];
        [encoder dispatchThreadgroups:environmentGroups
                 threadsPerThreadgroup:applyThreads];
        barrier();
    }
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error(
            "braided-bag command failed: " + errorText(commandBuffer.error)
        );
    }

    RunResult result;
    const auto* outputNodes = static_cast<const NumiBraidedBagNode*>(
        nodeBuffer.contents
    );
    result.nodes.assign(outputNodes, outputNodes + initial.nodes.size());
    const auto* outputBalls = static_cast<const NumiBraidedBagBall*>(
        ballBuffer.contents
    );
    result.balls.assign(outputBalls, outputBalls + initial.balls.size());
    const auto* outputStatuses = static_cast<const NumiBraidedBagStatus*>(
        bagStatusBuffer.contents
    );
    result.statuses.assign(
        outputStatuses,
        outputStatuses + environmentCount
    );
    const auto* outputSolverStatuses =
        static_cast<const NumiTemporalConeIslandStatus*>(
            solverStatusBuffer.contents
        );
    result.solverStatuses.assign(
        outputSolverStatuses,
        outputSolverStatuses + environmentCount
    );
    if (commandBuffer.GPUEndTime >= commandBuffer.GPUStartTime) {
        result.seconds = commandBuffer.GPUEndTime - commandBuffer.GPUStartTime;
    }
    return result;
}

bool sameResult(const RunResult& first, const RunResult& second) {
    return first.nodes.size() == second.nodes.size() &&
        first.balls.size() == second.balls.size() &&
        first.statuses.size() == second.statuses.size() &&
        first.solverStatuses.size() == second.solverStatuses.size() &&
        std::memcmp(
            first.nodes.data(),
            second.nodes.data(),
            first.nodes.size() * sizeof(NumiBraidedBagNode)
        ) == 0 &&
        std::memcmp(
            first.balls.data(),
            second.balls.data(),
            first.balls.size() * sizeof(NumiBraidedBagBall)
        ) == 0 &&
        std::memcmp(
            first.statuses.data(),
            second.statuses.data(),
            first.statuses.size() * sizeof(NumiBraidedBagStatus)
        ) == 0 &&
        std::memcmp(
            first.solverStatuses.data(),
            second.solverStatuses.data(),
            first.solverStatuses.size() *
                sizeof(NumiTemporalConeIslandStatus)
        ) == 0;
}

double maximumPositionLInfDelta(
    const RunResult& first,
    const RunResult& second
) {
    if (first.nodes.size() != second.nodes.size() ||
        first.balls.size() != second.balls.size()) {
        return std::numeric_limits<double>::infinity();
    }
    double maximum = 0.0;
    for (std::size_t index = 0u; index < first.nodes.size(); ++index) {
        const auto& a = first.nodes[index].positionAndInverseMass;
        const auto& b = second.nodes[index].positionAndInverseMass;
        maximum = std::max({
            maximum,
            std::abs(static_cast<double>(a.x) - b.x),
            std::abs(static_cast<double>(a.y) - b.y),
            std::abs(static_cast<double>(a.z) - b.z),
        });
    }
    for (std::size_t index = 0u; index < first.balls.size(); ++index) {
        const auto& a = first.balls[index].positionAndInverseMass;
        const auto& b = second.balls[index].positionAndInverseMass;
        maximum = std::max({
            maximum,
            std::abs(static_cast<double>(a.x) - b.x),
            std::abs(static_cast<double>(a.y) - b.y),
            std::abs(static_cast<double>(a.z) - b.z),
        });
    }
    return maximum;
}

std::uint64_t hashResult(const RunResult& result) {
    std::uint64_t hash = 1469598103934665603ull;
    const auto append = [&](const void* data, const std::size_t bytes) {
        const auto* values = static_cast<const std::uint8_t*>(data);
        for (std::size_t index = 0u; index < bytes; ++index) {
            hash ^= values[index];
            hash *= 1099511628211ull;
        }
    };
    append(
        result.nodes.data(),
        result.nodes.size() * sizeof(NumiBraidedBagNode)
    );
    append(
        result.balls.data(),
        result.balls.size() * sizeof(NumiBraidedBagBall)
    );
    append(
        result.statuses.data(),
        result.statuses.size() * sizeof(NumiBraidedBagStatus)
    );
    return hash;
}

void dumpOBJ(
    const std::string& path,
    const InitialState& initial,
    const RunResult& result
) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open OBJ output: " + path);
    }
    output << "# Numi Solver braided bag, environment 0\n";
    for (std::size_t node = 0u;
         node < NUMI_BRAIDED_BAG_NODE_COUNT;
         ++node) {
        const auto& position = result.nodes[node].positionAndInverseMass;
        output << "v " << position.x << ' ' << position.y << ' '
               << position.z << '\n';
    }
    for (const auto& edge : initial.edges) {
        output << "l " << edge.nodes.x + 1u << ' '
               << edge.nodes.y + 1u << '\n';
    }
    for (std::size_t ball = 0u;
         ball < NUMI_BRAIDED_BAG_BALL_COUNT;
         ++ball) {
        const auto& position = result.balls[ball].positionAndInverseMass;
        output << "# ball " << ball << " center "
               << position.x << ' ' << position.y << ' ' << position.z
               << " radius " << result.balls[ball].velocityAndRadius.w
               << '\n';
    }
}

int run(const int argc, const char* const* argv) {
    std::size_t environmentCount = 128u;
    std::uint32_t stepCount = 480u;
    std::uint32_t replayCount = 2u;
    float timestep = 1.0f / 480.0f;
    BagPreset preset = BagPreset::baseline;
    bool refinement = false;
    std::string metallibPath = NUMI_TEMPORAL_CONE_METALLIB;
    std::string objPath;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string_view value(argv[argument]);
        if (value == "--environments" && argument + 1 < argc) {
            environmentCount = std::stoull(argv[++argument]);
        } else if (value == "--steps" && argument + 1 < argc) {
            stepCount = static_cast<std::uint32_t>(
                std::stoul(argv[++argument])
            );
        } else if (value == "--replays" && argument + 1 < argc) {
            replayCount = static_cast<std::uint32_t>(
                std::stoul(argv[++argument])
            );
        } else if (value == "--timestep" && argument + 1 < argc) {
            timestep = std::stof(argv[++argument]);
        } else if (value == "--preset" && argument + 1 < argc) {
            preset = parsePreset(argv[++argument]);
        } else if (value == "--refinement") {
            refinement = true;
        } else if (value == "--metallib" && argument + 1 < argc) {
            metallibPath = argv[++argument];
        } else if (value == "--dump-obj" && argument + 1 < argc) {
            objPath = argv[++argument];
        } else if (value == "--help") {
            std::cout
                << "usage: numi-solver-braided-bag "
                   "[--environments N] [--steps N] [--replays N] "
                   "[--timestep DT] [--preset baseline|stiff-braid|"
                   "low-cfm|high-friction] [--refinement] "
                   "[--metallib PATH] [--dump-obj PATH]\n";
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + std::string(value));
        }
    }
    environmentCount = std::max<std::size_t>(environmentCount, 1u);
    stepCount = std::max<std::uint32_t>(stepCount, 1u);
    replayCount = std::max<std::uint32_t>(replayCount, 2u);
    if (!(timestep > 0.0f) || !std::isfinite(timestep)) {
        throw std::runtime_error("timestep must be finite and positive");
    }
    if (environmentCount > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("environment count exceeds ABI");
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (device == nil || queue == nil) {
        throw std::runtime_error("no Apple Metal command queue is available");
    }
    NSError* error = nil;
    NSString* path = [NSString stringWithUTF8String:metallibPath.c_str()];
    id<MTLLibrary> library = [device
        newLibraryWithURL:[NSURL fileURLWithPath:path]
                    error:&error];
    if (library == nil) {
        throw std::runtime_error("failed to load metallib: " + errorText(error));
    }
    const Pipelines pipelines{
        makePipeline(device, library, @"numi_braided_bag_free_motion"),
        makePipeline(device, library, @"numi_braided_bag_build_contacts"),
        makePipeline(device, library, @"numi_temporal_cone_island_solve"),
        makePipeline(device, library, @"numi_temporal_cone_stream_solve"),
        makePipeline(device, library, @"numi_braided_bag_apply"),
    };
    if (pipelines.buildContacts.threadExecutionWidth != 32u ||
        pipelines.denseSolver.threadExecutionWidth != 32u ||
        pipelines.streamedSolver.threadExecutionWidth != 32u ||
        pipelines.apply.maxTotalThreadsPerThreadgroup < 64u) {
        throw std::runtime_error("braided-bag pipelines violate SIMD contract");
    }

    const InitialState initial = makeInitialState(
        environmentCount,
        preset,
        timestep
    );
    (void)runGPU(
        device,
        queue,
        pipelines,
        initial,
        std::min<std::uint32_t>(stepCount, 8u),
        NUMI_BRAIDED_BAG_PATH_DENSE
    );
    (void)runGPU(
        device,
        queue,
        pipelines,
        initial,
        std::min<std::uint32_t>(stepCount, 8u),
        NUMI_BRAIDED_BAG_PATH_STREAMED
    );

    std::vector<RunResult> denseReplays;
    std::vector<RunResult> streamedReplays;
    denseReplays.reserve(replayCount);
    streamedReplays.reserve(replayCount);
    for (std::uint32_t replay = 0u; replay < replayCount; ++replay) {
        denseReplays.push_back(runGPU(
            device,
            queue,
            pipelines,
            initial,
            stepCount,
            NUMI_BRAIDED_BAG_PATH_DENSE
        ));
        streamedReplays.push_back(runGPU(
            device,
            queue,
            pipelines,
            initial,
            stepCount,
            NUMI_BRAIDED_BAG_PATH_STREAMED
        ));
    }

    bool denseDeterministic = true;
    bool streamedDeterministic = true;
    for (std::size_t replay = 1u; replay < replayCount; ++replay) {
        denseDeterministic = denseDeterministic &&
            sameResult(denseReplays[0], denseReplays[replay]);
        streamedDeterministic = streamedDeterministic &&
            sameResult(streamedReplays[0], streamedReplays[replay]);
    }
    const bool denseStreamBitwise = sameResult(
        denseReplays[0],
        streamedReplays[0]
    );
    const double conditionUpperLimit = preset == BagPreset::lowCFM
        ? 8192.0
        : 4096.0;

    std::uint32_t refinedStepCount = 0u;
    std::uint64_t refinedFailedSteps = 0u;
    std::uint32_t refinedEscapedMask = 0u;
    bool refinementDeterministic = true;
    bool refinementAccepted = true;
    double refinementPositionLInfDelta = 0.0;
    if (refinement) {
        if (stepCount > std::numeric_limits<std::uint32_t>::max() / 2u) {
            throw std::runtime_error("refined step count exceeds ABI");
        }
        refinedStepCount = 2u * stepCount;
        const InitialState refinedInitial = makeInitialState(
            environmentCount,
            preset,
            0.5f * timestep
        );
        const RunResult refinedFirst = runGPU(
            device,
            queue,
            pipelines,
            refinedInitial,
            refinedStepCount,
            NUMI_BRAIDED_BAG_PATH_STREAMED
        );
        const RunResult refinedSecond = runGPU(
            device,
            queue,
            pipelines,
            refinedInitial,
            refinedStepCount,
            NUMI_BRAIDED_BAG_PATH_STREAMED
        );
        refinementDeterministic = sameResult(refinedFirst, refinedSecond);
        refinementPositionLInfDelta = maximumPositionLInfDelta(
            streamedReplays[0],
            refinedFirst
        );
        for (const auto& status : refinedFirst.statuses) {
            refinedFailedSteps += status.control.x;
            refinedEscapedMask |= status.control.w;
            refinementAccepted = refinementAccepted &&
                status.control.y == refinedStepCount &&
                status.solverMetrics.x <= 0.06f &&
                status.solverMetrics.y <= 0.50f &&
                status.solverMetrics.z <= 2.0e-6f &&
                status.solverMetrics.w <= 2.0e-6f &&
                status.certificateMetrics.x <= 2.0e-6f &&
                status.certificateMetrics.y <= 2.0e-4f &&
                std::isfinite(status.certificateMetrics.z) &&
                std::isfinite(status.certificateMetrics.w) &&
                status.certificateMetrics.w <= conditionUpperLimit;
        }
        refinementAccepted = refinementAccepted &&
            refinedFailedSteps == 0u &&
            refinedEscapedMask == 0u &&
            refinementDeterministic &&
            refinementPositionLInfDelta <= 0.075;
    }

    std::uint64_t failedSteps = 0u;
    std::uint32_t escapedMask = 0u;
    std::uint32_t maximumIterations = 0u;
    double maximumPenetration = 0.0;
    double maximumStretch = 0.0;
    double maximumKKT = 0.0;
    double maximumCone = 0.0;
    double maximumComplementarity = 0.0;
    double maximumPositiveObjective = 0.0;
    double maximumOperatorInfinityNorm = 0.0;
    double maximumConditionUpper = 0.0;
    double maximumBallRadius = 0.0;
    double minimumBallHeight = std::numeric_limits<double>::infinity();
    double maximumBallSpeed = 0.0;
    double maximumNodeSpeed = 0.0;
    std::uint32_t minimumActiveContacts =
        std::numeric_limits<std::uint32_t>::max();
    std::uint32_t maximumActiveContacts = 0u;
    std::uint64_t accumulatedActiveContacts = 0u;
    std::uint32_t maximumActiveBlocks = 0u;
    bool completed = true;
    for (const auto& status : streamedReplays[0].statuses) {
        failedSteps += status.control.x;
        completed = completed && status.control.y == stepCount;
        maximumIterations = std::max(maximumIterations, status.control.z);
        escapedMask |= status.control.w;
        maximumPenetration = std::max<double>(
            maximumPenetration,
            status.solverMetrics.x
        );
        maximumStretch = std::max<double>(
            maximumStretch,
            status.solverMetrics.y
        );
        maximumKKT = std::max<double>(maximumKKT, status.solverMetrics.z);
        maximumCone = std::max<double>(maximumCone, status.solverMetrics.w);
        maximumComplementarity = std::max<double>(
            maximumComplementarity,
            status.certificateMetrics.x
        );
        maximumPositiveObjective = std::max<double>(
            maximumPositiveObjective,
            status.certificateMetrics.y
        );
        maximumOperatorInfinityNorm = std::max<double>(
            maximumOperatorInfinityNorm,
            status.certificateMetrics.z
        );
        maximumConditionUpper = std::max<double>(
            maximumConditionUpper,
            status.certificateMetrics.w
        );
        maximumBallRadius = std::max<double>(
            maximumBallRadius,
            status.physicalMetrics.x
        );
        minimumBallHeight = std::min<double>(
            minimumBallHeight,
            status.physicalMetrics.y
        );
        maximumBallSpeed = std::max<double>(
            maximumBallSpeed,
            status.physicalMetrics.z
        );
        maximumNodeSpeed = std::max<double>(
            maximumNodeSpeed,
            status.physicalMetrics.w
        );
        minimumActiveContacts = std::min(
            minimumActiveContacts,
            status.topologyMetrics.x
        );
        maximumActiveContacts = std::max(
            maximumActiveContacts,
            status.topologyMetrics.y
        );
        accumulatedActiveContacts += status.topologyMetrics.z;
        maximumActiveBlocks = std::max(
            maximumActiveBlocks,
            status.topologyMetrics.w
        );
    }
    double denseSeconds = 0.0;
    double streamedSeconds = 0.0;
    for (std::size_t replay = 0u; replay < replayCount; ++replay) {
        denseSeconds += denseReplays[replay].seconds;
        streamedSeconds += streamedReplays[replay].seconds;
    }
    denseSeconds /= replayCount;
    streamedSeconds /= replayCount;
    const double speedup = streamedSeconds > 0.0
        ? denseSeconds / streamedSeconds
        : 0.0;
    const double environmentMicrosteps =
        static_cast<double>(environmentCount) * stepCount;
    const double averageActiveContacts =
        static_cast<double>(accumulatedActiveContacts) /
        environmentMicrosteps;
    const std::uint64_t denseOperatorBytes =
        environmentCount * NUMI_TEMPORAL_CONE_ISLAND_MATRIX_ELEMENTS *
        sizeof(float);
    const std::uint64_t streamedOperatorBytes =
        initial.rowOffsets.size() * sizeof(std::uint32_t) +
        initial.columnIndices.size() * sizeof(std::uint32_t) +
        environmentCount * NUMI_BRAIDED_BAG_STREAM_BLOCK_COUNT * 9u *
            sizeof(float);
    const double operatorMemoryRatio = static_cast<double>(
        streamedOperatorBytes
    ) / static_cast<double>(denseOperatorBytes);
    const bool passed =
        completed && failedSteps == 0u && escapedMask == 0u &&
        maximumPenetration <= 0.06 && maximumStretch <= 0.50 &&
        maximumKKT <= 2.0e-6 && maximumCone <= 2.0e-6 &&
        maximumComplementarity <= 2.0e-6 &&
        maximumPositiveObjective <= 2.0e-4 &&
        std::isfinite(maximumOperatorInfinityNorm) &&
        std::isfinite(maximumConditionUpper) &&
        maximumConditionUpper <= conditionUpperLimit &&
        minimumActiveContacts <= maximumActiveContacts &&
        maximumActiveContacts <= NUMI_BRAIDED_BAG_CONTACT_COUNT &&
        maximumActiveBlocks <= NUMI_BRAIDED_BAG_STREAM_BLOCK_COUNT &&
        denseDeterministic && streamedDeterministic && denseStreamBitwise &&
        (environmentCount < 4u || speedup > 1.0) && refinementAccepted;

    if (!objPath.empty()) {
        dumpOBJ(objPath, initial, streamedReplays[0]);
    }
    std::cout << std::fixed << std::setprecision(9)
              << "device=" << device.name.UTF8String << '\n'
              << "preset=" << presetName(preset) << ' '
              << "environments=" << environmentCount
              << " steps=" << stepCount
              << " simulated_seconds="
              << stepCount * initial.config.timing.x
              << " braid_nodes=" << NUMI_BRAIDED_BAG_NODE_COUNT
              << " braid_edges=" << NUMI_BRAIDED_BAG_EDGE_COUNT
              << " balls=" << NUMI_BRAIDED_BAG_BALL_COUNT
              << " candidate_contacts_per_step="
              << NUMI_BRAIDED_BAG_CONTACT_COUNT
              << " failed_steps=" << failedSteps
              << " escaped_mask=" << escapedMask << '\n'
              << "min_active_contacts=" << minimumActiveContacts
              << " max_active_contacts=" << maximumActiveContacts
              << " average_active_contacts=" << averageActiveContacts
              << " max_active_blocks=" << maximumActiveBlocks << '\n'
              << "max_penetration=" << maximumPenetration
              << " max_relative_stretch=" << maximumStretch
              << " max_kkt_residual=" << maximumKKT
              << " max_cone_violation=" << maximumCone
              << " max_complementarity_residual="
              << maximumComplementarity
              << " max_positive_objective=" << maximumPositiveObjective
              << " max_operator_infinity_norm="
              << maximumOperatorInfinityNorm
              << " max_condition_upper=" << maximumConditionUpper
              << " condition_upper_limit=" << conditionUpperLimit
              << " max_iterations=" << maximumIterations << '\n'
              << "max_ball_radius=" << maximumBallRadius
              << " min_ball_height=" << minimumBallHeight
              << " max_ball_speed=" << maximumBallSpeed
              << " max_node_speed=" << maximumNodeSpeed << '\n'
              << "refinement=" << (refinement ? "true" : "false")
              << " refined_steps=" << refinedStepCount
              << " refined_failed_steps=" << refinedFailedSteps
              << " refined_escaped_mask=" << refinedEscapedMask
              << " refinement_deterministic="
              << (refinementDeterministic ? "true" : "false")
              << " refinement_position_linf_delta="
              << refinementPositionLInfDelta << '\n'
              << "dense_deterministic="
              << (denseDeterministic ? "true" : "false")
              << " streamed_deterministic="
              << (streamedDeterministic ? "true" : "false")
              << " dense_stream_bitwise="
              << (denseStreamBitwise ? "true" : "false")
              << " state_hash=0x" << std::hex << hashResult(streamedReplays[0])
              << std::dec << '\n'
              << "dense_gpu_seconds=" << denseSeconds
              << " streamed_gpu_seconds=" << streamedSeconds
              << " dense_to_stream_speedup=" << speedup
              << " dense_environment_microsteps_per_second="
              << environmentMicrosteps / denseSeconds
              << " streamed_environment_microsteps_per_second="
              << environmentMicrosteps / streamedSeconds << '\n'
              << "dense_operator_bytes=" << denseOperatorBytes
              << " streamed_operator_bytes=" << streamedOperatorBytes
              << " stream_to_dense_operator_memory=" << operatorMemoryRatio
              << " dense_solver_threadgroup_memory="
              << pipelines.denseSolver.staticThreadgroupMemoryLength
              << " streamed_solver_threadgroup_memory="
              << pipelines.streamedSolver.staticThreadgroupMemoryLength
              << '\n'
              << "result=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}

} // namespace

int main(const int argc, const char* const* argv) {
    @autoreleasepool {
        try {
            return run(argc, argv);
        } catch (const std::exception& error) {
            std::cerr << "error: " << error.what() << '\n';
            return 1;
        }
    }
}
