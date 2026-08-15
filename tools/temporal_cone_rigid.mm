#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/temporal_cone_island.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef NUMI_TEMPORAL_CONE_METALLIB
#error "NUMI_TEMPORAL_CONE_METALLIB must name the built solver metallib"
#endif

namespace {

constexpr std::uint32_t kStatic = NUMI_TEMPORAL_CONE_RIGID_STATIC_BODY;

mr_float4 f4(float x, float y, float z, float w = 0.0f) {
    return {x, y, z, w};
}

mr_uint4 u4(
    std::uint32_t x,
    std::uint32_t y,
    std::uint32_t z,
    std::uint32_t w
) {
    return {x, y, z, w};
}

std::string errorText(NSError* error) {
    return error == nil
        ? std::string("unknown Metal error")
        : std::string(error.localizedDescription.UTF8String);
}

struct Batch {
    std::vector<NumiTemporalConeRigidHeader> rigidHeaders;
    std::vector<NumiTemporalConeIntegrationHeader> integrationHeaders;
    std::vector<NumiTemporalConeAssemblyHeader> assemblyHeaders;
    std::vector<NumiTemporalConeRigidBody> bodies;
    std::vector<NumiTemporalConeRigidPose> poses;
    std::vector<NumiTemporalConeRigidContact> rigidContacts;
    std::vector<NumiTemporalConeAssemblyContactSpan> spans;
    std::vector<NumiTemporalConeAssemblyTerm> terms;
    std::vector<float> jacobians;
    std::vector<float> responses;
    std::vector<NumiTemporalConeIslandContact> solverContacts;
    std::vector<float> regularization;
    std::vector<std::uint32_t> rowOffsets;
    std::vector<std::uint32_t> columns;
    std::vector<bool> expectedFailure;
    std::size_t validContactCount = 0u;
    std::size_t validBodyCount = 0u;
};

struct Result {
    std::vector<NumiTemporalConeAssemblyContactSpan> spans;
    std::vector<NumiTemporalConeAssemblyTerm> terms;
    std::vector<float> jacobians;
    std::vector<float> responses;
    std::vector<NumiTemporalConeIslandContact> solverContacts;
    std::vector<NumiTemporalConeRigidStatus> responseStatuses;
    std::vector<NumiTemporalConeStreamHeader> streamHeaders;
    std::vector<float> blocks;
    std::vector<NumiTemporalConeAssemblyStatus> assemblyStatuses;
    std::vector<mr_float4> impulses;
    std::vector<NumiTemporalConeIslandStatus> solverStatuses;
    std::vector<NumiTemporalConeRigidBody> outputBodies;
    std::vector<NumiTemporalConeRigidStatus> publishStatuses;
    std::vector<NumiTemporalConeRigidPose> outputPoses;
    std::vector<NumiTemporalConeIntegrationStatus> integrationStatuses;
    double seconds = 0.0;
};

NumiTemporalConeRigidBody body(
    const mr_float4 linearAndInverseMass,
    const mr_float4 angular,
    const float inverseInertiaX = 1.0f,
    const float inverseInertiaY = 1.0f,
    const float inverseInertiaZ = 1.0f
) {
    NumiTemporalConeRigidBody value = {};
    value.linearVelocityAndInverseMass = linearAndInverseMass;
    value.angularVelocity = angular;
    value.inverseInertiaRow0 = f4(inverseInertiaX, 0.0f, 0.0f);
    value.inverseInertiaRow1 = f4(0.0f, inverseInertiaY, 0.0f);
    value.inverseInertiaRow2 = f4(0.0f, 0.0f, inverseInertiaZ);
    return value;
}

NumiTemporalConeRigidContact contact(
    const std::uint32_t bodyA,
    const std::uint32_t bodyB,
    const mr_float4 offsetA = {},
    const mr_float4 offsetB = {},
    const float frictionU = 0.0f,
    const float frictionV = 0.0f
) {
    NumiTemporalConeRigidContact value = {};
    value.bodies = u4(bodyA, bodyB, 0u, 0u);
    value.offsetA = offsetA;
    value.offsetB = offsetB;
    value.normalAndFrictionU = f4(1.0f, 0.0f, 0.0f, frictionU);
    value.tangentUAndFrictionV = f4(0.0f, 1.0f, 0.0f, frictionV);
    value.tangentVAndMaximumNormal = f4(0.0f, 0.0f, 1.0f, 0.0f);
    return value;
}

bool shareBody(
    const NumiTemporalConeRigidContact& first,
    const NumiTemporalConeRigidContact& second
) {
    for (const std::uint32_t a : {first.bodies.x, first.bodies.y}) {
        if (a == kStatic) {
            continue;
        }
        if (a == second.bodies.x || a == second.bodies.y) {
            return true;
        }
    }
    return false;
}

void appendProblem(Batch& batch, const std::size_t problem, const bool invalid) {
    const std::size_t bodyBase = batch.bodies.size();
    const std::size_t contactBase = batch.rigidContacts.size();
    const std::size_t rowBase = batch.rowOffsets.size();
    const std::size_t blockBase = batch.columns.size();
    const std::size_t regularizationBase = batch.regularization.size();
    std::vector<NumiTemporalConeRigidBody> localBodies;
    std::vector<NumiTemporalConeRigidContact> localContacts;
    float regularization = 0.02f;

    if (problem == 0u) {
        // Two redundant contacts against the same unit-mass body produce
        // W_n=[[2,1],[1,2]], lambda=(1/3,1/3), and v_x'=1/3.
        localBodies.push_back(body(f4(1.0f, 0.0f, 0.0f, 1.0f), {}));
        localContacts.push_back(contact(0u, kStatic));
        localContacts.push_back(contact(0u, kStatic));
        regularization = 1.0f;
    } else if (problem % 31u == 0u) {
        localBodies.push_back(body(
            f4(0.7f + 0.01f * static_cast<float>(problem % 7u),
               0.1f, -0.04f, 0.8f),
            f4(0.02f, -0.03f, 0.01f),
            0.7f, 0.9f, 1.1f
        ));
        for (std::uint32_t index = 0u;
             index < NUMI_TEMPORAL_CONE_ISLAND_MAX_CONTACTS;
             ++index) {
            const float offset = 0.005f * static_cast<float>(index) - 0.08f;
            localContacts.push_back(contact(
                0u,
                kStatic,
                f4(0.0f, offset, 0.5f * offset),
                {},
                0.35f,
                0.5f
            ));
        }
        regularization = 0.1f;
    } else if (problem % 5u == 1u) {
        localBodies.push_back(body(f4(0.5f, 0.0f, 0.0f, 1.0f), {}));
        localBodies.push_back(body(f4(-0.5f, 0.0f, 0.0f, 1.0f), {}));
        localContacts.push_back(contact(0u, 1u, {}, {}, 0.4f, 0.4f));
    } else if (problem % 5u == 2u) {
        localBodies.push_back(body(
            f4(0.6f, 0.15f, 0.0f, 0.75f),
            f4(0.0f, 0.0f, 0.2f),
            0.5f, 0.8f, 1.2f
        ));
        localContacts.push_back(contact(
            0u, kStatic, f4(0.0f, 0.4f, 0.0f), {}, 0.6f, 0.45f
        ));
    } else if (problem % 5u == 3u) {
        constexpr std::uint32_t kBodies = 8u;
        for (std::uint32_t index = 0u; index < kBodies; ++index) {
            localBodies.push_back(body(
                f4(
                    0.7f - 0.2f * static_cast<float>(index),
                    0.02f * static_cast<float>(index % 3u),
                    0.0f,
                    0.7f + 0.05f * static_cast<float>(index)
                ),
                {},
                0.6f + 0.03f * static_cast<float>(index),
                0.8f,
                1.0f
            ));
        }
        for (std::uint32_t index = 0u; index + 1u < kBodies; ++index) {
            localContacts.push_back(contact(
                index, index + 1u, {}, {}, 0.3f, 0.3f
            ));
        }
    } else if (problem % 5u == 4u) {
        localBodies.push_back(body(
            f4(1.0f, 0.35f, -0.1f, 1.25f),
            f4(0.05f, -0.02f, 0.08f),
            0.9f, 0.7f, 0.8f
        ));
        localContacts.push_back(contact(
            0u, kStatic, f4(0.0f, 0.25f, 0.1f), {}, 0.7f, 0.5f
        ));
        localContacts.push_back(contact(
            0u, kStatic, f4(0.0f, -0.2f, -0.15f), {}, 0.7f, 0.5f
        ));
    } else {
        localBodies.push_back(body(f4(0.8f, 0.0f, 0.0f, 1.0f), {}));
        localContacts.push_back(contact(0u, kStatic));
    }

    if (invalid) {
        localContacts.front().tangentVAndMaximumNormal =
            f4(0.0f, 1.0f, 0.0f, 0.0f);
    }

    const std::uint32_t bodyCount =
        static_cast<std::uint32_t>(localBodies.size());
    const std::uint32_t contactCount =
        static_cast<std::uint32_t>(localContacts.size());
    batch.bodies.insert(batch.bodies.end(), localBodies.begin(), localBodies.end());
    for (std::uint32_t index = 0u; index < bodyCount; ++index) {
        NumiTemporalConeRigidPose pose = {};
        pose.position = f4(
            0.1f * static_cast<float>(index),
            0.01f * static_cast<float>(problem % 11u),
            -0.03f * static_cast<float>(index)
        );
        pose.orientation = f4(0.0f, 0.0f, 0.0f, 1.0f);
        batch.poses.push_back(pose);
    }
    batch.rigidContacts.insert(
        batch.rigidContacts.end(), localContacts.begin(), localContacts.end()
    );
    batch.spans.resize(batch.spans.size() + contactCount);
    batch.terms.resize(batch.terms.size() + 2u * contactCount);
    batch.jacobians.resize(
        batch.jacobians.size() + 2u * contactCount *
            NUMI_TEMPORAL_CONE_RIGID_VALUES_PER_TERM
    );
    batch.responses.resize(
        batch.responses.size() + 2u * contactCount *
            NUMI_TEMPORAL_CONE_RIGID_VALUES_PER_TERM
    );
    batch.solverContacts.resize(batch.solverContacts.size() + contactCount);
    for (std::uint32_t index = 0u; index < contactCount; ++index) {
        for (std::uint32_t element = 0u; element < 9u; ++element) {
            batch.regularization.push_back(
                element % 4u == 0u ? regularization : 0.0f
            );
        }
    }
    batch.rowOffsets.push_back(0u);
    std::uint32_t relativeBlocks = 0u;
    for (std::uint32_t row = 0u; row < contactCount; ++row) {
        for (std::uint32_t column = 0u; column < contactCount; ++column) {
            if (row == column || shareBody(localContacts[row], localContacts[column])) {
                batch.columns.push_back(column);
                ++relativeBlocks;
            }
        }
        batch.rowOffsets.push_back(relativeBlocks);
    }

    NumiTemporalConeRigidHeader rigidHeader = {};
    rigidHeader.control = u4(
        NUMI_TEMPORAL_CONE_RIGID_ABI_VERSION, bodyCount, contactCount, 0u
    );
    rigidHeader.inputRanges = u4(
        static_cast<std::uint32_t>(bodyBase),
        static_cast<std::uint32_t>(contactBase), 0u, 0u
    );
    rigidHeader.responseRanges = u4(
        static_cast<std::uint32_t>(contactBase),
        static_cast<std::uint32_t>(2u * contactBase),
        static_cast<std::uint32_t>(
            2u * contactBase * NUMI_TEMPORAL_CONE_RIGID_VALUES_PER_TERM
        ),
        static_cast<std::uint32_t>(
            2u * contactBase * NUMI_TEMPORAL_CONE_RIGID_VALUES_PER_TERM
        )
    );
    rigidHeader.solverRanges = u4(
        static_cast<std::uint32_t>(contactBase),
        static_cast<std::uint32_t>(bodyBase), 0u, 0u
    );
    batch.rigidHeaders.push_back(rigidHeader);

    NumiTemporalConeIntegrationHeader integrationHeader = {};
    integrationHeader.control = u4(
        NUMI_TEMPORAL_CONE_INTEGRATION_ABI_VERSION, bodyCount, 0u, 0u
    );
    integrationHeader.ranges = u4(
        static_cast<std::uint32_t>(bodyBase),
        static_cast<std::uint32_t>(bodyBase),
        static_cast<std::uint32_t>(bodyBase),
        0u
    );
    integrationHeader.timestep = f4(1.0f / 240.0f, 0.0f, 0.0f, 0.0f);
    batch.integrationHeaders.push_back(integrationHeader);

    NumiTemporalConeAssemblyHeader assemblyHeader = {};
    assemblyHeader.control = u4(
        NUMI_TEMPORAL_CONE_ASSEMBLY_ABI_VERSION,
        contactCount,
        2u,
        1024u
    );
    assemblyHeader.outputRanges = u4(
        static_cast<std::uint32_t>(contactBase),
        static_cast<std::uint32_t>(rowBase),
        static_cast<std::uint32_t>(blockBase),
        relativeBlocks
    );
    assemblyHeader.inputRanges = u4(
        static_cast<std::uint32_t>(contactBase),
        static_cast<std::uint32_t>(regularizationBase), 0u, 0u
    );
    assemblyHeader.tolerances = f4(2.0e-6f, 2.0e-6f, 1.0f, 0.0f);
    batch.assemblyHeaders.push_back(assemblyHeader);
    batch.expectedFailure.push_back(invalid);
    if (!invalid) {
        batch.validContactCount += contactCount;
        batch.validBodyCount += bodyCount;
    }
}

Batch makeBatch(const std::size_t problemCount) {
    Batch batch;
    for (std::size_t problem = 0u; problem < problemCount; ++problem) {
        appendProblem(batch, problem, problem + 1u == problemCount);
    }
    return batch;
}

template <typename T>
bool exactVector(const std::vector<T>& first, const std::vector<T>& second) {
    return first.size() == second.size() &&
        std::memcmp(first.data(), second.data(), first.size() * sizeof(T)) == 0;
}

Result runGPU(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    const std::array<id<MTLComputePipelineState>, 5>& pipelines,
    const Batch& batch
) {
    const std::uint32_t problemCount =
        static_cast<std::uint32_t>(batch.rigidHeaders.size());
    const auto makeBytes = [&](const auto& values) -> id<MTLBuffer> {
        return [device newBufferWithBytes:values.data()
                                  length:values.size() * sizeof(values.front())
                                 options:MTLResourceStorageModeShared];
    };
    const auto makeOutput = [&](const std::size_t bytes) -> id<MTLBuffer> {
        id<MTLBuffer> value = [device newBufferWithLength:bytes
                                                  options:MTLResourceStorageModeShared];
        if (value != nil) {
            std::memset(value.contents, 0, value.length);
        }
        return value;
    };
    id<MTLBuffer> rigidHeaderBuffer = makeBytes(batch.rigidHeaders);
    id<MTLBuffer> integrationHeaderBuffer = makeBytes(batch.integrationHeaders);
    id<MTLBuffer> assemblyHeaderBuffer = makeBytes(batch.assemblyHeaders);
    id<MTLBuffer> bodyBuffer = makeBytes(batch.bodies);
    id<MTLBuffer> poseBuffer = makeBytes(batch.poses);
    id<MTLBuffer> rigidContactBuffer = makeBytes(batch.rigidContacts);
    id<MTLBuffer> spanBuffer = makeOutput(batch.spans.size() * sizeof(batch.spans.front()));
    id<MTLBuffer> termBuffer = makeOutput(batch.terms.size() * sizeof(batch.terms.front()));
    id<MTLBuffer> jacobianBuffer = makeOutput(batch.jacobians.size() * sizeof(float));
    id<MTLBuffer> responseBuffer = makeOutput(batch.responses.size() * sizeof(float));
    id<MTLBuffer> solverContactBuffer = makeOutput(
        batch.solverContacts.size() * sizeof(batch.solverContacts.front())
    );
    id<MTLBuffer> responseStatusBuffer = makeOutput(
        problemCount * sizeof(NumiTemporalConeRigidStatus)
    );
    id<MTLBuffer> regularizationBuffer = makeBytes(batch.regularization);
    id<MTLBuffer> rowBuffer = makeBytes(batch.rowOffsets);
    id<MTLBuffer> columnBuffer = makeBytes(batch.columns);
    id<MTLBuffer> blockBuffer = makeOutput(batch.columns.size() * 9u * sizeof(float));
    id<MTLBuffer> streamHeaderBuffer = makeOutput(
        problemCount * sizeof(NumiTemporalConeStreamHeader)
    );
    id<MTLBuffer> assemblyStatusBuffer = makeOutput(
        problemCount * sizeof(NumiTemporalConeAssemblyStatus)
    );
    id<MTLBuffer> impulseBuffer = makeOutput(
        batch.rigidContacts.size() * sizeof(mr_float4)
    );
    id<MTLBuffer> solverStatusBuffer = makeOutput(
        problemCount * sizeof(NumiTemporalConeIslandStatus)
    );
    id<MTLBuffer> outputBodyBuffer = makeOutput(
        batch.bodies.size() * sizeof(NumiTemporalConeRigidBody)
    );
    id<MTLBuffer> publishStatusBuffer = makeOutput(
        problemCount * sizeof(NumiTemporalConeRigidStatus)
    );
    id<MTLBuffer> outputPoseBuffer = makeOutput(
        batch.poses.size() * sizeof(NumiTemporalConeRigidPose)
    );
    id<MTLBuffer> integrationStatusBuffer = makeOutput(
        problemCount * sizeof(NumiTemporalConeIntegrationStatus)
    );
    if (rigidHeaderBuffer == nil || integrationHeaderBuffer == nil ||
        assemblyHeaderBuffer == nil || bodyBuffer == nil || poseBuffer == nil ||
        rigidContactBuffer == nil || spanBuffer == nil ||
        termBuffer == nil || jacobianBuffer == nil || responseBuffer == nil ||
        solverContactBuffer == nil || responseStatusBuffer == nil ||
        regularizationBuffer == nil || rowBuffer == nil || columnBuffer == nil ||
        blockBuffer == nil || streamHeaderBuffer == nil ||
        assemblyStatusBuffer == nil || impulseBuffer == nil ||
        solverStatusBuffer == nil || outputBodyBuffer == nil ||
        publishStatusBuffer == nil || outputPoseBuffer == nil ||
        integrationStatusBuffer == nil) {
        throw std::runtime_error("failed to allocate rigid qualification buffers");
    }

    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> response = [command computeCommandEncoder];
    [response setComputePipelineState:pipelines[0]];
    [response setBuffer:rigidHeaderBuffer offset:0 atIndex:0];
    [response setBuffer:bodyBuffer offset:0 atIndex:1];
    [response setBuffer:rigidContactBuffer offset:0 atIndex:2];
    [response setBuffer:spanBuffer offset:0 atIndex:3];
    [response setBuffer:termBuffer offset:0 atIndex:4];
    [response setBuffer:jacobianBuffer offset:0 atIndex:5];
    [response setBuffer:responseBuffer offset:0 atIndex:6];
    [response setBuffer:solverContactBuffer offset:0 atIndex:7];
    [response setBuffer:responseStatusBuffer offset:0 atIndex:8];
    [response setBytes:&problemCount length:sizeof(problemCount) atIndex:9];
    const mr_uint4 structural = u4(
        static_cast<std::uint32_t>(batch.bodies.size()),
        static_cast<std::uint32_t>(batch.rigidContacts.size()),
        static_cast<std::uint32_t>(batch.spans.size()),
        static_cast<std::uint32_t>(batch.terms.size())
    );
    const mr_uint4 values = u4(
        static_cast<std::uint32_t>(batch.jacobians.size()),
        static_cast<std::uint32_t>(batch.responses.size()),
        static_cast<std::uint32_t>(batch.solverContacts.size()), 0u
    );
    [response setBytes:&structural length:sizeof(structural) atIndex:10];
    [response setBytes:&values length:sizeof(values) atIndex:11];
    [response dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
                  threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [response endEncoding];

    id<MTLComputeCommandEncoder> assembly = [command computeCommandEncoder];
    [assembly setComputePipelineState:pipelines[1]];
    [assembly setBuffer:assemblyHeaderBuffer offset:0 atIndex:0];
    [assembly setBuffer:spanBuffer offset:0 atIndex:1];
    [assembly setBuffer:termBuffer offset:0 atIndex:2];
    [assembly setBuffer:jacobianBuffer offset:0 atIndex:3];
    [assembly setBuffer:responseBuffer offset:0 atIndex:4];
    [assembly setBuffer:regularizationBuffer offset:0 atIndex:5];
    [assembly setBuffer:rowBuffer offset:0 atIndex:6];
    [assembly setBuffer:columnBuffer offset:0 atIndex:7];
    [assembly setBuffer:blockBuffer offset:0 atIndex:8];
    [assembly setBuffer:streamHeaderBuffer offset:0 atIndex:9];
    [assembly setBuffer:assemblyStatusBuffer offset:0 atIndex:10];
    [assembly setBytes:&problemCount length:sizeof(problemCount) atIndex:11];
    const mr_uint4 assemblyInputs = u4(
        static_cast<std::uint32_t>(batch.spans.size()),
        static_cast<std::uint32_t>(batch.terms.size()),
        static_cast<std::uint32_t>(batch.jacobians.size()),
        static_cast<std::uint32_t>(batch.responses.size())
    );
    const mr_uint4 assemblyOutputs = u4(
        static_cast<std::uint32_t>(batch.rowOffsets.size()),
        static_cast<std::uint32_t>(batch.columns.size()),
        static_cast<std::uint32_t>(batch.regularization.size()),
        static_cast<std::uint32_t>(batch.columns.size() * 9u)
    );
    [assembly setBytes:&assemblyInputs length:sizeof(assemblyInputs) atIndex:12];
    [assembly setBytes:&assemblyOutputs length:sizeof(assemblyOutputs) atIndex:13];
    [assembly dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
                  threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [assembly endEncoding];

    id<MTLComputeCommandEncoder> solver = [command computeCommandEncoder];
    [solver setComputePipelineState:pipelines[2]];
    [solver setBuffer:streamHeaderBuffer offset:0 atIndex:0];
    [solver setBuffer:rowBuffer offset:0 atIndex:1];
    [solver setBuffer:columnBuffer offset:0 atIndex:2];
    [solver setBuffer:blockBuffer offset:0 atIndex:3];
    [solver setBuffer:solverContactBuffer offset:0 atIndex:4];
    [solver setBuffer:impulseBuffer offset:0 atIndex:5];
    [solver setBuffer:solverStatusBuffer offset:0 atIndex:6];
    [solver setBytes:&problemCount length:sizeof(problemCount) atIndex:7];
    const mr_uint4 solverCapacities = u4(
        static_cast<std::uint32_t>(batch.solverContacts.size()),
        static_cast<std::uint32_t>(batch.rowOffsets.size()),
        static_cast<std::uint32_t>(batch.columns.size()),
        static_cast<std::uint32_t>(batch.solverContacts.size())
    );
    [solver setBytes:&solverCapacities length:sizeof(solverCapacities) atIndex:8];
    [solver dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [solver endEncoding];

    id<MTLComputeCommandEncoder> publish = [command computeCommandEncoder];
    [publish setComputePipelineState:pipelines[3]];
    [publish setBuffer:rigidHeaderBuffer offset:0 atIndex:0];
    [publish setBuffer:bodyBuffer offset:0 atIndex:1];
    [publish setBuffer:rigidContactBuffer offset:0 atIndex:2];
    [publish setBuffer:impulseBuffer offset:0 atIndex:3];
    [publish setBuffer:responseStatusBuffer offset:0 atIndex:4];
    [publish setBuffer:solverStatusBuffer offset:0 atIndex:5];
    [publish setBuffer:outputBodyBuffer offset:0 atIndex:6];
    [publish setBuffer:publishStatusBuffer offset:0 atIndex:7];
    [publish setBytes:&problemCount length:sizeof(problemCount) atIndex:8];
    const mr_uint4 publishCapacities = u4(
        static_cast<std::uint32_t>(batch.bodies.size()),
        static_cast<std::uint32_t>(batch.rigidContacts.size()),
        static_cast<std::uint32_t>(batch.rigidContacts.size()),
        static_cast<std::uint32_t>(batch.bodies.size())
    );
    [publish setBytes:&publishCapacities length:sizeof(publishCapacities) atIndex:9];
    [publish dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
                 threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [publish endEncoding];

    id<MTLComputeCommandEncoder> integrate = [command computeCommandEncoder];
    [integrate setComputePipelineState:pipelines[4]];
    [integrate setBuffer:integrationHeaderBuffer offset:0 atIndex:0];
    [integrate setBuffer:poseBuffer offset:0 atIndex:1];
    [integrate setBuffer:outputBodyBuffer offset:0 atIndex:2];
    [integrate setBuffer:publishStatusBuffer offset:0 atIndex:3];
    [integrate setBuffer:outputPoseBuffer offset:0 atIndex:4];
    [integrate setBuffer:integrationStatusBuffer offset:0 atIndex:5];
    [integrate setBytes:&problemCount length:sizeof(problemCount) atIndex:6];
    const mr_uint4 integrationCapacities = u4(
        static_cast<std::uint32_t>(batch.poses.size()),
        static_cast<std::uint32_t>(batch.bodies.size()),
        static_cast<std::uint32_t>(batch.poses.size()),
        0u
    );
    [integrate setBytes:&integrationCapacities
                  length:sizeof(integrationCapacities)
                 atIndex:7];
    [integrate dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
                   threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [integrate endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted || command.error != nil) {
        throw std::runtime_error("Metal rigid chain failed: " + errorText(command.error));
    }

    Result result;
    const auto copy = [](auto& destination, id<MTLBuffer> source) {
        using Value = typename std::decay_t<decltype(destination)>::value_type;
        const auto* begin = static_cast<const Value*>(source.contents);
        destination.assign(begin, begin + source.length / sizeof(Value));
    };
    copy(result.spans, spanBuffer);
    copy(result.terms, termBuffer);
    copy(result.jacobians, jacobianBuffer);
    copy(result.responses, responseBuffer);
    copy(result.solverContacts, solverContactBuffer);
    copy(result.responseStatuses, responseStatusBuffer);
    copy(result.streamHeaders, streamHeaderBuffer);
    copy(result.blocks, blockBuffer);
    copy(result.assemblyStatuses, assemblyStatusBuffer);
    copy(result.impulses, impulseBuffer);
    copy(result.solverStatuses, solverStatusBuffer);
    copy(result.outputBodies, outputBodyBuffer);
    copy(result.publishStatuses, publishStatusBuffer);
    copy(result.outputPoses, outputPoseBuffer);
    copy(result.integrationStatuses, integrationStatusBuffer);
    if (command.GPUEndTime >= command.GPUStartTime) {
        result.seconds = command.GPUEndTime - command.GPUStartTime;
    }
    return result;
}

struct FreeFlightResult {
    NumiTemporalConeRigidPose pose = {};
    NumiTemporalConeIntegrationStatus status = {};
    double seconds = 0.0;
};

FreeFlightResult runFreeFlight(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> integrationPipeline,
    const std::uint32_t stepCount
) {
    NumiTemporalConeIntegrationHeader header = {};
    header.control = u4(NUMI_TEMPORAL_CONE_INTEGRATION_ABI_VERSION, 1u, 0u, 0u);
    header.ranges = u4(0u, 0u, 0u, 0u);
    header.timestep = f4(1.0f / 240.0f, 0.0f, 0.0f, 0.0f);
    NumiTemporalConeRigidPose initial = {};
    initial.position = f4(0.2f, -0.1f, 0.4f, 7.0f);
    initial.orientation = f4(0.0f, 0.0f, 0.0f, 1.0f);
    const NumiTemporalConeRigidBody velocity = body(
        f4(1.0f, -0.5f, 0.25f, 1.0f),
        f4(0.15f, -0.2f, 0.4f)
    );
    NumiTemporalConeRigidStatus upstream = {};
    upstream.control = u4(NUMI_TEMPORAL_CONE_RIGID_SUCCESS, 1u, 0u, 0u);
    id<MTLBuffer> headerBuffer = [device newBufferWithBytes:&header
                                                    length:sizeof(header)
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> poseA = [device newBufferWithBytes:&initial
                                              length:sizeof(initial)
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> poseB = [device newBufferWithLength:sizeof(initial)
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> velocityBuffer = [device newBufferWithBytes:&velocity
                                                      length:sizeof(velocity)
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> upstreamBuffer = [device newBufferWithBytes:&upstream
                                                      length:sizeof(upstream)
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> statusBuffer = [device
        newBufferWithLength:sizeof(NumiTemporalConeIntegrationStatus)
                    options:MTLResourceStorageModeShared];
    if (headerBuffer == nil || poseA == nil || poseB == nil ||
        velocityBuffer == nil || upstreamBuffer == nil || statusBuffer == nil) {
        throw std::runtime_error("failed to allocate free-flight buffers");
    }
    const std::uint32_t problemCount = 1u;
    const mr_uint4 capacities = u4(1u, 1u, 1u, 0u);
    id<MTLCommandBuffer> command = [queue commandBuffer];
    for (std::uint32_t step = 0u; step < stepCount; ++step) {
        const bool even = step % 2u == 0u;
        id<MTLComputeCommandEncoder> integrate = [command computeCommandEncoder];
        [integrate setComputePipelineState:integrationPipeline];
        [integrate setBuffer:headerBuffer offset:0 atIndex:0];
        [integrate setBuffer:even ? poseA : poseB offset:0 atIndex:1];
        [integrate setBuffer:velocityBuffer offset:0 atIndex:2];
        [integrate setBuffer:upstreamBuffer offset:0 atIndex:3];
        [integrate setBuffer:even ? poseB : poseA offset:0 atIndex:4];
        [integrate setBuffer:statusBuffer offset:0 atIndex:5];
        [integrate setBytes:&problemCount length:sizeof(problemCount) atIndex:6];
        [integrate setBytes:&capacities length:sizeof(capacities) atIndex:7];
        [integrate dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
                       threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
        [integrate endEncoding];
    }
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted || command.error != nil) {
        throw std::runtime_error("Metal free flight failed: " + errorText(command.error));
    }
    const id<MTLBuffer> finalPose = stepCount % 2u == 0u ? poseA : poseB;
    FreeFlightResult result;
    result.pose = *static_cast<const NumiTemporalConeRigidPose*>(finalPose.contents);
    result.status = *static_cast<const NumiTemporalConeIntegrationStatus*>(
        statusBuffer.contents
    );
    if (command.GPUEndTime >= command.GPUStartTime) {
        result.seconds = command.GPUEndTime - command.GPUStartTime;
    }
    return result;
}

std::array<double, 3> xyz(const mr_float4 value) {
    return {value.x, value.y, value.z};
}

std::array<double, 3> cross(
    const std::array<double, 3>& a,
    const std::array<double, 3>& b
) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

double dot(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

std::array<double, 3> inertiaMultiply(
    const NumiTemporalConeRigidBody& bodyValue,
    const std::array<double, 3>& value
) {
    return {
        bodyValue.inverseInertiaRow0.x * value[0] +
            bodyValue.inverseInertiaRow0.y * value[1] +
            bodyValue.inverseInertiaRow0.z * value[2],
        bodyValue.inverseInertiaRow1.x * value[0] +
            bodyValue.inverseInertiaRow1.y * value[1] +
            bodyValue.inverseInertiaRow1.z * value[2],
        bodyValue.inverseInertiaRow2.x * value[0] +
            bodyValue.inverseInertiaRow2.y * value[1] +
            bodyValue.inverseInertiaRow2.z * value[2]
    };
}

double kineticEnergy(const NumiTemporalConeRigidBody& value) {
    const auto linear = xyz(value.linearVelocityAndInverseMass);
    const auto angular = xyz(value.angularVelocity);
    const double mass = 1.0 / value.linearVelocityAndInverseMass.w;
    // Qualification bodies use diagonal inverse inertia.
    const double rotational =
        angular[0] * angular[0] / value.inverseInertiaRow0.x +
        angular[1] * angular[1] / value.inverseInertiaRow1.y +
        angular[2] * angular[2] / value.inverseInertiaRow2.z;
    return 0.5 * (mass * dot(linear, linear) + rotational);
}

std::array<double, 3> axis(
    const NumiTemporalConeRigidContact& value,
    const std::size_t index
) {
    return index == 0u ? xyz(value.normalAndFrictionU)
        : index == 1u ? xyz(value.tangentUAndFrictionV)
        : xyz(value.tangentVAndMaximumNormal);
}

double expectedCoefficient(
    const Batch& batch,
    const std::size_t problem,
    const std::size_t target,
    const std::size_t source,
    const std::size_t targetAxis,
    const std::size_t sourceAxis
) {
    const auto& header = batch.rigidHeaders[problem];
    const auto& targetContact = batch.rigidContacts[header.inputRanges.y + target];
    const auto& sourceContact = batch.rigidContacts[header.inputRanges.y + source];
    double result = target == source && targetAxis == sourceAxis
        ? batch.regularization[
            batch.assemblyHeaders[problem].inputRanges.y + 4u * targetAxis + target * 9u
        ]
        : 0.0;
    for (std::uint32_t localBody = 0u; localBody < header.control.y; ++localBody) {
        const auto role = [&](const NumiTemporalConeRigidContact& c) {
            if (c.bodies.x == localBody) {
                return std::pair{-1.0, xyz(c.offsetA)};
            }
            if (c.bodies.y == localBody) {
                return std::pair{1.0, xyz(c.offsetB)};
            }
            return std::pair{0.0, std::array<double, 3>{}};
        };
        const auto [targetSign, targetOffset] = role(targetContact);
        const auto [sourceSign, sourceOffset] = role(sourceContact);
        if (targetSign == 0.0 || sourceSign == 0.0) {
            continue;
        }
        auto targetLinear = axis(targetContact, targetAxis);
        auto sourceLinear = axis(sourceContact, sourceAxis);
        for (double& component : targetLinear) component *= targetSign;
        for (double& component : sourceLinear) component *= sourceSign;
        const auto targetAngular = cross(targetOffset, targetLinear);
        const auto sourceAngular = cross(sourceOffset, sourceLinear);
        const auto& bodyValue = batch.bodies[header.inputRanges.x + localBody];
        result += bodyValue.linearVelocityAndInverseMass.w *
            dot(targetLinear, sourceLinear) +
            dot(targetAngular, inertiaMultiply(bodyValue, sourceAngular));
    }
    return result;
}

int run(const int argc, const char* const* argv) {
    std::size_t problemCount = 256u;
    std::uint32_t replayCount = 3u;
    std::string metallibPath = NUMI_TEMPORAL_CONE_METALLIB;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string_view value(argv[argument]);
        if (value == "--islands" && argument + 1 < argc) {
            problemCount = std::stoull(argv[++argument]);
        } else if (value == "--replays" && argument + 1 < argc) {
            replayCount = static_cast<std::uint32_t>(std::stoul(argv[++argument]));
        } else if (value == "--metallib" && argument + 1 < argc) {
            metallibPath = argv[++argument];
        } else if (value == "--help") {
            std::cout << "usage: numi-solver-rigid [--islands N] [--replays N] "
                         "[--metallib PATH]\n";
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + std::string(value));
        }
    }
    problemCount = std::max<std::size_t>(problemCount, 32u);
    replayCount = std::max<std::uint32_t>(replayCount, 2u);
    if (problemCount > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("island count exceeds rigid ABI");
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (device == nil || queue == nil) {
        throw std::runtime_error("no Apple Metal command queue is available");
    }
    NSError* error = nil;
    NSString* path = [NSString stringWithUTF8String:metallibPath.c_str()];
    id<MTLLibrary> library = [device
        newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
    if (library == nil) {
        throw std::runtime_error("failed to load metallib: " + errorText(error));
    }
    const std::array<NSString*, 5> names = {
        @"numi_temporal_cone_rigid_response",
        @"numi_temporal_cone_stream_assemble",
        @"numi_temporal_cone_stream_solve",
        @"numi_temporal_cone_rigid_publish",
        @"numi_temporal_cone_rigid_integrate"
    };
    std::array<id<MTLComputePipelineState>, 5> pipelines;
    for (std::size_t index = 0u; index < pipelines.size(); ++index) {
        id<MTLFunction> function = [library newFunctionWithName:names[index]];
        pipelines[index] = [device newComputePipelineStateWithFunction:function error:&error];
        if (pipelines[index] == nil || pipelines[index].threadExecutionWidth != 32u) {
            throw std::runtime_error("failed to create SIMD32 rigid pipeline: " + errorText(error));
        }
    }

    const Batch batch = makeBatch(problemCount);
    (void)runGPU(device, queue, pipelines, batch);
    std::vector<Result> replays;
    for (std::uint32_t replay = 0u; replay < replayCount; ++replay) {
        replays.push_back(runGPU(device, queue, pipelines, batch));
    }
    const Result& result = replays.front();
    constexpr std::uint32_t kFreeFlightSteps = 240u;
    (void)runFreeFlight(device, queue, pipelines[4], kFreeFlightSteps);
    const FreeFlightResult freeFlight = runFreeFlight(
        device, queue, pipelines[4], kFreeFlightSteps
    );
    const FreeFlightResult freeFlightReplay = runFreeFlight(
        device, queue, pipelines[4], kFreeFlightSteps
    );
    bool deterministic = true;
    for (std::size_t replay = 1u; replay < replays.size(); ++replay) {
        deterministic = deterministic &&
            exactVector(result.spans, replays[replay].spans) &&
            exactVector(result.terms, replays[replay].terms) &&
            exactVector(result.jacobians, replays[replay].jacobians) &&
            exactVector(result.responses, replays[replay].responses) &&
            exactVector(result.solverContacts, replays[replay].solverContacts) &&
            exactVector(result.responseStatuses, replays[replay].responseStatuses) &&
            exactVector(result.streamHeaders, replays[replay].streamHeaders) &&
            exactVector(result.blocks, replays[replay].blocks) &&
            exactVector(result.assemblyStatuses, replays[replay].assemblyStatuses) &&
            exactVector(result.impulses, replays[replay].impulses) &&
            exactVector(result.solverStatuses, replays[replay].solverStatuses) &&
            exactVector(result.outputBodies, replays[replay].outputBodies) &&
            exactVector(result.publishStatuses, replays[replay].publishStatuses) &&
            exactVector(result.outputPoses, replays[replay].outputPoses) &&
            exactVector(result.integrationStatuses, replays[replay].integrationStatuses);
    }

    double maximumOperatorError = 0.0;
    double maximumFreeVelocityError = 0.0;
    double maximumPublicationError = 0.0;
    double maximumPoseError = 0.0;
    double maximumQuaternionNormError = 0.0;
    double maximumMomentumError = 0.0;
    double maximumEnergyIncrease = 0.0;
    double maximumKKT = 0.0;
    double maximumCone = 0.0;
    std::uint32_t maximumIterations = 0u;
    std::uint32_t maximumAccelerationRestarts = 0u;
    std::size_t acceleratedIslands = 0u;
    std::size_t failedValid = 0u;
    bool failureRollback = true;
    std::vector<std::uint32_t> iterations;
    for (std::size_t problem = 0u; problem < problemCount; ++problem) {
        const auto& rigidHeader = batch.rigidHeaders[problem];
        const auto& assemblyHeader = batch.assemblyHeaders[problem];
        if (batch.expectedFailure[problem]) {
            failureRollback = failureRollback &&
                result.responseStatuses[problem].control.x ==
                    NUMI_TEMPORAL_CONE_RIGID_INVALID_INPUT &&
                result.assemblyStatuses[problem].control.x !=
                    NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS &&
                result.solverStatuses[problem].control.x !=
                    NUMI_TEMPORAL_CONE_ISLAND_SUCCESS &&
                result.publishStatuses[problem].control.x ==
                    NUMI_TEMPORAL_CONE_RIGID_UPSTREAM_FAILURE &&
                result.integrationStatuses[problem].control.x ==
                    NUMI_TEMPORAL_CONE_INTEGRATION_UPSTREAM_FAILURE &&
                std::memcmp(
                    batch.bodies.data() + rigidHeader.inputRanges.x,
                    result.outputBodies.data() + rigidHeader.solverRanges.y,
                    rigidHeader.control.y * sizeof(NumiTemporalConeRigidBody)
                ) == 0 &&
                std::memcmp(
                    batch.poses.data() + rigidHeader.inputRanges.x,
                    result.outputPoses.data() + rigidHeader.solverRanges.y,
                    rigidHeader.control.y * sizeof(NumiTemporalConeRigidPose)
                ) == 0;
            continue;
        }
        const bool success =
            result.responseStatuses[problem].control.x ==
                NUMI_TEMPORAL_CONE_RIGID_SUCCESS &&
            result.assemblyStatuses[problem].control.x ==
                NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS &&
            result.solverStatuses[problem].control.x ==
                NUMI_TEMPORAL_CONE_ISLAND_SUCCESS &&
            result.publishStatuses[problem].control.x ==
                NUMI_TEMPORAL_CONE_RIGID_SUCCESS &&
            result.integrationStatuses[problem].control.x ==
                NUMI_TEMPORAL_CONE_INTEGRATION_SUCCESS;
        if (!success) {
            ++failedValid;
            continue;
        }
        maximumKKT = std::max<double>(maximumKKT, result.solverStatuses[problem].residuals.x);
        maximumCone = std::max<double>(maximumCone, result.solverStatuses[problem].residuals.y);
        maximumIterations = std::max(maximumIterations, result.solverStatuses[problem].control.y);
        const auto restartCount = static_cast<std::uint32_t>(
            result.solverStatuses[problem].diagnostics.w
        );
        maximumAccelerationRestarts = std::max(
            maximumAccelerationRestarts,
            restartCount
        );
        acceleratedIslands += restartCount > 0u ? 1u : 0u;
        iterations.push_back(result.solverStatuses[problem].control.y);
        for (std::uint32_t index = 0u; index < rigidHeader.control.z; ++index) {
            const auto& c = batch.rigidContacts[rigidHeader.inputRanges.y + index];
            std::array<double, 3> pointA{};
            std::array<double, 3> pointB{};
            if (c.bodies.x != kStatic) {
                const auto& value = batch.bodies[rigidHeader.inputRanges.x + c.bodies.x];
                const auto rotational = cross(xyz(value.angularVelocity), xyz(c.offsetA));
                const auto linear = xyz(value.linearVelocityAndInverseMass);
                for (std::size_t component = 0u; component < 3u; ++component) {
                    pointA[component] = linear[component] + rotational[component];
                }
            }
            if (c.bodies.y != kStatic) {
                const auto& value = batch.bodies[rigidHeader.inputRanges.x + c.bodies.y];
                const auto rotational = cross(xyz(value.angularVelocity), xyz(c.offsetB));
                const auto linear = xyz(value.linearVelocityAndInverseMass);
                for (std::size_t component = 0u; component < 3u; ++component) {
                    pointB[component] = linear[component] + rotational[component];
                }
            }
            std::array<double, 3> relative{};
            for (std::size_t component = 0u; component < 3u; ++component) {
                relative[component] = pointB[component] - pointA[component];
            }
            const std::array<double, 3> expectedFree = {
                dot(axis(c, 0u), relative) + c.bias.x,
                dot(axis(c, 1u), relative) + c.bias.y,
                dot(axis(c, 2u), relative) + c.bias.z
            };
            const auto& actual = result.solverContacts[rigidHeader.solverRanges.x + index];
            maximumFreeVelocityError = std::max({
                maximumFreeVelocityError,
                std::abs(expectedFree[0] - actual.freeVelocityAndFrictionU.x),
                std::abs(expectedFree[1] - actual.freeVelocityAndFrictionU.y),
                std::abs(expectedFree[2] - actual.freeVelocityAndFrictionU.z)
            });
        }
        for (std::uint32_t row = 0u; row < rigidHeader.control.z; ++row) {
            const std::uint32_t begin = batch.rowOffsets[assemblyHeader.outputRanges.y + row];
            const std::uint32_t end = batch.rowOffsets[assemblyHeader.outputRanges.y + row + 1u];
            for (std::uint32_t relative = begin; relative < end; ++relative) {
                const std::uint32_t column = batch.columns[assemblyHeader.outputRanges.z + relative];
                for (std::uint32_t a = 0u; a < 3u; ++a) {
                    for (std::uint32_t b = 0u; b < 3u; ++b) {
                        const double expected = expectedCoefficient(
                            batch, problem, row, column, a, b
                        );
                        const double actual = result.blocks[
                            (assemblyHeader.outputRanges.z + relative) * 9u + 3u * a + b
                        ];
                        maximumOperatorError = std::max(
                            maximumOperatorError, std::abs(actual - expected)
                        );
                    }
                }
            }
        }
        bool hasStatic = false;
        for (std::uint32_t index = 0u; index < rigidHeader.control.z; ++index) {
            const auto& c = batch.rigidContacts[rigidHeader.inputRanges.y + index];
            hasStatic = hasStatic || c.bodies.x == kStatic || c.bodies.y == kStatic;
        }
        std::array<double, 3> momentumBefore{};
        std::array<double, 3> momentumAfter{};
        double energyBefore = 0.0;
        double energyAfter = 0.0;
        for (std::uint32_t index = 0u; index < rigidHeader.control.y; ++index) {
            const auto& before = batch.bodies[rigidHeader.inputRanges.x + index];
            const auto& after = result.outputBodies[rigidHeader.solverRanges.y + index];
            auto expectedLinear = xyz(before.linearVelocityAndInverseMass);
            auto expectedAngular = xyz(before.angularVelocity);
            for (std::uint32_t contactIndex = 0u;
                 contactIndex < rigidHeader.control.z;
                 ++contactIndex) {
                const auto& c = batch.rigidContacts[
                    rigidHeader.inputRanges.y + contactIndex
                ];
                double sign = 0.0;
                std::array<double, 3> offset{};
                if (c.bodies.x == index) {
                    sign = -1.0;
                    offset = xyz(c.offsetA);
                } else if (c.bodies.y == index) {
                    sign = 1.0;
                    offset = xyz(c.offsetB);
                }
                if (sign == 0.0) {
                    continue;
                }
                const auto& lambda = result.impulses[
                    rigidHeader.solverRanges.x + contactIndex
                ];
                std::array<double, 3> impulse{};
                for (std::size_t component = 0u; component < 3u; ++component) {
                    impulse[component] = sign * (
                        axis(c, 0u)[component] * lambda.x +
                        axis(c, 1u)[component] * lambda.y +
                        axis(c, 2u)[component] * lambda.z
                    );
                    expectedLinear[component] +=
                        before.linearVelocityAndInverseMass.w * impulse[component];
                }
                const auto angularDelta = inertiaMultiply(
                    before, cross(offset, impulse)
                );
                for (std::size_t component = 0u; component < 3u; ++component) {
                    expectedAngular[component] += angularDelta[component];
                }
            }
            maximumPublicationError = std::max({
                maximumPublicationError,
                std::abs(expectedLinear[0] - after.linearVelocityAndInverseMass.x),
                std::abs(expectedLinear[1] - after.linearVelocityAndInverseMass.y),
                std::abs(expectedLinear[2] - after.linearVelocityAndInverseMass.z),
                std::abs(expectedAngular[0] - after.angularVelocity.x),
                std::abs(expectedAngular[1] - after.angularVelocity.y),
                std::abs(expectedAngular[2] - after.angularVelocity.z)
            });
            const auto& inputPose = batch.poses[rigidHeader.inputRanges.x + index];
            const auto& outputPose = result.outputPoses[rigidHeader.solverRanges.y + index];
            const double timestep = batch.integrationHeaders[problem].timestep.x;
            const std::array<double, 3> angular = xyz(after.angularVelocity);
            const double angle = timestep * std::sqrt(dot(angular, angular));
            const double halfAngle = 0.5 * angle;
            const double rotationScale = angle > 1.0e-6
                ? std::sin(halfAngle) / (angle / timestep)
                : timestep * (0.5 - angle * angle / 48.0);
            const std::array<double, 4> expectedOrientation = {
                angular[0] * rotationScale,
                angular[1] * rotationScale,
                angular[2] * rotationScale,
                std::cos(halfAngle)
            };
            maximumPoseError = std::max({
                maximumPoseError,
                std::abs(
                    inputPose.position.x + timestep * after.linearVelocityAndInverseMass.x -
                    outputPose.position.x
                ),
                std::abs(
                    inputPose.position.y + timestep * after.linearVelocityAndInverseMass.y -
                    outputPose.position.y
                ),
                std::abs(
                    inputPose.position.z + timestep * after.linearVelocityAndInverseMass.z -
                    outputPose.position.z
                ),
                std::abs(expectedOrientation[0] - outputPose.orientation.x),
                std::abs(expectedOrientation[1] - outputPose.orientation.y),
                std::abs(expectedOrientation[2] - outputPose.orientation.z),
                std::abs(expectedOrientation[3] - outputPose.orientation.w)
            });
            maximumQuaternionNormError = std::max(
                maximumQuaternionNormError,
                std::abs(
                    static_cast<double>(outputPose.orientation.x) * outputPose.orientation.x +
                    static_cast<double>(outputPose.orientation.y) * outputPose.orientation.y +
                    static_cast<double>(outputPose.orientation.z) * outputPose.orientation.z +
                    static_cast<double>(outputPose.orientation.w) * outputPose.orientation.w - 1.0
                )
            );
            const double mass = 1.0 / before.linearVelocityAndInverseMass.w;
            momentumBefore[0] += mass * before.linearVelocityAndInverseMass.x;
            momentumBefore[1] += mass * before.linearVelocityAndInverseMass.y;
            momentumBefore[2] += mass * before.linearVelocityAndInverseMass.z;
            momentumAfter[0] += mass * after.linearVelocityAndInverseMass.x;
            momentumAfter[1] += mass * after.linearVelocityAndInverseMass.y;
            momentumAfter[2] += mass * after.linearVelocityAndInverseMass.z;
            energyBefore += kineticEnergy(before);
            energyAfter += kineticEnergy(after);
        }
        if (!hasStatic) {
            maximumMomentumError = std::max(
                maximumMomentumError,
                std::max({
                    std::abs(momentumAfter[0] - momentumBefore[0]),
                    std::abs(momentumAfter[1] - momentumBefore[1]),
                    std::abs(momentumAfter[2] - momentumBefore[2])
                })
            );
        }
        maximumEnergyIncrease = std::max(
            maximumEnergyIncrease, energyAfter - energyBefore
        );
    }
    std::sort(iterations.begin(), iterations.end());
    const auto percentile = [&](const double value) {
        return iterations[static_cast<std::size_t>(
            std::floor(value * static_cast<double>(iterations.size() - 1u))
        )];
    };
    const double analyticImpulseError = std::max(
        std::abs(static_cast<double>(result.impulses[0].x) - 1.0 / 3.0),
        std::abs(static_cast<double>(result.impulses[1].x) - 1.0 / 3.0)
    );
    const double analyticVelocityError = std::abs(
        static_cast<double>(result.outputBodies[0].linearVelocityAndInverseMass.x) -
        1.0 / 3.0
    );
    const double flightTime = static_cast<double>(kFreeFlightSteps) / 240.0;
    const std::array<double, 3> flightAngular = {0.15, -0.2, 0.4};
    const double flightAngularSpeed = std::sqrt(dot(flightAngular, flightAngular));
    const double flightHalfAngle = 0.5 * flightTime * flightAngularSpeed;
    const double flightScale = std::sin(flightHalfAngle) / flightAngularSpeed;
    const std::array<double, 4> expectedFlightOrientation = {
        flightAngular[0] * flightScale,
        flightAngular[1] * flightScale,
        flightAngular[2] * flightScale,
        std::cos(flightHalfAngle)
    };
    const double freeFlightError = std::max({
        std::abs(static_cast<double>(freeFlight.pose.position.x) - 1.2),
        std::abs(static_cast<double>(freeFlight.pose.position.y) + 0.6),
        std::abs(static_cast<double>(freeFlight.pose.position.z) - 0.65),
        std::abs(static_cast<double>(freeFlight.pose.position.w) - 7.0),
        std::abs(freeFlight.pose.orientation.x - expectedFlightOrientation[0]),
        std::abs(freeFlight.pose.orientation.y - expectedFlightOrientation[1]),
        std::abs(freeFlight.pose.orientation.z - expectedFlightOrientation[2]),
        std::abs(freeFlight.pose.orientation.w - expectedFlightOrientation[3])
    });
    const double freeFlightNormError = std::abs(
        static_cast<double>(freeFlight.pose.orientation.x) * freeFlight.pose.orientation.x +
        static_cast<double>(freeFlight.pose.orientation.y) * freeFlight.pose.orientation.y +
        static_cast<double>(freeFlight.pose.orientation.z) * freeFlight.pose.orientation.z +
        static_cast<double>(freeFlight.pose.orientation.w) * freeFlight.pose.orientation.w - 1.0
    );
    const bool freeFlightDeterministic =
        std::memcmp(&freeFlight.pose, &freeFlightReplay.pose, sizeof(freeFlight.pose)) == 0 &&
        std::memcmp(
            &freeFlight.status,
            &freeFlightReplay.status,
            sizeof(freeFlight.status)
        ) == 0;
    const double averageSeconds = std::accumulate(
        replays.begin(), replays.end(), 0.0,
        [](double total, const Result& replay) { return total + replay.seconds; }
    ) / static_cast<double>(replays.size());
    const bool pass = deterministic && failureRollback && failedValid == 0u &&
        maximumOperatorError <= 2.0e-6 && maximumFreeVelocityError <= 2.0e-6 &&
        maximumPublicationError <= 2.0e-6 && maximumMomentumError <= 2.0e-6 &&
        maximumPoseError <= 2.0e-6 && maximumQuaternionNormError <= 2.0e-6 &&
        freeFlight.status.control.x == NUMI_TEMPORAL_CONE_INTEGRATION_SUCCESS &&
        freeFlightError <= 2.0e-5 && freeFlightNormError <= 2.0e-6 &&
        freeFlightDeterministic &&
        maximumEnergyIncrease <= 2.0e-5 && maximumKKT <= 6.0e-6 &&
        maximumCone <= 2.0e-6 && analyticImpulseError <= 3.0e-6 &&
        analyticVelocityError <= 4.0e-6;

    std::cout << std::fixed << std::setprecision(9)
              << "device=" << device.name.UTF8String << '\n'
              << "islands=" << problemCount
              << " valid_bodies=" << batch.validBodyCount
              << " valid_contacts=" << batch.validContactCount
              << " blocks=" << batch.columns.size() << '\n'
              << "operator_max_abs_error=" << maximumOperatorError
              << " free_velocity_max_abs_error=" << maximumFreeVelocityError
              << " publication_max_abs_error=" << maximumPublicationError << '\n'
              << "pose_max_abs_error=" << maximumPoseError
              << " quaternion_norm_max_error=" << maximumQuaternionNormError << '\n'
              << "free_flight_steps=" << kFreeFlightSteps
              << " free_flight_max_abs_error=" << freeFlightError
              << " free_flight_norm_error=" << freeFlightNormError
              << " free_flight_deterministic="
              << (freeFlightDeterministic ? "yes" : "no") << '\n'
              << "analytic_impulse_error=" << analyticImpulseError
              << " analytic_velocity_error=" << analyticVelocityError << '\n'
              << "momentum_max_abs_error=" << maximumMomentumError
              << " energy_max_increase=" << maximumEnergyIncrease << '\n'
              << "kkt_max=" << maximumKKT
              << " cone_max=" << maximumCone
              << " iterations_max=" << maximumIterations
              << " p50=" << percentile(0.50)
              << " p95=" << percentile(0.95)
              << " p99=" << percentile(0.99)
              << " accelerated_islands=" << acceleratedIslands
              << " max_acceleration_restarts="
              << maximumAccelerationRestarts << '\n'
              << "deterministic=" << (deterministic ? "yes" : "no")
              << " invalid_frame_rollback=" << (failureRollback ? "yes" : "no")
              << " failed_valid=" << failedValid << '\n'
              << "one_command_buffer=yes cpu_readback_between_stages=no stages=5\n"
              << "average_chain_seconds=" << averageSeconds
              << " islands_per_second=" << problemCount / averageSeconds
              << " contacts_per_second=" << batch.validContactCount / averageSeconds
              << '\n'
              << (pass ? "PASS" : "FAIL") << '\n';
    return pass ? 0 : 1;
}

}  // namespace

int main(const int argc, const char* const* argv) {
    @autoreleasepool {
        try {
            return run(argc, argv);
        } catch (const std::exception& error) {
            std::cerr << "FAIL: " << error.what() << '\n';
            return 1;
        }
    }
}
