#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/engine_types.h"
#include "numi/temporal_cone_island.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
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

constexpr std::uint32_t kDofs = 2u;
constexpr std::uint32_t kContacts = 2u;
constexpr std::uint32_t kBodies = 3u;
constexpr double kLink1HalfLength = 0.5;
constexpr double kLink2HalfLength = 0.4;
constexpr double kLink1Mass = 2.0;
constexpr double kLink2Mass = 1.5;
constexpr double kLink1InertiaY = 0.15;
constexpr double kLink2InertiaY = 0.10;
constexpr double kArmature0 = 0.02;
constexpr double kArmature1 = 0.03;

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

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

MRBodyPropertiesGPU body(
    std::uint32_t parent,
    std::uint32_t inbound,
    float mass,
    float ix,
    float iy,
    float iz
) {
    MRBodyPropertiesGPU value = {};
    value.articulationIndex = 0u;
    value.parentBody = parent;
    value.inboundJoint = inbound;
    value.motionType = MR_MOTION_DYNAMIC;
    value.massAndInverseMass = f4(mass, 1.0f / mass, 0.0f);
    value.inertiaRow0 = f4(ix, 0.0f, 0.0f);
    value.inertiaRow1 = f4(0.0f, iy, 0.0f);
    value.inertiaRow2 = f4(0.0f, 0.0f, iz);
    value.inverseInertiaRow0 = f4(1.0f / ix, 0.0f, 0.0f);
    value.inverseInertiaRow1 = f4(0.0f, 1.0f / iy, 0.0f);
    value.inverseInertiaRow2 = f4(0.0f, 0.0f, 1.0f / iz);
    value.dampingAndSpeedLimits = f4(0.0f, 0.0f, 1.0e6f, 1.0e6f);
    return value;
}

struct Model {
    MRWorldGPU world = {};
    MRArticulationGPU articulation = {};
    std::array<MRJointDescriptorGPU, 2> joints = {};
    std::array<MRDofPropertiesGPU, 2> dofs = {};
    std::array<MRBodyPropertiesGPU, 3> bodies = {};
};

Model makeModel() {
    Model model;
    model.world.abiVersion = MR_ENGINE_ABI_VERSION;
    model.world.bodyCount = kBodies;
    model.world.articulationCount = 1u;
    model.world.jointCount = 2u;
    model.world.nq = kDofs;
    model.world.nv = kDofs;
    model.world.pairCapacity = kContacts;
    model.world.contactCapacity = kContacts;
    model.world.constraintCapacity = kContacts;
    model.world.islandCapacity = 1u;
    model.world.solverType = MR_SOLVER_TEMPORAL_CONE;
    model.world.frictionConeType = MR_FRICTION_CONE_ELLIPTIC;
    model.world.gravityAndTimestep = f4(0.0f, 0.0f, -9.81f, 1.0f / 240.0f);
    model.world.solverScales = f4(1.0e-6f, 1.0e-7f, 2.0f, 1.0e-4f);

    model.articulation.rootBody = 0u;
    model.articulation.rootType = MR_ROOT_FIXED;
    model.articulation.firstBody = 0u;
    model.articulation.bodyCount = kBodies;
    model.articulation.firstJoint = 0u;
    model.articulation.jointCount = 2u;
    model.articulation.nq = kDofs;
    model.articulation.nv = kDofs;

    model.bodies[0] = body(MR_INVALID_INDEX, MR_INVALID_INDEX, 1.0f,
        0.2f, 0.25f, 0.3f);
    model.bodies[1] = body(0u, 0u, static_cast<float>(kLink1Mass),
        0.11f, static_cast<float>(kLink1InertiaY), 0.13f);
    model.bodies[2] = body(1u, 1u, static_cast<float>(kLink2Mass),
        0.08f, static_cast<float>(kLink2InertiaY), 0.09f);

    for (std::uint32_t jointIndex = 0u; jointIndex < 2u; ++jointIndex) {
        auto& joint = model.joints[jointIndex];
        joint.parentBody = jointIndex;
        joint.childBody = jointIndex + 1u;
        joint.jointType = MR_JOINT_REVOLUTE;
        joint.qOffset = jointIndex;
        joint.nq = 1u;
        joint.vOffset = jointIndex;
        joint.nv = 1u;
        joint.axis0 = f4(0.0f, 1.0f, 0.0f);
        joint.parentRotation = f4(0.0f, 0.0f, 0.0f, 1.0f);
        joint.childRotation = f4(0.0f, 0.0f, 0.0f, 1.0f);
        auto& dof = model.dofs[jointIndex];
        dof.articulationIndex = 0u;
        dof.jointIndex = jointIndex;
        dof.qIndex = jointIndex;
        dof.vIndex = jointIndex;
        dof.localDof = 0u;
    }
    model.joints[0].parentAnchor = f4(0.0f, 0.0f, 0.0f);
    model.joints[0].childAnchor = f4(0.0f, 0.0f, -0.5f);
    model.joints[1].parentAnchor = f4(0.0f, 0.0f, 0.5f);
    model.joints[1].childAnchor = f4(0.0f, 0.0f, -0.4f);
    model.dofs[0].drive = f4(0.0f, 0.0f, static_cast<float>(kArmature0), 0.0f);
    model.dofs[1].drive = f4(0.0f, 0.0f, static_cast<float>(kArmature1), 0.0f);
    return model;
}

NumiTemporalConeRigidLaw law(float gap, float restitution) {
    constexpr float timestep = 1.0f / 240.0f;
    NumiTemporalConeRigidLaw value = {};
    value.stiffnessAndRestitution = f4(18000.0f, 900.0f, 900.0f, restitution);
    value.dampingAndImpactThreshold = f4(260.0f, 40.0f, 40.0f, 0.05f);
    value.stabilization = f4(gap, 0.0005f, 1.5f, timestep);
    return value;
}

NumiTemporalConeArticulatedContact contact(std::uint32_t point) {
    NumiTemporalConeArticulatedContact value = {};
    value.control = u4(point, 0u, 0u, 0u);
    value.normalAndFrictionU = f4(0.0f, 0.0f, 1.0f, 0.65f);
    value.tangentUAndFrictionV = f4(1.0f, 0.0f, 0.0f, 0.45f);
    value.tangentVAndMaximumNormal = f4(0.0f, 1.0f, 0.0f, 0.0f);
    return value;
}

struct Batch {
    Model model;
    MRArticulatedOperatorDispatchGPU operatorDispatch = {};
    std::vector<float> q;
    std::vector<float> velocities;
    std::vector<MRArticulatedPointImpulseGPU> points;
    std::vector<NumiTemporalConeArticulatedHeader> articulatedHeaders;
    std::vector<NumiTemporalConeAssemblyHeader> assemblyHeaders;
    std::vector<NumiTemporalConeArticulatedContact> contacts;
    std::vector<NumiTemporalConeRigidLaw> laws;
    std::vector<std::uint32_t> rowOffsets;
    std::vector<std::uint32_t> columns;
    std::size_t validProblems = 0u;
};

Batch makeBatch(std::size_t problemCount) {
    Batch batch;
    batch.model = makeModel();
    batch.validProblems = problemCount - 3u;
    batch.operatorDispatch.articulationIndex = 0u;
    batch.operatorDispatch.environmentCount = static_cast<mr_u32>(problemCount);
    batch.operatorDispatch.pointCount = kContacts;
    batch.operatorDispatch.flags = MR_ARTICULATED_OPERATOR_WRITE_CHOLESKY_FACTOR;
    batch.operatorDispatch.qStride = kDofs;
    batch.operatorDispatch.pointStride = kContacts;
    batch.operatorDispatch.bodyPoseStride = kBodies;
    batch.operatorDispatch.pointWorldStride = kContacts;
    batch.operatorDispatch.massMatrixStride = kDofs * kDofs;
    batch.operatorDispatch.pointJacobianStride = kContacts * 3u * kDofs;
    batch.operatorDispatch.generalizedStride = kDofs;
    for (std::size_t problem = 0u; problem < problemCount; ++problem) {
        const float q0 = -1.0f + 2.0f *
            static_cast<float>(problem % 97u) / 96.0f;
        const float q1 = -1.2f + 2.4f *
            static_cast<float>(problem % 89u) / 88.0f;
        batch.q.push_back(q0);
        batch.q.push_back(q1);
        if (problem + 3u == problemCount) {
            batch.q[2u * problem] = std::numeric_limits<float>::quiet_NaN();
        }
        batch.velocities.push_back(0.80f + 0.01f * static_cast<float>(problem % 5u));
        batch.velocities.push_back(0.40f - 0.01f * static_cast<float>(problem % 3u));
        MRArticulatedPointImpulseGPU first = {};
        first.bodyIndex = 1u;
        first.localPoint = f4(0.0f, 0.0f, 0.5f);
        MRArticulatedPointImpulseGPU second = {};
        second.bodyIndex = 2u;
        second.localPoint = f4(0.0f, 0.0f, 0.4f);
        batch.points.push_back(first);
        batch.points.push_back(second);
        batch.contacts.push_back(contact(0u));
        batch.contacts.push_back(contact(1u));
        batch.laws.push_back(law(-0.0020f, 0.25f));
        batch.laws.push_back(law(-0.0015f, 0.20f));
        if (problem + 2u == problemCount) {
            batch.contacts[2u * problem + 0u].tangentVAndMaximumNormal =
                f4(1.0f, 0.0f, 0.0f);
        } else if (problem + 1u == problemCount) {
            batch.laws[2u * problem + 0u].stiffnessAndRestitution.w = 1.5f;
        }

        NumiTemporalConeArticulatedHeader articulated = {};
        articulated.control = u4(
            NUMI_TEMPORAL_CONE_ARTICULATED_ABI_VERSION,
            kDofs,
            kContacts,
            static_cast<std::uint32_t>(1000u + problem)
        );
        articulated.inputRanges = u4(
            static_cast<std::uint32_t>(problem * kDofs * kDofs),
            static_cast<std::uint32_t>(problem * kContacts * 3u * kDofs),
            static_cast<std::uint32_t>(problem * kDofs),
            static_cast<std::uint32_t>(problem * kContacts)
        );
        articulated.responseRanges = u4(
            static_cast<std::uint32_t>(problem * kContacts),
            static_cast<std::uint32_t>(problem * kContacts),
            static_cast<std::uint32_t>(problem * kContacts * 3u * kDofs),
            static_cast<std::uint32_t>(problem * kContacts * 3u * kDofs)
        );
        articulated.solverRanges = u4(
            static_cast<std::uint32_t>(problem * kContacts),
            static_cast<std::uint32_t>(problem * kDofs),
            static_cast<std::uint32_t>(problem * kContacts),
            static_cast<std::uint32_t>(problem * kContacts * 9u)
        );
        articulated.operatorRanges = u4(static_cast<std::uint32_t>(problem), 0u, 0u, 0u);
        batch.articulatedHeaders.push_back(articulated);

        const std::uint32_t rowBase = static_cast<std::uint32_t>(batch.rowOffsets.size());
        const std::uint32_t blockBase = static_cast<std::uint32_t>(batch.columns.size());
        batch.rowOffsets.push_back(0u);
        batch.rowOffsets.push_back(2u);
        batch.rowOffsets.push_back(4u);
        batch.columns.insert(batch.columns.end(), {0u, 1u, 0u, 1u});
        NumiTemporalConeAssemblyHeader assembly = {};
        assembly.control = u4(
            NUMI_TEMPORAL_CONE_ASSEMBLY_ABI_VERSION,
            kContacts,
            4u,
            512u
        );
        assembly.outputRanges = u4(
            static_cast<std::uint32_t>(problem * kContacts),
            rowBase,
            blockBase,
            4u
        );
        assembly.inputRanges = u4(
            static_cast<std::uint32_t>(problem * kContacts),
            static_cast<std::uint32_t>(problem * kContacts * 9u),
            0u,
            0u
        );
        assembly.tolerances = f4(2.0e-6f, 2.0e-6f, 1.0f, 0.0f);
        batch.assemblyHeaders.push_back(assembly);
    }
    return batch;
}

struct Result {
    std::vector<MRArticulatedBodyPoseGPU> poses;
    std::vector<MRArticulatedPointWorldGPU> pointWorld;
    std::vector<float> factors;
    std::vector<float> pointJacobians;
    std::vector<MRArticulatedOperatorStatusGPU> operatorStatuses;
    std::vector<NumiTemporalConeAssemblyContactSpan> spans;
    std::vector<NumiTemporalConeAssemblyTerm> terms;
    std::vector<float> jacobians;
    std::vector<float> responses;
    std::vector<NumiTemporalConeIslandContact> solverContacts;
    std::vector<float> regularization;
    std::vector<NumiTemporalConeArticulatedStatus> responseStatuses;
    std::vector<NumiTemporalConeStreamHeader> streamHeaders;
    std::vector<float> blocks;
    std::vector<NumiTemporalConeAssemblyStatus> assemblyStatuses;
    std::vector<mr_float4> impulses;
    std::vector<NumiTemporalConeIslandStatus> solverStatuses;
    std::vector<float> outputVelocities;
    std::vector<NumiTemporalConeArticulatedStatus> publishStatuses;
    double seconds = 0.0;
};

std::size_t operatorThreadgroupBytes(std::size_t bodies, std::size_t dofs) {
    const auto aligned = [](std::size_t value) { return (value + 15u) & ~std::size_t{15u}; };
    std::size_t bytes = 0u;
    const auto append = [&](std::size_t value) { bytes = aligned(bytes); bytes += value; };
    append(16u * bodies);
    append(16u * bodies);
    append(16u * bodies);
    append(16u * bodies);
    append(sizeof(std::uint32_t) * bodies);
    append(sizeof(std::uint32_t) * bodies);
    append(sizeof(std::uint8_t) * bodies);
    append(sizeof(float) * dofs * dofs);
    append(sizeof(float) * dofs);
    append(sizeof(float) * dofs);
    append(sizeof(float) * dofs);
    return aligned(bytes);
}

template <typename T>
bool exactVector(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
        std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

Result runGPU(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    const std::array<id<MTLComputePipelineState>, 5>& pipelines,
    const Batch& batch
) {
    const std::uint32_t problemCount =
        static_cast<std::uint32_t>(batch.articulatedHeaders.size());
    const auto makeBytes = [&](const auto& values) -> id<MTLBuffer> {
        return [device newBufferWithBytes:values.data()
                                  length:values.size() * sizeof(values.front())
                                 options:MTLResourceStorageModeShared];
    };
    const auto makeOne = [&](const auto& value) -> id<MTLBuffer> {
        return [device newBufferWithBytes:&value
                                  length:sizeof(value)
                                 options:MTLResourceStorageModeShared];
    };
    const auto output = [&](std::size_t bytes) -> id<MTLBuffer> {
        id<MTLBuffer> buffer = [device newBufferWithLength:bytes
                                                  options:MTLResourceStorageModeShared];
        if (buffer != nil) std::memset(buffer.contents, 0, buffer.length);
        return buffer;
    };
    const auto& model = batch.model;
    id<MTLBuffer> worldBuffer = makeOne(model.world);
    id<MTLBuffer> articulationBuffer = makeOne(model.articulation);
    id<MTLBuffer> jointBuffer = makeBytes(model.joints);
    id<MTLBuffer> dofBuffer = makeBytes(model.dofs);
    id<MTLBuffer> bodyBuffer = makeBytes(model.bodies);
    id<MTLBuffer> operatorDispatchBuffer = makeOne(batch.operatorDispatch);
    id<MTLBuffer> qBuffer = makeBytes(batch.q);
    id<MTLBuffer> pointBuffer = makeBytes(batch.points);
    id<MTLBuffer> poseBuffer = output(problemCount * kBodies * sizeof(MRArticulatedBodyPoseGPU));
    id<MTLBuffer> pointWorldBuffer = output(problemCount * kContacts * sizeof(MRArticulatedPointWorldGPU));
    id<MTLBuffer> factorBuffer = output(problemCount * kDofs * kDofs * sizeof(float));
    id<MTLBuffer> pointJacobianBuffer = output(problemCount * kContacts * 3u * kDofs * sizeof(float));
    id<MTLBuffer> generalizedBuffer = output(problemCount * kDofs * sizeof(float));
    id<MTLBuffer> operatorDeltaBuffer = output(problemCount * kDofs * sizeof(float));
    id<MTLBuffer> operatorStatusBuffer = output(problemCount * sizeof(MRArticulatedOperatorStatusGPU));
    id<MTLBuffer> articulatedHeaderBuffer = makeBytes(batch.articulatedHeaders);
    id<MTLBuffer> assemblyHeaderBuffer = makeBytes(batch.assemblyHeaders);
    id<MTLBuffer> velocityBuffer = makeBytes(batch.velocities);
    id<MTLBuffer> contactBuffer = makeBytes(batch.contacts);
    id<MTLBuffer> lawBuffer = makeBytes(batch.laws);
    id<MTLBuffer> spanBuffer = output(problemCount * kContacts * sizeof(NumiTemporalConeAssemblyContactSpan));
    id<MTLBuffer> termBuffer = output(problemCount * kContacts * sizeof(NumiTemporalConeAssemblyTerm));
    id<MTLBuffer> jacobianBuffer = output(problemCount * kContacts * 3u * kDofs * sizeof(float));
    id<MTLBuffer> responseBuffer = output(problemCount * kContacts * 3u * kDofs * sizeof(float));
    id<MTLBuffer> solverContactBuffer = output(problemCount * kContacts * sizeof(NumiTemporalConeIslandContact));
    id<MTLBuffer> regularizationBuffer = output(problemCount * kContacts * 9u * sizeof(float));
    id<MTLBuffer> responseStatusBuffer = output(problemCount * sizeof(NumiTemporalConeArticulatedStatus));
    id<MTLBuffer> rowBuffer = makeBytes(batch.rowOffsets);
    id<MTLBuffer> columnBuffer = makeBytes(batch.columns);
    id<MTLBuffer> blockBuffer = output(batch.columns.size() * 9u * sizeof(float));
    id<MTLBuffer> streamHeaderBuffer = output(problemCount * sizeof(NumiTemporalConeStreamHeader));
    id<MTLBuffer> assemblyStatusBuffer = output(problemCount * sizeof(NumiTemporalConeAssemblyStatus));
    id<MTLBuffer> impulseBuffer = output(problemCount * kContacts * sizeof(mr_float4));
    id<MTLBuffer> solverStatusBuffer = output(problemCount * sizeof(NumiTemporalConeIslandStatus));
    id<MTLBuffer> outputVelocityBuffer = output(problemCount * kDofs * sizeof(float));
    id<MTLBuffer> publishStatusBuffer = output(problemCount * sizeof(NumiTemporalConeArticulatedStatus));
    require(worldBuffer && articulationBuffer && jointBuffer && dofBuffer &&
        bodyBuffer && operatorDispatchBuffer && qBuffer && pointBuffer &&
        poseBuffer && pointWorldBuffer && factorBuffer && pointJacobianBuffer &&
        generalizedBuffer && operatorDeltaBuffer && operatorStatusBuffer &&
        articulatedHeaderBuffer && assemblyHeaderBuffer && velocityBuffer &&
        contactBuffer && lawBuffer && spanBuffer && termBuffer && jacobianBuffer &&
        responseBuffer && solverContactBuffer && regularizationBuffer &&
        responseStatusBuffer && rowBuffer && columnBuffer && blockBuffer &&
        streamHeaderBuffer && assemblyStatusBuffer && impulseBuffer &&
        solverStatusBuffer && outputVelocityBuffer && publishStatusBuffer,
        "failed to allocate articulated qualification buffers");

    const auto start = std::chrono::steady_clock::now();
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> op = [command computeCommandEncoder];
    [op setComputePipelineState:pipelines[0]];
    const std::array<id<MTLBuffer>, 15> operatorBuffers = {
        worldBuffer, articulationBuffer, jointBuffer, dofBuffer, bodyBuffer,
        operatorDispatchBuffer, qBuffer, pointBuffer, poseBuffer,
        pointWorldBuffer, factorBuffer, pointJacobianBuffer,
        generalizedBuffer, operatorDeltaBuffer, operatorStatusBuffer
    };
    for (std::size_t index = 0u; index < operatorBuffers.size(); ++index) {
        [op setBuffer:operatorBuffers[index] offset:0 atIndex:index];
    }
    [op setThreadgroupMemoryLength:operatorThreadgroupBytes(kBodies, kDofs) atIndex:0];
    [op dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
          threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [op endEncoding];

    id<MTLComputeCommandEncoder> response = [command computeCommandEncoder];
    [response setComputePipelineState:pipelines[1]];
    const std::array<id<MTLBuffer>, 14> responseBuffers = {
        articulatedHeaderBuffer, operatorStatusBuffer, factorBuffer,
        pointJacobianBuffer, velocityBuffer, contactBuffer, lawBuffer,
        spanBuffer, termBuffer, jacobianBuffer, responseBuffer,
        solverContactBuffer, regularizationBuffer, responseStatusBuffer
    };
    for (std::size_t index = 0u; index < responseBuffers.size(); ++index) {
        [response setBuffer:responseBuffers[index] offset:0 atIndex:index];
    }
    [response setBytes:&problemCount length:sizeof(problemCount) atIndex:14];
    const mr_uint4 inputCaps = u4(
        problemCount * kDofs * kDofs,
        problemCount * kContacts * 3u * kDofs,
        problemCount * kDofs,
        problemCount * kContacts
    );
    const mr_uint4 responseCaps = u4(
        problemCount * kContacts,
        problemCount * kContacts,
        problemCount * kContacts * 3u * kDofs,
        problemCount * kContacts * 3u * kDofs
    );
    const mr_uint4 responseSolverCaps = u4(
        problemCount * kContacts,
        problemCount * kContacts,
        problemCount * kContacts * 9u,
        problemCount
    );
    [response setBytes:&inputCaps length:sizeof(inputCaps) atIndex:15];
    [response setBytes:&responseCaps length:sizeof(responseCaps) atIndex:16];
    [response setBytes:&responseSolverCaps length:sizeof(responseSolverCaps) atIndex:17];
    [response dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [response endEncoding];

    id<MTLComputeCommandEncoder> assembly = [command computeCommandEncoder];
    [assembly setComputePipelineState:pipelines[2]];
    const std::array<id<MTLBuffer>, 11> assemblyBuffers = {
        assemblyHeaderBuffer, spanBuffer, termBuffer, jacobianBuffer,
        responseBuffer, regularizationBuffer, rowBuffer, columnBuffer,
        blockBuffer, streamHeaderBuffer, assemblyStatusBuffer
    };
    for (std::size_t index = 0u; index < assemblyBuffers.size(); ++index) {
        [assembly setBuffer:assemblyBuffers[index] offset:0 atIndex:index];
    }
    [assembly setBytes:&problemCount length:sizeof(problemCount) atIndex:11];
    const mr_uint4 assemblyInputs = responseCaps;
    const mr_uint4 assemblyOutputs = u4(
        static_cast<std::uint32_t>(batch.rowOffsets.size()),
        static_cast<std::uint32_t>(batch.columns.size()),
        problemCount * kContacts * 9u,
        static_cast<std::uint32_t>(batch.columns.size() * 9u)
    );
    [assembly setBytes:&assemblyInputs length:sizeof(assemblyInputs) atIndex:12];
    [assembly setBytes:&assemblyOutputs length:sizeof(assemblyOutputs) atIndex:13];
    [assembly dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [assembly endEncoding];

    id<MTLComputeCommandEncoder> solver = [command computeCommandEncoder];
    [solver setComputePipelineState:pipelines[3]];
    const std::array<id<MTLBuffer>, 7> solverBuffers = {
        streamHeaderBuffer, rowBuffer, columnBuffer, blockBuffer,
        solverContactBuffer, impulseBuffer, solverStatusBuffer
    };
    for (std::size_t index = 0u; index < solverBuffers.size(); ++index) {
        [solver setBuffer:solverBuffers[index] offset:0 atIndex:index];
    }
    [solver setBytes:&problemCount length:sizeof(problemCount) atIndex:7];
    const mr_uint4 solverCaps = u4(
        problemCount * kContacts,
        static_cast<std::uint32_t>(batch.rowOffsets.size()),
        static_cast<std::uint32_t>(batch.columns.size()),
        problemCount * kContacts
    );
    [solver setBytes:&solverCaps length:sizeof(solverCaps) atIndex:8];
    [solver dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
              threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [solver endEncoding];

    id<MTLComputeCommandEncoder> publish = [command computeCommandEncoder];
    [publish setComputePipelineState:pipelines[4]];
    const std::array<id<MTLBuffer>, 8> publishBuffers = {
        articulatedHeaderBuffer, velocityBuffer, responseBuffer, impulseBuffer,
        responseStatusBuffer, solverStatusBuffer, outputVelocityBuffer,
        publishStatusBuffer
    };
    for (std::size_t index = 0u; index < publishBuffers.size(); ++index) {
        [publish setBuffer:publishBuffers[index] offset:0 atIndex:index];
    }
    [publish setBytes:&problemCount length:sizeof(problemCount) atIndex:8];
    const mr_uint4 publishCaps = u4(
        problemCount * kDofs,
        problemCount * kContacts * 3u * kDofs,
        problemCount * kContacts,
        problemCount * kDofs
    );
    [publish setBytes:&publishCaps length:sizeof(publishCaps) atIndex:9];
    [publish dispatchThreadgroups:MTLSizeMake(problemCount, 1u, 1u)
               threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
    [publish endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted && command.error == nil,
        "Metal articulated chain failed: " + errorText(command.error));
    const auto end = std::chrono::steady_clock::now();
    const double gpuSeconds = command.GPUEndTime > command.GPUStartTime
        ? command.GPUEndTime - command.GPUStartTime
        : 0.0;

    Result result;
    const auto copy = [](auto& destination, id<MTLBuffer> source) {
        using Value = typename std::decay_t<decltype(destination)>::value_type;
        const auto* begin = static_cast<const Value*>(source.contents);
        destination.assign(begin, begin + source.length / sizeof(Value));
    };
    copy(result.poses, poseBuffer);
    copy(result.pointWorld, pointWorldBuffer);
    copy(result.factors, factorBuffer);
    copy(result.pointJacobians, pointJacobianBuffer);
    copy(result.operatorStatuses, operatorStatusBuffer);
    copy(result.spans, spanBuffer);
    copy(result.terms, termBuffer);
    copy(result.jacobians, jacobianBuffer);
    copy(result.responses, responseBuffer);
    copy(result.solverContacts, solverContactBuffer);
    copy(result.regularization, regularizationBuffer);
    copy(result.responseStatuses, responseStatusBuffer);
    copy(result.streamHeaders, streamHeaderBuffer);
    copy(result.blocks, blockBuffer);
    copy(result.assemblyStatuses, assemblyStatusBuffer);
    copy(result.impulses, impulseBuffer);
    copy(result.solverStatuses, solverStatusBuffer);
    copy(result.outputVelocities, outputVelocityBuffer);
    copy(result.publishStatuses, publishStatusBuffer);
    result.seconds = gpuSeconds > 0.0
        ? gpuSeconds
        : std::chrono::duration<double>(end - start).count();
    return result;
}

using Vec2 = std::array<double, 2>;
using Mat2 = std::array<double, 4>;

Vec2 point(double angle, double length) {
    return {length * std::sin(angle), length * std::cos(angle)};
}

Vec2 add(Vec2 a, Vec2 b) { return {a[0] + b[0], a[1] + b[1]}; }
Vec2 crossY(Vec2 radius) { return {radius[1], -radius[0]}; }
double dot(Vec2 a, Vec2 b) { return a[0] * b[0] + a[1] * b[1]; }

struct Oracle {
    Mat2 mass = {};
    std::array<Vec2, 2> pointPositions = {};
    std::array<std::array<Vec2, 2>, 2> pointJacobians = {};
};

Oracle oracle(double q0, double q1) {
    Oracle value;
    const Vec2 link1Com = point(q0, kLink1HalfLength);
    const Vec2 joint2 = point(q0, 2.0 * kLink1HalfLength);
    const Vec2 link2Com = add(joint2, point(q0 + q1, kLink2HalfLength));
    const Vec2 tip = add(joint2, point(q0 + q1, 2.0 * kLink2HalfLength));
    value.pointPositions = {joint2, tip};
    value.pointJacobians[0][0] = crossY(joint2);
    value.pointJacobians[0][1] = {0.0, 0.0};
    value.pointJacobians[1][0] = crossY(tip);
    value.pointJacobians[1][1] = crossY({
        tip[0] - joint2[0], tip[1] - joint2[1]
    });
    const Vec2 j1 = crossY(link1Com);
    const Vec2 j20 = crossY(link2Com);
    const Vec2 j21 = crossY({
        link2Com[0] - joint2[0], link2Com[1] - joint2[1]
    });
    value.mass[0] = kLink1Mass * dot(j1, j1) + kLink1InertiaY +
        kLink2Mass * dot(j20, j20) + kLink2InertiaY + kArmature0;
    value.mass[1] = kLink2Mass * dot(j20, j21) + kLink2InertiaY;
    value.mass[2] = value.mass[1];
    value.mass[3] = kLink2Mass * dot(j21, j21) + kLink2InertiaY + kArmature1;
    return value;
}

Vec2 solve(Mat2 m, Vec2 rhs) {
    const double determinant = m[0] * m[3] - m[1] * m[2];
    require(determinant > 1.0e-12, "FP64 articulated mass matrix is not SPD");
    return {
        (m[3] * rhs[0] - m[1] * rhs[1]) / determinant,
        (-m[2] * rhs[0] + m[0] * rhs[1]) / determinant
    };
}

double maxAbs(double a, double b) { return std::max(std::abs(a), std::abs(b)); }

int run(int argc, const char* const* argv) {
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
            std::cout << "usage: numi-solver-articulated [--islands N] [--replays N] [--metallib PATH]\n";
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + std::string(value));
        }
    }
    problemCount = std::max<std::size_t>(problemCount, 8u);
    replayCount = std::max<std::uint32_t>(replayCount, 2u);
    require(problemCount <= std::numeric_limits<std::uint32_t>::max() / 100u,
        "island count exceeds articulated ABI");
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    require(device != nil && queue != nil, "no Apple Metal command queue is available");
    NSError* error = nil;
    NSString* path = [NSString stringWithUTF8String:metallibPath.c_str()];
    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
    require(library != nil, "failed to load metallib: " + errorText(error));
    const std::array<NSString*, 5> names = {
        @"mr_articulated_operator",
        @"numi_temporal_cone_articulated_response",
        @"numi_temporal_cone_stream_assemble",
        @"numi_temporal_cone_stream_solve",
        @"numi_temporal_cone_articulated_publish"
    };
    std::array<id<MTLComputePipelineState>, 5> pipelines = {};
    for (std::size_t index = 0u; index < pipelines.size(); ++index) {
        id<MTLFunction> function = [library newFunctionWithName:names[index]];
        pipelines[index] = [device newComputePipelineStateWithFunction:function error:&error];
        require(pipelines[index] != nil && pipelines[index].threadExecutionWidth == 32u,
            "failed to create SIMD32 articulated pipeline: " + errorText(error));
    }
    const Batch batch = makeBatch(problemCount);
    (void)runGPU(device, queue, pipelines, batch);
    std::vector<Result> replays;
    for (std::uint32_t replay = 0u; replay < replayCount; ++replay) {
        replays.push_back(runGPU(device, queue, pipelines, batch));
    }
    const Result& result = replays.front();
    bool deterministic = true;
    for (std::size_t replay = 1u; replay < replays.size(); ++replay) {
        deterministic = deterministic &&
            exactVector(result.poses, replays[replay].poses) &&
            exactVector(result.pointWorld, replays[replay].pointWorld) &&
            exactVector(result.factors, replays[replay].factors) &&
            exactVector(result.pointJacobians, replays[replay].pointJacobians) &&
            exactVector(result.operatorStatuses, replays[replay].operatorStatuses) &&
            exactVector(result.spans, replays[replay].spans) &&
            exactVector(result.terms, replays[replay].terms) &&
            exactVector(result.jacobians, replays[replay].jacobians) &&
            exactVector(result.responses, replays[replay].responses) &&
            exactVector(result.solverContacts, replays[replay].solverContacts) &&
            exactVector(result.regularization, replays[replay].regularization) &&
            exactVector(result.responseStatuses, replays[replay].responseStatuses) &&
            exactVector(result.streamHeaders, replays[replay].streamHeaders) &&
            exactVector(result.blocks, replays[replay].blocks) &&
            exactVector(result.assemblyStatuses, replays[replay].assemblyStatuses) &&
            exactVector(result.impulses, replays[replay].impulses) &&
            exactVector(result.solverStatuses, replays[replay].solverStatuses) &&
            exactVector(result.outputVelocities, replays[replay].outputVelocities) &&
            exactVector(result.publishStatuses, replays[replay].publishStatuses);
    }

    double maxMassError = 0.0;
    double maxJacobianError = 0.0;
    double maxFiniteDifferenceError = 0.0;
    double maxResponseError = 0.0;
    double maxBlockError = 0.0;
    double maxFreeVelocityError = 0.0;
    double maxPublicationError = 0.0;
    double maxEnergyBudgetViolation = 0.0;
    double maxFp64Residual = 0.0;
    double maxGpuBackwardError = 0.0;
    double maxConditionProxy = 0.0;
    std::uint32_t maxIterations = 0u;
    std::size_t failedValid = 0u;
    for (std::size_t problem = 0u; problem < batch.validProblems; ++problem) {
        const Oracle ref = oracle(batch.q[2u * problem], batch.q[2u * problem + 1u]);
        const std::size_t factorBase = 4u * problem;
        const double l00 = result.factors[factorBase + 0u];
        const double l10 = result.factors[factorBase + 2u];
        const double l11 = result.factors[factorBase + 3u];
        const Mat2 gpuMass = {l00 * l00, l00 * l10, l00 * l10, l10 * l10 + l11 * l11};
        for (std::size_t i = 0u; i < 4u; ++i) {
            maxMassError = std::max(maxMassError, std::abs(gpuMass[i] - ref.mass[i]));
        }
        const double q0 = batch.q[2u * problem];
        const double q1 = batch.q[2u * problem + 1u];
        constexpr double h = 1.0e-5;
        for (std::size_t pointIndex = 0u; pointIndex < 2u; ++pointIndex) {
            for (std::size_t dof = 0u; dof < 2u; ++dof) {
                const std::size_t worldBase = problem * 12u + pointIndex * 6u;
                const double gpuX = result.pointJacobians[worldBase + dof];
                const double gpuZ = result.pointJacobians[worldBase + 4u + dof];
                maxJacobianError = std::max(maxJacobianError,
                    maxAbs(gpuX - ref.pointJacobians[pointIndex][dof][0],
                           gpuZ - ref.pointJacobians[pointIndex][dof][1]));
                const Oracle plus = oracle(q0 + (dof == 0u ? h : 0.0),
                    q1 + (dof == 1u ? h : 0.0));
                const Oracle minus = oracle(q0 - (dof == 0u ? h : 0.0),
                    q1 - (dof == 1u ? h : 0.0));
                const Vec2 finiteDifference = {
                    (plus.pointPositions[pointIndex][0] - minus.pointPositions[pointIndex][0]) / (2.0 * h),
                    (plus.pointPositions[pointIndex][1] - minus.pointPositions[pointIndex][1]) / (2.0 * h)
                };
                maxFiniteDifferenceError = std::max(maxFiniteDifferenceError,
                    maxAbs(gpuX - finiteDifference[0], gpuZ - finiteDifference[1]));
            }
        }
        for (std::size_t contactIndex = 0u; contactIndex < 2u; ++contactIndex) {
            const auto& jWorld = ref.pointJacobians[contactIndex];
            const std::array<Vec2, 3> rhs = {{
                {jWorld[0][1], jWorld[1][1]},
                {jWorld[0][0], jWorld[1][0]},
                {0.0, 0.0}
            }};
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                const Vec2 x = solve(ref.mass, rhs[axis]);
                const std::size_t responseBase = problem * 12u + contactIndex * 6u;
                maxResponseError = std::max(maxResponseError,
                    maxAbs(result.responses[responseBase + axis] - x[0],
                           result.responses[responseBase + 3u + axis] - x[1]));
                const Vec2 action = {
                    ref.mass[0] * x[0] + ref.mass[1] * x[1],
                    ref.mass[2] * x[0] + ref.mass[3] * x[1]
                };
                maxFp64Residual = std::max(maxFp64Residual,
                    maxAbs(action[0] - rhs[axis][0], action[1] - rhs[axis][1]));
            }
            const auto& solverContact = result.solverContacts[2u * problem + contactIndex];
            const Vec2 velocity = {batch.velocities[2u * problem], batch.velocities[2u * problem + 1u]};
            const double rawNormal = jWorld[0][1] * velocity[0] + jWorld[1][1] * velocity[1];
            const double rawTangent = jWorld[0][0] * velocity[0] + jWorld[1][0] * velocity[1];
            const auto& lawValue = batch.laws[2u * problem + contactIndex];
            const double dt = lawValue.stabilization.w;
            const double denominator = lawValue.dampingAndImpactThreshold.x +
                dt * lawValue.stiffnessAndRestitution.x;
            const double penetration = std::min<double>(
                lawValue.stabilization.x + lawValue.stabilization.y, 0.0);
            const double recovery = std::min<double>(lawValue.stabilization.z,
                -lawValue.stiffnessAndRestitution.x * penetration / denominator);
            const double rebound = rawNormal < -lawValue.dampingAndImpactThreshold.w
                ? -lawValue.stiffnessAndRestitution.w * rawNormal : 0.0;
            const double target = std::max(recovery, rebound);
            maxFreeVelocityError = std::max(maxFreeVelocityError,
                maxAbs(solverContact.freeVelocityAndFrictionU.x - (rawNormal - target),
                       solverContact.freeVelocityAndFrictionU.y - rawTangent));
        }
        for (std::size_t target = 0u; target < 2u; ++target) {
            for (std::size_t source = 0u; source < 2u; ++source) {
                for (std::size_t row = 0u; row < 3u; ++row) {
                    for (std::size_t column = 0u; column < 3u; ++column) {
                        double expected = 0.0;
                        for (std::size_t dof = 0u; dof < 2u; ++dof) {
                            expected += result.jacobians[problem * 12u + target * 6u + row * 2u + dof] *
                                result.responses[problem * 12u + source * 6u + dof * 3u + column];
                        }
                        if (target == source && row == column) {
                            expected += result.regularization[problem * 18u + target * 9u + 4u * row];
                        }
                        const double actual = result.blocks[problem * 36u + (target * 2u + source) * 9u + 3u * row + column];
                        maxBlockError = std::max(maxBlockError, std::abs(actual - expected));
                    }
                }
            }
        }
        Vec2 expectedVelocity = {batch.velocities[2u * problem], batch.velocities[2u * problem + 1u]};
        for (std::size_t contactIndex = 0u; contactIndex < 2u; ++contactIndex) {
            const auto impulse = result.impulses[2u * problem + contactIndex];
            const std::size_t base = problem * 12u + contactIndex * 6u;
            expectedVelocity[0] += result.responses[base + 0u] * impulse.x +
                result.responses[base + 1u] * impulse.y + result.responses[base + 2u] * impulse.z;
            expectedVelocity[1] += result.responses[base + 3u] * impulse.x +
                result.responses[base + 4u] * impulse.y + result.responses[base + 5u] * impulse.z;
        }
        maxPublicationError = std::max(maxPublicationError,
            maxAbs(result.outputVelocities[2u * problem] - expectedVelocity[0],
                   result.outputVelocities[2u * problem + 1u] - expectedVelocity[1]));
        const Vec2 initialVelocity = {batch.velocities[2u * problem], batch.velocities[2u * problem + 1u]};
        const Vec2 finalVelocity = {result.outputVelocities[2u * problem], result.outputVelocities[2u * problem + 1u]};
        const auto kinetic = [&](Vec2 v) {
            return 0.5 * (v[0] * (ref.mass[0] * v[0] + ref.mass[1] * v[1]) +
                v[1] * (ref.mass[2] * v[0] + ref.mass[3] * v[1]));
        };
        const double deltaEnergy = kinetic(finalVelocity) - kinetic(initialVelocity);
        double budget = 0.0;
        for (std::size_t c = 0u; c < 2u; ++c) {
            const auto impulse = result.impulses[2u * problem + c];
            const auto solverContact = result.solverContacts[2u * problem + c];
            const double targetNormal = -solverContact.freeVelocityAndFrictionU.x +
                (result.jacobians[problem * 12u + c * 6u + 0u] * initialVelocity[0] +
                 result.jacobians[problem * 12u + c * 6u + 1u] * initialVelocity[1]);
            budget += impulse.x * targetNormal;
            budget -= 0.5 * (
                result.regularization[problem * 18u + c * 9u + 0u] * impulse.x * impulse.x +
                result.regularization[problem * 18u + c * 9u + 4u] * impulse.y * impulse.y +
                result.regularization[problem * 18u + c * 9u + 8u] * impulse.z * impulse.z
            );
        }
        maxEnergyBudgetViolation = std::max(maxEnergyBudgetViolation, deltaEnergy - budget);
        maxGpuBackwardError = std::max<double>(maxGpuBackwardError,
            result.responseStatuses[problem].diagnostics.y);
        maxConditionProxy = std::max<double>(maxConditionProxy,
            result.responseStatuses[problem].conditioning.z);
        maxIterations = std::max(maxIterations, result.solverStatuses[problem].control.y);
        const bool valid = result.operatorStatuses[problem].code == MR_ARTICULATED_OPERATOR_SUCCESS &&
            result.responseStatuses[problem].control.x == NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS &&
            result.assemblyStatuses[problem].control.x == NUMI_TEMPORAL_CONE_ASSEMBLY_SUCCESS &&
            result.solverStatuses[problem].control.x == NUMI_TEMPORAL_CONE_ISLAND_SUCCESS &&
            result.publishStatuses[problem].control.x == NUMI_TEMPORAL_CONE_ARTICULATED_SUCCESS;
        if (!valid) ++failedValid;
    }
    bool rollback = true;
    for (std::size_t problem = batch.validProblems; problem < problemCount; ++problem) {
        const bool operatorFailure = problem == batch.validProblems;
        const std::uint32_t expectedResponse = operatorFailure
            ? NUMI_TEMPORAL_CONE_ARTICULATED_UPSTREAM_FAILURE
            : NUMI_TEMPORAL_CONE_ARTICULATED_INVALID_INPUT;
        bool operatorPayloadUntouched = true;
        if (operatorFailure) {
            operatorPayloadUntouched =
                result.operatorStatuses[problem].code ==
                    MR_ARTICULATED_OPERATOR_NONFINITE_INPUT;
            for (std::size_t value = 0u; value < 4u; ++value) {
                operatorPayloadUntouched = operatorPayloadUntouched &&
                    result.factors[4u * problem + value] == 0.0f;
            }
            for (std::size_t value = 0u; value < 12u; ++value) {
                operatorPayloadUntouched = operatorPayloadUntouched &&
                    result.pointJacobians[12u * problem + value] == 0.0f;
            }
        } else {
            operatorPayloadUntouched =
                result.operatorStatuses[problem].code ==
                    MR_ARTICULATED_OPERATOR_SUCCESS;
        }
        rollback = rollback && operatorPayloadUntouched &&
            result.responseStatuses[problem].control.x == expectedResponse &&
            result.publishStatuses[problem].control.x ==
                NUMI_TEMPORAL_CONE_ARTICULATED_UPSTREAM_FAILURE &&
            std::memcmp(&result.outputVelocities[2u * problem],
                &batch.velocities[2u * problem], 2u * sizeof(float)) == 0;
    }
    require(deterministic, "articulated chain is not byte deterministic");
    require(rollback, "articulated invalid-input rollback failed");
    require(failedValid == 0u, "valid articulated islands failed");
    require(maxMassError < 3.0e-6, "articulated mass oracle mismatch");
    require(maxJacobianError < 2.0e-6, "articulated Jacobian oracle mismatch");
    require(maxFiniteDifferenceError < 2.0e-5, "articulated finite-difference mismatch");
    require(maxResponseError < 4.0e-6, "articulated response-column mismatch");
    require(maxBlockError < 2.0e-6, "articulated Delassus block mismatch");
    require(maxFreeVelocityError < 2.0e-6, "articulated free-velocity mismatch");
    require(maxPublicationError < 2.0e-6, "articulated publication mismatch");
    require(maxEnergyBudgetViolation < 2.0e-5, "articulated energy budget violated");

    const double bestSeconds = std::min_element(
        replays.begin(), replays.end(), [](const Result& a, const Result& b) {
            return a.seconds < b.seconds;
        })->seconds;
    std::cout << std::setprecision(9)
              << "numi-solver-articulated device=\"" << device.name.UTF8String << "\""
              << " islands=" << problemCount
              << " valid=" << batch.validProblems
              << " contacts=" << batch.validProblems * kContacts
              << " stages=5 command_buffers=1 readbacks=0"
              << " deterministic=" << (deterministic ? "yes" : "no")
              << " rollback=" << (rollback ? "yes" : "no")
              << " failed_valid=" << failedValid << "\n"
              << "mass_max_abs_error=" << maxMassError
              << " jacobian_max_abs_error=" << maxJacobianError
              << " finite_difference_max_abs_error=" << maxFiniteDifferenceError
              << " response_max_abs_error=" << maxResponseError
              << " delassus_max_abs_error=" << maxBlockError << "\n"
              << "free_velocity_max_abs_error=" << maxFreeVelocityError
              << " publication_max_abs_error=" << maxPublicationError
              << " fp64_solve_residual=" << maxFp64Residual
              << " gpu_response_backward_error=" << maxGpuBackwardError << "\n"
              << "condition_proxy=" << maxConditionProxy
              << " energy_budget_violation=" << std::max(0.0, maxEnergyBudgetViolation)
              << " max_iterations=" << maxIterations << "\n"
              << "gpu_seconds=" << bestSeconds
              << " islands_per_second=" << static_cast<double>(problemCount) / bestSeconds
              << " contacts_per_second=" << static_cast<double>(batch.validProblems * kContacts) / bestSeconds
              << "\n";
    return 0;
}

} // namespace

int main(int argc, const char* const* argv) {
    @autoreleasepool {
        try {
            return run(argc, argv);
        } catch (const std::exception& error) {
            std::cerr << "numi-solver-articulated: " << error.what() << '\n';
            return 1;
        }
    }
}
