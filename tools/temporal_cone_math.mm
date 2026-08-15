#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "numi/temporal_cone_probe.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
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

constexpr double kMatrixFloor = 1.0e-10;
constexpr double kContactMatrixRegularization = 1.0e-2;

struct OracleOutput {
    std::array<double, 3> impulse{};
    std::array<std::array<double, 3>, 3> inverse{};
    std::array<double, 3> residual{};
    double maximumDelta = 0.0;
    double coneViolation = 0.0;
    std::uint32_t status = NUMI_TEMPORAL_CONE_PROBE_SUCCESS;
};

struct GPUResult {
    std::vector<NumiTemporalConeProbeOutput> outputs;
    double seconds = 0.0;
};

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
    const std::uint32_t z = 0u,
    const std::uint32_t w = 0u
) {
    return {x, y, z, w};
}

NumiTemporalConeProbeInput makeInput(
    const std::array<std::array<float, 3>, 3>& response,
    const std::array<float, 3>& freeVelocity,
    const std::array<float, 3>& warmImpulse,
    const float frictionU,
    const float frictionV,
    const float maximumNormalImpulse = 0.0f,
    const std::uint32_t iterations = 8u
) {
    NumiTemporalConeProbeInput input{};
    input.responseRow0 = f4(
        response[0][0], response[0][1], response[0][2]
    );
    input.responseRow1 = f4(
        response[1][0], response[1][1], response[1][2]
    );
    input.responseRow2 = f4(
        response[2][0], response[2][1], response[2][2]
    );
    input.freeVelocityAndFrictionU = f4(
        freeVelocity[0], freeVelocity[1], freeVelocity[2], frictionU
    );
    input.warmImpulseAndFrictionV = f4(
        warmImpulse[0], warmImpulse[1], warmImpulse[2], frictionV
    );
    input.limits = f4(maximumNormalImpulse, 0.0f, 0.0f, 0.0f);
    input.control = u4(
        NUMI_TEMPORAL_CONE_PROBE_ABI_VERSION,
        iterations
    );
    return input;
}

std::array<double, 3> projectCone(
    std::array<double, 3> impulse,
    const double frictionU,
    const double frictionV,
    const double maximumNormalImpulse
) {
    impulse[0] = std::max(impulse[0], 0.0);
    if (maximumNormalImpulse > 0.0) {
        impulse[0] = std::min(impulse[0], maximumNormalImpulse);
    }
    const double limitU = frictionU * impulse[0];
    const double limitV = frictionV * impulse[0];
    if (!(limitU > 0.0) || !(limitV > 0.0)) {
        impulse[1] = 0.0;
        impulse[2] = 0.0;
        return impulse;
    }
    const double normalizedSquared =
        impulse[1] * impulse[1] / (limitU * limitU) +
        impulse[2] * impulse[2] / (limitV * limitV);
    if (normalizedSquared > 1.0) {
        const double scale = 1.0 / std::sqrt(normalizedSquared);
        impulse[1] *= scale;
        impulse[2] *= scale;
    }
    return impulse;
}

bool conditionedInverse(
    const std::array<std::array<double, 3>, 3>& matrix,
    std::array<std::array<double, 3>, 3>& inverse
) {
    double scale = 0.0;
    for (const auto& row : matrix) {
        for (const double value : row) {
            if (!std::isfinite(value)) {
                return false;
            }
            scale = std::max(scale, std::abs(value));
        }
    }
    if (!(scale > kMatrixFloor)) {
        return false;
    }

    std::array<std::array<double, 3>, 3> regularized{};
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            regularized[row][column] =
                0.5 * (matrix[row][column] + matrix[column][row]) /
                scale;
            if (row == column) {
                regularized[row][column] +=
                    kContactMatrixRegularization;
            }
        }
    }

    const double c00 =
        regularized[1][1] * regularized[2][2] -
        regularized[1][2] * regularized[2][1];
    const double c01 =
        regularized[1][2] * regularized[2][0] -
        regularized[1][0] * regularized[2][2];
    const double c02 =
        regularized[1][0] * regularized[2][1] -
        regularized[1][1] * regularized[2][0];
    const double determinant =
        regularized[0][0] * c00 +
        regularized[0][1] * c01 +
        regularized[0][2] * c02;
    if (!(determinant > kMatrixFloor) || !std::isfinite(determinant)) {
        return false;
    }
    const double reciprocal = 1.0 / (determinant * scale);
    inverse[0][0] = c00 * reciprocal;
    inverse[0][1] =
        (regularized[0][2] * regularized[2][1] -
         regularized[0][1] * regularized[2][2]) * reciprocal;
    inverse[0][2] =
        (regularized[0][1] * regularized[1][2] -
         regularized[0][2] * regularized[1][1]) * reciprocal;
    inverse[1][0] = c01 * reciprocal;
    inverse[1][1] =
        (regularized[0][0] * regularized[2][2] -
         regularized[0][2] * regularized[2][0]) * reciprocal;
    inverse[1][2] =
        (regularized[0][2] * regularized[1][0] -
         regularized[0][0] * regularized[1][2]) * reciprocal;
    inverse[2][0] = c02 * reciprocal;
    inverse[2][1] =
        (regularized[0][1] * regularized[2][0] -
         regularized[0][0] * regularized[2][1]) * reciprocal;
    inverse[2][2] =
        (regularized[0][0] * regularized[1][1] -
         regularized[0][1] * regularized[1][0]) * reciprocal;
    for (const auto& row : inverse) {
        for (const double value : row) {
            if (!std::isfinite(value)) {
                return false;
            }
        }
    }
    return true;
}

OracleOutput solveOracle(const NumiTemporalConeProbeInput& input) {
    OracleOutput output;
    if (input.control.x != NUMI_TEMPORAL_CONE_PROBE_ABI_VERSION) {
        output.status = NUMI_TEMPORAL_CONE_PROBE_INVALID_ABI;
        return output;
    }
    if (input.control.y == 0u ||
        input.control.y > NUMI_TEMPORAL_CONE_PROBE_MAX_ITERATIONS) {
        output.status = NUMI_TEMPORAL_CONE_PROBE_INVALID_INPUT;
        return output;
    }

    const std::array<std::array<double, 3>, 3> response{{
        {{input.responseRow0.x, input.responseRow0.y, input.responseRow0.z}},
        {{input.responseRow1.x, input.responseRow1.y, input.responseRow1.z}},
        {{input.responseRow2.x, input.responseRow2.y, input.responseRow2.z}},
    }};
    if (!conditionedInverse(response, output.inverse)) {
        output.status = NUMI_TEMPORAL_CONE_PROBE_FACTORIZATION_FAILED;
        return output;
    }

    const std::array<double, 3> freeVelocity{{
        input.freeVelocityAndFrictionU.x,
        input.freeVelocityAndFrictionU.y,
        input.freeVelocityAndFrictionU.z,
    }};
    const double frictionU = input.freeVelocityAndFrictionU.w;
    const double frictionV = input.warmImpulseAndFrictionV.w;
    const double maximumNormalImpulse = input.limits.x;
    output.impulse = projectCone(
        {{
            input.warmImpulseAndFrictionV.x,
            input.warmImpulseAndFrictionV.y,
            input.warmImpulseAndFrictionV.z,
        }},
        frictionU,
        frictionV,
        maximumNormalImpulse
    );

    for (std::uint32_t iteration = 0u;
         iteration < input.control.y;
         ++iteration) {
        std::array<double, 3> residual{};
        for (std::size_t row = 0; row < 3; ++row) {
            residual[row] = freeVelocity[row];
            for (std::size_t column = 0; column < 3; ++column) {
                residual[row] +=
                    response[row][column] * output.impulse[column];
            }
        }
        std::array<double, 3> proposed = output.impulse;
        for (std::size_t row = 0; row < 3; ++row) {
            for (std::size_t column = 0; column < 3; ++column) {
                proposed[row] -=
                    output.inverse[row][column] * residual[column];
            }
        }
        const auto candidate = projectCone(
            proposed,
            frictionU,
            frictionV,
            maximumNormalImpulse
        );
        output.maximumDelta = 0.0;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            output.maximumDelta = std::max(
                output.maximumDelta,
                std::abs(candidate[axis] - output.impulse[axis])
            );
        }
        output.impulse = candidate;
    }

    for (std::size_t row = 0; row < 3; ++row) {
        output.residual[row] = freeVelocity[row];
        for (std::size_t column = 0; column < 3; ++column) {
            output.residual[row] +=
                response[row][column] * output.impulse[column];
        }
    }
    const double limitU = frictionU * output.impulse[0];
    const double limitV = frictionV * output.impulse[0];
    if (limitU > 0.0 && limitV > 0.0) {
        output.coneViolation = std::max(
            std::sqrt(
                output.impulse[1] * output.impulse[1] /
                    (limitU * limitU) +
                output.impulse[2] * output.impulse[2] /
                    (limitV * limitV)
            ) - 1.0,
            0.0
        );
    } else {
        output.coneViolation = std::hypot(
            output.impulse[1], output.impulse[2]
        );
    }
    return output;
}

std::vector<NumiTemporalConeProbeInput> makeProblems(
    const std::size_t count
) {
    const std::array<std::array<float, 3>, 3> identity{{
        {{1.0f, 0.0f, 0.0f}},
        {{0.0f, 1.0f, 0.0f}},
        {{0.0f, 0.0f, 1.0f}},
    }};
    std::vector<NumiTemporalConeProbeInput> inputs;
    inputs.reserve(count);
    inputs.push_back(makeInput(
        identity, {{1.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f}},
        0.6f, 0.6f
    ));
    inputs.push_back(makeInput(
        identity, {{-2.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f}},
        0.0f, 0.0f
    ));
    inputs.push_back(makeInput(
        identity, {{-1.0f, -0.2f, 0.1f}}, {{0.1f, 0.0f, 0.0f}},
        0.6f, 0.6f
    ));
    inputs.push_back(makeInput(
        identity, {{-1.0f, -2.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f}},
        0.5f, 0.5f
    ));
    inputs.push_back(makeInput(
        identity, {{-1.0f, -1.0f, 1.0f}}, {{0.0f, 0.0f, 0.0f}},
        0.8f, 0.35f
    ));
    inputs.push_back(makeInput(
        {{{{1.0f, 0.0f, 0.0f}},
          {{0.0f, 1.0e-10f, 0.0f}},
          {{0.0f, 0.0f, 1.0e-10f}}}},
        {{-0.5f, -0.1f, 0.05f}}, {{0.0f, 0.0f, 0.0f}},
        0.7f, 0.4f
    ));
    inputs.push_back(makeInput(
        identity, {{-5.0f, -1.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f}},
        0.5f, 0.5f, 1.25f
    ));
    inputs.push_back(makeInput(
        {{{{1.2f, 0.08f, -0.02f}},
          {{0.07f, 0.9f, 0.04f}},
          {{-0.01f, 0.03f, 0.7f}}}},
        {{-0.8f, -0.4f, 0.2f}}, {{0.05f, 0.01f, -0.01f}},
        0.55f, 0.45f
    ));
    inputs.push_back(makeInput(
        {{{{0.0f, 0.0f, 0.0f}},
          {{0.0f, 0.0f, 0.0f}},
          {{0.0f, 0.0f, 0.0f}}}},
        {{-1.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f}},
        0.5f, 0.5f
    ));

    for (std::size_t index = inputs.size(); index < count; ++index) {
        const float a = 0.25f + 0.01f * static_cast<float>(index % 71u);
        const float b = 0.35f + 0.01f * static_cast<float>(index % 53u);
        const float c = 0.45f + 0.01f * static_cast<float>(index % 37u);
        const float d = 0.03f * static_cast<float>(
            static_cast<int>(index % 9u) - 4
        );
        const float e = 0.02f * static_cast<float>(
            static_cast<int>(index % 7u) - 3
        );
        const float f = 0.025f * static_cast<float>(
            static_cast<int>(index % 5u) - 2
        );
        const std::array<std::array<float, 3>, 3> response{{
            {{a * a, a * d, a * e}},
            {{a * d, d * d + b * b, d * e + b * f}},
            {{a * e, d * e + b * f, e * e + f * f + c * c}},
        }};
        const float normalVelocity =
            index % 5u == 0u
            ? 0.1f + 0.01f * static_cast<float>(index % 17u)
            : -0.05f - 0.01f * static_cast<float>(index % 113u);
        const std::array<float, 3> freeVelocity{{
            normalVelocity,
            0.5f * std::sin(static_cast<float>(index) * 0.17f),
            0.5f * std::cos(static_cast<float>(index) * 0.11f),
        }};
        const std::array<float, 3> warm{{
            0.01f * static_cast<float>(index % 13u),
            0.005f * static_cast<float>(
                static_cast<int>(index % 7u) - 3
            ),
            0.004f * static_cast<float>(
                static_cast<int>(index % 9u) - 4
            ),
        }};
        inputs.push_back(makeInput(
            response,
            freeVelocity,
            warm,
            0.15f + 0.05f * static_cast<float>(index % 12u),
            0.20f + 0.04f * static_cast<float>(index % 10u),
            index % 29u == 0u ? 2.0f : 0.0f,
            8u
        ));
    }
    return inputs;
}

std::string errorText(NSError* error) {
    return error == nil
        ? std::string("unknown Metal error")
        : std::string(error.localizedDescription.UTF8String);
}

GPUResult runGPU(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> pipeline,
    const std::vector<NumiTemporalConeProbeInput>& inputs
) {
    const NSUInteger inputBytes =
        inputs.size() * sizeof(NumiTemporalConeProbeInput);
    const NSUInteger outputBytes =
        inputs.size() * sizeof(NumiTemporalConeProbeOutput);
    id<MTLBuffer> inputBuffer = [device
        newBufferWithBytes:inputs.data()
                   length:inputBytes
                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [device
        newBufferWithLength:outputBytes
                    options:MTLResourceStorageModeShared];
    if (inputBuffer == nil || outputBuffer == nil) {
        throw std::runtime_error("failed to allocate shared Metal buffers");
    }
    std::memset(outputBuffer.contents, 0, outputBytes);

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (commandBuffer == nil || encoder == nil) {
        throw std::runtime_error("failed to create Metal command encoder");
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:inputBuffer offset:0 atIndex:0];
    [encoder setBuffer:outputBuffer offset:0 atIndex:1];
    const std::uint32_t problemCount =
        static_cast<std::uint32_t>(inputs.size());
    [encoder setBytes:&problemCount length:sizeof(problemCount) atIndex:2];
    const NSUInteger width = std::min<NSUInteger>(
        pipeline.maxTotalThreadsPerThreadgroup,
        256u
    );
    [encoder dispatchThreads:MTLSizeMake(inputs.size(), 1u, 1u)
          threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted ||
        commandBuffer.error != nil) {
        throw std::runtime_error(
            "Metal command failed: " + errorText(commandBuffer.error)
        );
    }

    GPUResult result;
    const auto* values = static_cast<const NumiTemporalConeProbeOutput*>(
        outputBuffer.contents
    );
    result.outputs.assign(values, values + inputs.size());
    if (commandBuffer.GPUEndTime >= commandBuffer.GPUStartTime) {
        result.seconds =
            commandBuffer.GPUEndTime - commandBuffer.GPUStartTime;
    }
    return result;
}

double normalizedError(const double actual, const double expected) {
    return std::abs(actual - expected) /
        std::max(1.0, std::abs(expected));
}

int run(const int argc, const char* const* argv) {
    std::size_t problemCount = 65536u;
    std::uint32_t replayCount = 2u;
    std::string metallibPath = NUMI_TEMPORAL_CONE_METALLIB;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string_view value(argv[argument]);
        if (value == "--cases" && argument + 1 < argc) {
            problemCount = std::stoull(argv[++argument]);
        } else if (value == "--replays" && argument + 1 < argc) {
            replayCount = static_cast<std::uint32_t>(
                std::stoul(argv[++argument])
            );
        } else if (value == "--metallib" && argument + 1 < argc) {
            metallibPath = argv[++argument];
        } else if (value == "--help") {
            std::cout
                << "usage: numi-solver-math [--cases N] [--replays N] "
                   "[--metallib PATH]\n";
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + std::string(value));
        }
    }
    problemCount = std::max<std::size_t>(problemCount, 9u);
    replayCount = std::max<std::uint32_t>(replayCount, 2u);
    if (problemCount > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("case count exceeds the probe ABI");
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        throw std::runtime_error("no Apple Metal device is available");
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
        throw std::runtime_error("failed to create the Metal command queue");
    }
    NSString* metalLibraryPath = [NSString
        stringWithUTF8String:metallibPath.c_str()];
    NSError* error = nil;
    id<MTLLibrary> library = [device
        newLibraryWithURL:[NSURL fileURLWithPath:metalLibraryPath]
                    error:&error];
    if (library == nil) {
        throw std::runtime_error(
            "failed to load solver metallib: " + errorText(error)
        );
    }
    id<MTLFunction> function =
        [library newFunctionWithName:@"numi_temporal_cone_probe"];
    if (function == nil) {
        throw std::runtime_error("solver metallib lacks the math probe kernel");
    }
    id<MTLComputePipelineState> pipeline = [device
        newComputePipelineStateWithFunction:function
                                      error:&error];
    if (pipeline == nil) {
        throw std::runtime_error(
            "failed to create the math probe pipeline: " + errorText(error)
        );
    }

    const auto inputs = makeProblems(problemCount);
    std::vector<GPUResult> replays;
    replays.reserve(replayCount);
    for (std::uint32_t replay = 0u; replay < replayCount; ++replay) {
        replays.push_back(runGPU(device, queue, pipeline, inputs));
    }

    bool deterministic = true;
    for (std::size_t replay = 1; replay < replays.size(); ++replay) {
        deterministic = deterministic &&
            std::memcmp(
                replays[0].outputs.data(),
                replays[replay].outputs.data(),
                replays[0].outputs.size() *
                    sizeof(NumiTemporalConeProbeOutput)
            ) == 0;
    }

    std::size_t failedCases = 0u;
    double maximumRelativeError = 0.0;
    double maximumConeViolation = 0.0;
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        const auto oracle = solveOracle(inputs[index]);
        const auto& gpu = replays[0].outputs[index];
        bool valid = gpu.status.x == oracle.status;
        if (gpu.status.x == NUMI_TEMPORAL_CONE_PROBE_SUCCESS) {
            const std::array<double, 3> gpuImpulse{{
                gpu.impulseAndDelta.x,
                gpu.impulseAndDelta.y,
                gpu.impulseAndDelta.z,
            }};
            const std::array<double, 3> gpuResidual{{
                gpu.residualAndConeViolation.x,
                gpu.residualAndConeViolation.y,
                gpu.residualAndConeViolation.z,
            }};
            for (std::size_t axis = 0; axis < 3; ++axis) {
                maximumRelativeError = std::max(
                    maximumRelativeError,
                    normalizedError(gpuImpulse[axis], oracle.impulse[axis])
                );
                maximumRelativeError = std::max(
                    maximumRelativeError,
                    normalizedError(gpuResidual[axis], oracle.residual[axis])
                );
                valid = valid && std::isfinite(gpuImpulse[axis]);
            }
            maximumConeViolation = std::max(
                maximumConeViolation,
                static_cast<double>(gpu.residualAndConeViolation.w)
            );
            valid = valid && gpuImpulse[0] >= -1.0e-6;
            if (inputs[index].limits.x > 0.0f) {
                valid = valid &&
                    gpuImpulse[0] <= inputs[index].limits.x + 1.0e-5;
            }
            valid = valid && gpu.residualAndConeViolation.w <= 2.0e-5f;
        }
        if (!valid) {
            ++failedCases;
        }
    }

    const auto& separating = replays[0].outputs[0];
    const bool separatingAcceptedZero =
        std::abs(separating.impulseAndDelta.x) <= 1.0e-6f &&
        std::abs(separating.impulseAndDelta.y) <= 1.0e-6f &&
        std::abs(separating.impulseAndDelta.z) <= 1.0e-6f;
    const auto& sliding = replays[0].outputs[3];
    const double slidingLimit = 0.5 * sliding.impulseAndDelta.x;
    const bool slidingOnCone =
        sliding.impulseAndDelta.x > 0.0f &&
        std::abs(std::abs(sliding.impulseAndDelta.y) - slidingLimit) <=
            2.0e-5;
    const auto& capped = replays[0].outputs[6];
    const bool normalCapRespected =
        capped.impulseAndDelta.x <= 1.25f + 1.0e-5f;

    double totalGPUSeconds = 0.0;
    for (const auto& replay : replays) {
        totalGPUSeconds += replay.seconds;
    }
    const double averageGPUSeconds =
        totalGPUSeconds / static_cast<double>(replays.size());
    const double casesPerSecond = averageGPUSeconds > 0.0
        ? static_cast<double>(problemCount) / averageGPUSeconds
        : 0.0;

    const bool passed =
        failedCases == 0u &&
        maximumRelativeError <= 5.0e-4 &&
        maximumConeViolation <= 2.0e-5 &&
        deterministic &&
        separatingAcceptedZero &&
        slidingOnCone &&
        normalCapRespected;

    std::cout << std::fixed << std::setprecision(9)
              << "device=" << device.name.UTF8String << '\n'
              << "cases=" << problemCount
              << " replays=" << replayCount
              << " failed_cases=" << failedCases << '\n'
              << "max_fp64_relative_error=" << maximumRelativeError
              << " max_cone_violation=" << maximumConeViolation << '\n'
              << "deterministic_replay=" << (deterministic ? "true" : "false")
              << " separating_zero="
              << (separatingAcceptedZero ? "true" : "false")
              << " sliding_on_cone=" << (slidingOnCone ? "true" : "false")
              << " normal_cap=" << (normalCapRespected ? "true" : "false")
              << '\n'
              << "average_gpu_seconds=" << averageGPUSeconds
              << " cases_per_second=" << casesPerSecond << '\n'
              << "result=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
}

} // namespace

int main(const int argc, const char* const* argv) {
    @autoreleasepool {
        try {
            return run(argc, argv);
        } catch (const std::exception& error) {
            std::cerr << "numi-solver-math: " << error.what() << '\n';
            return 2;
        }
    }
}
