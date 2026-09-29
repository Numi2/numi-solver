cmake_minimum_required(VERSION 3.28)

project(
    NumiSolver
    VERSION 0.1.0
    DESCRIPTION "Apple Metal Temporal Cone contact solver"
    LANGUAGES CXX
)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(
        CMAKE_BUILD_TYPE Release
        CACHE STRING "Build type for single-configuration generators" FORCE
    )
endif()

if(NOT APPLE)
    message(FATAL_ERROR "Numi Solver requires macOS and the Apple Metal toolchain.")
endif()

find_program(XCRUN_EXECUTABLE xcrun REQUIRED)
enable_language(OBJCXX)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)

set(NUMI_SOLVER_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(NUMI_SOLVER_AIR "${NUMI_SOLVER_SHADER_DIR}/NumiTemporalCone.air")
set(NUMI_SOLVER_BRAIDED_BAG_AIR "${NUMI_SOLVER_SHADER_DIR}/NumiBraidedBag.air")
set(NUMI_SOLVER_CLOTH_BAG_AIR "${NUMI_SOLVER_SHADER_DIR}/NumiClothBag.air")
set(NUMI_SOLVER_ARTICULATED_AIR "${NUMI_SOLVER_SHADER_DIR}/NumiArticulatedOperator.air")
set(NUMI_SOLVER_INVERSE_ABA_AIR "${NUMI_SOLVER_SHADER_DIR}/NumiArticulatedInverseABA.air")
set(NUMI_SOLVER_METALLIB "${NUMI_SOLVER_SHADER_DIR}/NumiTemporalCone.metallib")
set(NUMI_SOLVER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/MetalWorldContact.metal")
set(
    NUMI_SOLVER_BRAIDED_BAG_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/BraidedBag.metal"
)
set(
    NUMI_SOLVER_CLOTH_BAG_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/ClothBag.metal"
)
set(
    NUMI_SOLVER_ARTICULATED_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/ArticulatedOperator.metal"
)
set(
    NUMI_SOLVER_INVERSE_ABA_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/ArticulatedInverseMass.metal"
)
set(
    NUMI_SOLVER_HEADERS
    include/metalrobo/constraint_ir_shared.h
    include/metalrobo/engine_types.h
    include/metalrobo/gpu_types.h
    include/metalrobo/parallel_aba_shared.h
    include/metalrobo/rod_gpu_shared.h
    include/metalrobo/unified_quality_shared.h
    include/numi/temporal_cone_island.h
    include/numi/temporal_cone_probe.h
    include/numi/braided_bag.h
    include/numi/cloth_bag_gpu.h
    include/numi/cloth_gpu_dispatch_guards.h
    include/numi/static_contact_geometry.h
)

add_custom_command(
    OUTPUT "${NUMI_SOLVER_METALLIB}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${NUMI_SOLVER_SHADER_DIR}"
    COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal
        -std=metal4.0
        -O3
        -fno-fast-math
        -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
        -c "${NUMI_SOLVER_SOURCE}"
        -o "${NUMI_SOLVER_AIR}"
    COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal
        -std=metal4.0
        -O3
        -fno-fast-math
        -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
        -c "${NUMI_SOLVER_BRAIDED_BAG_SOURCE}"
        -o "${NUMI_SOLVER_BRAIDED_BAG_AIR}"
    COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal
        -std=metal4.0
        -O3
        -fno-fast-math
        -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
        -c "${NUMI_SOLVER_CLOTH_BAG_SOURCE}"
        -o "${NUMI_SOLVER_CLOTH_BAG_AIR}"
    COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal
        -std=metal4.0
        -O3
        -fno-fast-math
        -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
        -c "${NUMI_SOLVER_ARTICULATED_SOURCE}"
        -o "${NUMI_SOLVER_ARTICULATED_AIR}"
    COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal
        -std=metal4.0
        -O3
        -fno-fast-math
        -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
        -DMR_ARTICULATED_ABA_MAX_BODIES=33u
        -DMR_INVERSE_MASS_KERNEL_NAME=numi_articulated_inverse_aba_stream
        -DMR_INVERSE_MASS_STREAMING_RHS=1
        -c "${NUMI_SOLVER_INVERSE_ABA_SOURCE}"
        -o "${NUMI_SOLVER_INVERSE_ABA_AIR}"
    COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metallib
        "${NUMI_SOLVER_AIR}"
        "${NUMI_SOLVER_BRAIDED_BAG_AIR}"
        "${NUMI_SOLVER_CLOTH_BAG_AIR}"
        "${NUMI_SOLVER_ARTICULATED_AIR}"
        "${NUMI_SOLVER_INVERSE_ABA_AIR}"
        -o "${NUMI_SOLVER_METALLIB}"
    DEPENDS
        "${NUMI_SOLVER_SOURCE}"
        "${NUMI_SOLVER_BRAIDED_BAG_SOURCE}"
        "${NUMI_SOLVER_CLOTH_BAG_SOURCE}"
        "${NUMI_SOLVER_ARTICULATED_SOURCE}"
        "${NUMI_SOLVER_INVERSE_ABA_SOURCE}"
        ${NUMI_SOLVER_HEADERS}
    COMMENT "Compiling the Temporal Cone solver metallib"
    VERBATIM
)

add_custom_target(numi_temporal_cone ALL DEPENDS "${NUMI_SOLVER_METALLIB}")

add_library(numi_temporal_cone_abi INTERFACE)
target_include_directories(
    numi_temporal_cone_abi
    INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)

# Keep volumetric development in a separate library so frozen cloth replay
# binaries and their metallib fingerprints remain meaningful.
set(NUMI_DEFORMABLE_TET_METALLIB "${NUMI_SOLVER_SHADER_DIR}/NumiDeformableTet.metallib")
add_custom_command(
    OUTPUT "${NUMI_DEFORMABLE_TET_METALLIB}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${NUMI_SOLVER_SHADER_DIR}"
    COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math
        -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
        -c "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/DeformableTet.metal"
        -o "${NUMI_SOLVER_SHADER_DIR}/NumiDeformableTet.air"
    COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metallib
        "${NUMI_SOLVER_SHADER_DIR}/NumiDeformableTet.air" -o "${NUMI_DEFORMABLE_TET_METALLIB}"
    DEPENDS src/metal/DeformableTet.metal include/numi/deformable_tet.h include/metalrobo/gpu_types.h
    VERBATIM
)
add_custom_target(numi_deformable_tet DEPENDS "${NUMI_DEFORMABLE_TET_METALLIB}")
add_executable(numi-solver-deformable-tet tools/deformable_tet.mm)
add_dependencies(numi-solver-deformable-tet numi_deformable_tet)
target_link_libraries(numi-solver-deformable-tet PRIVATE numi_temporal_cone_abi
    "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
target_compile_features(numi-solver-deformable-tet PRIVATE cxx_std_23)
target_compile_options(numi-solver-deformable-tet PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
target_compile_definitions(numi-solver-deformable-tet PRIVATE
    NUMI_DEFORMABLE_TET_METALLIB="${NUMI_DEFORMABLE_TET_METALLIB}")

set(NUMI_DEFORMABLE_MESH_METALLIB "${NUMI_SOLVER_SHADER_DIR}/NumiDeformableMesh.metallib")
add_custom_command(
    OUTPUT "${NUMI_DEFORMABLE_MESH_METALLIB}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${NUMI_SOLVER_SHADER_DIR}"
    COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math
        -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
        -c "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/DeformableMesh.metal"
        -o "${NUMI_SOLVER_SHADER_DIR}/NumiDeformableMesh.air"
    COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metallib
        "${NUMI_SOLVER_SHADER_DIR}/NumiDeformableMesh.air" -o "${NUMI_DEFORMABLE_MESH_METALLIB}"
    DEPENDS src/metal/DeformableMesh.metal src/metal/DeformableTet.metal
        include/numi/deformable_mesh.h include/numi/deformable_tet.h include/metalrobo/gpu_types.h
    VERBATIM
)
add_custom_target(numi_deformable_mesh DEPENDS "${NUMI_DEFORMABLE_MESH_METALLIB}")
add_executable(numi-solver-deformable-mesh tools/deformable_mesh.mm)
add_dependencies(numi-solver-deformable-mesh numi_deformable_mesh)
target_link_libraries(numi-solver-deformable-mesh PRIVATE numi_temporal_cone_abi
    "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")
target_compile_features(numi-solver-deformable-mesh PRIVATE cxx_std_23)
target_compile_options(numi-solver-deformable-mesh PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
target_compile_definitions(numi-solver-deformable-mesh PRIVATE
    NUMI_DEFORMABLE_MESH_METALLIB="${NUMI_DEFORMABLE_MESH_METALLIB}")

add_executable(numi-solver-math tools/temporal_cone_math.mm)
add_dependencies(numi-solver-math numi_temporal_cone)
target_link_libraries(
    numi-solver-math
    PRIVATE
        numi_temporal_cone_abi
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
target_compile_features(numi-solver-math PRIVATE cxx_std_23)
target_compile_options(numi-solver-math PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic)
target_compile_definitions(
    numi-solver-math
    PRIVATE
        NUMI_TEMPORAL_CONE_METALLIB="${NUMI_SOLVER_METALLIB}"
)

add_executable(numi-solver-islands tools/temporal_cone_islands.mm)
add_dependencies(numi-solver-islands numi_temporal_cone)
target_link_libraries(
    numi-solver-islands
    PRIVATE
        numi_temporal_cone_abi
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
target_compile_features(numi-solver-islands PRIVATE cxx_std_23)
target_compile_options(
    numi-solver-islands
    PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic
)
target_compile_definitions(
    numi-solver-islands
    PRIVATE
        NUMI_TEMPORAL_CONE_METALLIB="${NUMI_SOLVER_METALLIB}"
)

add_executable(numi-solver-assembly tools/temporal_cone_assembly.mm)
add_dependencies(numi-solver-assembly numi_temporal_cone)
target_link_libraries(
    numi-solver-assembly
    PRIVATE
        numi_temporal_cone_abi
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
target_compile_features(numi-solver-assembly PRIVATE cxx_std_23)
target_compile_options(
    numi-solver-assembly
    PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic
)
target_compile_definitions(
    numi-solver-assembly
    PRIVATE
        NUMI_TEMPORAL_CONE_METALLIB="${NUMI_SOLVER_METALLIB}"
)

add_executable(numi-solver-rigid tools/temporal_cone_rigid.mm)
add_dependencies(numi-solver-rigid numi_temporal_cone)
target_link_libraries(
    numi-solver-rigid
    PRIVATE
        numi_temporal_cone_abi
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
target_compile_features(numi-solver-rigid PRIVATE cxx_std_23)
target_compile_options(
    numi-solver-rigid
    PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic
)
target_compile_definitions(
    numi-solver-rigid
    PRIVATE
        NUMI_TEMPORAL_CONE_METALLIB="${NUMI_SOLVER_METALLIB}"
)

add_executable(numi-solver-braided-bag tools/braided_bag.mm)
add_dependencies(numi-solver-braided-bag numi_temporal_cone)
target_link_libraries(
    numi-solver-braided-bag
    PRIVATE
        numi_temporal_cone_abi
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
target_compile_features(numi-solver-braided-bag PRIVATE cxx_std_23)
target_compile_options(
    numi-solver-braided-bag
    PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic
)
target_compile_definitions(
    numi-solver-braided-bag
    PRIVATE
        NUMI_TEMPORAL_CONE_METALLIB="${NUMI_SOLVER_METALLIB}"
)

# Independent native static-geometry qualification does not rebuild the frozen
# cloth replay library or imply qualification of full cloth/bench responses.
set(NUMI_STATIC_GEOMETRY_METALLIB "${NUMI_SOLVER_SHADER_DIR}/NumiStaticGeometry.metallib")
add_custom_command(
    OUTPUT "${NUMI_STATIC_GEOMETRY_METALLIB}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${NUMI_SOLVER_SHADER_DIR}"
    COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math
        -I "${CMAKE_CURRENT_SOURCE_DIR}/include"
        -c "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/StaticContactGeometry.metal"
        -o "${NUMI_SOLVER_SHADER_DIR}/NumiStaticGeometry.air"
    COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metallib
        "${NUMI_SOLVER_SHADER_DIR}/NumiStaticGeometry.air" -o "${NUMI_STATIC_GEOMETRY_METALLIB}"
    DEPENDS src/metal/StaticContactGeometry.metal include/numi/static_contact_geometry.h
    VERBATIM)
add_custom_target(numi_static_geometry DEPENDS "${NUMI_STATIC_GEOMETRY_METALLIB}")
add_executable(numi-solver-static-geometry tools/static_contact_geometry.cpp)
target_include_directories(numi-solver-static-geometry PRIVATE include)
target_compile_features(numi-solver-static-geometry PRIVATE cxx_std_23)
target_compile_options(numi-solver-static-geometry PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
add_executable(numi-solver-static-segment tools/static_segment_geometry.cpp)
target_include_directories(numi-solver-static-segment PRIVATE include)
target_compile_features(numi-solver-static-segment PRIVATE cxx_std_23)
target_compile_options(numi-solver-static-segment PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
add_executable(numi-solver-static-velocity tools/static_velocity_contact.cpp)
target_include_directories(numi-solver-static-velocity PRIVATE include)
target_compile_features(numi-solver-static-velocity PRIVATE cxx_std_23)
target_compile_options(numi-solver-static-velocity PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
add_executable(numi-solver-cloth-dispatch-guard tests/dispatch_guard_host.cpp)
target_include_directories(numi-solver-cloth-dispatch-guard PRIVATE include)
target_compile_features(numi-solver-cloth-dispatch-guard PRIVATE cxx_std_20)
add_executable(numi-solver-transactional-substep tools/transactional_substep.cpp)
target_include_directories(numi-solver-transactional-substep PRIVATE include)
target_compile_features(numi-solver-transactional-substep PRIVATE cxx_std_23)
target_compile_options(numi-solver-transactional-substep PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
add_executable(numi-solver-elastic-yarn tools/elastic_yarn_contact_probe.cpp)
target_include_directories(numi-solver-elastic-yarn PRIVATE include)
target_compile_features(numi-solver-elastic-yarn PRIVATE cxx_std_23)
target_compile_options(numi-solver-elastic-yarn PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
add_executable(numi-solver-static-geometry-metal tools/static_contact_geometry_metal.mm)
add_dependencies(numi-solver-static-geometry-metal numi_static_geometry)
target_include_directories(numi-solver-static-geometry-metal PRIVATE include)
set_target_properties(numi-solver-static-geometry-metal PROPERTIES OBJCXX_STANDARD 23 OBJCXX_STANDARD_REQUIRED YES)
target_compile_options(numi-solver-static-geometry-metal PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic -ffp-contract=off)
target_compile_definitions(numi-solver-static-geometry-metal PRIVATE
    NUMI_STATIC_GEOMETRY_METALLIB="${NUMI_STATIC_GEOMETRY_METALLIB}")
target_link_libraries(numi-solver-static-geometry-metal PRIVATE "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")

add_executable(numi-solver-cloth-bag tools/cloth_bag.cpp)
add_executable(numi-solver-finite-bench tools/finite_bench.cpp)
target_include_directories(numi-solver-finite-bench PRIVATE include)
target_compile_features(numi-solver-finite-bench PRIVATE cxx_std_23)
target_compile_options(numi-solver-finite-bench PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(numi-solver-cloth-bag PRIVATE numi_temporal_cone_abi)
target_compile_features(numi-solver-cloth-bag PRIVATE cxx_std_23)
target_compile_options(
    numi-solver-cloth-bag
    PRIVATE -Wall -Wextra -Wpedantic
)

add_executable(numi-solver-cloth-calibration tools/cloth_calibration.cpp)
target_compile_features(numi-solver-cloth-calibration PRIVATE cxx_std_23)
target_compile_options(
    numi-solver-cloth-calibration
    PRIVATE -Wall -Wextra -Wpedantic
)

add_executable(numi-solver-cloth-metal tools/cloth_bag_metal.mm)
add_dependencies(numi-solver-cloth-metal numi_temporal_cone)
target_link_libraries(
    numi-solver-cloth-metal
    PRIVATE
        numi_temporal_cone_abi
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
target_compile_features(numi-solver-cloth-metal PRIVATE cxx_std_23)
target_compile_options(
    numi-solver-cloth-metal
    PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic
)
target_compile_definitions(
    numi-solver-cloth-metal
    PRIVATE
        NUMI_TEMPORAL_CONE_METALLIB="${NUMI_SOLVER_METALLIB}"
)

add_executable(numi-solver-articulated tools/temporal_cone_articulated.mm)
add_dependencies(numi-solver-articulated numi_temporal_cone)
target_link_libraries(
    numi-solver-articulated
    PRIVATE
        numi_temporal_cone_abi
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
target_compile_features(numi-solver-articulated PRIVATE cxx_std_23)
target_compile_options(
    numi-solver-articulated
    PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic
)
target_compile_definitions(
    numi-solver-articulated
    PRIVATE
        NUMI_TEMPORAL_CONE_METALLIB="${NUMI_SOLVER_METALLIB}"
)

add_executable(
    numi-solver-articulated-capacity
    tools/temporal_cone_articulated.mm
)
add_dependencies(numi-solver-articulated-capacity numi_temporal_cone)
target_link_libraries(
    numi-solver-articulated-capacity
    PRIVATE
        numi_temporal_cone_abi
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
target_compile_features(numi-solver-articulated-capacity PRIVATE cxx_std_23)
target_compile_options(
    numi-solver-articulated-capacity
    PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic
)
target_compile_definitions(
    numi-solver-articulated-capacity
    PRIVATE
        NUMI_ARTICULATED_ARMATURE_SCALE=16.0
        NUMI_ARTICULATED_CHAIN_DOF=32u
        NUMI_TEMPORAL_CONE_METALLIB="${NUMI_SOLVER_METALLIB}"
)

add_executable(
    numi-solver-articulated-conditioning
    tools/temporal_cone_articulated.mm
)
add_dependencies(numi-solver-articulated-conditioning numi_temporal_cone)
target_link_libraries(
    numi-solver-articulated-conditioning
    PRIVATE
        numi_temporal_cone_abi
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
target_compile_features(numi-solver-articulated-conditioning PRIVATE cxx_std_23)
target_compile_options(
    numi-solver-articulated-conditioning
    PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic
)
target_compile_definitions(
    numi-solver-articulated-conditioning
    PRIVATE
        NUMI_ARTICULATED_ARMATURE_SCALE=1.0
        NUMI_ARTICULATED_CHAIN_DOF=32u
        NUMI_ARTICULATED_EXPECT_CONDITION_FAILURE=1
        NUMI_TEMPORAL_CONE_METALLIB="${NUMI_SOLVER_METALLIB}"
)

add_executable(
    numi-solver-articulated-zero-armature
    tools/temporal_cone_articulated.mm
)
add_dependencies(numi-solver-articulated-zero-armature numi_temporal_cone)
target_link_libraries(
    numi-solver-articulated-zero-armature
    PRIVATE
        numi_temporal_cone_abi
        "${FOUNDATION_FRAMEWORK}"
        "${METAL_FRAMEWORK}"
)
target_compile_features(
    numi-solver-articulated-zero-armature PRIVATE cxx_std_23
)
target_compile_options(
    numi-solver-articulated-zero-armature
    PRIVATE -fobjc-arc -Wall -Wextra -Wpedantic
)
target_compile_definitions(
    numi-solver-articulated-zero-armature
    PRIVATE
        NUMI_ARTICULATED_ARMATURE_SCALE=0.0
        NUMI_ARTICULATED_CHAIN_DOF=32u
        NUMI_ARTICULATED_EXPECT_CONDITION_FAILURE=1
        NUMI_TEMPORAL_CONE_METALLIB="${NUMI_SOLVER_METALLIB}"
)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME static_contact.host COMMAND $<TARGET_FILE:numi-solver-static-geometry>)
    add_test(NAME static_contact.whole_yarn_host COMMAND $<TARGET_FILE:numi-solver-static-segment>)
    set_tests_properties(static_contact.whole_yarn_host PROPERTIES LABELS "cpu;physics;contact;finite_bench;ccd" TIMEOUT 60)
    add_test(NAME static_contact.velocity_slack_host COMMAND $<TARGET_FILE:numi-solver-static-velocity>)
    add_test(NAME cloth_bag.dispatch_guard_host COMMAND $<TARGET_FILE:numi-solver-cloth-dispatch-guard>)
    set_tests_properties(cloth_bag.dispatch_guard_host PROPERTIES LABELS "cpu;physics;cloth;capacity" TIMEOUT 30)
    set_tests_properties(static_contact.velocity_slack_host PROPERTIES LABELS "cpu;physics;contact;finite_bench;energy;friction" TIMEOUT 30)
    add_test(NAME static_contact.transactional_substep_host COMMAND $<TARGET_FILE:numi-solver-transactional-substep>)
    set_tests_properties(static_contact.transactional_substep_host PROPERTIES LABELS "cpu;physics;contact;finite_bench;rollback" TIMEOUT 30)
    add_test(NAME elastic_yarn.normal_host COMMAND $<TARGET_FILE:numi-solver-elastic-yarn>)
    add_test(NAME elastic_yarn.fem_boundary_host COMMAND $<TARGET_FILE:numi-solver-elastic-yarn> --fixture "${CMAKE_CURRENT_SOURCE_DIR}/docs/assets/elastic-yarn-fem-frame38-five-nodes.txt")
    set_tests_properties(elastic_yarn.normal_host elastic_yarn.fem_boundary_host PROPERTIES LABELS "cpu;physics;deformable;contact;reciprocal;energy" TIMEOUT 30)
    add_test(NAME static_contact.metal COMMAND $<TARGET_FILE:numi-solver-static-geometry-metal>)
    set_tests_properties(static_contact.host PROPERTIES LABELS "cpu;physics;contact;finite_bench" TIMEOUT 30)
    set_tests_properties(static_contact.metal PROPERTIES LABELS "metal;physics;contact;finite_bench" TIMEOUT 30)
    add_test(NAME finite_bench.reference COMMAND $<TARGET_FILE:numi-solver-finite-bench>)
    add_test(NAME cloth_bag.finite_bench COMMAND $<TARGET_FILE:numi-solver-cloth-bag> --finite-bench-probe)
    add_test(NAME cloth_bag.coupled_yarn COMMAND $<TARGET_FILE:numi-solver-cloth-bag> --coupled-yarn-probe)
    add_test(NAME cloth_bag.coupled_yarn_peak COMMAND $<TARGET_FILE:numi-solver-cloth-bag>
        --coupled-yarn-snapshot-probe "${CMAKE_CURRENT_SOURCE_DIR}/docs/assets/finite-bench-load-pickup-96-failed-peak.obj")
    set_tests_properties(cloth_bag.coupled_yarn cloth_bag.coupled_yarn_peak
        PROPERTIES LABELS "cpu;physics;cloth;contact;finite_bench;coupled" TIMEOUT 30)
    set_tests_properties(finite_bench.reference cloth_bag.finite_bench
        PROPERTIES LABELS "physics;cloth;contact;ccd;finite_bench" TIMEOUT 30)
    add_test(NAME deformable_tet.constitutive COMMAND $<TARGET_FILE:numi-solver-deformable-tet>)
    set_tests_properties(deformable_tet.constitutive PROPERTIES LABELS "metal;physics;deformable" TIMEOUT 30)
    add_test(NAME deformable_mesh.topology COMMAND $<TARGET_FILE:numi-solver-deformable-mesh> --topology-probe)
    add_test(NAME deformable_mesh.shortest_topology COMMAND $<TARGET_FILE:numi-solver-deformable-mesh> --topology-probe --refinement-diagonal shortest)
    set_tests_properties(deformable_mesh.shortest_topology PROPERTIES LABELS "cpu;geometry;deformable;mesh_refinement" TIMEOUT 30)
    set_tests_properties(deformable_mesh.topology
        PROPERTIES LABELS "cpu;geometry;deformable;mesh_refinement" TIMEOUT 30)
    add_test(NAME deformable_mesh.drop COMMAND $<TARGET_FILE:numi-solver-deformable-mesh>)
    set_tests_properties(deformable_mesh.drop PROPERTIES LABELS "metal;physics;deformable;contact" TIMEOUT 60)
    add_test(NAME deformable_mesh.refined_drop
        COMMAND $<TARGET_FILE:numi-solver-deformable-mesh> --mesh-refinement 1)
    set_tests_properties(deformable_mesh.refined_drop
        PROPERTIES LABELS "metal;physics;deformable;contact;mesh_refinement" TIMEOUT 60)
    add_test(NAME deformable_mesh.refined_drop_1280
        COMMAND $<TARGET_FILE:numi-solver-deformable-mesh> --mesh-refinement 2)
    set_tests_properties(deformable_mesh.refined_drop_1280
        PROPERTIES LABELS "metal;physics;deformable;contact;mesh_refinement" TIMEOUT 180)
    add_test(NAME deformable_mesh.refined_drop_10240
        COMMAND $<TARGET_FILE:numi-solver-deformable-mesh> --mesh-refinement 3)
    set_tests_properties(deformable_mesh.refined_drop_10240
        PROPERTIES LABELS "metal;physics;deformable;contact;mesh_refinement" TIMEOUT 300)
    add_test(
        NAME temporal_cone.math
        COMMAND $<TARGET_FILE:numi-solver-math>
            --cases 4096
            --replays 2
            --iterations 16
    )
    set_tests_properties(
        temporal_cone.math
        PROPERTIES LABELS "metal;physics;math" TIMEOUT 60
    )
    add_test(
        NAME temporal_cone.islands
        COMMAND $<TARGET_FILE:numi-solver-islands>
            --islands 96
            --replays 2
    )
    set_tests_properties(
        temporal_cone.islands
        PROPERTIES LABELS "metal;physics;math;islands;streamed" TIMEOUT 60
    )
    add_test(
        NAME temporal_cone.assembly
        COMMAND $<TARGET_FILE:numi-solver-assembly>
            --islands 96
            --replays 2
    )
    set_tests_properties(
        temporal_cone.assembly
        PROPERTIES LABELS "metal;physics;assembly;streamed" TIMEOUT 60
    )
    add_test(
        NAME temporal_cone.rigid
        COMMAND $<TARGET_FILE:numi-solver-rigid>
            --islands 96
            --replays 2
    )
    add_test(
        NAME temporal_cone.braided_bag
        COMMAND $<TARGET_FILE:numi-solver-braided-bag>
            --environments 16
            --steps 480
            --replays 2
    )
    set_tests_properties(
        temporal_cone.braided_bag
        PROPERTIES
            LABELS "metal;physics;braid;rigid;contact;streamed"
            TIMEOUT 60
    )
    add_test(
        NAME temporal_cone.braided_bag_freefall
        COMMAND $<TARGET_FILE:numi-solver-braided-bag>
            --environments 1
            --steps 1
            --replays 2
    )
    set_tests_properties(
        temporal_cone.braided_bag_freefall
        PROPERTIES
            LABELS "metal;physics;braid;zero_contact;streamed"
            TIMEOUT 60
    )
    foreach(NUMI_BAG_PRESET IN ITEMS stiff-braid low-cfm high-friction)
        string(REPLACE "-" "_" NUMI_BAG_TEST_SUFFIX "${NUMI_BAG_PRESET}")
        add_test(
            NAME temporal_cone.braided_bag_${NUMI_BAG_TEST_SUFFIX}
            COMMAND $<TARGET_FILE:numi-solver-braided-bag>
                --environments 4
                --steps 480
                --replays 2
                --preset ${NUMI_BAG_PRESET}
        )
        set_tests_properties(
            temporal_cone.braided_bag_${NUMI_BAG_TEST_SUFFIX}
            PROPERTIES
                LABELS "metal;physics;braid;contact;conditioning;streamed"
                TIMEOUT 60
        )
    endforeach()
    add_test(
        NAME temporal_cone.braided_bag_refinement
        COMMAND $<TARGET_FILE:numi-solver-braided-bag>
            --environments 1
            --steps 480
            --replays 2
            --refinement
    )
    set_tests_properties(
        temporal_cone.braided_bag_refinement
        PROPERTIES
            LABELS "metal;physics;braid;contact;refinement;streamed"
            TIMEOUT 60
    )
    add_test(
        NAME temporal_cone.braided_bag_long_horizon
        COMMAND $<TARGET_FILE:numi-solver-braided-bag>
            --environments 4
            --steps 1920
            --replays 2
    )
    set_tests_properties(
        temporal_cone.braided_bag_long_horizon
        PROPERTIES
            LABELS "metal;physics;braid;contact;long_horizon;streamed"
            TIMEOUT 60
    )
    add_test(
        NAME cloth_bag.reference
        COMMAND $<TARGET_FILE:numi-solver-cloth-bag>
            --steps 30
            --substeps 24
            --iterations 32
    )
    add_test(
        NAME cloth_calibration.synthetic
        COMMAND $<TARGET_FILE:numi-solver-cloth-calibration>
            --parameters
            ${CMAKE_CURRENT_SOURCE_DIR}/calibration/synthetic/parameters.csv
            --observations
            ${CMAKE_CURRENT_SOURCE_DIR}/calibration/synthetic/qualified-observations.csv
            --material-output
            ${CMAKE_CURRENT_BINARY_DIR}/synthetic-cloth-material.txt
    )
    set_tests_properties(
        cloth_calibration.synthetic
        PROPERTIES
            LABELS "physics;cloth;calibration;math;heldout"
            PASS_REGULAR_EXPRESSION "qualified=true"
            TIMEOUT 30
    )
    add_test(
        NAME cloth_calibration.reject_rank_deficiency
        COMMAND ${CMAKE_COMMAND}
            -DEXECUTABLE=$<TARGET_FILE:numi-solver-cloth-calibration>
            -DPARAMETERS=${CMAKE_CURRENT_SOURCE_DIR}/calibration/synthetic/rank-deficient-parameters.csv
            -DOBSERVATIONS=${CMAKE_CURRENT_SOURCE_DIR}/calibration/synthetic/rank-deficient-observations.csv
            -DEXPECT=rank=1
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ExpectCalibrationFailure.cmake
    )
    set_tests_properties(
        cloth_calibration.reject_rank_deficiency
        PROPERTIES
            LABELS "physics;cloth;calibration;math;rejection"
            TIMEOUT 30
    )
    add_test(
        NAME cloth_calibration.reject_heldout_failure
        COMMAND ${CMAKE_COMMAND}
            -DEXECUTABLE=$<TARGET_FILE:numi-solver-cloth-calibration>
            -DPARAMETERS=${CMAKE_CURRENT_SOURCE_DIR}/calibration/synthetic/parameters.csv
            -DOBSERVATIONS=${CMAKE_CURRENT_SOURCE_DIR}/calibration/synthetic/heldout-failure-observations.csv
            -DEXPECT=qualified=false
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ExpectCalibrationFailure.cmake
    )
    set_tests_properties(
        cloth_calibration.reject_heldout_failure
        PROPERTIES
            LABELS "physics;cloth;calibration;heldout;rejection"
            TIMEOUT 30
    )
    add_test(
        NAME cloth_calibration.reject_trial_leakage
        COMMAND ${CMAKE_COMMAND}
            -DEXECUTABLE=$<TARGET_FILE:numi-solver-cloth-calibration>
            -DPARAMETERS=${CMAKE_CURRENT_SOURCE_DIR}/calibration/synthetic/parameters.csv
            -DOBSERVATIONS=${CMAKE_CURRENT_SOURCE_DIR}/calibration/synthetic/trial-leakage-observations.csv
            -DEXPECT=trial.IDs.may.not.cross
            -DEXPECTED_RESULT=2
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ExpectCalibrationFailure.cmake
    )
    set_tests_properties(
        cloth_calibration.reject_trial_leakage
        PROPERTIES
            LABELS "physics;cloth;calibration;heldout;leakage;rejection"
            TIMEOUT 30
    )
    add_test(
        NAME cloth_calibration.runtime_pipeline
        COMMAND ${CMAKE_COMMAND}
            -DCALIBRATOR=$<TARGET_FILE:numi-solver-cloth-calibration>
            -DCPU_EXECUTABLE=$<TARGET_FILE:numi-solver-cloth-bag>
            -DMETAL_EXECUTABLE=$<TARGET_FILE:numi-solver-cloth-metal>
            -DPARAMETERS=${CMAKE_CURRENT_SOURCE_DIR}/calibration/cloth-parameters-v1.csv
            -DWORK_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckClothCalibrationRuntime.cmake
    )
    set_tests_properties(
        cloth_calibration.runtime_pipeline
        PROPERTIES
            LABELS "metal;physics;cloth;calibration;runtime;pipeline"
            RUN_SERIAL TRUE
            TIMEOUT 120
    )
    add_test(
        NAME cloth_material.cpu_default_parity
        COMMAND ${CMAKE_COMMAND}
            -DEXECUTABLE=$<TARGET_FILE:numi-solver-cloth-bag>
            -DMATERIAL=${CMAKE_CURRENT_SOURCE_DIR}/calibration/synthetic/default-material.txt
            -DKIND=cpu
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckClothMaterialParity.cmake
    )
    set_tests_properties(
        cloth_material.cpu_default_parity
        PROPERTIES
            LABELS "physics;cloth;calibration;runtime;parity;cpu"
            TIMEOUT 60
    )
    add_test(
        NAME cloth_material.metal_default_parity
        COMMAND ${CMAKE_COMMAND}
            -DEXECUTABLE=$<TARGET_FILE:numi-solver-cloth-metal>
            -DMATERIAL=${CMAKE_CURRENT_SOURCE_DIR}/calibration/synthetic/default-material.txt
            -DKIND=metal
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckClothMaterialParity.cmake
    )
    set_tests_properties(
        cloth_material.metal_default_parity
        PROPERTIES
            LABELS "metal;physics;cloth;calibration;runtime;parity"
            RUN_SERIAL TRUE
            TIMEOUT 180
    )
    add_test(
        NAME cloth_material.reject_invalid
        COMMAND ${CMAKE_COMMAND}
            -DEXECUTABLE=$<TARGET_FILE:numi-solver-cloth-bag>
            -DMATERIAL=${CMAKE_CURRENT_SOURCE_DIR}/calibration/synthetic/invalid-material.txt
            -DEXPECT=cloth.material.artifact.is.missing
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ExpectMaterialFailure.cmake
    )
    set_tests_properties(
        cloth_material.reject_invalid
        PROPERTIES
            LABELS "physics;cloth;calibration;runtime;rejection"
            TIMEOUT 30
    )
    add_test(
        NAME cloth_grip_trajectory.cpu_metal
        COMMAND ${CMAKE_COMMAND}
            -DCPU_EXECUTABLE=$<TARGET_FILE:numi-solver-cloth-bag>
            -DMETAL_EXECUTABLE=$<TARGET_FILE:numi-solver-cloth-metal>
            -DIDENTITY_TRAJECTORY=${CMAKE_CURRENT_SOURCE_DIR}/calibration/trajectories/synthetic-identity.csv
            -DROTATION_TRAJECTORY=${CMAKE_CURRENT_SOURCE_DIR}/calibration/trajectories/synthetic-rotation.csv
            -DREGRAB_TRAJECTORY=${CMAKE_CURRENT_SOURCE_DIR}/calibration/trajectories/synthetic-regrab.csv
            -DPATCH_REGRAB_TRAJECTORY=${CMAKE_CURRENT_SOURCE_DIR}/calibration/trajectories/synthetic-patch-regrab.csv
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckGripTrajectory.cmake
    )
    set_tests_properties(
        cloth_grip_trajectory.cpu_metal
        PROPERTIES
            LABELS "metal;physics;cloth;grip;trajectory;6dof;runtime;replay"
            RUN_SERIAL TRUE
            TIMEOUT 240
    )
    add_test(
        NAME cloth_grip_trajectory.reject_reactivation
        COMMAND ${CMAKE_COMMAND}
            -DEXECUTABLE=$<TARGET_FILE:numi-solver-cloth-bag>
            -DTRAJECTORY=${CMAKE_CURRENT_SOURCE_DIR}/calibration/trajectories/invalid-reactivation.csv
            -DEXPECT=cannot.reactivate.after.release
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ExpectGripTrajectoryFailure.cmake
    )
    set_tests_properties(
        cloth_grip_trajectory.reject_reactivation
        PROPERTIES
            LABELS "physics;cloth;grip;trajectory;6dof;rejection"
            TIMEOUT 30
    )
    add_test(
        NAME cloth_grip_trajectory.reject_distant_regrab
        COMMAND ${CMAKE_COMMAND}
            -DEXECUTABLE=$<TARGET_FILE:numi-solver-cloth-bag>
            -DTRAJECTORY=${CMAKE_CURRENT_SOURCE_DIR}/calibration/trajectories/invalid-distant-regrab.csv
            -DEXPECT=reattachment.exceeds.capture.radius
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ExpectGripTrajectoryFailure.cmake
    )
    set_tests_properties(
        cloth_grip_trajectory.reject_distant_regrab
        PROPERTIES
            LABELS "physics;cloth;grip;trajectory;6dof;rejection"
            TIMEOUT 30
    )
    add_test(
        NAME cloth_grip_trajectory.reject_distant_patch_regrab
        COMMAND ${CMAKE_COMMAND}
            -DEXECUTABLE=$<TARGET_FILE:numi-solver-cloth-bag>
            -DTRAJECTORY=${CMAKE_CURRENT_SOURCE_DIR}/calibration/trajectories/invalid-distant-patch-regrab.csv
            -DEXPECT=reattachment.exceeds.capture.radius
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ExpectGripTrajectoryFailure.cmake
    )
    set_tests_properties(
        cloth_grip_trajectory.reject_distant_patch_regrab
        PROPERTIES
            LABELS "physics;cloth;grip;trajectory;6dof;rejection"
            TIMEOUT 30
    )
    add_test(
        NAME cloth_bag.local_node_contact
        COMMAND $<TARGET_FILE:numi-solver-cloth-bag> --local-node-probe
    )
    add_test(
        NAME cloth_bag.metal_local_node_contact
        COMMAND $<TARGET_FILE:numi-solver-cloth-metal> --local-node-probe
    )
    set_tests_properties(
        cloth_bag.local_node_contact cloth_bag.metal_local_node_contact
        PROPERTIES LABELS "physics;cloth;contact;ccd;ground;local_topology" TIMEOUT 30
    )
    add_test(
        NAME cloth_bag.initial_geometry
        COMMAND $<TARGET_FILE:numi-solver-cloth-metal> --initial-state-probe
    )
    set_tests_properties(
        cloth_bag.initial_geometry
        PROPERTIES LABELS "physics;cloth;fruit;geometry" TIMEOUT 15
    )
    add_test(
        NAME cloth_bag.metal_finite_bench_response
        COMMAND $<TARGET_FILE:numi-solver-cloth-metal> --finite-bench-probe
    )
    set_tests_properties(
        cloth_bag.metal_finite_bench_response
        PROPERTIES LABELS "metal;physics;cloth;fruit;finite_bench;friction" TIMEOUT 120
    )
    add_test(
        NAME cloth_bag.metal_fruit_flight
        COMMAND $<TARGET_FILE:numi-solver-cloth-metal> --fruit-flight-probe
    )
    set_tests_properties(
        cloth_bag.metal_fruit_flight
        PROPERTIES LABELS "metal;physics;cloth;fruit;gravity;ground;refinement" TIMEOUT 60
    )
    add_test(
        NAME cloth_bag.metal_internal_contact
        COMMAND $<TARGET_FILE:numi-solver-cloth-metal>
            --replays 2
            --iterations 32
            --strain-sweeps 3
    )
    set_tests_properties(
        cloth_bag.metal_internal_contact
        PROPERTIES
            LABELS "metal;physics;cloth;reference;strain;reconciliation;knot;bend;fruit;contact;ccd;yarn;self_contact;self_friction;ground;friction;rolling;aerodynamics;mouth;release;pickup"
            TIMEOUT 60
    )
    add_test(
        NAME cloth_bag.rolling_friction
        COMMAND $<TARGET_FILE:numi-solver-cloth-bag> --rolling-probe
    )
    add_test(
        NAME cloth_bag.continuous_yarn_self_contact
        COMMAND $<TARGET_FILE:numi-solver-cloth-bag> --self-ccd-probe
    )
    add_test(
        NAME cloth_bag.strain_limit
        COMMAND $<TARGET_FILE:numi-solver-cloth-bag> --strain-probe
    )
    add_test(
        NAME cloth_bag.self_friction
        COMMAND $<TARGET_FILE:numi-solver-cloth-bag> --self-friction-probe
    )
    add_test(
        NAME cloth_bag.deformable_response
        COMMAND $<TARGET_FILE:numi-solver-cloth-bag>
            --deformable-response-probe
    )
    add_test(
        NAME cloth_bag.yarn_mechanics
        COMMAND $<TARGET_FILE:numi-solver-cloth-bag>
            --yarn-mechanics-probe
    )
    add_test(
        NAME cloth_bag.aerodynamics
        COMMAND $<TARGET_FILE:numi-solver-cloth-bag>
            --aerodynamics-probe
    )
    add_test(
        NAME cloth_bag.cloth_ground_friction
        COMMAND $<TARGET_FILE:numi-solver-cloth-bag>
            --cloth-ground-friction-probe
    )
    add_test(
        NAME cloth_bag.rolling_resistance
        COMMAND $<TARGET_FILE:numi-solver-cloth-bag>
            --rolling-resistance-probe
    )
    add_test(
        NAME cloth_bag.release_aware_escape
        COMMAND $<TARGET_FILE:numi-solver-cloth-bag> --escape-classification-probe
    )
    set_tests_properties(
        cloth_bag.release_aware_escape
        PROPERTIES LABELS "physics;cloth;fruit;release;ground;classification" TIMEOUT 30
    )
    add_test(
        NAME cloth_bag.mouth_release
        COMMAND $<TARGET_FILE:numi-solver-cloth-bag>
            --mouth-release-probe
    )
    set_tests_properties(
        cloth_bag.mouth_release
        PROPERTIES
            LABELS "physics;cloth;reference;contact;release;mouth;yarn"
            TIMEOUT 30
    )
    set_tests_properties(
        cloth_bag.rolling_resistance
        PROPERTIES
            LABELS "physics;cloth;reference;contact;friction;rolling;fruit"
            TIMEOUT 30
    )
    set_tests_properties(
        cloth_bag.cloth_ground_friction
        PROPERTIES
            LABELS "physics;cloth;reference;contact;friction;ground;yarn"
            TIMEOUT 30
    )
    set_tests_properties(
        cloth_bag.aerodynamics
        PROPERTIES
            LABELS "physics;cloth;reference;aerodynamics;yarn;fruit"
            TIMEOUT 30
    )
    set_tests_properties(
        cloth_bag.yarn_mechanics
        PROPERTIES
            LABELS "physics;cloth;reference;contact;ccd;yarn;knot"
            TIMEOUT 30
    )
    set_tests_properties(
        cloth_bag.deformable_response
        PROPERTIES
            LABELS "physics;cloth;reference;contact;deformable;coupled"
            TIMEOUT 30
    )
    set_tests_properties(
        cloth_bag.self_friction
        PROPERTIES
            LABELS "physics;cloth;reference;friction;self_contact"
            TIMEOUT 30
    )
    set_tests_properties(
        cloth_bag.strain_limit
        PROPERTIES
            LABELS "physics;cloth;reference;strain;barrier"
            TIMEOUT 30
    )
    set_tests_properties(
        cloth_bag.continuous_yarn_self_contact
        PROPERTIES
            LABELS "physics;cloth;reference;contact;ccd;self_contact"
            TIMEOUT 30
    )
    set_tests_properties(
        cloth_bag.rolling_friction
        PROPERTIES
            LABELS "physics;cloth;reference;friction;rolling"
            TIMEOUT 30
    )
    set_tests_properties(
        cloth_bag.reference
        PROPERTIES
            LABELS "physics;cloth;reference;contact;determinism"
            RUN_SERIAL TRUE
            TIMEOUT 180
    )
    add_test(
        NAME cloth_bag.airborne_spin
        COMMAND $<TARGET_FILE:numi-solver-cloth-bag>
            --scenario spin
            --steps 60
            --substeps 48
            --iterations 32
            --dump-frames ${CMAKE_CURRENT_BINARY_DIR}/cloth-spin-test
    )
    set_tests_properties(
        cloth_bag.airborne_spin
        PROPERTIES
            LABELS "physics;cloth;reference;contact;determinism;spin"
            RUN_SERIAL TRUE
            TIMEOUT 300
    )
    add_test(
        NAME cloth_bag.pickup
        COMMAND $<TARGET_FILE:numi-solver-cloth-bag>
            --scenario pickup
            --steps 240
            --substeps 48
            --iterations 32
            --dump-frames ${CMAKE_CURRENT_BINARY_DIR}/cloth-pickup-test
            --dump-every 10
    )
    set_tests_properties(
        cloth_bag.pickup
        PROPERTIES
            LABELS "physics;cloth;reference;contact;determinism;pickup"
            RUN_SERIAL TRUE
            TIMEOUT 900
    )
    set_tests_properties(
        temporal_cone.rigid
        PROPERTIES LABELS "metal;physics;rigid;assembly;streamed" TIMEOUT 60
    )
    add_test(
        NAME temporal_cone.articulated
        COMMAND $<TARGET_FILE:numi-solver-articulated>
            --islands 96
            --replays 2
    )
    set_tests_properties(
        temporal_cone.articulated
        PROPERTIES
            LABELS "metal;physics;articulated;assembly;streamed;inverse_aba"
            TIMEOUT 60
    )
    add_test(
        NAME temporal_cone.articulated_capacity
        COMMAND $<TARGET_FILE:numi-solver-articulated-capacity>
            --islands 16
            --replays 2
    )
    set_tests_properties(
        temporal_cone.articulated_capacity
        PROPERTIES
            LABELS "metal;physics;articulated;capacity;assembly;streamed;inverse_aba"
            TIMEOUT 60
    )
    add_test(
        NAME temporal_cone.articulated_conditioning
        COMMAND $<TARGET_FILE:numi-solver-articulated-conditioning>
            --islands 8
            --replays 2
    )
    set_tests_properties(
        temporal_cone.articulated_conditioning
        PROPERTIES
            LABELS "metal;physics;articulated;conditioning;rollback;inverse_aba"
            TIMEOUT 60
    )
    add_test(
        NAME temporal_cone.articulated_zero_armature
        COMMAND $<TARGET_FILE:numi-solver-articulated-zero-armature>
            --islands 8
            --replays 2
    )
    set_tests_properties(
        temporal_cone.articulated_zero_armature
        PROPERTIES
            LABELS "metal;physics;articulated;conditioning;rollback;inverse_aba"
            TIMEOUT 60
    )
endif()

install(FILES "${NUMI_SOLVER_METALLIB}" DESTINATION lib/numi-solver)
install(FILES "${NUMI_DEFORMABLE_TET_METALLIB}" DESTINATION lib/numi-solver)
install(FILES "${NUMI_DEFORMABLE_MESH_METALLIB}" DESTINATION lib/numi-solver)
install(DIRECTORY include/ DESTINATION include)
install(
    TARGETS
        numi-solver-math
        numi-solver-deformable-tet
        numi-solver-deformable-mesh
        numi-solver-islands
        numi-solver-assembly
        numi-solver-rigid
        numi-solver-braided-bag
        numi-solver-cloth-bag
        numi-solver-cloth-calibration
        numi-solver-articulated
        numi-solver-articulated-capacity
        numi-solver-articulated-conditioning
        numi-solver-articulated-zero-armature
    RUNTIME DESTINATION bin
)
