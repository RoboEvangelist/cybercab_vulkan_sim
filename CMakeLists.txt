cmake_minimum_required(VERSION 3.15)
project(CybercabVulkanSim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Vulkan REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SDL2 REQUIRED sdl2)
pkg_check_modules(YAMLCPP REQUIRED yaml-cpp)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)

find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)

# Compile all GLSL shaders to SPIR-V inside build/shaders/
file(GLOB SHADER_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.vert"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.frag"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp"
)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/shaders")

set(SPIRV_OUTPUTS)
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(FILE_NAME ${SHADER} NAME)
    set(SPV_OUT "${CMAKE_CURRENT_BINARY_DIR}/shaders/${FILE_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPV_OUT}
        COMMAND ${GLSLANG_VALIDATOR}
                -I${CMAKE_CURRENT_SOURCE_DIR}/shaders/include
                -V ${SHADER}
                -o ${SPV_OUT}
        DEPENDS ${SHADER} "${CMAKE_CURRENT_SOURCE_DIR}/shaders/include/common.glsl"
                          "${CMAKE_CURRENT_SOURCE_DIR}/shaders/include/photon.glsl"
        COMMENT "Compiling GLSL -> SPIR-V: ${FILE_NAME}"
        VERBATIM
    )
    list(APPEND SPIRV_OUTPUTS ${SPV_OUT})
endforeach()

add_custom_target(compile_shaders ALL DEPENDS ${SPIRV_OUTPUTS})

# Copy config YAML files into build/config/ so build-dir invocations resolve cleanly
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/config/cybercab_sim.yaml"
          "${CMAKE_CURRENT_SOURCE_DIR}/config/v15_policy.yaml"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/config")

set(CYBERCAB_COMMON_DEFS
    CYBERCAB_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    CYBERCAB_BUILD_DIR="${CMAKE_CURRENT_BINARY_DIR}"
)

set(CYBERCAB_WARN_FLAGS -Wall -Wextra -Wpedantic)

# Shared modular libraries
add_library(cybercab_core SHARED
    src/config/SimConfig.cpp
    src/config/V15PolicyConfig.cpp
    src/core/VulkanContext.cpp
    src/core/VulkanBuffer.cpp
    src/core/VulkanImage.cpp
    src/core/VulkanPipeline.cpp
    src/core/GpuProfiler.cpp
    src/core/TraceLogger.cpp
)
target_include_directories(cybercab_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    ${SDL2_INCLUDE_DIRS}
    ${YAMLCPP_INCLUDE_DIRS}
    ${JSONCPP_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)
target_compile_definitions(cybercab_core PUBLIC ${CYBERCAB_COMMON_DEFS})
target_compile_options(cybercab_core PRIVATE ${CYBERCAB_WARN_FLAGS})
target_link_libraries(cybercab_core PUBLIC
    Vulkan::Vulkan
    ${SDL2_LIBRARIES}
    ${YAMLCPP_LIBRARIES}
    ${JSONCPP_LIBRARIES}
)

add_library(cybercab_world SHARED
    src/world/CybercabMesh.cpp
    src/world/WorldEngine.cpp
)
target_link_libraries(cybercab_world PUBLIC cybercab_core)
target_compile_options(cybercab_world PRIVATE ${CYBERCAB_WARN_FLAGS})

add_library(cybercab_perception SHARED
    src/sensors/CameraRigRenderer.cpp
    src/perception/SoilDetector.cpp
    src/perception/BackboneRTDETR.cpp
    src/perception/ByteTrack.cpp
    src/perception/BEVFormer.cpp
    src/perception/OccupancyFlowNet.cpp
    src/perception/WorldModel.cpp
    src/perception/PerceptionPipeline.cpp
    src/perception/SignalClassifier.cpp
    src/perception/SignalFilter.cpp
    src/perception/v15/FrequencySplit.cpp
    src/perception/v15/V15Pipeline.cpp
    src/perception/v15/CalibSolver.cpp
    src/perception/v15/CameraSelfCalib.cpp
    src/perception/v15/VirtualCameraRectifier.cpp
    src/perception/v15/AudioTokenizer.cpp
    src/perception/v15/KinematicNavEncoder.cpp
    src/perception/v15/PatchTokenizer.cpp
    src/perception/v15/StaticSpatialMemory.cpp
    src/perception/v15/GaussianWorldModel.cpp
    src/perception/v15/LatentPredictor.cpp
    src/perception/sota/VoxelToGaussianInit.cpp
    src/perception/sota/DeformableAttention3D.cpp
    src/perception/sota/GaussianToVoxelSplatter.cpp
    src/perception/sota/GaussianFormer3DPipeline.cpp
)
add_dependencies(cybercab_perception compile_shaders)
target_link_libraries(cybercab_perception PUBLIC cybercab_world)
target_compile_options(cybercab_perception PRIVATE ${CYBERCAB_WARN_FLAGS})

add_library(cybercab_planning SHARED
    src/app/V15HudOverlay.cpp
    src/planning/TrajectoryPlanner.cpp
    src/planning/v15/AutoregSafety.cpp
    src/planning/v15/GaussianRolloutScorer.cpp
    src/planning/v15/AutoregressivePolicy.cpp
)
target_link_libraries(cybercab_planning PUBLIC cybercab_perception)
target_compile_options(cybercab_planning PRIVATE ${CYBERCAB_WARN_FLAGS})

# Main executable
add_executable(cybercab_sim
    src/main.cpp
    src/app/VulkanViewer.cpp
)
add_dependencies(cybercab_sim compile_shaders)
target_link_libraries(cybercab_sim PRIVATE cybercab_planning)
target_compile_options(cybercab_sim PRIVATE ${CYBERCAB_WARN_FLAGS})

# GoogleTest + CTest Suite
enable_testing()
add_executable(cybercab_tests
    tests/test_math_camera.cpp
    tests/test_physics_sat.cpp
    tests/test_planning_tracking_trace.cpp
    tests/test_vulkan_pipeline.cpp
    tests/test_system_sync.cpp
    tests/test_bytetrack.cpp
    tests/test_rtdetr.cpp
    tests/test_occupancy_net.cpp
    tests/test_fmea_suite.cpp
    tests/test_fmea_perception_safety.cpp
    tests/test_core_world_coverage.cpp
    tests/test_model_quality_and_pr.cpp
    tests/test_tracking_and_occlusion.cpp
    tests/test_world_model_and_latency.cpp
    tests/test_v15_config.cpp
    tests/test_v15_photon_frontend.cpp
    tests/test_v15_photon_noise.cpp
    tests/test_v15_freq_split.cpp
    tests/test_v15_pipeline.cpp
    tests/test_v15_calib_solver.cpp
    tests/test_v15_calib.cpp
    tests/test_v15_virtual_rectify.cpp
    tests/test_v15_audio_siren.cpp
    tests/test_v15_patch_tokens.cpp
    tests/test_v15_static_memory.cpp
    tests/test_v15_gaussian_world.cpp
    tests/test_v15_latent_predictor.cpp
    tests/test_v15_autoreg_planner.cpp
    tests/test_v15_gaussian_planner.cpp
    tests/test_v15_telemetry_and_reality_hud.cpp
    tests/test_v15_hdr_rccb_closed_loop.cpp
    tests/test_v15_params_and_actions.cpp
    tests/test_v15_autoshift_and_calib_closed_loop.cpp
    tests/test_v15_voxels_detections_and_hallucinations.cpp
    tests/test_v15_mission_collisions_and_zero_alloc.cpp
    tests/test_v15_cybercab_camera_res.cpp
    tests/test_camera_corrob_and_bounds.cpp
    tests/test_cybercab_res_full_validation.cpp
    tests/test_all_res_v14_v15_validation.cpp
    tests/test_all_res_fmea_safety.cpp
    tests/test_all_res_fmea_env_and_sync.cpp
    tests/test_av_mot_idsw_and_lane_assoc.cpp
    tests/test_av_ece_confusion_and_slicing.cpp
    tests/test_av_velocity_stopping_margin.cpp
    tests/test_av_road_structure_and_poles.cpp
    tests/test_av_confidence_calibration.cpp
    tests/test_av_vision_signal_fsm_and_assoc.cpp
    tests/test_av_open_set_obstacles_and_freespace.cpp
    tests/test_av_shadow_replay_and_hierarchy.cpp
    tests/test_v15_e2e_closed_loop.cpp
    tests/test_sota_gaussianformer3d.cpp
    tests/test_sota_gaussianformer3d_closed_loop.cpp
    tests/test_sota_gaussianformer3d_ab_metrics.cpp
    tests/test_sota_gaussianformer3d_ab_missions_nfr.cpp
    tests/test_sota_gaussianformer3d_ab_robustness.cpp
)
add_dependencies(cybercab_tests compile_shaders)
target_link_libraries(cybercab_tests PRIVATE
    cybercab_planning
    GTest::GTest
    GTest::Main
    pthread
)
target_compile_options(cybercab_tests PRIVATE ${CYBERCAB_WARN_FLAGS})

include(GoogleTest)
gtest_discover_tests(cybercab_tests)
