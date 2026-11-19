#pragma once

#include "config/SimConfig.hpp"
#include "core/VulkanBuffer.hpp"
#include "core/VulkanContext.hpp"
#include "core/VulkanImage.hpp"
#include "perception/PerceptionPipeline.hpp"
#include "perception/PerceptionTypes.hpp"
#include "sensors/CameraRigRenderer.hpp"
#include "world/WorldEngine.hpp"
#include <SDL2/SDL.h>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace cybercab {

enum class ViewMode : uint8_t {
    Chase3D = 0,
    FsdVoxel = 1,
    BevTactical = 2
};

enum class CamOverlayMode : uint8_t {
    Neural = 0,
    Rgb = 1,
    Calib = 2,
    SoilHeatmap = 3,
    VoxelReproj = 4
};

struct ViewerToggleState {
    ViewMode viewMode = ViewMode::Chase3D;
    CamOverlayMode camOverlay = CamOverlayMode::Neural;
    int selectedCam = 0;
    bool showVoxels = true;
    bool showShadows = true;
    bool showFlow = true;
    bool showWorldGhosts = true;
    bool showRss = true;
    bool showColliders = true;
    bool showFrustums = false;
    bool showPlanner = true;
};

enum class UiAction : uint8_t {
    None = 0,
    SetCamOverlay,
    SelectCam,
    SetViewMode,
    ToggleRes,
    ToggleDriveMode,
    ToggleAeb,
    ToggleSoilSelected,
    ToggleLayerVoxels,
    ToggleLayerShadows,
    ToggleLayerFlow,
    ToggleLayerGhosts,
    ToggleLayerRss,
    ToggleLayerColliders,
    ToggleLayerFrustums,
    ToggleLayerPlanner,
    ScenarioJaywalk,
    ScenarioOccluded,
    ScenarioHardBrake,
    ScenarioSoilFront,
    ToggleForcedRed,
    ScenarioReset
};

struct UiButton {
    float x = 0.0f;
    float y = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
    UiAction action = UiAction::None;
    int param = 0;

    bool contains(float px, float py) const {
        return px >= x && px <= (x + w) && py >= y && py <= (y + h);
    }
};

struct LineVertex {
    float x, y, z;
    float r, g, b, a;
};

struct UiVertex {
    float x, y;
    float u, v;
    float r, g, b, a;
    float mode;
};

class VulkanViewer {
public:
    VulkanViewer() = default;
    ~VulkanViewer();

    void init(SimConfig& cfg, WorldEngine& world, CameraRigRenderer& rig,
              PerceptionPipeline& pipeline, VulkanContext& ctx);
    void destroy();

    // Processes SDL events; returns false when user closes window / presses Esc.
    bool pollEvents(SimConfig& cfg, WorldEngine& world, CameraRigRenderer& rig,
                    PerceptionPipeline& pipeline, ManualInput& manual, ViewerToggleState& toggles,
                    std::string& toastMsg);

