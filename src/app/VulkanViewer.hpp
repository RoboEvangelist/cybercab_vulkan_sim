#pragma once

#include "app/V15HudOverlay.hpp"
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
  BevTactical = 2,
  RealityView = 3
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

    void renderFrame(const SimConfig& cfg, const WorldEngine& world, const CameraRigRenderer& rig,
                     const PerceptionFrameOutput& perc, const PlannerCommand& cmd,
                     const ViewerToggleState& toggles, const std::string& toastMsg);

    SDL_Window* window() const { return window_; }
    void refreshCameraDescriptors(const CameraRigRenderer& rig);

private:
    SDL_Window* window_ = nullptr;
    const VulkanContext* ctx_ = nullptr;

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat swapFormat_ = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D swapExtent_{1600, 920};
    std::vector<VkImage> swapImages_;
    std::vector<VkImageView> swapViews_;
    std::vector<VkFramebuffer> swapFramebuffers_;
    VulkanImage depthImage_;

    VkRenderPass mainPass_ = VK_NULL_HANDLE;
    VkPipelineLayout sceneLayout_ = VK_NULL_HANDLE;
    VkPipeline scenePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout linesLayout_ = VK_NULL_HANDLE;
    VkPipeline linesPipeline_ = VK_NULL_HANDLE;

    VkDescriptorSetLayout uiSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout uiLayout_ = VK_NULL_HANDLE;
    VkPipeline uiPipeline_ = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, v15_hud::kTotalUiSetCount> uiSets_{};

    VulkanImage fontAtlas_;
    VulkanImage realityImg_;

    // Pre-allocated dynamic geometry & UI vertex/index buffers (zero hot-path allocations)
    VulkanBuffer dynWorldVB_, dynWorldIB_;
    VulkanBuffer voxelVB_, voxelIB_;
    VulkanBuffer lineVB_;
    VulkanBuffer uiVB_, uiIB_;

    std::vector<SceneVertex> dynVertsScratch_;
    std::vector<uint32_t> dynIdxScratch_;
    std::vector<SceneVertex> voxelVertsScratch_;
    std::vector<uint32_t> voxelIdxScratch_;
    std::vector<LineVertex> lineVertsScratch_;
    std::vector<UiVertex> uiVertsScratch_;
    std::vector<uint32_t> uiIdxScratch_;

    struct UiDrawBatch {
        uint32_t indexOffset = 0;
        uint32_t indexCount = 0;
        int camSetIdx = 0;
    };
    std::vector<UiDrawBatch> uiBatchesScratch_;
    std::vector<UiButton> uiButtonsScratch_;
    struct PanelBounds { float x, y, w, h; };
    std::vector<PanelBounds> uiPanelsScratch_;

    VkSemaphore imageAvailSem_ = VK_NULL_HANDLE;
    VkSemaphore renderDoneSem_ = VK_NULL_HANDLE;
    VkFence inFlightFence_ = VK_NULL_HANDLE;
    VkCommandBuffer frameCmd_ = VK_NULL_HANDLE;

    // Orbit camera state
    float orbitYaw_ = 0.0f;
    float orbitPitch_ = 0.34f;
    float orbitDist_ = 13.5f;
    bool dragging_ = false;
    int lastMouseX_ = 0;
    int lastMouseY_ = 0;
    bool framebufferResized_ = false;

    void applyUiAction(UiAction action, int param, SimConfig& cfg, WorldEngine& world,
                       CameraRigRenderer& rig, PerceptionPipeline& pipeline,
                       ViewerToggleState& toggles, std::string& toastMsg);

    void createSwapchain();
    void cleanupSwapchain();
    void recreateSwapchain();
    void createMainRenderPass();
    void createPipelines();
    void createSyncObjects();

    void buildVoxelAndGhostGeometry(const WorldEngine& world, const PerceptionFrameOutput& perc,
                                    const PlannerCommand& cmd, const ViewerToggleState& toggles);
    void build3DLines(const SimConfig& cfg, const WorldEngine& world, const PerceptionFrameOutput& perc,
                      const PlannerCommand& cmd, const ViewerToggleState& toggles);
    void buildUiOverlay(const SimConfig& cfg, const WorldEngine& world, const CameraRigRenderer& rig,
                        const PerceptionFrameOutput& perc, const PlannerCommand& cmd,
                        const ViewerToggleState& toggles, const std::string& toastMsg);

    // Immediate UI batch builders (write directly into uiVertsScratch_ / uiIdxScratch_)
    void uiRect(float x, float y, float w, float h, Vec4 col);
    void uiRectOutline(float x, float y, float w, float h, float t, Vec4 col);
    void uiLine(float x0, float y0, float x1, float y1, float t, Vec4 col);
    void uiText(float x, float y, float scale, Vec4 col, const std::string& text);
    void uiTexQuad(float x, float y, float w, float h, Vec4 tint);
};

} // namespace cybercab
