#include "app/VulkanViewer.hpp"
#include "app/BitmapFont.hpp"
#include "core/VulkanPipeline.hpp"
#include "core/VulkanUtils.hpp"

#include <SDL2/SDL_vulkan.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace cybercab {

namespace {

struct alignas(16) ScenePC {
    Mat4 viewProj;
    Vec4 sunDirAmbient;
    Vec4 fogColorDensity;
};

struct alignas(16) LinesPC {
    Mat4 viewProj;
};

struct alignas(16) UiPC {
    Vec2 screenSize;
};

void appendCube(std::vector<SceneVertex>& verts, std::vector<uint32_t>& idx,
                Vec3 center, Vec3 size, float yaw, Vec3 color, float alpha, float emissive) {
    const float hx = size.x * 0.5f, hy = size.y * 0.5f, hz = size.z * 0.5f;
    const float c = std::cos(yaw), s = std::sin(yaw);
    auto xf = [&](Vec3 p) -> Vec3 {
        return {center.x + p.x * c + p.z * s, center.y + p.y, center.z - p.x * s + p.z * c};
    };
    const Vec3 normals[6] = {{0,0,1},{0,0,-1},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0}};
    const Vec3 pts[6][4] = {
        {{-hx,-hy, hz},{ hx,-hy, hz},{ hx, hy, hz},{-hx, hy, hz}},
        {{ hx,-hy,-hz},{-hx,-hy,-hz},{-hx, hy,-hz},{ hx, hy,-hz}},
        {{ hx,-hy, hz},{ hx,-hy,-hz},{ hx, hy,-hz},{ hx, hy, hz}},
        {{-hx,-hy,-hz},{-hx,-hy, hz},{-hx, hy, hz},{-hx, hy,-hz}},
        {{-hx, hy, hz},{ hx, hy, hz},{ hx, hy,-hz},{-hx, hy,-hz}},
        {{-hx,-hy,-hz},{ hx,-hy,-hz},{ hx,-hy, hz},{-hx,-hy, hz}},
    };
    for (int f = 0; f < 6; ++f) {
        const uint32_t base = static_cast<uint32_t>(verts.size());
        const Vec3 n = normals[f];
        const Vec3 wn{n.x * c + n.z * s, n.y, -n.x * s + n.z * c};
        for (int i = 0; i < 4; ++i) {
            const Vec3 wp = xf(pts[f][i]);
            verts.push_back({wp.x, wp.y, wp.z, wn.x, wn.y, wn.z, color.x, color.y, color.z, alpha, emissive, 0, 0, 0});
        }
        idx.push_back(base + 0); idx.push_back(base + 1); idx.push_back(base + 2);
        idx.push_back(base + 0); idx.push_back(base + 2); idx.push_back(base + 3);
    }
}

void appendBoxLines(std::vector<LineVertex>& out, Vec3 center, Vec3 he, float yaw, Vec4 col) {
    const float c = std::cos(yaw), s = std::sin(yaw);
    auto xf = [&](float lx, float ly, float lz) -> Vec3 {
        return {center.x + lx * c + lz * s, center.y + ly, center.z - lx * s + lz * c};
    };
    Vec3 v[8] = {
        xf(-he.x, -he.y, -he.z), xf( he.x, -he.y, -he.z),
        xf( he.x,  he.y, -he.z), xf(-he.x,  he.y, -he.z),
        xf(-he.x, -he.y,  he.z), xf( he.x, -he.y,  he.z),
        xf( he.x,  he.y,  he.z), xf(-he.x,  he.y,  he.z),
    };
    const int e[12][2] = {{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};
    for (int i = 0; i < 12; ++i) {
        const Vec3& a = v[e[i][0]];
        const Vec3& b = v[e[i][1]];
        out.push_back({a.x, a.y, a.z, col.x, col.y, col.z, col.w});
        out.push_back({b.x, b.y, b.z, col.x, col.y, col.z, col.w});
    }
}

Vec3 semanticColor(SemanticClass sem) {
    switch (sem) {
        case SemanticClass::Vehicle:    return {0.220f, 0.741f, 0.973f};
        case SemanticClass::Pedestrian: return {0.957f, 0.247f, 0.369f};
        case SemanticClass::SignalSign: return {0.984f, 0.749f, 0.141f};
        case SemanticClass::Overhang:   return {0.753f, 0.518f, 0.988f};
        case SemanticClass::Curb:       return {0.392f, 0.455f, 0.545f};
        case SemanticClass::OccShadow:  return {0.706f, 0.325f, 0.035f};
        default:                        return {0.580f, 0.639f, 0.722f};
    }
}

} // namespace

VulkanViewer::~VulkanViewer() {
    destroy();
}

void VulkanViewer::init(SimConfig& cfg, WorldEngine& world, CameraRigRenderer& rig,
                        PerceptionPipeline& pipeline, VulkanContext& ctx) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        throw std::runtime_error(std::string("SDL_Init failed: ") + SDL_GetError());
    }
    window_ = SDL_CreateWindow(
        "Tesla Cybercab — 8-Camera Vulkan Deep Perception, Occupancy & SAT Physics",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        1600, 920,
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!window_) {
        throw std::runtime_error(std::string("SDL_CreateWindow failed: ") + SDL_GetError());
    }

    ctx.init(cfg.useVulkanValidation, window_);
    ctx_ = &ctx;

    rig.init(ctx, cfg, world);
    pipeline.init(ctx, rig);

    createSwapchain();
    createMainRenderPass();
    createPipelines();
    createSyncObjects();

    // Font atlas
    const auto fontPixels = BitmapFontAtlas::buildAtlasRGBA8();
    fontAtlas_.create(*ctx_, BitmapFontAtlas::kAtlasW, BitmapFontAtlas::kAtlasH, VK_FORMAT_R8G8B8A8_UNORM,
                      VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT, true, VK_FILTER_NEAREST);
    fontAtlas_.uploadRGBA8(fontPixels.data(), BitmapFontAtlas::kAtlasW, BitmapFontAtlas::kAtlasH);

    for (int c = 0; c < 8; ++c) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
                                       ctx_->descriptorPool(), 1, &uiSetLayout_};
        VK_CHECK(vkAllocateDescriptorSets(ctx_->device(), &ai, &uiSets_[c]));
    }
    refreshCameraDescriptors(rig);

    // Pre-allocate dynamic GPU vertex/index buffers
    dynWorldVB_.create(*ctx_, 65536 * sizeof(SceneVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    dynWorldIB_.create(*ctx_, 262144 * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    voxelVB_.create(*ctx_, 120000 * sizeof(SceneVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    voxelIB_.create(*ctx_, 200000 * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    lineVB_.create(*ctx_, 32768 * sizeof(LineVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    uiVB_.create(*ctx_, 65536 * sizeof(UiVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    uiIB_.create(*ctx_, 131072 * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    dynVertsScratch_.reserve(32768);
    dynIdxScratch_.reserve(65536);
    voxelVertsScratch_.reserve(100000);
    voxelIdxScratch_.reserve(160000);
    lineVertsScratch_.reserve(16384);
    uiVertsScratch_.reserve(32768);
    uiIdxScratch_.reserve(65536);
    uiBatchesScratch_.reserve(16);
    uiButtonsScratch_.reserve(64);
    uiPanelsScratch_.reserve(8);
}

void VulkanViewer::refreshCameraDescriptors(const CameraRigRenderer& rig) {
    if (!ctx_) return;
    for (int c = 0; c < 8; ++c) {
        VkDescriptorImageInfo camInfo{rig.distortedImage(c).sampler(), rig.distortedImage(c).view(),
                                      VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo fontInfo{fontAtlas_.sampler(), fontAtlas_.view(),
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet writes[2]{};
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, uiSets_[c], 0, 0, 1,
                     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &camInfo, nullptr, nullptr};
        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, uiSets_[c], 1, 0, 1,
                     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &fontInfo, nullptr, nullptr};
        vkUpdateDescriptorSets(ctx_->device(), 2, writes, 0, nullptr);
    }
}

void VulkanViewer::destroy() {
    if (!ctx_) return;
