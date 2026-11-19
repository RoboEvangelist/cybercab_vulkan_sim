#pragma once

#include "core/VulkanContext.hpp"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace cybercab {

class GpuProfiler {
public:
    GpuProfiler() = default;
    ~GpuProfiler();

    void init(const VulkanContext& ctx, uint32_t maxQueries = 64);
    void destroy();

    void reset(VkCommandBuffer cmd);
    void beginStage(VkCommandBuffer cmd, const std::string& name, VkPipelineStageFlagBits stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
    void endStage(VkCommandBuffer cmd, const std::string& name, VkPipelineStageFlagBits stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

    // Call after queue completion / fence wait to read GPU durations in milliseconds.
    std::map<std::string, double> resolveTimingsMs();

private:
    const VulkanContext* ctx_ = nullptr;
    VkQueryPool queryPool_ = VK_NULL_HANDLE;
    uint32_t maxQueries_ = 0;
    uint32_t nextQuery_ = 0;

    struct StageSpan {
        std::string name;
        uint32_t startIdx = 0;
        uint32_t endIdx = 0;
        bool closed = false;
    };
    std::vector<StageSpan> spans_;
};

} // namespace cybercab
