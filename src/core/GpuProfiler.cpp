#include "core/GpuProfiler.hpp"
#include "core/VulkanUtils.hpp"

namespace cybercab {

GpuProfiler::~GpuProfiler() {
    destroy();
}

void GpuProfiler::init(const VulkanContext& ctx, uint32_t maxQueries) {
    destroy();
    ctx_ = &ctx;
    maxQueries_ = maxQueries;
    VkQueryPoolCreateInfo qpci{};
    qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpci.queryCount = maxQueries_;
    VK_CHECK(vkCreateQueryPool(ctx_->device(), &qpci, nullptr, &queryPool_));
}

void GpuProfiler::destroy() {
    if (ctx_ && ctx_->device() != VK_NULL_HANDLE && queryPool_ != VK_NULL_HANDLE) {
        vkDestroyQueryPool(ctx_->device(), queryPool_, nullptr);
        queryPool_ = VK_NULL_HANDLE;
    }
    ctx_ = nullptr;
    spans_.clear();
    nextQuery_ = 0;
}

void GpuProfiler::reset(VkCommandBuffer cmd) {
    if (queryPool_ == VK_NULL_HANDLE) return;
    vkCmdResetQueryPool(cmd, queryPool_, 0, maxQueries_);
    nextQuery_ = 0;
    spans_.clear();
}

void GpuProfiler::beginStage(VkCommandBuffer cmd, const std::string& name, VkPipelineStageFlagBits stage) {
    if (queryPool_ == VK_NULL_HANDLE || nextQuery_ + 2 > maxQueries_) return;
    uint32_t idx = nextQuery_++;
    vkCmdWriteTimestamp(cmd, stage, queryPool_, idx);
    spans_.push_back({name, idx, 0, false});
}

void GpuProfiler::endStage(VkCommandBuffer cmd, const std::string& name, VkPipelineStageFlagBits stage) {
    if (queryPool_ == VK_NULL_HANDLE || nextQuery_ >= maxQueries_) return;
    for (auto it = spans_.rbegin(); it != spans_.rend(); ++it) {
        if (!it->closed && it->name == name) {
            uint32_t idx = nextQuery_++;
            vkCmdWriteTimestamp(cmd, stage, queryPool_, idx);
            it->endIdx = idx;
            it->closed = true;
            return;
        }
    }
}

std::map<std::string, double> GpuProfiler::resolveTimingsMs() {
    std::map<std::string, double> out;
    if (queryPool_ == VK_NULL_HANDLE || nextQuery_ == 0) return out;
    std::vector<uint64_t> raw(nextQuery_, 0);
    VkResult res = vkGetQueryPoolResults(
        ctx_->device(), queryPool_, 0, nextQuery_,
        raw.size() * sizeof(uint64_t), raw.data(), sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (res != VK_SUCCESS) return out;

    double nsPerTick = static_cast<double>(ctx_->timestampPeriodNs());
    for (const auto& s : spans_) {
        if (s.closed && raw[s.endIdx] >= raw[s.startIdx]) {
            double ms = static_cast<double>(raw[s.endIdx] - raw[s.startIdx]) * nsPerTick * 1e-6;
            out[s.name] += ms;
        }
    }
    return out;
}

} // namespace cybercab
