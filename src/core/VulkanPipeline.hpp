#pragma once

#include "core/VulkanContext.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace cybercab {

struct DescriptorBindingSpec {
    uint32_t binding = 0;
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    VkShaderStageFlags stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    uint32_t count = 1;
};

class VulkanComputePipeline {
public:
    VulkanComputePipeline() = default;
    ~VulkanComputePipeline();

    VulkanComputePipeline(const VulkanComputePipeline&) = delete;
    VulkanComputePipeline& operator=(const VulkanComputePipeline&) = delete;

    void create(const VulkanContext& ctx,
                const std::string& spvPath,
                const std::vector<DescriptorBindingSpec>& bindings,
                uint32_t pushConstantBytes = 0,
                VkShaderStageFlags pushStages = VK_SHADER_STAGE_COMPUTE_BIT);
    void destroy();

    VkDescriptorSet allocateDescriptorSet() const;
    void bindAndDispatch(VkCommandBuffer cmd, VkDescriptorSet set,
                         uint32_t gx, uint32_t gy, uint32_t gz,
                         const void* pushData = nullptr, uint32_t pushBytes = 0) const;

    VkPipeline pipeline() const { return pipeline_; }
    VkPipelineLayout layout() const { return pipelineLayout_; }
    VkDescriptorSetLayout setLayout() const { return setLayout_; }

private:
    const VulkanContext* ctx_ = nullptr;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    mutable std::vector<VkDescriptorSet> allocatedSets_;
};

std::vector<uint32_t> readSpirvFile(const std::string& path);
VkShaderModule createShaderModule(VkDevice device, const std::vector<uint32_t>& code);

void cmdComputeBarrier(VkCommandBuffer cmd);

std::string resolveAssetPath(const std::string& relativePath);

} // namespace cybercab
