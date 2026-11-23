#include "core/VulkanPipeline.hpp"
#include "core/VulkanUtils.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace cybercab {

std::vector<uint32_t> readSpirvFile(const std::string& path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open SPIR-V shader: " + path);
    }
    size_t fileSize = static_cast<size_t>(f.tellg());
    if (fileSize == 0 || (fileSize % 4) != 0) {
        throw std::runtime_error("Invalid SPIR-V size for: " + path);
    }
    std::vector<uint32_t> buffer(fileSize / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(fileSize));
    return buffer;
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<uint32_t>& code) {
    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = code.size() * sizeof(uint32_t);
    smci.pCode = code.data();
    VkShaderModule mod = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &smci, nullptr, &mod));
    return mod;
}

void cmdComputeBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
}

std::string resolveAssetPath(const std::string& relativePath) {
    namespace fs = std::filesystem;
    if (fs::exists(relativePath)) return relativePath;
#ifdef CYBERCAB_BUILD_DIR
    {
        fs::path p = fs::path(CYBERCAB_BUILD_DIR) / relativePath;
        if (fs::exists(p)) return p.string();
    }
#endif
#ifdef CYBERCAB_SOURCE_DIR
    {
        fs::path p = fs::path(CYBERCAB_SOURCE_DIR) / relativePath;
        if (fs::exists(p)) return p.string();
    }
#endif
    return relativePath;
}

VulkanComputePipeline::~VulkanComputePipeline() {
    destroy();
}

void VulkanComputePipeline::create(const VulkanContext& ctx,
                                   const std::string& spvPath,
                                   const std::vector<DescriptorBindingSpec>& bindings,
                                   uint32_t pushConstantBytes,
                                   VkShaderStageFlags pushStages) {
    destroy();
    ctx_ = &ctx;

    std::vector<VkDescriptorSetLayoutBinding> vkBindings;
    vkBindings.reserve(bindings.size());
    for (const auto& b : bindings) {
        VkDescriptorSetLayoutBinding lb{};
        lb.binding = b.binding;
        lb.descriptorType = b.type;
        lb.descriptorCount = b.count;
        lb.stageFlags = b.stageFlags;
        vkBindings.push_back(lb);
    }

    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = static_cast<uint32_t>(vkBindings.size());
    dslci.pBindings = vkBindings.empty() ? nullptr : vkBindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(ctx_->device(), &dslci, nullptr, &setLayout_));

    VkPushConstantRange pcr{pushStages, 0, pushConstantBytes};
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &setLayout_;
    plci.pushConstantRangeCount = (pushConstantBytes > 0) ? 1 : 0;
    plci.pPushConstantRanges = (pushConstantBytes > 0) ? &pcr : nullptr;
    VK_CHECK(vkCreatePipelineLayout(ctx_->device(), &plci, nullptr, &pipelineLayout_));

    auto spirv = readSpirvFile(resolveAssetPath(spvPath));
    VkShaderModule mod = createShaderModule(ctx_->device(), spirv);

    VkPipelineShaderStageCreateInfo ssci{};
    ssci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ssci.module = mod;
    ssci.pName = "main";

    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage = ssci;
    cpci.layout = pipelineLayout_;
    VK_CHECK(vkCreateComputePipelines(ctx_->device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline_));

    vkDestroyShaderModule(ctx_->device(), mod, nullptr);
}

void VulkanComputePipeline::destroy() {
    if (ctx_ && ctx_->device() != VK_NULL_HANDLE) {
        if (!allocatedSets_.empty() &&
            ctx_->descriptorPool() != VK_NULL_HANDLE) {
            vkFreeDescriptorSets(
                ctx_->device(), ctx_->descriptorPool(),
                static_cast<uint32_t>(allocatedSets_.size()),
                allocatedSets_.data());
        }
        allocatedSets_.clear();
        if (pipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(ctx_->device(), pipeline_, nullptr);
            pipeline_ = VK_NULL_HANDLE;
        }
        if (pipelineLayout_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(ctx_->device(), pipelineLayout_, nullptr);
            pipelineLayout_ = VK_NULL_HANDLE;
        }
        if (setLayout_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(ctx_->device(), setLayout_, nullptr);
            setLayout_ = VK_NULL_HANDLE;
        }
    }
    ctx_ = nullptr;
}

VkDescriptorSet VulkanComputePipeline::allocateDescriptorSet() const {
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = ctx_->descriptorPool();
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &setLayout_;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateDescriptorSets(ctx_->device(), &dsai, &set));
    allocatedSets_.push_back(set);
    return set;
}

void VulkanComputePipeline::bindAndDispatch(VkCommandBuffer cmd, VkDescriptorSet set,
                                            uint32_t gx, uint32_t gy, uint32_t gz,
                                            const void* pushData, uint32_t pushBytes) const {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    if (set != VK_NULL_HANDLE) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &set, 0, nullptr);
    }
    if (pushData && pushBytes > 0) {
        vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes, pushData);
    }
    vkCmdDispatch(cmd, gx, gy, gz);
}

} // namespace cybercab
