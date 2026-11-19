#include "core/VulkanImage.hpp"
#include "core/VulkanBuffer.hpp"
#include "core/VulkanUtils.hpp"

namespace cybercab {

VulkanImage::~VulkanImage() {
    destroy();
}

VulkanImage::VulkanImage(VulkanImage&& other) noexcept {
    *this = std::move(other);
}

VulkanImage& VulkanImage::operator=(VulkanImage&& other) noexcept {
    if (this != &other) {
        destroy();
        ctx_ = other.ctx_;
        image_ = other.image_;
        memory_ = other.memory_;
        view_ = other.view_;
        sampler_ = other.sampler_;
        format_ = other.format_;
        aspect_ = other.aspect_;
        currentLayout_ = other.currentLayout_;
        width_ = other.width_;
        height_ = other.height_;
        other.ctx_ = nullptr;
        other.image_ = VK_NULL_HANDLE;
        other.memory_ = VK_NULL_HANDLE;
        other.view_ = VK_NULL_HANDLE;
        other.sampler_ = VK_NULL_HANDLE;
    }
    return *this;
}

void VulkanImage::create(const VulkanContext& ctx, uint32_t width, uint32_t height,
                         VkFormat format, VkImageUsageFlags usage,
                         VkImageAspectFlags aspectFlags, bool createSampler,
                         VkFilter filter) {
    destroy();
    ctx_ = &ctx;
    width_ = width;
    height_ = height;
    format_ = format;
    aspect_ = aspectFlags;
    currentLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.extent = {width, height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.format = format;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ici.usage = usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateImage(ctx_->device(), &ici, nullptr, &image_));

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(ctx_->device(), image_, &req);

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = ctx_->findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(ctx_->device(), &mai, nullptr, &memory_));
    VK_CHECK(vkBindImageMemory(ctx_->device(), image_, memory_, 0));

    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = image_;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    vci.subresourceRange = {aspectFlags, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(ctx_->device(), &vci, nullptr, &view_));

    if (createSampler) {
        VkSamplerCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter = filter;
        sci.minFilter = filter;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        VK_CHECK(vkCreateSampler(ctx_->device(), &sci, nullptr, &sampler_));
    }
}

void VulkanImage::destroy() {
    if (ctx_ && ctx_->device() != VK_NULL_HANDLE) {
        if (sampler_ != VK_NULL_HANDLE) {
            vkDestroySampler(ctx_->device(), sampler_, nullptr);
            sampler_ = VK_NULL_HANDLE;
        }
        if (view_ != VK_NULL_HANDLE) {
            vkDestroyImageView(ctx_->device(), view_, nullptr);
            view_ = VK_NULL_HANDLE;
        }
        if (image_ != VK_NULL_HANDLE) {
            vkDestroyImage(ctx_->device(), image_, nullptr);
            image_ = VK_NULL_HANDLE;
        }
        if (memory_ != VK_NULL_HANDLE) {
            vkFreeMemory(ctx_->device(), memory_, nullptr);
            memory_ = VK_NULL_HANDLE;
        }
    }
    ctx_ = nullptr;
    width_ = 0;
    height_ = 0;
    currentLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
}

void VulkanImage::cmdTransitionLayout(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect,
                                      VkImageLayout oldLayout, VkImageLayout newLayout) {
    if (oldLayout == newLayout) return;
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {aspect, 0, 1, 0, 1};

    VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
        barrier.srcAccessMask = 0;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_GENERAL) {
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }

    if (newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
        dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    } else if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    } else if (newLayout == VK_IMAGE_LAYOUT_GENERAL) {
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    } else if (newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    } else if (newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void VulkanImage::transitionLayoutImmediate(VkImageLayout newLayout) {
    VkCommandBuffer cmd = ctx_->beginSingleTimeCommands();
    cmdTransitionLayout(cmd, image_, aspect_, currentLayout_, newLayout);
    ctx_->endSingleTimeCommands(cmd);
    currentLayout_ = newLayout;
}

void VulkanImage::uploadRGBA8(const uint8_t* pixels, uint32_t width, uint32_t height) {
    VkDeviceSize size = static_cast<VkDeviceSize>(width) * height * 4;
    VulkanBuffer staging;
    staging.create(*ctx_, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    staging.upload(pixels, size);

    VkCommandBuffer cmd = ctx_->beginSingleTimeCommands();
    cmdTransitionLayout(cmd, image_, aspect_, currentLayout_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy region{};
    region.imageSubresource = {aspect_, 0, 0, 1};
    region.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(cmd, staging.handle(), image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    cmdTransitionLayout(cmd, image_, aspect_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    ctx_->endSingleTimeCommands(cmd);
    currentLayout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

std::vector<uint8_t> VulkanImage::downloadRGBA8() const {
    VkDeviceSize size = static_cast<VkDeviceSize>(width_) * height_ * 4;
    VulkanBuffer staging;
    staging.create(*ctx_, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkCommandBuffer cmd = ctx_->beginSingleTimeCommands();
    cmdTransitionLayout(cmd, image_, aspect_, currentLayout_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkBufferImageCopy region{};
    region.imageSubresource = {aspect_, 0, 0, 1};
    region.imageExtent = {width_, height_, 1};
    vkCmdCopyImageToBuffer(cmd, image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging.handle(), 1, &region);
    cmdTransitionLayout(cmd, image_, aspect_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, currentLayout_);
    ctx_->endSingleTimeCommands(cmd);

    std::vector<uint8_t> out(size);
    staging.download(out.data(), size);
    return out;
}

} // namespace cybercab
