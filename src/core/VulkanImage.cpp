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
