#pragma once

#include "core/VulkanContext.hpp"
#include <cstdint>
#include <vector>

namespace cybercab {

class VulkanImage {
public:
    VulkanImage() = default;
    ~VulkanImage();

    VulkanImage(const VulkanImage&) = delete;
    VulkanImage& operator=(const VulkanImage&) = delete;
    VulkanImage(VulkanImage&& other) noexcept;
    VulkanImage& operator=(VulkanImage&& other) noexcept;

    void create(const VulkanContext& ctx, uint32_t width, uint32_t height,
                VkFormat format, VkImageUsageFlags usage,
                VkImageAspectFlags aspectFlags, bool createSampler = false,
                VkFilter filter = VK_FILTER_LINEAR);
    void destroy();

    static void cmdTransitionLayout(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect,
                                    VkImageLayout oldLayout, VkImageLayout newLayout);
    void transitionLayoutImmediate(VkImageLayout newLayout);

    void uploadRGBA8(const uint8_t* pixels, uint32_t width, uint32_t height);
    std::vector<uint8_t> downloadRGBA8() const;

    VkImage handle() const { return image_; }
    VkImageView view() const { return view_; }
    VkSampler sampler() const { return sampler_; }
    VkFormat format() const { return format_; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    VkImageLayout layout() const { return currentLayout_; }
    void setLayout(VkImageLayout l) { currentLayout_ = l; }

private:
    const VulkanContext* ctx_ = nullptr;
    VkImage image_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkImageView view_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspect_ = VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageLayout currentLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
};

} // namespace cybercab
