#pragma once

#include "core/VulkanContext.hpp"
#include <cstddef>
#include <vector>

namespace cybercab {

class VulkanBuffer {
public:
    VulkanBuffer() = default;
    ~VulkanBuffer();

    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;
    VulkanBuffer(VulkanBuffer&& other) noexcept;
    VulkanBuffer& operator=(VulkanBuffer&& other) noexcept;

    void create(const VulkanContext& ctx, VkDeviceSize size,
                VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps);
    void createPreferred(const VulkanContext& ctx, VkDeviceSize size,
                         VkBufferUsageFlags usage,
                         VkMemoryPropertyFlags preferredMemProps,
                         VkMemoryPropertyFlags fallbackMemProps);
    void destroy();

    void upload(const void* data, VkDeviceSize bytes, VkDeviceSize offset = 0);
    void download(void* dst, VkDeviceSize bytes, VkDeviceSize offset = 0) const;
    void invalidateCache() const;
    void flushCache() const;

    template <typename T>
    void uploadVector(const std::vector<T>& vec) {
        if (!vec.empty()) upload(vec.data(), vec.size() * sizeof(T), 0);
    }

    template <typename T>
    void downloadVector(std::vector<T>& vec) const {
        if (!vec.empty()) download(vec.data(), vec.size() * sizeof(T), 0);
    }

    void* map();
    void unmap();
    void* mappedData() const { return mapped_; }
    bool isHostCached() const { return (memProps_ & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0; }

    VkBuffer handle() const { return buffer_; }
    VkDeviceSize size() const { return size_; }
    VkDescriptorBufferInfo descriptorInfo(VkDeviceSize offset = 0, VkDeviceSize range = VK_WHOLE_SIZE) const {
        return {buffer_, offset, range};
    }

private:
    const VulkanContext* ctx_ = nullptr;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    VkMemoryPropertyFlags memProps_ = 0;
    void* mapped_ = nullptr;
};

} // namespace cybercab
