#include "core/VulkanBuffer.hpp"
#include "core/VulkanUtils.hpp"
#include <cstring>

namespace cybercab {

VulkanBuffer::~VulkanBuffer() {
    destroy();
}

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept {
    *this = std::move(other);
}

VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& other) noexcept {
    if (this != &other) {
        destroy();
        ctx_ = other.ctx_;
        buffer_ = other.buffer_;
        memory_ = other.memory_;
        size_ = other.size_;
        memProps_ = other.memProps_;
        mapped_ = other.mapped_;
        other.ctx_ = nullptr;
        other.buffer_ = VK_NULL_HANDLE;
        other.memory_ = VK_NULL_HANDLE;
        other.size_ = 0;
        other.mapped_ = nullptr;
    }
    return *this;
}

void VulkanBuffer::create(const VulkanContext& ctx, VkDeviceSize size,
                          VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps) {
    createPreferred(ctx, size, usage, memProps, memProps);
}

void VulkanBuffer::createPreferred(const VulkanContext& ctx, VkDeviceSize size,
                                   VkBufferUsageFlags usage,
                                   VkMemoryPropertyFlags preferredMemProps,
                                   VkMemoryPropertyFlags fallbackMemProps) {
    destroy();
    ctx_ = &ctx;
    size_ = std::max<VkDeviceSize>(size, 16);

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size_;
    bci.usage = usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(ctx_->device(), &bci, nullptr, &buffer_));

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx_->device(), buffer_, &req);

    uint32_t memIdx = 0;
    if (ctx_->tryFindMemoryType(req.memoryTypeBits, preferredMemProps, memIdx)) {
        memProps_ = preferredMemProps;
    } else {
        memIdx = ctx_->findMemoryType(req.memoryTypeBits, fallbackMemProps);
        memProps_ = fallbackMemProps;
    }

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = memIdx;
    VK_CHECK(vkAllocateMemory(ctx_->device(), &mai, nullptr, &memory_));
    VK_CHECK(vkBindBufferMemory(ctx_->device(), buffer_, memory_, 0));
}

void VulkanBuffer::destroy() {
    if (ctx_ && ctx_->device() != VK_NULL_HANDLE) {
        if (mapped_) {
            vkUnmapMemory(ctx_->device(), memory_);
            mapped_ = nullptr;
        }
        if (buffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(ctx_->device(), buffer_, nullptr);
            buffer_ = VK_NULL_HANDLE;
        }
        if (memory_ != VK_NULL_HANDLE) {
            vkFreeMemory(ctx_->device(), memory_, nullptr);
            memory_ = VK_NULL_HANDLE;
        }
    }
    ctx_ = nullptr;
    size_ = 0;
}

void* VulkanBuffer::map() {
    if (!mapped_) {
        VK_CHECK(vkMapMemory(ctx_->device(), memory_, 0, size_, 0, &mapped_));
    }
    return mapped_;
}

void VulkanBuffer::unmap() {
    if (mapped_) {
        vkUnmapMemory(ctx_->device(), memory_);
        mapped_ = nullptr;
    }
}

void VulkanBuffer::flushCache() const {
    if ((memProps_ & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 &&
        (memProps_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0 && ctx_) {
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, memory_, 0, VK_WHOLE_SIZE};
        vkFlushMappedMemoryRanges(ctx_->device(), 1, &range);
    }
}

void VulkanBuffer::invalidateCache() const {
    if ((memProps_ & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 &&
        (memProps_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0 && ctx_) {
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, memory_, 0, VK_WHOLE_SIZE};
        vkInvalidateMappedMemoryRanges(ctx_->device(), 1, &range);
    }
}

void VulkanBuffer::upload(const void* data, VkDeviceSize bytes, VkDeviceSize offset) {
    if (bytes == 0) return;
    if ((memProps_ & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        void* p = map();
        std::memcpy(static_cast<char*>(p) + offset, data, bytes);
        if ((memProps_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
            const VkDeviceSize atom = ctx_->deviceProperties().limits.nonCoherentAtomSize;
            const VkDeviceSize alignedOff = (atom > 0) ? (offset - (offset % atom)) : 0;
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, memory_, alignedOff, VK_WHOLE_SIZE};
            vkFlushMappedMemoryRanges(ctx_->device(), 1, &range);
        }
        return;
    }

    VulkanBuffer staging;
    staging.create(*ctx_, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    staging.upload(data, bytes, 0);

    VkCommandBuffer cmd = ctx_->beginSingleTimeCommands();
    VkBufferCopy copy{0, offset, bytes};
    vkCmdCopyBuffer(cmd, staging.handle(), buffer_, 1, &copy);
    ctx_->endSingleTimeCommands(cmd);
}

void VulkanBuffer::download(void* dst, VkDeviceSize bytes, VkDeviceSize offset) const {
    if (bytes == 0) return;
    if ((memProps_ & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        void* p = nullptr;
        if (mapped_) {
            p = mapped_;
        } else {
            VK_CHECK(vkMapMemory(ctx_->device(), memory_, 0, size_, 0, &p));
        }
        if ((memProps_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
            const VkDeviceSize atom = ctx_->deviceProperties().limits.nonCoherentAtomSize;
            const VkDeviceSize alignedOff = (atom > 0) ? (offset - (offset % atom)) : 0;
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, memory_, alignedOff, VK_WHOLE_SIZE};
            vkInvalidateMappedMemoryRanges(ctx_->device(), 1, &range);
        }
        std::memcpy(dst, static_cast<const char*>(p) + offset, bytes);
        if (!mapped_) {
            vkUnmapMemory(ctx_->device(), memory_);
        }
        return;
    }

    VulkanBuffer staging;
    staging.create(*ctx_, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkCommandBuffer cmd = ctx_->beginSingleTimeCommands();
    VkBufferCopy copy{offset, 0, bytes};
    vkCmdCopyBuffer(cmd, buffer_, staging.handle(), 1, &copy);
    ctx_->endSingleTimeCommands(cmd);
    staging.download(dst, bytes, 0);
}

} // namespace cybercab
