#pragma once

#include <vulkan/vulkan.h>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

struct SDL_Window;

namespace cybercab {

class VulkanContext {
public:
    VulkanContext() = default;
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    // If window == nullptr, initializes in pure headless offscreen + compute mode.
    void init(bool enableValidation, SDL_Window* window = nullptr);
    void cleanup();

    bool tryFindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, uint32_t& outIndex) const;
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
    VkFormat findDepthFormat() const;

    VkCommandBuffer beginSingleTimeCommands() const;
    void endSingleTimeCommands(VkCommandBuffer cmd) const;

    VkInstance instance() const { return instance_; }
    VkPhysicalDevice physicalDevice() const { return physicalDevice_; }
    VkDevice device() const { return device_; }
    VkSurfaceKHR surface() const { return surface_; }
    VkQueue graphicsQueue() const { return graphicsQueue_; }
    uint32_t graphicsQueueFamily() const { return graphicsQueueFamily_; }
    VkCommandPool commandPool() const { return commandPool_; }
    VkDescriptorPool descriptorPool() const { return descriptorPool_; }
    const VkPhysicalDeviceProperties& deviceProperties() const { return deviceProps_; }
    std::string deviceName() const { return deviceProps_.deviceName; }
    float timestampPeriodNs() const { return deviceProps_.limits.timestampPeriod; }
    bool validationEnabled() const { return validationEnabled_; }
    uint32_t validationIssueCount() const { return validationIssueCount_.load(std::memory_order_relaxed); }

private:
    friend VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT,
        const VkDebugUtilsMessengerCallbackDataEXT*, void*);
    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    uint32_t graphicsQueueFamily_ = 0;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties deviceProps_{};
    bool validationEnabled_ = false;
    bool initialized_ = false;
    std::atomic<uint32_t> validationIssueCount_{0};

    void createInstance(bool enableValidation, SDL_Window* window);
    void pickPhysicalDevice(bool requireSurface);
    void createLogicalDevice(bool requireSwapchain);
    void createCommandAndDescriptorPools();
};

} // namespace cybercab
