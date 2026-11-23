#include "core/VulkanContext.hpp"
#include "core/VulkanUtils.hpp"

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <cstring>
#include <iostream>
#include <vector>

namespace cybercab {

VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        if (auto* ctx = static_cast<VulkanContext*>(pUserData)) {
            ctx->validationIssueCount_.fetch_add(1, std::memory_order_relaxed);
        }
        std::cerr << "[VulkanValidation] " << pCallbackData->pMessage << "\n";
    }
    return VK_FALSE;
}

namespace {

bool checkValidationLayerSupport(const char* layerName) {
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    for (const auto& l : layers) {
        if (std::strcmp(l.layerName, layerName) == 0) return true;
    }
    return false;
}

} // namespace

VulkanContext::~VulkanContext() {
    cleanup();
}

void VulkanContext::init(bool enableValidation, SDL_Window* window) {
    if (initialized_) return;
    validationIssueCount_.store(0, std::memory_order_relaxed);
    createInstance(enableValidation, window);
    if (window) {
        if (!SDL_Vulkan_CreateSurface(window, instance_, &surface_)) {
            throw std::runtime_error(std::string("SDL_Vulkan_CreateSurface failed: ") + SDL_GetError());
        }
    }
    pickPhysicalDevice(window != nullptr);
    createLogicalDevice(window != nullptr);
    createCommandAndDescriptorPools();
    initialized_ = true;
}

void VulkanContext::cleanup() {
    if (!initialized_) return;
    vkDeviceWaitIdle(device_);
    if (descriptorPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        descriptorPool_ = VK_NULL_HANDLE;
    }
    if (commandPool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, commandPool_, nullptr);
        commandPool_ = VK_NULL_HANDLE;
    }
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    if (surface_ != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        surface_ = VK_NULL_HANDLE;
    }
    if (debugMessenger_ != VK_NULL_HANDLE) {
        auto fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (fn) fn(instance_, debugMessenger_, nullptr);
        debugMessenger_ = VK_NULL_HANDLE;
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
    initialized_ = false;
}

void VulkanContext::createInstance(bool enableValidation, SDL_Window* window) {
    const char* validationLayer = "VK_LAYER_KHRONOS_validation";
    const char* fallbackLayer = "VK_LAYER_LUNARG_standard_validation";
    std::vector<const char*> layers;
    if (enableValidation) {
        if (checkValidationLayerSupport(validationLayer)) {
            layers.push_back(validationLayer);
            validationEnabled_ = true;
        } else if (checkValidationLayerSupport(fallbackLayer)) {
            layers.push_back(fallbackLayer);
            validationEnabled_ = true;
        }
    }

    std::vector<const char*> extensions;
    if (window) {
        unsigned int count = 0;
        if (!SDL_Vulkan_GetInstanceExtensions(window, &count, nullptr)) {
            throw std::runtime_error("SDL_Vulkan_GetInstanceExtensions failed");
        }
        extensions.resize(count);
        if (!SDL_Vulkan_GetInstanceExtensions(window, &count, extensions.data())) {
            throw std::runtime_error("SDL_Vulkan_GetInstanceExtensions failed");
        }
    }
    if (validationEnabled_) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Tesla Cybercab Vulkan Perception Sim";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "CybercabVkEngine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &appInfo;
    ci.enabledLayerCount = static_cast<uint32_t>(layers.size());
    ci.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();
    ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    ci.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();

    VK_CHECK(vkCreateInstance(&ci, nullptr, &instance_));

    if (validationEnabled_) {
        auto fn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
        if (fn) {
            VkDebugUtilsMessengerCreateInfoEXT dci{};
            dci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            dci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            dci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            dci.pfnUserCallback = vulkanDebugCallback;
            dci.pUserData = this;
            fn(instance_, &dci, nullptr, &debugMessenger_);
        }
    }
}

void VulkanContext::pickPhysicalDevice(bool requireSurface) {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (count == 0) {
        throw std::runtime_error("No Vulkan physical devices found");
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, devices.data());

    int bestScore = -1;
    for (VkPhysicalDevice dev : devices) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(dev, &props);

        uint32_t qCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qCount, nullptr);
        std::vector<VkQueueFamilyProperties> qProps(qCount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qCount, qProps.data());

        int chosenFamily = -1;
        for (uint32_t i = 0; i < qCount; ++i) {
            bool hasGfx = (qProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
            bool hasCmp = (qProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
            if (!hasGfx || !hasCmp) continue;
            if (requireSurface) {
                VkBool32 present = VK_FALSE;
                vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface_, &present);
                if (!present) continue;
            }
            chosenFamily = static_cast<int>(i);
            break;
        }
        if (chosenFamily < 0) continue;

        int score = 10;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 1000;
        else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score += 100;

        if (score > bestScore) {
            bestScore = score;
            physicalDevice_ = dev;
            graphicsQueueFamily_ = static_cast<uint32_t>(chosenFamily);
            deviceProps_ = props;
        }
    }

    if (physicalDevice_ == VK_NULL_HANDLE) {
        throw std::runtime_error("Failed to find suitable Vulkan GPU with graphics+compute queue");
    }
}

void VulkanContext::createLogicalDevice(bool requireSwapchain) {
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = graphicsQueueFamily_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    std::vector<const char*> devExts;
    if (requireSwapchain) {
        devExts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    VkPhysicalDeviceFeatures features{};
    features.fillModeNonSolid = VK_TRUE;
    features.wideLines = VK_TRUE;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<uint32_t>(devExts.size());
    dci.ppEnabledExtensionNames = devExts.empty() ? nullptr : devExts.data();
    dci.pEnabledFeatures = &features;

    VK_CHECK(vkCreateDevice(physicalDevice_, &dci, nullptr, &device_));
    vkGetDeviceQueue(device_, graphicsQueueFamily_, 0, &graphicsQueue_);
}

void VulkanContext::createCommandAndDescriptorPools() {
    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = graphicsQueueFamily_;
    VK_CHECK(vkCreateCommandPool(device_, &cpci, nullptr, &commandPool_));

    std::vector<VkDescriptorPoolSize> sizes = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 256},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 512},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 128},
    };
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 512;
    dpci.poolSizeCount = static_cast<uint32_t>(sizes.size());
    dpci.pPoolSizes = sizes.data();
    VK_CHECK(vkCreateDescriptorPool(device_, &dpci, nullptr, &descriptorPool_));
}

bool VulkanContext::tryFindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, uint32_t& outIndex) const {
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((typeFilter & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
            outIndex = i;
            return true;
        }
    }
    return false;
}

uint32_t VulkanContext::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const {
    uint32_t idx = 0;
    if (!tryFindMemoryType(typeFilter, properties, idx)) {
        throw std::runtime_error("Failed to find suitable Vulkan memory type");
    }
    return idx;
}

VkFormat VulkanContext::findDepthFormat() const {
    for (VkFormat fmt : {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT}) {
        VkFormatProperties props{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice_, fmt, &props);
        if ((props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0) {
            return fmt;
        }
    }
    throw std::runtime_error("Failed to find supported depth format");
}

VkCommandBuffer VulkanContext::beginSingleTimeCommands() const {
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandPool = commandPool_;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(device_, &ai, &cmd));

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    return cmd;
}

void VulkanContext::endSingleTimeCommands(VkCommandBuffer cmd) const {
    VK_CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(graphicsQueue_, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(graphicsQueue_));
    vkFreeCommandBuffers(device_, commandPool_, 1, &cmd);
}

} // namespace cybercab
