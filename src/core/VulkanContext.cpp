#include "core/VulkanContext.hpp"
#include "core/VulkanUtils.hpp"

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <cstring>
#include <iostream>
#include <vector>

namespace cybercab {

namespace {

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void*) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "[VulkanValidation] " << pCallbackData->pMessage << "\n";
    }
    return VK_FALSE;
}

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
            dci.pfnUserCallback = debugCallback;
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

