#pragma once

#include <vulkan/vulkan.h>
#include <stdexcept>
#include <string>

namespace cybercab {

inline const char* vkResultToString(VkResult res) {
    switch (res) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
        case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
        default: return "VK_ERROR_UNKNOWN";
    }
}

#define VK_CHECK(expr)                                                                         \
    do {                                                                                       \
        VkResult _vk_res = (expr);                                                             \
        if (_vk_res != VK_SUCCESS) {                                                           \
            throw std::runtime_error(std::string("Vulkan error ") +                            \
                                     cybercab::vkResultToString(_vk_res) + " at " + __FILE__ + \
                                     ":" + std::to_string(__LINE__));                          \
        }                                                                                      \
    } while (0)

inline VkDeviceSize alignUp(VkDeviceSize v, VkDeviceSize align) {
    return (v + align - 1) & ~(align - 1);
}

} // namespace cybercab
