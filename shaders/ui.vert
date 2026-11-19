#version 450

layout(location = 0) in vec2 inPos;   // pixels, origin top-left
layout(location = 1) in vec2 inUv;
layout(location = 2) in vec4 inColor;
layout(location = 3) in float inMode; // 0 = solid, 1 = camera texture, 2 = font atlas

layout(push_constant) uniform UiPC {
    vec2 screenSize;
} pc;

layout(location = 0) out vec2 vUv;
layout(location = 1) out vec4 vColor;
layout(location = 2) flat out float vMode;

void main() {
    const vec2 ndc = inPos / pc.screenSize * 2.0 - 1.0; // Vulkan NDC is y-down, like pixels
    gl_Position = vec4(ndc, 0.0, 1.0);
    vUv = inUv;
    vColor = inColor;
    vMode = inMode;
}
