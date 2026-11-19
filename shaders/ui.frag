#version 450

layout(binding = 0) uniform sampler2D uImage;
layout(binding = 1) uniform sampler2D uFont;

layout(location = 0) in vec2 vUv;
layout(location = 1) in vec4 vColor;
layout(location = 2) flat in float vMode;

layout(location = 0) out vec4 outColor;

void main() {
    if (vMode < 0.5) {
        outColor = vColor;
    } else if (vMode < 1.5) {
        outColor = vec4(texture(uImage, vUv).rgb, 1.0) * vColor;
    } else {
        const float coverage = texture(uFont, vUv).r;
        outColor = vec4(vColor.rgb, vColor.a * coverage);
    }
}
