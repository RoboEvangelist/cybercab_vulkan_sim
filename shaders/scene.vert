#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inColor;
layout(location = 3) in float inEmissive;
layout(location = 4) in float inSemId;

layout(push_constant) uniform ScenePC {
    mat4 viewProj;
    vec4 sunDirIntensity; // xyz = normalized sun dir, w = ambient
    vec4 fogColorDensity; // rgb = sky/fog color, w = fog density
} pc;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec4 outColor;
layout(location = 3) out float outEmissive;
layout(location = 4) out float outSemId;

void main() {
    vec4 wp = vec4(inPos, 1.0);
    gl_Position = pc.viewProj * wp;
    outWorldPos = inPos;
    outNormal = normalize(inNormal);
    outColor = inColor;
    outEmissive = inEmissive;
    outSemId = inSemId;
}
