#version 450

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inColor;
layout(location = 3) in float inEmissive;
layout(location = 4) in float inSemId;

layout(push_constant) uniform ScenePC {
    mat4 viewProj;
    vec4 sunDirAmbient;   // xyz = sun direction, w = ambient
    vec4 fogColorDensity; // rgb = fog color, w = FogExp2 density
    vec4 hdrParams;       // x = max output radiance (1.0 = v14 display clamp, >1 = v15 linear HDR headroom)
} pc;

layout(location = 0) out vec4 outColor;

void main() {
    const vec3 N = normalize(inNormal);
    const vec3 L = normalize(pc.sunDirAmbient.xyz);
    const float ndotl = max(dot(N, L), 0.0);
    const float hemi = 0.5 + 0.5 * N.y;
    const float ambient = pc.sunDirAmbient.w * (0.6 + 0.4 * hemi);
    vec3 lit = inColor.rgb * (ambient + ndotl * 0.85) + inColor.rgb * inEmissive;

    // Soft metallic / clearcoat automotive sheen on vehicle bodies & wheels (SemId == 1)
    if (abs(inSemId - 1.0) < 0.25 && inEmissive < 0.1) {
        const vec3 halfH = normalize(L + vec3(0.12, 0.38, 0.18));
        const float spec = pow(max(dot(N, halfH), 0.0), 22.0) * 0.22;
        const float crestSheen = pow(max(N.y, 0.0), 2.5) * 0.10;
        const float warmBounce = max(-N.y, 0.0) * 0.14;
        lit += inColor.rgb * (crestSheen + warmBounce) + vec3(spec, spec * 0.94, spec * 0.80);
    }

    // FogExp2 on view depth (1 / w of clip space).
    const float viewDist = 1.0 / max(gl_FragCoord.w, 1e-4);
    const float fd = pc.fogColorDensity.w * viewDist;
    const float fogFactor = clamp(1.0 - exp(-fd * fd), 0.0, 1.0);
    outColor = vec4(clamp(mix(lit, pc.fogColorDensity.rgb, fogFactor), 0.0, pc.hdrParams.x), inColor.a);
}
