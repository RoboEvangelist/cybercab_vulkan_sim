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

    // Soft metallic clearcoat & tinted canopy glass shading on Cybercab ego mesh (SemId == 1.25)
    if (abs(inSemId - 1.25) < 0.08 && inEmissive < 0.1) {
        const float kSpecTightPow = 42.0, kSpecTightGain = 0.24;
        const float kSpecBroadPow = 10.0, kSpecBroadGain = 0.09;
        const float kGlassSpecPow = 96.0, kGlassSpecGain = 0.55;
        const float kGlassMaxLuma = 0.085, kGlassBlueRatio = 1.45;
        const vec3 kSkyZenith = vec3(0.42, 0.62, 0.88);
        const vec3 kSkyHorizon = vec3(0.70, 0.82, 0.95);
        const vec3 kPaintWarmTint = vec3(1.0, 0.94, 0.80);

        const float luma = dot(inColor.rgb, vec3(0.299, 0.587, 0.114));
        const bool isTintedGlass = (luma < kGlassMaxLuma) && (inColor.b > inColor.r * kGlassBlueRatio);
        const vec3 halfH = normalize(L + vec3(0.14, 0.42, 0.20));
        const float ndoth = max(dot(N, halfH), 0.0);
        const float skyRefl = clamp(N.y * 0.75 + 0.35, 0.0, 1.0);
        const vec3 skyCol = mix(kSkyHorizon, kSkyZenith, skyRefl);

        if (isTintedGlass) {
            const float fresnelGlass = 0.10 + 0.32 * pow(1.0 - abs(N.y), 2.5);
            const float glassSpec = pow(ndoth, kGlassSpecPow) * kGlassSpecGain;
            lit = inColor.rgb * (ambient * 0.8 + ndotl * 0.45) + skyCol * fresnelGlass + vec3(glassSpec);
        } else {
            const float spec = pow(ndoth, kSpecTightPow) * kSpecTightGain + pow(ndoth, kSpecBroadPow) * kSpecBroadGain;
            const float crestSheen = pow(max(N.y, 0.0), 2.5) * 0.10;
            const float warmBounce = max(-N.y, 0.0) * 0.13;
            const float rimFresnel = smoothstep(0.0, 0.45, 1.0 - abs(N.y)) * 0.05;
            lit += inColor.rgb * (crestSheen + warmBounce) + kPaintWarmTint * spec + skyCol * rimFresnel;
        }
    }

    // FogExp2 on view depth (1 / w of clip space).
    const float viewDist = 1.0 / max(gl_FragCoord.w, 1e-4);
    const float fd = pc.fogColorDensity.w * viewDist;
    const float fogFactor = clamp(1.0 - exp(-fd * fd), 0.0, 1.0);
    outColor = vec4(clamp(mix(lit, pc.fogColorDensity.rgb, fogFactor), 0.0, pc.hdrParams.x), inColor.a);
}
