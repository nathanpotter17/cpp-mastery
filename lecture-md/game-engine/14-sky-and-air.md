# Chapter 14: Sky and air

By the end of this chapter, the sky is lit by light that has scattered any number of times, and the air between the camera and the scene is visible: distant things fade toward the sky's color and take on its tint, as they do outdoors. Chapter 13 removed the far plane, so the engine can now show ground kilometres away. Without air, that ground looks as crisp as ground at your feet, and the eye reads it as near.

Chapter 8's sky was **single scattering**: sunlight that scatters once in the air, toward the camera. That leaves out all the light that scatters again, so its sky was darker than a real one, its twilight went black too early, and it was only a background: nothing in the scene was seen through it.

This chapter replaces it with Hillaire's atmosphere ("A Scalable and Production Ready Sky and Atmosphere Rendering Technique", EGSR 2020). The atmosphere is still a sphere of air, haze and ozone around a spherical planet, but it's computed as a few small **lookup tables**, each cheap to build and cheap to read:
1. **Transmittance:** how much light gets through the air, from any height, in any direction, to space. 256 × 64 texels, built once.
2. **Multiple scattering:** how much light, scattered twice or more, reaches each height, for each angle of the sun. 32 × 32 texels, built once.
3. **The sky view:** the sky around the camera, in every direction. 192 × 108 texels, the size in Hillaire's sample code (the paper uses 200 × 100), every frame.
4. **Aerial perspective:** the air between the camera and the scene, over the screen and out to 32 km, 32 × 32 × 32. Hillaire stores it as one volume, with the transmittance averaged over the colors in alpha; we keep two, so the transmittance keeps its color: the light the air adds, and the share of the scene's light it lets through. Every frame.

The two per-frame tables take about 0.02 ms of GPU time on an RTX 5070 Laptop, and their cost doesn't depend on the screen's resolution.

Two more pieces come with it:
- **A ground:** a flat plane 65.5 km across, under the scene, so there's something to see the air over.
- **Energy compensation for rough metals:** Kulla and Conty's (2017) idea, in Turquin's (2019) simpler form. The GGX reflection Chapter 6 built counts light that bounces once off the microfacets. On a rough surface, much of the light bounces between them and still leaves. Without it, a fully rough white metal reflects only about 40% of the light it gets, averaged over the angles it's lit from, and rough surfaces look dull.

The photographed sky stays, as an option for the background and its light. It has no air to go with it, so with it there's no haze.

This chapter builds on [Chapter 13](13-range.md).

## 14.1 The atmosphere's model: `atmosphere.slangh`

### Why
The tables, the background and the scene's shaders all need the same air, the same geometry, and the same mappings between a table's texels and what they stand for.

### How
- **The planet and the air:** a sphere 6,360 km in radius, with the atmosphere's top 100 km above it. Positions are relative to the planet's centre, with the camera above its top, on the +Y axis. Over the few kilometres a scene spans, the ground is taken as flat: only the camera's height moves it on the sphere.
- **The medium (`medium_at`):** what the air at a height scatters and absorbs, per metre:
  - **Air molecules (Rayleigh scattering):** (5.802, 13.558, 33.1) × 10⁻⁶ per metre at the ground, for red, green and blue, thinning by a factor of e every 8 km. Blue scatters most, which is why the sky is blue and the low sun red.
  - **Haze (Mie scattering):** 3.996 × 10⁻⁶ per metre, the same for every color, plus a little absorption, thinning every 1.2 km. It scatters mostly forward, which makes the glow around the sun.
  - **Ozone** absorbs, mostly green and red (its Chappuis band, centred on orange), hardly any blue, in a layer peaking 25 km up and fading to nothing 15 km above and below. At sunset, when the light crosses most of it, it keeps the sky overhead blue.
  - **Extinction** is all of the above added up: how much light a metre of the medium takes out of a ray, scattered elsewhere or absorbed.
  - **Below the sphere:** the scene's ground is flat, the plane touching the planet's top, so 32 km away it's about 80 m above the curved ground (d²/2R), and the air there is taken at that height. But the aerial perspective volume reaches past the scene's ground: along a downward ray, past the ground is under the sphere, and the volume's filtering blends those parts in near the ground. Down there, `medium_at` takes the air to be as it is at the ground, rather than growing ever denser. Without it, the ground glows in blocks at night.
  
  These are the values of Bruneton's and Hillaire's models, as Chapter 8 used them.
- **Phase functions:** how much light scatters at each angle. Rayleigh's, as much back as forward; Cornette and Shanks's for haze, mostly forward, with an asymmetry parameter of 0.8.
- **Spheres:** `sphere_exit` finds where a ray leaves a sphere from inside, and `ground_hit` where it hits the planet, if it does.
- **The transmittance table (`transmittance_uv`, `transmittance_r_mu`):** each texel stands for a radius `r` and a zenith cosine `mu`, and holds the share of light that crosses the atmosphere from there, along that direction, to space. The mapping is Bruneton's, from Bruneton and Neyret's 2008 paper's parameterization, in the form of his 2017 reference implementation: texture x is how far the ray's distance to the top of the atmosphere is between the shortest and longest possible from that radius, and texture y is the distance to the horizon. Both spread the texels toward the horizon, where transmittance changes fastest.
- **Sunlight (`sunlight_at`):** the sun's illuminance outside the atmosphere, 128,000 lux, times the transmittance toward the sun. Where the sun is below the horizon, seen from the point, the planet is in the way and there's none (Bruneton's ground test): that's the planet's shadow, which at dusk climbs the sky from the east. Points below the sphere count as on it.
- **The multiple scattering table (`multiscatter_uv`):** by the sun's zenith cosine across and height down. The texel centres at the ends fall exactly on the ends of the range.
- **Light along a ray (`integrate_ray`):** what a ray sees of the atmosphere over a distance. It's split into segments, and in each one the medium is taken as constant:
  - **The source:** what each metre of the segment scatters toward the ray. Single scattering, sunlight times each scattering coefficient times its phase function; plus multiple scattering, the table's value for that height and sun times the medium's scattering. The multiple scattering table already spreads its light evenly in every direction, so it needs no phase function.
  - **Integrated exactly:** across the segment, the light it adds is itself dimmed as it comes from deeper in. Integrated with the segment's own transmittance, the segment adds `source × (1 − transmittance) / extinction` (Hillaire, "Physically Based and Unified Volumetric Rendering in Frostbite", 2015). With few steps, this stays energy-conserving where a plain sum wouldn't: however thick the segment, it never adds more light than its medium could scatter.
  - It returns both the light and the transmittance, `Scattered`.
- **The sky-view table (`sky_view_uv`, `sky_view_direction`):** Hillaire's, as in his sample code: by the view's angle around the up axis, measured from the sun's, across; and by its angle from the zenith, down. The zenith angle is squeezed toward the horizon, top half above it and bottom half below, because that's where the sky changes fastest. The angle around is squeezed toward the sun too. One table covers every sun direction around the up axis, since only the angle between the view and the sun matters.
- **Aerial perspective (`aerial_slice_distance`, `aerial_depth`):** 32 slices in depth, out to 32 km. The slices are spaced quadratically, slice `s` reaching `((s + 1) / 32)² × 32 km`, so the near ones, where things are seen in detail, are thin: the first is 31 m deep. `aerial_depth` turns a distance into the volume's depth coordinate. Closer than the first slice's far edge, a point gets a share of the first slice's air, in proportion to its distance, so the haze fades to nothing at the camera. The slices' spacing is our choice; Hillaire's sample code also squares it, but spreads the slices further.

### Code
`game-engine/shaders/atmosphere.slangh`:
```slang
// The atmosphere: Earth's air, haze and ozone, and how light crosses them.
// Hillaire 2020, "A Scalable and Production Ready Sky and Atmosphere
// Rendering Technique", with Bruneton's transmittance lookup (from Bruneton
// and Neyret 2008, as in his 2017 reference implementation). Shared by atmosphere.slang, which builds the lookup tables,
// background.slang, which draws the sky from one of them, and mesh.slang,
// which reads the haze.
//
// The planet is a sphere; the camera stands above its top, at x = z = 0.
// Every position here is relative to the planet's centre, in metres, and
// the world's +Y is up. Over the few kilometres a scene spans, the ground
// is taken as flat: only the camera's height moves it on the sphere.

static const float atmosphere_pi = 3.14159265;

// --- The medium ----------------------------------------------------------------

static const float planet_radius = 6360e3;
static const float atmosphere_radius = 6460e3;

// Air molecules (Rayleigh scattering): far more for blue than red, which is
// why the sky is blue, and why the sun reddens near the horizon.
static const float3 rayleigh_scattering = float3(5.802e-6, 13.558e-6, 33.1e-6);
static const float rayleigh_height = 8000.0;  // density falls to 1/e of its value every 8 km

// Haze (Mie scattering): the same for every color, and mostly forward.
static const float mie_scattering = 3.996e-6;
static const float mie_extinction = 4.440e-6;  // scattering plus absorption
static const float mie_height = 1200.0;
static const float mie_anisotropy = 0.8;

// Ozone absorbs, mostly green and red, in a layer 25 km up: at sunset, when the
// light crosses most of it, it keeps the sky blue overhead.
static const float3 ozone_absorption = float3(0.650e-6, 1.881e-6, 0.085e-6);

// The sun outside the atmosphere, in lux, and the ground's albedo for the
// light it reflects back up.
static const float sun_illuminance_in_space = 128000.0;
static const float ground_albedo = 0.2;

// How far the aerial perspective volume reaches, in metres.
static const float aerial_distance = 32000.0;

// What the medium is like at `altitude`: how much it scatters, and how much
// it scatters or absorbs in all (extinction), per metre. Below the sphere
// counts as on it: the aerial perspective volume reaches past the scene's
// ground, and along a downward ray, past the ground is under the sphere.
struct Medium {
    float3 rayleigh;    // Rayleigh scattering
    float mie;          // Mie scattering
    float3 extinction;
};

Medium medium_at(float altitude) {
    altitude = max(altitude, 0.0);
    const float air = exp(-altitude / rayleigh_height);
    const float haze = exp(-altitude / mie_height);
    const float ozone = max(0.0, 1.0 - abs(altitude - 25000.0) / 15000.0);

    Medium medium;
    medium.rayleigh = rayleigh_scattering * air;
    medium.mie = mie_scattering * haze;
    medium.extinction = medium.rayleigh + mie_extinction * haze + ozone_absorption * ozone;
    return medium;
}

// Rayleigh's phase function: how much light scatters by angle (cosine mu);
// as much forward as back, least at right angles.
float rayleigh_phase(float mu) {
    return 3.0 / (16.0 * atmosphere_pi) * (1.0 + mu * mu);
}

// The Cornette-Shanks phase function for haze: strongly forward.
float mie_phase(float mu) {
    const float g = mie_anisotropy;
    const float g2 = g * g;
    return 3.0 / (8.0 * atmosphere_pi) * (1.0 - g2) * (1.0 + mu * mu)
        / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * g * mu, 1.5));
}

// --- Spheres -------------------------------------------------------------------

// Distance along a ray from `origin` (relative to the planet's centre) in
// direction `direction` to where it leaves a sphere of `radius` from inside,
// or -1 if it misses.
float sphere_exit(float3 origin, float3 direction, float radius) {
    const float b = dot(origin, direction);
    const float c = dot(origin, origin) - radius * radius;
    const float discriminant = b * b - c;
    return discriminant < 0.0 ? -1.0 : -b + sqrt(discriminant);
}

// Distance to where a ray hits the ground, or -1 if it doesn't.
float ground_hit(float3 origin, float3 direction) {
    const float b = dot(origin, direction);
    const float c = dot(origin, origin) - planet_radius * planet_radius;
    const float discriminant = b * b - c;

    if (discriminant < 0.0) {
        return -1.0;
    }

    const float t = -b - sqrt(discriminant);
    return t > 0.0 ? t : -1.0;
}

// --- Lookup tables ---------------------------------------------------------------

// Transmittance: the share of light, per channel, that crosses the
// atmosphere from a point at radius r looking up at zenith cosine mu, all
// the way out. Bruneton's mapping: texture x is how far along the
// possible distances to the top the ray's is, texture y the radius, both
// spaced so that the horizon, where it changes fastest, gets the most texels.
float2 transmittance_uv(float r, float mu) {
    const float h = sqrt(atmosphere_radius * atmosphere_radius - planet_radius * planet_radius);
    const float rho = sqrt(max(0.0, r * r - planet_radius * planet_radius));
    const float discriminant = r * r * (mu * mu - 1.0) + atmosphere_radius * atmosphere_radius;
    const float d = max(0.0, -r * mu + sqrt(max(discriminant, 0.0)));
    const float d_min = atmosphere_radius - r;
    const float d_max = rho + h;
    return float2((d - d_min) / (d_max - d_min), rho / h);
}

// The inverse: the radius and zenith cosine a texel stands for.
void transmittance_r_mu(float2 uv, out float r, out float mu) {
    const float h = sqrt(atmosphere_radius * atmosphere_radius - planet_radius * planet_radius);
    const float rho = h * uv.y;
    r = sqrt(rho * rho + planet_radius * planet_radius);
    const float d_min = atmosphere_radius - r;
    const float d_max = rho + h;
    const float d = d_min + uv.x * (d_max - d_min);
    mu = d == 0.0 ? 1.0 : clamp((h * h - rho * rho - d * d) / (2.0 * r * d), -1.0, 1.0);
}

// The sunlight that reaches `point`: the sun's illuminance outside, times
// the transmittance toward it, or nothing where the planet is in the way:
// where the sun is below the horizon seen from radius r (Bruneton's
// test). Points below the sphere count as on it, as in medium_at.
float3 sunlight_at(Texture2D transmittance, SamplerState clamped, float3 point, float3 sun) {
    const float length_point = length(point);
    const float r = max(length_point, planet_radius);
    const float mu = dot(point / length_point, sun);

    if (mu < 0.0 && r * r * (mu * mu - 1.0) + planet_radius * planet_radius >= 0.0) {
        return float3(0.0);
    }

    return sun_illuminance_in_space * transmittance.SampleLevel(clamped, transmittance_uv(r, mu), 0.0).rgb;
}

// Multiple scattering (Hillaire 2020, section 5.5): light scattered more
// than once, per unit of sun illuminance, by a point at radius r with the
// sun at zenith cosine mu. Texture x is the sun's zenith cosine, y the
// altitude; both pulled half a texel in from the edges, so the texel
// centres at the ends land exactly on them.
static const float multiscatter_size = 32.0;

float2 multiscatter_uv(float r, float sun_mu) {
    const float2 unit = float2(sun_mu * 0.5 + 0.5, (r - planet_radius) / (atmosphere_radius - planet_radius));
    return (saturate(unit) * (multiscatter_size - 1.0) + 0.5) / multiscatter_size;
}

// --- The light scattered along a ray ------------------------------------------------

// What a ray from `origin` in `direction` sees of the atmosphere over its
// first `distance` metres: the light scattered toward it (in nits, sun
// included through `sunlight_at`, plus multiple scattering), and the
// transmittance of that stretch. Integrated in `steps` segments; within each,
// the medium is taken as constant, and the light it adds is integrated
// exactly against the transmittance falling across it (Hillaire 2015).
struct Scattered {
    float3 light;
    float3 transmittance;
};

Scattered integrate_ray(
    Texture2D transmittance, Texture2D multiscatter, SamplerState clamped,
    float3 origin, float3 direction, float distance, float3 sun, uint steps
) {
    const float mu = dot(direction, sun);
    const float phase_rayleigh = rayleigh_phase(mu);
    const float phase_mie = mie_phase(mu);
    const float dt = distance / float(steps);

    Scattered result;
    result.light = 0.0;
    result.transmittance = 1.0;

    for (uint i = 0; i < steps; ++i) {
        const float3 point = origin + direction * (dt * (float(i) + 0.5));
        const float r = length(point);
        const Medium medium = medium_at(r - planet_radius);
        const float3 up = point / r;

        // Single scattering, toward the ray, by angle; multiple scattering,
        // already spread evenly, by how much the medium scatters at all.
        const float3 scattering = medium.rayleigh * phase_rayleigh + medium.mie * phase_mie;
        const float3 multiple = multiscatter.SampleLevel(clamped, multiscatter_uv(r, dot(up, sun)), 0.0).rgb;
        const float3 source = sunlight_at(transmittance, clamped, point, sun) * scattering
            + sun_illuminance_in_space * multiple * (medium.rayleigh + medium.mie);

        // The segment's own transmittance, and what it adds: the source
        // integrated over the segment, dimmed as it goes.
        const float3 segment = exp(-medium.extinction * dt);
        const float3 added = (source - source * segment) / max(medium.extinction, 1e-12);

        result.light += result.transmittance * added;
        result.transmittance *= segment;
    }

    return result;
}

// --- The sky-view table ---------------------------------------------------------------

// The sky around the camera, by direction: Hillaire 2020's sky-view table,
// 192 x 108 texels. Texture y is the view's angle from the zenith: the top
// half from the zenith down to the horizon, the bottom half from the horizon
// to the nadir, each squeezed so the horizon, where the sky changes fastest,
// gets the most texels. Texture x is the view's angle around the up axis,
// from the sun (x = 0) to opposite it (x = 1), also squeezed toward the sun.
static const float2 sky_view_size = float2(192.0, 108.0);

// The camera's position for the tables: above the planet's top by `altitude`.
float3 camera_origin(float altitude) {
    return float3(0.0, planet_radius + max(altitude, 1.0), 0.0);
}

// The angle from the zenith down to the horizon, at radius r: past the
// horizon, a ray hits the ground.
float horizon_zenith_angle(float r) {
    const float to_horizon = sqrt(max(r * r - planet_radius * planet_radius, 0.0));
    return atmosphere_pi - acos(to_horizon / r);
}

float2 sky_view_uv(float r, float view_zenith_cos, float light_view_cos) {
    const float zenith_horizon = horizon_zenith_angle(r);
    const float beta = atmosphere_pi - zenith_horizon;  // from the horizon to the nadir
    const float angle = acos(clamp(view_zenith_cos, -1.0, 1.0));

    float2 uv;
    if (angle < zenith_horizon) {
        const float coord = 1.0 - sqrt(max(1.0 - angle / zenith_horizon, 0.0));
        uv.y = coord * 0.5;
    } else {
        const float coord = sqrt(max((angle - zenith_horizon) / beta, 0.0));
        uv.y = coord * 0.5 + 0.5;
    }
    uv.x = sqrt(saturate(-light_view_cos * 0.5 + 0.5));

    // Pulled half a texel in from the edges, as for multiple scattering.
    return (uv * (sky_view_size - 1.0) + 0.5) / sky_view_size;
}

// The inverse: the view direction a texel stands for, in a frame where the
// sun lies in the plane of +X and +Y.
float3 sky_view_direction(float r, float2 uv) {
    uv = saturate((uv * sky_view_size - 0.5) / (sky_view_size - 1.0));

    const float zenith_horizon = horizon_zenith_angle(r);
    const float beta = atmosphere_pi - zenith_horizon;

    float angle;
    if (uv.y < 0.5) {
        const float coord = 1.0 - uv.y * 2.0;
        angle = zenith_horizon * (1.0 - coord * coord);
    } else {
        const float coord = uv.y * 2.0 - 1.0;
        angle = zenith_horizon + beta * coord * coord;
    }

    const float light_view_cos = -(uv.x * uv.x * 2.0 - 1.0);
    const float sin_light_view = sqrt(max(1.0 - light_view_cos * light_view_cos, 0.0));
    const float sin_zenith = sin(angle);
    return float3(sin_zenith * light_view_cos, cos(angle), sin_zenith * sin_light_view);
}

// The sky in world direction `direction`, from the sky-view table: its
// angle from the zenith, and its angle around the up axis from the sun's.
float3 sky_view_radiance(Texture2D sky_view, SamplerState clamped, float altitude, float3 direction, float3 sun) {
    const float r = planet_radius + max(altitude, 1.0);
    const float2 view_horizontal = direction.xz;
    const float2 sun_horizontal = sun.xz;
    const float lengths = length(view_horizontal) * length(sun_horizontal);
    const float light_view_cos = lengths > 1e-6 ? dot(view_horizontal, sun_horizontal) / lengths : 1.0;
    return sky_view.SampleLevel(clamped, sky_view_uv(r, direction.y, light_view_cos), 0.0).rgb;
}

// --- Aerial perspective ------------------------------------------------------------

// The air between the camera and a point: 32 x 32 texels across the screen,
// 32 slices in depth, spread quadratically out to aerial_distance, so the
// near slices, where things are seen in detail, are thin. Slice s holds the
// air from the camera out to its far edge, ((s + 1) / 32)^2 x 32 km.
static const float aerial_slices = 32.0;

float aerial_slice_distance(float slice) {
    const float edge = (slice + 1.0) / aerial_slices;
    return edge * edge * aerial_distance;
}

// The texture's depth coordinate for a point `distance` away, and how much
// of the first slice's air it has: in front of slice 0's far edge, the air
// is slice 0's, in proportion to the distance.
float aerial_depth(float distance, out float first_slice_share) {
    const float edge = sqrt(saturate(distance / aerial_distance)) * aerial_slices;
    first_slice_share = saturate(edge * edge);
    return (max(edge, 1.0) - 0.5) / aerial_slices;
}
```

## 14.2 Building the tables: `atmosphere.slang`, `cube.slangh`, `environment.slang`

### Why
Each table is one compute shader. They run in order, each reading the ones before.

### How
- **`transmittanceMain`:** for each texel's radius and angle, marches up to the top of the atmosphere in 40 steps, adding up the extinction. The transmittance is `exp` of minus the sum.
- **`multiscatterMain`:** Hillaire's approximation of every bounce after the first, for each height and sun angle:
  - **64 directions,** spread evenly over the sphere: 8 cosines of the polar angle, evenly spaced, times 8 angles around. Each is marched to the top of the atmosphere or the ground, in 20 steps.
  - **Two sums:** `L`, the light scattered once toward the point from all around it, as if the point then scattered it evenly; and `f`, the share of evenly scattered light that would come back to the point. Ground the march hits reflects its sunlight with an albedo of 0.2.
  - **Every bounce:** the light that comes back gets scattered again, and a share `f` of it comes back again, and so on. All the bounces together add up to `L (1 + f + f² + …) = L / (1 − f)`.
  - Stored per unit of sun illuminance; `integrate_ray` multiplies it back.
- **`sky_radiance`:** the sky in one direction: `integrate_ray` in 32 steps to the top of the atmosphere or the ground. Where the ray hits the ground, the ground's own reflected sunlight is added, seen through the air.
- **`skyViewMain`:** one thread per sky-view texel, in a frame where the sun lies in the +X, +Y plane, at its real height.
- **`aerialMain`:** one thread per column of the volume, 32 × 32:
  - **The ray** goes from the camera through the column's point on the screen. Its direction comes from a point on the near plane, unprojected: in camera-relative space, that point's position is its direction.
  - **The march:** slice by slice, `integrate_ray` over each slice's depth in 2 steps, each slice's light dimmed by the transmittance of the slices in front of it. Each slice stores the totals so far, from the camera to its far edge.
- **`skyCubeMain`:** the sky in every direction into mip 0 of the sky cube, as Chapter 8's `skyMain` did: the rest of the environment, the irradiance and the specular cube, is built from it as before. Thread 0 also works out the sunlight at the camera, for the scene's sun.
- **`cube.slangh`:** `cube_direction` moves here from `environment.slang`, since `atmosphere.slang` needs it too.
- **`environment.slang`** loses Chapter 8's atmosphere and `skyMain`. `photograph_nits` moves here too, from the push data: it never changed.

### Code
`game-engine/shaders/atmosphere.slang`:
```slang
// The atmosphere's lookup tables (atmosphere.slangh), in compute shaders:
//   transmittanceMain  how much light crosses the atmosphere, from any
//                      height and angle: once
//   multiscatterMain   the light scattered more than once: once, from the
//                      transmittance
//   skyViewMain        the sky around the camera: every frame
//   aerialMain         the air between the camera and the scene, as two
//                      volumes over the view: every frame
//   skyCubeMain        the sky, for lighting, into the environment's sky
//                      cube, whenever the sun moves

#include "shared.slangh"
#include "atmosphere.slangh"
#include "cube.slangh"

// --- Data shared with C++ (src/includes/shader_types.h) ----------------------

struct AtmospherePushData {
    FrameData* frame;         // aerialMain: this frame's FrameData
    EnvironmentInfo* info;    // skyCubeMain: where the sunlight goes
    float3 sun_direction;     // toward the sun
    float altitude;           // the camera's height above the ground, in metres
    uint transmittance;       // sampled slot
    uint multiscatter;        // sampled slot
    uint target;              // storage slot this step writes
    uint second_target;       // aerialMain: the transmittance volume
    uint size;                // skyCubeMain: the cube's face size
    uint sampler;             // sampler heap index: the clamp sampler
};

[[vk::push_constant]]
ConstantBuffer<AtmospherePushData> push;

Texture2D transmittance_table() {
    return Texture2D.Handle(uint2(push.transmittance, 0));
}

Texture2D multiscatter_table() {
    return Texture2D.Handle(uint2(push.multiscatter, 0));
}

SamplerState clamped() {
    return SamplerState.Handle(uint2(push.sampler, 0));
}

// The sky seen from `origin` in `direction`: the atmosphere up to its top,
// or to the ground, and then the ground itself, lit by the sun and seen
// through the air in between.
float3 sky_radiance(float3 origin, float3 direction, float3 sun) {
    const float ground = ground_hit(origin, direction);
    const float distance = ground > 0.0 ? ground : sphere_exit(origin, direction, atmosphere_radius);

    const Scattered air = integrate_ray(transmittance_table(), multiscatter_table(), clamped(),
        origin, direction, distance, sun, 32);
    float3 radiance = air.light;

    if (ground > 0.0) {
        const float3 point = origin + direction * ground;
        const float3 up = normalize(point);
        const float3 sunlight = sunlight_at(transmittance_table(), clamped(), point + up, sun);
        radiance += ground_albedo / atmosphere_pi * sunlight * max(dot(up, sun), 0.0) * air.transmittance;
    }

    return radiance;
}

// --- Transmittance ---------------------------------------------------------------

[shader("compute")]
[numthreads(8, 8, 1)]
void transmittanceMain(uint3 id : SV_DispatchThreadID) {
    RWTexture2D<float4> table = RWTexture2D<float4>.Handle(uint2(push.target, 0));
    uint width, height;
    table.GetDimensions(width, height);

    if (id.x >= width || id.y >= height) {
        return;
    }

    float r, mu;
    transmittance_r_mu((float2(id.xy) + 0.5) / float2(width, height), r, mu);

    // March up to the top of the atmosphere, adding up the extinction.
    const float3 origin = float3(0.0, r, 0.0);
    const float3 direction = float3(sqrt(max(1.0 - mu * mu, 0.0)), mu, 0.0);
    const float distance = sphere_exit(origin, direction, atmosphere_radius);

    const uint steps = 40;
    const float dt = distance / float(steps);
    float3 optical_depth = 0.0;

    for (uint i = 0; i < steps; ++i) {
        const float3 point = origin + direction * (dt * (float(i) + 0.5));
        optical_depth += medium_at(length(point) - planet_radius).extinction * dt;
    }

    table[id.xy] = float4(exp(-optical_depth), 1.0);
}

// --- Multiple scattering ----------------------------------------------------------

// For each height and sun angle (Hillaire 2020, section 5.5): march in 64
// directions spread evenly over the sphere, and add up
//   L  the light scattered once toward the point, with an even (isotropic)
//      phase function: the second bounce, if the point scatters it again,
//   f  how much light the point would get back, as a share, from scattering
//      that's itself spread evenly.
// Every further bounce gives back f times the one before, so all of them
// together are L (1 + f + f^2 + ...) = L / (1 - f). Per unit of sunlight.
[shader("compute")]
[numthreads(8, 8, 1)]
void multiscatterMain(uint3 id : SV_DispatchThreadID) {
    RWTexture2D<float4> table = RWTexture2D<float4>.Handle(uint2(push.target, 0));

    if (any(id.xy >= uint2(multiscatter_size))) {
        return;
    }

    // The texel's sun angle and height: multiscatter_uv, backwards. Texel
    // centres span 0 to 1 exactly, ends included.
    const float2 unit = float2(id.xy) / (multiscatter_size - 1.0);
    const float sun_mu = unit.x * 2.0 - 1.0;
    const float3 sun = float3(sqrt(max(1.0 - sun_mu * sun_mu, 0.0)), sun_mu, 0.0);
    const float3 origin = float3(0.0, planet_radius + max(unit.y * (atmosphere_radius - planet_radius), 1.0), 0.0);

    const float isotropic = 1.0 / (4.0 * atmosphere_pi);
    const uint rows = 8;
    const uint steps = 20;

    float3 light = 0.0;
    float3 returned = 0.0;

    for (uint a = 0; a < rows; ++a) {
        for (uint b = 0; b < rows; ++b) {
            // Evenly over the sphere: cosines of the polar angle evenly
            // spaced, and the angle around evenly too.
            const float cos_theta = 1.0 - 2.0 * (float(a) + 0.5) / float(rows);
            const float sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
            const float phi = 2.0 * atmosphere_pi * (float(b) + 0.5) / float(rows);
            const float3 direction = float3(sin_theta * cos(phi), cos_theta, sin_theta * sin(phi));

            const float ground = ground_hit(origin, direction);
            const float distance = ground > 0.0 ? ground : sphere_exit(origin, direction, atmosphere_radius);
            const float dt = distance / float(steps);

            float3 throughput = 1.0;

            for (uint i = 0; i < steps; ++i) {
                const float3 point = origin + direction * (dt * (float(i) + 0.5));
                const Medium medium = medium_at(length(point) - planet_radius);
                const float3 scattering = medium.rayleigh + medium.mie;
                const float3 segment = exp(-medium.extinction * dt);

                // Over the segment, dimmed as it goes, as in integrate_ray.
                const float3 integral = throughput * (1.0 - segment) / max(medium.extinction, 1e-12);
                const float3 sunlight = sunlight_at(transmittance_table(), clamped(), point, sun) / sun_illuminance_in_space;

                light += integral * scattering * isotropic * sunlight;
                returned += integral * scattering;
                throughput *= segment;
            }

            // The ground reflects the sunlight it gets, evenly (Lambertian).
            if (ground > 0.0) {
                const float3 point = origin + direction * ground;
                const float3 up = normalize(point);
                const float3 sunlight = sunlight_at(transmittance_table(), clamped(), point + up, sun) / sun_illuminance_in_space;
                light += throughput * ground_albedo / atmosphere_pi * sunlight * max(dot(up, sun), 0.0);
            }
        }
    }

    // The average over the 64 directions: their sum times the sphere's
    // solid angle, shared among them, times the isotropic phase, 1 / 4 pi.
    const float directions = float(rows * rows);
    light /= directions;
    returned /= directions;

    table[id.xy] = float4(light / (1.0 - returned), 1.0);
}

// --- The sky view ------------------------------------------------------------------

[shader("compute")]
[numthreads(8, 8, 1)]
void skyViewMain(uint3 id : SV_DispatchThreadID) {
    RWTexture2D<float4> table = RWTexture2D<float4>.Handle(uint2(push.target, 0));

    if (any(float2(id.xy) >= sky_view_size)) {
        return;
    }

    // The table's frame has the sun in the plane of +X and +Y, at its real
    // angle from the zenith.
    const float3 origin = camera_origin(push.altitude);
    const float sun_mu = push.sun_direction.y;
    const float3 sun = float3(sqrt(max(1.0 - sun_mu * sun_mu, 0.0)), sun_mu, 0.0);
    const float3 direction = sky_view_direction(origin.y, (float2(id.xy) + 0.5) / sky_view_size);

    table[id.xy] = float4(sky_radiance(origin, direction, sun), 1.0);
}

// --- Aerial perspective ---------------------------------------------------------------

// One thread per column of the volume: the ray from the camera through that
// point of the screen, marched slice by slice, each slice's light and
// transmittance added to the ones before it. The same light and
// transmittance as the sky's, but stopping at each slice's far edge.
[shader("compute")]
[numthreads(8, 8, 1)]
void aerialMain(uint3 id : SV_DispatchThreadID) {
    RWTexture3D<float4> inscatter = RWTexture3D<float4>.Handle(uint2(push.target, 0));
    RWTexture3D<float4> through = RWTexture3D<float4>.Handle(uint2(push.second_target, 0));

    if (any(id.xy >= uint2(aerial_slices))) {
        return;
    }

    // The ray through this column's point on the screen, from a point on the
    // near plane: camera-relative, so its direction is its position.
    const float2 ndc = (float2(id.xy) + 0.5) / aerial_slices * 2.0 - 1.0;
    const float4 near_point = mul(push.frame.inverse_view_projection, float4(ndc, 1.0, 1.0));
    const float3 direction = normalize(near_point.xyz / near_point.w);

    const float3 origin = camera_origin(push.altitude);

    float3 light = 0.0;
    float3 transmittance = 1.0;
    float start = 0.0;

    for (uint slice = 0; slice < uint(aerial_slices); ++slice) {
        const float end = aerial_slice_distance(float(slice));
        const Scattered segment = integrate_ray(transmittance_table(), multiscatter_table(), clamped(),
            origin + direction * start, direction, end - start, push.sun_direction, 2);

        light += transmittance * segment.light;
        transmittance *= segment.transmittance;
        start = end;

        inscatter[uint3(id.xy, slice)] = float4(light, 1.0);
        through[uint3(id.xy, slice)] = float4(transmittance, 1.0);
    }
}

// --- The sky cube, for lighting ---------------------------------------------------------

// The sky in every direction from the camera, into mip 0 of the
// environment's sky cube: one thread per texel, the six faces as the
// dispatch's z. Thread 0 also works out the sunlight at the camera, for the
// scene's sun.
[shader("compute")]
[numthreads(8, 8, 1)]
void skyCubeMain(uint3 texel : SV_DispatchThreadID) {
    const float3 origin = camera_origin(push.altitude);

    if (all(texel == 0)) {
        push.info->sun_illuminance = sunlight_at(transmittance_table(), clamped(), origin, push.sun_direction);
    }

    if (texel.x >= push.size || texel.y >= push.size) {
        return;
    }

    // Kept below 65504, the largest 16-bit float, like the photograph.
    RWTexture2DArray<float4> cube = RWTexture2DArray<float4>.Handle(uint2(push.target, 0));
    const float3 direction = cube_direction(texel, push.size);
    cube[texel] = float4(min(sky_radiance(origin, direction, push.sun_direction), 65504.0), 1.0);
}
```

`game-engine/shaders/cube.slangh`:
```slang
// Cube map directions, shared by environment.slang and atmosphere.slang.

// --- Cube maps ---------------------------------------------------------------

// The direction through texel `texel` of cube face `face`, `size` texels
// across. Vulkan's cube faces are +X, -X, +Y, -Y, +Z, -Z, and each face's
// u and v run along the axes its table in the Vulkan spec gives; this is
// that table read backwards. u and v are -1..1 across the face, through
// texel centers.
float3 cube_direction(uint3 texel, uint size) {
    const float2 uv = (float2(texel.xy) + 0.5) / float(size) * 2.0 - 1.0;

    float3 direction;
    switch (texel.z) {
        case 0: direction = float3(1.0, -uv.y, -uv.x); break;
        case 1: direction = float3(-1.0, -uv.y, uv.x); break;
        case 2: direction = float3(uv.x, 1.0, uv.y); break;
        case 3: direction = float3(uv.x, -1.0, -uv.y); break;
        case 4: direction = float3(uv.x, -uv.y, 1.0); break;
        default: direction = float3(-uv.x, -uv.y, -1.0); break;
    }

    return normalize(direction);
}
```

`game-engine/shaders/environment.slang`:
```slang
// Compute shaders that build the environment the scene is lit by, from a
// sky cube: atmosphere.slang's skyCubeMain renders the simulated sky into it,
// and this file's
//   equirectMain    an HDR photograph of the sky, projected into the sky cube
//   irradianceMain  the sky's diffuse light, as spherical harmonics
//   prefilterMain   the sky's specular light, blurred per roughness
//   brdfLutMain     the BRDF table the specular light is scaled by

// --- Data shared with C++ (src/includes/shader_types.h) ----------------------

struct EnvironmentInfo {
    float3 irradiance_sh[9];
    float3 sun_illuminance;
};

struct EnvironmentPushData {
    EnvironmentInfo* info;
    uint source;
    uint target;
    uint size;
    float roughness;
    uint sampler;
    uint source_size;
};

[[vk::push_constant]]
ConstantBuffer<EnvironmentPushData> push;

static const float pi = 3.14159265;

#include "cube.slangh"

// A cube map is written as a 2D array of 6 layers, one per face.
RWTexture2DArray<float4> storage_target() {
    return RWTexture2DArray<float4>.Handle(uint2(push.target, 0));
}

TextureCube source_cube() {
    return TextureCube.Handle(uint2(push.source, 0));
}

SamplerState clamp_sampler() {
    return SamplerState.Handle(uint2(push.sampler, 0));
}

// --- An HDR photograph of the sky ----------------------------------------------

// The photograph's values are relative: a camera records light, not its
// units. This sets how bright a stored 1.0 is, in nits; with it, the
// kloppenheim sky's average is about that of an overcast evening.
static const float photograph_nits = 2000.0;

// Projects an equirectangular image (longitude across, latitude down: the
// usual layout for sky photographs) into mip 0 of the sky cube. The file's
// values are relative, so photograph_nits turns them into nits.
[shader("compute")]
[numthreads(8, 8, 1)]
void equirectMain(uint3 texel : SV_DispatchThreadID) {
    if (texel.x >= push.size || texel.y >= push.size) {
        return;
    }

    const float3 d = cube_direction(texel, push.size);

    // Longitude 0 (the image's center) looks down -Z; latitude runs from
    // straight up (top row) to straight down (bottom row).
    const float2 uv = float2(atan2(d.x, -d.z) / (2.0 * pi) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / pi);

    const Texture2D image = Texture2D.Handle(uint2(push.source, 0));
    const float3 radiance = image.SampleLevel(clamp_sampler(), uv, 0.0).rgb * photograph_nits;

    // In nits, the brightest texels can pass 65504, the largest 16-bit float
    // the cube can hold; what the GPU stores instead isn't defined, and an
    // infinity would spread into the mips and the lighting. Clamp it.
    storage_target()[texel] = float4(min(radiance, 65504.0), 1.0);
}

// --- Diffuse light: spherical harmonics ----------------------------------------

// A diffuse surface gathers light from its whole hemisphere, so what it
// receives (irradiance) changes slowly with its normal. Nine spherical
// harmonics, the 3D equivalent of a few Fourier terms, capture it to within
// a few percent (Ramamoorthi and Hanrahan, 2001).

// The nine real spherical harmonics, bands 0 to 2, at unit vector n.
void sh_basis(float3 n, out float basis[9]) {
    basis[0] = 0.282095;
    basis[1] = 0.488603 * n.y;
    basis[2] = 0.488603 * n.z;
    basis[3] = 0.488603 * n.x;
    basis[4] = 1.092548 * n.x * n.y;
    basis[5] = 1.092548 * n.y * n.z;
    basis[6] = 0.315392 * (3.0 * n.z * n.z - 1.0);
    basis[7] = 1.092548 * n.x * n.z;
    basis[8] = 0.546274 * (n.x * n.x - n.y * n.y);
}

static const uint irradiance_threads = 64;
static const uint irradiance_face_size = 32;

groupshared float3 partial_sh[irradiance_threads][9];

// One workgroup projects the sky onto the nine harmonics. It reads the sky
// cube at the mip level that's 32 texels across (a detailed sky only blurs
// into irradiance anyway), each texel weighted by the solid angle it covers.
// The sum is then convolved with the cosine lobe a diffuse surface sees,
// which in this basis is just a factor per band: pi, 2pi/3 and pi/4.
[shader("compute")]
[numthreads(irradiance_threads, 1, 1)]
void irradianceMain(uint thread : SV_GroupIndex) {
    const uint size = irradiance_face_size;
    const float lod = log2(float(push.source_size) / float(size));

    float3 sum[9];
    for (uint k = 0; k < 9; ++k) {
        sum[k] = 0.0;
    }

    for (uint i = thread; i < size * size * 6; i += irradiance_threads) {
        const uint3 texel = uint3(i % size, (i / size) % size, i / (size * size));
        const float3 direction = cube_direction(texel, size);

        // A texel at (u, v) on a face one unit away covers (2 / size)^2 of
        // area, seen at a slant: its solid angle shrinks by (1 + u^2 + v^2)^1.5.
        const float2 uv = (float2(texel.xy) + 0.5) / float(size) * 2.0 - 1.0;
        const float solid_angle = 4.0 / (size * size) / pow(1.0 + dot(uv, uv), 1.5);

        const float3 radiance = source_cube().SampleLevel(clamp_sampler(), direction, lod).rgb;

        float basis[9];
        sh_basis(direction, basis);

        for (uint k = 0; k < 9; ++k) {
            sum[k] += radiance * basis[k] * solid_angle;
        }
    }

    for (uint k = 0; k < 9; ++k) {
        partial_sh[thread][k] = sum[k];
    }

    GroupMemoryBarrierWithGroupSync();

    if (thread == 0) {
        static const float band[9] = {
            pi,
            2.0 * pi / 3.0, 2.0 * pi / 3.0, 2.0 * pi / 3.0,
            pi / 4.0, pi / 4.0, pi / 4.0, pi / 4.0, pi / 4.0,
        };

        for (uint k = 0; k < 9; ++k) {
            float3 total = 0.0;
            for (uint t = 0; t < irradiance_threads; ++t) {
                total += partial_sh[t][k];
            }
            push.info->irradiance_sh[k] = total * band[k];
        }
    }
}

// --- Specular light: prefiltering -----------------------------------------------

// The GGX distribution, as in mesh.slang.
float distribution_ggx(float n_dot_h, float alpha) {
    const float alpha2 = alpha * alpha;
    const float f = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
    return alpha2 / (pi * f * f);
}

// A low-discrepancy sequence: point i of n, spread evenly over the unit
// square. Reversing the bits of i gives the second coordinate.
float2 hammersley(uint i, uint n) {
    return float2(float(i) / float(n), float(reversebits(i)) * 2.3283064365386963e-10);
}

// A half vector around `normal`, distributed like GGX's microfacets: more of
// them where D is high. `xi` picks which one.
float3 sample_ggx(float2 xi, float3 normal, float alpha) {
    const float phi = 2.0 * pi * xi.x;
    const float cos_theta = sqrt((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y));
    const float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    const float3 h = float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);

    // From the sample's frame, around +Z, into the normal's.
    const float3 helper = abs(normal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    const float3 tangent = normalize(cross(helper, normal));
    const float3 bitangent = cross(normal, tangent);
    return normalize(tangent * h.x + bitangent * h.y + normal * h.z);
}

// One mip level of the specular cube: the sky as a surface of `roughness`
// reflects it, looking straight at it (n = v = r, the split-sum assumption).
// Rougher levels are smaller: their light is blurrier.
//   - Importance sampling: 128 directions, chosen where GGX reflects the
//     most light, weighted by n.l.
//   - Filtered importance sampling (Krivanek and Colbert, 2008): each
//     sample reads the sky at a mip level matching the solid angle it stands
//     for, so a few samples cover the lobe with little visible noise.
[shader("compute")]
[numthreads(8, 8, 1)]
void prefilterMain(uint3 texel : SV_DispatchThreadID) {
    if (texel.x >= push.size || texel.y >= push.size) {
        return;
    }

    const float3 normal = cube_direction(texel, push.size);

    // Smooth: a mirror. Read the sky at the mip level this size matches.
    if (push.roughness == 0.0) {
        const float lod = log2(float(push.source_size) / float(push.size));
        storage_target()[texel] = source_cube().SampleLevel(clamp_sampler(), normal, lod);
        return;
    }

    const float alpha = push.roughness * push.roughness;
    const uint samples = 128;
    const float texel_solid_angle = 4.0 * pi / (6.0 * push.source_size * push.source_size);

    float3 sum = 0.0;
    float weight = 0.0;

    for (uint i = 0; i < samples; ++i) {
        const float3 h = sample_ggx(hammersley(i, samples), normal, alpha);
        const float3 l = 2.0 * dot(normal, h) * h - normal;
        const float n_dot_l = dot(normal, l);

        if (n_dot_l > 0.0) {
            // With n = v, the probability of this direction is D / 4.
            const float n_dot_h = max(dot(normal, h), 0.0);
            const float pdf = distribution_ggx(n_dot_h, alpha) / 4.0;
            const float sample_solid_angle = 1.0 / (samples * pdf + 1e-6);
            const float lod = max(0.5 * log2(sample_solid_angle / texel_solid_angle), 0.0);

            sum += source_cube().SampleLevel(clamp_sampler(), l, lod).rgb * n_dot_l;
            weight += n_dot_l;
        }
    }

    storage_target()[texel] = float4(sum / max(weight, 1e-6), 1.0);
}

// --- The BRDF table ------------------------------------------------------------------

// Smith's height-correlated visibility, as in mesh.slang.
float visibility_smith(float n_dot_l, float n_dot_v, float alpha) {
    const float alpha2 = alpha * alpha;
    const float from_view = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - alpha2) + alpha2);
    const float from_light = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - alpha2) + alpha2);
    const float sum = from_view + from_light;
    return sum > 0.0 ? 0.5 / sum : 0.0;
}

// The split sum's second half: how much light the specular BRDF reflects
// overall, for a view angle (n.v across) and a roughness (down), as a scale
// and a bias on the reflectance at normal incidence, F0:
//     reflected = F0 * scale + bias
// Computed once, by importance-sampling GGX like the prefilter.
[shader("compute")]
[numthreads(8, 8, 1)]
void brdfLutMain(uint3 texel : SV_DispatchThreadID) {
    if (texel.x >= push.size || texel.y >= push.size) {
        return;
    }

    const float n_dot_v = (texel.x + 0.5) / push.size;
    const float roughness = (texel.y + 0.5) / push.size;
    const float alpha = roughness * roughness;

    const float3 normal = float3(0.0, 0.0, 1.0);
    const float3 view = float3(sqrt(1.0 - n_dot_v * n_dot_v), 0.0, n_dot_v);

    const uint samples = 512;
    float scale = 0.0;
    float bias = 0.0;

    for (uint i = 0; i < samples; ++i) {
        const float3 h = sample_ggx(hammersley(i, samples), normal, alpha);
        const float3 l = 2.0 * dot(view, h) * h - view;
        const float n_dot_l = l.z;

        if (n_dot_l > 0.0) {
            const float n_dot_h = max(h.z, 0.0);
            const float v_dot_h = max(dot(view, h), 0.0);

            // BRDF x n.l / pdf, with Fresnel left out: V * D * n.l over
            // D * n.h / (4 v.h).
            const float reflected = visibility_smith(n_dot_l, n_dot_v, alpha) * 4.0 * n_dot_l * v_dot_h / max(n_dot_h, 1e-6);

            // Schlick's Fresnel, F0 + (1 - F0) * f, split into F0's share and the rest.
            const float fresnel = pow(1.0 - v_dot_h, 5.0);
            scale += (1.0 - fresnel) * reflected;
            bias += fresnel * reflected;
        }
    }

    RWTexture2D<float4> table = RWTexture2D<float4>.Handle(uint2(push.target, 0));
    table[texel.xy] = float4(scale / samples, bias / samples, 0.0, 1.0);
}
```

## 14.3 The tables in the data: `shader_types.h`, `shared.slangh`

### Why
The new compute shaders need push data, and the scene's shaders need to find the per-frame tables.

### How
- **`AtmospherePushData`:** the frame's address (for the aerial perspective's view), the environment info's address (for the sun's light), the sun's direction, the camera's height, the table slots each step reads and writes, the sky cube's size and the sampler. 56 bytes, no padding.
- **`EnvironmentPushData`** loses `sun_direction`, now in `AtmospherePushData`, and `source_scale`, now `photograph_nits` in the shader. 32 bytes, no padding.
- **`FrameData`** gains the sky view's and the two volumes' slots, and `atmosphere`: 1 with the simulated sky, 0 with the photograph. 312 bytes, no padding.

### Code
`game-engine/src/includes/shader_types.h`:
```cpp
#pragma once

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

// C++ mirrors of the structs in the shaders (shaders/*.slang).
// The GPU reads these bytes as they are, so the two sides must agree on every
// size and offset; the static_asserts catch a mismatch at compile time.

// --- Vertex ------------------------------------------------------------------

// Slang lays out data behind a pointer like C: each member aligned only to
// the size of its scalar type. Every member here is made of 4-byte floats,
// so nothing needs padding, and glm agrees member for member. All of these
// structs are packed tight like this: no padding anywhere.
//   - A normal of (0, 0, 0) means the file had none (see mesh.slang).
//   - A tangent of (0, 0, 0, 0) means the file had none; the shader then
//     works the tangent out from the texture coordinates.
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec4 tangent;  // xyz: the direction of +u on the surface; w: +1 or -1, the bitangent's sign
    glm::vec2 uv0;      // TEXCOORD_0
    glm::vec2 uv1;      // TEXCOORD_1
    glm::vec4 color;    // COLOR_0, linear RGBA; white when the file has none
};

static_assert(sizeof(Vertex) == 72);
static_assert(offsetof(Vertex, tangent) == 24);
static_assert(offsetof(Vertex, uv0) == 40);
static_assert(offsetof(Vertex, uv1) == 48);
static_assert(offsetof(Vertex, color) == 56);

// --- Per-draw data -----------------------------------------------------------

// One per draw, in a GPU buffer the shaders index. A draw is placed in a
// world cell (cells.h): its model matrix moves the primitive into the cell,
// measured from the cell's corner. A shadow ray that hits a draw's triangle
// finds the triangle's vertices through first_index and vertex_offset, as
// drawIndexed does. The cull tests the draw's box.
struct DrawData {
    glm::mat4 model;            // this primitive's space -> its cell, from the cell's corner
    glm::mat4 normal_matrix;    // transposed inverse of model: keeps normals perpendicular under any scale
    std::uint32_t material;     // index into the material buffer
    std::uint32_t first_index;  // where the primitive's indices start in the index buffer
    std::int32_t vertex_offset; // added to each index: where its vertices start in the vertex buffer
    glm::ivec3 cell;            // the world cell the draw is placed in
    glm::vec3 bounds_min;       // the draw's box, in its cell, which holds all of its triangles
    glm::vec3 bounds_max;
};

static_assert(sizeof(DrawData) == 176);
static_assert(offsetof(DrawData, normal_matrix) == 64);
static_assert(offsetof(DrawData, material) == 128);
static_assert(offsetof(DrawData, cell) == 140);
static_assert(offsetof(DrawData, bounds_min) == 152);

// --- Materials ---------------------------------------------------------------

// glTF's three ways of using a material's alpha. Each gets its own pipeline,
// and the shader reads the mode as a specialization constant.
enum class AlphaMode : std::uint32_t {
    opaque,  // alpha is ignored
    mask,    // fully opaque or fully transparent: cut out below alpha_cutoff
    blend,   // see-through: blended over what's behind it
};

// Which texture a material slot samples, with which sampler and which set of
// texture coordinates. Heap indices: texture 0 is a 1x1 white texture and
// sampler 0 the default sampler, for slots the file leaves empty.
struct TextureSlot {
    std::uint32_t texture = 0;  // resource heap index
    std::uint32_t sampler = 0;  // sampler heap index
    std::uint32_t uv_set = 0;   // 0: TEXCOORD_0, 1: TEXCOORD_1
};

static_assert(sizeof(TextureSlot) == 12);

// A glTF metallic-roughness material: every factor and texture of the core
// spec. Each texture is multiplied by its factor; see mesh.slang for how.
struct Material {
    glm::vec4 base_color_factor;     // linear RGBA
    glm::vec3 emissive_factor;       // linear RGB light the surface gives off
    float metallic_factor;           // 1: metal, 0: not
    float roughness_factor;          // 1: fully rough, 0: mirror smooth
    float normal_scale;              // how strongly the normal map tilts the normal
    float occlusion_strength;        // 0: ignore the occlusion map, 1: use it fully
    float alpha_cutoff;              // AlphaMode::mask: alpha below this is cut out
    std::uint32_t double_sided;      // 1: both sides are drawn and lit
    AlphaMode alpha_mode;            // for shadow rays, which meet every kind of material
    TextureSlot base_color;          // RGBA, sRGB
    TextureSlot metallic_roughness;  // G: roughness, B: metallic
    TextureSlot normal;              // tangent-space normal
    TextureSlot occlusion;           // R: how much ambient light reaches the surface
    TextureSlot emissive;            // RGB, sRGB
};

static_assert(sizeof(Material) == 116);
static_assert(offsetof(Material, emissive_factor) == 16);
static_assert(offsetof(Material, metallic_factor) == 28);
static_assert(offsetof(Material, double_sided) == 48);
static_assert(offsetof(Material, alpha_mode) == 52);
static_assert(offsetof(Material, base_color) == 56);
static_assert(offsetof(Material, emissive) == 104);

// --- Lights ------------------------------------------------------------------

// KHR_lights_punctual's three kinds of light. "Punctual" means infinitely
// small: all of a light's power comes from one point, or one direction.
enum class LightType : std::uint32_t {
    directional,  // like the sun: parallel rays, intensity in lux
    point,        // shines in every direction, intensity in candela
    spot,         // a point light limited to a cone, intensity in candela
};

// One light from the file, placed in a world cell (cells.h); its direction is
// along the world's axes.
struct Light {
    glm::vec3 offset;     // point and spot lights: where in `cell` the light is (cells.h)
    float range;          // distance where the light fades to nothing; 0 for no limit
    glm::vec3 direction;  // spot and directional lights: the way the light shines
    float spot_scale;     // spot cone falloff: 1 / (cos(inner) - cos(outer))
    glm::vec3 intensity;  // color times intensity
    float spot_offset;    // spot cone falloff: -cos(outer) * spot_scale
    glm::ivec3 cell;      // point and spot lights: the world cell the light is in
    LightType type;
};

static_assert(sizeof(Light) == 64);
static_assert(offsetof(Light, direction) == 16);
static_assert(offsetof(Light, intensity) == 32);
static_assert(offsetof(Light, cell) == 48);
static_assert(offsetof(Light, type) == 60);

// --- Views -------------------------------------------------------------------

// What the fragment shader outputs: the shaded scene, one material input on
// its own, for checking that each one loaded correctly, the ambient
// occlusion, or how much of the sun's light reaches each point. Keys 1-9
// and 0 pick one.
enum class View : std::uint32_t {
    lit,
    base_color,
    normal,         // the final normal, normal map included
    vertex_normal,  // the interpolated vertex normal, without the normal map
    metallic,
    roughness,
    occlusion,
    emissive,
    ambient_occlusion,  // GTAO's visibility: white open, black occluded
    shadow,             // the sun's light that gets through: white all of it, black none
};

// --- The environment -----------------------------------------------------------

// What the environment's compute shaders tell the CPU and the scene shader
// about the sky, in host-visible memory both can read.
//   - irradiance_sh: the light falling on a surface from the whole sky, as 9
//     spherical harmonics coefficients per color channel (see environment.slang).
//   - sun_illuminance: the sun's light at the ground after the atmosphere, in
//     lux on a surface facing it; 0 when the sun is down or the sky is an image.
struct EnvironmentInfo {
    std::array<glm::vec3, 9> irradiance_sh;
    glm::vec3 sun_illuminance;
};

static_assert(sizeof(EnvironmentInfo) == 120);

// --- Per-frame data -----------------------------------------------------------

// Everything the shaders need that's the same for every draw in a frame. Each
// frame in flight has its own copy in host-visible memory, rewritten by the
// CPU before the frame is recorded. Push data points at it.
//   - Lighting values are physical: lux for illuminance, nits (candela per
//     square meter) for the brightness of the sky.
//   - Pointers come right after the matrices, so all of them land on 8-byte
//     boundaries with no padding.
// Shaders work in camera-relative space: the world's axes, with the camera
// at the origin. A position given by a cell and an offset (cells.h) is moved
// into it by subtracting the camera's cell and offset.
struct FrameData {
    glm::mat4 view_projection;          // camera-relative space -> clip space
    glm::mat4 inverse_view_projection;  // clip space -> camera-relative space
    vk::DeviceAddress vertices;         // the scene's vertices
    vk::DeviceAddress indices;          // the scene's indices, for shadow rays' alpha tests
    vk::DeviceAddress draws;            // one DrawData per draw
    vk::DeviceAddress instances;        // the cull's visible draws: what each instance draws
    vk::DeviceAddress materials;        // the scene's materials
    vk::DeviceAddress lights;           // the file's lights
    vk::DeviceAddress environment;      // the EnvironmentInfo
    vk::DeviceAddress scene_tlas;       // the top-level acceleration structure, for ray queries
    glm::ivec3 camera_cell;             // the world cell the camera is in
    float exposure;                     // scales light into the 0..1 range the tone mapper expects
    glm::vec3 sun_direction;            // unit vector pointing toward the sun
    std::uint32_t light_count;          // how many Lights `lights` holds
    glm::vec3 sun_illuminance;          // lux, per color channel, on a surface facing the sun
    View view;                          // what the fragment shader outputs
    std::uint32_t sky_cube;             // resource heap slot: the sky, full detail
    std::uint32_t specular_cube;        // resource heap slot: the sky prefiltered per roughness
    std::uint32_t brdf_lut;             // resource heap slot: the split-sum BRDF table
    std::uint32_t clamp_sampler;        // sampler heap index: trilinear, clamped to the edge
    std::uint32_t specular_mips;        // mip levels of specular_cube: roughness 0 to 1
    float sun_angular_radius;           // radians: half the sun's apparent width
    std::uint32_t ambient_occlusion;    // resource heap slot: the GTAO image, full resolution
    std::uint32_t ao_enabled;           // 0: ignore it, to compare
    glm::vec3 camera_offset;            // where in its cell the camera is
    glm::vec3 tlas_offset;              // the camera, measured from the TLAS's origin (acceleration.h)
    std::uint32_t sky_view;             // resource heap slot: the sky-view table
    std::uint32_t aerial_inscatter;     // resource heap slot: the air's light, as a volume over the view
    std::uint32_t aerial_transmittance; // resource heap slot: the air's transmittance, likewise
    std::uint32_t atmosphere;           // 1: the simulated sky, with its haze; 0: the photograph, without
};

static_assert(sizeof(FrameData) == 312);
static_assert(offsetof(FrameData, vertices) == 128);
static_assert(offsetof(FrameData, instances) == 152);
static_assert(offsetof(FrameData, scene_tlas) == 184);
static_assert(offsetof(FrameData, camera_cell) == 192);
static_assert(offsetof(FrameData, sun_direction) == 208);
static_assert(offsetof(FrameData, sun_illuminance) == 224);
static_assert(offsetof(FrameData, sky_cube) == 240);
static_assert(offsetof(FrameData, sun_angular_radius) == 260);
static_assert(offsetof(FrameData, ambient_occlusion) == 264);
static_assert(offsetof(FrameData, camera_offset) == 272);
static_assert(offsetof(FrameData, tlas_offset) == 284);
static_assert(offsetof(FrameData, sky_view) == 296);

// --- Push data ---------------------------------------------------------------

// Written with vkCmdPushDataEXT before each pipeline's draws: where this
// frame's data is. Which DrawData an instance draws, the vertex shader looks
// up among the cull's instances.
struct PushData {
    vk::DeviceAddress frame;
};

static_assert(sizeof(PushData) == 8);

// --- GPU culling (culling.h, cull.slang) ---------------------------------------

// Draws of one primitive in one draw list, a run of the cull's order: drawn
// as one instanced command, of as many instances as are in view.
struct DrawGroup {
    std::uint32_t first;          // where its draws start in the order
    std::uint32_t count;          // how many draws
    std::uint32_t list;           // its draw list
    std::uint32_t index_count;    // the primitive's drawIndexed arguments
    std::uint32_t first_index;
    std::int32_t vertex_offset;
};

static_assert(sizeof(DrawGroup) == 24);

// A draw list's groups, a run of the group table. Each group has one command
// slot, so it's the list's run of commands too.
struct DrawListRange {
    std::uint32_t first_group = 0;
    std::uint32_t group_count = 0;
};

static_assert(sizeof(DrawListRange) == 8);

// Where all of the cull's buffers are, in one table the steps read.
struct CullTables {
    vk::DeviceAddress order;        // draw indices, by list, then primitive
    vk::DeviceAddress groups;       // one DrawGroup per group
    vk::DeviceAddress lists;        // one DrawListRange per list
    vk::DeviceAddress visible;      // per draw in the order: 1 if in view
    vk::DeviceAddress draw_slots;   // prefix sums of visible, then their total
    vk::DeviceAddress group_flags;  // per group: 1 if any of its draws is in view
    vk::DeviceAddress group_slots;  // prefix sums of group_flags, then their total
    vk::DeviceAddress instances;    // the visible draws' indices
    vk::DeviceAddress commands;     // one VkDrawIndexedIndirectCommand per group
    vk::DeviceAddress counts;       // per list, how many commands to draw
    std::uint32_t draw_count;
    std::uint32_t group_count;
};

static_assert(sizeof(CullTables) == 88);

// Every cull step's push data: the frame (the draws and the view) and the tables.
struct CullPushData {
    vk::DeviceAddress frame;
    vk::DeviceAddress tables;
};

static_assert(sizeof(CullPushData) == 16);

// The tone-mapping pass's push data: which resource heap slot holds the HDR
// image, and the view, so material views can skip tone mapping.
struct TonemapPushData {
    std::uint32_t hdr_image;
    View view;
};

// The transparency composite's push data (composite.slang): the resource heap
// slots of the transparency pass's two sums.
struct CompositePushData {
    std::uint32_t accum;
    std::uint32_t reveal;
};

// The environment compute shaders' push data. Each dispatch sets only what
// its shader reads, so every member has a default.
struct EnvironmentPushData {
    vk::DeviceAddress info = 0;     // where the EnvironmentInfo goes
    std::uint32_t source = 0;       // resource heap slot to read
    std::uint32_t target = 0;       // resource heap slot to write (a storage image)
    std::uint32_t size = 0;         // the target's width and height in texels
    float roughness = 0.0f;         // prefiltering: the roughness this mip level is for
    std::uint32_t sampler = 0;      // sampler heap index: the clamp sampler
    std::uint32_t source_size = 0;  // the source cube's face size at mip 0
};

static_assert(sizeof(EnvironmentPushData) == 32);
static_assert(offsetof(EnvironmentPushData, size) == 16);

// The atmosphere's compute shaders' push data (atmosphere.slang). Each step
// reads what it needs: the per-frame ones the camera's height and the sun,
// the aerial perspective the frame's view. Push data follows std430 rules,
// where a vec3 starts on a 16-byte boundary: the two pointers fill the
// first 16 bytes, so sun_direction lands on one.
struct AtmospherePushData {
    vk::DeviceAddress frame = 0;       // aerial perspective: this frame's FrameData
    vk::DeviceAddress info = 0;        // the sky cube: where the sunlight at the camera goes
    glm::vec3 sun_direction{0.0f};     // toward the sun
    float altitude = 0.0f;             // the camera's height above the ground, in metres
    std::uint32_t transmittance = 0;   // resource heap slot: the transmittance table, sampled
    std::uint32_t multiscatter = 0;    // resource heap slot: the multiple scattering table, sampled
    std::uint32_t target = 0;          // resource heap slot this step writes (storage)
    std::uint32_t second_target = 0;   // aerial perspective: the transmittance volume (storage)
    std::uint32_t size = 0;            // the sky cube's face size
    std::uint32_t sampler = 0;         // sampler heap index: the clamp sampler
};

static_assert(sizeof(AtmospherePushData) == 56);
static_assert(offsetof(AtmospherePushData, sun_direction) == 16);
static_assert(offsetof(AtmospherePushData, transmittance) == 32);

// The ambient occlusion compute shaders' push data (ao.slang), the same for
// all four steps. The half-resolution images and `source` and `target` are
// storage images; each step reads and writes the ones it needs.
struct AoPushData {
    vk::DeviceAddress frame = 0;   // this frame's FrameData
    std::uint32_t depth = 0;       // resource heap slot: the depth buffer, sampled
    std::uint32_t normals = 0;     // resource heap slot: the prepass's normals, sampled
    std::uint32_t ao_depth = 0;    // resource heap slot: half resolution, nearest distance (storage)
    std::uint32_t ao_normals = 0;  // resource heap slot: half resolution, its normal (storage)
    std::uint32_t source = 0;      // resource heap slot: what this step reads (storage)
    std::uint32_t target = 0;      // resource heap slot: what this step writes (storage)
    std::uint32_t width = 0;       // the full-resolution images' size in pixels
    std::uint32_t height = 0;
    float radius = 0.0f;           // meters: how far around a point occluders are looked for
    std::uint32_t slices = 0;      // directions around the view vector
    std::uint32_t steps = 0;       // samples along each direction, each way
    std::uint32_t blur_axis = 0;   // the blur's direction: 0 across, 1 down
};

static_assert(sizeof(AoPushData) == 56);
```

`game-engine/shaders/shared.slangh`:
```slang
// The structs every scene shader shares with C++ (src/includes/shader_types.h),
// and a few helpers, included by mesh.slang, background.slang, ao.slang and
// cull.slang.
// Each of those declares its own push data block. A .slangh file isn't
// compiled on its own: CMakeLists.txt only compiles .slang files.

// Data behind a pointer is laid out like C: each member aligned only to the
// size of its scalar type. Every member here is made of 4-byte floats, so
// there's no padding, and this matches the C++ Vertex exactly (72 bytes).
// The other structs follow the same rule and match theirs.
struct Vertex {
    float3 position;
    float3 normal;   // (0, 0, 0) when the file had no normals
    float4 tangent;  // (0, 0, 0, 0) when the file had no tangents
    float2 uv0;
    float2 uv1;
    float4 color;
};

struct DrawData {
    float4x4 model;          // this primitive's space -> its cell, from the cell's corner
    float4x4 normal_matrix;  // transposed inverse of model
    uint material;           // index into the materials
    uint first_index;        // where the primitive's indices start
    int vertex_offset;       // added to each index
    int3 cell;               // the world cell the draw is placed in
    float3 bounds_min;       // the draw's box, in its cell
    float3 bounds_max;
};

// Which texture, sampler and texture coordinates a material slot uses.
struct TextureSlot {
    uint texture;  // resource heap index; 0 is plain white
    uint sampler;  // sampler heap index; 0 is the default sampler
    uint uv_set;   // 0: TEXCOORD_0, 1: TEXCOORD_1
};

struct Material {
    float4 base_color_factor;
    float3 emissive_factor;
    float metallic_factor;
    float roughness_factor;
    float normal_scale;
    float occlusion_strength;
    float alpha_cutoff;
    uint double_sided;
    uint alpha_mode;                 // 0 opaque, 1 mask, 2 blend
    TextureSlot base_color;          // RGBA
    TextureSlot metallic_roughness;  // G: roughness, B: metallic
    TextureSlot normal;              // tangent-space normal
    TextureSlot occlusion;           // R
    TextureSlot emissive;            // RGB
};

// KHR_lights_punctual light types (LightType in C++).
static const uint light_directional = 0;
static const uint light_point = 1;
static const uint light_spot = 2;

struct Light {
    float3 offset;     // where in its cell the light is
    float range;       // 0: no limit
    float3 direction;  // the way the light shines
    float spot_scale;
    float3 intensity;  // lux (directional) or candela (point, spot), per channel
    float spot_offset;
    int3 cell;         // the world cell the light is in
    uint type;
};

// View: what the fragment shader outputs (keys 1-9, then 0).
static const uint view_lit = 0;
static const uint view_base_color = 1;
static const uint view_normal = 2;
static const uint view_vertex_normal = 3;
static const uint view_metallic = 4;
static const uint view_roughness = 5;
static const uint view_occlusion = 6;
static const uint view_emissive = 7;
static const uint view_ambient_occlusion = 8;
static const uint view_shadow = 9;

// What the environment's compute shaders found out about the sky.
struct EnvironmentInfo {
    float3 irradiance_sh[9];  // diffuse light, as spherical harmonics
    float3 sun_illuminance;   // lux at the ground; 0 for a photographed sky
};

// The same for every draw in a frame. Natural layout, like the C++ struct:
// the pointers land on 8-byte boundaries, right after the matrices.
struct FrameData {
    float4x4 view_projection;          // camera-relative space -> clip space
    float4x4 inverse_view_projection;  // clip space -> camera-relative space
    Vertex* vertices;                  // the scene's vertices
    uint* indices;                     // the scene's indices
    DrawData* draws;                   // one DrawData per draw
    uint* instances;                   // the cull's visible draws: what each instance draws
    Material* materials;               // the scene's materials
    Light* lights;                     // the file's lights
    EnvironmentInfo* environment;      // the sky's diffuse light and the sun
    uint64_t scene_tlas;               // the top-level acceleration structure's address
    int3 camera_cell;                  // the world cell the camera is in
    float exposure;                    // scene nits -> tone mapper input
    float3 sun_direction;              // toward the sun
    uint light_count;
    float3 sun_illuminance;            // lux, facing the sun
    uint view;                         // what to output
    uint sky_cube;                     // resource heap slots: the sky in full detail,
    uint specular_cube;                //   prefiltered per roughness,
    uint brdf_lut;                     //   and the split-sum BRDF table
    uint clamp_sampler;                // sampler heap index
    uint specular_mips;                // mip levels of specular_cube
    float sun_angular_radius;          // radians
    uint ambient_occlusion;            // resource heap slot: the GTAO image
    uint ao_enabled;                   // 0: ignore it
    float3 camera_offset;              // where in its cell the camera is
    float3 tlas_offset;                // the camera, from the TLAS's origin
    uint sky_view;                     // resource heap slot: the sky-view table
    uint aerial_inscatter;             // resource heap slot: the air's light, over the view
    uint aerial_transmittance;         // resource heap slot: the air's transmittance, likewise
    uint atmosphere;                   // 1: the simulated sky and its haze; 0: the photograph
};

// --- Camera-relative positions ---------------------------------------------------

// The side of a world cell, in metres: cells.h's cell_size.
static const float cell_size = 64.0;

// A position given as a world cell and an offset in it (cells.h), relative
// to the camera. The cells are subtracted as integers, exactly; only their
// difference, and the offsets' difference, become floats. Near the camera,
// both are small, and keep a float's full precision anywhere in the world.
float3 camera_relative(FrameData* frame, int3 cell, float3 offset) {
    return float3(cell - frame.camera_cell) * cell_size + (offset - frame.camera_offset);
}

// Written with vkCmdPushDataEXT before each pipeline's draws.
struct PushData {
    FrameData* frame;  // this frame's data
};

// --- Normals in two numbers ------------------------------------------------------

// Octahedral encoding (Meyer et al. 2010): a unit vector is projected onto
// the octahedron |x| + |y| + |z| = 1, whose lower half is folded up over the
// upper; flattened, that's a square, so two numbers in -1..1 hold any
// direction, evenly enough for 16-bit floats.
float2 encode_octahedral(float3 n) {
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    const float2 folded = (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return n.z >= 0.0 ? n.xy : folded;
}

float3 decode_octahedral(float2 e) {
    float3 n = float3(e, 1.0 - abs(e.x) - abs(e.y));
    const float fold = saturate(-n.z);
    n.x += n.x >= 0.0 ? -fold : fold;
    n.y += n.y >= 0.0 ? -fold : fold;
    return normalize(n);
}
```

## 14.4 The environment on the CPU: `environment.h`, `environment.cpp`

### Why
The tables are images like the environment's others, with slots in the resource heap, and the environment builds and updates them.

### How
- **The images:** five, all 16-bit float RGBA, each with a sampled slot and a storage slot. The two aerial perspective volumes are 3D images; `create_flat` takes a depth for them.
- **Once, at startup:** `create_environment` computes the transmittance table, then the multiple scattering table, which reads it. Each ends with a barrier from the compute shader's writes to the compute shaders' reads that follow.
- **When the sun moves (`update_environment`):** in the simulated sky, `skyCubeMain` renders the sky cube for the camera's height; the rest of the update is as before.
- **Every frame (`record_atmosphere`):**
  - **Barriers in:** the per-frame tables are rewritten from scratch, so their old contents can be dropped, from layout `eUndefined`. The frames in flight share them, so the barrier also waits for the previous frame's fragment shaders to finish reading them. A barrier's first scope covers every command submitted before it on the queue, so that includes the previous frame's.
  - **The sky view,** then **the aerial perspective,** which don't depend on each other.
  - **Barriers out,** to the fragment shaders' reads.
  - **Why every frame:** the aerial perspective follows the view, and the sky view the camera's height, which can change every frame. Both are small enough that it costs almost nothing.

### Code
`game-engine/src/includes/environment.h`:
```cpp
#pragma once

#include "includes/buffer.h"
#include "includes/descriptor_heap.h"
#include "includes/shader_types.h"
#include "includes/vulkan_setup.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <filesystem>

// --- The environment ---------------------------------------------------------

// Where the scene's sky comes from.
enum class SkySource {
    atmosphere,  // simulated, for the time of day, with the sun as a light
    photograph,  // an HDR image of a real sky; its sun is already in the image
};

// An image and its memory. There's no VkImageView: shaders reach it through
// descriptor heap slots, written from view descriptions.
struct GpuImage {
    vk::raii::DeviceMemory memory = nullptr;
    vk::raii::Image handle = nullptr;
};

// Every image is 16-bit float RGBA: sky radiance needs the range, up to 65504 nits.
constexpr vk::Format environment_format = vk::Format::eR16G16B16A16Sfloat;

constexpr std::uint32_t sky_cube_size = 512;      // per face, at mip 0
constexpr std::uint32_t sky_cube_mips = 10;       // 512 down to 1
constexpr std::uint32_t specular_cube_size = 128;
constexpr std::uint32_t specular_mips = 6;        // roughness 0, 0.2, ... 1
constexpr std::uint32_t brdf_lut_size = 128;

// The atmosphere's tables (atmosphere.slangh), each 16-bit float RGBA too.
constexpr std::uint32_t transmittance_width = 256;   // by angle
constexpr std::uint32_t transmittance_height = 64;   // by height
constexpr std::uint32_t multiscatter_size = 32;
constexpr std::uint32_t sky_view_width = 192;        // around the up axis
constexpr std::uint32_t sky_view_height = 108;       // from the zenith to the nadir
constexpr std::uint32_t aerial_size = 32;            // across, down and in depth

// The resource heap slots the environment uses, consecutive from `first`:
// sampled descriptors for reading, storage descriptors for compute shaders
// to write.
struct EnvironmentSlots {
    std::uint32_t sky_cube;          // sampled, every mip
    std::uint32_t sky_target;        // storage, mip 0
    std::uint32_t specular_cube;     // sampled, every mip
    std::uint32_t specular_targets;  // storage, one per mip: this slot and the next specular_mips - 1
    std::uint32_t brdf_lut;          // sampled
    std::uint32_t brdf_target;       // storage
    std::uint32_t photograph;        // sampled: the HDR image as loaded

    // The atmosphere's tables, each sampled and as storage.
    std::uint32_t transmittance;
    std::uint32_t transmittance_target;
    std::uint32_t multiscatter;
    std::uint32_t multiscatter_target;
    std::uint32_t sky_view;
    std::uint32_t sky_view_target;
    std::uint32_t aerial_inscatter;
    std::uint32_t aerial_inscatter_target;
    std::uint32_t aerial_transmittance;
    std::uint32_t aerial_transmittance_target;
};

constexpr std::uint32_t environment_slot_count = 6 + specular_mips + 10;

// The sky in its forms for lighting: a full-detail cube map, its diffuse
// light as spherical harmonics (in `info`), a cube prefiltered for specular
// light, and the BRDF table that goes with it. And the atmosphere's tables:
// transmittance and multiple scattering, computed once, and the sky view and
// aerial perspective, every frame.
struct Environment {
    GpuImage sky_cube;
    GpuImage specular_cube;
    GpuImage brdf_lut;
    GpuImage photograph;
    GpuImage transmittance;
    GpuImage multiscatter;
    GpuImage sky_view;
    GpuImage aerial_inscatter;
    GpuImage aerial_transmittance;
    std::uint32_t photograph_width = 0;
    std::uint32_t photograph_height = 0;

    Buffer info;                       // one EnvironmentInfo, host-visible
    EnvironmentInfo* mapped = nullptr;

    vk::raii::Pipeline equirect = nullptr;
    vk::raii::Pipeline irradiance = nullptr;
    vk::raii::Pipeline prefilter = nullptr;
    vk::raii::Pipeline brdf = nullptr;
    vk::raii::Pipeline transmittance_table = nullptr;
    vk::raii::Pipeline multiscatter_table = nullptr;
    vk::raii::Pipeline sky_view_table = nullptr;
    vk::raii::Pipeline aerial = nullptr;
    vk::raii::Pipeline sky_cube_atmosphere = nullptr;

    EnvironmentSlots slots;
    std::uint32_t clamp_sampler = 0;

    // The environment's own command buffer, re-recorded for each update.
    vk::raii::CommandBuffer commands = nullptr;
};

// Creates the images and pipelines, writes their descriptors from slot
// `first_slot` on, loads the HDR photograph at `photograph`, and computes
// what never changes: the BRDF table and the atmosphere's transmittance and
// multiple scattering. Call update_environment before drawing.
Environment create_environment(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::Queue& queue,
    const vk::raii::CommandPool& pool,
    DescriptorHeaps& heaps,
    std::uint32_t first_slot,
    const std::filesystem::path& photograph
);

// Rebuilds the sky from `source`, with the sun toward `sun_direction` and
// the camera `altitude` metres above the ground, then its diffuse and
// specular light. Waits for the GPU before and after, so it must not be
// called while a frame is being recorded.
void update_environment(
    Environment& environment,
    const vk::raii::Device& device,
    const vk::raii::Queue& queue,
    const DescriptorHeaps& heaps,
    SkySource source,
    glm::vec3 sun_direction,
    float altitude
);

// Records the atmosphere's per-frame tables, the sky view and the aerial
// perspective, for the frame whose FrameData is at `frame`. On return both
// are ready for fragment shaders.
void record_atmosphere(
    const vk::raii::CommandBuffer& commands,
    const Environment& environment,
    vk::DeviceAddress frame,
    glm::vec3 sun_direction,
    float altitude
);

// The sky's irradiance, in lux, on a surface facing `normal`: the spherical
// harmonics evaluated on the CPU, for the light meter.
glm::vec3 sky_irradiance(const EnvironmentInfo& info, glm::vec3 normal);
```

`game-engine/src/environment.cpp`:
```cpp
#include "includes/environment.h"

#include "includes/pipeline.h"
#include "includes/radiance_hdr.h"

#include <glm/gtc/packing.hpp>

#include <array>
#include <cstddef>
#include <functional>
#include <limits>

namespace {

// --- Images ------------------------------------------------------------------

GpuImage create_gpu_image(const vk::raii::Device& device, const GpuChoice& gpu, const vk::ImageCreateInfo& info) {
    GpuImage image;
    image.handle = vk::raii::Image(device, info);

    const vk::MemoryRequirements requirements = image.handle.getMemoryRequirements();

    image.memory = vk::raii::DeviceMemory(device, vk::MemoryAllocateInfo{
        .allocationSize = requirements.size,
        .memoryTypeIndex = find_memory_type(gpu, requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
    });

    image.handle.bindMemory(*image.memory, 0);
    return image;
}

// A cube map: a 2D image with 6 array layers that may be viewed as a cube.
GpuImage create_cube(const vk::raii::Device& device, const GpuChoice& gpu, std::uint32_t size, std::uint32_t mips, vk::ImageUsageFlags usage) {
    return create_gpu_image(device, gpu, vk::ImageCreateInfo{
        .flags = vk::ImageCreateFlagBits::eCubeCompatible,
        .imageType = vk::ImageType::e2D,
        .format = environment_format,
        .extent = {size, size, 1},
        .mipLevels = mips,
        .arrayLayers = 6,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
}

// A 2D image, or a 3D one `depth` deep.
GpuImage create_flat(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    std::uint32_t width,
    std::uint32_t height,
    vk::ImageUsageFlags usage,
    std::uint32_t depth = 1
) {
    return create_gpu_image(device, gpu, vk::ImageCreateInfo{
        .imageType = depth > 1 ? vk::ImageType::e3D : vk::ImageType::e2D,
        .format = environment_format,
        .extent = {width, height, depth},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
}

// How shaders see an image: `type` (2D, 2D array, cube, 3D), from mip
// `base_mip`, `mips` levels, all of its layers.
vk::ImageViewCreateInfo view_of(const GpuImage& image, vk::ImageViewType type, std::uint32_t base_mip, std::uint32_t mips, std::uint32_t layers) {
    return vk::ImageViewCreateInfo{
        .image = *image.handle,
        .viewType = type,
        .format = environment_format,
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = base_mip,
            .levelCount = mips,
            .baseArrayLayer = 0,
            .layerCount = layers,
        },
    };
}

// --- Barriers ----------------------------------------------------------------

// Moves mips [base_mip, base_mip + mips) of every layer of `image` between
// layouts, after `src` work and before `dst` work.
void barrier(
    const vk::raii::CommandBuffer& commands,
    const GpuImage& image,
    vk::ImageLayout from,
    vk::ImageLayout to,
    vk::PipelineStageFlags2 src_stage,
    vk::AccessFlags2 src_access,
    vk::PipelineStageFlags2 dst_stage,
    vk::AccessFlags2 dst_access,
    std::uint32_t base_mip = 0,
    std::uint32_t mips = vk::RemainingMipLevels
) {
    const vk::ImageMemoryBarrier2 image_barrier{
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
        .oldLayout = from,
        .newLayout = to,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = *image.handle,
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = base_mip,
            .levelCount = mips,
            .baseArrayLayer = 0,
            .layerCount = vk::RemainingArrayLayers,
        },
    };

    commands.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &image_barrier});
}

// Makes compute shaders' writes to memory, through pointers, visible to the
// `dst` work: the CPU once the submission is done, or later shaders.
void memory_barrier(const vk::raii::CommandBuffer& commands, vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access) {
    const vk::MemoryBarrier2 memory{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
    };

    commands.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &memory});
}

// --- Submitting --------------------------------------------------------------

// Records `record` into the environment's command buffer, submits it and
// waits. The same command buffer is reset and reused every time rather than
// allocated and freed like submit_and_wait's: these commands bind the
// descriptor heaps, and the validation layer (1.4.341) wrongly reports
// conflicting heap ranges after a command buffer that bound them is freed.
void run(
    const Environment& environment,
    const vk::raii::Device& device,
    const vk::raii::Queue& queue,
    const std::function<void(const vk::raii::CommandBuffer&)>& record
) {
    const vk::raii::CommandBuffer& commands = environment.commands;

    commands.reset();
    commands.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    record(commands);
    commands.end();

    const vk::raii::Fence done(device, vk::FenceCreateInfo{});
    const vk::CommandBufferSubmitInfo command_info{.commandBuffer = *commands};

    queue.submit2(vk::SubmitInfo2{
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &command_info,
    }, *done);

    (void)device.waitForFences(*done, vk::True, std::numeric_limits<std::uint64_t>::max());
}

// --- Dispatching -------------------------------------------------------------

// Runs `pipeline` over a `size` x `size` image with `layers` layers, one
// thread per texel, in the shaders' 8 x 8 workgroups.
void dispatch(
    const vk::raii::CommandBuffer& commands,
    const vk::raii::Pipeline& pipeline,
    const EnvironmentPushData& push,
    std::uint32_t layers
) {
    commands.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
    commands.pushDataEXT(vk::PushDataInfoEXT{
        .offset = 0,
        .data = {.address = &push, .size = sizeof(push)},
    });

    const std::uint32_t groups = (push.size + 7) / 8;
    commands.dispatch(groups, groups, layers);
}

// Runs one of atmosphere.slang's steps over `width` x `height` x `depth`
// threads, in its 8 x 8 workgroups.
void dispatch_atmosphere(
    const vk::raii::CommandBuffer& commands,
    const vk::raii::Pipeline& pipeline,
    const AtmospherePushData& push,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t depth = 1
) {
    commands.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
    commands.pushDataEXT(vk::PushDataInfoEXT{
        .offset = 0,
        .data = {.address = &push, .size = sizeof(push)},
    });
    commands.dispatch((width + 7) / 8, (height + 7) / 8, depth);
}

// Fills mips 1 and up of the sky cube from mip 0. Each level is the one above
// shrunk to half size by a linearly filtered blit, all six faces at once.
// On entry mip 0 is eTransferSrcOptimal and the rest eUndefined; on return
// every mip is eShaderReadOnlyOptimal.
void generate_sky_mips(const vk::raii::CommandBuffer& commands, const GpuImage& cube) {
    barrier(commands, cube, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
        vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
        vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferWrite, 1);

    for (std::uint32_t mip = 1; mip < sky_cube_mips; ++mip) {
        const auto from = static_cast<std::int32_t>(sky_cube_size >> (mip - 1));
        const auto to = static_cast<std::int32_t>(sky_cube_size >> mip);

        commands.blitImage(*cube.handle, vk::ImageLayout::eTransferSrcOptimal,
            *cube.handle, vk::ImageLayout::eTransferDstOptimal,
            vk::ImageBlit{
                .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = mip - 1, .baseArrayLayer = 0, .layerCount = 6},
                .srcOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{from, from, 1}},
                .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = mip, .baseArrayLayer = 0, .layerCount = 6},
                .dstOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{to, to, 1}},
            },
            vk::Filter::eLinear);

        // This level is the next blit's source.
        barrier(commands, cube, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal,
            vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferRead, mip, 1);
    }

    barrier(commands, cube, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eTransferRead,
        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eFragmentShader,
        vk::AccessFlagBits2::eShaderSampledRead);
}

// --- The photograph ----------------------------------------------------------

// Loads the HDR file and uploads it as 16-bit floats. glm::packHalf4x16 packs
// four floats into four halves, 8 bytes: one RGBA texel. Halves top out at
// 65504: fine for the file's relative values (this sky's peak is about 33),
// though once scaled to nits its brightest texels pass it, and equirectMain
// clamps them.
void upload_photograph(
    Environment& environment,
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::Queue& queue,
    const vk::raii::CommandPool& pool,
    const std::filesystem::path& path
) {
    const HdrImage image = load_radiance_hdr(path);
    const std::size_t texels = static_cast<std::size_t>(image.width) * image.height;

    const Buffer staging = create_buffer(device, gpu, texels * sizeof(std::uint64_t),
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    auto* halves = static_cast<std::uint64_t*>(staging.memory.mapMemory(0, staging.size));

    for (std::size_t i = 0; i < texels; ++i) {
        const glm::vec4 rgba{image.rgb[i * 3], image.rgb[i * 3 + 1], image.rgb[i * 3 + 2], 1.0f};
        halves[i] = glm::packHalf4x16(rgba);
    }

    staging.memory.unmapMemory();

    environment.photograph_width = image.width;
    environment.photograph_height = image.height;
    environment.photograph = create_flat(device, gpu, image.width, image.height,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled);

    submit_and_wait(device, queue, pool, [&](const vk::raii::CommandBuffer& commands) {
        barrier(commands, environment.photograph, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
            vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
            vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);

        commands.copyBufferToImage(*staging.handle, *environment.photograph.handle, vk::ImageLayout::eTransferDstOptimal,
            vk::BufferImageCopy{
                .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
                .imageExtent = {image.width, image.height, 1},
            });

        barrier(commands, environment.photograph, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderSampledRead);
    });
}

}  // namespace

// --- Creating ----------------------------------------------------------------

Environment create_environment(
    const vk::raii::Device& device,
    const GpuChoice& gpu,
    const vk::raii::Queue& queue,
    const vk::raii::CommandPool& pool,
    DescriptorHeaps& heaps,
    std::uint32_t first_slot,
    const std::filesystem::path& photograph
) {
    Environment environment;
    environment.clamp_sampler = heaps.clamp_sampler;

    // --- Images --------------------------------------------------------------

    // The sky cube is written by a compute shader (storage), shrunk into its
    // mips by blits (transfer), and sampled.
    environment.sky_cube = create_cube(device, gpu, sky_cube_size, sky_cube_mips,
        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled
        | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst);

    environment.specular_cube = create_cube(device, gpu, specular_cube_size, specular_mips,
        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled);

    environment.brdf_lut = create_flat(device, gpu, brdf_lut_size, brdf_lut_size,
        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled);

    // The atmosphere's tables: written by compute shaders, then sampled.
    constexpr vk::ImageUsageFlags table_usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled;
    environment.transmittance = create_flat(device, gpu, transmittance_width, transmittance_height, table_usage);
    environment.multiscatter = create_flat(device, gpu, multiscatter_size, multiscatter_size, table_usage);
    environment.sky_view = create_flat(device, gpu, sky_view_width, sky_view_height, table_usage);
    environment.aerial_inscatter = create_flat(device, gpu, aerial_size, aerial_size, table_usage, aerial_size);
    environment.aerial_transmittance = create_flat(device, gpu, aerial_size, aerial_size, table_usage, aerial_size);

    upload_photograph(environment, device, gpu, queue, pool, photograph);

    // --- Descriptors ---------------------------------------------------------

    // Cubes are sampled as cubes, but written as 2D arrays of six layers,
    // one mip at a time: a storage descriptor covers one mip level.
    EnvironmentSlots& slots = environment.slots;
    slots.sky_cube = first_slot;
    slots.sky_target = first_slot + 1;
    slots.specular_cube = first_slot + 2;
    slots.specular_targets = first_slot + 3;
    slots.brdf_lut = slots.specular_targets + specular_mips;
    slots.brdf_target = slots.brdf_lut + 1;
    slots.photograph = slots.brdf_target + 1;
    slots.transmittance = slots.photograph + 1;
    slots.transmittance_target = slots.transmittance + 1;
    slots.multiscatter = slots.transmittance_target + 1;
    slots.multiscatter_target = slots.multiscatter + 1;
    slots.sky_view = slots.multiscatter_target + 1;
    slots.sky_view_target = slots.sky_view + 1;
    slots.aerial_inscatter = slots.sky_view_target + 1;
    slots.aerial_inscatter_target = slots.aerial_inscatter + 1;
    slots.aerial_transmittance = slots.aerial_inscatter_target + 1;
    slots.aerial_transmittance_target = slots.aerial_transmittance + 1;

    constexpr auto storage = vk::DescriptorType::eStorageImage;

    write_image_descriptor(device, heaps, slots.sky_cube, view_of(environment.sky_cube, vk::ImageViewType::eCube, 0, sky_cube_mips, 6));
    write_image_descriptor(device, heaps, slots.sky_target, view_of(environment.sky_cube, vk::ImageViewType::e2DArray, 0, 1, 6), storage);
    write_image_descriptor(device, heaps, slots.specular_cube, view_of(environment.specular_cube, vk::ImageViewType::eCube, 0, specular_mips, 6));

    for (std::uint32_t mip = 0; mip < specular_mips; ++mip) {
        write_image_descriptor(device, heaps, slots.specular_targets + mip,
            view_of(environment.specular_cube, vk::ImageViewType::e2DArray, mip, 1, 6), storage);
    }

    write_image_descriptor(device, heaps, slots.brdf_lut, view_of(environment.brdf_lut, vk::ImageViewType::e2D, 0, 1, 1));
    write_image_descriptor(device, heaps, slots.brdf_target, view_of(environment.brdf_lut, vk::ImageViewType::e2D, 0, 1, 1), storage);
    write_image_descriptor(device, heaps, slots.photograph, view_of(environment.photograph, vk::ImageViewType::e2D, 0, 1, 1));

    // Each table, sampled and as storage.
    const auto describe_table = [&](const GpuImage& image, vk::ImageViewType type, std::uint32_t sampled, std::uint32_t target) {
        write_image_descriptor(device, heaps, sampled, view_of(image, type, 0, 1, 1));
        write_image_descriptor(device, heaps, target, view_of(image, type, 0, 1, 1), storage);
    };
    describe_table(environment.transmittance, vk::ImageViewType::e2D, slots.transmittance, slots.transmittance_target);
    describe_table(environment.multiscatter, vk::ImageViewType::e2D, slots.multiscatter, slots.multiscatter_target);
    describe_table(environment.sky_view, vk::ImageViewType::e2D, slots.sky_view, slots.sky_view_target);
    describe_table(environment.aerial_inscatter, vk::ImageViewType::e3D, slots.aerial_inscatter, slots.aerial_inscatter_target);
    describe_table(environment.aerial_transmittance, vk::ImageViewType::e3D, slots.aerial_transmittance, slots.aerial_transmittance_target);

    // --- The info buffer and pipelines ---------------------------------------

    // Written by compute shaders through its address, read by the CPU and
    // the scene shader.
    environment.info = create_buffer(device, gpu, sizeof(EnvironmentInfo),
        vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    environment.mapped = static_cast<EnvironmentInfo*>(environment.info.memory.mapMemory(0, sizeof(EnvironmentInfo)));
    *environment.mapped = EnvironmentInfo{};

    environment.equirect = create_compute_pipeline(device, "environment", "equirectMain");
    environment.irradiance = create_compute_pipeline(device, "environment", "irradianceMain");
    environment.prefilter = create_compute_pipeline(device, "environment", "prefilterMain");
    environment.brdf = create_compute_pipeline(device, "environment", "brdfLutMain");
    environment.transmittance_table = create_compute_pipeline(device, "atmosphere", "transmittanceMain");
    environment.multiscatter_table = create_compute_pipeline(device, "atmosphere", "multiscatterMain");
    environment.sky_view_table = create_compute_pipeline(device, "atmosphere", "skyViewMain");
    environment.aerial = create_compute_pipeline(device, "atmosphere", "aerialMain");
    environment.sky_cube_atmosphere = create_compute_pipeline(device, "atmosphere", "skyCubeMain");

    // --- The BRDF table and the atmosphere's fixed tables, once ---------------

    // The pool was created with eResetCommandBuffer, so this one can be
    // re-recorded for every update.
    environment.commands = std::move(vk::raii::CommandBuffers(device, vk::CommandBufferAllocateInfo{
        .commandPool = *pool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    })[0]);

    run(environment, device, queue, [&](const vk::raii::CommandBuffer& commands) {
        bind_descriptor_heaps(commands, heaps);

        barrier(commands, environment.brdf_lut, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
            vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite);

        dispatch(commands, environment.brdf, EnvironmentPushData{.target = slots.brdf_target, .size = brdf_lut_size}, 1);

        barrier(commands, environment.brdf_lut, vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
            vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead);

        // Transmittance, then multiple scattering, which reads it. Both are
        // read by compute shaders from then on: the sky's tables and cube.
        AtmospherePushData atmosphere{
            .transmittance = slots.transmittance,
            .multiscatter = slots.multiscatter,
            .sampler = environment.clamp_sampler,
        };

        for (const GpuImage* table : {&environment.transmittance, &environment.multiscatter}) {
            barrier(commands, *table, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
                vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
                vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite);
        }

        atmosphere.target = slots.transmittance_target;
        dispatch_atmosphere(commands, environment.transmittance_table, atmosphere, transmittance_width, transmittance_height);

        barrier(commands, environment.transmittance, vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderSampledRead);

        atmosphere.target = slots.multiscatter_target;
        dispatch_atmosphere(commands, environment.multiscatter_table, atmosphere, multiscatter_size, multiscatter_size);

        barrier(commands, environment.multiscatter, vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderSampledRead);
    });

    return environment;
}

// --- Updating ----------------------------------------------------------------

void update_environment(
    Environment& environment,
    const vk::raii::Device& device,
    const vk::raii::Queue& queue,
    const DescriptorHeaps& heaps,
    SkySource source,
    glm::vec3 sun_direction,
    float altitude
) {
    // Frames in flight may still be sampling the cubes this rewrites.
    device.waitIdle();

    // A photographed sky has its sun in the picture, so there's no separate
    // sun light. The atmosphere's compute shader writes this itself.
    if (source == SkySource::photograph) {
        environment.mapped->sun_illuminance = glm::vec3{0.0f};
    }

    const EnvironmentSlots& slots = environment.slots;

    const EnvironmentPushData push{
        .info = environment.info.address,
        .source = slots.sky_cube,
        .size = sky_cube_size,
        .sampler = environment.clamp_sampler,
        .source_size = sky_cube_size,
    };

    run(environment, device, queue, [&](const vk::raii::CommandBuffer& commands) {
        bind_descriptor_heaps(commands, heaps);

        // 1. The sky into mip 0 of the sky cube. Its old contents don't matter.
        barrier(commands, environment.sky_cube, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
            vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite, 0, 1);

        EnvironmentPushData sky = push;
        sky.target = slots.sky_target;

        if (source == SkySource::photograph) {
            sky.source = slots.photograph;
            dispatch(commands, environment.equirect, sky, 6);
        } else {
            const AtmospherePushData atmosphere{
                .info = environment.info.address,
                .sun_direction = sun_direction,
                .altitude = altitude,
                .transmittance = slots.transmittance,
                .multiscatter = slots.multiscatter,
                .target = slots.sky_target,
                .size = sky_cube_size,
                .sampler = environment.clamp_sampler,
            };
            dispatch_atmosphere(commands, environment.sky_cube_atmosphere, atmosphere, sky_cube_size, sky_cube_size, 6);
        }

        // 2. Its mip chain, by blits.
        barrier(commands, environment.sky_cube, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
            vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferRead, 0, 1);

        generate_sky_mips(commands, environment.sky_cube);

        // 3. Diffuse light: one workgroup sums the sky into nine coefficients.
        commands.bindPipeline(vk::PipelineBindPoint::eCompute, *environment.irradiance);
        commands.pushDataEXT(vk::PushDataInfoEXT{.offset = 0, .data = {.address = &push, .size = sizeof(push)}});
        commands.dispatch(1, 1, 1);

        // 4. Specular light: each mip level of the specular cube for its roughness.
        barrier(commands, environment.specular_cube, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
            vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite);

        for (std::uint32_t mip = 0; mip < specular_mips; ++mip) {
            EnvironmentPushData level = push;
            level.target = slots.specular_targets + mip;
            level.size = specular_cube_size >> mip;
            level.roughness = static_cast<float>(mip) / static_cast<float>(specular_mips - 1);
            dispatch(commands, environment.prefilter, level, 6);
        }

        barrier(commands, environment.specular_cube, vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
            vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead);

        // The coefficients and the sun went into the info buffer: make them
        // visible to the CPU, which reads them once this submission is done,
        // and to the scene's fragment shaders, which read them every frame.
        // Waiting for the fence orders the work, but doesn't by itself make
        // one shader's writes visible to another's reads.
        memory_barrier(commands,
            vk::PipelineStageFlagBits2::eHost | vk::PipelineStageFlagBits2::eFragmentShader,
            vk::AccessFlagBits2::eHostRead | vk::AccessFlagBits2::eShaderStorageRead);
    });
}

// --- The atmosphere, every frame ---------------------------------------------

void record_atmosphere(
    const vk::raii::CommandBuffer& commands,
    const Environment& environment,
    vk::DeviceAddress frame,
    glm::vec3 sun_direction,
    float altitude
) {
    const EnvironmentSlots& slots = environment.slots;

    AtmospherePushData push{
        .frame = frame,
        .sun_direction = sun_direction,
        .altitude = altitude,
        .transmittance = slots.transmittance,
        .multiscatter = slots.multiscatter,
        .sampler = environment.clamp_sampler,
    };

    // Rewritten from scratch every frame. Frames in flight share them, so
    // this also waits for the previous frame's fragment shaders to finish
    // reading them.
    for (const GpuImage* table : {&environment.sky_view, &environment.aerial_inscatter, &environment.aerial_transmittance}) {
        barrier(commands, *table, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
            vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite);
    }

    // The sky view: one thread per texel.
    push.target = slots.sky_view_target;
    dispatch_atmosphere(commands, environment.sky_view_table, push, sky_view_width, sky_view_height);

    // Aerial perspective: one thread per column of the volume, marching
    // through its slices.
    push.target = slots.aerial_inscatter_target;
    push.second_target = slots.aerial_transmittance_target;
    dispatch_atmosphere(commands, environment.aerial, push, aerial_size, aerial_size);

    // The background reads the sky view, and the scene's shaders the volumes.
    for (const GpuImage* table : {&environment.sky_view, &environment.aerial_inscatter, &environment.aerial_transmittance}) {
        barrier(commands, *table, vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
            vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead);
    }
}

// --- Reading the irradiance on the CPU ---------------------------------------

glm::vec3 sky_irradiance(const EnvironmentInfo& info, glm::vec3 n) {
    // The same nine basis functions as environment.slang.
    const float basis[9] = {
        0.282095f,
        0.488603f * n.y,
        0.488603f * n.z,
        0.488603f * n.x,
        1.092548f * n.x * n.y,
        1.092548f * n.y * n.z,
        0.315392f * (3.0f * n.z * n.z - 1.0f),
        1.092548f * n.x * n.z,
        0.546274f * (n.x * n.x - n.y * n.y),
    };

    glm::vec3 irradiance{0.0f};
    for (std::size_t k = 0; k < 9; ++k) {
        irradiance += info.irradiance_sh[k] * basis[k];
    }

    return glm::max(irradiance, glm::vec3{0.0f});
}
```

## 14.5 Seeing through the air: `mesh.slang`, `background.slang`

### Why
The scene's shaders apply the aerial perspective and the energy compensation; the background draws the sky from the sky-view table.

### How
- **The aerial perspective (`aerial_perspective`):**
  - **Where to read:** the point's place on the screen, from the view-projection, and its depth coordinate in the volume, from its distance to the camera.
  - **What it does:** the surface's light, emission included, is multiplied by the transmittance, and the air's own light is added: `radiance × transmittance + inscatter`. Hillaire's composition, the same as the sky's.
  - **With the photograph,** transmittance is 1 and there's nothing to add.
- **Energy compensation (`energy_compensation`):**
  - **What's missing:** the BRDF table's two numbers (Chapter 8), at F0 = 1, add up to `E`, the share of light a perfectly reflective surface sends back after one bounce. `1 − E` is what the single bounce misses: almost nothing on a smooth surface, and on a fully rough one, about 70% seen head-on, down to about 15% near grazing angles.
  - **The fix,** Turquin's (2019) form of Kulla and Conty's compensation, as Filament uses it: scale the specular reflection by `1 + F0 × (1 / E − 1)`. A surface that reflects everything (F0 = 1) gets all of `1 − E` back; one that reflects 4% of the light, a dielectric, gets only a little.
  - **Where:** in `Surface`, applied to the direct specular in `shade` and the environment's specular in `shade_environment`.
- **The background:** with the simulated sky, from the sky-view table, which holds more detail near the horizon than the cube does; with the photograph, from the cube, as before. The camera's height is its cell's height plus its offset's.

### Code
`game-engine/shaders/mesh.slang`:
```slang
// Draws one glTF primitive: its vertices come from the scene's vertex buffer,
// its place in the world from its DrawData, and its surface from its glTF
// material, whose textures are read from the descriptor heap. Shaded with
// glTF's physically based BRDF, lit by the sun and the file's lights, with
// ray-traced shadows, and by the sky around the scene, already exposed.
// Three fragment shaders:
//   prepassMain      the depth prepass's vertex normal
//   fragmentMain     opaque and masked surfaces, into the HDR image
//   transparentMain  blended surfaces, into weighted blended transparency's sums

// --- Data shared with C++ (src/includes/shader_types.h) ----------------------

#include "shared.slangh"
#include "atmosphere.slangh"

// In a descriptor heap pipeline, the push_constant block is where push data lands.
[[vk::push_constant]]
ConstantBuffer<PushData> push;

// The alpha mode this pipeline was built for (AlphaMode in C++):
// 0 opaque, 1 mask, 2 blend. A specialization constant: its value is
// fixed when the pipeline is created, so each pipeline's fragment shader
// keeps only the code its mode needs.
[vk::constant_id(0)]
const uint alpha_mode = 0;

static const uint alpha_opaque = 0;
static const uint alpha_mask = 1;
static const uint alpha_blend = 2;

// --- Stage interface ---------------------------------------------------------

// What the vertex shader hands to the rasterizer. SV_Position is the
// clip-space position; every other field but draw_index is interpolated
// across the triangle. Vulkan requires integer fields to be flat, which
// nointerpolation makes them.
struct VertexOutput {
    float4 position : SV_Position;
    float3 relative_position : POSITION;  // camera-relative: the world's axes, the camera at the origin
    float3 normal : NORMAL;
    float3 tangent : TANGENT;
    float3 bitangent : BINORMAL;
    float2 uv0 : TEXCOORD0;
    float2 uv1 : TEXCOORD1;
    float4 color : COLOR;
    nointerpolation uint draw_index : DRAW_INDEX;  // the same for a whole triangle, so never interpolated
};

// --- Vertex shader -----------------------------------------------------------

// SV_VulkanVertexID is Vulkan's own gl_VertexIndex, which includes the draw's
// vertexOffset: each primitive's indices start at 0, and the draw adds where
// that primitive's vertices begin in the shared buffer. SV_VulkanInstanceID
// is gl_InstanceIndex, which likewise counts from the command's
// firstInstance: the cull points that at the command's run of visible draws
// in `instances`, so each instance finds its draw there.
[shader("vertex")]
VertexOutput vertexMain(uint vertex_id : SV_VulkanVertexID, uint instance : SV_VulkanInstanceID) {
    FrameData* frame = push.frame;
    const Vertex vertex = frame.vertices[vertex_id];
    const uint draw_index = frame.instances[instance];
    const DrawData draw = frame.draws[draw_index];

    // The vertex in its draw's cell, then relative to the camera.
    const float3 in_cell = mul(draw.model, float4(vertex.position, 1.0)).xyz;
    const float3 relative_position = camera_relative(frame, draw.cell, in_cell);

    // Tangent and bitangent lie along the surface, so they move with the
    // model matrix, like positions; only the normal needs the normal matrix.
    // The bitangent is built before the transform, from glTF's rule
    // B = cross(N, T) * w: a mirroring transform then mirrors it too.
    const float3 bitangent = cross(vertex.normal, vertex.tangent.xyz) * vertex.tangent.w;

    VertexOutput output;
    output.position = mul(frame.view_projection, float4(relative_position, 1.0));
    output.relative_position = relative_position;
    output.normal = mul((float3x3)draw.normal_matrix, vertex.normal);
    output.tangent = mul((float3x3)draw.model, vertex.tangent.xyz);
    output.bitangent = mul((float3x3)draw.model, bitangent);
    output.uv0 = vertex.uv0;
    output.uv1 = vertex.uv1;
    output.color = vertex.color;
    output.draw_index = draw_index;
    return output;
}

// --- Material textures -------------------------------------------------------

// Samples a material slot: its texture, with its sampler, at its set of
// texture coordinates. Descriptor heap access: a handle made from an index
// reads that descriptor from the bound heap.
float4 sample_slot(TextureSlot slot, VertexOutput input) {
    const Texture2D texture = Texture2D.Handle(uint2(slot.texture, 0));
    const SamplerState sampler = SamplerState.Handle(uint2(slot.sampler, 0));
    const float2 uv = slot.uv_set == 0 ? input.uv0 : input.uv1;
    return texture.Sample(sampler, uv);
}

// --- Normals -----------------------------------------------------------------

// The direction the surface faces at this pixel, for lighting.
//   1. The interpolated vertex normal. Without normals in the file, glTF asks
//      for flat shading: the triangle's own normal is the cross product of
//      how the position changes across neighbouring pixels (ddx, ddy).
//   2. A normal map tilts it, per texel, within the surface's tangent frame.
//   3. On a double-sided material's back face, the surface faces the other way.
float3 surface_normal(VertexOutput input, Material material, bool front_face, bool apply_normal_map) {
    float3 normal = input.normal;

    // cross(ddy, ddx), not cross(ddx, ddy): Vulkan's screen Y points down,
    // so this order is the one that points toward the camera.
    if (all(normal == 0.0)) {
        normal = cross(ddy(input.relative_position), ddx(input.relative_position));
    }

    normal = normalize(normal);

    // Texture 0 is white: no normal map.
    const bool mapped = apply_normal_map && material.normal.texture != 0;

    float3 tangent = input.tangent;
    float3 bitangent = input.bitangent;

    // Without tangents in the file, work the frame out from how position and
    // texture coordinates change between neighbouring pixels (ddx, ddy):
    //     dp/dx = P_u * du/dx + P_v * dv/dx
    //     dp/dy = P_u * du/dy + P_v * dv/dy
    // Solving these for P_u and P_v, how position changes per unit of u and v,
    // gives the tangent (+u) and bitangent. glTF's v runs down the image while
    // a normal map's +Y points up, so the bitangent is -P_v. Only the
    // directions matter, so the determinant's sign stands in for dividing by
    // it. This can differ slightly from the MikkTSpace tangents glTF
    // specifies, but needs no precomputation.
    if (mapped && all(tangent == 0.0)) {
        const float2 uv = material.normal.uv_set == 0 ? input.uv0 : input.uv1;
        const float3 dp_dx = ddx(input.relative_position);
        const float3 dp_dy = ddy(input.relative_position);
        const float2 duv_dx = ddx(uv);
        const float2 duv_dy = ddy(uv);

        const float determinant = duv_dx.x * duv_dy.y - duv_dy.x * duv_dx.y;
        const float orientation = determinant < 0.0 ? -1.0 : 1.0;

        tangent = (dp_dx * duv_dy.y - dp_dy * duv_dx.y) * orientation;
        bitangent = -(dp_dy * duv_dx.x - dp_dx * duv_dy.x) * orientation;
    }

    // The frame's three axes flip together, keeping the map's tilt correct.
    if (material.double_sided != 0 && !front_face) {
        normal = -normal;
        tangent = -tangent;
        bitangent = -bitangent;
    }

    // Texture coordinates that don't change across the triangle give no frame
    // at all; the plain normal is all we have then.
    if (!mapped || all(tangent == 0.0) || all(bitangent == 0.0)) {
        return normal;
    }

    // All three axes must be unit length, or the map's tilt is scaled with
    // them. The normal already is; the other two grow and shrink with the
    // model matrix, and interpolation shortens them between vertices.
    tangent = normalize(tangent);
    bitangent = normalize(bitangent);

    // The map stores each component in 0..1; unpack to -1..1. normal_scale
    // scales the tilt: X and Y only, as glTF specifies.
    float3 tangent_space = sample_slot(material.normal, input).xyz * 2.0 - 1.0;
    tangent_space.xy *= material.normal_scale;

    return normalize(tangent * tangent_space.x + bitangent * tangent_space.y + normal * tangent_space.z);
}

// --- The glTF BRDF -----------------------------------------------------------

// glTF's metallic-roughness model, as its specification's Appendix B writes
// it. A BRDF says how much of the light arriving from one direction leaves
// toward another: here from the light (l) toward the viewer (v), around the
// half vector h between them.

static const float pi = 3.14159265;

// What shading needs to know about the surface at this pixel.
struct Surface {
    float3 base_color;
    float metallic;
    float alpha;   // roughness squared: the "alpha" of GGX
    float3 normal;
    float3 view;   // unit vector toward the camera
    float3 energy_compensation;  // what the specular reflection is scaled by (energy_compensation)
};

// D: the GGX (Trowbridge-Reitz) distribution of microfacet normals. Smooth
// surfaces have nearly all their tiny facets aligned with the normal, so D
// is a tall, narrow peak around h = n; rough ones spread it out.
float distribution_ggx(float n_dot_h, float alpha) {
    const float alpha2 = alpha * alpha;
    const float f = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
    return alpha2 / (pi * f * f);
}

// V: Smith's height-correlated visibility, the share of facets neither in
// shadow nor hidden, with the BRDF's 1 / (4 n.l n.v) folded in.
float visibility_smith(float n_dot_l, float n_dot_v, float alpha) {
    const float alpha2 = alpha * alpha;
    const float from_view = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - alpha2) + alpha2);
    const float from_light = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - alpha2) + alpha2);
    const float sum = from_view + from_light;
    return sum > 0.0 ? 0.5 / sum : 0.0;
}

// The light leaving toward the viewer, in nits, from light arriving from
// direction `l` with illuminance `illuminance` (lux, on a surface facing it).
float3 shade(Surface surface, float3 l, float3 illuminance) {
    const float n_dot_l = dot(surface.normal, l);

    if (n_dot_l <= 0.0) {
        return float3(0.0);
    }

    const float3 h = normalize(l + surface.view);
    const float n_dot_v = max(dot(surface.normal, surface.view), 1e-4);
    const float n_dot_h = max(dot(surface.normal, h), 0.0);
    const float v_dot_h = max(dot(surface.view, h), 0.0);

    // Mirror-like reflection off the facets: white, for every material.
    const float specular = visibility_smith(n_dot_l, n_dot_v, surface.alpha) * distribution_ggx(n_dot_h, surface.alpha);

    // Schlick's Fresnel: every surface reflects more at grazing angles.
    const float fresnel = pow(1.0 - v_dot_h, 5.0);

    // Metals tint their reflection with the base color and have no diffuse
    // part. Dielectrics (everything else) reflect 4% head-on, rising to 100%
    // at grazing angles, and the rest enters the surface and scatters back
    // out as Lambertian diffuse light, colored by the base color.
    const float3 compensated = specular * surface.energy_compensation;
    const float3 metal = compensated * (surface.base_color + (1.0 - surface.base_color) * fresnel);
    const float3 dielectric = lerp(surface.base_color / pi, compensated, 0.04 + 0.96 * fresnel);
    const float3 brdf = lerp(dielectric, metal, surface.metallic);

    // Light falling at an angle spreads over more surface: the n.l factor.
    return brdf * illuminance * n_dot_l;
}

// --- Shadows -------------------------------------------------------------------

// How far a ray toward the sun, or another light infinitely far away, may go.
static const float infinite_distance = 1e9;

// A ray starting exactly on a surface can hit that same surface: the hit
// point's rounding puts it a hair below. This moves the origin off the
// surface along its geometric normal by up to 256 units in the last place
// (ULPs) of each coordinate: an offset that grows with the coordinates, so
// it suits any distance from the origin, where a fixed distance would be too
// much near it and too little far away. The constants are the authors',
// found by experiment. From "A Fast and Robust Method for Avoiding
// Self-Intersection" (Wachter and Binder, Ray Tracing Gems, 2019).
float3 offset_ray_origin(float3 position, float3 normal) {
    const float near_origin = 1.0 / 32.0;
    const float float_scale = 1.0 / 65536.0;
    const float int_scale = 256.0;

    float3 offset;
    for (int axis = 0; axis < 3; ++axis) {
        // Step the float's bits, as an integer, away from the surface.
        const int step = int(int_scale * normal[axis]);
        const float stepped = asfloat(asint(position[axis]) + (position[axis] < 0.0 ? -step : step));

        // Close to 0 a few units in the last place are tiny, so add a small
        // fixed distance there instead.
        offset[axis] = abs(position[axis]) < near_origin ? position[axis] + float_scale * normal[axis] : stepped;
    }

    return offset;
}

// The alpha of a masked or blended triangle a ray met, at the hit point. The
// hit's barycentric coordinates weight the triangle's three vertices; the
// texture is read at full resolution, since there are no neighbouring pixels
// to pick a mip level from. Rays from neighbouring pixels can hit different
// materials, so the texture's heap index differs between them: that's fine,
// since descriptor heap access is non-uniform unless the SPIR-V marks it
// uniform, and Slang doesn't.
float candidate_alpha(FrameData* frame, DrawData draw, Material material, uint triangle, float2 barycentrics) {
    const uint first = draw.first_index + triangle * 3;
    const Vertex v0 = frame.vertices[int(frame.indices[first]) + draw.vertex_offset];
    const Vertex v1 = frame.vertices[int(frame.indices[first + 1]) + draw.vertex_offset];
    const Vertex v2 = frame.vertices[int(frame.indices[first + 2]) + draw.vertex_offset];
    const float3 weight = float3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);

    const float2 uv = material.base_color.uv_set == 0
        ? v0.uv0 * weight.x + v1.uv0 * weight.y + v2.uv0 * weight.z
        : v0.uv1 * weight.x + v1.uv1 * weight.y + v2.uv1 * weight.z;
    const float vertex_alpha = v0.color.a * weight.x + v1.color.a * weight.y + v2.color.a * weight.z;

    const Texture2D texture = Texture2D.Handle(uint2(material.base_color.texture, 0));
    const SamplerState sampler = SamplerState.Handle(uint2(material.base_color.sampler, 0));
    return material.base_color_factor.a * vertex_alpha * texture.SampleLevel(sampler, uv, 0.0).a;
}

// How much of a light gets from `origin` to `distance` along `direction`: 0
// when something solid is in the way, otherwise the share every see-through
// layer on the way lets through. A ray query walks the TLAS and BLASes:
//   - an opaque triangle ends it at once: any blocking hit will do,
//   - a masked one comes back as a candidate, which blocks where its alpha
//     reaches the cutoff, and lets the light through its cut-out texels,
//   - a blended one comes back as a candidate that lets 1 - alpha of the
//     light through, as the transparency pass's reveal sum does. It never
//     ends the ray: the light goes on, dimmed, to whatever is behind.
float light_visibility(FrameData* frame, float3 origin, float3 direction, float distance) {
    const RaytracingAccelerationStructure scene = RaytracingAccelerationStructure(frame.scene_tlas);

    RayDesc ray;
    ray.Origin = origin;
    ray.TMin = 0.0;
    ray.Direction = direction;
    ray.TMax = distance;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, ray);

    float transmittance = 1.0;

    while (query.Proceed()) {
        if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) {
            continue;
        }

        const DrawData draw = frame.draws[query.CandidateInstanceID()];
        const Material material = frame.materials[draw.material];
        const float alpha = candidate_alpha(frame, draw, material, query.CandidatePrimitiveIndex(),
            query.CandidateTriangleBarycentrics());

        if (material.alpha_mode == alpha_blend) {
            transmittance *= 1.0 - saturate(alpha);
        } else if (alpha >= material.alpha_cutoff) {
            query.CommitNonOpaqueTriangleHit();
        }
    }

    return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 0.0 : transmittance;
}

// Where a ray toward the light `l` starts: off the surface on the light's
// side, along `face_normal`, the triangle's own flat normal. An interpolated
// or normal-mapped normal can disagree about which side the light is on.
//
// The TLAS's space is measured from its origin cell, near the camera, not
// from the camera itself (acceleration.h): tlas_offset, the camera's
// position in it, moves the camera-relative position there first, so the
// offset grows with the coordinates the ray is really traced at.
float3 shadow_ray_origin(FrameData* frame, float3 position, float3 face_normal, float3 l) {
    return offset_ray_origin(position + frame.tlas_offset, dot(face_normal, l) >= 0.0 ? face_normal : -face_normal);
}

// shade(), times how much of the light gets through. A ray is only traced
// when the light could reach the surface at all.
float3 shade_shadowed(
    Surface surface, FrameData* frame, float3 position, float3 face_normal,
    float3 l, float3 illuminance, float distance
) {
    if (dot(surface.normal, l) <= 0.0 || all(illuminance == 0.0)) {
        return float3(0.0);
    }

    const float reaching = light_visibility(frame, shadow_ray_origin(frame, position, face_normal, l), l, distance);
    return reaching > 0.0 ? shade(surface, l, illuminance) * reaching : float3(0.0);
}

// --- Lights --------------------------------------------------------------------

// The direction toward a light, how far away it is, and the illuminance it
// gives here, following KHR_lights_punctual. Point and spot lights fade with
// the square of the distance, then smoothly to nothing at `range`; spot
// lights also fade from the inner cone to the outer one. A directional
// light is infinitely far away.
float3 punctual_light(FrameData* frame, Light light, float3 position, out float3 l, out float distance) {
    if (light.type == light_directional) {
        l = -light.direction;
        distance = infinite_distance;
        return light.intensity;
    }

    const float3 to_light = camera_relative(frame, light.cell, light.offset) - position;
    const float distance2 = max(dot(to_light, to_light), 1e-8);
    distance = sqrt(distance2);
    l = to_light / distance;

    float attenuation = 1.0 / distance2;

    if (light.range > 0.0) {
        const float ratio = distance / light.range;
        attenuation *= saturate(1.0 - ratio * ratio * ratio * ratio);
    }

    if (light.type == light_spot) {
        const float cone = saturate(dot(light.direction, -l) * light.spot_scale + light.spot_offset);
        attenuation *= cone * cone;
    }

    return light.intensity * attenuation;
}

// --- Image-based lighting ------------------------------------------------------

// Light from the whole sky at once, from what environment.slang prepared.

// The sky's irradiance on a surface facing `n`, in lux: its nine spherical
// harmonics coefficients, each weighted by its basis function at `n`.
float3 sky_irradiance(EnvironmentInfo* environment, float3 n) {
    const float basis[9] = {
        0.282095,
        0.488603 * n.y,
        0.488603 * n.z,
        0.488603 * n.x,
        1.092548 * n.x * n.y,
        1.092548 * n.y * n.z,
        0.315392 * (3.0 * n.z * n.z - 1.0),
        1.092548 * n.x * n.z,
        0.546274 * (n.x * n.x - n.y * n.y),
    };

    float3 irradiance = 0.0;
    for (uint k = 0; k < 9; ++k) {
        irradiance += environment.irradiance_sh[k] * basis[k];
    }

    return max(irradiance, 0.0);
}

// --- Ambient occlusion -------------------------------------------------------------

// The rotation that turns `from` into `to`, applied to `v`: Rodrigues'
// formula rewritten without angles (Moller and Hughes 1999), from the cross
// and dot products alone. `from` and `to` are never opposite here: the bent
// normal averages directions in the hemisphere around the normal.
float3 rotate_from_to(float3 from, float3 to, float3 v) {
    const float3 axis = cross(from, to);
    const float c = dot(from, to);

    if (c > 0.9999) {
        return v;
    }

    return v * c + cross(axis, v) + axis * (dot(axis, v) / (1.0 + c));
}

// Ambient occlusion counts light that's blocked, but light also bounces off
// the occluders, and more so the brighter they are. Jimenez et al.'s fit,
// from the same GTAO paper, brightens the visibility by the surface's own
// albedo, standing in for its surroundings'.
float3 multi_bounce(float visibility, float3 albedo) {
    const float3 a = 2.0404 * albedo - 0.3324;
    const float3 b = -4.7951 * albedo + 0.6417;
    const float3 c = 2.7552 * albedo + 0.6903;
    return max(float3(visibility), ((visibility * a + b) * visibility + c) * visibility);
}

// How much of the sky's reflection a partly occluded point still sees
// (Lagarde and de Rousiers 2014): smooth surfaces, looking straight on, keep
// more of it than occlusion alone suggests; rough ones lose about as much.
// Its roughness is GGX's alpha, roughness squared.
float specular_occlusion(float n_dot_v, float visibility, float alpha) {
    return saturate(pow(n_dot_v + visibility, exp2(-16.0 * alpha - 1.0)) - 1.0 + visibility);
}

// The sky's light reflected toward the viewer.
//   - Diffuse: a Lambertian surface reflects base color / pi of the
//     irradiance falling on it, read along `irradiance_normal` (the bent
//     normal: the direction the open sky lies in), dimmed by the visibility
//     and brightened again by multiple bounces.
//   - Specular, the "split sum": the light (the prefiltered sky along the
//     reflected ray, at the mip level for this roughness) times how much
//     the BRDF reflects overall (the table, as a scale and bias on F0),
//     dimmed by the specular occlusion.
//   - A roughness-aware Fresnel term splits the light between the two.
float3 shade_environment(Surface surface, FrameData* frame, float roughness, float visibility, float3 irradiance_normal) {
    const float n_dot_v = max(dot(surface.normal, surface.view), 1e-4);
    const float3 f0 = lerp(float3(0.04), surface.base_color, surface.metallic);
    const float3 fresnel = f0 + (max(float3(1.0 - roughness), f0) - f0) * pow(1.0 - n_dot_v, 5.0);

    const SamplerState clamped = SamplerState.Handle(uint2(frame.clamp_sampler, 0));

    const float3 diffuse_color = surface.base_color * (1.0 - surface.metallic);
    const float3 diffuse = (1.0 - fresnel) * diffuse_color * sky_irradiance(frame.environment, irradiance_normal) / pi
        * multi_bounce(visibility, diffuse_color);

    const TextureCube specular_cube = TextureCube.Handle(uint2(frame.specular_cube, 0));
    const float3 reflected = reflect(-surface.view, surface.normal);
    const float lod = roughness * float(frame.specular_mips - 1);
    const float3 prefiltered = specular_cube.SampleLevel(clamped, reflected, lod).rgb;

    const Texture2D brdf_lut = Texture2D.Handle(uint2(frame.brdf_lut, 0));
    const float2 brdf = brdf_lut.SampleLevel(clamped, float2(n_dot_v, roughness), 0.0).rg;
    const float3 specular = prefiltered * (f0 * brdf.x + brdf.y) * specular_occlusion(n_dot_v, visibility, surface.alpha)
        * surface.energy_compensation;

    return diffuse + specular;
}

// --- Energy compensation -------------------------------------------------------------

// The GGX specular reflection counts light that bounces once off the
// microfacets; on a rough surface, much of it bounces again, between them,
// and still leaves. Without it, rough metals come out too dark: a fully
// rough white metal reflects only about 40% of the light. The BRDF
// table's two numbers, at f0 = 1, add up to E, the share that one bounce
// leaves with; scaling the reflection by 1 + f0 (1 / E - 1) puts the rest
// back, in proportion to how much the surface reflects at all. Kulla and
// Conty's (2017) idea, in Turquin's (2019) simpler scaled form, which
// Filament uses.
float3 energy_compensation(FrameData* frame, float3 base_color, float metallic, float roughness, float n_dot_v) {
    const SamplerState clamped = SamplerState.Handle(uint2(frame.clamp_sampler, 0));
    const Texture2D brdf_lut = Texture2D.Handle(uint2(frame.brdf_lut, 0));
    const float2 brdf = brdf_lut.SampleLevel(clamped, float2(n_dot_v, roughness), 0.0).rg;
    const float3 f0 = lerp(float3(0.04), base_color, metallic);
    return 1.0 + f0 * (1.0 / max(brdf.x + brdf.y, 1e-3) - 1.0);
}

// --- Aerial perspective --------------------------------------------------------------

// The light the air between the camera and `position` (camera-relative)
// adds, and, in `transmittance`, the share of the surface's light it lets
// through: the aerial perspective volumes, at the point's place on the
// screen and its distance. Closer than the first slice's far edge, a share
// of the first slice's air.
float3 aerial_perspective(FrameData* frame, float3 position, out float3 transmittance) {
    const float4 clip = mul(frame.view_projection, float4(position, 1.0));
    const float2 screen = clip.xy / clip.w * 0.5 + 0.5;

    float first_slice_share;
    const float depth = aerial_depth(length(position), first_slice_share);
    const float3 coordinate = float3(screen, depth);

    const SamplerState clamped = SamplerState.Handle(uint2(frame.clamp_sampler, 0));
    const float3 inscatter = Texture3D.Handle(uint2(frame.aerial_inscatter, 0)).SampleLevel(clamped, coordinate, 0.0).rgb;
    const float3 through = Texture3D.Handle(uint2(frame.aerial_transmittance, 0)).SampleLevel(clamped, coordinate, 0.0).rgb;

    transmittance = lerp(float3(1.0), through, first_slice_share);
    return inscatter * first_slice_share;
}

// --- Depth and normal prepass ------------------------------------------------------

// The interpolated vertex normal, facing the viewer on a double-sided
// material's back face; flat when the file has no normals. This is the
// surface at the scale the mesh describes it, which ambient occlusion
// searches against: a normal map's detail isn't in the depth buffer.
float3 vertex_normal(VertexOutput input, Material material, bool front_face) {
    float3 normal = input.normal;

    if (all(normal == 0.0)) {
        normal = cross(ddy(input.relative_position), ddx(input.relative_position));
    }

    normal = normalize(normal);
    return material.double_sided != 0 && !front_face ? -normal : normal;
}

// The prepass draws every opaque and masked surface first, writing only its
// depth and its vertex normal, octahedrally encoded. Masked surfaces cut out
// their transparent texels here too, so the depth buffer holds exactly the
// surfaces the lighting pass will shade.
[shader("fragment")]
float2 prepassMain(VertexOutput input, bool front_face : SV_IsFrontFace) : SV_Target {
    FrameData* frame = push.frame;
    const Material material = frame.materials[frame.draws[input.draw_index].material];

    if (alpha_mode == alpha_mask) {
        const float alpha = material.base_color_factor.a * sample_slot(material.base_color, input).a * input.color.a;
        if (alpha < material.alpha_cutoff) {
            discard;
        }
    }

    return encode_octahedral(vertex_normal(input, material, front_face));
}

// --- Shading a fragment ------------------------------------------------------

// The surface at this fragment: its exposed radiance (or one input, in a
// debug view), and its alpha. The lighting pass writes it as it is; the
// transparency pass adds it into its sums.
// `front_face`: whether this triangle faces the camera.
float4 shade_fragment(VertexOutput input, bool front_face) {
    FrameData* frame = push.frame;
    const Material material = frame.materials[frame.draws[input.draw_index].material];

    // Base color: factor x texture x vertex color. sRGB textures are decoded
    // to linear by the sampler, so all three are linear.
    float4 base_color = material.base_color_factor * sample_slot(material.base_color, input) * input.color;

    if (alpha_mode == alpha_opaque) {
        base_color.a = 1.0;
    } else if (alpha_mode == alpha_mask) {
        // Below the cutoff the pixel is cut out entirely: no color, no depth.
        if (base_color.a < material.alpha_cutoff) {
            discard;
        }
        base_color.a = 1.0;
    }

    // Metallic and roughness share one texture: blue and green.
    const float4 metallic_roughness = sample_slot(material.metallic_roughness, input);
    const float metallic = material.metallic_factor * metallic_roughness.b;
    const float roughness = material.roughness_factor * metallic_roughness.g;

    // Occlusion darkens creases that ambient light can't reach. Strength
    // blends between no effect (0) and the full map (1).
    const float occlusion = 1.0 + material.occlusion_strength * (sample_slot(material.occlusion, input).r - 1.0);

    const float3 emissive = material.emissive_factor * sample_slot(material.emissive, input).rgb;

    const float3 normal = surface_normal(input, material, front_face, frame.view != view_vertex_normal);

    // Ambient occlusion, from the AO pass's image at this pixel. It combines
    // with the occlusion map by min, not product: both estimate the same
    // thing, at two scales. The bent normal is a deflection from the vertex
    // normal; turning the shading normal by the same deflection keeps the
    // normal map's detail. See-through surfaces aren't in the prepass, so
    // the image there holds whatever is behind them: they use the map alone.
    const float4 gtao = Texture2D.Handle(uint2(frame.ambient_occlusion, 0)).Load(int3(int2(input.position.xy), 0));
    float visibility = occlusion;
    float3 irradiance_normal = normal;

    if (frame.ao_enabled != 0 && alpha_mode != alpha_blend) {
        visibility = min(occlusion, gtao.w);
        irradiance_normal = normalize(rotate_from_to(vertex_normal(input, material, front_face), gtao.xyz, normal));
    }

    // The debug views show one input each. Directions are shown as colors:
    // each component's -1..1 mapped to 0..1.
    switch (frame.view) {
        case view_base_color: return base_color;
        case view_normal:
        case view_vertex_normal: return float4(normal * 0.5 + 0.5, 1.0);
        case view_metallic: return float4(metallic.xxx, 1.0);
        case view_roughness: return float4(roughness.xxx, 1.0);
        case view_occlusion: return float4(occlusion.xxx, 1.0);
        case view_emissive: return float4(emissive, 1.0);
        case view_ambient_occlusion: return float4(gtao.www, 1.0);
        default: break;
    }

    // The triangle's flat normal, from how the position changes across
    // neighbouring pixels: exact up to rounding, since a triangle is flat.
    // Shadow rays start off the surface along it. A triangle seen exactly
    // edge-on has no area on screen, and no such normal: then the shading
    // normal stands in, rather than a division by zero.
    const float3 face_cross = cross(ddy(input.relative_position), ddx(input.relative_position));
    const float3 face_normal = dot(face_cross, face_cross) > 1e-24 ? normalize(face_cross) : normal;

    // The shadow view: how much of the sun's light gets through to each point.
    if (frame.view == view_shadow) {
        const float sun = frame.sun_direction.y > 0.0
            ? light_visibility(frame, shadow_ray_origin(frame, input.relative_position, face_normal, frame.sun_direction),
                frame.sun_direction, infinite_distance)
            : 0.0;
        return float4(sun.xxx, 1.0);
    }

    // A perfectly smooth surface would reflect a punctual light from a single
    // point, too small for any pixel to catch; a floor on roughness keeps
    // highlights visible.
    const float3 view = normalize(-input.relative_position);  // toward the camera, at the origin
    const Surface surface = {
        base_color.rgb,
        metallic,
        max(roughness, 0.045) * max(roughness, 0.045),
        normal,
        view,
        energy_compensation(frame, base_color.rgb, metallic, roughness, max(dot(normal, view), 1e-4)),
    };

    // Direct light, shadowed: the sun, then every light in the file.
    float3 radiance = shade_shadowed(surface, frame, input.relative_position, face_normal,
        frame.sun_direction, frame.sun_illuminance, infinite_distance);

    for (uint i = 0; i < frame.light_count; ++i) {
        float3 l;
        float distance;
        const float3 illuminance = punctual_light(frame, frame.lights[i], input.relative_position, l, distance);
        radiance += shade_shadowed(surface, frame, input.relative_position, face_normal, l, illuminance, distance);
    }

    // Indirect light from the sky, darkened by occlusion. Ambient occlusion
    // only ever reaches this indirect light: the sun and the lights are
    // direct, and only a shadow can block them.
    radiance += shade_environment(surface, frame, roughness, visibility, irradiance_normal);

    // The air between the camera and the surface, with the simulated sky:
    // it dims the surface's light, emission included, and adds its own. The
    // photograph has no air to go with it.
    float3 transmittance = 1.0;
    float3 inscatter = 0.0;

    if (frame.atmosphere != 0) {
        inscatter = aerial_perspective(frame, input.relative_position, transmittance);
    }

    // Exposure scales nits into the tone mapper's range here, before the
    // 16-bit HDR image could overflow. glTF defines emission in nits, but, as
    // its spec notes many engines do, we take it as already exposed: an
    // emissive value of 1 shows as near-white, whatever the exposure.
    return float4((radiance * transmittance + inscatter) * frame.exposure + emissive * transmittance, base_color.a);
}

// --- Fragment shaders ----------------------------------------------------------

// The lighting pass, for opaque and masked surfaces.
// SV_Target: the value written to color attachment 0.
// SV_IsFrontFace: whether this triangle faces the camera.
[shader("fragment")]
float4 fragmentMain(VertexOutput input, bool front_face : SV_IsFrontFace) : SV_Target {
    return shade_fragment(input, front_face);
}

// --- Weighted blended transparency ---------------------------------------------

// The transparency pass, for blended surfaces (McGuire and Bavoil 2013,
// "Weighted Blended Order-Independent Transparency"). Blending one surface
// over another depends on which is in front, so blended surfaces would have
// to be sorted back to front, per pixel. Instead, every fragment adds into
// two sums, in any order:
//   accum   (premultiplied color, coverage) times a weight that falls with
//           distance, added up
//   reveal  the share of the scene that shows through: 1, times every
//           fragment's (1 - coverage)
// The composite (composite.slang) divides accum's color by its coverage,
// a weighted average of the layers, and lays it over the scene by
// 1 - reveal. One layer comes out as blending would draw it, to 16-bit
// precision; where layers overlap, the nearer one counts for more.
struct TransparentOutput {
    float4 accum : SV_Target0;
    float reveal : SV_Target1;
};

// The weight: McGuire and Bavoil's equation 7, tuned for 16-bit float sums
// and distances from 0.1 m to 500 m. Past a few hundred metres it bottoms
// out at its floor, so distant layers that overlap count equally. It falls
// steeply with the distance in front of the camera, so where layers overlap,
// the nearest dominates; the
// clamp keeps it between 1e-2 and 3e3. Colors are clamped to
// transparent_max, which tone mapping already shows as nearly white: one
// fragment then adds at most 4 x 3e3 = 12000, well below the 65504 a
// 16-bit float holds, so many layers can stack up before the sums overflow.
static const float transparent_max = 4.0;

float transparent_weight(float coverage, float view_depth) {
    const float a = view_depth / 5.0;
    const float b = view_depth / 200.0;
    return coverage * clamp(10.0 / (1e-5 + a * a + b * b * b * b * b * b), 1e-2, 3e3);
}

[shader("fragment")]
TransparentOutput transparentMain(VertexOutput input, bool front_face : SV_IsFrontFace) {
    const float4 color = shade_fragment(input, front_face);
    const float coverage = saturate(color.a);

    // The distance in front of the camera, along its view direction: the
    // clip-space w a perspective projection leaves.
    const float view_depth = mul(push.frame.view_projection, float4(input.relative_position, 1.0)).w;
    const float weight = transparent_weight(coverage, view_depth);

    TransparentOutput output;
    output.accum = float4(min(color.rgb, transparent_max) * coverage, coverage) * weight;
    output.reveal = coverage;
    return output;
}
```

`game-engine/shaders/background.slang`:
```slang
// Draws the sky behind the scene: one full-screen triangle at depth 0,
// infinitely far away, depth-tested so it only covers pixels nothing else
// has drawn on.

#include "shared.slangh"
#include "atmosphere.slangh"

// In a descriptor heap pipeline, the push_constant block is where push data lands.
[[vk::push_constant]]
ConstantBuffer<PushData> push;

// --- Vertex shader -----------------------------------------------------------

struct VertexOutput {
    float4 position : SV_Position;
    float2 clip : TEXCOORD0;  // this point's clip-space x and y
};

// The full-screen triangle of tonemap.slang, at depth 0: with reverse-Z and
// no far plane, that's infinitely far, so the depth test lets it through
// only where the depth buffer still holds its cleared 0. Its clip-space corners are also
// passed on: interpolated across the triangle, they arrive at each pixel as
// that pixel's own clip-space position.
[shader("vertex")]
VertexOutput vertexMain(uint vertex_id : SV_VulkanVertexID) {
    const float2 corner = float2((vertex_id << 1) & 2, vertex_id & 2) * 2.0 - 1.0;

    VertexOutput output;
    output.position = float4(corner, 0.0, 1.0);
    output.clip = corner;
    return output;
}

// --- Fragment shader ---------------------------------------------------------

// Each pixel looks along the ray from the camera through it. Turning the
// pixel's clip-space position at the near plane (depth 1) back into
// camera-relative space gives a point on that ray; the direction is from the
// camera, at the origin, to it. The far plane won't do: with none, depth 0
// is infinitely far, and w there is 0.
[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    FrameData* frame = push.frame;

    const float4 near_point = mul(frame.inverse_view_projection, float4(input.clip, 1.0, 1.0));
    const float3 direction = normalize(near_point.xyz / near_point.w);

    // The simulated sky from its sky-view table, which holds more detail
    // near the horizon than the cube; the photograph from the cube.
    const SamplerState clamped = SamplerState.Handle(uint2(frame.clamp_sampler, 0));
    float3 radiance;

    if (frame.atmosphere != 0) {
        const float altitude = float(frame.camera_cell.y) * cell_size + frame.camera_offset.y;
        const Texture2D sky_view = Texture2D.Handle(uint2(frame.sky_view, 0));
        radiance = sky_view_radiance(sky_view, clamped, altitude, direction, frame.sun_direction);
    } else {
        const TextureCube sky = TextureCube.Handle(uint2(frame.sky_cube, 0));
        radiance = sky.SampleLevel(clamped, direction, 0.0).rgb;
    }

    // The sun's disk, which neither the table nor the cube holds: its illuminance spread
    // over the tiny solid angle it covers, pi r^2 for an angular radius r.
    if (dot(direction, frame.sun_direction) > cos(frame.sun_angular_radius)) {
        const float solid_angle = 3.14159265 * frame.sun_angular_radius * frame.sun_angular_radius;
        radiance += frame.sun_illuminance / solid_angle;
    }

    // Exposed like the scene, and kept below the 16-bit float limit: the
    // sun's disk is over a billion nits.
    return float4(min(radiance * frame.exposure, 60000.0), 1.0);
}
```

## 14.6 A ground: `scene.h`, `scene.cpp`

### Why
Something to see the air over, out to where the aerial perspective ends, 32 km away.

### How
- **`add_ground`** adds a square of flat ground to a scene:
  - **One tile,** a two-triangle square 1,024 m across, as one primitive, with its own plain material: dark, rough, not metal, no textures.
  - **Drawn once per tile,** 64 × 64 = 4,096 draws, each placed in the cell its corner is in. A single 65.5 km square in one cell would have coordinates up to 32.8 km along each axis, where floats are 2 to 4 mm apart; this way, every tile keeps its precision, wherever the camera is.
  - **The cull** throws away all but the tiles in view, and groups the rest into one instanced draw (Chapter 12).
- **Its height:** 1 cm below the scene's lowest point, so it never fights the scene's own floor for the same depth.

### Code
`game-engine/src/includes/scene.h`:
```cpp
#pragma once

#include "includes/shader_types.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

// One glTF primitive: a run of indices in the scene's index buffer, drawn
// against the vertices starting at `vertex_offset` in the vertex buffer.
struct Primitive {
    std::uint32_t first_index = 0;
    std::uint32_t index_count = 0;
    std::int32_t vertex_offset = 0;
    std::uint32_t material = 0;  // index into Scene::materials
};

// glTF's texture filters and wrap modes, as the file stores them: OpenGL
// enum values (9728 GL_NEAREST, 10497 GL_REPEAT, ...). A filter of -1 means
// the file leaves it to the renderer.
struct SceneSampler {
    int mag_filter = -1;
    int min_filter = -1;
    int wrap_s = 10497;  // GL_REPEAT, glTF's default
    int wrap_t = 10497;
};

// A material's reference to one texture.
struct TextureRef {
    std::int32_t image = -1;    // index into Scene::images, or -1 for none
    std::int32_t sampler = -1;  // index into Scene::samplers, or -1 for the default
    std::uint32_t uv_set = 0;   // which texture coordinates: 0 or 1
};

// A glTF metallic-roughness material. The defaults are glTF's: a material
// that sets nothing is white, fully metallic and fully rough.
struct SceneMaterial {
    glm::vec4 base_color_factor{1.0f};
    TextureRef base_color;
    float metallic_factor = 1.0f;
    float roughness_factor = 1.0f;
    TextureRef metallic_roughness;
    TextureRef normal;
    float normal_scale = 1.0f;
    TextureRef occlusion;
    float occlusion_strength = 1.0f;
    glm::vec3 emissive_factor{0.0f};
    TextureRef emissive;
    AlphaMode alpha_mode = AlphaMode::opaque;
    float alpha_cutoff = 0.5f;
    bool double_sided = false;
};

// An image as the file stores it: still encoded as PNG, JPEG, ...
struct SceneImage {
    std::vector<unsigned char> encoded;
    std::string name;   // the file name or glTF name, for messages
    bool srgb = false;  // holds colors (base color, emissive) rather than data like normals
};

// One thing to draw: a primitive, placed in the world by a node's transform.
// A mesh used by several nodes is drawn once per node. The draw is placed in
// a world cell (cells.h): `model` moves the primitive into it, measured from
// the cell's corner, and the box is measured from there too.
struct MeshDraw {
    glm::mat4 model{1.0f};
    glm::ivec3 cell{0};
    std::uint32_t primitive = 0;

    // A transform that mirrors the primitive (a negative scale) reverses the
    // order its triangles' corners appear in, which decides which side is
    // the front.
    bool mirrored = false;

    // The draw's box, in its cell: the primitive's box, moved by `model`, and
    // boxed again. It holds every triangle of the draw, if a little loosely
    // when the transform rotates.
    glm::vec3 bounds_min{0.0f};
    glm::vec3 bounds_max{0.0f};
};

// Everything from a glTF file that drawing its geometry needs, flattened into
// arrays ready to upload: every primitive's vertices and indices back to back.
struct Scene {
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<Primitive> primitives;
    std::vector<MeshDraw> draws;

    // The file's materials, plus a plain white one at the end for primitives
    // that don't name a material.
    std::vector<SceneMaterial> materials;
    std::vector<SceneImage> images;
    std::vector<SceneSampler> samplers;

    // KHR_lights_punctual lights, placed in the world by their nodes.
    std::vector<Light> lights;

    // World-space box around everything drawn, roughly, in floats. Nothing
    // uses it yet: it's there for a camera that frames the scene.
    glm::vec3 bounds_min{std::numeric_limits<float>::max()};
    glm::vec3 bounds_max{std::numeric_limits<float>::lowest()};
};

// Loads the default scene of a .gltf or .glb file, with its materials,
// samplers, lights and images, still encoded.
// The scene is placed with its origin at `origin`, in metres from the
// world's.
Scene load_gltf(const std::filesystem::path& path, const glm::dvec3& origin = glm::dvec3{0.0});

// Adds a flat, plain ground to `scene`: a square `size` metres across,
// centred under `centre` at height `height`, made of `tile`-metre tiles. The
// tiles are one primitive drawn once per tile, each placed in its own cell,
// so the ground holds a float's precision wherever the camera is on it.
void add_ground(Scene& scene, const glm::dvec3& centre, double height, double size, double tile);
```

In `game-engine/src/scene.cpp`, add `#include <array>` after `#include <algorithm>`.

In `game-engine/src/scene.cpp`, add this section before `Scene load_gltf(`:
```cpp
void add_ground(Scene& scene, const glm::dvec3& centre, double height, double size, double tile) {
    // One tile: a square from its corner, facing up. Both triangles wind
    // counter-clockwise seen from above, glTF's front.
    const auto edge = static_cast<float>(tile);
    const std::array<glm::vec3, 4> corners{
        glm::vec3{0.0f, 0.0f, 0.0f},
        glm::vec3{0.0f, 0.0f, edge},
        glm::vec3{edge, 0.0f, edge},
        glm::vec3{edge, 0.0f, 0.0f},
    };

    const Primitive primitive{
        .first_index = static_cast<std::uint32_t>(scene.indices.size()),
        .index_count = 6,
        .vertex_offset = static_cast<std::int32_t>(scene.vertices.size()),
        .material = static_cast<std::uint32_t>(scene.materials.size()),
    };

    for (const glm::vec3& corner : corners) {
        scene.vertices.push_back(Vertex{
            .position = corner,
            .normal = {0.0f, 1.0f, 0.0f},
            .tangent = {1.0f, 0.0f, 0.0f, 1.0f},
            .uv0 = {corner.x, corner.z},
            .uv1 = {corner.x, corner.z},
            .color = glm::vec4{1.0f},
        });
    }

    for (const std::uint32_t index : {0u, 1u, 2u, 0u, 2u, 3u}) {
        scene.indices.push_back(index);
    }

    // Dry, bare ground: dark, rough, not metal, no textures.
    SceneMaterial ground;
    ground.base_color_factor = {0.3f, 0.28f, 0.25f, 1.0f};
    ground.metallic_factor = 0.0f;
    ground.roughness_factor = 0.9f;
    scene.materials.push_back(ground);

    const auto primitive_index = static_cast<std::uint32_t>(scene.primitives.size());
    scene.primitives.push_back(primitive);

    // The tiles, row by row, each placed in the cell its corner is in.
    const auto tiles = static_cast<int>(size / tile);
    const glm::dvec3 first{centre.x - size * 0.5, height, centre.z - size * 0.5};

    for (int row = 0; row < tiles; ++row) {
        for (int column = 0; column < tiles; ++column) {
            const glm::dvec3 corner = first + glm::dvec3{column * tile, 0.0, row * tile};
            const CellPosition placed = to_cell(corner);

            scene.draws.push_back(MeshDraw{
                .model = glm::translate(glm::mat4{1.0f}, placed.offset),
                .cell = placed.cell,
                .primitive = primitive_index,
                .mirrored = false,
                .bounds_min = placed.offset,
                .bounds_max = placed.offset + glm::vec3{edge, 0.0f, edge},
            });
        }
    }
}
```

## 14.7 Every frame: `main.cpp`

### Why
`main` adds the ground, keeps the sky built for the camera's height, and records the atmosphere's tables each frame.

### How
- **The ground** goes under the scene, 65.5 km across, right after loading it.
- **The camera's height:** its y position, taken as above ground at y = 0, at least 1 m.
- **Rebuilding the sky cube:** as before when the hour changes, and now also when the camera's height has moved 100 m since it was built. The scene's sky light changes slowly with height, and the rebuild waits for the GPU.
- **`record_frame`** records the atmosphere's tables right after the cull, before the depth prepass. `DrawList` carries the sun's direction and the camera's height.
- **`FrameData`** gets the tables' slots and `atmosphere`.

### Code
`game-engine/src/main.cpp`:
```cpp
#include "includes/acceleration.h"
#include "includes/ambient_occlusion.h"
#include "includes/buffer.h"
#include "includes/camera.h"
#include "includes/cells.h"
#include "includes/culling.h"
#include "includes/daylight.h"
#include "includes/descriptor_heap.h"
#include "includes/environment.h"
#include "includes/pipeline.h"
#include "includes/scene.h"
#include "includes/sdl.h"
#include "includes/shader_types.h"
#include "includes/swapchain.h"
#include "includes/texture.h"
#include "includes/vulkan_setup.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <limits>
#include <print>
#include <span>
#include <string>
#include <vector>

namespace {

// --- Frames in flight --------------------------------------------------------

// How many frames the CPU may record ahead of the GPU.
constexpr std::size_t frames_in_flight = 2;

constexpr std::uint64_t no_timeout = std::numeric_limits<std::uint64_t>::max();

// How far, in cells each way, the camera may stray from the TLAS's origin
// cell before the TLAS is rebuilt around it: 16 cells of 64 m, about 1 km.
// Within that, a ray near the camera is traced at coordinates under about
// 1.1 km, where floats are 0.06 to 0.12 mm apart.
constexpr int tlas_reach_cells = 16;

// What each in-flight frame needs for itself. `data` holds this frame's
// FrameData; the GPU may still be reading the other frame's while the CPU
// writes this one.
struct Frame {
    vk::raii::CommandBuffer commands = nullptr;
    vk::raii::Semaphore image_acquired = nullptr;  // swapchain image is ready to draw into
    vk::raii::Fence done = nullptr;                // GPU finished this frame's commands
    Buffer data;                                   // one FrameData, host-visible
    FrameData* mapped = nullptr;                   // `data`, mapped for the CPU to write
    Buffer cull_totals;                            // the cull's totals, copied out, host-visible
    const CullTotals* totals = nullptr;            // `cull_totals`, mapped for the CPU to read
};

// --- Recording a frame -------------------------------------------------------

// Moves `image` between layouts, and makes the `dst` work wait for the `src` work.
// `aspect` is which part of the image: its color, or its depth.
void transition(
    const vk::raii::CommandBuffer& commands,
    vk::Image image,
    vk::ImageLayout from,
    vk::ImageLayout to,
    vk::PipelineStageFlags2 src_stage,
    vk::AccessFlags2 src_access,
    vk::PipelineStageFlags2 dst_stage,
    vk::AccessFlags2 dst_access,
    vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor
) {
    const vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
        .oldLayout = from,
        .newLayout = to,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = image,
        .subresourceRange = {
            .aspectMask = aspect,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    commands.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier,
    });
}

// The alpha modes the prepass and the lighting pass draw, in the order of
// their pipelines. The see-through mode, AlphaMode::blend, has only the
// transparency pass's.
constexpr std::array solid_modes{AlphaMode::opaque, AlphaMode::mask};

// glTF's front faces wind counter-clockwise, seen from the front. Our
// projection's Y flip (see camera.cpp) only undoes the difference between
// OpenGL's upward Y and Vulkan's downward one, so on screen they still wind
// counter-clockwise. A mirroring transform reverses that.
constexpr vk::FrontFace front_face = vk::FrontFace::eCounterClockwise;
constexpr vk::FrontFace mirrored_front_face = vk::FrontFace::eClockwise;

// Resource heap slots of the swapchain's images, written by describe_screen()
// in main and rewritten whenever the swapchain is rebuilt.
struct ScreenSlots {
    std::uint32_t hdr = 0;             // sampled, by tone mapping
    std::uint32_t depth = 0;           // sampled, by ambient occlusion
    std::uint32_t normals = 0;         // sampled, by ambient occlusion
    std::uint32_t ao = 0;              // sampled, by the lighting pass
    AoTargets ao_targets;              // storage, for the AO pass
    std::uint32_t accum = 0;           // sampled, by the transparency composite
    std::uint32_t reveal = 0;          // sampled, by the transparency composite
};

constexpr std::uint32_t screen_slot_count = 11;

// Every graphics pipeline a frame uses. The prepass and the lighting pass
// have one per solid alpha mode, in solid_modes' order; see-through surfaces
// have only the transparency pass's.
struct ScenePipelines {
    std::vector<vk::raii::Pipeline> prepass;
    std::vector<vk::raii::Pipeline> lighting;
    vk::raii::Pipeline transparency = nullptr;
    vk::raii::Pipeline background = nullptr;
    vk::raii::Pipeline composite = nullptr;
    vk::raii::Pipeline tonemap = nullptr;
};

// What a frame draws and where it finishes: the scene's index buffer, the
// frame's data, the screen images' slots, and where the cull's totals go.
struct DrawList {
    vk::Buffer index_buffer;
    vk::DeviceAddress frame = 0;   // this frame's FrameData
    ScreenSlots screen;
    View view = View::lit;
    vk::Buffer readback;           // this frame's copy of the cull's totals
    bool see_through = false;      // whether the scene has blended draws at all
    glm::vec3 sun_direction{0.0f}; // toward the sun
    float altitude = 0.0f;         // the camera's height above the ground, in metres
};

// Draws every list of alpha mode `mode` with `pipeline`: one indirect call
// per list, after setting the list's cull mode and front face. Push data
// says where the frame's data is; each instance finds its DrawData through
// the cull's instances.
void draw_mode(
    const vk::raii::CommandBuffer& commands,
    const DrawCulling& culling,
    const DrawList& draws,
    AlphaMode mode,
    const vk::raii::Pipeline& pipeline
) {
    commands.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);

    const PushData push{.frame = draws.frame};

    commands.pushDataEXT(vk::PushDataInfoEXT{
        .offset = 0,
        .data = {.address = &push, .size = sizeof(push)},
    });

    for (const bool double_sided : {false, true}) {
        for (const bool mirrored : {false, true}) {
            const std::uint32_t list = draw_list_index(mode, double_sided, mirrored);

            if (culling.list_ranges[list].group_count == 0) {
                continue;
            }

            // Single-sided surfaces are invisible from behind, so the GPU can
            // skip their back faces before running the fragment shader.
            commands.setCullMode(double_sided ? vk::CullModeFlagBits::eNone : vk::CullModeFlagBits::eBack);
            commands.setFrontFace(mirrored ? mirrored_front_face : front_face);
            draw_list(commands, culling, list);
        }
    }
}

// The viewport and scissor every pass uses: the whole image. The pipelines
// leave both dynamic.
void set_viewport(const vk::raii::CommandBuffer& commands, vk::Extent2D extent) {
    commands.setViewport(0, vk::Viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(extent.width),
        .height = static_cast<float>(extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    });
    commands.setScissor(0, vk::Rect2D{.offset = {0, 0}, .extent = extent});
}

// Records a frame: the cull, then five passes:
//   1. the depth prepass: every solid surface's depth and vertex normal,
//   2. ambient occlusion, from those, in compute shaders,
//   3. the lighting, into the HDR image: each solid alpha mode's lists with
//      that mode's pipeline, against the prepass's depth, then the sky
//      behind them,
//   4. transparency, if anything is see-through: the blended lists into two
//      sums, in any order, then those laid over the HDR image,
//   5. tone mapping, from the HDR image into the swapchain image, which is
//      then ready to present.
void record_frame(
    const vk::raii::CommandBuffer& commands,
    const Swapchain& swapchain,
    std::uint32_t image_index,
    const ScenePipelines& pipelines,
    const AmbientOcclusion& ambient_occlusion,
    const DrawCulling& culling,
    const Environment& environment,
    const DescriptorHeaps& heaps,
    const DrawList& draws
) {
    const vk::Image image = swapchain.images[image_index];
    const vk::Image hdr = *swapchain.hdr.handle;
    const vk::Image depth = *swapchain.depth.handle;
    const vk::Image normals = *swapchain.normals.handle;

    commands.reset();
    commands.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    // Every texture and sampler the shaders read comes from these two heaps.
    // They stay bound for every pass, graphics and compute.
    bind_descriptor_heaps(commands, heaps);

    // One index buffer for the whole scene. Indices go through the GPU's
    // fixed-function index fetch, which also lets it reuse vertices shared
    // between neighbouring triangles.
    commands.bindIndexBuffer(draws.index_buffer, 0, vk::IndexType::eUint32);

    // --- The cull --------------------------------------------------------------

    // Every pass below draws only what the cull keeps, from its lists.
    record_culling(commands, culling, draws.frame, draws.readback);

    // --- The atmosphere ------------------------------------------------------

    // The sky around the camera and the air in front of it, for this frame's
    // sun, height and view: the background and the lighting read them.
    record_atmosphere(commands, environment, draws.frame, draws.sun_direction, draws.altitude);

    // --- Pass 1: the depth prepass -------------------------------------------

    // The depth buffer and the normals are shared by the frames in flight, so
    // these also wait for the previous frame to finish reading them: the AO
    // pass reads both, and the lighting pass depth-tests against the depth.
    transition(commands, depth,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthAttachmentOptimal,
        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::AccessFlagBits2::eShaderSampledRead | vk::AccessFlagBits2::eDepthStencilAttachmentRead,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::ImageAspectFlagBits::eDepth
    );

    transition(commands, normals,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderSampledRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite
    );

    // Reverse-Z: 0 is infinitely far. The depth is stored this time: the AO
    // pass and the lighting pass both read it.
    const vk::RenderingAttachmentInfo normal_attachment{
        .imageView = *swapchain.normals.view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.float32 = std::array{0.0f, 0.0f, 0.0f, 0.0f}}},
    };

    const vk::RenderingAttachmentInfo prepass_depth{
        .imageView = *swapchain.depth.view,
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{.depth = 0.0f}},
    };

    const vk::Rect2D whole_image{.offset = {0, 0}, .extent = swapchain.extent};

    commands.beginRendering(vk::RenderingInfo{
        .renderArea = whole_image,
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &normal_attachment,
        .pDepthAttachment = &prepass_depth,
    });

    set_viewport(commands, swapchain.extent);

    // Solid surfaces only: opaque, then masked.
    for (std::size_t i = 0; i < solid_modes.size(); ++i) {
        draw_mode(commands, culling, draws, solid_modes[i], pipelines.prepass[i]);
    }

    commands.endRendering();

    // --- Pass 2: ambient occlusion -------------------------------------------

    // From here on the depth is only read: by the AO pass, and as the
    // lighting pass's depth test, which eDepthReadOnlyOptimal allows at once.
    transition(commands, depth,
        vk::ImageLayout::eDepthAttachmentOptimal, vk::ImageLayout::eDepthReadOnlyOptimal,
        vk::PipelineStageFlagBits2::eLateFragmentTests, vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eEarlyFragmentTests
            | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::AccessFlagBits2::eShaderSampledRead | vk::AccessFlagBits2::eDepthStencilAttachmentRead,
        vk::ImageAspectFlagBits::eDepth
    );

    transition(commands, normals,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderSampledRead
    );

    record_ambient_occlusion(commands, ambient_occlusion, swapchain, draws.screen.ao_targets,
        AoPushData{.frame = draws.frame, .depth = draws.screen.depth, .normals = draws.screen.normals});

    // --- Pass 3: the lighting --------------------------------------------------

    // The HDR image is shared by the frames in flight too: this waits for
    // the previous frame's tone mapping to finish reading it.
    transition(commands, hdr,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
        vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite
    );

    // Every pixel is drawn over, by the scene or the sky; clearing is just
    // cheaper than loading what was there.
    const vk::RenderingAttachmentInfo hdr_attachment{
        .imageView = *swapchain.hdr.view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.float32 = std::array{0.0f, 0.0f, 0.0f, 1.0f}}},
    };

    // The prepass's depth, loaded and tested against but not written.
    const vk::RenderingAttachmentInfo lighting_depth{
        .imageView = *swapchain.depth.view,
        .imageLayout = vk::ImageLayout::eDepthReadOnlyOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eNone,
    };

    commands.beginRendering(vk::RenderingInfo{
        .renderArea = whole_image,
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &hdr_attachment,
        .pDepthAttachment = &lighting_depth,
    });

    set_viewport(commands, swapchain.extent);

    for (std::size_t i = 0; i < solid_modes.size(); ++i) {
        draw_mode(commands, culling, draws, solid_modes[i], pipelines.lighting[i]);
    }

    // The sky goes in once everything solid is drawn: it only covers pixels
    // still at depth 0, infinitely far.
    commands.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines.background);

    const PushData sky_push{.frame = draws.frame};
    commands.pushDataEXT(vk::PushDataInfoEXT{
        .offset = 0,
        .data = {.address = &sky_push, .size = sizeof(sky_push)},
    });

    commands.draw(3, 1, 0, 0);
    commands.endRendering();

    // --- Pass 4: transparency ------------------------------------------------

    if (draws.see_through) {
        const vk::Image accum = *swapchain.accum.handle;
        const vk::Image reveal = *swapchain.reveal.handle;

        // Both sums are shared by the frames in flight: this waits for the
        // previous frame's composite to finish reading them.
        for (const vk::Image sum : {accum, reveal}) {
            transition(commands, sum,
                vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
                vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite
            );
        }

        // Before any layer: nothing summed, and the whole scene showing through.
        const std::array sum_attachments{
            vk::RenderingAttachmentInfo{
                .imageView = *swapchain.accum.view,
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.float32 = std::array{0.0f, 0.0f, 0.0f, 0.0f}}},
            },
            vk::RenderingAttachmentInfo{
                .imageView = *swapchain.reveal.view,
                .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .loadOp = vk::AttachmentLoadOp::eClear,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.float32 = std::array{1.0f, 0.0f, 0.0f, 0.0f}}},
            },
        };

        // The same depth as the lighting pass: tested, not written, so a
        // see-through surface behind a solid one is hidden, and one behind
        // another see-through one still counts.
        commands.beginRendering(vk::RenderingInfo{
            .renderArea = whole_image,
            .layerCount = 1,
            .colorAttachmentCount = static_cast<std::uint32_t>(sum_attachments.size()),
            .pColorAttachments = sum_attachments.data(),
            .pDepthAttachment = &lighting_depth,
        });

        // The viewport and scissor set earlier still apply: dynamic state
        // lasts for the whole command buffer.
        draw_mode(commands, culling, draws, AlphaMode::blend, pipelines.transparency);
        commands.endRendering();

        // The composite reads both sums, and blends into the HDR image, which
        // the lighting pass just wrote: the same layout, but its writes must
        // land before the blend reads them.
        for (const vk::Image sum : {accum, reveal}) {
            transition(commands, sum,
                vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
                vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead
            );
        }

        transition(commands, hdr,
            vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eColorAttachmentOptimal,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite
        );

        // The lit scene is kept and blended into.
        const vk::RenderingAttachmentInfo scene_attachment{
            .imageView = *swapchain.hdr.view,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
        };

        commands.beginRendering(vk::RenderingInfo{
            .renderArea = whole_image,
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &scene_attachment,
        });

        commands.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines.composite);

        const CompositePushData composite_push{.accum = draws.screen.accum, .reveal = draws.screen.reveal};
        commands.pushDataEXT(vk::PushDataInfoEXT{
            .offset = 0,
            .data = {.address = &composite_push, .size = sizeof(composite_push)},
        });

        commands.draw(3, 1, 0, 0);
        commands.endRendering();
    }

    // --- Pass 5: tone mapping ------------------------------------------------

    // The scene is finished: the tone-mapping shader may read it now.
    transition(commands, hdr,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead
    );

    // Undefined: every pixel is about to be overwritten.
    transition(commands, image,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eNone,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite
    );

    // The full-screen triangle writes every pixel, so there's nothing to
    // clear or load first.
    const vk::RenderingAttachmentInfo color_attachment{
        .imageView = *swapchain.views[image_index],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eDontCare,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };

    commands.beginRendering(vk::RenderingInfo{
        .renderArea = whole_image,
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment,
    });

    commands.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines.tonemap);

    const TonemapPushData push{.hdr_image = draws.screen.hdr, .view = draws.view};

    commands.pushDataEXT(vk::PushDataInfoEXT{
        .offset = 0,
        .data = {.address = &push, .size = sizeof(push)},
    });

    commands.draw(3, 1, 0, 0);
    commands.endRendering();

    transition(commands, image,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone
    );

    commands.end();
}

// --- Events ------------------------------------------------------------------

// What the keyboard controls, besides the camera.
struct Settings {
    View view = View::lit;
    SkySource sky = SkySource::atmosphere;
    float hours = 10.0f;                    // time of day, 0 to 24
    float exposure_compensation = 0.0f;     // stops brighter (+) or darker (-) than metered
    bool ambient_occlusion = true;
};

// The views' names, in View's order, for the window title.
constexpr std::array view_names{
    "Lit", "Base color", "Normal", "Vertex normal", "Metallic", "Roughness", "Occlusion", "Emissive", "Ambient occlusion",
    "Sun shadow",
};

// Handles every pending event and fills in `input` for this frame. False once
// the window was closed or Escape pressed.
//   1-9 0 pick the view: 0 is the tenth, as on the keyboard
//   e     switch between the simulated sky and the photographed one
//   o     switch ambient occlusion off and on, to compare
//   [ ]   time of day, a quarter of an hour earlier or later
//   - =   exposure, half a stop darker or brighter: like a camera's
//         exposure compensation, + is brighter
// Holding a key repeats it.
bool poll_events(SDL_Window* window, CameraInput& input, Settings& settings) {
    input = CameraInput{};
    SDL_Event event;

    while (SDL_PollEvent(&event)) {
        const bool escape = event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE;

        if (event.type == SDL_EVENT_QUIT || escape) {
            return false;
        }

        if (event.type == SDL_EVENT_KEY_DOWN) {
            const SDL_Keycode key = event.key.key;

            // SDLK_1 to SDLK_9 are consecutive key codes; SDLK_0 comes before them.
            if (key >= SDLK_1 && key <= SDLK_9) {
                settings.view = static_cast<View>(key - SDLK_1);
            } else if (key == SDLK_0) {
                settings.view = View::shadow;
            } else if (key == SDLK_O) {
                settings.ambient_occlusion = !settings.ambient_occlusion;
            } else if (key == SDLK_E) {
                settings.sky = settings.sky == SkySource::atmosphere ? SkySource::photograph : SkySource::atmosphere;
            } else if (key == SDLK_LEFTBRACKET) {
                settings.hours = std::fmod(settings.hours + 23.75f, 24.0f);
            } else if (key == SDLK_RIGHTBRACKET) {
                settings.hours = std::fmod(settings.hours + 0.25f, 24.0f);
            } else if (key == SDLK_MINUS) {
                settings.exposure_compensation -= 0.5f;
            } else if (key == SDLK_EQUALS) {
                settings.exposure_compensation += 0.5f;
            }
        }

        if (event.type == SDL_EVENT_MOUSE_MOTION) {
            input.mouse_delta += glm::vec2{event.motion.xrel, event.motion.yrel};
        } else if (event.type == SDL_EVENT_MOUSE_WHEEL) {
            input.wheel += event.wheel.y;
        }
    }

    // Which buttons are held right now.
    const SDL_MouseButtonFlags buttons = SDL_GetMouseState(nullptr, nullptr);
    input.right_button = (buttons & SDL_BUTTON_RMASK) != 0;
    input.left_button = (buttons & SDL_BUTTON_LMASK) != 0;
    input.middle_button = (buttons & SDL_BUTTON_MMASK) != 0;

    // While a button is held, relative mode hides the cursor and keeps
    // reporting movement, so a drag can't run into the edge of the screen.
    const bool dragging = input.right_button || input.left_button || input.middle_button;

    if (dragging != SDL_GetWindowRelativeMouseMode(window)) {
        SDL_SetWindowRelativeMouseMode(window, dragging);
    }

    return true;
}

}  // namespace

int main() {
    try {
        // --- Window and instance ---------------------------------------------

        SdlContext sdl;
        const int version = SDL_GetVersion();
        std::println("SDL {}.{}.{} on {}", SDL_VERSIONNUM_MAJOR(version), SDL_VERSIONNUM_MINOR(version),
            SDL_VERSIONNUM_MICRO(version), SdlContext::video_driver());

        // Loads libvulkan at runtime, so nothing has to link against it.
        vk::raii::Context context;

#ifdef NDEBUG
        const bool validation = false;
#else
        const bool validation = validation_layer_available(context);
#endif
        std::println("Validation layer {}", validation ? "on" : "off");

        // Declaration order matters: each object is destroyed before the ones above it.
        // The window comes first: creating it loads Vulkan into SDL, which
        // required_vulkan_extensions() needs.
        Window window = make_vulkan_window(1920, 1080, "game-engine", true);

        vk::raii::Instance instance = create_instance(context, SdlContext::required_vulkan_extensions(), validation);
        vk::raii::DebugUtilsMessengerEXT messenger = validation
            ? create_debug_messenger(instance)
            : vk::raii::DebugUtilsMessengerEXT(nullptr);

        vk::raii::SurfaceKHR surface = create_surface(instance, window.get());

        // --- GPU, device and swapchain ---------------------------------------

        std::println("GPUs:");
        std::optional<GpuChoice> gpu = pick_gpu(instance, surface);

        if (!gpu) {
            std::println(stderr, "No GPU has Vulkan 1.4, the descriptor heap and ray queries, and can present to this window");
            return EXIT_FAILURE;
        }

        std::println("Using {}", gpu->device.getProperties().deviceName.data());
        print_descriptor_heap_properties(*gpu);

        vk::raii::Device device = create_device(*gpu);
        vk::raii::Queue queue = device.getQueue(gpu->queue_family, 0);
        Swapchain swapchain = create_swapchain(device, *gpu, surface, window.get());

        // --- Pipelines -------------------------------------------------------

        // recreate_swapchain() picks the same formats again, so the pipelines
        // stay valid across resizes.
        //   - The prepass draws normals and depth, and the lighting pass the
        //     HDR image, for opaque and masked materials. The transparency
        //     pass draws blended ones into its two sums.
        //   - The sky draws into the HDR image, behind the scene, and the
        //     composite over it; tone mapping writes the swapchain image.
        ScenePipelines pipelines;

        for (const AlphaMode mode : solid_modes) {
            pipelines.prepass.push_back(create_mesh_pipeline(device, std::array{normal_format},
                depth_format, mode, MeshPass::depth_normals));
            pipelines.lighting.push_back(create_mesh_pipeline(device, std::array{hdr_format},
                depth_format, mode, MeshPass::lighting));
        }

        pipelines.transparency = create_mesh_pipeline(device, std::array{accum_format, reveal_format},
            depth_format, AlphaMode::blend, MeshPass::transparency);

        pipelines.background = create_fullscreen_pipeline(device, "background", hdr_format, depth_format);
        pipelines.composite = create_fullscreen_pipeline(device, "composite", hdr_format, vk::Format::eUndefined, ColorBlend::over);
        pipelines.tonemap = create_fullscreen_pipeline(device, "tonemap", swapchain.format);

        const AmbientOcclusion ambient_occlusion = create_ambient_occlusion(device);

        // --- Per-frame resources ---------------------------------------------

        // eResetCommandBuffer lets us re-record each frame's command buffer.
        vk::raii::CommandPool command_pool(device, vk::CommandPoolCreateInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = gpu->queue_family,
        });

        vk::raii::CommandBuffers command_buffers(device, vk::CommandBufferAllocateInfo{
            .commandPool = *command_pool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = frames_in_flight,
        });

        std::vector<Frame> frames;
        for (vk::raii::CommandBuffer& commands : command_buffers) {
            // Host-coherent: the CPU's writes reach the GPU without a flush.
            Buffer data = create_buffer(device, *gpu, sizeof(FrameData),
                vk::BufferUsageFlagBits::eShaderDeviceAddress,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
            auto* mapped = static_cast<FrameData*>(data.memory.mapMemory(0, sizeof(FrameData)));

            // Starts zeroed: the first wait on each frame reads it before
            // the GPU has written it.
            Buffer cull_totals = create_buffer(device, *gpu, sizeof(CullTotals), vk::BufferUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
            auto* totals = static_cast<CullTotals*>(cull_totals.memory.mapMemory(0, sizeof(CullTotals)));
            *totals = CullTotals{.visible_draws = 0, .commands = 0};

            frames.push_back(Frame{
                .commands = std::move(commands),
                .image_acquired = vk::raii::Semaphore(device, vk::SemaphoreCreateInfo{}),
                // Start signalled, so the first wait on each frame returns straight away.
                .done = vk::raii::Fence(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled}),
                .data = std::move(data),
                .mapped = mapped,
                .cull_totals = std::move(cull_totals),
                .totals = totals,
            });
        }

        // --- Scene -----------------------------------------------------------

        // The glTF file to draw, under lecture-md/game-engine/assets.
        const std::filesystem::path scene_file = std::filesystem::path(ASSET_DIR) / "Sponza/Sponza.gltf";

        // Where in the world the scene is placed, in metres: its origin. Move
        // it far away, to {100000.0, 0.0, 100000.0} say, 141 km out, and the
        // image stays the same: everything is drawn relative to the camera.
        const glm::dvec3 scene_origin{0.0, 0.0, 0.0};

        const std::uint64_t load_start = SDL_GetTicksNS();
        Scene scene = load_gltf(scene_file, scene_origin);
        std::println("Loaded {}: {} vertices, {} triangles, {} primitives, {} draws, {} materials, {} images",
            scene_file.filename().string(), scene.vertices.size(), scene.indices.size() / 3,
            scene.primitives.size(), scene.draws.size(), scene.materials.size(), scene.images.size());

        // Ground under the scene, out to 32 km each way: something to see the
        // air over, as far as the aerial perspective reaches. 1 cm below the
        // scene's lowest point, so it never fights the scene's own floor.
        add_ground(scene, scene_origin, static_cast<double>(scene.bounds_min.y) - 0.01, 65536.0, 1024.0);

        // Each draw's matrices, triangles and box, and what the cull needs to
        // group it. The normal matrix is the transposed inverse of the model
        // matrix: under non-uniform scale, transforming a normal by the model
        // matrix itself would tilt it off the surface.
        std::vector<DrawData> draw_data;
        std::vector<CullDraw> cull_draws;
        std::array<std::size_t, 3> mode_draws{};  // how many draws of each alpha mode

        for (const MeshDraw& draw : scene.draws) {
            const Primitive& primitive = scene.primitives[draw.primitive];
            const SceneMaterial& material = scene.materials[primitive.material];
            const std::uint32_t list = draw_list_index(material.alpha_mode, material.double_sided, draw.mirrored);

            draw_data.push_back(DrawData{
                .model = draw.model,
                .normal_matrix = glm::transpose(glm::inverse(draw.model)),
                .material = primitive.material,
                .first_index = primitive.first_index,
                .vertex_offset = primitive.vertex_offset,
                .cell = draw.cell,
                .bounds_min = draw.bounds_min,
                .bounds_max = draw.bounds_max,
            });
            cull_draws.push_back(CullDraw{
                .list = list,
                .primitive = draw.primitive,
                .index_count = primitive.index_count,
                .first_index = primitive.first_index,
                .vertex_offset = primitive.vertex_offset,
            });
            ++mode_draws[static_cast<std::size_t>(material.alpha_mode)];
        }

        std::println("Draws: {} opaque, {} masked, {} blended", mode_draws[0], mode_draws[1], mode_draws[2]);

        // Vertices and draw data are read through pointers; indices go to the
        // GPU's index fetch, so that buffer is an index buffer. Vertices and
        // indices are also what the acceleration structures are built from,
        // and shadow rays read indices through a pointer too.
        const vk::BufferUsageFlags build_input = vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR
            | vk::BufferUsageFlagBits::eShaderDeviceAddress;
        const Buffer vertex_buffer = upload_buffer(device, *gpu, queue, command_pool,
            std::as_bytes(std::span(scene.vertices)), build_input);
        const Buffer index_buffer = upload_buffer(device, *gpu, queue, command_pool,
            std::as_bytes(std::span(scene.indices)), vk::BufferUsageFlagBits::eIndexBuffer | build_input);
        const Buffer draw_buffer = upload_buffer(device, *gpu, queue, command_pool,
            std::as_bytes(std::span(draw_data)), vk::BufferUsageFlagBits::eShaderDeviceAddress);

        // The cull, its groups and its draw lists, worked out for this scene's draws.
        const DrawCulling culling = create_draw_culling(device, *gpu, queue, command_pool, cull_draws);

        std::println("Culling: {} draws in {} groups", culling.draw_count, culling.group_count);

        // Most files have no lights, and a buffer can't be empty: then there's
        // no buffer, and the shader's light count is 0.
        const Buffer light_buffer = scene.lights.empty() ? Buffer{} : upload_buffer(device, *gpu, queue, command_pool,
            std::as_bytes(std::span(scene.lights)), vk::BufferUsageFlagBits::eShaderDeviceAddress);

        std::println("Lights: {} from the file, plus the sun", scene.lights.size());

        // --- Acceleration structures -----------------------------------------

        const std::uint64_t acceleration_start = SDL_GetTicksNS();
        // Around the cell the camera starts in: the scene's origin.
        AccelerationStructures acceleration = build_acceleration_structures(device, *gpu, queue, command_pool,
            scene, vertex_buffer, index_buffer, to_cell(scene_origin).cell);

        std::println("Acceleration structures: {} BLAS, {} instances in {:.0f} ms", acceleration.blases.size(),
            scene.draws.size(), static_cast<double>(SDL_GetTicksNS() - acceleration_start) * 1e-6);

        // --- Textures and materials ------------------------------------------

        // Decode every image, upload them with mipmaps, and describe them in
        // the descriptor heap. Texture 0 is white; scene image i is texture i + 1.
        const std::uint64_t texture_start = SDL_GetTicksNS();
        const std::vector<Texture> textures = create_scene_textures(device, *gpu, queue, command_pool, scene);
        // After the textures: the swapchain images' slots, then the environment's.
        DescriptorHeaps heaps = create_descriptor_heaps(device, *gpu, queue, command_pool, textures, scene.samplers,
            screen_slot_count + environment_slot_count);

        const auto first_screen_slot = static_cast<std::uint32_t>(textures.size());
        const ScreenSlots screen{
            .hdr = first_screen_slot,
            .depth = first_screen_slot + 1,
            .normals = first_screen_slot + 2,
            .ao = first_screen_slot + 3,
            .ao_targets = {
                .ao_depth = first_screen_slot + 4,
                .ao_normals = first_screen_slot + 5,
                .ao_raw = first_screen_slot + 6,
                .ao_blur = first_screen_slot + 7,
                .ao = first_screen_slot + 8,
            },
            .accum = first_screen_slot + 9,
            .reveal = first_screen_slot + 10,
        };

        // The swapchain's images are recreated with it, so their descriptors
        // are rewritten every time: after this, only while the GPU is idle.
        const auto describe_screen = [&] {
            const auto whole = [](const Image& image, vk::ImageAspectFlags aspect) {
                return vk::ImageViewCreateInfo{
                    .image = *image.handle,
                    .viewType = vk::ImageViewType::e2D,
                    .format = image.format,
                    .subresourceRange = {.aspectMask = aspect, .levelCount = 1, .layerCount = 1},
                };
            };
            constexpr auto color = vk::ImageAspectFlagBits::eColor;
            constexpr auto storage = vk::DescriptorType::eStorageImage;

            write_image_descriptor(device, heaps, screen.hdr, whole(swapchain.hdr, color));
            write_image_descriptor(device, heaps, screen.depth, whole(swapchain.depth, vk::ImageAspectFlagBits::eDepth));
            write_image_descriptor(device, heaps, screen.normals, whole(swapchain.normals, color));
            write_image_descriptor(device, heaps, screen.ao, whole(swapchain.ao, color));
            write_image_descriptor(device, heaps, screen.ao_targets.ao_depth, whole(swapchain.ao_depth, color), storage);
            write_image_descriptor(device, heaps, screen.ao_targets.ao_normals, whole(swapchain.ao_normals, color), storage);
            write_image_descriptor(device, heaps, screen.ao_targets.ao_raw, whole(swapchain.ao_raw, color), storage);
            write_image_descriptor(device, heaps, screen.ao_targets.ao_blur, whole(swapchain.ao_blur, color), storage);
            write_image_descriptor(device, heaps, screen.ao_targets.ao, whole(swapchain.ao, color), storage);
            write_image_descriptor(device, heaps, screen.accum, whole(swapchain.accum, color));
            write_image_descriptor(device, heaps, screen.reveal, whole(swapchain.reveal, color));
        };

        describe_screen();

        // recreate_swapchain() waits for the GPU to go idle, so the slots are
        // free to rewrite straight afterwards.
        const auto resize = [&] {
            recreate_swapchain(swapchain, device, *gpu, surface, window.get());
            describe_screen();
        };

        // --- The environment -------------------------------------------------

        const std::uint64_t environment_start = SDL_GetTicksNS();
        Environment environment = create_environment(device, *gpu, queue, command_pool, heaps, first_screen_slot + screen_slot_count,
            std::filesystem::path(ASSET_DIR) / "environments/kloppenheim_06_puresky_2k.hdr");

        std::println("Environment: {:.0f} ms", static_cast<double>(SDL_GetTicksNS() - environment_start) * 1e-6);

        std::println("Textures: {} in {:.0f} ms (whole load {:.0f} ms)", textures.size(),
            static_cast<double>(SDL_GetTicksNS() - texture_start) * 1e-6,
            static_cast<double>(SDL_GetTicksNS() - load_start) * 1e-6);

        // Heap indices are one past the scene's: image i is texture i + 1 and
        // sampler i is sampler i + 1, so "none" (-1) becomes 0, the white
        // texture or the default sampler.
        const auto slot = [](const TextureRef& ref) {
            return TextureSlot{
                .texture = static_cast<std::uint32_t>(ref.image + 1),
                .sampler = static_cast<std::uint32_t>(ref.sampler + 1),
                .uv_set = ref.uv_set,
            };
        };

        std::vector<Material> materials;

        for (const SceneMaterial& material : scene.materials) {
            materials.push_back(Material{
                .base_color_factor = material.base_color_factor,
                .emissive_factor = material.emissive_factor,
                .metallic_factor = material.metallic_factor,
                .roughness_factor = material.roughness_factor,
                .normal_scale = material.normal_scale,
                .occlusion_strength = material.occlusion_strength,
                .alpha_cutoff = material.alpha_cutoff,
                .double_sided = material.double_sided ? 1u : 0u,
                .alpha_mode = material.alpha_mode,
                .base_color = slot(material.base_color),
                .metallic_roughness = slot(material.metallic_roughness),
                .normal = slot(material.normal),
                .occlusion = slot(material.occlusion),
                .emissive = slot(material.emissive),
            });
        }

        const Buffer material_buffer = upload_buffer(device, *gpu, queue, command_pool,
            std::as_bytes(std::span(materials)), vk::BufferUsageFlagBits::eShaderDeviceAddress);

        // Spawns at the scene's origin, looking down -Z.
        FlyCamera camera{.position = scene_origin};
        CameraInput input;
        Settings settings;
        Settings shown_settings{.hours = -1.0f};  // what the title shows; differs at first
        CullTotals shown_totals{};                // the cull's totals the title shows
        Settings sky_settings{.hours = -1.0f};    // what the environment was built for
        float sky_altitude = 0.0f;                // the camera's height it was built for

        std::uint64_t previous_ticks = SDL_GetTicksNS();

        // --- Frame loop ------------------------------------------------------

        std::uint64_t frame_count = 0;

        while (poll_events(window.get(), input, settings)) {
            // Minimised: nothing to draw into, so sleep until something happens.
            int width = 0;
            int height = 0;
            SDL_GetWindowSizeInPixels(window.get(), &width, &height);

            if (width == 0 || height == 0 || (SDL_GetWindowFlags(window.get()) & SDL_WINDOW_MINIMIZED)) {
                SDL_WaitEvent(nullptr);
                continue;
            }

            if (width != swapchain.window_width || height != swapchain.window_height) {
                resize();
            }

            // --- Update -----------------------------------------------------

            // Seconds since the last frame, so movement doesn't depend on frame rate.
            const std::uint64_t ticks = SDL_GetTicksNS();
            const float seconds = static_cast<float>(ticks - previous_ticks) * 1e-9f;
            previous_ticks = ticks;

            update_camera(camera, input, seconds);

            const float aspect = static_cast<float>(swapchain.extent.width) / static_cast<float>(swapchain.extent.height);

            // The sky: rebuilt whenever its source changes, or the time of day
            // moves the sun in the simulated one. That takes a few milliseconds
            // and waits for the GPU, which is fine for a key press.
            const glm::vec3 sun_direction = sun_direction_at(settings.hours);
            // The camera's height above the ground, which is at y = 0. The
            // simulated sky is lit for it too, so climbing far enough, 100 m,
            // rebuilds it.
            const auto altitude = static_cast<float>(std::max(camera.position.y, 1.0));
            const bool sky_moved = settings.sky == SkySource::atmosphere
                && (settings.hours != sky_settings.hours || std::abs(altitude - sky_altitude) > 100.0f);

            if (settings.sky != sky_settings.sky || sky_moved) {
                update_environment(environment, device, queue, heaps, settings.sky, sun_direction, altitude);
                sky_settings = settings;
                sky_altitude = altitude;
            }

            // Light and exposure. The meter reads the light falling on flat
            // ground: the sky's irradiance on an upward-facing surface, plus
            // the sun's share at its angle (Rec. 709 luminance of each).
            // Compensation works like a camera's: +1 is a stop brighter, which
            // means a lower EV (EV measures the light the camera expects).
            const auto luminance = [](glm::vec3 c) { return 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b; };
            const glm::vec3 sun_illuminance = environment.mapped->sun_illuminance;
            const float ground_illuminance = luminance(sky_irradiance(*environment.mapped, {0.0f, 1.0f, 0.0f}))
                + luminance(sun_illuminance) * std::max(sun_direction.y, 0.0f);

            const float ev100 = std::clamp(metered_ev100(ground_illuminance), -2.0f, 16.0f) - settings.exposure_compensation;
            const float exposure = exposure_from_ev100(ev100);

            // --- Render -----------------------------------------------------

            Frame& frame = frames[frame_count % frames_in_flight];

            // 1. Wait until the GPU is done with this frame's command buffer and
            //    data from last time, then write this frame's data.
            (void)device.waitForFences(*frame.done, vk::True, no_timeout);

            // What the cull kept the last time this frame's resources were
            // used: two frames ago.
            const CullTotals totals = *frame.totals;

            // The title shows the view, the sky, the time, the exposure,
            // whether ambient occlusion is on and what the cull kept,
            // whenever one changes.
            if (settings.view != shown_settings.view || settings.sky != shown_settings.sky
                || settings.hours != shown_settings.hours
                || settings.exposure_compensation != shown_settings.exposure_compensation
                || settings.ambient_occlusion != shown_settings.ambient_occlusion
                || totals.visible_draws != shown_totals.visible_draws || totals.commands != shown_totals.commands) {
                const auto minutes = static_cast<int>(std::lround(settings.hours * 60.0f));
                const std::string sky = settings.sky == SkySource::atmosphere
                    ? std::format("sky {:02}:{:02}", minutes / 60, minutes % 60)
                    : std::string("photographed sky");
                const std::string title = std::format("game-engine: {}, {}, EV {:.1f}, AO {}, drawn {} of {} in {} commands",
                    view_names[static_cast<std::size_t>(settings.view)], sky, ev100,
                    settings.ambient_occlusion ? "on" : "off", totals.visible_draws, culling.draw_count, totals.commands);

                SDL_SetWindowTitle(window.get(), title.c_str());
                shown_settings = settings;
                shown_totals = totals;
            }

            // Where the camera is: a cell and an offset, like everything the
            // GPU places. Once it's far from the TLAS's origin cell, the TLAS
            // is rebuilt around the camera's cell. That waits for the GPU to
            // stop using the old one: a brief pause, once per kilometre or so.
            const CellPosition camera_at = to_cell(camera.position);

            if (glm::any(glm::greaterThan(glm::abs(camera_at.cell - acceleration.origin_cell), glm::ivec3(tlas_reach_cells)))) {
                device.waitIdle();
                build_tlas(device, *gpu, queue, command_pool, acceleration, scene, camera_at.cell);
                std::println("TLAS rebuilt around cell ({}, {}, {})", camera_at.cell.x, camera_at.cell.y, camera_at.cell.z);
            }

            // The view-projection matrix works in camera-relative space: the
            // view only turns the world, the camera being at its origin.
            const glm::mat4 view_projection = camera.projection(aspect) * camera.view();

            *frame.mapped = FrameData{
                .view_projection = view_projection,
                .inverse_view_projection = glm::inverse(view_projection),
                .vertices = vertex_buffer.address,
                .indices = index_buffer.address,
                .draws = draw_buffer.address,
                .instances = culling.instances.address,
                .materials = material_buffer.address,
                .lights = light_buffer.address,
                .environment = environment.info.address,
                .scene_tlas = acceleration.tlas_address,
                .camera_cell = camera_at.cell,
                .exposure = exposure,
                .sun_direction = sun_direction,
                .light_count = static_cast<std::uint32_t>(scene.lights.size()),
                .sun_illuminance = sun_illuminance,
                .view = settings.view,
                .sky_cube = environment.slots.sky_cube,
                .specular_cube = environment.slots.specular_cube,
                .brdf_lut = environment.slots.brdf_lut,
                .clamp_sampler = environment.clamp_sampler,
                .specular_mips = specular_mips,
                .sun_angular_radius = sun_angular_radius,
                .ambient_occlusion = screen.ao,
                .ao_enabled = settings.ambient_occlusion ? 1u : 0u,
                .camera_offset = camera_at.offset,
                .tlas_offset = glm::vec3(camera.position - glm::dvec3(acceleration.origin_cell) * cell_size),
                .sky_view = environment.slots.sky_view,
                .aerial_inscatter = environment.slots.aerial_inscatter,
                .aerial_transmittance = environment.slots.aerial_transmittance,
                .atmosphere = settings.sky == SkySource::atmosphere ? 1u : 0u,
            };

            const DrawList draws{
                .index_buffer = *index_buffer.handle,
                .frame = frame.data.address,
                .screen = screen,
                .view = settings.view,
                .readback = *frame.cull_totals.handle,
                .see_through = mode_draws[static_cast<std::size_t>(AlphaMode::blend)] > 0,
                .sun_direction = sun_direction,
                .altitude = altitude,
            };

            // 2. Ask the swapchain for an image; `image_acquired` is signalled once it's free.
            const auto acquired = swapchain.handle.acquireNextImage(no_timeout, *frame.image_acquired);

            if (acquired.result == vk::Result::eErrorOutOfDateKHR) {
                resize();
                continue;
            }

            const std::uint32_t image_index = acquired.value;
            device.resetFences(*frame.done);

            // 3. Record and submit: wait for the image, draw, signal `rendered` and `done`.
            record_frame(frame.commands, swapchain, image_index, pipelines, ambient_occlusion, culling, environment, heaps, draws);

            const vk::SemaphoreSubmitInfo wait_info{
                .semaphore = *frame.image_acquired,
                .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            };
            const vk::CommandBufferSubmitInfo command_info{.commandBuffer = *frame.commands};
            const vk::SemaphoreSubmitInfo signal_info{
                .semaphore = *swapchain.rendered[image_index],
                .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            };

            queue.submit2(vk::SubmitInfo2{
                .waitSemaphoreInfoCount = 1,
                .pWaitSemaphoreInfos = &wait_info,
                .commandBufferInfoCount = 1,
                .pCommandBufferInfos = &command_info,
                .signalSemaphoreInfoCount = 1,
                .pSignalSemaphoreInfos = &signal_info,
            }, *frame.done);

            // 4. Present once `rendered` is signalled.
            const vk::Semaphore rendered = *swapchain.rendered[image_index];
            const vk::SwapchainKHR swapchain_handle = *swapchain.handle;

            const vk::Result presented = queue.presentKHR(vk::PresentInfoKHR{
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = &rendered,
                .swapchainCount = 1,
                .pSwapchains = &swapchain_handle,
                .pImageIndices = &image_index,
            });

            if (presented == vk::Result::eErrorOutOfDateKHR || presented == vk::Result::eSuboptimalKHR) {
                resize();
            }

            ++frame_count;
        }

        // --- Shutdown --------------------------------------------------------

        // Everything above is destroyed on the way out of this scope; the GPU must be idle first.
        device.waitIdle();
        std::println("Presented {} frames", frame_count);
    } catch (const std::exception& e) {
        std::println(stderr, "Error: {}", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```

## 14.8 Build and run

```bash
./game-engine/build.bash
```

Press Enter (option 7) to build in debug with clang and run.

**Expected:**
- **Sponza** stands on a plain of bare ground. Inside the courtyard, it looks almost as before; rough metal and stone are a little brighter, from the energy compensation, and the sky is brighter overhead, from multiple scattering.
- **Fly up and away,** a few hundred metres up and a few kilometres out, and look back. The ground fades toward the sky's color with distance and meets it at the horizon. At noon the haze is pale blue-grey; toward sunset it turns warm.
- **Change the hour:** at dusk the sky keeps its glow for a while after the sun has set, and the ground, in the planet's shadow, goes dark, faintly lit by the sky.
- **Climb** a few kilometres: the horizon drops, the haze below thickens toward it, and the sky overhead darkens.
- **Switch to the photographed sky:** the background and the light are the photograph's, and there's no haze.
- **No `[validation …]` lines.**

Next, in Chapter 15, distance anti-aliasing: what happens to fine detail, specular highlights and alpha-tested foliage when it's far enough away to be smaller than a pixel.
