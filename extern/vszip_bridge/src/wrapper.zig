// Copyright 2026 Ludicon LLC. All Rights Reserved.
//
// C-ABI bridge over dnjulek/vapoursynth-zip's standalone SSIMULACRA2 filter
// (src/filters/ssimulacra2.zig). That `process` entry point takes three planes
// of *linear* RGB f32, so we linearize the sRGB bytes here (same transfer the
// other backends use) into tightly-packed planar buffers (stride == width).

const std = @import("std");
const ssimu2 = @import("ssimu2_core");

const allocator = std.heap.c_allocator;

fn srgbToLinear(c: f32) f32 {
    return if (c <= 0.04045) c / 12.92 else std.math.pow(f32, (c + 0.055) / 1.055, 2.4);
}

// Deinterleave RGBA8 -> three planar linear-RGB f32 planes packed into `dst`
// (plane p occupies dst[p*n .. (p+1)*n]).
fn rgba8ToLinearPlanar(src: [*]const u8, dst: []f32, n: usize) void {
    var i: usize = 0;
    while (i < n) : (i += 1) {
        dst[0 * n + i] = srgbToLinear(@as(f32, @floatFromInt(src[4 * i + 0])) / 255.0);
        dst[1 * n + i] = srgbToLinear(@as(f32, @floatFromInt(src[4 * i + 1])) / 255.0);
        dst[2 * n + i] = srgbToLinear(@as(f32, @floatFromInt(src[4 * i + 2])) / 255.0);
    }
}

// Returns the SSIMULACRA2 score, or NaN on allocation failure.
// `orig` and `dist` must each point to at least w*h*4 bytes of RGBA8.
export fn vszip_ssimu2_score(orig: [*]const u8, dist: [*]const u8, w: u32, h: u32) f64 {
    const n: usize = @as(usize, w) * @as(usize, h);

    const buf = allocator.alloc(f32, n * 6) catch return std.math.nan(f64);
    defer allocator.free(buf);

    rgba8ToLinearPlanar(orig, buf[0 .. n * 3], n);
    rgba8ToLinearPlanar(dist, buf[n * 3 .. n * 6], n);

    const p1 = [3][]const f32{ buf[0 * n .. 1 * n], buf[1 * n .. 2 * n], buf[2 * n .. 3 * n] };
    const p2 = [3][]const f32{ buf[3 * n .. 4 * n], buf[4 * n .. 5 * n], buf[5 * n .. 6 * n] };

    return ssimu2.process(p1, p2, w, w, h);
}
