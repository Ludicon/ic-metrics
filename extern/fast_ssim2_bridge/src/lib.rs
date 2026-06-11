// Copyright 2026 Ludicon LLC. All Rights Reserved.
//
// C-ABI bridge over imazen/fast-ssim2.
// Linearizes sRGB bytes with fast_ssim2's own srgb_u8_to_linear (the canonical
// conversion its ToLinearRgb impls use) and calls compute_ssimulacra2.

use fast_ssim2::{compute_ssimulacra2, srgb_u8_to_linear, LinearRgbImage};

fn rgba8_to_linear(buf: &[u8]) -> Vec<[f32; 3]> {
    buf.chunks_exact(4)
        .map(|c| [srgb_u8_to_linear(c[0]), srgb_u8_to_linear(c[1]), srgb_u8_to_linear(c[2])])
        .collect()
}

/// # Safety
/// `orig` and `dist` must each point to at least `w * h * 4` bytes of RGBA8 data.
#[no_mangle]
pub unsafe extern "C" fn fast_ssim2_score(
    orig: *const u8,
    dist: *const u8,
    w: u32,
    h: u32,
) -> f64 {
    let pixel_count = (w as usize) * (h as usize);
    let orig_slice = std::slice::from_raw_parts(orig, pixel_count * 4);
    let dist_slice = std::slice::from_raw_parts(dist, pixel_count * 4);

    let src = LinearRgbImage::new(rgba8_to_linear(orig_slice), w as usize, h as usize);
    let dst = LinearRgbImage::new(rgba8_to_linear(dist_slice), w as usize, h as usize);

    compute_ssimulacra2(src, dst).unwrap_or(0.0)
}
