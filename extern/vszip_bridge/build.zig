// Copyright 2026 Ludicon LLC. All Rights Reserved.
//
// Builds a small C-ABI shared library around vapoursynth-zip's standalone
// SSIMULACRA2 filter. We pull in only src/filters/ssimulacra2.zig from the
// sibling submodule (it imports nothing but std), so none of the VapourSynth
// plugin machinery is needed.

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const strip: bool = optimize == .ReleaseFast;

    // The upstream filter, as an importable module.
    const core = b.createModule(.{
        .root_source_file = b.path("../vapoursynth-zip/src/filters/ssimulacra2.zig"),
        .target = target,
        .optimize = optimize,
    });

    const lib = b.addLibrary(.{
        .name = "vszip_ssimu2",
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/wrapper.zig"),
            .target = target,
            .optimize = optimize,
            .strip = strip,
        }),
    });
    lib.root_module.addImport("ssimu2_core", core);
    lib.linkLibC();

    b.installArtifact(lib);
    b.installFile("include/vszip_ssimu2.h", "include/vszip_ssimu2.h");
}
