
#if __clang__
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
    #pragma clang diagnostic ignored "-Wsign-conversion"
    #pragma clang diagnostic ignored "-Wimplicit-int-conversion"
    #pragma clang diagnostic ignored "-Wundef"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "ic_ssimulacra2.h"

#include "ic_shared.h"

#include <stdio.h>
#include <stdlib.h>


int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <original.png> <distorted.png> <error_map.png>\n", argv[0]);
        return 1;
    }

    const char* orig_filename = argv[1];
    const char* dist_filename = argv[2];
    const char* error_filename = argv[3];

    int w1, h1, c1;
    u8* orig = stbi_load(orig_filename, &w1, &h1, &c1, 4); // force RGBA (ssimulacra2 API)
    defer { stbi_image_free(orig); };
    if (!orig) {
        fprintf(stderr, "Failed to load original image: %s\n", orig_filename);
        return 1;
    }

    int w2, h2, c2;
    u8* dist = stbi_load(dist_filename, &w2, &h2, &c2, 4); // force RGBA (ssimulacra2 API)
    defer { stbi_image_free(dist); };
    if (!dist) {
        fprintf(stderr, "Failed to load distorted image: %s\n", dist_filename);
        return 1;
    }

    if (w1 != w2 || h1 != h2) {
        fprintf(stderr, "Image dimensions must match (%dx%d vs %dx%d)\n", w1, h1, w2, h2);
        return 1;
    }

    int w = w1, h = h1;
    size_t pixel_count = (size_t)w * (size_t)h;
    u8* error_map = (u8*)malloc(4 * pixel_count);
    defer { free(error_map); };
    if (!error_map) {
        fprintf(stderr, "Failed to allocate memory for error map\n");
        return 1;
    }

    void* scratch = malloc(ic_ssimulacra2_score_scratch_size(w, h));
    defer { free(scratch); };
    if (!scratch) {
        fprintf(stderr, "Failed to allocate scratch buffer\n");
        return 1;
    }

    double score = ic_ssimulacra2_score(w, h, orig, dist, scratch, error_map);
    printf("SSIMULACRA2 score: %f\n", score);

    if (!stbi_write_png(error_filename, w, h, 4, error_map, w * 4)) {
        fprintf(stderr, "Failed to write error map image: %s\n", error_filename);
    }

    return 0;
}
