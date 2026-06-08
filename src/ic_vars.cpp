// Copyright 2022-2026 Ludicon LLC. All Rights Reserved.
#define _CRT_SECURE_NO_WARNINGS

#include "ic_vars.h"
#include "ic_shared.h" // u64, ic_assert

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h> // sscanf, snprintf, printf, fopen, fgets, fclose
#include <stdlib.h> // atoi, atof
#include <string.h> // strcmp
#include <ctype.h> // isspace

#if !IC_DISABLE_VARS

#if IC_OS_WINDOWS
    #define WIN32_LEAN_AND_MEAN
    #define VC_EXTRALEAN
    #define NOMINMAX
    #include <windows.h>
#endif

#if IC_OS_MACOS || IC_OS_ANDROID || IC_OS_IOS || IC_OS_LINUX
    #include <sys/stat.h>
    #include <time.h>
#endif

#define IC_VAR_MAX_COUNT             1024
#define IC_VAR_MAX_FILES             4

enum VarType {
    VarType_Bool,
    VarType_Int,
    VarType_Float,
};

struct Var {
    const char* name;
    VarType type;
    union {
        //void * v;
        bool * b;
        int * i;
        float * f;
    } ptr;
    union {
        bool b;
        int i;
        float f;
    } initial_value;
};

struct VarFile {
    char name[256];
    bool watch;
    u64 modtime;
};

static struct VarSystem {
    u64 initialized;

    int count;
    Var array[IC_VAR_MAX_COUNT];

    int file_count;
    VarFile file[IC_VAR_MAX_FILES];
} vars;


inline void init_vars() {
    if (vars.initialized != 0xBADBADBADBADBEEFULL) {
        vars.initialized = 0xBADBADBADBADBEEFULL;
        vars.count = 0;
        vars.file_count = 0;
    }
}

void ic_vars_register(const char* name, bool * b) {
    init_vars();
    ic_assert(vars.count < IC_VAR_MAX_COUNT);
    vars.array[vars.count].name = name;
    vars.array[vars.count].type = VarType_Bool;
    vars.array[vars.count].ptr.b = b;
    vars.array[vars.count].initial_value.b = *b;
    vars.count++;
}

void ic_vars_register(const char* name, int * i) {
    init_vars();
    ic_assert(vars.count < IC_VAR_MAX_COUNT);
    vars.array[vars.count].name = name;
    vars.array[vars.count].type = VarType_Int;
    vars.array[vars.count].ptr.i = i;
    vars.array[vars.count].initial_value.i = *i;
    vars.count++;
}

void ic_vars_register(const char* name, float * f) {
    init_vars();
    ic_assert(vars.count < IC_VAR_MAX_COUNT);
    vars.array[vars.count].name = name;
    vars.array[vars.count].type = VarType_Float;
    vars.array[vars.count].ptr.f = f;
    vars.array[vars.count].initial_value.f = *f;
    vars.count++;
}


static Var * find_variable(const char* name) {
    for (int i = 0; i < vars.count; i++) {
        if (strcmp(vars.array[i].name, name) == 0) {
            return vars.array + i;
        }
    }
    return nullptr;
}

#if IC_OS_WINDOWS

    static u64 get_file_modtime(const char* filepath) {
        HANDLE hFile = CreateFile(filepath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) {
            return 0;
        }

        FILETIME creation_time, access_time, write_time;
        if (GetFileTime(hFile, &creation_time, &access_time, &write_time) == 0) {
            CloseHandle(hFile);
            return 0;
        }

        CloseHandle(hFile);
        return (u64(write_time.dwHighDateTime) << 32) | u64(write_time.dwLowDateTime);
    }

#endif

#if IC_OS_MACOS || IC_OS_ANDROID || IC_OS_IOS || IC_OS_LINUX

    static u64 get_file_modtime(const char* filepath) {
        struct stat file_stat;
        if (stat(filepath, &file_stat) == -1) {
            return 0;
        }

        return file_stat.st_mtime;
    }

#endif

static void parse_line(const char *line, const char* filepath, int line_number) {
    char name[64], value[64];
    int items_read = sscanf(line, "%63s %63s", name, value);

    if (items_read < 2) {
        printf("%s:%d error: Invalid variable declaration.\n", filepath, line_number);
        return;
    }

    Var * var = find_variable(name);
    if (var == nullptr) {
        printf("%s:%d error: Unknown variable '%s'.\n", filepath, line_number, name);
        return;
    }

    if (var->type == VarType_Bool) {
        *var->ptr.b = (strcmp(value, "true") == 0);
    }
    else if (var->type == VarType_Int) {
        *var->ptr.i = atoi(value);
    }
    else if (var->type == VarType_Float) {
        *var->ptr.f = strtof(value, nullptr);
    }
    else {
        ic_assert(false);
    }
}

static u64 load_config(const char* filepath, u64 previous_modtime) {

    u64 modtime = get_file_modtime(filepath);
    if (modtime == 0 || previous_modtime == modtime) {
        // File does not exist, or has not changed.
        if (modtime == 0) 
            printf("config file %s not found\n", filepath);
        return modtime;
    }

    FILE* fp = fopen(filepath, "rb");
    if (fp == nullptr) {
        printf("unexpected error loading %s\n", filepath);
        return 0;
    }

    // Parse config file.
    int line_number = 0;
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        line_number++;

        // Skip comments or empty lines
        char *trimmed = line;
        while (isspace(*trimmed)) ++trimmed;  // Skip leading whitespace
        if (*trimmed == '#' || *trimmed == '\0') continue;

        // Remove inline comments by finding '#' and terminating the string there
        char *comment = strchr(trimmed, '#');
        if (comment) *comment = '\0';

        // Parse the line into a Variable struct
        parse_line(trimmed, filepath, line_number);
    }

    fclose(fp);

    return modtime;
}

void ic_vars_load_config(const char* filename, bool watch) {
    ic_assert(vars.file_count < IC_VAR_MAX_FILES);

    // Try to find config file in our list.
    int i;
    for (i = 0; i < vars.file_count; i++) {
        if (strcmp(filename, vars.file[i].name) == 0) {
            break;
        }
    }

    // If not found, add a new file item.
    if (i == vars.file_count) {
        snprintf(vars.file[i].name, sizeof(vars.file[i].name), "%s", filename);
        vars.file_count++;
    }

    // Update file attributes.
    vars.file[i].watch = watch;
    vars.file[i].modtime = load_config(filename, 0);
}

bool ic_vars_update() {
    bool any_update = false;

    // Go over all the files, and reload the ones with the watch flag if the file modtime has changed.
    for (int f = 0; f < vars.file_count; f++) {
        if (vars.file[f].watch) {
            u64 new_modtime = load_config(vars.file[f].name, vars.file[f].modtime);
            if (new_modtime != vars.file[f].modtime) {
                vars.file[f].modtime = new_modtime;
                any_update = true;
            }
        }
    }

    return any_update;
}


void ic_vars_reset() {

    // Restore all the initial values.
    for (int i = 0; i < vars.count; i++) {
        Var* var = &vars.array[i];
        if (var->type == VarType_Bool) {
            *var->ptr.b = var->initial_value.b;
        }
        else if (var->type == VarType_Int) {
            *var->ptr.i = var->initial_value.i;
        }
        else if (var->type == VarType_Float) {
            *var->ptr.f = var->initial_value.f;
        }
    }
}

void ic_vars_save_values() {

    for (int i = 0; i < vars.count; i++) {
        Var* var = &vars.array[i];
        if (var->type == VarType_Bool) {
            var->initial_value.b = *var->ptr.b;
        }
        else if (var->type == VarType_Int) {
            var->initial_value.i = *var->ptr.i;
        }
        else if (var->type == VarType_Float) {
            var->initial_value.f = *var->ptr.f;
        }
    }
}

// #else

// void ic_vars_load_config(const char* config_file, bool watch) {}
// bool ic_vars_update() { return false; }
// void ic_vars_reset() {}
// void ic_vars_save_values() {}

#endif // !IC_DISABLE_VARS
