// Copyright 2022-2026 Ludicon LLC. All Rights Reserved.
#pragma once

// Disable vars in Final/no-developer builds.
#if FINAL && !ENABLE_DEVELOPER && !defined(IC_DISABLE_VARS)
    #define IC_DISABLE_VARS 1
#endif

// Example usage:
// IC_VAR_BOOL(load_renderdoc, false);

// We use a namespace instead of a unique name. This allows detecting conflicts between variables with the same name
// at compile time.

#ifdef IC_DISABLE_VARS
#define IC_VAR(type, name, value) namespace var { const type name = value; }
#define IC_VAR_DECL(type, name) namespace var { extern const type name; }
#else
#define IC_VAR(type, name, value) namespace var { type name = value; } namespace startup { static VarRegister name(#name, &var::name); }
#define IC_VAR_DECL(type, name) namespace var { extern type name; }
#endif

#define IC_VAR_BOOL(name, value) IC_VAR(bool, name, value)
#define IC_VAR_INT(name, value) IC_VAR(int, name, value)
#define IC_VAR_FLOAT(name, value) IC_VAR(float, name, value)

//#define IC_VAR_DECL(type, name) namespace var { extern type name; }

#if !IC_DISABLE_VARS

    void ic_vars_register(const char* name, bool * v);
    void ic_vars_register(const char* name, int * v);
    void ic_vars_register(const char* name, float * v);

    struct VarRegister {
        VarRegister(const char* name, bool * v) { ic_vars_register(name, v); }
        VarRegister(const char* name, int * v) { ic_vars_register(name, v); }
        VarRegister(const char* name, float * v) { ic_vars_register(name, v); }
    };

#endif

// Load config file and optionally add to watch list.
void ic_vars_load_config(const char* config_file, bool watch);


// Reload watched config files if contents have changed. Returns true if anything has changed.
bool ic_vars_update();

// Restore all variables to their initial values.
void ic_vars_reset();

// Save the current values as the initial values that we can restore later.
void ic_vars_save_values();


#if IC_DISABLE_VARS

    inline void ic_vars_load_config(const char* config_file, bool watch) {}
    inline bool ic_vars_update() { return false; }
    inline void ic_vars_reset() {}
    inline void ic_vars_save_values() {}

#endif