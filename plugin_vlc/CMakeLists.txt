cmake_minimum_required(VERSION 3.0)
project(plugin_vlc)

set(PLUGIN_NAME intergrain_plugin)
if(MINGW OR UNIX)
    add_compile_options("-fPIC")
endif()

# Indiquez à CMake où trouver les headers et les bibliothèques de VLC :
set(VLC_SDK "" CACHE PATH "Root dir to the VLC SDK") 
include_directories(${VLC_SDK}/include)
include_directories(${VLC_SDK}/include/vlc/plugins)
include_directories(../src)


#Define plugin sources and lib
set(SOURCES plugin.c plugin.h)
set(libsources ../src/vfgs_fw.c ../src/vfgs_hw.c)
add_library(libvfgs ${libsources})
add_library(${PLUGIN_NAME} SHARED ${SOURCES})
target_link_directories(${PLUGIN_NAME} PUBLIC ${VLC_SDK}/lib)
target_link_directories(${PLUGIN_NAME} PUBLIC ${VLC_SDK}/lib64)

# Ajoutez les bibliothèques VLC auxquelles votre plugin doit être lié :
target_link_libraries(${PLUGIN_NAME} vlc vlccore libvfgs)
target_compile_definitions(${PLUGIN_NAME} PRIVATE __PLUGIN__)
