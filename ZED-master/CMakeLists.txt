# CMAKE_MINIMUM_REQUIRED(VERSION 3.16)
# PROJECT(ZED)

# # set the LANGUAGE
# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS ON)
# set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
# set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.8)

# set(CMAKE_BUILD_TYPE Release)
# # set(CMAKE_BUILD_TYPE Debug)

# find_package(ZED 3 REQUIRED)
# find_package(OpenCV REQUIRED)
# find_package(CUDA REQUIRED)
# find_package(Threads REQUIRED)
# find_package(PCL 1.10 REQUIRED COMPONENTS common io visualization)

# include_directories(${CUDA_INCLUDE_DIRS})
# include_directories(${ZED_INCLUDE_DIRS})
# include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(${PCL_INCLUDE_DIRS})
# include_directories(include)

# link_directories(${ZED_LIBRARY_DIR})
# link_directories(${CUDA_LIBRARY_DIRS})
# link_directories(${PCL_LIBRARY_DIRS})
# add_definitions(${PCL_DEFINITIONS})

# FILE(GLOB_RECURSE SRC_FILES src/*.cpp  src/Zed_cuda.cu)
# FILE(GLOB_RECURSE HDR_FILES include/*.hpp include/*.h)

# CUDA_ADD_EXECUTABLE(run main.cpp ${HDR_FILES} ${SRC_FILES})
# CUDA_ADD_EXECUTABLE(detect detect.cpp ${HDR_FILES} ${SRC_FILES})
# target_link_libraries(run ${ZED_LIBRARIES})
# target_link_libraries(run ${OpenCV_LIBS})
# target_link_libraries(run Threads::Threads)
# target_link_libraries(run ${PCL_LIBRARIES})

##############version directed by lyk
cmake_minimum_required(VERSION 3.16)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc" CACHE FILEPATH "" FORCE)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89")
project(ZED LANGUAGES CXX CUDA)

# ---------- 编译选项 ----------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)  # 或 Debug
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---------- 依赖 ----------
find_package(ZED 3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(Threads REQUIRED)
find_package(TensorRT-YOLO REQUIRED)


# ---------- 头文件路径 ----------
include_directories(
    ${ZED_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${TensorRT-YOLO_INCLUDE_DIRS}     # ← 把 YOLO 的 include 放到这里
    ${CMAKE_SOURCE_DIR}/include       # 这里面有 Zed.hpp / trtyolo.hpp 等
)
add_definitions(${PCL_DEFINITIONS})
# 编译工具链配置
#-------------------------------------------------------------------------------
function(set_target_compile_options target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/O2 /Oi /fp:fast>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/Od /Ob0 /Zi /RTC1>
        )
        set_target_properties(${target} PROPERTIES MSVC_RUNTIME_LIBRARY
            "$<$<CONFIG:Debug>:MultiThreadedDebugDLL>$<$<CONFIG:Release>:MultiThreadedDLL>"
        )
    else()
        target_compile_options(${target} PRIVATE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3 -march=native -flto=auto -DNDEBUG>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0 -g3 -fno-omit-frame-pointer -fno-inline>
        )
        target_link_options(${target} PRIVATE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3 -flto=auto>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-g3>
        )
    endif()

    target_compile_definitions(${target} PRIVATE
        $<$<CONFIG:Debug>:DEBUG>
        $<$<NOT:$<CONFIG:Debug>>:NDEBUG>
    )
endfunction()

add_definitions(${PCL_DEFINITIONS})

# ---------- 源文件 ----------
file(GLOB_RECURSE SRC_FILES  src/*.cpp)
file(GLOB_RECURSE CUDA_FILES src/*.cu)
file(GLOB_RECURSE HDR_FILES  include/*.hpp include/*.h)

# ---------- 可执行文件 ----------
add_executable(run     main.cpp    ${CUDA_FILES} ${SRC_FILES} ${HDR_FILES})
add_executable(detect  detect.cpp  ${CUDA_FILES} ${SRC_FILES} ${HDR_FILES})
add_executable(detect01 detect01.cpp ${CUDA_FILES} ${SRC_FILES} ${HDR_FILES})


# ---------- 公共链接库 ----------
set(COMMON_LIBS
    ${ZED_LIBRARIES}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    Threads::Threads
    trtyolo # ← 注意这里用 LIBS，不是 INCLUDE_DIRS
)

# ---------- 目标链接 ----------
target_link_libraries(run     PRIVATE ${COMMON_LIBS})
target_link_libraries(detect  PRIVATE ${COMMON_LIBS})
target_link_libraries(detect01 PRIVATE ${COMMON_LIBS} pcl_sample_consensus)

