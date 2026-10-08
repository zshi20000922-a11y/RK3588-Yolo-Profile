cmake_minimum_required(VERSION 3.4.1)

project(rknn_yolov5_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pthread")

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5_demo_${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../../runtime/RK3588/${CMAKE_SYSTEM_NAME}/librknn_api
    CACHE PATH "RKNN API SDK directory (must provide include/rknn_api.h)")
set(LIB_ARCH aarch64)
set(RKNN_RT_LIB ${CMAKE_SOURCE_DIR}/include/librknnrt.so
    CACHE FILEPATH "Path to the board-matched librknnrt.so")

#rga
set(RGA_PATH ${CMAKE_SOURCE_DIR}/include/3rdparty/rga/RK3588
    CACHE PATH "RGA SDK root")
set(RGA_LIB ${RGA_PATH}/lib/Linux/${LIB_ARCH}/librga.so
    CACHE FILEPATH "Path to librga.so")
set(MPP_PATH ${CMAKE_SOURCE_DIR}/include/3rdparty/mpp/RK3588
    CACHE PATH "MPP SDK root")
set(MPP_LIB ${MPP_PATH}/lib/Linux/${LIB_ARCH}/librockchip_mpp.so
    CACHE FILEPATH "Path to librockchip_mpp.so")


include_directories(${RKNN_API_PATH}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/3rdparty)
include_directories(${RGA_PATH}/include)
include_directories(${MPP_PATH}/include/rockchip)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")


# rknn_yolov5_demo
include_directories( ${CMAKE_SOURCE_DIR}/include)

option(BUILD_LEGACY_DEMO "Build the legacy OpenCV throughput demo" OFF)
if(BUILD_LEGACY_DEMO)
find_package(OpenCV REQUIRED)
add_executable(rknn_yolov5_demo
        src/main.cc
        src/postprocess.cc
        src/preprocess.cc
        src/rkYolov5s.cc
)

target_link_libraries(rknn_yolov5_demo
  ${RKNN_RT_LIB}
  ${OpenCV_LIBS}
  ${RGA_LIB}
)
endif()

find_package(Threads REQUIRED)

# Modular RK3588 vision service. Each component is a static library so final
# images can exclude unused hardware and algorithms without changing sources.
option(RKVS_BUILD_SERVICE "Build the modular rk_vision_service" ON)
option(RKVS_WITH_V4L2 "Build local V4L2 DMA-BUF input" ON)
option(RKVS_WITH_RTSP_INPUT "Build RTSP/MPP DMA-BUF input" OFF)
option(RKVS_WITH_DISPLAY "Build Wayland/KMS display output" OFF)
option(RKVS_WITH_MPP_ENCODE "Build MPP encoder" OFF)
option(RKVS_WITH_UDP_OUTPUT "Build MPEG-TS/UDP output" OFF)
option(RKVS_WITH_RTSP_OUTPUT "Build RTSP server output" OFF)
option(RKVS_WITH_YOLOV5 "Build YOLOv5 adapter" ON)
option(RKVS_WITH_YOLOV8 "Build YOLOv8 adapter" ON)
option(RKVS_WITH_YOLO11 "Build YOLO11 adapter" ON)
option(RKVS_WITH_YOLO26 "Build YOLO26 adapter" ON)
option(RKVS_WITH_BYTETRACK "Build per-source ByteTrack" ON)
option(RKVS_WITH_FRAME_DIFF "Build three-frame motion detector" ON)

if(RKVS_WITH_FRAME_DIFF)
  if(CMAKE_CROSSCOMPILING)
    set(RKVS_OPENCV_INCLUDE /usr/include/opencv4)
    add_library(rkvs_opencv_core SHARED IMPORTED)
    add_library(rkvs_opencv_imgproc SHARED IMPORTED)
    add_library(rkvs_opencv_video SHARED IMPORTED)
    set_target_properties(rkvs_opencv_core PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/include/3rdparty/opencv/RK3588/lib/libopencv_core.so.4.9.0)
    set_target_properties(rkvs_opencv_imgproc PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/include/3rdparty/opencv/RK3588/lib/libopencv_imgproc.so.4.9.0)
    set_target_properties(rkvs_opencv_video PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/include/3rdparty/opencv/RK3588/lib/libopencv_video.so.4.9.0)
    set(RKVS_OPENCV_LIBS rkvs_opencv_video rkvs_opencv_imgproc rkvs_opencv_core)
  else()
    find_package(OpenCV REQUIRED COMPONENTS core imgproc video)
    set(RKVS_OPENCV_INCLUDE ${OpenCV_INCLUDE_DIRS})
    set(RKVS_OPENCV_LIBS ${OpenCV_LIBS})
  endif()
endif()

if((RKVS_WITH_UDP_OUTPUT OR RKVS_WITH_RTSP_OUTPUT) AND NOT RKVS_WITH_MPP_ENCODE)
  message(FATAL_ERROR "UDP/RTSP output requires RKVS_WITH_MPP_ENCODE=ON")
endif()

add_library(rkvs_core STATIC
        src/rkvs/config.cc
        src/rkvs/frame_hub.cc
        src/rkvs/fair_scheduler.cc
        src/rkvs/detector_adapter.cc
        src/rkvs/json.cc
        src/rkvs/unix_servers.cc)
target_include_directories(rkvs_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(rkvs_core PUBLIC Threads::Threads)

if(RKVS_WITH_BYTETRACK)
  add_library(rkvs_bytetrack STATIC src/rkvs/byte_tracker.cc)
  target_include_directories(rkvs_bytetrack PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(rkvs_bytetrack PUBLIC rkvs_core)
endif()

if(RKVS_WITH_FRAME_DIFF)
  add_library(rkvs_frame_diff STATIC src/rkvs/frame_difference.cc)
  target_include_directories(rkvs_frame_diff PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_include_directories(rkvs_frame_diff PRIVATE ${RKVS_OPENCV_INCLUDE})
  target_link_libraries(rkvs_frame_diff PUBLIC rkvs_core ${RKVS_OPENCV_LIBS})

  add_library(rkvs_rga_motion STATIC src/rkvs/rga_motion.cc)
  target_include_directories(rkvs_rga_motion PUBLIC
          ${CMAKE_SOURCE_DIR}/include ${RGA_PATH}/include)
  target_include_directories(rkvs_rga_motion PRIVATE ${RKVS_OPENCV_INCLUDE})
  target_link_libraries(rkvs_rga_motion PUBLIC
          rkvs_frame_diff rkvs_core ${RGA_LIB} ${RKVS_OPENCV_LIBS})
endif()

if(RKVS_WITH_DISPLAY OR RKVS_WITH_MPP_ENCODE)
  add_library(rkvs_mosaic STATIC src/rkvs/mosaic_compositor.cc)
  target_include_directories(rkvs_mosaic PUBLIC
          ${CMAKE_SOURCE_DIR}/include ${RGA_PATH}/include)
  target_link_libraries(rkvs_mosaic PUBLIC rkvs_core ${RGA_LIB})
endif()

if(RKVS_WITH_V4L2)
  add_library(rkvs_v4l2_source STATIC src/rkvs/v4l2_source.cc)
  target_include_directories(rkvs_v4l2_source PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(rkvs_v4l2_source PUBLIC rkvs_core Threads::Threads)
endif()

if(RKVS_WITH_RTSP_INPUT OR RKVS_WITH_DISPLAY OR RKVS_WITH_UDP_OUTPUT OR RKVS_WITH_RTSP_OUTPUT)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(RKVS_GST REQUIRED IMPORTED_TARGET
          gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0
          gstreamer-allocators-1.0 gstreamer-rtsp-1.0
          gstreamer-rtsp-server-1.0)
endif()

if(RKVS_WITH_RTSP_INPUT)
  add_library(rkvs_rtsp_source STATIC src/rkvs/rtsp_mpp_source.cc)
  target_include_directories(rkvs_rtsp_source PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(rkvs_rtsp_source PUBLIC rkvs_core PkgConfig::RKVS_GST)
endif()

if(RKVS_WITH_DISPLAY)
  add_library(rkvs_wayland_display STATIC src/rkvs/wayland_display.cc)
  target_include_directories(rkvs_wayland_display PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(rkvs_wayland_display PUBLIC rkvs_core PkgConfig::RKVS_GST)
endif()

if(RKVS_WITH_MPP_ENCODE)
  add_library(rkvs_mpp_encoder STATIC
          src/rkvs/encoded_packet.cc src/rkvs/mpp_encoder.cc)
  target_include_directories(rkvs_mpp_encoder PUBLIC
          ${CMAKE_SOURCE_DIR}/include ${MPP_PATH}/include/rockchip)
  target_link_libraries(rkvs_mpp_encoder PUBLIC rkvs_core ${MPP_LIB})
endif()

if(RKVS_WITH_UDP_OUTPUT)
  add_library(rkvs_udp_output STATIC src/rkvs/udp_output.cc)
  target_include_directories(rkvs_udp_output PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(rkvs_udp_output PUBLIC
          rkvs_core rkvs_mpp_encoder PkgConfig::RKVS_GST)
endif()

if(RKVS_WITH_RTSP_OUTPUT)
  add_library(rkvs_rtsp_output STATIC src/rkvs/rtsp_output.cc)
  target_include_directories(rkvs_rtsp_output PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(rkvs_rtsp_output PUBLIC
          rkvs_core rkvs_mpp_encoder PkgConfig::RKVS_GST)
endif()

if(RKVS_BUILD_SERVICE)
  add_library(rkvs_rknn_detector STATIC src/rkvs/rknn_detector.cc)
  target_include_directories(rkvs_rknn_detector PUBLIC
          ${CMAKE_SOURCE_DIR}/include
          ${RKNN_API_PATH}/include
          ${RGA_PATH}/include)
  target_link_libraries(rkvs_rknn_detector PUBLIC
          rkvs_core ${RKNN_RT_LIB} ${RGA_LIB})

  add_executable(rk_vision_service src/rkvs/main.cc)
  target_include_directories(rk_vision_service PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(rk_vision_service PRIVATE
          rkvs_core rkvs_rknn_detector Threads::Threads)
  add_executable(rkvs_control src/rkvs/rkvs_control.cc)
  add_executable(rk3588_yolo_benchmark tools/rk3588_yolo_benchmark.cc)
  target_include_directories(rk3588_yolo_benchmark PRIVATE
          ${CMAKE_SOURCE_DIR}/include ${RKNN_API_PATH}/include ${RGA_PATH}/include)
  target_link_libraries(rk3588_yolo_benchmark PRIVATE
          rkvs_core rkvs_rknn_detector Threads::Threads ${RKNN_RT_LIB} ${RGA_LIB})
  add_executable(rk3588_yolo_quality tools/rk3588_yolo_quality.cc)
  target_include_directories(rk3588_yolo_quality PRIVATE
          ${CMAKE_SOURCE_DIR}/include ${RKNN_API_PATH}/include ${RGA_PATH}/include)
  target_link_libraries(rk3588_yolo_quality PRIVATE
          rkvs_core rkvs_rknn_detector Threads::Threads ${RKNN_RT_LIB} ${RGA_LIB})
  if(RKVS_WITH_V4L2)
    target_link_libraries(rk_vision_service PRIVATE rkvs_v4l2_source)
    target_compile_definitions(rk_vision_service PRIVATE RKVS_HAS_V4L2=1)
  endif()
  if(RKVS_WITH_RTSP_INPUT)
    target_link_libraries(rk_vision_service PRIVATE rkvs_rtsp_source)
    target_compile_definitions(rk_vision_service PRIVATE RKVS_HAS_RTSP_INPUT=1)
  endif()
  if(RKVS_WITH_DISPLAY)
    target_link_libraries(rk_vision_service PRIVATE rkvs_wayland_display)
    target_compile_definitions(rk_vision_service PRIVATE RKVS_HAS_DISPLAY=1)
  endif()
  if(RKVS_WITH_MPP_ENCODE)
    target_link_libraries(rk_vision_service PRIVATE rkvs_mpp_encoder)
    target_compile_definitions(rk_vision_service PRIVATE RKVS_HAS_MPP_ENCODER=1)
  endif()
  if(RKVS_WITH_UDP_OUTPUT)
    target_link_libraries(rk_vision_service PRIVATE rkvs_udp_output)
    target_compile_definitions(rk_vision_service PRIVATE RKVS_HAS_UDP_OUTPUT=1)
  endif()
  if(RKVS_WITH_RTSP_OUTPUT)
    target_link_libraries(rk_vision_service PRIVATE rkvs_rtsp_output)
    target_compile_definitions(rk_vision_service PRIVATE RKVS_HAS_RTSP_OUTPUT=1)
  endif()
  if(RKVS_WITH_BYTETRACK)
    target_link_libraries(rk_vision_service PRIVATE rkvs_bytetrack)
  endif()
  if(RKVS_WITH_FRAME_DIFF)
    target_link_libraries(rk_vision_service PRIVATE rkvs_rga_motion)
    target_compile_definitions(rk_vision_service PRIVATE RKVS_HAS_FRAME_DIFF=1)
  endif()
  if(RKVS_WITH_DISPLAY OR RKVS_WITH_MPP_ENCODE)
    target_link_libraries(rk_vision_service PRIVATE rkvs_mosaic)
    target_compile_definitions(rk_vision_service PRIVATE RKVS_HAS_MOSAIC=1)
  endif()
endif()

# Low-latency dual-camera service.  It deliberately does not link OpenCV: the
# 4K detection path stays in DMA-BUF/RGA/RKNN memory.
option(BUILD_DUAL_DETECTOR "Build the existing dual-camera detector" ON)
if(BUILD_DUAL_DETECTOR)
add_executable(rknn_dual_camera_detector
        src/dual/main.cc
        src/dual/v4l2_camera.cc
        src/dual/yolo_postprocess.cc
        src/dual/rknn_worker.cc
        src/dual/detector_service.cc
        src/dual/preview_sink.cc
        src/dual/pip_sink.cc
        src/dual/stereo_calibration.cc
        src/dual/cross_camera_matcher.cc
        src/dual/roi_controller.cc
    src/dual/stereo_coordinator.cc
    src/dual/auto_roi_coordinator.cc
)
target_include_directories(rknn_dual_camera_detector PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/dual_detector
)
target_link_libraries(rknn_dual_camera_detector
        ${RKNN_RT_LIB}
        ${RGA_LIB}
        ${MPP_LIB}
        Threads::Threads
)

add_executable(stereo_roi_ctl
        src/dual/stereo_roi_ctl.cc
        src/dual/roi_controller.cc)
target_include_directories(stereo_roi_ctl PRIVATE
        ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/include/dual_detector)
endif()
if(RKVS_BUILD_SERVICE)
  install(TARGETS rk_vision_service rkvs_control rk3588_yolo_benchmark rk3588_yolo_quality DESTINATION ./)
  install(FILES config/rk_vision_service.example.yaml DESTINATION config)
  install(DIRECTORY model/manifests DESTINATION model)
  install(PROGRAMS ${RKNN_RT_LIB} ${RGA_LIB} DESTINATION lib)
endif()


# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5_demo_${CMAKE_SYSTEM_NAME})
if(BUILD_LEGACY_DEMO)
  install(TARGETS rknn_yolov5_demo DESTINATION ./)
endif()
if(BUILD_DUAL_DETECTOR)
  install(TARGETS rknn_dual_camera_detector DESTINATION ./)
  install(TARGETS stereo_roi_ctl DESTINATION ./)
  install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
  install(PROGRAMS ${RGA_LIB} DESTINATION lib)
  install(PROGRAMS ${MPP_LIB} DESTINATION lib)
endif()
install(DIRECTORY model DESTINATION ./)

include(CTest)
if(BUILD_TESTING)
  add_executable(rkvs_core_tests tests/rkvs_core_tests.cc)
  target_include_directories(rkvs_core_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_compile_definitions(rkvs_core_tests PRIVATE
    RKVS_TEST_CONFIG="${CMAKE_SOURCE_DIR}/config/rk_vision_service.example.yaml")
  target_link_libraries(rkvs_core_tests PRIVATE rkvs_core)
  if(RKVS_WITH_BYTETRACK)
    target_link_libraries(rkvs_core_tests PRIVATE rkvs_bytetrack)
  endif()
  if(RKVS_WITH_FRAME_DIFF)
    target_link_libraries(rkvs_core_tests PRIVATE rkvs_frame_diff)
  endif()
  if(NOT CMAKE_CROSSCOMPILING)
    add_test(NAME rkvs_core_tests COMMAND rkvs_core_tests)
  endif()

  if(BUILD_DUAL_DETECTOR)
    add_executable(dual_detector_unit_tests
    tests/dual_detector_tests.cc
    src/dual/v4l2_camera.cc
    src/dual/yolo_postprocess.cc
    src/dual/stereo_calibration.cc
    src/dual/cross_camera_matcher.cc
    src/dual/roi_controller.cc
    src/dual/stereo_coordinator.cc
    src/dual/auto_roi_coordinator.cc)
    target_include_directories(dual_detector_unit_tests PRIVATE
      ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/include/dual_detector)
    target_link_libraries(dual_detector_unit_tests Threads::Threads)
    if(NOT CMAKE_CROSSCOMPILING)
      add_test(NAME dual_detector_unit_tests COMMAND dual_detector_unit_tests)
    else()
      message(STATUS "Cross build: run dual_detector_unit_tests on the RK3588 target")
    endif()
  endif()
endif()
