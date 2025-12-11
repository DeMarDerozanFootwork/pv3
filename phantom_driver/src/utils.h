/*******************************************************************************
 * @file    pv3.h
 * @date    01/2025
 *
 * @attention Copyright (c) 2022-2025
 * @attention Phantom AI, Inc.
 * @attention All rights reserved.
 *******************************************************************************/

#ifndef PHANTOM_DRIVER_SRC_UTILS_H
#define PHANTOM_DRIVER_SRC_UTILS_H

#include "common.h"

namespace phantom_ai {
namespace driver {
namespace utils {

// Infer the capacity of an array of a known constant size.  The array must have
// had not decayed to a pointer in the context this function is used.

template <typename T, uint32_t N>
inline constexpr uint32_t
capacity(const T (&)[N]) noexcept {
  return N;
}

// Convert a Phantom OS CameraID to a PV3 camera::View.

inline vision::camera::View
convert(const CameraID location) noexcept {
  using View = vision::camera::View;
  View view = View::Max;

  switch (location) {
    case CAM_FRONT_CENTER:
      view = View::Front_Center;
      break;

    case CAM_FRONT_CENTER_CROP:
      view = View::Front_Center_Crop;
      break;

    case CAM_FRONT_CENTER_MIDDLE:
      view = View::Front_Center_Middle;
      break;

    case CAM_FRONT_CENTER_WHOLE:
      view = View::Front_Center_Whole;
      break;

    case CAM_FRONT_CENTER_FULL:
      view = View::Front_Center_Full;
      break;

    case CAM_FRONT_CENTER_NARROW:
      view = View::Front_Center_Narrow;
      break;

    case CAM_FRONT_CENTER_NARROW_CROP:
      view = View::Front_Center_Narrow_Crop;
      break;

    case CAM_FRONT_CENTER_NARROW_FULL:
      view = View::Front_Center_Narrow_Full;
      break;

    case CAM_FRONT_CENTER_SVM:
      view = View::Front_Center_SVM;
      break;

    case CAM_FRONT_CENTER_SVM_CROP:
      view = View::Front_Center_SVM_Crop;
      break;

    case CAM_FRONT_CENTER_SVM_FULL:
      view = View::Front_Center_SVM_Full;
      break;

    case CAM_FRONT_LEFT:
      view = View::Front_Left;
      break;

    case CAM_FRONT_LEFT_CROP:
      view = View::Front_Left_Crop;
      break;

    case CAM_FRONT_RIGHT:
      view = View::Front_Right;
      break;

    case CAM_FRONT_RIGHT_CROP:
      view = View::Front_Right_Crop;
      break;

    case CAM_REAR_CENTER:
      view = View::Rear_Center;
      break;

    case CAM_REAR_CENTER_CROP:
      view = View::Rear_Center_Crop;
      break;

    case CAM_REAR_CENTER_FULL:
      view = View::Rear_Center_Full;
      break;

    case CAM_REAR_CENTER_SVM:
      view = View::Rear_Center_SVM;
      break;

    case CAM_REAR_CENTER_SVM_CROP:
      view = View::Rear_Center_SVM_Crop;
      break;

    case CAM_REAR_CENTER_SVM_FULL:
      view = View::Rear_Center_SVM_Full;
      break;

    case CAM_REAR_SIDE_LEFT:
      view = View::Rear_Left;
      break;

    case CAM_REAR_SIDE_LEFT_CROP:
      view = View::Rear_Left_Crop;
      break;

    case CAM_REAR_SIDE_LEFT_FULL:
      view = View::Rear_Left_Full;
      break;

    case CAM_REAR_SIDE_RIGHT:
      view = View::Rear_Right;
      break;

    case CAM_REAR_SIDE_RIGHT_CROP:
      view = View::Rear_Right_Crop;
      break;

    case CAM_REAR_SIDE_RIGHT_FULL:
      view = View::Rear_Right_Full;
      break;

    case CAM_SIDE_LEFT:
      view = View::Side_Left;
      break;

    case CAM_SIDE_LEFT_FULL:
      view = View::Side_Left_Full;
      break;

    case CAM_SIDE_RIGHT:
      view = View::Side_Right;
      break;

    case CAM_SIDE_RIGHT_FULL:
      view = View::Side_Right_Full;
      break;

    case CAM_SIDE_FRONT_LEFT:
      view = View::Side_Front_Left;
      break;

    case CAM_SIDE_FRONT_LEFT_FULL:
      view = View::Side_Front_Left_Full;
      break;

    case CAM_SIDE_FRONT_RIGHT:
      view = View::Side_Front_Right;
      break;

    case CAM_SIDE_FRONT_RIGHT_FULL:
      view = View::Side_Front_Right_Full;
      break;

    case CAM_SIDE_REAR_LEFT:
      view = View::Side_Rear_Left;
      break;

    case CAM_SIDE_REAR_LEFT_FULL:
      view = View::Side_Rear_Left_Full;
      break;

    case CAM_SIDE_REAR_RIGHT:
      view = View::Side_Rear_Right;
      break;

    case CAM_SIDE_REAR_RIGHT_FULL:
      view = View::Side_Rear_Right_Full;
      break;

    case CAM_GROUND:
      view = View::Ground;
      break;

    case CAM_FRONT_CENTER_CROP_BAYER:
    case CAM_RESERVED:
    case NUM_MAX_CAM_IDS:
    default:
      break;
  }

  PV3_ASSERT(View::Max != view, "Invalid!");
  return view;
}

// Convert a Phantom OS camera_format to a PV3 core::image::Format.

inline vision::core::image::Format
convert(const camera_format fmt) noexcept {
  using Format = vision::core::image::Format;
  Format format = Format::Unknown;

  switch (fmt) {
    case camera_format::format_yuv420sp:
      format = Format::pNV12;
      break;

    case camera_format::format_bgr:
      format = Format::pBGR;
      break;

    case camera_format::format_rgb:
      format = Format::pRGB;

    case camera_format::format_yuv422:
    default:
      break;
  }

  PV3_ASSERT(Format::Unknown != format, "Invalid!");
  return format;
}

// Convert a cv::Mat to its equivalent PV3 image::Handle.  Note that
//   1. PV3 image::Handles are thin objects that simply describe, and point to,
//      a location in memory and do not own the underlying image data and its
//      lifetime by design.  Think of them as a pointer with some metadata.
//   2. The OpenCV matrix owns the underlying memory.
//   3. This function does not copy any substansive amount of data.
//   4. The combination of (1), (2), and (3) means that the matrix must outlive
//      the image handle! Invalid memory accesses will occur otherwise!

inline vision::image::Handle
convert(const cv::Mat mat) noexcept {
  vision::image::Handle image{};

  switch (mat.depth()) {
    case CV_8UC1: {
      // Assuming CV_8UC1 == pNV12.  Not very robust!
      image.format = vision::core::image::Format::pNV12;
      image.width = cast::numeric<uint16_t>(mat.cols);
      image.stride = cast::numeric<uint16_t>(image.width * sizeof(uint8_t));
      image.height = cast::numeric<uint16_t>((mat.rows * 2U) / 3U);

      // Planes
      enum Plane : uint32_t {
        kLuma = 0U,
        kChroma = 1U,
      };

      // Luma
      image.planes[kLuma].as_void_ptr = mat.ptr();
      image.planes[kLuma].size = image.height * image.stride;

      // Chroma
      image.planes[kChroma].as_void_ptr = mat.ptr(image.height);
      image.planes[kChroma].size = image.planes[kLuma].size / 2U;
    } break;

    default:
      break;
  }

  PV3_ASSERT(image, "Invalid!");
  return image;
}

inline void
calculate_yaw_rate_bias(                //
    const EspMeasurements& esp,         //
    const WheelSpeedMeasurements& wsm,  //
    float& rate,                        //
    float& bias,                        //
    float& count) {
  constexpr float kMinNumSamples = 50.0F;
  constexpr float kMaxYawRateCnt = 10000.0F;
  constexpr float kStationaryVehicleSpeedThresholdKph = 1.0F;
  constexpr float kYawRateThresholdInStationarySituationDeg = 1.5F;
  const float wspeed = 0.5F * (wsm.rear_left_ + wsm.rear_right_);

  if ((wspeed < kStationaryVehicleSpeedThresholdKph) &&
      (std::abs(esp.yaw_rate_) < kYawRateThresholdInStationarySituationDeg)) {
    bias = ((bias * count) + esp.yaw_rate_) / (count + 1.0F);
    count = std::min(count + 1.0F, kMaxYawRateCnt);
  }

  rate = (count > kMinNumSamples) ? bias : 0.0F;
}

inline void
decode(                                //
    cv::Mat dst,                       //
    const vision::image::Handle mask,  //
    const vision::core::container::Span<const uint8_t> lut) noexcept {
  // TODO: This is a quick implementation.  To be robust, this code must be
  // amended to handle scenarios where the segmentation mask's width is not a
  // multiple of 64.  Supporting LUTs greater than 16 elements is also possible.

  assert(mask && "Invalid!");
  assert((dst.type() == CV_8UC1) && "Invalid!");
  assert((vision::core::image::Format::Gray == mask.format) && "Invalid!");
  assert((dst.rows == mask.height) && (dst.cols == mask.width) && "Invalid!");
  assert((lut.count() <= 16U) && "Unsupported!");
  const uint8x16_t lut_u8 = vld1q_u8(lut.data());

  // Process one row at a time.
  for (uint32_t row = 0U, col0 = 0U;  //
       row < mask.height;             //
       ++row, col0 = 0U) {
    uint8_t* const __restrict dst_row_ptr = dst.ptr(row);
    const uint8_t* const __restrict mask_row_ptr =  //
        mask.data.as_byte_ptr +                     //
        (row * mask.stride);

    // Process 64 pixels at a time.
    for (uint16_t col1 = 16U, col2 = 32U, col3 = 48U;  //
         (col0 + 64U) <= mask.width;                   //
         col0 += 64U, col1 += 64U, col2 += 64U, col3 += 64U) {
      vst1q_u8(dst_row_ptr + col0,
               vqtbl1q_u8(lut_u8, vld1q_u8(mask_row_ptr + col0)));

      vst1q_u8(dst_row_ptr + col1,
               vqtbl1q_u8(lut_u8, vld1q_u8(mask_row_ptr + col1)));

      vst1q_u8(dst_row_ptr + col2,
               vqtbl1q_u8(lut_u8, vld1q_u8(mask_row_ptr + col2)));

      vst1q_u8(dst_row_ptr + col3,
               vqtbl1q_u8(lut_u8, vld1q_u8(mask_row_ptr + col3)));
    }

    // TODO: Process left-overs here.
  }
}

inline void
visualize(                                                     //
    cv::Mat dst,                                               //
    const cv::Mat src,                                         //
    const cv::Mat mask,                                        //
    const vision::core::container::Span<const uint8_t> b_lut,  //
    const vision::core::container::Span<const uint8_t> g_lut,  //
    const vision::core::container::Span<const uint8_t> r_lut,  //
    const vision::core::container::Span<const uint8_t> a_lut) noexcept {
  // TODO: This is a quick implementation.  To be robust, this code must be
  // amended to handle scenarios where the segmentation mask's width is not a
  // multiple of 8.  Supporting LUTs greater than 16 elements is also possible.

  assert((dst.type() == CV_8UC3) && "Invalid!");
  assert((src.type() == CV_8UC3) && "Invalid!");
  assert((mask.type() == CV_8UC1) && "Invalid!");
  assert((dst.rows == src.rows) && (dst.cols == src.cols) && "Invalid!");
  assert((dst.rows == mask.rows) && (dst.cols == mask.cols) && "Invalid!");
  assert((b_lut.count() <= 16U) && (g_lut.count() <= 16U) &&
         (r_lut.count() <= 16U) && (a_lut.count() <= 16U) && "Unsupported!");

  const uint8x16_t b_lut_u8 = vld1q_u8(b_lut.data());
  const uint8x16_t g_lut_u8 = vld1q_u8(g_lut.data());
  const uint8x16_t r_lut_u8 = vld1q_u8(r_lut.data());
  const uint8x16_t a_lut_u8 = vld1q_u8(a_lut.data());

  uint16x8_t b_u16{}, g_u16{}, r_u16{};

  // Process one row at a time.
  for (int32_t row = 0, col0 = 0;  //
       row < src.rows;             //
       ++row, col0 = 0) {
    uint8_t* const __restrict dst_row_ptr = dst.ptr(row);
    const uint8_t* const __restrict src_row_ptr = src.ptr(row);
    const uint8_t* const __restrict mask_row_ptr = mask.ptr(row);

    // Proess 8 pixels at a time.
    for (int32_t col2 = 0;        //
         (col0 + 8) <= src.cols;  //
         col0 += 8, col2 += 24) {
      const uint8x8_t mask_u8 = vld1_u8(mask_row_ptr + col0);
      const uint8x8x3_t src_u24 = vld3_u8(src_row_ptr + col2);

      const uint8x8_t mask_a_u8 = vqtbl1_u8(a_lut_u8, mask_u8);
      const uint8x8_t mask_b_u8 = vqtbl1_u8(b_lut_u8, mask_u8);
      const uint8x8_t mask_g_u8 = vqtbl1_u8(g_lut_u8, mask_u8);
      const uint8x8_t mask_r_u8 = vqtbl1_u8(r_lut_u8, mask_u8);
      const uint8x8_t src_a_u8 = vmvn_u8(mask_a_u8);

      b_u16 = vmull_u8(src_u24.val[0U], src_a_u8);
      g_u16 = vmull_u8(src_u24.val[1U], src_a_u8);
      r_u16 = vmull_u8(src_u24.val[2U], src_a_u8);

      b_u16 = vmlal_u8(b_u16, mask_b_u8, mask_a_u8);
      g_u16 = vmlal_u8(g_u16, mask_g_u8, mask_a_u8);
      r_u16 = vmlal_u8(r_u16, mask_r_u8, mask_a_u8);

      vst3_u8(dst_row_ptr + col2,  //
              uint8x8x3_t{
                  vshrn_n_u16(b_u16, 8U),
                  vshrn_n_u16(g_u16, 8U),
                  vshrn_n_u16(r_u16, 8U),
              });
    }

    // TODO: Process left-overs here.
  }
}

}  // namespace utils
}  // namespace driver
}  // namespace phantom_ai

#endif /* PHANTOM_DRIVER_SRC_UTILS_H */
