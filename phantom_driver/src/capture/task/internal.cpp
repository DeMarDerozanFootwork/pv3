/*******************************************************************************
 * @file    internal.cpp
 * @date    01/2025
 *
 * @attention Copyright (c) 2022-2025
 * @attention Phantom AI, Inc.
 * @attention All rights reserved.
 *******************************************************************************/

#include "driver.h"

namespace phantom_ai {
namespace driver {

//
// IMPORTANT NOTE!
// This function MUST BE kept as light as possible!  Do NOT perform any extra
// processing here except for the absolute bare minimum!  Ideally, that simply
// takes the form of calling PV3 in a loop and nothing else!  Keep in mind that
// PV3 is the front end of the vision pipeline, and produces data that the rest
// of the perception stack has a blocking dependency on.  Furthermore, PV3 runs
// inference which currently is the performance bottleneck in our pipeline.
// Because of these two reasons, the rate at which PV3 runs places a ceiling on
// the performance of the entirety of the vision system.  Any slowdowns here,
// introduced as a result of any extra processing performed, will impact the
// performance of the entirety of the vision pipeline!
//

Status
process(Context& driver) {
  return vision::process(  //
      driver.pv3.context,  //
      vision::Input{},     //
      driver.pv3.output);
}

}  // namespace driver
}  // namespace phantom_ai
