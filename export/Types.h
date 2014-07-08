//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2009 Sony Pictures Imageworks Inc
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the
 * distribution.  Neither the name of Sony Pictures Imageworks nor the
 * names of its contributors may be used to endorse or promote
 * products derived from this software without specific prior written
 * permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//----------------------------------------------------------------------------//

/*! \file Types.h
  \brief Contains typedefs for the commonly used types in Field3D.
  \ingroup field
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_Types_H_
#define _INCLUDED_Field3D_Types_H_

//----------------------------------------------------------------------------//

#include <vector>

#ifdef FIELD3D_CUSTOM_MATH_LIB
#  include FIELD3D_MATH_LIB_INCLUDE
#else
#  include "StdMathLib.h"
#endif

//----------------------------------------------------------------------------//
// Interval
//----------------------------------------------------------------------------//

//! Represents a single integration interval. 
//! The interval is assumed to be inclusive, i.e. [t0,t1].
struct Interval
{
  // Constructor ---------------------------------------------------------------

  //! Default constructor
  Interval(double start, double end, double step)
    : t0(start), t1(end), stepLength(step) 
  { }

  // Public data members -------------------------------------------------------

  //! The start of the interval (inclusive)
  double t0;
  //! The end of the interval (inclusive)
  double t1;
  //! The world space step length that is reasonable to use for the given 
  //! interval.
  double stepLength;
};

//----------------------------------------------------------------------------//

typedef std::vector<Interval> IntervalVec;

//----------------------------------------------------------------------------//
// Ogawa Types
//----------------------------------------------------------------------------//

#if !defined(_MSC_VER)
using ::uint8_t;
using ::int8_t;
using ::uint16_t;
using ::int16_t;
using ::uint32_t;
using ::int32_t;
using ::uint64_t;
using ::int64_t;
#else
typedef unsigned char           uint8_t;
typedef signed char             int8_t;
typedef unsigned short          uint16_t;
typedef signed short            int16_t;
typedef unsigned int            uint32_t;
typedef int                     int32_t;
typedef unsigned long long      uint64_t;
typedef long long               int64_t;
#endif

typedef half                    float16_t;
typedef float                   float32_t;
typedef double                  float64_t;

typedef Field3D::V3h            vec16_t;
typedef Field3D::V3f            vec32_t;
typedef Field3D::V3d            vec64_t;

//----------------------------------------------------------------------------//

#endif // Include guard

