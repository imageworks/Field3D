//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2015 Sony Pictures Imageworks Inc,
 *                    Pixar Animation Studios
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

/*! \file MinMaxUtil.h
  \brief Contains MIP-related utility functions
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_MinMaxUtil_H_
#define _INCLUDED_Field3D_MinMaxUtil_H_

//----------------------------------------------------------------------------//

#include <vector>

#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>

#include "MIPField.h"
#include "MIPUtil.h"
#include "TemporalFieldUtil.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Functions
//----------------------------------------------------------------------------//

//! Constructs a min/max MIP representation of the given field.
template <typename MIPField_T>
std::pair<typename MIPField_T::Ptr, typename MIPField_T::Ptr>
makeMinMax(const typename MIPField_T::NestedType &base, 
           const float resMult, const size_t numThreads);

//! Constructs a min/max MIP representation of the given TemporalField.
template <typename Data_T>
std::pair<typename MIPField<SparseField<Data_T> >::Ptr,
          typename MIPField<SparseField<Data_T> >::Ptr>
makeMinMax(const typename TemporalField<Data_T>::Ptr &base, 
           const float resMult, const size_t numThreads);

//----------------------------------------------------------------------------//
// Constants
//----------------------------------------------------------------------------//

//! The standard 'min' suffix - "_min"
extern const char* k_minSuffix;
//! The standard 'max' suffix - "_max"
extern const char* k_maxSuffix;

//----------------------------------------------------------------------------//
// Function implementations
//----------------------------------------------------------------------------//

template <typename MIPField_T>
std::pair<typename MIPField_T::Ptr, typename MIPField_T::Ptr>
makeMinMax(const typename MIPField_T::NestedType &base, 
           const float resMult, const size_t numThreads)
{
  typedef typename MIPField_T::Ptr             MipPtr;
  typedef typename MIPField_T::NestedType      Field;
  typedef typename MIPField_T::NestedType::Ptr FieldPtr;

  // Storage for results
  std::pair<MipPtr, MipPtr> result;

  // First, downsample the field into a min and max representation ---

  V3i srcRes = base.dataWindow().size() + Field3D::V3i(1);
  V3i res    = V3f(srcRes) * std::min(1.0f, resMult);

  // Corner case handling
  res.x = std::max(res.x, 2);
  res.y = std::max(res.y, 2);
  res.z = std::max(res.z, 2);

  // Storage for min/max fields
  FieldPtr minSrc(new Field);
  FieldPtr maxSrc(new Field);

  // Resample 
  resample(base, *minSrc, res, MinFilter());
  resample(base, *maxSrc, res, MaxFilter());

  // Second, generate MIP representations ---

  result.first  = makeMIP<MIPField_T, MinFilter>(*minSrc, 2, numThreads);
  result.second = makeMIP<MIPField_T, MaxFilter>(*maxSrc, 2, numThreads);

  return result;
}

//----------------------------------------------------------------------------//

template <typename Data_T>
std::pair<typename MIPField<SparseField<Data_T> >::Ptr,
          typename MIPField<SparseField<Data_T> >::Ptr>
makeMinMax(const typename TemporalField<Data_T>::Ptr &base, 
           const float resMult, const size_t numThreads)
{
  typedef SparseField<Data_T>     Field;
  typedef typename Field::Ptr     FieldPtr;
  typedef MIPField<Field>         MIPField;
  typedef typename MIPField::Ptr  MIPPtr;

  // Storage for results
  std::pair<MIPPtr, MIPPtr> result;

  // First, downsample the field into a min and max representation ---

  V3i srcRes = base->dataWindow().size() + Field3D::V3i(1);
  V3i res    = V3f(srcRes) * std::min(1.0f, resMult);

  // Corner case handling
  res.x = std::max(res.x, 2);
  res.y = std::max(res.y, 2);
  res.z = std::max(res.z, 2);

  // Get min/max values from temporal samples in base
  FieldPtr baseMin = minTemporalField<Data_T>(base, -1.0f, 1.0f);
  FieldPtr baseMax = maxTemporalField<Data_T>(base, -1.0f, 1.f);

  // Storage for min/max fields
  FieldPtr minSrc(new Field);
  FieldPtr maxSrc(new Field);

  // Resample 
  resample(*baseMin, *minSrc, res, MinFilter());
  resample(*baseMax, *maxSrc, res, MaxFilter());

  // Second, generate MIP representations ---

  result.first  = makeMIP<MIPField, MinFilter>(*minSrc, 2, numThreads);
  result.second = makeMIP<MIPField, MaxFilter>(*maxSrc, 2, numThreads);

  return result;

}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
