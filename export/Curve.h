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

/*! \file Curve.h
  \brief Contains the Curve class which is used to interpolate attributes
  in time.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_Curve_H_
#define _INCLUDED_Field3D_Curve_H_

//----------------------------------------------------------------------------//

#include <algorithm>
#include <utility>
#include <vector>

#include <boost/lexical_cast.hpp>

#include <OpenEXR/ImathFun.h>
#include <OpenEXR/ImathMatrix.h>

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Curve
//----------------------------------------------------------------------------//

/*! \class Curve
  \brief Implements a simple function curve where samples of type T can 
  be added along a 1D axis. Once samples exist they can be interpolated
  using the linear() call.
 */

//----------------------------------------------------------------------------//

template <typename T>
class Curve
{
public:

  // Typedefs ------------------------------------------------------------------

  typedef std::pair<float, T> Sample;
  typedef std::vector<Sample> SampleVec;

  // Main methods --------------------------------------------------------------

  //! Adds a sample point to the curve.
  //! \param t Sample position
  //! \param value Sample value
  void addSample(const float t, const T &value);

  //! Linearly interpolates a value from the curve.
  //! \param t Position along curve
  T linear(const float t) const;

  //! Returns the number of samples in the curve
  size_t numSamples() const
  { return m_samples.size(); }

  //! Returns a const reference to the samples in the curve.
  const SampleVec& samples() const
  { return m_samples; }

  //! Clears all samples in curve
  void clear() 
  { SampleVec().swap(m_samples); }

private:
  
  // Structs -------------------------------------------------------------------

  //! Used when finding values in the m_samples vector.
  struct CheckTGreaterThan : 
    public std::unary_function<std::pair<float, T>, bool>
  {
    CheckTGreaterThan(float match)
      : m_match(match)
    { }
    bool operator()(std::pair<float, T> test)
    {
      return test.first > m_match;
    }
  private:
    float m_match;
  };

  //! Used when finding values in the m_samples vector.
  struct CheckTEqual : 
    public std::unary_function<std::pair<float, T>, bool>
  {
    CheckTEqual(float match)
      : m_match(match)
    { }
    bool operator()(std::pair<float, T> test)
    {
      return test.first == m_match;
    }
  private:
    float m_match;
  };

  // Utility methods -----------------------------------------------------------

  //! The default return value is used when no sample points are available.
  //! This defaults to zero, but for some types (for example Quaternion), 
  //! We need more arguments to the constructor. In these cases the method
  //! is specialized for the given T type.
  T defaultReturnValue() const
  { return T(0); }

  //! The default implementation for linear interpolation. Works for all classes
  //! for which Imath::lerp is implemented (i.e float/double, V2f, V3f).
  //! For other types this method needs to be specialized.
  T lerp(const Sample &lower, const Sample &upper, const float t) const
  { return Imath::lerp(lower.second, upper.second, t); }

  // Private data members ------------------------------------------------------

  //! Stores the samples that define the curve. Sample insertion ensures 
  //! that the samples are sorted according to Sample.first.
  SampleVec m_samples;

};

//----------------------------------------------------------------------------//
// Template implementations
//----------------------------------------------------------------------------//

template <typename T>
void Curve<T>::addSample(const float t, const T &value)
{
  using namespace std;
  // Check that sample time is not already in curve
  typename SampleVec::iterator i =                                        
    find_if(m_samples.begin(), m_samples.end(), CheckTEqual(t));
  if (i != m_samples.end()) {
    // Sample position already exists, so we replace it
    i->second = value;
    return;
  }
  // Find the first sample location that is greater than the interpolation
  // position                                                             
  i = find_if(m_samples.begin(), m_samples.end(), CheckTGreaterThan(t));
  // If we get something other than end() back then we insert the new     
  // sample before that. If there wasn't a larger value we add this sample
  // to the end of the vector.
  if (i != m_samples.end()) {
    m_samples.insert(i, make_pair(t, value));
  } else {
    m_samples.push_back(make_pair(t, value));
  }
}

//----------------------------------------------------------------------------//

template <typename T>
T Curve<T>::linear(const float t) const
{
  using namespace std;
  // If there are no samples, return zero
  if (m_samples.size() == 0) {
    return defaultReturnValue();
  }
  // Find the first sample location that is greater than the interpolation
  // position
  typename SampleVec::const_iterator i =
    find_if(m_samples.begin(), m_samples.end(), CheckTGreaterThan(t));
  // If we get end() back then there was no sample larger, so we return the
  // last value. If we got the first value then there is only one value and
  // we return that.
  if (i == m_samples.end()) {
    return m_samples.back().second;
  } else if (i == m_samples.begin()) {
    return m_samples.front().second;
  }
  // Interpolate between the nearest two samples.
  const Sample &upper = *i;
  const Sample &lower = *(--i);
  const float interpT = Imath::lerpfactor(t, lower.first, upper.first);
  return lerp(lower, upper, interpT);
}

//----------------------------------------------------------------------------//
// Template specializations
//----------------------------------------------------------------------------//

template <>
inline Imath::Matrix44<float> 
Curve<Imath::Matrix44<float> >::defaultReturnValue() const
{ 
  Imath::Matrix44<float> identity;
  identity.makeIdentity();
  return identity;
}

//----------------------------------------------------------------------------//

template <>
inline Imath::Matrix44<double> 
Curve<Imath::Matrix44<double> >::defaultReturnValue() const
{ 
  Imath::Matrix44<double> identity;
  identity.makeIdentity();
  return identity;
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
