//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2015 Sony Pictures Imageworks, Inc.,
 *                    Pixar Animation Studios, Inc.
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

/*! \file StochasticInterp.h
  \ingroup field
  \brief Contains the StochasticInterp class, which works with all basic
  field types.
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_StochasticInterp_H_
#define _INCLUDED_Field3D_StochasticInterp_H_

#include "FieldInterp.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// GenericStochasticInterp
//----------------------------------------------------------------------------//

/*! \class GenericStochasticInterp
  \ingroup field
  \brief Interpolator that stochastically chooses a single voxel in a field
  such that with multiple samples, the average converges to what would be
  the fully interpolated value.
  \note An interpolator always interpolates in voxel space.
*/

template <class Field_T>
class GenericStochasticInterp : public RefBase
{
public:
  
  // Typedefs ------------------------------------------------------------------

  typedef typename Field_T::value_type value_type;
  typedef boost::intrusive_ptr<GenericStochasticInterp> Ptr;

  // RTTI replacement ----------------------------------------------------------

  typedef GenericStochasticInterp class_type;
  DEFINE_FIELD_RTTI_CONCRETE_CLASS;

  static const char *staticClassName()
  {
    return "GenericStochasticInterp";
  }
  
  static const char* staticClassType()
  {
    return ms_classType.name();
  }

  // Main methods --------------------------------------------------------------

  value_type linear(const Field_T &data, const V3d &vsP,
                    const float xiX, const float xiY, const float xiZ) const
  {
    // Voxel coords
    V3i c1, c2;
    // Interpolation weights
    FIELD3D_VEC3_T<double> f1, f2;

    getLerpInfo(vsP, data.dataWindow(), f1, f2, c1, c2);

    // Choose c1 or c2 based on random variables
    return data.fastValue(xiX < f1.x ? c1.x : c2.x, 
                          xiY < f1.y ? c1.y : c2.y,
                          xiZ < f1.z ? c1.z : c2.z);
  }
  
private:

  // Static data members -------------------------------------------------------

  static TemplatedFieldType<GenericStochasticInterp<Field_T> > ms_classType;

  // Typedefs ------------------------------------------------------------------

  //! Convenience typedef for referring to base class
  typedef RefBase base;    

};

//----------------------------------------------------------------------------//
// Static member instantiation
//----------------------------------------------------------------------------//

FIELD3D_CLASSTYPE_TEMPL_INSTANTIATION(GenericStochasticInterp);

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
