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

/*! \file ProceduralField.h
  \brief Contains the ProceduralField class.
  \ingroup field
  
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_ProceduralField_H_
#define _INCLUDED_Field3D_ProceduralField_H_

//----------------------------------------------------------------------------//

#include "Field.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Forward declarations 
//----------------------------------------------------------------------------//

template <class T>
class ProceduralFieldLookup; 

//----------------------------------------------------------------------------//
// Utility macros
//----------------------------------------------------------------------------//

#define REGISTER_FIELD_TYPES(FIELDCLASS) \
  factory.registerField(FIELDCLASS<half>::create); \
  factory.registerField(FIELDCLASS<float>::create); \
  factory.registerField(FIELDCLASS<double>::create); \
  factory.registerField(FIELDCLASS<V3h>::create); \
  factory.registerField(FIELDCLASS<V3f>::create); \
  factory.registerField(FIELDCLASS<V3d>::create)

#define INSTANTIATE_FIELD_TYPES(FIELDCLASS) \
  template class FIELDCLASS<half>; \
  template class FIELDCLASS<float>; \
  template class FIELDCLASS<double>; \
  template class FIELDCLASS<V3h>; \
  template class FIELDCLASS<V3f>; \
  template class FIELDCLASS<V3d>

//----------------------------------------------------------------------------//
// ProceduralField
//----------------------------------------------------------------------------//

/*! \class ProceduralField
  \ingroup field

  This class generalizes the Field concept to fields that don't necessarily
  contain voxel data. This is technically wrong (it's not true that a perlin 
  noise volume IS-A Field) but it lends great flexibility to the Field concept.
  
  So what happens when a ProceduralField is accessed using the Field interface?
  ProceduralField itself implements value() such that it automatically 
  point-samples the field instead of accessing a particular voxel. This makes
  the ProceduralField seem as a regular Field to anyone using that interface.

  The interesting part comes when we look at interpolation. Regular Field
  objects use Interpolator objects to produce values in-between voxels. For
  ProceduralField, we instead want to point-sample in space, but other than that
  the resultant value can be handled the same.

  It is also fine to interpolate the ProceduralField outside its bounds - just
  as it is with regular Fields. The difference is that ProceduralFields 
  still return valid values outside its bounds. 

  By using ProceduralField, we can support purely procedural and hybrid
  procedural/discrete volumes, but to the functions that use them, they all
  function the same.

*/

//----------------------------------------------------------------------------//

template <class Data_T>
class ProceduralField : public Field<Data_T>
{

public:

  // Typedefs ------------------------------------------------------------------

  typedef boost::intrusive_ptr<ProceduralField> Ptr;

  typedef ProceduralFieldLookup<Data_T> LinearInterp;
  typedef ProceduralFieldLookup<Data_T> CubicInterp;

  // Constructors --------------------------------------------------------------

  //! Destructor
  virtual ~ProceduralField()
  { /* Empty */ }

  // To be implemented by subclasses -------------------------------------------

  virtual Data_T lsSample(const V3d &lsP) const = 0;

  // From Field ----------------------------------------------------------------

  //! Transforms the point from voxel space to subclass's space and calls
  //! the appropriate sample function
  virtual Data_T value(int i, int j, int k) const = 0;

  // Main methods --------------------------------------------------------------

  //! Calls either sampleIntMetadata() if the ProceduralField is
  //! scalar (half, float, or double), or sampleVecIntMetadata() if
  //! the field is vector (V3h, V3f, V3d)
  Data_T typedIntMetadata(const std::string &name, 
                                const Data_T& defaultVal) const;
  //! Calls either sampleFloatMetadata() if the ProceduralField is
  //! scalar (half, float, or double), or sampleVecFloatMetadata() if
  //! the field is vector (V3h, V3f, V3d)
  Data_T typedFloatMetadata(const std::string &name,
                                  const Data_T& defaultVal) const;

  // RTTI replacement ----------------------------------------------------------

  typedef ProceduralField<Data_T> class_type;
  DEFINE_FIELD_RTTI_ABSTRACT_CLASS

  virtual std::string className() const
    { return std::string("ProceduralField"); }

  // Typedefs ------------------------------------------------------------------

  typedef Field<Data_T> base;

};

//----------------------------------------------------------------------------//
// Typedefs
//----------------------------------------------------------------------------//

typedef ProceduralField<half>   ProceduralFieldh;
typedef ProceduralField<float>  ProceduralFieldf;
typedef ProceduralField<double> ProceduralFieldd;
typedef ProceduralField<V3h>    ProceduralField3h;
typedef ProceduralField<V3f>    ProceduralField3f;
typedef ProceduralField<V3d>    ProceduralField3d;

//----------------------------------------------------------------------------//
// Template specializations
//----------------------------------------------------------------------------//

template <>
inline half
ProceduralField<half>::typedIntMetadata(const std::string &name,
                                        const half& defaultVal) const
{
  return metadata().intMetadata(name, static_cast<int>(defaultVal));
}

//----------------------------------------------------------------------------//

template <>
inline float
ProceduralField<float>::typedIntMetadata(const std::string &name,
                                         const float& defaultVal) const
{
  return metadata().intMetadata(name, static_cast<int>(defaultVal));
}

//----------------------------------------------------------------------------//

template <>
inline double
ProceduralField<double>::typedIntMetadata(const std::string &name,
                                          const double& defaultVal) const
{
  return metadata().intMetadata(name, static_cast<int>(defaultVal));
}

//----------------------------------------------------------------------------//

template <>
inline V3h
ProceduralField<V3h>::typedIntMetadata(const std::string &name,
                                       const V3h& defaultVal) const
{
  return V3h(metadata().vecIntMetadata(name, defaultVal));
}

//----------------------------------------------------------------------------//

template <>
inline V3f
ProceduralField<V3f>::typedIntMetadata(const std::string &name,
                                       const V3f& defaultVal) const
{
  return V3f(metadata().vecIntMetadata(name, defaultVal));
}

//----------------------------------------------------------------------------//

template <>
inline V3d
ProceduralField<V3d>::typedIntMetadata(const std::string &name,
                                       const V3d& defaultVal) const
{
  return V3d(metadata().vecIntMetadata(name, defaultVal));
}

//----------------------------------------------------------------------------//

template <>
inline half
ProceduralField<half>::typedFloatMetadata(const std::string &name, 
                                          const half& defaultVal) const
{
  return metadata().floatMetadata(name, static_cast<float>(defaultVal));
}

//----------------------------------------------------------------------------//

template <>
inline float
ProceduralField<float>::typedFloatMetadata(const std::string &name, 
                                           const float& defaultVal) const
{
  return metadata().floatMetadata(name, defaultVal);
}

//----------------------------------------------------------------------------//

template <>
inline double
ProceduralField<double>::typedFloatMetadata(const std::string &name, 
                                            const double& defaultVal) const
{
  return metadata().floatMetadata(name, static_cast<float>(defaultVal));
}

//----------------------------------------------------------------------------//

template <>
inline V3h
ProceduralField<V3h>::typedFloatMetadata(const std::string &name, 
                                         const V3h& defaultVal) const
{
  return V3h(metadata().vecFloatMetadata(name, defaultVal));
}

//----------------------------------------------------------------------------//

template <>
inline V3f
ProceduralField<V3f>::typedFloatMetadata(const std::string &name, 
                                         const V3f& defaultVal) const
{
  return V3f(metadata().vecFloatMetadata(name, defaultVal));
}

//----------------------------------------------------------------------------//

template <>
inline V3d
ProceduralField<V3d>::typedFloatMetadata(const std::string &name, 
                                         const V3d& defaultVal) const
{
  return V3d(metadata().vecFloatMetadata(name, defaultVal));
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard

