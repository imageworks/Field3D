//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2018 Pixar Animation Studios
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

/*! \file CSparseField.h
  \brief Contains the CSparseField (Compressed SparseField) class
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_CSparseField_H_
#define _INCLUDED_Field3D_CSparseField_H_

//----------------------------------------------------------------------------//

#include <vector>

#include "SparseField.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Forward declarations
//----------------------------------------------------------------------------//

template <class Field_T>
class LinearGenericFieldInterp;
template <class Field_T>
class CubicGenericFieldInterp;

class CSparseFieldImpl;

//----------------------------------------------------------------------------//
// CSparseField
//----------------------------------------------------------------------------//

/*! \class CSparseField
  \ingroup field
  \brief A compressed SparseField, storing values using the ZFP compression lib.
  \note Although it is possible to compile half/double/V3h/V3d versions
        of CSparseField, these are not the intended use, and internally
        the ZFP representation is identical.
  \note All compressed fields are immutable.
*/

//----------------------------------------------------------------------------//

template <typename Data_T>
class CSparseField
  : public Field<Data_T>
{
public:

  // Typedefs ------------------------------------------------------------------

  typedef CSparseField                     class_type;
  typedef boost::intrusive_ptr<class_type> Ptr;
  typedef std::vector<Ptr>                 Vec;

  typedef LinearGenericFieldInterp<class_type> LinearInterp;
  typedef CubicGenericFieldInterp<class_type>  CubicInterp;
  typedef GenericStochasticInterp<class_type>  StochasticInterp;

  // RTTI replacement ----------------------------------------------------------

  DEFINE_FIELD_RTTI_CONCRETE_CLASS;

  static const char *staticClassName()
  {
    return "CSparseField";
  }

  static const char *staticClassType()
  {
    return CSparseField<Data_T>::ms_classType.name();
  }

  // Ctors, dtor ---------------------------------------------------------------

  CSparseField();

  // Threading-related ---------------------------------------------------------

  //! Number of 'grains' to use with threaded access
  virtual size_t numGrains() const;
  //! Bounding box of the given 'grain'
  //! \return Whether the grain is contiguous in memory
  virtual bool   getGrainBounds(const size_t idx, Box3i &vsBounds) const;

  // From Field base class -----------------------------------------------------

  //! \name From Field
  //! \{
  virtual long long int memSize() const;
  virtual size_t        voxelCount() const;
  virtual Data_T        value(int i, int j, int k) const;
  //! \}

  // From FieldBase ------------------------------------------------------------

  //! \name From FieldBase
  //! \{
  
  FIELD3D_CLASSNAME_CLASSTYPE_IMPLEMENTATION;

  virtual FieldBase::Ptr clone() const
  { return Ptr(new CSparseField(*this)); }

  //! \}

  // Main methods --------------------------------------------------------------

  //! Converts the provided field to compressed form
  bool compress(const SparseField<Data_T> &field, const int bitRate);
  //! Configures the field at load time
  void configure(const Box3i &extents,
                 const Box3i &dataWindow,
                 const int    blockOrder,
                 const V3i   &blockRes,
                 const int    bitRate,
                 const std::vector<int> &blockMap);
  //! Converts the field to uncompressed form
  typename SparseField<Data_T>::Ptr decompress() const;
  //! Non-virtual access
  Data_T fastValue(int i, int j, int k) const;
  //! Whether a block is allocated
  bool blockIsAllocated(const int bi, const int bj, const int bk) const;
  //! Block empty value
  Data_T getBlockEmptyValue(const int bi, const int bj, const int bk) const;
  //! Returns the block size
  int blockSize() const;
  //! Returns the resolution of the block array
  V3i blockRes() const;
  //! Returns the bitrate
  int bitRate() const;

protected:

  friend class CSparseFieldIO;
  
  // Typedefs ------------------------------------------------------------------

  typedef Field<Data_T> base;
  
  // Main methods --------------------------------------------------------------

  CSparseFieldImpl* impl();

  // Data members --------------------------------------------------------------

  boost::shared_ptr<CSparseFieldImpl>  m_implPtr;
  CSparseFieldImpl                    *m_impl;

private:

  // Static data members -------------------------------------------------------

  static TemplatedFieldType<class_type> ms_classType;

};

//----------------------------------------------------------------------------//
// Static member instantiations
//----------------------------------------------------------------------------//

FIELD3D_CLASSTYPE_TEMPL_INSTANTIATION(CSparseField);

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
