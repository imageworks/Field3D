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

/*! \file SparseField.h
  \brief Contains the SparseField class
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_SparseField_H_
#define _INCLUDED_Field3D_SparseField_H_

//----------------------------------------------------------------------------//

#include <vector>

#include <boost/lexical_cast.hpp>

#include "Field.h"
#include "SparseFile.h"

#define BLOCK_ORDER 4 // 2^BLOCK_ORDER is the block size along each axis

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

//----------------------------------------------------------------------------//
// LinearSparseFieldInterp
//----------------------------------------------------------------------------//

/* \class LinearSparseFieldInterp
   \ingroup field
   \brief Linear interpolator optimized for fields with a fastValue function
*/

//----------------------------------------------------------------------------//

template <typename Data_T>
class LinearSparseFieldInterp : public RefBase
{
public:

  // Typedefs ------------------------------------------------------------------

  typedef Data_T value_type;
  typedef boost::intrusive_ptr<LinearSparseFieldInterp> Ptr;

  // RTTI replacement ----------------------------------------------------------

  typedef LinearSparseFieldInterp class_type;
  DEFINE_FIELD_RTTI_CONCRETE_CLASS;

  static const char *staticClassName()
  {
    return "LinearSparseFieldInterp";
  }

  static const char* classType()
  {
    return ms_classType.name();
  }

  // Main methods --------------------------------------------------------------

  value_type sample(const SparseField<Data_T> &field, const V3d &vsP) const
  {
    // Pixel centers are at .5 coordinates
    // NOTE: Don't use contToDisc for this, we're looking for sample
    // point locations, not coordinate shifts.
    FIELD3D_VEC3_T<double> p(vsP - FIELD3D_VEC3_T<double>(0.5));

    // Lower left corner
    V3i c1(static_cast<int>(floor(p.x)),
           static_cast<int>(floor(p.y)),
           static_cast<int>(floor(p.z)));
    // Upper right corner
    V3i c2(c1 + V3i(1));
    // C1 fractions
    FIELD3D_VEC3_T<double> f1(static_cast<FIELD3D_VEC3_T<double> >(c2) - p);
    // C2 fraction
    FIELD3D_VEC3_T<double> f2(static_cast<FIELD3D_VEC3_T<double> >(1.0) - f1);

    const Box3i &dataWindow = field.dataWindow();

    // Clamp the coordinates
    c1.x = std::min(dataWindow.max.x, std::max(dataWindow.min.x, c1.x));
    c1.y = std::min(dataWindow.max.y, std::max(dataWindow.min.y, c1.y));
    c1.z = std::min(dataWindow.max.z, std::max(dataWindow.min.z, c1.z));
    c2.x = std::min(dataWindow.max.x, std::max(dataWindow.min.x, c2.x));
    c2.y = std::min(dataWindow.max.y, std::max(dataWindow.min.y, c2.y));
    c2.z = std::min(dataWindow.max.z, std::max(dataWindow.min.z, c2.z));

    // Determine which block we're in
    int i = c1.x, j = c1.y, k = c1.z, vi, vj, vk, bi, bj, bk;
    field.applyDataWindowOffset(i, j, k);
    field.getVoxelInBlock(i, j, k, vi, vj, vk);
    field.getBlockCoord(i, j, k, bi, bj, bk);
    int blockSize = 1 << field.blockOrder();

    // If in the middle of a block, optimize lookup stencil
    if (vi < blockSize - 1 && vj < blockSize - 1 && vk < blockSize - 1) {
      // Only do work if the block is allocated
      const Data_T * const p = field.blockData(bi, bj, bk);
      if (p) {
        const Data_T * const c111 = p + vi + vj * blockSize + vk * blockSize * blockSize;
        const Data_T * const c121 = c111 + blockSize * (c2.y - c1.y);
        const Data_T * const c112 = c111 + blockSize * blockSize * (c2.z - c1.z);
        const Data_T * const c122 = c112 + blockSize * (c2.y - c1.y);
        int xInc = c2.x - c1.x;
        return static_cast<Data_T>
          (f1.x * (f1.y * (f1.z * *c111 +
                           f2.z * *c112) +
                   f2.y * (f1.z * *c121 +
                           f2.z * *c122)) +
           f2.x * (f1.y * (f1.z * *(c111 + xInc) +
                           f2.z * *(c112 + xInc)) +
                   f2.y * (f1.z * *(c121 + xInc) +
                           f2.z * *(c122 + xInc))));
      } else {
        return static_cast<Data_T>(field.getBlockEmptyValue(bi, bj, bk));
      }
    } else {
      return static_cast<Data_T>
        (f1.x * (f1.y * (f1.z * field.fastValue(c1.x, c1.y, c1.z) +
                         f2.z * field.fastValue(c1.x, c1.y, c2.z)) +
                 f2.y * (f1.z * field.fastValue(c1.x, c2.y, c1.z) +
                         f2.z * field.fastValue(c1.x, c2.y, c2.z))) +
         f2.x * (f1.y * (f1.z * field.fastValue(c2.x, c1.y, c1.z) +
                         f2.z * field.fastValue(c2.x, c1.y, c2.z)) +
                 f2.y * (f1.z * field.fastValue(c2.x, c2.y, c1.z) +
                         f2.z * field.fastValue(c2.x, c2.y, c2.z))));
    }

  }

private:

  // Static data members -------------------------------------------------------

  static TemplatedFieldType<LinearSparseFieldInterp<Data_T> > ms_classType;

  // Typedefs ------------------------------------------------------------------

  //! Convenience typedef for referring to base class
  typedef RefBase base;

};

//----------------------------------------------------------------------------//

FIELD3D_CLASSTYPE_TEMPL_INSTANTIATION(LinearSparseFieldInterp);

//----------------------------------------------------------------------------//
// SparseBlock
//----------------------------------------------------------------------------//

//! Namespace for sparse field specifics
//! \ingroup field_int
namespace Sparse {


//! \class SparseBlock
//! \ingroup field_int
//! Storage for one individual block of a SparseField
template <typename Data_T>
struct SparseBlock : boost::noncopyable
{
  // Constructors --------------------------------------------------------------

  //! Ctor
  SparseBlock()
    : isAllocated(false),
      emptyValue(static_cast<Data_T>(0)),
      data(NULL)
  { /* Empty */ }

  //! Dtor
  ~SparseBlock()
  {
    if (data) {
      delete[] data;
    }
  }

  // Main methods --------------------------------------------------------------

  //! Gets the value of a given voxel
  inline Data_T& value(int i, int j, int k, int blockOrder)
  //! \note Bit shift should be ok, indices are always positive.
  { return data[(k << blockOrder << blockOrder) + (j << blockOrder) + i]; }

  //! Gets the const value of a given voxel
  //! \note Bit shift should be ok, indices are always positive.
  inline const Data_T& value(int i, int j, int k, int blockOrder) const
  { return data[(k << blockOrder << blockOrder) + (j << blockOrder) + i]; }

  //! Alloc data
  void resize(int n)
  {
    if (data) {
      delete[] data;
    }
    data = new Data_T[n];
    isAllocated = true;
    std::fill_n(data, n, emptyValue);
  }

  //! Remove data
  void clear()
  {
    if (data) {
      delete[] data;
      data = NULL;
    }
    isAllocated = false;
  }

  //! Copy data from another block
  void copy(const SparseBlock &other, size_t n)
  {
    if(other.isAllocated) {
      if (!data) {
        resize(n);
      }
      Data_T *p = data, *end = data + n, *o = other.data;
      while (p != end) {
        *p++ = *o++;
      }
    }
    else
      clear();

  }

  // Data members --------------------------------------------------------------

  //! Whether the block is allocated or not
  bool isAllocated;

  //! The value to use if the block isn't allocated. We allow setting this
  //! per block so that we for example can have different inside/outside
  //! values when storing narrow-band levelsets
  Data_T emptyValue;

  //! Pointer to data. Null if block is unallocated
  Data_T *data;

private:

  //! Non-copyable
  SparseBlock(const SparseBlock&);
  //! Non-copyable
  const SparseBlock& operator=(const SparseBlock&);

};

} // namespace Sparse

//----------------------------------------------------------------------------//
// SparseField
//----------------------------------------------------------------------------//

/*! \class SparseField
  \ingroup field
  \brief This Field subclass stores voxel data in block-allocated arrays.

  Empty blocks aren't allocated. This effectively optimizes away memory use
  for "empty" voxels.

  Refer to \ref using_fields for examples of how to use this in your code.

  \todo Make this class thread safe!
*/

//----------------------------------------------------------------------------//

template <class Data_T>
class SparseField
  : public ResizableField<Data_T>
{
public:

  // Typedefs ------------------------------------------------------------------

  typedef boost::intrusive_ptr<SparseField> Ptr;
  typedef std::vector<Ptr> Vec;

  typedef LinearSparseFieldInterp<Data_T> LinearInterp;
  typedef CubicGenericFieldInterp<SparseField<Data_T> > CubicInterp;

  // RTTI replacement ----------------------------------------------------------

  typedef SparseField<Data_T> class_type;
  DEFINE_FIELD_RTTI_CONCRETE_CLASS;

  static const char *staticClassName()
  {
    return "SparseField";
  }

  static const char *classType()
  {
    return SparseField<Data_T>::ms_classType.name();
  }

  // Constructors --------------------------------------------------------------

  //! \name Constructors & destructor
  //! \{

  //! Constructs an empty buffer
  SparseField();

  //! Copy constructor
  SparseField(const SparseField &o);

  //! Destructor
  ~SparseField();

  //! Assignment operator.  For cache-managed fields, it creates a new
  //! file reference, and for non-managed fields, it copies the data
  SparseField& operator=(const SparseField &o);

  // \}

  // Main methods --------------------------------------------------------------

  //! Clears all the voxels in the storage
  virtual void clear(const Data_T &value);

  //! Sets the block order (i.e. the power-of-2 to use as block size.
  //! \note This will clear out any existing data.
  void setBlockOrder(int order);

  //! Returns the block order
  int blockOrder() const;

  //! Returns the block size
  int blockSize() const;

  //! Checks if a voxel is in an allocated block
  bool voxelIsInAllocatedBlock(int i, int j, int k) const;

  //! Checks if a block is allocated
  bool blockIsAllocated(int bi, int bj, int bk) const;

  //! Returns the constant value of an block, whether it's allocated
  //! already or not..
  const Data_T getBlockEmptyValue(int bi, int bj, int bk) const;

  //! Sets the constant value of an block.
  //! If the block is already allocated, it gets deallocated
  void setBlockEmptyValue(int bi, int bj, int bk, const Data_T &val);

  //! Returns whether a block index is valid
  bool blockIndexIsValid(int bi, int bj, int bk) const;

  //! Returns the resolution of the block array
  V3i blockRes() const;

  //! Releases any blocks that are deemed empty.
  //! This can be used to clean up after algorithms that write "zero" values
  //! to the buffer, as well as after any narrow band levelset algorithms.
  //! \param func A function object with the method "bool check(SparseBlock&)"
  //! \returns Number of released blocks
  template <typename Functor_T>
  int releaseBlocks(Functor_T func);

  //! Calculates the block number based on a block i,j,k index
  int blockId(int blockI, int blockJ, int blockK) const;

  //! Calculates the block coordinates that a given set of voxel coords are in
  //! \note The i,j,k coordinates are strictly positive, and refer to the
  //! coordinates of a voxel -after- the data window offset has been applied.
  void getBlockCoord(int i, int j, int k, int &bi, int &bj, int &bk) const;

  //! Calculates the coordinates in a block for the given voxel index
  //! \note The i,j,k coordinates are strictly positive, and refer to the
  //! coordinates of a voxel -after- the data window offset has been applied.
  void getVoxelInBlock(int i, int j, int k, int &vi, int &vj, int &vk) const;

  //! Applies data window offset
  void applyDataWindowOffset(int &i, int &j, int &k) const
  {
    i -= base::m_dataWindow.min.x;
    j -= base::m_dataWindow.min.y;
    k -= base::m_dataWindow.min.z;
  }

  // From Field base class -----------------------------------------------------

  //! \name From Field
  //! \{
  virtual Data_T value(int i, int j, int k) const;
  virtual long long int memSize() const;
  //! \}

  // From WritableField base class ---------------------------------------------

  //! \name From WritableField
  //! \{
  virtual Data_T& lvalue(int i, int j, int k);
  //! \}

  // Concrete voxel access -----------------------------------------------------

  //! Read access to voxel. Notice that this is non-virtual.
  Data_T fastValue(int i, int j, int k) const;
  //! Write access to voxel. Notice that this is non-virtual.
  Data_T& fastLValue(int i, int j, int k);

  //! Returns a pointer to the data in a block, or null if the given block is
  //! unallocated
  Data_T* blockData(int bi, int bj, int bk) const;

  // From FieldBase ------------------------------------------------------------

  //! \name From FieldBase
  //! \{
  virtual std::string className() const
  { return staticClassName(); }

  virtual FieldBase::Ptr clone() const
  { return Ptr(new SparseField(*this)); }

  //! \}

  // Iterators -----------------------------------------------------------------

  //! \name Iterators
  //! \{

  //! Const iterator for traversing the values in a SparseField object.
  class const_iterator;

  //! Const iterator to first element. "cbegin" matches the tr1 c++ standard.
  const_iterator cbegin() const;
  //! Const iterator to first element of specific subset
  const_iterator cbegin(const Box3i &subset) const;
  //! Const iterator pointing one element past the last valid one.
  const_iterator cend() const;
  //! Const iterator pointing one element past the last valid one (for a
  //! subset)
  const_iterator cend(const Box3i &subset) const;

  //! Non-const iterator for traversing the values in a SparseField object.
  //! When dereferencing a non-const iterator, you will potentially allocate
  //! the block that it refers to.
  class iterator;

  //! Iterator to first element.
  iterator begin();
  //! Iterator to first element of specific subset
  iterator begin(const Box3i &subset);
  //! Iterator pointing one element past the last valid one.
  iterator end();
  //! Iterator pointing one element past the last valid one (for a
  //! subset)
  iterator end(const Box3i &subset);

  //! Iterator for traversing the field block-by-block.
  //! \note There is no direct access to voxels from this iterator, it mainly
  //! exists to return the integer coordinate and bounding box of each block
  class block_iterator;

  block_iterator blockBegin() const;
  //! Const iterator pointing to element one past the last valid block
  block_iterator blockEnd() const;

  //! \}

  // Internal utility functions ------------------------------------------------

  //! Internal function to create a Reference for the current field,
  //! for use in dynamic reading.
  void addReference(const std::string &filename, const std::string &layerPath,
                    int valuesPerBlock, int occupiedBlocks);
  //! Internal function to setup the Reference's block pointers, for
  //! use with dynamic reading.
  void setupReferenceBlocks();

 protected:

  friend class SparseFieldIO;

  // Typedefs ------------------------------------------------------------------

  typedef ResizableField<Data_T> base;
  typedef Sparse::SparseBlock<Data_T> Block;

  // From ResizableField class -------------------------------------------------

  virtual void sizeChanged()
  {
    // Call base class
    base::sizeChanged();
    setupBlocks();
  }

  // Convenience methods -------------------------------------------------------

  //! \name Convenience methods
  //! \{

  //! Initializes the block structure. Will clear any existing data
  void setupBlocks();

  //! Deallocated the data of the given block and sets its empty value
  void deallocBlock(Block &block, const Data_T &emptyValue);

  //! \}

  // Data members --------------------------------------------------------------

  //! Block order (size = 2^blockOrder)
  int m_blockOrder;
  //! Block array resolution
  V3i m_blockRes;
  //! Block array res.x * res.y
  int m_blockXYSize;
  //! Array of blocks. Not using std::vector since SparseBlock is noncopyable
  Block *m_blocks;
  //! Number of blocks in field.
  size_t m_numBlocks;

  //! Pointer to SparseFileManager. Used when doing dynamic reading.
  //! NULL if not in use.
  SparseFileManager *m_fileManager;
  //! File id. Used with m_fileManager if active. Otherwise -1
  int m_fileId;

  //! Dummy value used when needing to return but indicating a failed call.
  Data_T m_dummy;

private:

  // Static data members -------------------------------------------------------

  static TemplatedFieldType<SparseField<Data_T> > ms_classType;

  // Utility methods -----------------------------------------------------------

  //! Copies internal data, including blocks, from another SparseField,
  //! used by copy constructor and operator=
  void copySparseField(const SparseField &o);

  //! Internal function to copy empty values and allocated flags,
  //! without copying data, used when copying a dynamically read field
  void copyBlockStates(const SparseField<Data_T> &o);

};

//----------------------------------------------------------------------------//
// Static member instantiations
//----------------------------------------------------------------------------//

FIELD3D_CLASSTYPE_TEMPL_INSTANTIATION(SparseField);

//----------------------------------------------------------------------------//
// Typedefs
//----------------------------------------------------------------------------//

typedef SparseField<half>   SparseFieldh;
typedef SparseField<float>  SparseFieldf;
typedef SparseField<double> SparseFieldd;
typedef SparseField<V3h>    SparseField3h;
typedef SparseField<V3f>    SparseField3f;
typedef SparseField<V3d>    SparseField3d;

//----------------------------------------------------------------------------//
// Helper functors
//----------------------------------------------------------------------------//

namespace Sparse {

//! Checks if all the values in the SparseBlock are equal.
//! Used by SparseField::releaseBlocks().
template <typename Data_T>
struct CheckAllEqual
{
  //! Checks whether a given block can be released. It's safe to assume
  //! that the block is allocated if this functor is called.
  //! \param block Reference to the block to check
  //! \param retEmptyValue If the block is to be removed, store the
  //! "empty value" that replaces it in this variable
  //! \param validSize Number of voxels per dim within field data window
  //! \param blockSize Number of voxels actually allocated per dim
  //! \returns Whether or not the supplied block can be released.
  bool check(const SparseBlock<Data_T> &block, Data_T &retEmptyValue,
             const V3i &validSize, const V3i &blockSize)
  {
    // Store first value
    Data_T first = block.data[0];
    // Iterate over rest
    bool match = true;
    size_t len = blockSize.x * blockSize.y * blockSize.z;
    if (validSize == blockSize) {
      // interior block so look at all voxels
      for (size_t i = 0; i < len; i++) {
        if (block.data[i] != first) {
          match = false;
          break;
        }
      }
    } else {
      // only look at valid voxels
      int x=0, y=0, z=0;
      for (size_t i = 0; i < len; i++, x++) {
        if (x >= blockSize.x) {
          x = 0;
          ++y;
          if (y >= blockSize.y) {
            y = 0;
            ++z;
          }
        }
        if (x >= validSize.x || y >= validSize.y || z >= validSize.z) {
          continue;
        }
        if (block.data[i] != first) {
          match = false;
          break;
        }
      }
    } // end of interior block test

    if (match) {
      retEmptyValue = first;
      return true;
    } else {
      return false;
    }
  }
};

//----------------------------------------------------------------------------//

template <typename Data_T>
inline bool isAnyLess(const Data_T &left, const Data_T &right)
{
  return (std::abs(left) < right);
}

//----------------------------------------------------------------------------//

template <>
inline bool isAnyLess(const V3h &left, const V3h &right)
{
  return (std::abs(left.x) < right.x ||
          std::abs(left.y) < right.y ||
          std::abs(left.z) < right.z );
}

//----------------------------------------------------------------------------//

template <>
inline bool isAnyLess(const V3f &left, const V3f &right)
{
  return (std::abs(left.x) < right.x ||
          std::abs(left.y) < right.y ||
          std::abs(left.z) < right.z );
}

//----------------------------------------------------------------------------//

template <>
inline bool isAnyLess(const V3d &left, const V3d &right)
{
  return (std::abs(left.x) < right.x ||
          std::abs(left.y) < right.y ||
          std::abs(left.z) < right.z );
}

//----------------------------------------------------------------------------//

//! Checks if all the absolute values in the SparseBlock are greater than
//! some number. Useful for making narrow band levelsets
//! Used by SparseField::releaseBlocks().
template <typename Data_T>
struct CheckMaxAbs
{
  //! Constructor. Takes max value
  CheckMaxAbs(Data_T maxValue)
    : m_maxValue(maxValue)
  { }
  //! Checks whether a given block can be released. It's safe to assume
  //! that the block is allocated if this functor is called.
  //! \param block Reference to the block to check
  //! \param retEmptyValue If the block is to be removed, store the
  //! "empty value" that replaces it in this variable
  //! \param validSize Number of voxels per dim within field data window
  //! \param blockSize Number of voxels actually allocated per dim
  //! \returns Whether or not the supplied block can be released.
  bool check(const SparseBlock<Data_T> &block, Data_T &retEmptyValue,
             const V3i &validSize, const V3i &blockSize)
  {
    // Store first value
    Data_T first = block.data[0];
    // Iterate over rest
    bool allGreater = true;
    size_t len = blockSize.x * blockSize.y * blockSize.z;

    if (validSize == blockSize) {
      // interior block so look at all voxels
      for (size_t i = 0; i < len; i++) {
        if (isAnyLess<Data_T>(block.data[i], m_maxValue)) {
          allGreater = false;
          break;
        }
      }
    } else {
      // only look at valid voxels
      int x=0, y=0, z=0;
      for (size_t i = 0; i < len; i++, x++) {
        if (x >= blockSize.x) {
          x = 0;
          ++y;
          if (y >= blockSize.y) {
            y = 0;
            ++z;
          }
        }
        if (x >= validSize.x || y >= validSize.y || z >= validSize.z) {
          continue;
        }
        if (isAnyLess<Data_T>(block.data[i], m_maxValue)) {
          allGreater = false;
          break;
        }
      }
    } // end of interior block test

    if (allGreater) {
      retEmptyValue = first;
      return true;
    } else {
      return false;
    }
  }
private:
  Data_T m_maxValue;
};

//----------------------------------------------------------------------------//

} // namespace Sparse

//----------------------------------------------------------------------------//
// SparseField::const_iterator
//----------------------------------------------------------------------------//

//! \todo Code duplication between this and iterator!!!!!!!!!!!!!!!!!!!!!!
template <class Data_T>
class SparseField<Data_T>::const_iterator
{
 public:
#if defined(WIN32) || __MAC_OS_X_VERSION_MIN_REQUIRED >= 1090
  typedef std::forward_iterator_tag iterator_category;
  typedef Data_T value_type;
  typedef ptrdiff_t difference_type;
  typedef ptrdiff_t distance_type;
  typedef Data_T *pointer;
  typedef Data_T& reference;
#endif

  typedef SparseField<Data_T> class_type;
  const_iterator(const class_type &field,
                 const Box3i &window,
                 const V3i &currentPos, int blockOrder)
    : x(currentPos.x), y(currentPos.y), z(currentPos.z),
      m_p(NULL), m_blockIsActivated(false),
      m_blockStepsTicker(0), m_blockOrder(blockOrder),
      m_blockId(-1), m_window(window), m_field(&field)
  {
    m_manager = m_field->m_fileManager;
    setupNextBlock(x, y, z);
  }
  ~const_iterator() {
    if (m_manager && m_blockId >= 0 &&
        m_blockId < static_cast<int>(m_field->m_numBlocks)) {
      if (m_field->m_blocks[m_blockId].isAllocated)
        m_manager->decBlockRef<Data_T>(m_field->m_fileId, m_blockId);
    }
  }
  const const_iterator& operator ++ ()
  {
    bool resetPtr = false;
    // Check against end of data window
    if (x == m_window.max.x) {
      if (y == m_window.max.y) {
        x = m_window.min.x;
        y = m_window.min.y;
        ++z;
        resetPtr = true;
      } else {
        x = m_window.min.x;
        ++y;
        resetPtr = true;
      }
    } else {
      ++x;
    }
    // These can both safely be incremented here
    ++m_blockStepsTicker;
    // ... but only step forward if we're in a non-empty block
    if (!m_isEmptyBlock && (!m_manager || m_blockIsActivated))
      ++m_p;
    // Check if we've reached the end of this block
    if (m_blockStepsTicker == (1 << m_blockOrder))
      resetPtr = true;
    if (resetPtr) {
      // If we have, we need to reset the current block, etc.
      m_blockStepsTicker = 0;
      setupNextBlock(x, y, z);
    }
    return *this;
  }
  template <class Iter_T>
  inline bool operator == (const Iter_T &rhs) const
  {
    return x == rhs.x && y == rhs.y && z == rhs.z;
  }
  template <class Iter_T>
  inline bool operator != (const Iter_T &rhs) const
  {
    return x != rhs.x || y != rhs.y || z != rhs.z;
  }
  inline const Data_T& operator * () const
  {
    if (!m_isEmptyBlock && m_manager && !m_blockIsActivated) {
      m_manager->activateBlock<Data_T>(m_field->m_fileId, m_blockId);
      m_blockIsActivated = true;
      const Block &block = m_field->m_blocks[m_blockId];
      int vi, vj, vk;
      m_field->getVoxelInBlock(x, y, z, vi, vj, vk);
      m_p = &block.value(vi, vj, vk, m_blockOrder);
    }
    return *m_p;
  }
  inline const Data_T* operator -> () const
  {
    if (!m_isEmptyBlock && m_manager && !m_blockIsActivated) {
      SparseFileManager *manager = m_field->m_fileManager;
      manager->activateBlock<Data_T>(m_field->m_fileId, m_blockId);
      m_blockIsActivated = true;
      const Block &block = m_field->m_blocks[m_blockId];
      int vi, vj, vk;
      m_field->getVoxelInBlock(x, y, z, vi, vj, vk);
      m_p = &block.value(vi, vj, vk, m_blockOrder);
    }
    return m_p;
  }

  // Public data members -------------------------------------------------------

  //! Current x/y/z coord
  int x, y, z;

private:

  // Typedefs ------------------------------------------------------------------

  typedef Sparse::SparseBlock<Data_T> Block;

  // Convenience methods -------------------------------------------------------

  void setupNextBlock(int i, int j, int k)
  {
    m_field->applyDataWindowOffset(i, j, k);
    m_field->getBlockCoord(i, j, k, m_blockI, m_blockJ, m_blockK);
    int oldBlockId = m_blockId;
    m_blockId = m_field->blockId(m_blockI, m_blockJ, m_blockK);
    if (m_manager && oldBlockId != m_blockId &&
        oldBlockId >= 0 &&
        oldBlockId < static_cast<int>(m_field->m_numBlocks) &&
        m_field->m_blocks[oldBlockId].isAllocated) {
      m_manager->decBlockRef<Data_T>(m_field->m_fileId, oldBlockId);
    }
    if (m_blockId >= m_field->m_blockXYSize * m_field->m_blockRes.z) {
      m_isEmptyBlock = true;
      return;
    }

    const Block &block = m_field->m_blocks[m_blockId];
    int vi, vj, vk;
    m_field->getVoxelInBlock(i, j, k, vi, vj, vk);
    m_blockStepsTicker = vi;
    if (block.isAllocated) {
      if (m_manager && oldBlockId != m_blockId && m_blockId >= 0) {
        m_manager->incBlockRef<Data_T>(m_field->m_fileId, m_blockId);
        // this is a managed field, so the block may not be loaded
        // yet, so don't bother setting m_p yet (it'll get set in the
        // * and -> operators when the block is activated)
      } else {
        // only set m_p to the voxel's address if this is not a
        // managed field, i.e., if the data is already in memory.
        m_p = &block.value(vi, vj, vk, m_blockOrder);
      }
      m_isEmptyBlock = false;
    } else {
      m_p = &block.emptyValue;
      m_isEmptyBlock = true;
    }
    if (m_field->m_fileManager) {
      m_blockIsActivated = false;
    }
  }

  //! Current pointed-to element
  mutable const Data_T *m_p;
  //! Whether we're at an empty block and we don't increment m_p
  bool m_isEmptyBlock;
  //! Used with delayed-load fields. Check if we've already activated the
  //! current blocks
  mutable bool m_blockIsActivated;
  //! Ticker for how many more steps to take before resetting the pointer
  int m_blockStepsTicker;
  //! Block size
  int m_blockOrder;
  //! Current block index
  int m_blockI, m_blockJ, m_blockK, m_blockId;
  //! Window to traverse
  Box3i m_window;
  //! Reference to field we're traversing
  const class_type *m_field;
  //! Pointer to the singleton file manager
  SparseFileManager *m_manager;
};

//----------------------------------------------------------------------------//
// SparseField::iterator
//----------------------------------------------------------------------------/

//! \todo Code duplication between this and const_iterator !!!!!!!!!!!!!
template <class Data_T>
class SparseField<Data_T>::iterator
{
 public:
#if defined(WIN32) || __MAC_OS_X_VERSION_MIN_REQUIRED >= 1090
  typedef std::forward_iterator_tag iterator_category;
  typedef Data_T value_type;
  typedef ptrdiff_t difference_type;
  typedef ptrdiff_t distance_type;
  typedef Data_T *pointer;
  typedef Data_T& reference;
#endif

  typedef SparseField<Data_T> class_type;
  iterator(class_type &field,
           const Box3i &window,
           const V3i &currentPos, int blockOrder)
    : x(currentPos.x), y(currentPos.y), z(currentPos.z),
      m_p(NULL), m_blockStepsTicker(0), m_blockOrder(blockOrder),
      m_blockId(-1), m_window(window), m_field(&field)
  {
    setupNextBlock(x, y, z);
  }
  const iterator& operator ++ ()
  {
    bool resetPtr = false;
    // Check against end of data window
    if (x == m_window.max.x) {
      if (y == m_window.max.y) {
        x = m_window.min.x;
        y = m_window.min.y;
        ++z;
        resetPtr = true;
      } else {
        x = m_window.min.x;
        ++y;
        resetPtr = true;
      }
    } else {
      ++x;
    }
    // These can both safely be incremented here
    ++m_blockStepsTicker;
    // ... but only step forward if we're in a non-empty block
    if (!m_isEmptyBlock)
      ++m_p;
    // Check if we've reached the end of this block
    if (m_blockStepsTicker == (1 << m_blockOrder))
      resetPtr = true;
    if (resetPtr) {
      // If we have, we need to reset the current block, etc.
      m_blockStepsTicker = 0;
      setupNextBlock(x, y, z);
    }
    return *this;
  }
  inline bool operator == (const iterator &rhs) const
  {
    return x == rhs.x && y == rhs.y && z == rhs.z;
  }
  inline bool operator != (const iterator &rhs) const
  {
    return x != rhs.x || y != rhs.y || z != rhs.z;
  }
  inline Data_T& operator * ()
  {
    if (m_field->m_fileManager) {
      assert(false && "Dereferencing iterator on a dynamic-read sparse field");
      Msg::print(Msg::SevWarning, "Dereferencing iterator on a dynamic-read "
                "sparse field");
      return *m_p;
    }
    // If the block is currently empty, we must allocate it
    if (m_isEmptyBlock) {
      // Touch the voxel to allocate the block
      m_field->lvalue(x, y, z);
      // Set up the block again
      setupNextBlock(x, y, z);
    }
    return *m_p;
  }
  inline Data_T* operator -> ()
  {
    if (m_field->m_fileManager) {
      assert(false && "Dereferencing iterator on a dynamic-read sparse field");
      Msg::print(Msg::SevWarning, "Dereferencing iterator on a dynamic-read "
                "sparse field");
      return m_p;
    }
    // If the block is currently empty, we must allocate it
    if (m_isEmptyBlock) {
      // Touch the voxel to allocate the block
      m_field->lvalue(x, y, z);
      // Set up the block again
      setupNextBlock(x, y, z);
    }
    return m_p;
  }
  // Public data members
  int x, y, z;
private:
  typedef Sparse::SparseBlock<Data_T> Block;
  //! Convenience
  void setupNextBlock(int i, int j, int k)
  {
    m_field->applyDataWindowOffset(i, j, k);
    m_field->getBlockCoord(i, j, k, m_blockI, m_blockJ, m_blockK);
    m_blockId = m_field->blockId(m_blockI, m_blockJ, m_blockK);
    if (m_blockId >= m_field->m_blockXYSize * m_field->m_blockRes.z) {
      m_isEmptyBlock = true;
      return;
    }
    Block &block = m_field->m_blocks[m_blockId];
    int vi, vj, vk;
    m_field->getVoxelInBlock(i, j, k, vi, vj, vk);
    m_blockStepsTicker = vi;
    if (block.isAllocated) {
      m_p = &block.value(vi, vj, vk, m_blockOrder);
      m_isEmptyBlock = false;
    } else {
      m_p = &block.emptyValue;
      m_isEmptyBlock = true;
    }
  }
  //! Current pointed-to element
  Data_T *m_p;
  //! Whether we're at an empty block and we don't increment m_p
  bool m_isEmptyBlock;
  //! Ticker for how many more steps to take before resetting the pointer
  int m_blockStepsTicker;
  //! Block size
  int m_blockOrder;
  //! Current block index
  int m_blockI, m_blockJ, m_blockK, m_blockId;
  //! Window to traverse
  Box3i m_window;
  //! Reference to field we're traversing
  class_type *m_field;
};

//----------------------------------------------------------------------------//
// SparseField::block_iterator
//----------------------------------------------------------------------------/

//! \note This iterator type can not be dereferenced. It's only used to
//! provide a bounding box and indices for each block.
template <class Data_T>
class SparseField<Data_T>::block_iterator
{
 public:
  //! Convenience typedef
  typedef SparseField<Data_T> class_type;
  //! Constructor
  block_iterator(const class_type &field, const Box3i &window,
                 const V3i &currentPos)
    : x(currentPos.x), y(currentPos.y), z(currentPos.z),
      m_window(window), m_field(field)
  {
    recomputeBlockBoundingBox();
  }
  //! Increment iterator
  const block_iterator& operator ++ ()
  {
    if (x == m_window.max.x) {
      if (y == m_window.max.y) {
        x = m_window.min.x;
        y = m_window.min.y;
        ++z;
      } else {
        x = m_window.min.x;
        ++y;
      }
    } else {
      ++x;
    }
    recomputeBlockBoundingBox();
    return *this;
  }
  //! Equality check
  inline bool operator == (const block_iterator &rhs) const
  {
    return x == rhs.x && y == rhs.y && z == rhs.z;
  }
  //! Inequality check
  inline bool operator != (const block_iterator &rhs) const
  {
    return x != rhs.x || y != rhs.y || z != rhs.z;
  }
  //! Returns a reference to the bounding box representing the current block
  const Box3i& blockBoundingBox()
  {
    return m_currentBlockWindow;
  }
  //! Current block index
  int x, y, z;
private:
  void recomputeBlockBoundingBox()
  {
    Box3i box;
    int blockSize = m_field.blockSize();
    box.min = V3i(x * blockSize, y * blockSize, z * blockSize);
    box.max = box.min + V3i(blockSize - 1, blockSize - 1, blockSize - 1);
    // Clamp the box
    box.min = FIELD3D_CLIP(box.min, m_field.dataWindow());
    box.max = FIELD3D_CLIP(box.max, m_field.dataWindow());
    // Set the member variable
    m_currentBlockWindow = box;
  }
  //! Bounding box for block indices
  Box3i m_window;
  //! Pointer to field we're traversing
  const class_type& m_field;
  //! Bounding box in voxel coordinates for the current block
  Box3i m_currentBlockWindow;
};

//----------------------------------------------------------------------------//
// SparseField implementations
//----------------------------------------------------------------------------//

template <class Data_T>
SparseField<Data_T>::SparseField()
  : base(),
    m_blockOrder(BLOCK_ORDER),
    m_blocks(NULL),
    m_fileManager(NULL)
{
  setupBlocks();
}

//----------------------------------------------------------------------------//

template <class Data_T>
SparseField<Data_T>::SparseField(const SparseField<Data_T> &o)
 : base(o),
   m_blockOrder(o.m_blockOrder),
   m_blocks(NULL),
   m_fileManager(o.m_fileManager)
{
  copySparseField(o);
}

//----------------------------------------------------------------------------//

template <class Data_T>
SparseField<Data_T>::~SparseField()
{
  if (m_fileManager) {
    // this file is dynamically managed, so we need to ensure the
    // cache doesn't point to this field's blocks because they are
    // about to be deleted
    m_fileManager->removeFieldFromCache<Data_T>(m_fileId);
  }
  if (m_blocks) {
    delete[] m_blocks;
  }
}

//----------------------------------------------------------------------------//

template <class Data_T>
SparseField<Data_T> &
SparseField<Data_T>::operator=(const SparseField<Data_T> &o)
{
  if (this != &o) {
    this->base::operator=(o);
    copySparseField(o);
  }
  return *this;
}

//----------------------------------------------------------------------------//

template <class Data_T>
void
SparseField<Data_T>::copySparseField(const SparseField<Data_T> &o)
{
  m_blockOrder = o.m_blockOrder;
  if (o.m_fileManager) {
    // allocate m_blocks, sets m_blockRes, m_blockXYSize, m_blocks
    setupBlocks();
    m_fileManager = o.m_fileManager;
    SparseFile::Reference<Data_T> &oldReference =
      m_fileManager->reference<Data_T>(o.m_fileId);
    addReference(oldReference.filename, oldReference.layerPath,
                 oldReference.valuesPerBlock,
                 oldReference.occupiedBlocks);
    copyBlockStates(o);
    setupReferenceBlocks();
  } else {
    // directly copy all values and blocks from the source, no extra setup
    m_blockRes = o.m_blockRes;
    m_blockXYSize = o.m_blockXYSize;
    if (m_blocks) {
      delete[] m_blocks;
    }
    m_numBlocks = o.m_numBlocks;
    m_blocks = new Block[m_numBlocks];
    for (size_t i = 0; i < m_numBlocks; ++i) {
      m_blocks[i].isAllocated = o.m_blocks[i].isAllocated;
      m_blocks[i].emptyValue = o.m_blocks[i].emptyValue;
      m_blocks[i].copy(o.m_blocks[i],
                       1 << m_blockOrder << m_blockOrder << m_blockOrder);
    }
    m_fileId = -1;
    m_fileManager = NULL;
  }
}

//----------------------------------------------------------------------------//

template <class Data_T>
void SparseField<Data_T>::addReference(const std::string &filename,
                                       const std::string &layerPath,
                                       int valuesPerBlock,
                                       int occupiedBlocks)
{
  m_fileManager = &SparseFileManager::singleton();
  m_fileId = m_fileManager->getNextId<Data_T>(filename, layerPath);
  // Set up the manager data
  SparseFile::Reference<Data_T> &reference =
    m_fileManager->reference<Data_T>(m_fileId);
  reference.valuesPerBlock = valuesPerBlock;
  reference.occupiedBlocks = occupiedBlocks;
  reference.setNumBlocks(m_numBlocks);
}

//----------------------------------------------------------------------------//

template <class Data_T>
void SparseField<Data_T>::copyBlockStates(const SparseField<Data_T> &o)
{
  if (m_numBlocks != o.m_numBlocks) return;

  for (size_t i = 0; i < m_numBlocks; ++i) {
    m_blocks[i].isAllocated = o.m_blocks[i].isAllocated;
    m_blocks[i].emptyValue = o.m_blocks[i].emptyValue;
    m_blocks[i].clear();
  }
}

//----------------------------------------------------------------------------//

template <class Data_T>
void SparseField<Data_T>::setupReferenceBlocks()
{
  if (!m_fileManager || m_fileId < 0) return;

  SparseFile::Reference<Data_T> &reference =
    m_fileManager->reference<Data_T>(m_fileId);

  std::vector<int>::iterator fb = reference.fileBlockIndices.begin();
  typename SparseFile::Reference<Data_T>::BlockPtrs::iterator bp =
    reference.blocks.begin();
  int nextBlockIdx = 0;
  for (size_t i = 0; i < m_numBlocks; ++i, ++fb, ++bp) {
    if (m_blocks[i].isAllocated) {
      *fb = nextBlockIdx;
      *bp = m_blocks + i;
      nextBlockIdx++;
    } else {
      *fb = -1;
    }
  }
}

//----------------------------------------------------------------------------//

template <class Data_T>
void SparseField<Data_T>::clear(const Data_T &value)
{
  // If we're clearing, we can get rid of all current blocks
  setupBlocks();
  Block *p = m_blocks, *end = m_blocks + m_numBlocks;
  while (p != end) {
    p->emptyValue = value;
    ++p;
  }
}

//----------------------------------------------------------------------------//

template <class Data_T>
void SparseField<Data_T>::setBlockOrder(int order)
{
  m_blockOrder = order;
  setupBlocks();
}

//----------------------------------------------------------------------------//

template <class Data_T>
int SparseField<Data_T>::blockOrder() const
{
  return m_blockOrder;
}

//----------------------------------------------------------------------------//

template <class Data_T>
int SparseField<Data_T>::blockSize() const
{
  return 1 << m_blockOrder;
}

//----------------------------------------------------------------------------//

template <class Data_T>
bool SparseField<Data_T>::voxelIsInAllocatedBlock(int i, int j, int k) const
{
  int bi, bj, bk;
  applyDataWindowOffset(i, j, k);
  getBlockCoord(i, j, k, bi, bj, bk);
  return blockIsAllocated(bi, bj, bk);
}

//----------------------------------------------------------------------------//

template <class Data_T>
bool SparseField<Data_T>::blockIsAllocated(int bi, int bj, int bk) const
{
  const Block &block = m_blocks[blockId(bi, bj, bk)];
  return block.isAllocated;
}

//----------------------------------------------------------------------------//

template <class Data_T>
const Data_T SparseField<Data_T>::getBlockEmptyValue(int bi, int bj, int bk) const
{
  return m_blocks[blockId(bi, bj, bk)].emptyValue;
}

//----------------------------------------------------------------------------//

template <class Data_T>
void SparseField<Data_T>::setBlockEmptyValue(int bi, int bj, int bk,
                                             const Data_T &val)
{
  Block &block = m_blocks[blockId(bi, bj, bk)];
  if (block.isAllocated) {
    deallocBlock(block, val);
  } else {
    block.emptyValue = val;
  }
}

//----------------------------------------------------------------------------//

template <class Data_T>
bool SparseField<Data_T>::blockIndexIsValid(int bi, int bj, int bk) const
{
  return bi >= 0 && bj >= 0 && bk >= 0 &&
    bi < m_blockRes.x && bj < m_blockRes.y && bk < m_blockRes.z;
}

//----------------------------------------------------------------------------//

template <class Data_T>
V3i SparseField<Data_T>::blockRes() const
{
  return m_blockRes;
}

//----------------------------------------------------------------------------//

template <class Data_T>
template <typename Functor_T>
int SparseField<Data_T>::releaseBlocks(Functor_T func)
{
  Data_T emptyValue;
  int numDeallocs = 0;

  // If the block is on the edge of the field, it may have unused
  // voxels, with undefined values.  We need to pass the range of
  // valid voxels into the check function, so it only looks at valid
  // voxels.
  V3i dataRes = FieldRes::dataResolution();
  V3i validSize;
  V3i blockAllocSize(blockSize());

  int bx = 0, by = 0, bz = 0;
  for (size_t i = 0; i < m_numBlocks; ++i, ++bx) {
    if (bx >= m_blockRes.x) {
      bx = 0;
      ++by;
      if (by >= m_blockRes.y) {
        by = 0;
        ++bz;
      }
    }
    validSize = blockAllocSize;
    if (bx == m_blockRes.x-1) {
      validSize.x = dataRes.x - bx * blockAllocSize.x;
    }
    if (by == m_blockRes.y-1) {
      validSize.y = dataRes.y - by * blockAllocSize.y;
    }
    if (bz == m_blockRes.z-1) {
      validSize.z = dataRes.z - bz * blockAllocSize.z;
    }

    if (m_blocks[i].isAllocated) {
      if (func.check(m_blocks[i], emptyValue, validSize, blockAllocSize)) {
        deallocBlock(m_blocks[i], emptyValue);
        numDeallocs++;
      }
    }
  }
  return numDeallocs;
}

//----------------------------------------------------------------------------//

template <class Data_T>
Data_T SparseField<Data_T>::value(int i, int j, int k) const
{
  return fastValue(i, j, k);
}

//----------------------------------------------------------------------------//

template <class Data_T>
Data_T& SparseField<Data_T>::lvalue(int i, int j, int k)
{
  return fastLValue(i, j, k);
}

//----------------------------------------------------------------------------//

template <class Data_T>
Data_T SparseField<Data_T>::fastValue(int i, int j, int k) const
{
  assert (i >= base::m_dataWindow.min.x);
  assert (i <= base::m_dataWindow.max.x);
  assert (j >= base::m_dataWindow.min.y);
  assert (j <= base::m_dataWindow.max.y);
  assert (k >= base::m_dataWindow.min.z);
  assert (k <= base::m_dataWindow.max.z);
  // Add crop window offset
  applyDataWindowOffset(i, j, k);
  // Find block coord
  int bi, bj, bk;
  getBlockCoord(i, j, k, bi, bj, bk);
  // Find coord in block
  int vi, vj, vk;
  getVoxelInBlock(i, j, k, vi, vj, vk);
  // Get the actual block
  int id = blockId(bi, bj, bk);
  const Block &block = m_blocks[id];
  // Check if block data is allocated
  if (block.isAllocated) {
    if (m_fileManager) {
      m_fileManager->incBlockRef<Data_T>(m_fileId, id);
      m_fileManager->activateBlock<Data_T>(m_fileId, id);
      Data_T tmpValue = block.value(vi, vj, vk, m_blockOrder);
      m_fileManager->decBlockRef<Data_T>(m_fileId, id);
      return tmpValue;
    } else {
      return block.value(vi, vj, vk, m_blockOrder);
    }
  } else {
    return block.emptyValue;
  }
}

//----------------------------------------------------------------------------//

//! \note Bit shift should be ok, indices are always positive.
template <class Data_T>
Data_T& SparseField<Data_T>::fastLValue(int i, int j, int k)
{
  assert (i >= base::m_dataWindow.min.x);
  assert (i <= base::m_dataWindow.max.x);
  assert (j >= base::m_dataWindow.min.y);
  assert (j <= base::m_dataWindow.max.y);
  assert (k >= base::m_dataWindow.min.z);
  assert (k <= base::m_dataWindow.max.z);

  if (m_fileManager) {
    assert(false && "Called fastLValue() on a dynamic-read sparse field");
    Msg::print(Msg::SevWarning, "Called fastLValue() on a dynamic-read "
              "sparse field");
    return m_dummy;
  }

  // Add crop window offset
  applyDataWindowOffset(i, j, k);
  // Find block coord
  int bi, bj, bk;
  getBlockCoord(i, j, k, bi, bj, bk);
  // Find coord in block
  int vi, vj, vk;
  getVoxelInBlock(i, j, k, vi, vj, vk);
  // Get the actual block
  int id = blockId(bi, bj, bk);
  Block &block = m_blocks[id];
  // If block is allocated, return a reference to the data
  if (block.isAllocated) {
    return block.value(vi, vj, vk, m_blockOrder);
  } else {
    // ... Otherwise, allocate block
    size_t blockSize = 1 << m_blockOrder << m_blockOrder << m_blockOrder;
    block.resize(blockSize);
    return block.value(vi, vj, vk, m_blockOrder);
  }
}

//----------------------------------------------------------------------------//

template <class Data_T>
Data_T* SparseField<Data_T>::blockData(int bi, int bj, int bk) const
{
  int id = blockId(bi, bj, bk);
  const Block &block = m_blocks[id];
  if (block.isAllocated) {
    return block.data;
  } else {
    return NULL;
  }
}

//----------------------------------------------------------------------------//

template <class Data_T>
long long int SparseField<Data_T>::memSize() const
{
  long long int blockSize = m_numBlocks * sizeof(Block);
  long long int dataSize = 0;
  typename std::vector<Block>::const_iterator i;

  for (size_t i = 0; i < m_numBlocks; ++i) {
    if (m_blocks[i].isAllocated) {
      dataSize += (1 << m_blockOrder << m_blockOrder << m_blockOrder) *
        sizeof(Data_T);
    }
  }

  return sizeof(*this) + dataSize + blockSize;
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename SparseField<Data_T>::const_iterator
SparseField<Data_T>::cbegin() const
{
  if (FieldRes::dataResolution() == V3i(0))
    return cend();
  return const_iterator(*this, base::m_dataWindow, base::m_dataWindow.min,
                        m_blockOrder);
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename SparseField<Data_T>::const_iterator
SparseField<Data_T>::cbegin(const Box3i &subset) const
{
  if (subset.isEmpty())
    return cend(subset);
  return const_iterator(*this, subset, subset.min, m_blockOrder);
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename SparseField<Data_T>::const_iterator
SparseField<Data_T>::cend() const
{
  return const_iterator(*this, base::m_dataWindow,
                        V3i(base::m_dataWindow.min.x,
                                   base::m_dataWindow.min.y,
                                   base::m_dataWindow.max.z + 1),
                        m_blockOrder);
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename SparseField<Data_T>::const_iterator
SparseField<Data_T>::cend(const Box3i &subset) const
{
  return const_iterator(*this, subset,
                        V3i(subset.min.x,
                                   subset.min.y,
                                   subset.max.z + 1), m_blockOrder);
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename SparseField<Data_T>::iterator
SparseField<Data_T>::begin()
{
  if (FieldRes::dataResolution() == V3i(0))
    return end();
  return iterator(*this, base::m_dataWindow,
                  base::m_dataWindow.min, m_blockOrder); }

//----------------------------------------------------------------------------//

template <class Data_T>
typename SparseField<Data_T>::iterator
SparseField<Data_T>::begin(const Box3i &subset)
{
  if (subset.isEmpty())
    return end(subset);
  return iterator(*this, subset, subset.min, m_blockOrder);
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename SparseField<Data_T>::iterator
SparseField<Data_T>::end()
{
  return iterator(*this, base::m_dataWindow,
                  V3i(base::m_dataWindow.min.x,
                             base::m_dataWindow.min.y,
                             base::m_dataWindow.max.z + 1), m_blockOrder);
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename SparseField<Data_T>::iterator
SparseField<Data_T>::end(const Box3i &subset)
{
  return iterator(*this, subset,
                  V3i(subset.min.x, subset.min.y, subset.max.z + 1),
                  m_blockOrder);
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename SparseField<Data_T>::block_iterator
SparseField<Data_T>::blockBegin() const
{
  if (FieldRes::dataResolution() == V3i(0))
    return blockEnd();
  return block_iterator(*this, Box3i(V3i(0), m_blockRes - V3i(1)),
                        V3i(0));
}

//----------------------------------------------------------------------------//

template <class Data_T>
typename SparseField<Data_T>::block_iterator
SparseField<Data_T>::blockEnd() const
{
  return block_iterator(*this, Box3i(V3i(0), m_blockRes - V3i(1)),
                        V3i(0, 0, m_blockRes.z));
}

//----------------------------------------------------------------------------//

template <class Data_T>
void SparseField<Data_T>::setupBlocks()
{
  // Do calculation in floating point so we can round up later
  V3f res(base::m_dataWindow.size() + V3i(1));
  V3f blockRes(res / (1 << m_blockOrder));
  blockRes.x = ceil(blockRes.x);
  blockRes.y = ceil(blockRes.y);
  blockRes.z = ceil(blockRes.z);
  V3i intBlockRes(static_cast<int>(blockRes.x),
                         static_cast<int>(blockRes.y),
                         static_cast<int>(blockRes.z));
  m_blockRes = intBlockRes;
  m_blockXYSize = m_blockRes.x * m_blockRes.y;
  if (m_blocks) {
    delete[] m_blocks;
  }
  m_numBlocks = intBlockRes.x * intBlockRes.y * intBlockRes.z;
  m_blocks = new Block[m_numBlocks];
}

//----------------------------------------------------------------------------//

template <class Data_T>
int SparseField<Data_T>::blockId(int blockI, int blockJ, int blockK) const
{
  return blockK * m_blockXYSize + blockJ * m_blockRes.x + blockI;
}

//----------------------------------------------------------------------------//

//! \note Bit shift should be ok, indices are always positive.
template <class Data_T>
void SparseField<Data_T>::getBlockCoord(int i, int j, int k,
                                        int &bi, int &bj, int &bk) const
{
  assert(i >= 0);
  assert(j >= 0);
  assert(k >= 0);
  bi = i >> m_blockOrder;
  bj = j >> m_blockOrder;
  bk = k >> m_blockOrder;
}

//----------------------------------------------------------------------------//

//! \note Bit shift should be ok, indices are always positive.
template <class Data_T>
void SparseField<Data_T>::getVoxelInBlock(int i, int j, int k,
                                          int &vi, int &vj, int &vk) const
{
  assert(i >= 0);
  assert(j >= 0);
  assert(k >= 0);
  vi = i & ((1 << m_blockOrder) - 1);
  vj = j & ((1 << m_blockOrder) - 1);
  vk = k & ((1 << m_blockOrder) - 1);
}

//----------------------------------------------------------------------------//

template <class Data_T>
void SparseField<Data_T>::deallocBlock(Block &block, const Data_T &emptyValue)
{
  block.isAllocated = false;
  //! Block::clear() deallocates the data
  block.clear();
  block.emptyValue = emptyValue;
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
