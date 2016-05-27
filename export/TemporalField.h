//----------------------------------------------------------------------------//

#ifndef _INCLUDED_TemporalField_H_
#define _INCLUDED_TemporalField_H_

//----------------------------------------------------------------------------//

#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/thread/mutex.hpp>

#include <OpenEXR/ImathFun.h>

#include <zlib.h>

#include "Field.h"

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// LinearTemporalFieldInterp
//----------------------------------------------------------------------------//

template <typename Data_T>
class TemporalField;

template <typename Data_T>
class LinearTemporalFieldInterp : public RefBase
{
public:

  // Typedefs ------------------------------------------------------------------

  typedef Data_T value_type;
  typedef boost::intrusive_ptr<LinearTemporalFieldInterp> Ptr;

  // RTTI replacement ----------------------------------------------------------

  typedef LinearTemporalFieldInterp class_type;
  DEFINE_FIELD_RTTI_CONCRETE_CLASS;

  static const char *staticClassName()
  {
    return "LinearTemporalFieldInterp";
  }

  static const char* staticClassType()
  {
    return ms_classType.name();
  }

  // Main methods --------------------------------------------------------------

  value_type sample(const TemporalField<Data_T> &field, const V3d &vsP,
                    const float time) const
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

    // If in the middle of a block, optimize lookup stencil
    return static_cast<Data_T>
      (f1.x * (f1.y * (f1.z * field.fastValue(c1.x, c1.y, c1.z, time) +
                       f2.z * field.fastValue(c1.x, c1.y, c2.z, time)) +
               f2.y * (f1.z * field.fastValue(c1.x, c2.y, c1.z, time) +
                       f2.z * field.fastValue(c1.x, c2.y, c2.z, time))) +
       f2.x * (f1.y * (f1.z * field.fastValue(c2.x, c1.y, c1.z, time) +
                       f2.z * field.fastValue(c2.x, c1.y, c2.z, time)) +
               f2.y * (f1.z * field.fastValue(c2.x, c2.y, c1.z, time) +
                       f2.z * field.fastValue(c2.x, c2.y, c2.z, time))));

  }

private:

  // Static data members -------------------------------------------------------

  static TemplatedFieldType<LinearTemporalFieldInterp<Data_T> > ms_classType;

  // Typedefs ------------------------------------------------------------------

  //! Convenience typedef for referring to base class
  typedef RefBase base;

};

//----------------------------------------------------------------------------//
// WATemporalFieldInterp
//----------------------------------------------------------------------------//

//! Weighted average temporal field interp
template <typename Data_T>
class WATemporalFieldInterp : public RefBase
{
public:

  // Typedefs ------------------------------------------------------------------

  typedef Data_T value_type;
  typedef boost::intrusive_ptr<WATemporalFieldInterp> Ptr;

  // RTTI replacement ----------------------------------------------------------

  typedef WATemporalFieldInterp class_type;
  DEFINE_FIELD_RTTI_CONCRETE_CLASS;

  static const char *staticClassName()
  {
    return "WATemporalFieldInterp";
  }

  static const char* staticClassType()
  {
    return ms_classType.name();
  }

  // Main methods --------------------------------------------------------------

  value_type sample(const TemporalField<Data_T> &field, const V3d &vsP,
                    const float time) const
  {
    // Pixel centers are at .5 coordinates
    // NOTE: Don't use contToDisc for this, we're looking for sample
    // point locations, not coordinate shifts.
    FIELD3D_VEC3_T<double> p(vsP - FIELD3D_VEC3_T<double>(0.5));
    FIELD3D_VEC3_T<double> floorP(std::floor(p.x), 
                                  std::floor(p.y),
                                  std::floor(p.z));

    // Lower left corner
    V3i c1(static_cast<int>(floorP.x),
           static_cast<int>(floorP.y),
           static_cast<int>(floorP.z));
    // Upper right corner
    V3i c2(c1 + V3i(1));
    // C1 fractions
    FIELD3D_VEC3_T<double> f1(static_cast<FIELD3D_VEC3_T<double> >(c2) - p);
    // C2 fraction
    FIELD3D_VEC3_T<double> f2(static_cast<FIELD3D_VEC3_T<double> >(1.0) - f1);
    // Data window
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

    // 4D coord of sample
    Imath::Vec4<double> sample(p.x, p.y, p.z, time);
    
    // Gather positions and values ---

    bool                present[16];
    Imath::Vec4<double> positions[16];
    Data_T              values[16];

    for (int s = 0; s < 8; ++s) {

      // Voxel offset
      const int iInc = ((s % 2) == 1) ? 1 : 0;
      const int jInc = ((s / 2) % 2 == 1) ? 1 : 0;
      const int kInc = ((s / 4) > 0) ? 1 : 0;
      const int i = c1.x + iInc;
      const int j = c1.y + jInc;
      const int k = c1.z + kInc;
      // Get temporal neighbors
      float t0 = 0.0, t1 = 0.0;
      Data_T v0(0.0), v1(0.0);
      field.getNearestSamples(i, j, k, time, present[s * 2], present[s * 2 + 1],
                              t0, t1, v0, v1);
      // We don't check for value presence here, that'll be done below.
      // Position
      positions[s * 2] = positions[s * 2 + 1] = Imath::V4d(floorP.x + iInc, 
                                                           floorP.y + jInc,
                                                           floorP.z + kInc, 
                                                           0.0);
      positions[s * 2].w     = t0;
      positions[s * 2 + 1].w = t1;
      // Value
      values[s * 2]     = v0;
      values[s * 2 + 1] = v1;

    }

    // Perform weighted average ---

    Data_T accumResult = 0.0;
    double accumWeight = 0.0;

    for (int i = 0; i < 16; ++i) {
      if (present[i]) {
        const double weight = 
          1.0 / std::max(1e-6, (sample - positions[i]).length());
        accumResult += weight * values[i];
        accumWeight += weight;
      }
    }

    // Return result ---

    if (accumWeight > 0.0) {
      return accumResult / accumWeight;
    } else {
      return 0.0;
    }

  }

private:

  // Static data members -------------------------------------------------------

  static TemplatedFieldType<WATemporalFieldInterp<Data_T> > ms_classType;

  // Typedefs ------------------------------------------------------------------

  //! Convenience typedef for referring to base class
  typedef RefBase base;

};

//----------------------------------------------------------------------------//
// TemporalStochasticInterp
//----------------------------------------------------------------------------//

template <typename Data_T>
class TemporalStochasticInterp : public RefBase
{
public:
  
  // Typedefs ------------------------------------------------------------------

  typedef Data_T                                         value_type;
  typedef boost::intrusive_ptr<TemporalStochasticInterp> Ptr;

  // RTTI replacement ----------------------------------------------------------

  typedef TemporalStochasticInterp class_type;
  DEFINE_FIELD_RTTI_CONCRETE_CLASS;

  static const char *staticClassName()
  {
    return "TemporalStochasticInterp";
  }
  
  static const char* staticClassType()
  {
    return ms_classType.name();
  }

  // Main methods --------------------------------------------------------------

  value_type linear(const TemporalField<Data_T> &data, 
                    const V3d &vsP, const float time,
                    const float xiX, const float xiY, const float xiZ) const
  {
    // Voxel coords
    V3i c1, c2;
    // Interpolation weights
    FIELD3D_VEC3_T<double> f1, f2;
    // Grab weights
    getLerpInfo(vsP, data.dataWindow(), f1, f2, c1, c2);

    // Choose c1 or c2 based on random variables
    return data.fastValue(xiX < f1.x ? c1.x : c2.x, 
                          xiY < f1.y ? c1.y : c2.y,
                          xiZ < f1.z ? c1.z : c2.z,
                          time);
  }
  
private:

  // Typedefs ------------------------------------------------------------------

  //! Convenience typedef for referring to base class
  typedef RefBase base;    

  // Static data members -------------------------------------------------------

  static TemplatedFieldType<TemporalStochasticInterp<Data_T> > ms_classType;

};

//----------------------------------------------------------------------------//

FIELD3D_CLASSTYPE_TEMPL_INSTANTIATION(LinearTemporalFieldInterp);
FIELD3D_CLASSTYPE_TEMPL_INSTANTIATION(WATemporalFieldInterp);
FIELD3D_CLASSTYPE_TEMPL_INSTANTIATION(TemporalStochasticInterp);

//----------------------------------------------------------------------------//
// TemporalBlock
//----------------------------------------------------------------------------//

/*! \class TemporalBlock
  \brief This class encapsulates a block of temporal functions.
  Each voxel in the block stores an index into the time[] and value[] arrays.

  There are some 'implicit' states:

  x If m_offsets == NULL and m_times == NULL but m_values != NULL,
    then m_values stores a single "empty value".
  x If all pointers are non-NULL, then it is an ordinary block

  The number of samples in a given voxel is found through:
    size_t numSamples = offset(i, j, k) - offset(i + 1, j, k);
  ... thus, m_offsets is always: numVoxels + 1

  \warning The data passed to setArrays() is assumed to be the ownership
  of TemporalBlock, and upon destruction of the TemporalBlock, those arrays
  are delete[]'d

 */

//----------------------------------------------------------------------------//

template <typename Data_T>
class TemporalBlock : public boost::noncopyable
{
public:
  
  // Enums ---------------------------------------------------------------------

  //! State enumerator
  enum State {
    ActiveState = 0,
    EmptyState,
    VoidState,
    InvalidState
  };
  
  // Ctors, dtor ---------------------------------------------------------------

  //! Constructor. Sets all pointers null.
  TemporalBlock();
  //! Destructor. Cleans out existing array data
  ~TemporalBlock();

  // Class methods -------------------------------------------------------------

  static size_t numVoxelsPerBlock(const int blockOrder)
  { return 1 << (3 * blockOrder); }

  // Main methods --------------------------------------------------------------

  //! Returns the state of the block
  State state() const;
  //! Returns the 'empty value'
  Data_T emptyValue() const;
  //! Returns the number of samples in the entire block
  int numSamples(const int blockOrder) const;
  //! Returns the number of samples used in a given voxel
  int numSamples(int i, int j, int k,
                 const int blockOrder) const;
  //! Returns pointer to the sample times for a given voxel
  float* sampleTimes(int i, int j, int k, const int blockOrder);
  //! Returns pointer to the sample times for a given voxel
  const float* sampleTimes(int i, int j, int k,
                           const int blockOrder) const;
  //! Returns pointer to the sample values for a given voxel
  Data_T* sampleValues(int i, int j, int k, const int blockOrder);
  //! Returns pointer to the sample values for a given voxel
  const Data_T* sampleValues(int i, int j, int k,
                             const int blockOrder) const;
  //! Interpolates in time.
  //! \note Assumes the caller knows the block is allocated.
  Data_T interp(const int i, const int j, const int k, const float t,
                const int blockOrder) const;
  //! Gets the nearest temporal samples
  void getNearestSamples(const int i, const int j, const int k, 
                         const int blockOrder, const float t, 
                         bool &hasValue0, bool &hasValue1,
                         float &t0, float &t1, Data_T &v0, Data_T &v1) const;
  //! Sets the arrays
  void setArrays(int *offsets, float *times, Data_T *values);
  //! Returns the pointer to the offset data
  const int* offsets() const
  { return m_offsets; }
  //! Returns the pointer to the time data
  const float* times() const
  { return m_times; }
  //! Returns the pointer to the value data
  const Data_T* values() const
  { return m_values; }
  //! Returns the m_offset array index for a given voxel
  //! \param i I index within block. Always positive.
  //! \param j J index within block. Always positive.
  //! \param k K index within block. Always positive.
  //! \param blockOrder the log2 size of the block.
  int offset(int i, int j, int k, int blockOrder) const;
  //! Returns the memory use in bytes.
  long long int memSize(int blockOrder) const;
  //! Copies the data from another block
  void copy(const TemporalBlock &other, size_t blockOrder);
  //! Clears the block, setting all pointers to null
  void clear();

private:

  friend class TemporalFieldIO;

  // Utility -------------------------------------------------------------------

  //! Used when finding values in the m_samples vector.
  struct CheckTGreaterThan : 
    public std::unary_function<std::pair<float, Data_T>, bool>
  {
    CheckTGreaterThan(float match)
      : m_match(match)
    { }
    bool operator()(float value)
    {
      return value > m_match;
    }
  private:
    float m_match;
  };

  // Data members --------------------------------------------------------------

  //! Offset value per voxel
  int *m_offsets;
  //! Time samples
  float *m_times;
  //! Value samples
  Data_T *m_values;

};  

//----------------------------------------------------------------------------//
// TemporalField
//----------------------------------------------------------------------------//

/*! \class TemporalField
  \brief This class implements the "temporal volume" concept.

  \note Temporal fields are neither writable nor resizable through the
  general interface. We rely on TemporalBlock being configured before
  being added to the field.
*/

//----------------------------------------------------------------------------//

template <typename Data_T>
class TemporalField : public Field3D::Field<Data_T>
{
public:

  // Typedefs ------------------------------------------------------------------
  
  typedef boost::intrusive_ptr<TemporalField> Ptr;
  typedef std::vector<Ptr> Vec;

  typedef LinearTemporalFieldInterp<Data_T> LinearInterp;
  typedef TemporalStochasticInterp<Data_T>  StochasticInterp;

  typedef TemporalBlock<Data_T> Block;

  // Constructors --------------------------------------------------------------

  //! \name Constructors & destructor
  //! \{

  //! Constructs an empty temporal field
  TemporalField();

  //! Copy constructor. Copies data
  TemporalField(const TemporalField &o);

  //! Assignment operator. Copies data
  TemporalField& operator=(const TemporalField &o);

  //! Destructor. Deallocates memory.
  ~TemporalField();

  // \}

  // From Field base class -----------------------------------------------------

  //! \name From Field
  //! \{  
  //! For a temporal field, the common value() call returns the state
  //! of the field at time=0
  virtual Data_T value(int i, int j, int k) const;
  //! Returns current memory use
  virtual long long int memSize() const;
  //! \}

  // RTTI replacement ----------------------------------------------------------

  typedef TemporalField<Data_T> class_type;
  DEFINE_FIELD_RTTI_CONCRETE_CLASS

  static const char *staticClassName()
  {
    return "TemporalField";
  }

  static const char *staticClassType()
  {
    return TemporalField<Data_T>::ms_classType.name();
  }
    
  // Concrete voxel access -----------------------------------------------------

  //! Read access to value at a given voxel. Time is implicitly 0.0
  //! \note Does not return a reference, since the value will be interpolated
  Data_T fastValue(int i, int j, int k) const;

  //! Read access to value at a given voxel and time.
  //! \note Does not return a reference, since the value will be interpolated
  Data_T fastValue(int i, int j, int k, float t) const;

  // From FieldBase ------------------------------------------------------------

  //! \name From FieldBase
  //! \{

  FIELD3D_CLASSNAME_CLASSTYPE_IMPLEMENTATION;
  
  virtual FieldBase::Ptr clone() const;

  //! \}

  // Main methods --------------------------------------------------------------

  //! Clears all data in the volume
  void clear();

  //! Returns a pointer to a block, given a data window
  Block* block(const Box3i &domain);
  //! Returns a pointer to a block, given a data window
  const Block* block(const Box3i &domain) const;
  //! Returns a pointer to a block, given its coordinate
  Block* block(const int bi, const int bj, const int bk);
  //! Returns a pointer to a block, given its coordinate
  const Block* block(const int bi, const int bj, const int bk) const;

  //! Returns the block resolution
  V3i blockRes() const;
  //! Returns the extents of a given block
  Box3i blockExtents(int bi, int bj, int bk) const;

  //! Returns the number of samples used in a given voxel
  int numSamples(int i, int j, int k) const;
  //! Returns pointer to the sample times for a given voxel
  float* sampleTimes(int i, int j, int k);
  //! Returns pointer to the sample times for a given voxel
  const float* sampleTimes(int i, int j, int k) const;
  //! Returns pointer to the sample values for a given voxel
  Data_T* sampleValues(int i, int j, int k);
  //! Returns pointer to the sample values for a given voxel
  const Data_T* sampleValues(int i, int j, int k) const;
  //! Gets the nearest temporal samples
  void getNearestSamples(int i, int j, int k,
                         const float t, bool &hasValue0, bool &hasValue1,
                         float &t0, float &t1, Data_T &v0, Data_T &v1) const;

  //! Returns the average sample count per voxel
  float averageNumSamples() const;

  //! Sets the block order (i.e. the power-of-2 to use as block size.
  //! \note This will clear out any existing data.
  void setBlockOrder(int order);
  //! Returns the block order
  int blockOrder() const;

  //! Applies data window offset
  void applyDataWindowOffset(int &i, int &j, int &k) const
  {
    i -= base::m_dataWindow.min.x;
    j -= base::m_dataWindow.min.y;
    k -= base::m_dataWindow.min.z;
  }

  //! Returns the size of a block
  int blockSize() const;

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

  // Size setup ----------------------------------------------------------------

  //! Resizes the object
  void setSize(const V3i &size);
  //! Resizes the object
  void setSize(const Box3i &extents);
  //! Resizes the object
  void setSize(const Box3i &extents, const Box3i &dataWindow);
  //! Resizes the object with padding
  void setSize(const V3i &size, int padding);
  //! Sets up this field so that resolution and mapping matches the other
  void matchDefinition(FieldRes::Ptr fieldToMatch);

  // Threading-related ---------------------------------------------------------

  //! Number of 'grains' to use with threaded access
  size_t numGrains() const;
  //! Bounding box of the given 'grain'
  //! \return Whether the grain is contiguous in memory
  bool   getGrainBounds(const size_t idx, Box3i &vsBounds) const;

protected:

  friend class TemporalFieldIO;

  // Typedefs ------------------------------------------------------------------

  typedef Field<Data_T> base;

  // Static data members -------------------------------------------------------

  static TemplatedFieldType<TemporalField<Data_T> > ms_classType;

  // Data members --------------------------------------------------------------

  //! Block order
  int m_blockOrder;
  //! Block array resolution
  V3i m_blockRes;
  //! Block x/y stride
  size_t m_blockXYSize; 
  
  //! Array of TemporalBlock instances
  Block *m_blocks;
  //! Number of blocks in array
  size_t m_numBlocks;

  // Utility methods -----------------------------------------------------------

  //! Called when the size of the field changes. Basically just clears all
  //! existing data
  void sizeChanged(); 
  //! Initializes the block structure. Will clear any existing data
  void setupBlocks();
  //! Copies from a second TemporalField
  void copyTemporalField(const TemporalField &other);

};

//----------------------------------------------------------------------------//
// Typedefs
//----------------------------------------------------------------------------//

typedef TemporalField<half>   TemporalFieldh;
typedef TemporalField<float>  TemporalFieldf;
typedef TemporalField<double> TemporalFieldd;
typedef TemporalField<V3h>    TemporalField3h;
typedef TemporalField<V3f>    TemporalField3f;
typedef TemporalField<V3d>    TemporalField3d;

//------------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------------

template <typename Data_T>
Box3i blockCoords(const Box3i &dvsBounds, const TemporalField<Data_T> *f)
{
  // Check empty bbox input
  if (!continuousBounds(dvsBounds).hasVolume()) {
    return Box3i();
  }
  // Discrete offset voxel space
  Box3i dovsBounds = dvsBounds;
  f->applyDataWindowOffset(dovsBounds.min.x, 
                           dovsBounds.min.y, 
                           dovsBounds.min.z);
  f->applyDataWindowOffset(dovsBounds.max.x, 
                           dovsBounds.max.y, 
                           dovsBounds.max.z);
  // Discrete block space bounds
  Box3i dbsBounds;
  if (f) {
    f->getBlockCoord(dovsBounds.min.x, dovsBounds.min.y, dovsBounds.min.z,
                     dbsBounds.min.x, dbsBounds.min.y, dbsBounds.min.z);
    f->getBlockCoord(dovsBounds.max.x, dovsBounds.max.y, dovsBounds.max.z,
                     dbsBounds.max.x, dbsBounds.max.y, dbsBounds.max.z);
  } 
  return dbsBounds;
}

//----------------------------------------------------------------------------//
// TemporalBlock implementations
//----------------------------------------------------------------------------//

template <typename Data_T>
TemporalBlock<Data_T>::TemporalBlock()
  : m_offsets(NULL), m_times(NULL), m_values(NULL)
{ }

//----------------------------------------------------------------------------//

template <typename Data_T>
TemporalBlock<Data_T>::~TemporalBlock()
{
  clear();
}

//----------------------------------------------------------------------------//

template <typename Data_T>
typename TemporalBlock<Data_T>::State 
TemporalBlock<Data_T>::state() const
{
  if (!m_offsets && !m_times) {
    if (m_values) {
      return EmptyState;
    } else {
      return VoidState;
    }
  } else if (m_offsets && m_times && m_values) {
    return ActiveState;
  } else {
    return InvalidState;
  }
}

//----------------------------------------------------------------------------//

template <typename Data_T>
Data_T TemporalBlock<Data_T>::emptyValue() const
{
  return *m_values;
}

//----------------------------------------------------------------------------//

template <class Data_T>
int TemporalBlock<Data_T>::numSamples(const int blockOrder) const
{
  if (!m_offsets) {
    return 0;
  }
  return offset((1 << blockOrder), 
                (1 << blockOrder) - 1, 
                (1 << blockOrder) - 1, 
                blockOrder);
}

//----------------------------------------------------------------------------//

template <class Data_T>
int TemporalBlock<Data_T>::numSamples(int i, int j, int k,
                                      const int blockOrder) const
{
  if (!m_offsets) {
    return 0;
  }
  return offset(i + 1, j, k, blockOrder) - offset(i, j, k, blockOrder);
}

//----------------------------------------------------------------------------//

template <class Data_T>
float* 
TemporalBlock<Data_T>::sampleTimes(int i, int j, int k, const int blockOrder) 
{
  return &m_times[offset(i, j, k, blockOrder)];
}

//----------------------------------------------------------------------------//

template <class Data_T>
const float* 
TemporalBlock<Data_T>::sampleTimes(int i, int j, int k,
                                   const int blockOrder) const
{
  return &m_times[offset(i, j, k, blockOrder)];
}

//----------------------------------------------------------------------------//

template <class Data_T>
Data_T* 
TemporalBlock<Data_T>::sampleValues(int i, int j, int k, const int blockOrder) 
{
  return &m_values[offset(i, j, k, blockOrder)];
}

//----------------------------------------------------------------------------//

template <class Data_T>
const Data_T* 
TemporalBlock<Data_T>::sampleValues(int i, int j, int k,
                                    const int blockOrder) const
{
  return &m_values[offset(i, j, k, blockOrder)];
}

//----------------------------------------------------------------------------//

template <typename Data_T>
Data_T TemporalBlock<Data_T>::interp(const int i, const int j, const int k, 
                                     const float t, const int blockOrder) const
{ 
  const int offsetThis = offset(i, j, k, blockOrder);
  const int offsetNext = offset(i + 1, j, k, blockOrder);
  const float *start = m_times + offsetThis;
  const float *end   = m_times + offsetNext;
  // If there are no samples, return zero
  if (start == end) {
    return static_cast<Data_T>(0.0);
  }
  // Find the first sample location that is greater than the interpolation
  // position
  const float *pos = find_if(start, end, CheckTGreaterThan(t));
  // If we get end() back then there was no sample larger, so we return the
  // last value. If we got the first value then there is only one value and
  // we return that.
  if (pos == end) {
    return m_values[offsetNext - 1];
  } else if (pos == start) {
    return m_values[offsetThis];
  }
  // Interpolate between the nearest two values.
  const float interpT = Imath::lerpfactor(t, *(pos - 1), *pos);
  return Imath::lerp(m_values[offsetThis + (pos - start) - 1], 
                     m_values[offsetThis + (pos - start)], 
                     interpT);
}

//----------------------------------------------------------------------------//

template <typename Data_T>
void 
TemporalBlock<Data_T>::getNearestSamples(const int i, const int j, const int k,
                                         const int blockOrder, 
                                         const float t, 
                                         bool &hasValue0, bool &hasValue1,
                                         float &t0, float &t1, 
                                         Data_T &v0, Data_T &v1) const
{
  const int offsetThis = offset(i, j, k, blockOrder);
  const int offsetNext = offset(i + 1, j, k, blockOrder);
  const float *start = m_times + offsetThis;
  const float *end   = m_times + offsetNext;
  // If there are no samples, return zero
  if (start == end) {
    hasValue0 = false;
    hasValue1 = false;
    return;
  }
  // Find the first sample location that is greater than the interpolation
  // position
  const float *pos = find_if(start, end, CheckTGreaterThan(t));
  // If we get end() back then there was no sample larger, so we return the
  // last value. If we got the first value then there is only one value and
  // we return that.
  if (pos == end) {
    hasValue0 = true;
    hasValue1 = false;
    t0 = m_times[offsetNext - 1];
    v0 = m_values[offsetNext - 1];
  } else if (pos == start) {
    hasValue0 = false;
    hasValue1 = true;
    t1 = m_times[offsetThis];
    v1 = m_values[offsetThis];
  }
  // Has both neighbors
  hasValue0 = hasValue1 = true;
  t0 = m_times[offsetThis + (pos - start) - 1];
  t1 = m_times[offsetThis + (pos - start)];
  v0 = m_values[offsetThis + (pos - start) - 1];
  v1 = m_values[offsetThis + (pos - start)];
}

//----------------------------------------------------------------------------//

template <typename Data_T>
void 
TemporalBlock<Data_T>::setArrays(int *offsets, float *times, Data_T *values)
{
  m_offsets = offsets;
  m_times = times;
  m_values = values;
}

//----------------------------------------------------------------------------//

template <typename Data_T>
int TemporalBlock<Data_T>::offset(int i, int j, int k, int blockOrder) const 
{
  return m_offsets[(k << blockOrder << blockOrder) + (j << blockOrder) + i];
} 


//----------------------------------------------------------------------------//

template <typename Data_T>
long long int TemporalBlock<Data_T>::memSize(int blockOrder) const
{
  size_t blockSize = 1 << blockOrder;
  int start = offset(0, 0, 0, blockOrder);
  int end   = offset(blockSize, blockSize - 1, blockSize - 1,
                     blockOrder);
  int arrayLength = end - start;
  return ((1 << (blockOrder * 3)) + 1) * sizeof(int) + 
    arrayLength * (sizeof(float) + sizeof(Data_T));
}

//----------------------------------------------------------------------------//

template <typename Data_T>
void TemporalBlock<Data_T>::clear()
{
  if (m_offsets) {
    delete[] m_offsets;
    m_offsets = NULL;
  }
  if (m_times) {
    delete[] m_times;
    m_times = NULL;
  }
  if (m_values) {
    delete[] m_values;
    m_values = NULL;
  }
}

//----------------------------------------------------------------------------//

template <typename Data_T>
void TemporalBlock<Data_T>::copy(const TemporalBlock &other, size_t blockOrder)
{
  size_t blockSize = 1 << blockOrder << blockOrder << blockOrder;

  clear();

  if (!other.m_offsets && !other.m_times) {
    if (other.m_values) {
      // EmptyState;
      m_values    = new Data_T[1];
      m_values[0] = other.m_values[0];
    } else {
      // VoidState;
    }
  } else if (other.m_offsets && other.m_times && other.m_values) {
    size_t numVoxels = blockSize;
    size_t numValues = other.numSamples(blockOrder);
    m_offsets = new int[numVoxels + 1];
    m_times   = new float[numValues];
    m_values  = new Data_T[numValues];
    std::copy(other.m_offsets, other.m_offsets + numVoxels + 1, m_offsets);
    std::copy(other.m_times, other.m_times + numValues, m_times);
    std::copy(other.m_values, other.m_values + numValues, m_values);
  } else {
    // InvalidState;
  }
}

//----------------------------------------------------------------------------//
// TemporalField implementations
//----------------------------------------------------------------------------//

template <typename Data_T>
TemporalField<Data_T>::TemporalField() 
  : base(),
    m_blocks(NULL)
{
  setBlockOrder(4);
}

//----------------------------------------------------------------------------//

template <typename Data_T>
TemporalField<Data_T>::TemporalField(const TemporalField &o)
  : base(o),
    m_blocks(NULL)
{
  copyTemporalField(o);
}

//----------------------------------------------------------------------------//

template <class Data_T>
TemporalField<Data_T> &
TemporalField<Data_T>::operator=(const TemporalField<Data_T> &o)
{
  if (this != &o) {
    this->base::operator=(o);
    copyTemporalField(o);
  }
  return *this;
}

//----------------------------------------------------------------------------//

template <typename Data_T>
TemporalField<Data_T>::~TemporalField()
{
  clear();
}

//----------------------------------------------------------------------------//

template <typename Data_T>
void TemporalField<Data_T>::clear()
{
  // Clear existing blocks
  if (m_blocks) {
    delete[] m_blocks;
  }
}

//----------------------------------------------------------------------------//

template <typename Data_T>
typename TemporalField<Data_T>::Block*
TemporalField<Data_T>::block(const Box3i &domain)
{
  return const_cast<Block*>(
      static_cast<const TemporalField*>(this)->block(domain));
}

//----------------------------------------------------------------------------//

template <typename Data_T>
const typename TemporalField<Data_T>::Block*
TemporalField<Data_T>::block(const Box3i &domain) const
{
  int bSize = blockSize();
  // Check that the size of the domain matches a block
  V3i size = domain.size() + V3i(1);
  // Get min corner
  V3i min = domain.min;
  V3i max = domain.max;
  // Add crop window offset
  applyDataWindowOffset(min.x, min.y, min.z);
  applyDataWindowOffset(max.x, max.y, max.z);
  // Check that min is at the corner of a block
  if (min.x % bSize != 0 || min.y % bSize != 0 || min.z % bSize) {
    return NULL;
  }
  // Check if max is at the edge of the data window and domain is smaller 
  // than a block (i.e. the edge blocks of the buffer).
  if ((domain.max.x == base::m_dataWindow.max.x ||
       domain.max.y == base::m_dataWindow.max.y ||
       domain.max.z == base::m_dataWindow.max.z) &&
      (size.x <= bSize && size.y <= bSize && size.z <= bSize)) {
    // Nothing, just here to support the else ifs below for clarity.
  } else if (size.x != bSize || size.y != bSize || size.z != bSize) {
    // Check that the size matches a block
    return NULL;
  } else if ((max.x + 1) % bSize != 0 ||
             (max.y + 1) % bSize != 0 ||
             (max.z + 1) % bSize != 0) {
    // Check that max is at the corner of a block
    return NULL;
  }
  // Ok, we're asking for a proper block. Return it.
  int bi, bj, bk;
  getBlockCoord(min.x, min.y, min.z, bi, bj, bk);
  return &m_blocks[blockId(bi, bj, bk)];
}

//----------------------------------------------------------------------------//

template <typename Data_T>
typename TemporalField<Data_T>::Block*
TemporalField<Data_T>::block(const int bi, const int bj, const int bk)
{
  return &m_blocks[blockId(bi, bj, bk)];
}

//----------------------------------------------------------------------------//

template <typename Data_T>
const typename TemporalField<Data_T>::Block*
TemporalField<Data_T>::block(const int bi, const int bj, const int bk) const
{
  return &m_blocks[blockId(bi, bj, bk)];
}

//----------------------------------------------------------------------------//

template <typename Data_T>
V3i TemporalField<Data_T>::blockRes() const
{ 
  return m_blockRes; 
}

//----------------------------------------------------------------------------//

template <typename Data_T>
Box3i TemporalField<Data_T>::blockExtents(int bi, int bj, int bk) const
{ 
  V3i lower(bi * (1 << m_blockOrder), 
            bj * (1 << m_blockOrder),
            bk * (1 << m_blockOrder));
  V3i upper = lower + V3i(1 << m_blockOrder) - V3i(1);
  applyDataWindowOffset(lower.x, lower.y, lower.z);
  applyDataWindowOffset(upper.x, upper.y, upper.z);
  upper.x = std::min(upper.x, base::m_dataWindow.max.x);
  upper.y = std::min(upper.y, base::m_dataWindow.max.y);
  upper.z = std::min(upper.z, base::m_dataWindow.max.z);
  return Box3i(lower, upper);
}

//----------------------------------------------------------------------------//

template <class Data_T>
int 
TemporalField<Data_T>::numSamples(int i, int j, int k) const
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
  typename Block::State state = block.state();
  if (state == Block::ActiveState) {
    return block.numSamples(vi, vj, vk, m_blockOrder);
  } else {
    return 0;
  } 
}

//----------------------------------------------------------------------------//

template <class Data_T>
float* 
TemporalField<Data_T>::sampleTimes(int i, int j, int k) 
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
  Block &block = m_blocks[id];
  // Check if block data is allocated
  typename Block::State state = block.state();
  if (state == Block::ActiveState) {
    return block.sampleTimes(vi, vj, vk, m_blockOrder);
  } else {
    return 0;
  } 
}

//----------------------------------------------------------------------------//

template <class Data_T>
const float* 
TemporalField<Data_T>::sampleTimes(int i, int j, int k) const
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
  typename Block::State state = block.state();
  if (state == Block::ActiveState) {
    return block.sampleTimes(vi, vj, vk, m_blockOrder);
  } else {
    return 0;
  } 
}

//----------------------------------------------------------------------------//

template <class Data_T>
Data_T* 
TemporalField<Data_T>::sampleValues(int i, int j, int k) 
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
  Block &block = m_blocks[id];
  // Check if block data is allocated
  typename Block::State state = block.state();
  if (state == Block::ActiveState) {
    return block.sampleValues(vi, vj, vk, m_blockOrder);
  } else {
    return 0;
  } 
}

//----------------------------------------------------------------------------//

template <class Data_T>
const Data_T* 
TemporalField<Data_T>::sampleValues(int i, int j, int k) const
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
  typename Block::State state = block.state();
  if (state == Block::ActiveState) {
    return block.sampleValues(vi, vj, vk, m_blockOrder);
  } else {
    return 0;
  } 
}

//----------------------------------------------------------------------------//

template <class Data_T>
void 
TemporalField<Data_T>::getNearestSamples(int i, int j, int k,
                                         const float t, 
                                         bool &hasValue0, bool &hasValue1,
                                         float &t0, float &t1, 
                                         Data_T &v0, Data_T &v1) const
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
  typename Block::State state = block.state();
  if (state == Block::ActiveState) {  
    // Pass on the request
    block.getNearestSamples(vi, vj, vk, m_blockOrder, t, hasValue0, hasValue1, 
                            t0, t1, v0, v1);
  } else {
    hasValue0 = false;
    hasValue1 = false;
  }
}

//----------------------------------------------------------------------------//

template <class Data_T>
float 
TemporalField<Data_T>::averageNumSamples() const
{
  V3i size = base::m_dataWindow.size() + V3i(1);
  size_t numVoxels = size.x * size.y * size.z;
  size_t numSamples = 0;
  for (size_t b = 0; b < m_numBlocks; ++b) {
    numSamples += m_blocks[b].numSamples(m_blockOrder);
  }
  return static_cast<float>(numSamples) / static_cast<float>(numVoxels);
}

//----------------------------------------------------------------------------//

template <class Data_T>
void TemporalField<Data_T>::setBlockOrder(int order)
{
  m_blockOrder = order;
  setupBlocks();
}

//----------------------------------------------------------------------------//

template <class Data_T>
int TemporalField<Data_T>::blockOrder() const
{
  return m_blockOrder;
}

//----------------------------------------------------------------------------//

template <typename Data_T>
Data_T TemporalField<Data_T>::value(int i, int j, int k) const
{
  return fastValue(i, j, k, 0.0);
}

//----------------------------------------------------------------------------//

template <typename Data_T>
long long int TemporalField<Data_T>::memSize() const
{ 
  long long int mem = 0;
  // Contained data
  for (size_t i = 0; i < m_numBlocks; ++i) {
    if (m_blocks[i].state() == Block::ActiveState) {
      mem += m_blocks[i].memSize(m_blockOrder);
    }
  }
  // Block array
  mem += m_numBlocks * sizeof(Block);
  // Final size, including *this
  return mem + sizeof(*this);
}

//----------------------------------------------------------------------------//

template <typename Data_T>
Data_T 
TemporalField<Data_T>::fastValue(int i, int j, int k) const
{
  return fastValue(i, j, k, 0.0f);
}

//----------------------------------------------------------------------------//

template <typename Data_T>
Data_T 
TemporalField<Data_T>::fastValue(int i, int j, int k, float t) const
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
  typename Block::State state = block.state();
  if (state == Block::ActiveState) {
    return block.interp(vi, vj, vk, t, m_blockOrder);
  } else if (state == Block::EmptyState) {
    return block.emptyValue();
  } else {
    return static_cast<Data_T>(0.0);
  }
}

//----------------------------------------------------------------------------//

template <typename Data_T>
FieldBase::Ptr TemporalField<Data_T>::clone() const
{ 
  return Ptr(new TemporalField(*this)); 
}

//----------------------------------------------------------------------------//

template <typename Data_T>
void TemporalField<Data_T>::sizeChanged()
{
  base::m_mapping->setExtents(base::m_extents);
  setupBlocks();
}

//----------------------------------------------------------------------------//

template <typename Data_T>
void TemporalField<Data_T>::setupBlocks()
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
  // Clear existing data
  clear();
  // Allocate new blocks
  m_numBlocks = m_blockRes.x * m_blockRes.y * m_blockRes.z;
  m_blocks = new Block[m_numBlocks];
}

//----------------------------------------------------------------------------//

template <typename Data_T>
void TemporalField<Data_T>::copyTemporalField(const TemporalField &other)
{
  clear();

  m_blockOrder = other.m_blockOrder;
  m_blockRes = other.m_blockRes;
  m_blockXYSize = other.m_blockXYSize;

  m_numBlocks = other.m_numBlocks;
  m_blocks = new Block[m_numBlocks];

  for (size_t i = 0; i < m_numBlocks; ++i) {
    m_blocks[i].copy(other.m_blocks[i], m_blockOrder);
  }
}

//----------------------------------------------------------------------------//

template <class Data_T>
int TemporalField<Data_T>::blockSize() const
{
  return 1 << m_blockOrder;
}

//----------------------------------------------------------------------------//

template <class Data_T>
int TemporalField<Data_T>::blockId(int blockI, int blockJ, int blockK) const
{
  return blockK * m_blockXYSize + blockJ * m_blockRes.x + blockI;
}

//----------------------------------------------------------------------------//

//! \note Bit shift should be ok, indices are always positive.
template <class Data_T>
void TemporalField<Data_T>::getBlockCoord(int i, int j, int k,
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
void TemporalField<Data_T>::getVoxelInBlock(int i, int j, int k,
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
void TemporalField<Data_T>::setSize(const V3i &size)
{
  Field<Data_T>::m_extents.min = V3i(0);
  Field<Data_T>::m_extents.max = size - V3i(1);
  Field<Data_T>::m_dataWindow = Field<Data_T>::m_extents;
  sizeChanged();
}

//----------------------------------------------------------------------------//

template <class Data_T>
void TemporalField<Data_T>::setSize(const Box3i &extents)
{ 
  Field<Data_T>::m_extents = extents;
  Field<Data_T>::m_dataWindow = extents;
  sizeChanged();
}

//----------------------------------------------------------------------------//

template <class Data_T>
void TemporalField<Data_T>::setSize(const Box3i &extents, 
                                     const Box3i &dataWindow)
{ 
    
  Field<Data_T>::m_extents = extents;
  Field<Data_T>::m_dataWindow = dataWindow;
  sizeChanged();
}

//----------------------------------------------------------------------------//

template <class Data_T>
void TemporalField<Data_T>::matchDefinition(FieldRes::Ptr fieldToMatch)
{
  setSize(fieldToMatch->extents(), fieldToMatch->dataWindow());
  FieldRes::setMapping(fieldToMatch->mapping());
}

//----------------------------------------------------------------------------//

template <class Data_T>
size_t TemporalField<Data_T>::numGrains() const
{
  return m_numBlocks;
}

//----------------------------------------------------------------------------//

template <class Data_T>
bool TemporalField<Data_T>::getGrainBounds
(const size_t idx, Box3i &bounds) const
{
  // Block size
  const size_t blockSide = (1 << m_blockOrder);
  // Block coordinate
  const V3i bCoord = indexToCoord(idx, m_blockRes);
  // Block bbox
  const V3i start(bCoord * blockSide + base::m_dataWindow.min);
  const V3i end  (start + Imath::V3i(blockSide - 1));
  // Bounds must be clipped against data window
  const Box3i unclipped(start, end);
  bounds = clipBounds(unclipped, base::m_dataWindow);
  // Whether it's a contiguous block
  return bounds == unclipped;
}

//----------------------------------------------------------------------------//
// Static data member instantiation
//----------------------------------------------------------------------------//

FIELD3D_CLASSTYPE_TEMPL_INSTANTIATION(TemporalField);

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // include guard
