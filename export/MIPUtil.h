//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2013 Sony Pictures Imageworks Inc
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

/*! \file MIPUtil.h
  \brief Contains MIP-related utility functions
*/

//----------------------------------------------------------------------------//

#ifndef _INCLUDED_Field3D_MIPUtil_H_
#define _INCLUDED_Field3D_MIPUtil_H_

//----------------------------------------------------------------------------//

#include <vector>

#include <boost/thread/thread.hpp>
#include <boost/thread/condition.hpp>

#include "SparseField.h"
#include "DenseField.h"

//----------------------------------------------------------------------------//

#include "ns.h"

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Enums
//----------------------------------------------------------------------------//

#if 0
enum MIPFilterType
{
  MIPBoxFilter = 0,
  MIPGaussianFilter,
  MIPBicubicFilter
};
#endif

//----------------------------------------------------------------------------//
// Utility functions
//----------------------------------------------------------------------------//

template <class Data_T1, class Data_T2>
bool copySparseFieldToSparseField(
  typename Field3D::SparseField<Data_T1>::Ptr src,
  typename Field3D::SparseField<Data_T2>::Ptr dest);

//----------------------------------------------------------------------------//

template <class Data_T>
bool resample(typename Field3D::DenseField<Data_T>::Ptr source,
              typename Field3D::DenseField<Data_T>::Ptr target,
              Field3D::V3i reSize,
              int filterType = 2,
              int numThreads = 8);

//----------------------------------------------------------------------------//

template <class Data_T>
bool resample(typename Field3D::SparseField<Data_T>::Ptr source,
              typename Field3D::SparseField<Data_T>::Ptr target,
              Field3D::V3i reSize,
              int filterType = 2,
              int numThreads = 8);

//----------------------------------------------------------------------------//

template <class MIPField_T>
typename MIPField_T::Ptr
makeMIP(typename MIPField_T::ContainedType::Ptr base, const int minSize = 32) 
{
  typedef typename MIPField_T::value_type    Data_T;
  typedef typename MIPField_T::ContainedType Src_T;
  typedef typename Src_T::Ptr                SrcPtr;
  typedef typename MIPField_T::Ptr           MIPPtr;
  typedef std::vector<typename Src_T::Ptr>   SrcVec;

  if (base->extents() != base->dataWindow()) {
    return MIPPtr();
  }
  
  // Initialize output vector with base resolution
  SrcVec result;
  result.push_back(base);

  // Iteration variables
  V3i res = base->extents().size() + V3i(1);
  SrcPtr currentField = base;
  
  // Loop until minimum size is found
  while (res.x > minSize || res.y > minSize || res.z > minSize) {
    // Set up current iteration
    V3i nextRes = (res + V3i(1)) / V3i(2);
    // Perform filtering
    SrcPtr nextField(new Src_T);
    resample<Data_T>(currentField, nextField, nextRes, 0, 8);
    // Set mapping
    nextField->setMapping(base->mapping());
    // Add to vector of filtered fields
    result.push_back(nextField);
    // Set up for next iteration
    currentField = nextField;
    res = nextRes;
  }

  MIPPtr mipField(new MIPField_T);
  mipField->setup(result);
  mipField->name = base->name;
  mipField->attribute = base->attribute;
  mipField->copyMetadata(*base);

  return mipField;
}

//----------------------------------------------------------------------------//
// Template implementations
//----------------------------------------------------------------------------//

// Class declarations

template <class Data_T>
class ResampleDenseCPU
{
 public:

  // Typedefs ----------------------------------------------------------------//
  typedef typename Field3D::DenseField<Data_T>::Ptr dense_ptr;
  typedef Imath::V3i V3i; 
  typedef Imath::Box3i iBox;

  // Enum for filter specification -------------------------------------------//
  enum filterType{ BOX, GAUSS, LANCZOS2 };

  // Forward declaration -----------------------------------------------------//
  
  //Class for doing all the work
  class Worker;

  // Constructors ------------------------------------------------------------//
   
  // Creates an empty Resample object
  ResampleDenseCPU();

 // Creates a Resample object of chosen filter type
  ResampleDenseCPU( V3i inSize );

 // Creates a Resample object of chosen filter type
  ResampleDenseCPU( V3i inSize, int inFilterType );

 // Creates a Resample object of chosen filter type and with specified samples
  ResampleDenseCPU( V3i inSize, int inFilterType, int inThreads );

  // Creates a Resample object of chosen filter type and with specified samples
  ResampleDenseCPU( V3i inSize, 
               int inFilterType, 
               int inThreads,
               unsigned int inFilterSamples);

  // Destructor --------------------------------------------------------------//
  virtual ~ResampleDenseCPU();

  // Functions ---------------------------------------------------------------//

  // Sets the size of the resampled field
  void setSize( V3i inSize );

  //Creates a filter of specified size
  void setFilter( int inFilterType );

  // Sets the number of filter samples and resample the filter
  void setFilterSamples( int inFilterSamples );

  // Sets number of threads
  void setThreads( int inThreads );

  // Performs the filtering
  bool resample( dense_ptr volPtr, dense_ptr resPtr );


 protected:

  //For creating the work buffer
  void createWorkBuffer( dense_ptr volPtr );

  // Filter sampling functions, should not be public -------------------------//
  void sampleBoxFilter( );
  void sampleGaussFilter( );
  void sampleLanc2Filter( );  

  // Variables --------------------------------------------------------------//

  float*     m_filtPtr;        // Filter lookup table
  float      m_filtSupport;    // Support for the filter
  int        m_filtSamples;    // Number of samples for the filter lookup table
  int        m_filtType;       // Filter type. Declared as int and not filterType 
                               // because as filterType we wouldn't be able to 
                               // define it with integers (no explicit cast
                               // between int -> enum)

  V3i  m_size;                 // Size of the resampled field

  bool m_active;               // If should be running or pause 
  bool m_running;              // If is currently running 

  int m_numWorkThreads;        // Number of worker threads;

  boost::mutex m_bufMutex;     // Mutex for reading from buffer

  std::vector<iBox> m_workBufX; // Buffer for holding the work data first conv
  std::vector<iBox> m_workBufY; // Buffer for holding the work data second conv
  std::vector<iBox> m_workBufZ; // Buffer for holding the work data third conv

};

//----------------------------------------------------------------------------//
// ResampleDenseCPU::Worker
//----------------------------------------------------------------------------//
 
template <class Data_T>
class ResampleDenseCPU<Data_T>::Worker
{
 public:
  // Constructors ------------------------------------------------------------//
  Worker( 
         ResampleDenseCPU<Data_T>* inBoss, 
         int inId, 
         dense_ptr inVolPtr, 
         dense_ptr inResPtr,
         int inFiltAxis)      : m_boss( inBoss ), 
                                m_workerId( inId ), 
                                m_filtAxis( inFiltAxis ),
                                m_volPtr( inVolPtr ), 
                                m_resPtr( inResPtr )
  {
    // Empty
  }

  // Destructor --------------------------------------------------------------//
  virtual ~Worker() 
  {
//    m_volPtr = dense_ptr();
//    m_resPtr = dense_ptr();
  }

  //--------------------------------------------------------------------------//
  // Main function
  void operator()()
  {
    // Check for filter dimension
    switch( m_filtAxis ) {
      case 0:
        filtX();
        break;
      case 1:
        filtY();
        break;
      case 2:
        filtZ();
        break;
      default:
        Msg::print(Msg::SevWarning, 
                   "Error::ResampleDenseCPU::Worker::operator()()::"
                   "Unknown filter axis");
        return;
    }

  }

//----------------------------------------------------------------------------//

  void filtX();

  void filtY();

  void filtZ();
 private:
  ResampleDenseCPU<Data_T>* m_boss;      // Pointer to parent boss instance
  int m_workerId;                   // Unique id for the worker

  int    m_filtAxis;                // Filter axis
//  Data_T m_support;                 // Support of the filter
//  Data_T m_supportInv;              // Inverted support value to be able to
                                    // multiply instead of dividing
  
  dense_ptr m_volPtr;               // Pointer to the incoming volume
  dense_ptr m_resPtr;               // Pointer to the outgoing result
};

//----------------------------------------------------------------------------//

template <class Data_T>
class ResampleSparseCPU
{
 public:

  // Typedefs ----------------------------------------------------------------//
  typedef typename Field3D::SparseField<Data_T>::Ptr sparse_ptr;
  typedef Imath::V3i V3i;
  typedef Imath::Box<V3i> iBox;

  // Enum for filter specification -------------------------------------------//
  enum filterType{ BOX, GAUSS, LANCZOS2 };

  // Forward declaration -----------------------------------------------------//
  
  //Class for doing all the work
  class Worker;

  // Constructors ------------------------------------------------------------//
   
  // Creates an empty Resample object
  ResampleSparseCPU();

 // Creates a Resample object of chosen filter type
  ResampleSparseCPU( V3i inSize );

 // Creates a Resample object of chosen filter type
  ResampleSparseCPU( V3i inSize, int inFilterType );

 // Creates a Resample object of chosen filter type and with specified samples
  ResampleSparseCPU( V3i inSize, int inFilterType, int inThreads );

  // Creates a Resample object of chosen filter type and with specified samples
  ResampleSparseCPU( V3i inSize, 
               int inFilterType, 
               int inThreads,
               unsigned int inFilterSamples);

  // Destructor --------------------------------------------------------------//
  virtual ~ResampleSparseCPU();

  // Functions ---------------------------------------------------------------//

  // Sets the size of the resampled field
  void setSize( V3i inSize );

  //Creates a filter of specified size
  void setFilter( int inFilterType );

  // Sets the number of filter samples and resample the filter
  void setFilterSamples( int inFilterSamples );

  // Sets number of threads
  void setThreads( int inThreads );

  // Performs the filtering
  bool resample( sparse_ptr volPtr, sparse_ptr resPtr );


 protected:

  //For creating the work buffer
  void createWorkBuffer( sparse_ptr volPtr, sparse_ptr resPtr );

  // Filter sampling functions, should not be public -------------------------//
  void sampleBoxFilter( );
  void sampleGaussFilter( );
  void sampleLanc2Filter( );  

  // Variables --------------------------------------------------------------//

  float*     m_filtPtr;        // Filter lookup table
  float      m_filtSupport;    // Support for the filter
  int        m_filtSamples;    // Number of samples for the filter lookup table
  int        m_filtType;       // Filter type. Declared as int and not filterType 
                               // because as filterType we wouldn't be able to 
                               // define it with integers (no explicit cast between int -> enum)

  V3i  m_size;                 // Size of the resampled field

  bool m_active;               // If should be running or pause 
  bool m_running;              // If is currently running 

  int m_numWorkThreads;        // Number of worker threads;

  boost::mutex m_bufMutex;     // Mutex for reading from buffer

  std::vector<iBox> m_workBufX; // Buffer for holding the work data first conv
  std::vector<iBox> m_workBufY; // Buffer for holding the work data second conv
  std::vector<iBox> m_workBufZ; // Buffer for holding the work data third conv

};

//----------------------------------------------------------------------------//
// ResampleSparseCPU::Worker
//----------------------------------------------------------------------------//
 
template <class Data_T>
class ResampleSparseCPU<Data_T>::Worker
{
 public:
  // Constructors ------------------------------------------------------------//
  Worker( 
         ResampleSparseCPU<Data_T>* inBoss, 
         int inId, 
         sparse_ptr inVolPtr, 
         sparse_ptr inResPtr,
         int inFiltAxis)      : m_boss( inBoss ), 
                                m_workerId( inId ),  
                                m_filtAxis( inFiltAxis ),
                                m_volPtr( inVolPtr ), 
                                m_resPtr( inResPtr )
  {
    // Empty
  }

  // Destructor --------------------------------------------------------------//
  virtual ~Worker() 
  {
//    m_volPtr = sparse_ptr();
//    m_resPtr = sparse_ptr();
  }

  //--------------------------------------------------------------------------//
  // Main function
  void operator()()
  {
    // Check for filter dimension
    switch( m_filtAxis ) {
      case 0:
        filtX();
        break;
      case 1:
        filtY();
        break;
      case 2:
        filtZ();
        break;
      default:
        Msg::print(Msg::SevWarning, "Error::ResampleSparseCPU::Worker::operator()()::"
                  "Unknown filter axis");
        return;
    }

  }

  void filtX();

  void filtY();

  void filtZ();
 private:
  ResampleSparseCPU<Data_T>* m_boss;      // Pointer to parent boss instance
  int m_workerId;                   // Unique id for the worker

  int    m_filtAxis;                // Filter axis
//  Data_T m_support;                 // Support of the filter
//  Data_T m_supportInv;              // Inverted support value to be able to
                                    // multiply instead of dividing
  
  sparse_ptr m_volPtr;               // Pointer to the incoming volume
  sparse_ptr m_resPtr;               // Pointer to the outgoing result
};

////////////////////////////////////////////////////////////////////////////////
//
// Implementations
//
////////////////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------//
// Constructors --------------------------------------------------------------//
//----------------------------------------------------------------------------//

// Creates an empty conv object
template <class Data_T>
ResampleDenseCPU<Data_T>::ResampleDenseCPU()
{
  m_numWorkThreads = 8; // FieldTools::getNumThreads(10000);

  m_size = V3i( 1, 1, 1 );

  m_filtPtr = 0;
  m_filtSamples = 1000;
  
  setFilter( LANCZOS2 );
}

//----------------------------------------------------------------------------//

template <class Data_T1, class Data_T2>
bool copySparseFieldToSparseField
(typename Field3D::SparseField<Data_T1>::Ptr src,
 typename Field3D::SparseField<Data_T2>::Ptr dest){

  using namespace Field3D;

  dest->setBlockOrder(src->blockOrder());
  dest->matchDefinition(src);
  dest->name = src->name;
  dest->attribute = src->attribute;

  dest->clear(std::numeric_limits<Data_T2>::max());
  FieldMapping::Ptr inMapping = src->mapping();
  FieldMapping::Ptr outMapping = dest->mapping();
  //V3d voxelSize = outMapping->wsVoxelSize(0, 0, 0);

  typename SparseField<Data_T1>::block_iterator srcblocki = src->blockBegin();
  typename SparseField<Data_T1>::block_iterator srcblockend = src->blockEnd();
  typename SparseField<Data_T2>::block_iterator destblocki = dest->blockBegin();

  for (; srcblocki != srcblockend; ++srcblocki, ++destblocki) {
    bool allocated = 
      src->blockIsAllocated(srcblocki.x, srcblocki.y, srcblocki.z);

    typename SparseField<Data_T1>::const_iterator srci = 
      src->cbegin(srcblocki.blockBoundingBox());
    typename SparseField<Data_T1>::const_iterator srcend = 
      src->cend(srcblocki.blockBoundingBox());
    typename SparseField<Data_T2>::iterator desti = 
      dest->begin(destblocki.blockBoundingBox());

    for (; srci != srcend; ++srci, ++desti) {
      if (allocated) {
        *desti = Data_T2(*srci);
      } else {
        dest->setBlockEmptyValue
          (destblocki.x, destblocki.y, destblocki.z,
           Data_T2(src->getBlockEmptyValue(srcblocki.x,
                                           srcblocki.y,
                                           srcblocki.z)));
      }
    }
  }

  return true;

}

//----------------------------------------------------------------------------//

 // Creates a Resample object of chosen filter type
template <class Data_T>
ResampleDenseCPU<Data_T>::ResampleDenseCPU( V3i inSize ) 
{
  m_numWorkThreads = 8; // FieldTools::getNumThreads(10000);

  if( inSize.x < 1 ) { 
    Msg::print(Msg::SevWarning, 
              "Warning::ResampleDenseCPU::ResampleDenseCPU()::x axis size < 1, setting to 1");
    inSize.x = 1;
  }
  if( inSize.y < 1 ) { 
    Msg::print(Msg::SevWarning, 
              "Warning::ResampleDenseCPU::ResampleDenseCPU()::y axis size < 1, setting to 1");
    inSize.y = 1;
  } 
  if( inSize.z < 1 ) { 
    Msg::print(Msg::SevWarning, 
              "Warning::ResampleDenseCPU::ResampleDenseCPU()::z axis size < 1, setting to 1");
     inSize.z = 1;
  }

  m_size = inSize;

  m_filtPtr = 0;
  m_filtSamples = 1000;

  setFilter( LANCZOS2 );
}
//----------------------------------------------------------------------------//

 // Creates a Resample object of chosen filter type
template <class Data_T>
ResampleDenseCPU<Data_T>::ResampleDenseCPU( V3i inSize, int inFilterType ) 
{
  m_numWorkThreads = 8; //FieldTools::getNumThreads(10000);

  if( inSize.x < 1 ) { 
    Msg::print(Msg::SevWarning,
              "Warning::ResampleDenseCPU::ResampleDenseCPU()::x axis size < 1, setting to 1");
    inSize.x = 1;
  }
  if( inSize.y < 1 ) { 
    Msg::print(Msg::SevWarning, 
              "Warning::ResampleDenseCPU::ResampleDenseCPU()::y axis size < 1, setting to 1");
    inSize.y = 1;
  } 
  if( inSize.z < 1 ) { 
    Msg::print(Msg::SevWarning, 
              "Warning::ResampleDenseCPU::ResampleDenseCPU()::z axis size < 1, setting to 1");
     inSize.z = 1;
  }

  m_size = inSize;

  m_filtPtr = 0;
  m_filtSamples = 1000;

  setFilter( inFilterType );
}
//----------------------------------------------------------------------------//

 // Creates a Resample object with specified number of threads
template <class Data_T>
ResampleDenseCPU<Data_T>::ResampleDenseCPU( 
  V3i inSize, 
  int inFilterType, 
  int inThreads )
{
  m_numWorkThreads = 8; //FieldTools::getNumThreads(10000);

  if( inSize.x < 1 ) { 
    Msg::print(Msg::SevWarning, 
              "Warning::ResampleDenseCPU::ResampleDenseCPU()::x axis size < 1, setting to 1");
    inSize.x = 1;
  }
  if( inSize.y < 1 ) { 
    Msg::print(Msg::SevWarning, 
              "Warning::ResampleDenseCPU::ResampleDenseCPU()::y axis size < 1, setting to 1");
    inSize.y = 1;
  } 
  if( inSize.z < 1 ) { 
    Msg::print(Msg::SevWarning,
              "Warning::ResampleDenseCPU::ResampleDenseCPU()::z axis size < 1, setting to 1");
     inSize.z = 1;
  }
  m_size = inSize;
  
  m_filtPtr = 0;
  m_filtSamples = 1000;

  setFilter( inFilterType );
  setThreads( inThreads );
}
//----------------------------------------------------------------------------//

 // Creates a Resample object of chosen filter type and with specified samples
template <class Data_T>
ResampleDenseCPU<Data_T>::ResampleDenseCPU( 
  V3i inSize, 
  int inFilterType,
  int inThreads,
  unsigned int inFilterSamples)
{
  if( inSize.x < 1 ) { 
    Msg::print(Msg::SevWarning,
              "Warning::ResampleDenseCPU::ResampleDenseCPU()::x axis size < 1, setting to 1");
    inSize.x = 1;
  }
  if( inSize.y < 1 ) { 
    Msg::print(Msg::SevWarning, 
              "Warning::ResampleDenseCPU::ResampleDenseCPU()::y axis size < 1, setting to 1");
    inSize.y = 1;
  } 
  if( inSize.z < 1 ) { 
    Msg::print(Msg::SevWarning, 
              "Warning::ResampleDenseCPU::ResampleDenseCPU()::z axis size < 1, setting to 1");
    inSize.z = 1;
  }
  m_size = inSize;
  
  m_filtPtr = 0;
  m_filtSamples = inFilterSamples;

  setFilter( inFilterType );
  setThreads( inThreads );
}

//----------------------------------------------------------------------------//
// Destructor ----------------------------------------------------------------//
//----------------------------------------------------------------------------//
template <class Data_T>
ResampleDenseCPU<Data_T>::~ResampleDenseCPU()
{
  if( m_filtPtr ) {
    delete[] m_filtPtr;
  }
}
//----------------------------------------------------------------------------//
// Functions -----------------------------------------------------------------//
//----------------------------------------------------------------------------//


// Sets the number of filter samples and resample the filter
template <class Data_T>
void ResampleDenseCPU<Data_T>::setFilterSamples( int inFilterSamples )
{
  if( inFilterSamples == m_filtSamples ) {
    return;
  }
  
  m_filtSamples = inFilterSamples;

  // Need to resample filter
  setFilter( m_filtType );
}
//----------------------------------------------------------------------------//

// Sets and samples a filter
template <class Data_T>
void ResampleDenseCPU<Data_T>::setFilter( int inFilterType )
{

  // The sample functions both allocate the filter memory and set filter type
  switch( inFilterType ) {
    case BOX:  
      sampleBoxFilter( );
      break;
    case GAUSS:  
      sampleGaussFilter( );
      break;
    case LANCZOS2:
    default:
      sampleLanc2Filter( );
      break;
  }
}
//----------------------------------------------------------------------------//

// Sets number of threads
template <class Data_T>
void ResampleDenseCPU<Data_T>::setThreads( int inThreads )
{
  if( inThreads < 1 ) {
    Msg::print(Msg::SevWarning, "Warning::ResampleDenseCPU::setThreads()::Threads < 1,"
              " setting active threads to " +
              //str(FieldTools::getNumThreads(10000) ) );
               boost::lexical_cast<std::string>(8));
    m_numWorkThreads = 8; // FieldTools::getNumThreads(10000);
  }
  else {
    m_numWorkThreads = inThreads;
  }
}
//----------------------------------------------------------------------------//
// Perform resampling
template <class Data_T>
bool ResampleDenseCPU<Data_T>::resample( dense_ptr volPtr, dense_ptr resPtr )
{

  // Resulting densefield pointer pointing to the incoming volume 
  // will give undesired result
  if ( volPtr.get() == resPtr.get() ) {
    Msg::print(Msg::SevWarning, "Error::ResampleDenseCPU::resample()::"
              "Outgoing field same as incoming"); // Change these to exceptions?
    return false;
  }
  
  // Create work buffer
  createWorkBuffer( volPtr );

  // Resolutions of volumes
  V3i volResolution = volPtr->mapping()->resolution();
  V3i resResolution = resPtr->mapping()->resolution();

  // if result volume isn't right size, resize it
  if ( ( resResolution.x != m_size.x ) ||
       ( resResolution.y != m_size.y ) ||
       ( resResolution.z != m_size.z )  ) {
//     Msg::print(Msg::SevWarning, 
//               "Warning::ResampleDenseCPU::resample()::Resulting field is not of same"
//               " size as given parameters, resizing");

    resPtr->setMapping(volPtr->mapping());
    resPtr->setSize( m_size );
    resResolution = resPtr->mapping()->resolution();
  }


  // If we need to resize the x axis we need a temporary field for that axis
  dense_ptr tmpField1;
  if( volResolution.x != m_size.x ) {

    V3i tmpSize1( m_size.x, 
                  static_cast<int>( volPtr->mapping()->resolution().y ), 
                  static_cast<int>( volPtr->mapping()->resolution().z ) );

    tmpField1 = dense_ptr( new Field3D::DenseField<Data_T> );
    tmpField1->setMapping(volPtr->mapping());
    tmpField1->setSize( tmpSize1 );
  }
  else {
    tmpField1 = volPtr;
  }

  // If we need to ressize the y axis we need a temporary field for that axis
  dense_ptr tmpField2;
  if( volResolution.y != m_size.y ) {

    V3i tmpSize2( m_size.x, 
                  m_size.y, 
                  static_cast<int>( volPtr->mapping()->resolution().z ) );

    tmpField2 = dense_ptr( new Field3D::DenseField<Data_T> );
    tmpField2->setMapping(volPtr->mapping());
    tmpField2->setSize( tmpSize2 );
  }
  else {
    tmpField2 = tmpField1;
  }

  // Filter the rows
  if( volResolution.x != m_size.x ) {

    // Thread group for the workers
    boost::thread_group group;

    int filtAxis = 0;

    //Create workers
    for( int i = 0; i < m_numWorkThreads; ++i ) {

      group.create_thread( Worker( this, i, volPtr, tmpField1, filtAxis) );
    }

    // Make sure every thread is finished before moving on
    group.join_all();
  }

  // Filter the columns
  if( volResolution.y != m_size.y ) {

    // Thread group for the workers
    boost::thread_group group;

    int filtAxis = 1;

    //Create workers
    for( int i = 0; i < m_numWorkThreads; ++i ) {

      group.create_thread( Worker( this, i, tmpField1, tmpField2, filtAxis) );
    }

    // Make sure every thread is finished before moving on
    group.join_all();
  }

  // Filter the depth
  if( volResolution.z != m_size.z ) {

    // Thread group for the workers
    boost::thread_group group;

    int filtAxis = 2;

    //Create workers
    for( int i = 0; i < m_numWorkThreads; ++i ) {

      group.create_thread( Worker( this, i, tmpField2, resPtr, filtAxis) );
    }

    // Make sure every thread is finished before moving on
    group.join_all();
  }
  else {
    
    // Just copy values
    resPtr->copyFrom(tmpField2);

  }

  // If we've come this far everything went alright.
  return true;
}
//----------------------------------------------------------------------------//

// Sample a Lanczos2 filter
template <class Data_T>
void ResampleDenseCPU<Data_T>::sampleLanc2Filter( )
{
  const float PI = 3.14159265;

  if( m_filtPtr ) {
    delete[] m_filtPtr;
  }
  // Set filter parameters
  m_filtType = LANCZOS2;
  m_filtPtr = new float[m_filtSamples];
  m_filtSupport = static_cast<float>( 2.0 );

  // t = 0 => Lanczos2 = 1
  m_filtPtr[0] = 1;

  // Step between samples
  float stepSize = m_filtSupport / static_cast<float>( m_filtSamples );
  float t = 0;
  float t1 = 0;
  float t2 = 0;

  // Sample the filter - According to Katana
/*   Data_T* p_m_filtPtr = m_filtPtr; */
/*   for(int i = 1; i <  m_filtSamples; ++i ) { */
/*     t += stepSize; */

/*     t1  = t * PI; */
/*     t2 = ( t / m_filtSupport ) * PI; */

/*     *++p_m_filtPtr = (std::sin( t1 ) / t1 ) /  */
/*                      (std::sin( t2 ) / t2 ); */
/*   } */

  // Sample the filter - Expanded function
  float PI_SQUARE = PI * PI;
  float* p_m_filtPtr = m_filtPtr;
  for(int i = 1; i <  m_filtSamples; ++i ) {
    t += stepSize;

    t1  = t * PI;
    t2 = ( PI / 2 ) * t;

    *++p_m_filtPtr = (m_filtSupport * std::sin( t1 ) * std::sin( t2 ) ) / 
                     ( PI_SQUARE * t * t );
  }
  
}
//----------------------------------------------------------------------------//

// Sample a Gauss filter, we use a gauss filter with sigma = 1/root2
template <class Data_T>
void ResampleDenseCPU<Data_T>::sampleGaussFilter( )
{
  if( m_filtPtr ) {
    delete[] m_filtPtr;
  }

  // Set filter parameters
  m_filtType = GAUSS;
  m_filtPtr = new float[m_filtSamples];
  m_filtSupport = static_cast<float>( 3.5 ) / std::sqrt( 2.0 );

  // Right now we only use the Gauss function sigma = 1/root2 but I will leave the
  // variables here if you want to change it. Will make it easier
  float sigma = 1.0 / std::sqrt( static_cast<float>( 2.0 ) );
  float two_sigma_squared = static_cast<float>( 2.0 ) * sigma * sigma;

  // Step between samples
  float stepSize = m_filtSupport / static_cast<float>( m_filtSamples );
  float t = 0;

//  float weight;
 // Sample the filter
  float* p_m_filtPtr = m_filtPtr;  
  for(int i = 0; i <  m_filtSamples; ++i ) {
    
    *p_m_filtPtr++ = exp( -( t * t ) / two_sigma_squared );
    t += stepSize;
  }
}
//----------------------------------------------------------------------------//

// Sample a box filter
template <class Data_T>
void ResampleDenseCPU<Data_T>::sampleBoxFilter( )
{
  if( m_filtPtr ) {
    delete[] m_filtPtr;
  }

  // Set filter parameters
  m_filtType = BOX;
  m_filtPtr = new float[m_filtSamples];
  m_filtSupport = static_cast<float>( 0.5001 );

 // Sample the filter
  float* p_m_filtPtr = m_filtPtr;  
  for(int i = 0; i <  m_filtSamples; ++i ) {

    *p_m_filtPtr++ = static_cast<float>( 1.0 );
  }
}
//----------------------------------------------------------------------------//

//For creating the work buffer
template <class Data_T>
void ResampleDenseCPU<Data_T>::createWorkBuffer( dense_ptr volPtr )
{
  // For x we have to filter every orignial y and z slice
  V3i volSize = volPtr->mapping()->resolution();

  for( int i = 0; i < volSize.y; ++i ) {
    for( int j = 0; j < volSize.z; ++j ) {
      m_workBufX.push_back( iBox( V3i( 0,  i, j ), 
                                  V3i( ( m_size.x - 1 ), i , j ) ) );
    }
  }
  
  // For y we only have to filter original z slice
  for( int i = 0; i < m_size.y; ++i ) {
    for( int j = 0; j < volSize.z; ++j ) {
      m_workBufY.push_back( iBox( V3i( 0,  i, j ), 
                                  V3i(( m_size.x- 1 ), i , j ) ) );
    }
  }

  // For z we use new sizes
  for( int i = 0; i < m_size.y; ++i ) {
    for( int j = 0; j < m_size.z; ++j ) {
      m_workBufZ.push_back( iBox( V3i( 0,  i, j ), 
                                  V3i(( m_size.x- 1 ), i , j ) ) );
    }
  }
}

//----------------------------------------------------------------------------//

// Filter in the x-axis
template <class Data_T>
inline void ResampleDenseCPU<Data_T>::Worker::filtX()
{

  // Get total size of the volume
  V3i volSize = m_volPtr->mapping()->resolution();
  V3i resSize = m_resPtr->mapping()->resolution();

  // Calculate variables needed, x is current sampling position
  float xInc      = static_cast<float>( volSize.x ) / 
    static_cast<float>( resSize.x );    

  float rSupport    = m_boss->m_filtSupport;

  // The filter support should be different depending on if we're oversampling or
  // undersampling. For oversampling keep support as it is (we always want 
  // field samples that the filter support), if we're undersampling we need to
  // take the scaling in regards so we cover the support of the new volume
  if( resSize.x < volSize.x ) {
    rSupport *= xInc;
  }
   
  // It's faster to multiply than divide
  float rSupportInv = static_cast<float>( 1.0 ) / rSupport; 


  //Start execution
  while( true ) {
    // Declaration for the current work segment
    iBox workSeg;

    // Lock work buffer while reading from it
    {
      boost::mutex::scoped_lock lock( m_boss->m_bufMutex );
	    
      // Get work from work buffer 1 if it is not empty
      if ( !m_boss->m_workBufX.empty() ) {
        // Fetch work
        workSeg = m_boss->m_workBufX.back();
          
        // Remove from work buffer
        m_boss->m_workBufX.pop_back();
      }
      // Or buffers empty, exit
      else {
        return;
      }
    }
    

    // Get iterator for the result volume
    typename Field3D::DenseField<Data_T>::iterator it =
      m_resPtr->begin( workSeg );

    // Get end iterator for the result volume
    typename Field3D::DenseField<Data_T>::iterator end =
      m_resPtr->end( workSeg );

    // Loop through and filter voxels in x axis, x is current sampel position
    float x = xInc / static_cast<float>( 2.0 );
    for( ; it != end; ++it ) {

      Data_T sum = static_cast<Data_T>(0);
      float sumWeights = 0;

      // Loop through contributing voxels in negative x direction
      float t = std::floor( x - 0.49999) + 0.5;

      while( ( t > ( x - rSupport ) ) && ( t > 0 ) ) {

        float weight =
          m_boss->m_filtPtr[(int)( ( x - t ) * rSupportInv * m_boss->m_filtSamples )];

        sum += Data_T(weight) * 
          m_volPtr->fastValue( static_cast<int>(std::floor( t )),
                               it.y, it.z );
        sumWeights += weight;
        --t;
      }
      // Loop through contributing voxels in positive x direction
      t = std::floor( x + 0.49999) + 0.5;
      while( ( t < ( x + rSupport ) ) && ( t < volSize.x ) ) {

        float weight =
          m_boss->m_filtPtr[(int)( ( t - x ) * rSupportInv * m_boss->m_filtSamples )];

        sum += Data_T(weight) * 
          m_volPtr->fastValue( static_cast<int>(std::floor( t )),
                               it.y, it.z );
        sumWeights += weight;
        ++t;
      }

      // store value
      if (sumWeights == 0.0)
        *it = static_cast<Data_T>(0);
      else *it = sum / sumWeights;
      x += xInc;
    }
  }
}

//----------------------------------------------------------------------------//

// Filter in the y-axis
// filtY and filtZ work a little different than the filtX in the way that the 
// workbuffer contains scanlines so don't increase the sampling position for y
// and z while we're working on the same scanline. We only increase x
template <class Data_T>
inline void ResampleDenseCPU<Data_T>::Worker::filtY()
{

  // Get total size of the volume
  V3i volSize = m_volPtr->mapping()->resolution();
  V3i resSize = m_resPtr->mapping()->resolution();

  // Calculate variables needed, y is current sampling position
  float yInc      = static_cast<float>( volSize.y ) / 
    static_cast<float>( resSize.y );
  float yStart = yInc / static_cast<float>( 2.0 );
   
  float rSupport    = m_boss->m_filtSupport;

  // The filter support should be different depending on if we're oversampling or
  // undersampling. For oversampling keep support as it is (we always want 
  // field samples that the filter support), if we're undersampling we need to
  // take the scaling in regards so we cover the support of the new volume
  if( resSize.y < volSize.y ) {
    rSupport *= yInc;
  }
   
  // It's faster to multiply than divide
  float rSupportInv = static_cast<float>( 1.0 ) / rSupport; 

  //Start execution
  while( true ) {
    // Declaration for the current work segment
    iBox workSeg;

    // Lock work buffer while reading from it
    {
      boost::mutex::scoped_lock lock( m_boss->m_bufMutex );
	    
      // Get work from work buffer 1 if it is not empty
      if ( !m_boss->m_workBufY.empty() ) {
        // Fetch work
        workSeg = m_boss->m_workBufY.back();
          
        // Remove from work buffer
        m_boss->m_workBufY.pop_back();
      }
      // Or buffers empty, exit
      else {
        return;
      }
    }
           
    // Get iterator for the result volume
    typename Field3D::DenseField<Data_T>::iterator it =
      m_resPtr->begin( workSeg );

    // Get end iterator for the result volume
    typename Field3D::DenseField<Data_T>::iterator end =
      m_resPtr->end( workSeg );

    // Unlike the xFilter we stay at the same y value for every scanline but need
    // to calculate the y sampling position for the scanline
    int yVox = workSeg.min.y;
    float y = yStart + yVox * yInc;


    // Loop through and filter voxels in y axis, y is current sample position
    for( ; it != end; ++it ) {

      Data_T sum =  static_cast<Data_T>(0);
      float sumWeights = 0;

      // Loop through contributing voxels in negative x-axis
      float t = std::floor( y - 0.49999) + 0.5;
      while( ( t > ( y - rSupport ) ) && ( t > 0 ) ) {

        float weight =
          m_boss->m_filtPtr[(int)( ( y - t ) * rSupportInv * m_boss->m_filtSamples )];
          
        sum += Data_T(weight) * m_volPtr->fastValue(it.x, 
                                            static_cast<int>(std::floor( t )),
                                            it.z);
        sumWeights += weight;

        --t;
      }

      // Loop through contributing voxels in positive x-axis
      t = std::floor( y + 0.49999) + 0.5;
      while( ( t < ( y + rSupport ) ) && ( t < volSize.y ) ) {

        float weight =
          m_boss->m_filtPtr[(int)( ( t - y ) * rSupportInv * m_boss->m_filtSamples )];

        sum += Data_T(weight) * m_volPtr->fastValue(it.x,
                                            static_cast<int>(std::floor( t )),
                                            it.z );
        sumWeights += weight;
        ++t;
      }
       
      // store value
      if (sumWeights == 0.0)
        *it = static_cast<Data_T>(0);
      else *it = sum / sumWeights;
    }
  }
}

//----------------------------------------------------------------------------//

// Filter in the z-axis
// filtY and filtZ work a little different than the filtX in the way that the 
// workbuffer contains scanlines so don't increase the sampling position for y
// and z while we're working on the same scanline. We only increase x
template <class Data_T>
inline void ResampleDenseCPU<Data_T>::Worker::filtZ()
{

  // Get total size of the volume
  V3i volSize = m_volPtr->mapping()->resolution();
  V3i resSize = m_resPtr->mapping()->resolution();

  // Calculate variables needed, z is current sampling position
  float zInc      = static_cast<float>( volSize.z ) /
    static_cast<float>( resSize.z );
  float zStart = zInc / static_cast<float>( 2.0 );

  float rSupport    = m_boss->m_filtSupport;

  // The filter support should be different depending on if we're oversampling or
  // undersampling. For oversampling keep support as it is (we always want
  // field samples that the filter support), if we're undersampling we need to
  // take the scaling in regards so we cover the support of the new volume
  if( resSize.z < volSize.z ) {
    rSupport *= zInc;
  }
   
  // It's faster to multiply than divide
  float rSupportInv = static_cast<float>( 1.0 ) / rSupport;

  //Start execution
  while( true ) {
    // Declaration for the current work segment
    iBox workSeg;

    // Lock work buffer while reading from it
    {
      boost::mutex::scoped_lock lock( m_boss->m_bufMutex );
	    
      // Get work from work buffer 1 if it is not empty
      if ( !m_boss->m_workBufZ.empty() ) {
        // Fetch work
        workSeg = m_boss->m_workBufZ.back();
          
        // Remove from work buffer
        m_boss->m_workBufZ.pop_back();
      }
      // Or buffers empty, exit
      else {
        return;
      }
    }
    

    // Get iterator for the result volume
    typename Field3D::DenseField<Data_T>::iterator it =
      m_resPtr->begin( workSeg );

    // Get end iterator for the result volume
    typename Field3D::DenseField<Data_T>::iterator end =
      m_resPtr->end( workSeg );

    // Unlike the xFilter we stay at the same y value for every scanline but need
    // to calculate the y sampling position for the scanline
    int zVox = workSeg.min.z;
    float z = zStart + zVox * zInc;

    for( ; it != end; ++it ) {

      Data_T sum = static_cast<Data_T>(0);
      float sumWeights = 0;


      // Loop through contributing voxels in negative x-axis
      float t = std::floor( z - 0.49999) + 0.5;
      while( ( t > ( z - rSupport ) ) && ( t > 0 ) ) {

        float weight =
          m_boss->m_filtPtr[(int)( ( z - t ) * rSupportInv * m_boss->m_filtSamples )];

        sum += Data_T(weight) * m_volPtr->fastValue(it.x, it.y, 
                                            static_cast<int>(std::floor( t )) );
        sumWeights += weight;

        --t;
      }

      // Loop through contributing voxels in positive x-axis
      t = std::floor( z + 0.49999) + 0.5;
      while( ( t < ( z + rSupport ) ) && ( t < volSize.z ) ) {
 
        float weight =
          m_boss->m_filtPtr[(int)( ( t - z ) * rSupportInv * m_boss->m_filtSamples )];

        sum += Data_T(weight) * m_volPtr->fastValue(it.x, it.y, 
                                            static_cast<int>(std::floor( t )) );
        sumWeights += weight;

        ++t;
      }

       
      // store value
      if (sumWeights == 0.0)
        *it = static_cast<Data_T>(0);
      else *it = sum / sumWeights;
    }
  }
}

//----------------------------------------------------------------------------//
// Constructors --------------------------------------------------------------//
//----------------------------------------------------------------------------//

// Creates an empty conv object
template <class Data_T>
ResampleSparseCPU<Data_T>::ResampleSparseCPU()
{
  m_numWorkThreads = 8; //FieldTools::getNumThreads(10000);

  m_size = V3i( 1, 1, 1 );

  m_filtPtr = 0;
  m_filtSamples = 1000;
  
  setFilter( LANCZOS2 );
}
//----------------------------------------------------------------------------//

 // Creates a Resample object of chosen filter type
template <class Data_T>
ResampleSparseCPU<Data_T>::ResampleSparseCPU( V3i inSize ) 
{
  m_numWorkThreads = 8; //FieldTools::getNumThreads(10000);

  if( inSize.x < 1 ) { 
    Msg::print(Msg::SevWarning, 
              "Warning::ResampleSparseCPU::ResampleSparseCPU()::x axis size < 1, setting to 1");
    inSize.x = 1;
  }
  if( inSize.y < 1 ) { 
    Msg::print(Msg::SevWarning, 
              "Warning::ResampleSparseCPU::ResampleSparseCPU()::y axis size < 1, setting to 1");
    inSize.y = 1;
  } 
  if( inSize.z < 1 ) { 
    Msg::print(Msg::SevWarning, 
              "Warning::ResampleSparseCPU::ResampleSparseCPU()::z axis size < 1, setting to 1");
     inSize.z = 1;
  }

  m_size = inSize;

  m_filtPtr = 0;
  m_filtSamples = 1000;

  setFilter( LANCZOS2 );
}
//----------------------------------------------------------------------------//

 // Creates a Resample object of chosen filter type
template <class Data_T>
ResampleSparseCPU<Data_T>::ResampleSparseCPU( V3i inSize, int inFilterType ) 
{
  m_numWorkThreads = 8; // FieldTools::getNumThreads(10000);

  if( inSize.x < 1 ) { 
    Msg::print(Msg::SevWarning,
              "Warning::ResampleSparseCPU::ResampleSparseCPU()::x axis size < 1, setting to 1");
    inSize.x = 1;
  }
  if( inSize.y < 1 ) { 
    Msg::print(Msg::SevWarning, 
              "Warning::ResampleSparseCPU::ResampleSparseCPU()::y axis size < 1, setting to 1");
    inSize.y = 1;
  } 
  if( inSize.z < 1 ) { 
    Msg::print(Msg::SevWarning, 
              "Warning::ResampleSparseCPU::ResampleSparseCPU()::z axis size < 1, setting to 1");
     inSize.z = 1;
  }

  m_size = inSize;

  m_filtPtr = 0;
  m_filtSamples = 1000;

  setFilter( inFilterType );
}
//----------------------------------------------------------------------------//

 // Creates a Resample object with specified number of threads
template <class Data_T>
ResampleSparseCPU<Data_T>::ResampleSparseCPU( 
  V3i inSize, 
  int inFilterType, 
  int inThreads )
{
  m_numWorkThreads = 8; // FieldTools::getNumThreads(10000);

  if( inSize.x < 1 ) { 
    Msg::print(Msg::SevWarning, 
              "Warning::ResampleSparseCPU::ResampleSparseCPU()::x axis size < 1, setting to 1");
    inSize.x = 1;
  }
  if( inSize.y < 1 ) { 
    Msg::print(Msg::SevWarning, 
              "Warning::ResampleSparseCPU::ResampleSparseCPU()::y axis size < 1, setting to 1");
    inSize.y = 1;
  } 
  if( inSize.z < 1 ) { 
    Msg::print(Msg::SevWarning,
              "Warning::ResampleSparseCPU::ResampleSparseCPU()::z axis size < 1, setting to 1");
     inSize.z = 1;
  }
  m_size = inSize;
  
  m_filtPtr = 0;
  m_filtSamples = 1000;

  setFilter( inFilterType );
  setThreads( inThreads );
}
//----------------------------------------------------------------------------//

 // Creates a Resample object of chosen filter type and with specified samples
template <class Data_T>
ResampleSparseCPU<Data_T>::ResampleSparseCPU( 
  V3i inSize, 
  int inFilterType,
  int inThreads,
  unsigned int inFilterSamples)
{
  if( inSize.x < 1 ) { 
    Msg::print(Msg::SevWarning,
              "Warning::ResampleSparseCPU::ResampleSparseCPU()::x axis size < 1, setting to 1");
    inSize.x = 1;
  }
  if( inSize.y < 1 ) { 
    Msg::print(Msg::SevWarning, 
              "Warning::ResampleSparseCPU::ResampleSparseCPU()::y axis size < 1, setting to 1");
    inSize.y = 1;
  } 
  if( inSize.z < 1 ) { 
    Msg::print(Msg::SevWarning, 
              "Warning::ResampleSparseCPU::ResampleSparseCPU()::z axis size < 1, setting to 1");
    inSize.z = 1;
  }
  m_size = inSize;
  
  m_filtPtr = 0;
  m_filtSamples = inFilterSamples;

  setFilter( inFilterType );
  setThreads( inThreads );
}

//----------------------------------------------------------------------------//
// Destructor ----------------------------------------------------------------//
//----------------------------------------------------------------------------//
template <class Data_T>
ResampleSparseCPU<Data_T>::~ResampleSparseCPU()
{
  if( m_filtPtr ) {
    delete[] m_filtPtr;
  }
}
//----------------------------------------------------------------------------//
// Functions -----------------------------------------------------------------//
//----------------------------------------------------------------------------//


// Sets the number of filter samples and resample the filter
template <class Data_T>
void ResampleSparseCPU<Data_T>::setFilterSamples( int inFilterSamples )
{
  if( inFilterSamples == m_filtSamples ) {
    return;
  }
  
  m_filtSamples = inFilterSamples;

  // Need to resample filter
  setFilter( m_filtType );
}
//----------------------------------------------------------------------------//

// Sets and samples a filter
template <class Data_T>
void ResampleSparseCPU<Data_T>::setFilter( int inFilterType )
{

  // The sample functions both allocate the filter memory and set filter type
  switch( inFilterType ) {
    case GAUSS:  
      sampleGaussFilter( );
      break;
    case BOX:  
      sampleBoxFilter( );
      break;
    case LANCZOS2:
    default:
      sampleLanc2Filter( );
      break;
  }
}
//----------------------------------------------------------------------------//

// Sets number of threads
template <class Data_T>
void ResampleSparseCPU<Data_T>::setThreads( int inThreads )
{
  if( inThreads < 1 ) {
    Msg::print(Msg::SevWarning, "Warning::ResampleSparseCPU::setThreads()::Threads < 1,"
              " setting active threads to " +
              // str(FieldTools::getNumThreads(10000) ) );
               boost::lexical_cast<std::string>(8));
    m_numWorkThreads = 8; //FieldTools::getNumThreads(10000);
  }
  else {
    m_numWorkThreads = inThreads;
  }
}
//----------------------------------------------------------------------------//
// Perform resampling
template <class Data_T>
bool ResampleSparseCPU<Data_T>::resample( sparse_ptr volPtr, sparse_ptr resPtr )
{

  // Resulting sparsefield pointer pointing to the incoming volume 
  // will give undesired result
  if ( volPtr.get() == resPtr.get() ) {
    Msg::print(Msg::SevWarning, "Error::ResampleSparseCPU::resample()::"
              "Outgoing field same as incoming"); // Change these to exceptions?
    return false;
  }
  
  // Create work buffer
  createWorkBuffer( volPtr, resPtr );

  // Resolutions of volumes
  V3i volResolution = volPtr->mapping()->resolution();
  V3i resResolution = resPtr->mapping()->resolution();

  // if result volume isn't right size, resize it
  if ( ( resResolution.x != m_size.x ) ||
       ( resResolution.y != m_size.y ) ||
       ( resResolution.z != m_size.z )  ) {
//     Msg::print(Msg::SevWarning, 
//               "Warning::ResampleSparseCPU::resample()::Resulting field is not of same"
//               " size as given parameters, resizing");

    resPtr->setMapping(volPtr->mapping());
    resPtr->setSize( m_size );
    resResolution = resPtr->mapping()->resolution();
  }


  // If we need to resize the x axis we need a temporary field for that axis
  sparse_ptr tmpField1;
  if( volResolution.x != m_size.x ) {

    V3i tmpSize1( m_size.x, 
                  static_cast<int>( volResolution.y ), 
                  static_cast<int>( volResolution.z ) );

    tmpField1 = sparse_ptr( new Field3D::SparseField<Data_T> );
    tmpField1->setBlockOrder(resPtr->blockOrder());
    tmpField1->setMapping(volPtr->mapping());
    tmpField1->setSize( tmpSize1 );
    tmpField1->name = volPtr->name;
    tmpField1->attribute = volPtr->attribute;
    tmpField1->copyMetadata(*volPtr);
  }
  else {
    tmpField1 = volPtr;
  }

  // If we need to resize the y axis we need a temporary field for that axis
  sparse_ptr tmpField2;
  if( volResolution.y != m_size.y ) {
  
    V3i tmpSize2( m_size.x, 
                  m_size.y, 
                  static_cast<int>( volResolution.z ) );

    tmpField2 = sparse_ptr( new Field3D::SparseField<Data_T> );
    tmpField2->setBlockOrder(resPtr->blockOrder());
    tmpField2->setMapping(volPtr->mapping());
    tmpField2->setSize( tmpSize2 );
    tmpField2->name = volPtr->name;
    tmpField2->attribute = volPtr->attribute;
    tmpField2->copyMetadata(*volPtr);
  }
  else {
    tmpField2 = tmpField1;
  }

  // Filter the rows
  if( volResolution.x != m_size.x ) {
    
    // Thread group for the workers
    boost::thread_group group;

    int filtAxis = 0;

    //Create workers
    for( int i = 0; i < m_numWorkThreads; ++i ) {
      group.create_thread( Worker( this, i, volPtr, tmpField1, filtAxis) );
    }

    // Make sure every thread is finished before moving on
    group.join_all();
  }

  // Filter the columns
  if( volResolution.y != m_size.y ) {
    
    // Thread group for the workers
    boost::thread_group group;

    int filtAxis = 1;

    //Create workers
    for( int i = 0; i < m_numWorkThreads; ++i ) {
      group.create_thread( Worker( this, i, tmpField1, tmpField2, filtAxis) );
    }

    // Make sure every thread is finished before moving on
    group.join_all();
  }

  // Filter the depth
  if( volResolution.z != m_size.z ) {
    
    // Thread group for the workers
    boost::thread_group group;

    int filtAxis = 2;

    //Create workers
    for( int i = 0; i < m_numWorkThreads; ++i ) {
      group.create_thread( Worker( this, i, tmpField2, resPtr, filtAxis) );
    }

    // Make sure every thread is finished before moving on
    group.join_all();
  }
  else {
    
    // Just copy values
    if (!copySparseFieldToSparseField<Data_T, Data_T>
        (tmpField2, resPtr))
      return false;
  }
  
  using namespace Field3D;
  // loop over unallocated blocks in the new field and set their empty
  // values to the corresponding empty values in the input field
  typename Field3D::SparseField<Data_T>::block_iterator rblocki = resPtr->blockBegin();
  typename Field3D::SparseField<Data_T>::block_iterator rblockend = resPtr->blockEnd();
  Field3D::V3d voxelRatio =
    Field3D::V3d(volResolution) / Field3D::V3d(resResolution);
  for (; rblocki != rblockend; ++rblocki) {
    if (!resPtr->blockIsAllocated(rblocki.x, rblocki.y, rblocki.z)) {
      Field3D::Box3i blockBox = rblocki.blockBoundingBox();
      Field3D::V3i middle =
        Field3D::V3i(Field3D::V3d(blockBox.max+blockBox.min)*0.5);
      Field3D::V3i srcVoxel = Field3D::V3i(Field3D::V3d(middle) * voxelRatio);
      resPtr->setBlockEmptyValue(
        rblocki.x, rblocki.y, rblocki.z,
        volPtr->fastValue(srcVoxel.x, srcVoxel.y, srcVoxel.z));
    }
  }

  // If we've come this far everything went alright.
  return true;
}
//----------------------------------------------------------------------------//

// Sample a Lanczos2 filter
template <class Data_T>
void ResampleSparseCPU<Data_T>::sampleLanc2Filter( )
{
  const float PI = 3.14159265;

  if( m_filtPtr ) {
    delete[] m_filtPtr;
  }
  // Set filter parameters
  m_filtType = LANCZOS2;
  m_filtPtr = new float[m_filtSamples];
  m_filtSupport = static_cast<float>( 2.0 );

  // t = 0 => Lanczos2 = 1
  m_filtPtr[0] = 1;

  // Step between samples
  float stepSize = m_filtSupport / static_cast<float>( m_filtSamples );
  float t = 0;
  float t1 = 0;
  float t2 = 0;

  // Sample the filter - According to Katana
/*   Data_T* p_m_filtPtr = m_filtPtr; */
/*   for(int i = 1; i <  m_filtSamples; ++i ) { */
/*     t += stepSize; */

/*     t1  = t * PI; */
/*     t2 = ( t / m_filtSupport ) * PI; */

/*     *++p_m_filtPtr = (std::sin( t1 ) / t1 ) /  */
/*                      (std::sin( t2 ) / t2 ); */
/*   } */

  // Sample the filter - Expanded function
  float PI_SQUARE = PI * PI;
  float* p_m_filtPtr = m_filtPtr;
  for(int i = 1; i <  m_filtSamples; ++i ) {
    t += stepSize;

    t1  = t * PI;
    t2 = ( PI / 2 ) * t;

    *++p_m_filtPtr = (m_filtSupport * std::sin( t1 ) * std::sin( t2 ) ) / 
                     ( PI_SQUARE * t * t );
  }
  
}
//----------------------------------------------------------------------------//

// Sample a Gauss filter, we use a gauss filter with sigma = 1/root2
template <class Data_T>
void ResampleSparseCPU<Data_T>::sampleGaussFilter( )
{
  if( m_filtPtr ) {
    delete[] m_filtPtr;
  }

  // Set filter parameters
  m_filtType = GAUSS;
  m_filtPtr = new float[m_filtSamples];
  m_filtSupport = static_cast<float>( 3.5 ) / std::sqrt( 2.0 );

  // Right now we only use the Gauss function sigma = 1/root2 but I will leave the
  // variables here if you want to change it. Will make it easier
  float sigma = 1.0 / std::sqrt( static_cast<float>( 2.0 ) );
  float two_sigma_squared = static_cast<float>( 2.0 ) * sigma * sigma;

  // Step between samples
  float stepSize = m_filtSupport / static_cast<float>( m_filtSamples );
  float t = 0;

//  float weight;
 // Sample the filter
  float* p_m_filtPtr = m_filtPtr;  
  for(int i = 0; i <  m_filtSamples; ++i ) {
    
    *p_m_filtPtr++ = exp( -( t * t ) / two_sigma_squared );
    t += stepSize;
  }
}
//----------------------------------------------------------------------------//

// Sample a box filter
template <class Data_T>
void ResampleSparseCPU<Data_T>::sampleBoxFilter( )
{
  if( m_filtPtr ) {
    delete[] m_filtPtr;
  }

  // Set filter parameters
  m_filtType = BOX;
  m_filtPtr = new float[m_filtSamples];
  m_filtSupport = static_cast<float>( 0.5001 );

 // Sample the filter
  float* p_m_filtPtr = m_filtPtr;  
  for(int i = 0; i <  m_filtSamples; ++i ) {

    *p_m_filtPtr++ = static_cast<float>( 1.0 );
  }
}
//----------------------------------------------------------------------------//

//For creating the work buffer
template <class Data_T>
void ResampleSparseCPU<Data_T>::createWorkBuffer( sparse_ptr volPtr,
                                                  sparse_ptr resPtr )
{
  using namespace Field3D;
  V3i volSize = volPtr->mapping()->resolution();

  // add work units for each soon-to-be-allocated resampled block

  // Note: this is a little hacky: we create temp fields with the
  // resolutions we'll be creating in the resample functions, and for
  // each allocated block in the old field, we mark all the
  // corresponding voxels in the target fields so the blocks are
  // allocated.  Then we add work units for each allocated block in
  // the temp fields.
  //
  // A more efficient way would be possible if we have access to the
  // voxel-to-block function, currently a protected function.  Then we
  // could just loop over the blocks needed without allocating blocks
  // in the temp fields.

  // work units for X filtering
  SparseField<unsigned char> xFiltAlloc;
  xFiltAlloc.setBlockOrder(resPtr->blockOrder());
  {

    xFiltAlloc.setSize(V3i(m_size.x, volSize.y, volSize.z));
    xFiltAlloc.clear(0);
    V3f scale(static_cast<float>(m_size.x) / static_cast<float>(volSize.x),
              1.0, 1.0);

    typename SparseField<Data_T>::block_iterator sblocki = volPtr->blockBegin();
    typename SparseField<Data_T>::block_iterator sblockend = volPtr->blockEnd();
    for (; sblocki != sblockend; ++sblocki) {
      if (volPtr->blockIsAllocated(sblocki.x, sblocki.y, sblocki.z)) {
        Box3i blockBox = sblocki.blockBoundingBox();
        
        V3i resStartPos = V3i(std::floor(blockBox.min.x*scale.x), 
                              std::floor(blockBox.min.y*scale.y), 
                              std::floor(blockBox.min.z*scale.z));
        V3i resEndPos = V3i(std::ceil(blockBox.max.x*scale.x), 
                            std::ceil(blockBox.max.y*scale.y), 
                            std::ceil(blockBox.max.z*scale.z));
        
        int startBlockX, startBlockY, startBlockZ, endBlockX, endBlockY, endBlockZ;
        xFiltAlloc.getBlockCoord(resStartPos.x, resStartPos.y, resStartPos.z, 
                                 startBlockX, startBlockY, startBlockZ);
        xFiltAlloc.getBlockCoord(resEndPos.x, resEndPos.y, resEndPos.z, 
                                 endBlockX, endBlockY, endBlockZ);

        V3i blockRes = xFiltAlloc.blockRes();
        if (endBlockX >= blockRes.x) endBlockX = blockRes.x - 1;
        if (endBlockY >= blockRes.y) endBlockY = blockRes.y - 1;
        if (endBlockZ >= blockRes.z) endBlockZ = blockRes.z - 1;
        
        for (int k=startBlockZ; k<=endBlockZ; ++k)
          for (int j=startBlockY; j<=endBlockY; ++j)
            for (int i=startBlockX; i<=endBlockX; ++i)
              xFiltAlloc.setBlockEmptyValue(i, j, k, 1);
      }
    }
    SparseField<unsigned char>::block_iterator ablocki = xFiltAlloc.blockBegin();
    SparseField<unsigned char>::block_iterator ablockend = xFiltAlloc.blockEnd();
    for (; ablocki != ablockend; ++ablocki) {
      if (xFiltAlloc.getBlockEmptyValue(ablocki.x, ablocki.y, ablocki.z)) {
        m_workBufX.push_back(ablocki.blockBoundingBox());
      }
    }
  }

  // work units for Y filtering
  SparseField<unsigned char> yFiltAlloc;
  yFiltAlloc.setBlockOrder(resPtr->blockOrder());
  {

    yFiltAlloc.setSize(V3i(m_size.x, m_size.y, volSize.z));
    yFiltAlloc.clear(0);
    V3f scale(1.0,
              static_cast<float>(m_size.y) / static_cast<float>(volSize.y),
              1.0);

    typename SparseField<unsigned char>::block_iterator sblocki = xFiltAlloc.blockBegin();
    typename SparseField<unsigned char>::block_iterator sblockend = xFiltAlloc.blockEnd();
    for (; sblocki != sblockend; ++sblocki) {
      if (xFiltAlloc.getBlockEmptyValue(sblocki.x, sblocki.y, sblocki.z)) {
        Box3i blockBox = sblocki.blockBoundingBox();
        
        V3i resStartPos = V3i(std::floor(blockBox.min.x*scale.x), 
                              std::floor(blockBox.min.y*scale.y), 
                              std::floor(blockBox.min.z*scale.z));
        V3i resEndPos = V3i(std::ceil(blockBox.max.x*scale.x), 
                            std::ceil(blockBox.max.y*scale.y), 
                            std::ceil(blockBox.max.z*scale.z));

        int startBlockX, startBlockY, startBlockZ, endBlockX, endBlockY, endBlockZ;
        yFiltAlloc.getBlockCoord(resStartPos.x, resStartPos.y, resStartPos.z, 
                                 startBlockX, startBlockY, startBlockZ);
        yFiltAlloc.getBlockCoord(resEndPos.x, resEndPos.y, resEndPos.z, 
                                 endBlockX, endBlockY, endBlockZ);

        V3i blockRes = yFiltAlloc.blockRes();
        if (endBlockX >= blockRes.x) endBlockX = blockRes.x - 1;
        if (endBlockY >= blockRes.y) endBlockY = blockRes.y - 1;
        if (endBlockZ >= blockRes.z) endBlockZ = blockRes.z - 1;
        
        for (int k=startBlockZ; k<=endBlockZ; ++k)
          for (int j=startBlockY; j<=endBlockY; ++j)
            for (int i=startBlockX; i<=endBlockX; ++i)
              yFiltAlloc.setBlockEmptyValue(i, j, k, 1);
      }
    }
    SparseField<unsigned char>::block_iterator ablocki = yFiltAlloc.blockBegin();
    SparseField<unsigned char>::block_iterator ablockend = yFiltAlloc.blockEnd();
    for (; ablocki != ablockend; ++ablocki) {
      if (yFiltAlloc.getBlockEmptyValue(ablocki.x, ablocki.y, ablocki.z)) {
        m_workBufY.push_back(ablocki.blockBoundingBox());
      }
    }
  }

  // work units for Z filtering
  SparseField<unsigned char> zFiltAlloc;
  zFiltAlloc.setBlockOrder(resPtr->blockOrder());
  {

    zFiltAlloc.setSize(V3i(m_size.x, m_size.y, m_size.z));
    zFiltAlloc.clear(0);
    V3f scale(1.0,
              1.0,
              static_cast<float>(m_size.z) / static_cast<float>(volSize.z));

    typename SparseField<unsigned char>::block_iterator sblocki = yFiltAlloc.blockBegin();
    typename SparseField<unsigned char>::block_iterator sblockend = yFiltAlloc.blockEnd();
    for (; sblocki != sblockend; ++sblocki) {
      if (yFiltAlloc.getBlockEmptyValue(sblocki.x, sblocki.y, sblocki.z)) {
        Box3i blockBox = sblocki.blockBoundingBox();
        
        V3i resStartPos = V3i(std::floor(blockBox.min.x*scale.x), 
                              std::floor(blockBox.min.y*scale.y), 
                              std::floor(blockBox.min.z*scale.z));
        V3i resEndPos = V3i(std::ceil(blockBox.max.x*scale.x), 
                            std::ceil(blockBox.max.y*scale.y), 
                            std::ceil(blockBox.max.z*scale.z));
        
        int startBlockX, startBlockY, startBlockZ, endBlockX, endBlockY, endBlockZ;
        zFiltAlloc.getBlockCoord(resStartPos.x, resStartPos.y, resStartPos.z, 
                                 startBlockX, startBlockY, startBlockZ);
        zFiltAlloc.getBlockCoord(resEndPos.x, resEndPos.y, resEndPos.z, 
                                 endBlockX, endBlockY, endBlockZ);

        V3i blockRes = zFiltAlloc.blockRes();
        if (endBlockX >= blockRes.x) endBlockX = blockRes.x - 1;
        if (endBlockY >= blockRes.y) endBlockY = blockRes.y - 1;
        if (endBlockZ >= blockRes.z) endBlockZ = blockRes.z - 1;
                                 
        for (int k=startBlockZ; k<=endBlockZ; ++k)
          for (int j=startBlockY; j<=endBlockY; ++j)
            for (int i=startBlockX; i<=endBlockX; ++i)
              zFiltAlloc.setBlockEmptyValue(i, j, k, 1);
      }
    }
    SparseField<unsigned char>::block_iterator ablocki = zFiltAlloc.blockBegin();
    SparseField<unsigned char>::block_iterator ablockend = zFiltAlloc.blockEnd();
    for (; ablocki != ablockend; ++ablocki) {
      if (zFiltAlloc.getBlockEmptyValue(ablocki.x, ablocki.y, ablocki.z)) {
        m_workBufZ.push_back(ablocki.blockBoundingBox());
      }
    }
  }

}

//----------------------------------------------------------------------------//

// Filter in the x-axis
template <class Data_T>
inline void ResampleSparseCPU<Data_T>::Worker::filtX()
{

  // Get total size of the volume
  V3i volSize = m_volPtr->mapping()->resolution();
  V3i resSize = m_resPtr->mapping()->resolution();

  // Calculate variables needed, x is current sampling position
  float xInc      = static_cast<float>( volSize.x ) / 
    static_cast<float>( resSize.x );    
  float xStart = xInc / static_cast<float>( 2.0 );

  float rSupport    = m_boss->m_filtSupport;

  // The filter support should be different depending on if we're
  // oversampling or undersampling. For oversampling keep support as
  // it is (we always want field samples that the filter support), if
  // we're undersampling we need to take the scaling in regards so we
  // cover the support of the new volume
  if( resSize.x < volSize.x ) {
    rSupport *= xInc;
  }
   
  // It's faster to multiply than divide
  float rSupportInv = static_cast<float>( 1.0 ) / rSupport; 


  //Start execution
  while( true ) {
    // Declaration for the current work segment
    iBox workSeg;

    // Lock work buffer while reading from it
    {
      boost::mutex::scoped_lock lock( m_boss->m_bufMutex );
	    
      // Get work from work buffer 1 if it is not empty
      if ( !m_boss->m_workBufX.empty() ) {
        // Fetch work
        workSeg = m_boss->m_workBufX.back();

        // Remove from work buffer
        m_boss->m_workBufX.pop_back();
      }
      // Or buffers empty, exit
      else {
        return;
      }
    }
    
    Imath::Box3i dataWindow = m_volPtr->dataWindow();

    // Get iterator for the result volume
    typename Field3D::SparseField<Data_T>::iterator it =
      m_resPtr->begin( workSeg );

    // Get end iterator for the result volume
    typename Field3D::SparseField<Data_T>::iterator end =
      m_resPtr->end( workSeg );

    // Loop through target filter voxels in the block, filter along x axis
    for( ; it != end; ++it ) {
      float x = xStart + it.x * xInc;

      Data_T sum = static_cast<Data_T>(0);
      float sumWeights = 0;

      // Loop through contributing voxels in negative x direction
      float t = std::floor( x - 0.49999) + 0.5;

      while( ( t > ( x - rSupport ) ) && ( t > 0 ) ) {

        float weight =
          m_boss->m_filtPtr[(int)( ( x - t ) * rSupportInv * m_boss->m_filtSamples )];

        int xInd = static_cast<int>(std::floor( t ));
        if (xInd <= dataWindow.max.x && xInd >= dataWindow.min.x) {
          sum += m_volPtr->fastValue(xInd, it.y, it.z) * weight;
          sumWeights += weight;
        }
        --t;
      }

      // Loop through contributing voxels in positive x direction
      t = std::floor( x + 0.49999) + 0.5;
      while( ( t < ( x + rSupport ) ) && ( t < volSize.x ) ) {

        float weight =
          m_boss->m_filtPtr[(int)( ( t - x ) * rSupportInv * m_boss->m_filtSamples )];

        int xInd = static_cast<int>(std::floor( t ));
        if (xInd <= dataWindow.max.x && xInd >= dataWindow.min.x) {
          sum += m_volPtr->fastValue(xInd, it.y, it.z) * weight;
          sumWeights += weight;
        }
        ++t;
      }

      // store value
      if (sumWeights == 0.0)
        *it = static_cast<Data_T>(0);
      else *it = sum / sumWeights;
    }
  }
}

//----------------------------------------------------------------------------//

// Filter in the y-axis
template <class Data_T>
inline void ResampleSparseCPU<Data_T>::Worker::filtY()
{

  // Get total size of the volume
  V3i volSize = m_volPtr->mapping()->resolution();
  V3i resSize = m_resPtr->mapping()->resolution();

  // Calculate variables needed, y is current sampling position
  float yInc      = static_cast<float>( volSize.y ) / 
    static_cast<float>( resSize.y );
  float yStart = yInc / static_cast<float>( 2.0 );
   
  float rSupport    = m_boss->m_filtSupport;

  // The filter support should be different depending on if we're
  // oversampling or undersampling. For oversampling keep support as
  // it is (we always want field samples that the filter support), if
  // we're undersampling we need to take the scaling in regards so we
  // cover the support of the new volume
  if( resSize.y < volSize.y ) {
    rSupport *= yInc;
  }
   
  // It's faster to multiply than divide
  float rSupportInv = static_cast<float>( 1.0 ) / rSupport; 

  //Start execution
  while( true ) {
    // Declaration for the current work segment
    iBox workSeg;

    // Lock work buffer while reading from it
    {
      boost::mutex::scoped_lock lock( m_boss->m_bufMutex );
	    
      // Get work from work buffer 1 if it is not empty
      if ( !m_boss->m_workBufY.empty() ) {
        // Fetch work
        workSeg = m_boss->m_workBufY.back();
          
        // Remove from work buffer
        m_boss->m_workBufY.pop_back();
      }
      // Or buffers empty, exit
      else {
        return;
      }
    }
           
    Imath::Box3i dataWindow = m_volPtr->dataWindow();

    // Get iterator for the result volume
    typename Field3D::SparseField<Data_T>::iterator it =
      m_resPtr->begin( workSeg );

    // Get end iterator for the result volume
    typename Field3D::SparseField<Data_T>::iterator end =
      m_resPtr->end( workSeg );

    // Loop through target filter voxels in the block, filter along y axis
    for( ; it != end; ++it ) {
      float y = yStart + it.y * yInc;

      Data_T sum =  static_cast<Data_T>(0);
      float sumWeights = 0;

      // Loop through contributing voxels in negative y direction
      float t = std::floor( y - 0.49999) + 0.5;
      while( ( t > ( y - rSupport ) ) && ( t > 0 ) ) {

        float weight =
          m_boss->m_filtPtr[(int)( ( y - t ) * rSupportInv * m_boss->m_filtSamples )];
          
        int yInd = static_cast<int>(std::floor( t ));
        if (yInd <= dataWindow.max.y && yInd >= dataWindow.min.y) {
          sum += m_volPtr->fastValue(it.x, yInd, it.z) * weight;
          sumWeights += weight;
        }
        --t;
      }

      // Loop through contributing voxels in positive y direction
      t = std::floor( y + 0.49999) + 0.5;
      while( ( t < ( y + rSupport ) ) && ( t < volSize.y ) ) {

        float weight =
          m_boss->m_filtPtr[(int)( ( t - y ) * rSupportInv * m_boss->m_filtSamples )];

        int yInd = static_cast<int>(std::floor( t ));
        if (yInd <= dataWindow.max.y && yInd >= dataWindow.min.y) {
          sum += m_volPtr->fastValue(it.x, yInd, it.z) * weight;
          sumWeights += weight;
        }
        ++t;
      }

      // store value
      if (sumWeights == 0.0)
        *it = static_cast<Data_T>(0);
      else *it = sum / sumWeights;
    }
  }
}

//----------------------------------------------------------------------------//

// Filter in the z-axis
template <class Data_T>
inline void ResampleSparseCPU<Data_T>::Worker::filtZ()
{

  // Get total size of the volume
  V3i volSize = m_volPtr->mapping()->resolution();
  V3i resSize = m_resPtr->mapping()->resolution();

  // Calculate variables needed, z is current sampling position
  float zInc      = static_cast<float>( volSize.z ) /
    static_cast<float>( resSize.z );
  float zStart = zInc / static_cast<float>( 2.0 );

  float rSupport    = m_boss->m_filtSupport;

  // The filter support should be different depending on if we're oversampling or
  // undersampling. For oversampling keep support as it is (we always want
  // field samples that the filter support), if we're undersampling we need to
  // take the scaling in regards so we cover the support of the new volume
  if( resSize.z < volSize.z ) {
    rSupport *= zInc;
  }
   
  // It's faster to multiply than divide
  float rSupportInv = static_cast<float>( 1.0 ) / rSupport;

  //Start execution
  while( true ) {
    // Declaration for the current work segment
    iBox workSeg;

    // Lock work buffer while reading from it
    {
      boost::mutex::scoped_lock lock( m_boss->m_bufMutex );
	    
      // Get work from work buffer 1 if it is not empty
      if ( !m_boss->m_workBufZ.empty() ) {
        // Fetch work
        workSeg = m_boss->m_workBufZ.back();
          
        // Remove from work buffer
        m_boss->m_workBufZ.pop_back();
      }
      // Or buffers empty, exit
      else {
        return;
      }
    }
    
    Imath::Box3i dataWindow = m_volPtr->dataWindow();

    // Get iterator for the result volume
    typename Field3D::SparseField<Data_T>::iterator it =
      m_resPtr->begin( workSeg );

    // Get end iterator for the result volume
    typename Field3D::SparseField<Data_T>::iterator end =
      m_resPtr->end( workSeg );

    // Loop through target filter voxels in the block, filter along z axis
    for( ; it != end; ++it ) {
      float z = zStart + it.z * zInc;

      Data_T sum = static_cast<Data_T>(0);
      float sumWeights = 0;

      // Loop through contributing voxels in negative z direction
      float t = std::floor( z - 0.49999) + 0.5;
      while( ( t > ( z - rSupport ) ) && ( t > 0 ) ) {

        float weight =
          m_boss->m_filtPtr[(int)( ( z - t ) * rSupportInv * m_boss->m_filtSamples )];

        int zInd = static_cast<int>(std::floor( t ));
        if (zInd <= dataWindow.max.z && zInd >= dataWindow.min.z) {
          sum += m_volPtr->fastValue(it.x, it.y, zInd) * weight;
          sumWeights += weight;
        }
        --t;
      }

      // Loop through contributing voxels in positive z direction
      t = std::floor( z + 0.49999) + 0.5;
      while( ( t < ( z + rSupport ) ) && ( t < volSize.z ) ) {
 
        float weight =
          m_boss->m_filtPtr[(int)( ( t - z ) * rSupportInv * m_boss->m_filtSamples )];

        int zInd = static_cast<int>(std::floor( t ));
        if (zInd <= dataWindow.max.z && zInd >= dataWindow.min.z) {
          sum += m_volPtr->fastValue(it.x, it.y, zInd) * weight;
          sumWeights += weight;
        }
        ++t;
      }
       
      // store value
      if (sumWeights == 0.0)
        *it = static_cast<Data_T>(0);
      else *it = sum / sumWeights;
    }
  }
}

//----------------------------------------------------------------------------//

template <class Data_T>
bool resample(typename DenseField<Data_T>::Ptr source,
              typename DenseField<Data_T>::Ptr target,
              V3i reSize,
              int filterType,
              int numThreads)
{
  // Compile-time error checks
  assert(source != NULL);
  assert(target != NULL);
  assert(reSize.x > 0);
  assert(reSize.y > 0);
  assert(reSize.z > 0);
  assert(numThreads >= 0);

  // Error checks
  if (source == NULL) {
    Msg::print(Msg::SevWarning, "FieldProcessing::volumeAdvect(): "
              "Got null source pointer");
    return false;
  }
  if (target == NULL) {
    Msg::print(Msg::SevWarning, "FieldProcessing::volumeAdvect(): "
              "Got null target pointer");
    return false;
  }
  if (source.get() == target.get()) {
    Msg::print(Msg::SevWarning, "FieldProcessing::volumeAdvect(): "
              "Cannot let source and target buffer be same pointer");
    return false;    
  }
  if( reSize.x < 1  ) {
    Msg::print(Msg::SevWarning, "FieldProcessing::volumeAdvect(): "
              "Resize in x axis too small");
    return false;    
  }
  if( reSize.y < 1  ) {
    Msg::print(Msg::SevWarning, "FieldProcessing::volumeAdvect(): "
              "Resize in y axis too small");
    return false;    
  }
  if( reSize.z < 1  ) {
    Msg::print(Msg::SevWarning, "FieldProcessing::volumeAdvect(): "
              "Resize in z axis too small");
    return false;    
  }
  if (numThreads <= 0) {
    Msg::print(Msg::SevWarning, "FieldProcessing::resample(): "
              "Threads < 1");
    return false;
  }

  ResampleDenseCPU<Data_T> resamp(reSize, filterType, numThreads);
  return resamp.resample(source, target);
}

//----------------------------------------------------------------------------//

template <class Data_T>
bool resample(typename SparseField<Data_T>::Ptr source,
              typename SparseField<Data_T>::Ptr target,
              V3i reSize,
              int filterType,
              int numThreads)
{
  // Compile-time error checks
  assert(source != NULL);
  assert(target != NULL);
  assert(reSize.x > 0);
  assert(reSize.y > 0);
  assert(reSize.z > 0);
  assert(numThreads >= 0);

  // Error checks
  if (source == NULL) {
    Msg::print(Msg::SevWarning, "FieldProcessing::volumeAdvect(): "
              "Got null source pointer");
    return false;
  }
  if (target == NULL) {
    Msg::print(Msg::SevWarning, "FieldProcessing::volumeAdvect(): "
              "Got null target pointer");
    return false;
  }
  if (source.get() == target.get()) {
    Msg::print(Msg::SevWarning, "FieldProcessing::volumeAdvect(): "
              "Cannot let source and target buffer be same pointer");
    return false;    
  }
  if( reSize.x < 1  ) {
    Msg::print(Msg::SevWarning, "FieldProcessing::volumeAdvect(): "
              "Resize in x axis too small");
    return false;    
  }
  if( reSize.y < 1  ) {
    Msg::print(Msg::SevWarning, "FieldProcessing::volumeAdvect(): "
              "Resize in y axis too small");
    return false;    
  }
  if( reSize.z < 1  ) {
    Msg::print(Msg::SevWarning, "FieldProcessing::volumeAdvect(): "
              "Resize in z axis too small");
    return false;    
  }
  if (numThreads <= 0) {
    Msg::print(Msg::SevWarning, "FieldProcessing::resample(): "
              "Threads < 1");
    return false;
  }

  ResampleSparseCPU<Data_T> resamp(reSize, filterType, numThreads);
  return resamp.resample(source, target);
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_HEADER_CLOSE

//----------------------------------------------------------------------------//

#endif // Include guard
