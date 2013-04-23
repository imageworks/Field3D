//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2012 Sony Pictures Imageworks and DreamWorks Animation LLC
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

#include <DenseField.h>
#include <FieldInterp.h>
#include <SparseField.h>
#include <Log.h>

#include <openvdb/openvdb.h>
#include <openvdb/tree/ValueAccessor.h>
#include <openvdb/util/Util.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/ValueTransformer.h>
#include <openvdb/Types.h>

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#ifdef __APPLE__
#include <mach/mach.h>
#elif __linux__
#include <unistd.h>
#include <ios>
#include <fstream>
#endif

//----------------------------------------------------------------------------//

using namespace boost;
using namespace std;
using namespace Field3D;
using namespace boost::posix_time;
using namespace openvdb;

typedef boost::mt19937 RNGType;

//----------------------------------------------------------------------------//
// Globals
//----------------------------------------------------------------------------//

static size_t g_baseRSS;

//----------------------------------------------------------------------------//
// Util
//----------------------------------------------------------------------------//

static size_t currentRSS()
{
#ifdef __APPLE__
  struct task_basic_info t_info;
  mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;
  if (KERN_SUCCESS != task_info(mach_task_self(),
                                TASK_BASIC_INFO, 
                                (task_info_t)&t_info, 
                                &t_info_count)) {
    return -1;
  }
  return t_info.resident_size;
#elif __linux__
  using std::ios_base;
  using std::ifstream;
  using std::string;

  ifstream stat_stream("/proc/self/stat", ios_base::in);

  string pid, comm, state, ppid, pgrp, session, tty_nr;
  string tpgid, flags, minflt, cminflt, majflt, cmajflt;
  string utime, stime, cutime, cstime, priority, nice;
  string O, itrealvalue, starttime;

  unsigned long vsize;
  long rss;

  stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
              >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
              >> utime >> stime >> cutime >> cstime >> priority >> nice
              >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

  stat_stream.close();

  long page_size = sysconf(_SC_PAGE_SIZE); // in case x86-64 is configured to use 2MB pages
  // vm_usage     = vsize / 1024.0;
  // resident_set = rss * page_size;
  return rss * page_size;
#endif
}

//----------------------------------------------------------------------------//

struct Stats
{
  Stats(size_t a, size_t t, size_t m, size_t r, size_t c = 0)
    : mAllocTime(a), mTime(t), mMemSize(m), mRSS(r), mCheckSum(c) 
  { }
  size_t mAllocTime, mTime, mMemSize, mRSS, mCheckSum;
};

//----------------------------------------------------------------------------//

struct Timer
{
  Timer()
    : m_startTime(microsec_clock::local_time())
  { }
  size_t ms()
  { 
    return size_t((microsec_clock::local_time() - 
                   m_startTime).total_milliseconds()); 
  }
  size_t us()
  { 
    return size_t((microsec_clock::local_time() - 
                   m_startTime).total_microseconds()); 
  }
private:
  ptime m_startTime;
};

//----------------------------------------------------------------------------//

void printStats(const std::string s, const Stats &stats)
{
  std::cout << std::left << " - " << std::setw (15) << s 
            << " | alloc: " << std::setw(9) << stats.mAllocTime 
            << " | run: " << std::setw(9) << stats.mTime 
            << " | total: " << std::setw(9) << stats.mTime + stats.mAllocTime
            << " | mem: " << std::setw (14) << stats.mMemSize  
            << " | rss: " << std::setw (14) << stats.mRSS - g_baseRSS
            << " | check sum: " << stats.mCheckSum << "\n";
}

//----------------------------------------------------------------------------//

// Check if region is properly filled, no over or under filling.

size_t checksumF3DDense(const DenseField<float> &field)
{
  double sum = 0.0;
  for (DenseField<float>::const_iterator i = field.cbegin(), end = field.cend(); 
       i != end; ++i) {
    sum += double(*i);
  }
  return size_t(sum);
}

//----------------------------------------------------------------------------//

size_t checksumF3DSparse(SparseField<float> &sparse)
{
  const int blockOrder = sparse.blockOrder();
  double sum = 0.0;
  for (SparseField<float>::block_iterator b = sparse.blockBegin(),
         bEnd = sparse.blockEnd(); b != bEnd; ++b) {

    Box3i box = b.blockBoundingBox();
        
    if (!sparse.voxelIsInAllocatedBlock(box.min.x, box.min.y, box.min.z)) 
      continue;
        
    const float *start = &sparse.fastLValue(box.min.x, box.min.y, box.min.z);
    const float *end = start + (1 << blockOrder << blockOrder << blockOrder);
    while (start != end) {
      sum += *start++;
    }
  }
  return size_t(sum);
}

//----------------------------------------------------------------------------//

template <typename TreeType>
size_t checksumVDB(const TreeType &tree)
{
  double sum = 0.0;
  for (typename TreeType::LeafCIter iter = tree.cbeginLeaf(); iter; ++iter) {
    const float *value = &iter->getFirstValue(), 
      *end = &iter->getLastValue() + 1;
    while (value != end) sum += *value++;
  }
  return size_t(sum);
}

//----------------------------------------------------------------------------//
// Defines
//----------------------------------------------------------------------------//

#define DECLARE_TIMING_VARIABLES                         \
  size_t runTime = std::numeric_limits<size_t>::max();   \
  size_t allocTime = std::numeric_limits<size_t>::max(); \
  size_t memUsage = 0;                                   \
  size_t checkSum = 0;                                   \
  size_t memRSS = 0;                                     \

#define ALLOC_TIMER \
  Timer allocTimer;
#define RUN_TIMER \
  Timer runTimer;

#define UPDATE_ALLOC_TIME(sec)                       \
  allocTime = std::min(allocTime, allocTimer.sec());
#define UPDATE_RUN_TIME(sec)                       \
  runTime = std::min(runTime, runTimer.sec());

//----------------------------------------------------------------------------//
// Tests
//----------------------------------------------------------------------------//

void  testContiguousWriteAccess(int size, int samples);
Stats testContiguousWriteAccessDense(int size, int samples);
Stats testContiguousWriteAccessSparse(int size, int blockOrder, int samples);
template<openvdb::Index Log2Dim>
Stats testContiguousWriteAccessVDB(int size, int samples);

// --

void  testContiguousPreAllocWriteAccess(int size, int samples);
Stats testContiguousPreAllocWriteAccessDense(int size, int samples);
Stats testContiguousPreAllocWriteAccessSparse(int size, int blockOrder, int samples);
template<openvdb::Index Log2Dim>
Stats testContiguousPreAllocWriteAccessVDB(int size, int samples);

// --

void  testContiguousReadAccess(int size, int samples);
Stats testContiguousReadAccessDense(int size, int samples);
Stats testContiguousReadAccessSparse(int size, int blockOrder, int samples);
template<openvdb::Index Log2Dim>
Stats testContiguousReadAccessVDB(int size, int samples);

// --

void  testMemoryCoherentWriteAccess(int size, int samples);
Stats testMemoryCoherentWriteAccessDense(int size, int samples);
Stats testMemoryCoherentWriteAccessSparse(int size, int blockOrder, int samples);
template<openvdb::Index Log2Dim>
Stats testMemoryCoherentWriteAccessVDB(int size, int samples);

// --

void  testMemoryCoherentPreAllocWriteAccess(int size, int samples);
Stats testMemoryCoherentPreAllocWriteAccessDense(int size, int samples);
Stats testMemoryCoherentPreAllocWriteAccessSparse(int size, int blockOrder, int samples);
template<openvdb::Index Log2Dim>
Stats testMemoryCoherentPreAllocWriteAccessVDB(int size, int samples);

// --

void  testMemoryCoherentReadAccess(int size, int samples);
Stats testMemoryCoherentReadAccessDense(int size, int samples);
Stats testMemoryCoherentReadAccessSparse(int size, int blockOrder, int samples);
template<openvdb::Index Log2Dim>
Stats testMemoryCoherentReadAccessVDB(int size, int samples);

// --

void testSparseFill(int size, int samples);
Stats testSparseFillSparse(int size, int order, int samples);
template<openvdb::Index Log2Dim>
Stats testSparseFillVDB(int size, int samples);

// --

void  testRandomWriteAccess(int numPoints, int size, int samples);
Stats testRandomWriteAccessDense(int numPoints, int size, int samples);
Stats testRandomWriteAccessSparse(int numPoints, int size, int blockOrder, int samples);
template<openvdb::Index Log2Dim>
Stats testRandomWriteAccessVDB(int numPoints, int size, int samples);

// --

void  testRandomPreAllocWriteAccess(int numPoints, int size, int samples);
Stats testRandomPreAllocWriteAccessDense(int numPoints, int size, int samples);
Stats testRandomPreAllocWriteAccessSparse(int numPoints, int size, int blockOrder, int samples);
template<openvdb::Index Log2Dim>
Stats testRandomPreAllocWriteAccessVDB(int numPoints, int size, int samples);

// --

void  testRandomReadAccess(int numPoints, int size, int samples);
Stats testRandomReadAccessDense(int numPoints, int size, int samples);
Stats testRandomReadAccessSparse(int numPoints, int size, int blockOrder, int samples);
template<openvdb::Index Log2Dim>
Stats testRandomReadAccessVDB(int numPoints, int size, int samples);

// --

void  testRandomPointInterpolation(int numPoints, int size, int samples);
Stats testRandomPointInterpolationDense(int numPoints, int size, int samples);
Stats testRandomPointInterpolationSparse(int numPoints, int size, int blockOrder, int samples);
template<openvdb::Index Log2Dim>
Stats testRandomPointInterpolationVDB(int numPoints, int size, int samples);

// --

void testUniformRaymarching(int numRays, double stepSize, int size, int samples);
Stats testUniformRaymarchingDense(int numRays, double stepSize, int size, int samples);
Stats testUniformRaymarchingSparse(int numRays, double stepSize, int size, int blockOrder, int samples);
template<openvdb::Index Log2Dim>
Stats testUniformRaymarchingVDB(int numRays, double stepSize, int size, int samples);

// --

void testDenseLevelSetSphere(int size, int samples);
Stats testDenseLevelSetSphereDense(int size, int samples);
Stats testDenseLevelSetSphereSparse(int size, int blockOrder, int samples);
template<openvdb::Index Log2Dim>
Stats testDenseLevelSetSphereVDB(int size, int samples);

// --

void  testNarrowBandLevelSetSphere(int halfWidth, int size, int samples);
Stats testNarrowBandLevelSetSphereDense(int halfWidth, int size, int samples);
Stats testNarrowBandLevelSetSphereSparse(int halfWidth, int size, int blockOrder, int samples);
template<openvdb::Index Log2Dim>
Stats testNarrowBandLevelSetSphereVDB(int halfWidth, int size, int samples);

//----------------------------------------------------------------------------//
// Main
//----------------------------------------------------------------------------//

int main()
{
  int samples = 10;
  int baseRes = 800;

  g_baseRSS = currentRSS();

  cout << "Dense Domain Tests - Field3D & OpenVDB (taking " << samples << 
    " samples, time in ms)" << endl;
  cout << "  Field3D - " << FIELD3D_MAJOR_VER << "." << FIELD3D_MINOR_VER 
       << "." << FIELD3D_MICRO_VER << endl;
  cout << "  OpenVDB - " << OPENVDB_LIBRARY_MAJOR_VERSION << "." 
       << OPENVDB_LIBRARY_MINOR_VERSION << "."
       << OPENVDB_LIBRARY_PATCH_VERSION << endl;

  testContiguousWriteAccess(baseRes, samples);
  testContiguousPreAllocWriteAccess(baseRes, samples);
  testContiguousReadAccess(baseRes, samples);

  testMemoryCoherentWriteAccess(baseRes, samples);
  testMemoryCoherentPreAllocWriteAccess(baseRes, samples);
  testMemoryCoherentReadAccess(baseRes, samples);

  testSparseFill(1024, samples);
  testSparseFill(2048, samples);

  int randomPoints = 200000;
  testRandomWriteAccess(randomPoints, baseRes, samples);
  testRandomPreAllocWriteAccess(randomPoints, baseRes, samples);

  randomPoints = 1000000;
  testRandomReadAccess(randomPoints, baseRes, samples);
  testRandomPointInterpolation(randomPoints, baseRes, samples);

  int numRays = 10000;
  double stepSize = 0.5;
  testUniformRaymarching(numRays, stepSize, baseRes, samples);

  testDenseLevelSetSphere(1000, samples);

  // MW: Width should be 5?
  testNarrowBandLevelSetSphere(3, 1000, samples);
  testNarrowBandLevelSetSphere(3, 2000, samples);
  testNarrowBandLevelSetSphere(3, 3000, samples);
  testNarrowBandLevelSetSphere(3, 4000, samples);
}


//----------------------------------------------------------------------------//
// Contiguous write access
//----------------------------------------------------------------------------//

void
testContiguousWriteAccess(int size, int samples)
{
  cout << "Contiguous write access "<< size << "^3\n";

  printStats("Dense", testContiguousWriteAccessDense(size, samples));
  
  printStats("Sparse 8", testContiguousWriteAccessSparse(size, 3, samples));
  printStats("VDB 8", testContiguousWriteAccessVDB<3>(size, samples));
  
  printStats("Sparse 16", testContiguousWriteAccessSparse(size, 4, samples));
  printStats("VDB 16", testContiguousWriteAccessVDB<4>(size, samples));

  printStats("Sparse 32", testContiguousWriteAccessSparse(size, 5, samples));
  printStats("VDB 32", testContiguousWriteAccessVDB<5>(size, samples));
}

//----------------------------------------------------------------------------//

Stats
testContiguousWriteAccessDense(int size, int samples)
{
  int rangeMin = 0, rangeMax = size - 1;
  
  DECLARE_TIMING_VARIABLES;

  for (int s = 0; s < samples; ++s) {

    ALLOC_TIMER;
  
    DenseField<float> dense;
    dense.setSize(Box3i(V3i(rangeMin), V3i(rangeMax)));

    UPDATE_ALLOC_TIME(ms);
    
    RUN_TIMER;
    
    for (int k = rangeMin; k <= rangeMax; ++k) {
      for (int j = rangeMin; j <= rangeMax; ++j) {
        for (int i = rangeMin; i <= rangeMax; ++i) {
          dense.fastLValue(i, j, k) = 1.0;
        }
      }
    }
    
    UPDATE_RUN_TIME(ms);
    
    checkSum = checksumF3DDense(dense);
    memUsage = dense.memSize();
    memRSS = currentRSS();
  }

  
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//

// MW: This text used to use pointer arithmetic. Too fast, apparently.
Stats
testContiguousWriteAccessSparse(int size, int blockOrder, int samples)
{
  int rangeMin = 0, rangeMax = size - 1;

  DECLARE_TIMING_VARIABLES;

  for (int s = 0; s < samples; ++s) {
  
    ALLOC_TIMER;
    
    SparseField<float> sparse;
    sparse.setBlockOrder(blockOrder);
    sparse.setSize(Box3i(V3i(rangeMin), V3i(rangeMax)));
    
    UPDATE_ALLOC_TIME(ms);
    
    RUN_TIMER;

    SparseField<float>::iterator i = sparse.begin();
    SparseField<float>::iterator end = sparse.end();
    for (; i != end; ++i) {
      *i = 1.0;
    }

    UPDATE_RUN_TIME(ms);

    checkSum = checksumF3DSparse(sparse);
    memUsage = sparse.memSize();
    memRSS = currentRSS();
  }
  

  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);  
}

//----------------------------------------------------------------------------//

template<openvdb::Index Log2Dim>
Stats 
testContiguousWriteAccessVDB(int size, int samples)
{
  typedef typename tree::Tree4<float, 5, 4, Log2Dim>::Type TreeType;

  int rangeMin = 0, rangeMax = size - 1;
    
  DECLARE_TIMING_VARIABLES;

  for (int s = 0; s < samples; ++s) {
  
    ALLOC_TIMER;
        
    TreeType tree(0.0);
    
    UPDATE_ALLOC_TIME(ms);

    tree::ValueAccessor<TreeType> accessor(tree);
    Coord ijk;
    
    RUN_TIMER;

    for (ijk[0] = rangeMin; ijk[0] <= rangeMax; ++ijk[0]) {
      for (ijk[1] = rangeMin; ijk[1] <= rangeMax; ++ijk[1]) {
        for (ijk[2] = rangeMin; ijk[2] <= rangeMax; ++ijk[2]) {
          accessor.setValueOnly(ijk, 1.0);
        }
      }
    }

    UPDATE_RUN_TIME(ms);

    checkSum = checksumVDB(tree);

    memUsage = tree.memUsage();
    memRSS = currentRSS();
  }

  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//
// Contiguous write access (prealloc)
//----------------------------------------------------------------------------//

void
testContiguousPreAllocWriteAccess(int size, int samples)
{
  cout << "Contiguous write access (preallocated) "<< size << "^3\n";

  printStats("Dense", testContiguousPreAllocWriteAccessDense(size, samples));

  printStats("Sparse 8", testContiguousPreAllocWriteAccessSparse(size, 3, samples));
  printStats("VDB 8", testContiguousPreAllocWriteAccessVDB<3>(size, samples));

  printStats("Sparse 16", testContiguousPreAllocWriteAccessSparse(size, 4, samples));
  printStats("VDB 16", testContiguousPreAllocWriteAccessVDB<4>(size, samples));

  printStats("Sparse 32", testContiguousPreAllocWriteAccessSparse(size, 5, samples));
  printStats("VDB 32", testContiguousPreAllocWriteAccessVDB<5>(size, samples));
}

//----------------------------------------------------------------------------//

Stats
testContiguousPreAllocWriteAccessDense(int size, int samples)
{
  int rangeMin = 0, rangeMax = size - 1;
  
  // pre generate volume
  DenseField<float> dense;
  dense.setSize(Box3i(V3i(rangeMin), V3i(rangeMax)));
  
  std::fill(dense.begin(), dense.end(), 0.0);


  DECLARE_TIMING_VARIABLES;

  ALLOC_TIMER;
  
  UPDATE_ALLOC_TIME(ms);
  
  for (int s = 0; s < samples; ++s) {

    RUN_TIMER;
    
    for (int k = rangeMin; k <= rangeMax; ++k) {
      for (int j = rangeMin; j <= rangeMax; ++j) {
        for (int i = rangeMin; i <= rangeMax; ++i) {
          dense.fastLValue(i, j, k) = 1.0;
        }
      }
    }

    UPDATE_RUN_TIME(ms);

    checkSum = checksumF3DDense(dense);

    memUsage = dense.memSize();
    memRSS = currentRSS();
  }
  
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//

// MW: This test used to use pointer arithmetic.
Stats
testContiguousPreAllocWriteAccessSparse(int size, int blockOrder, int samples)
{
  int rangeMin = 0, rangeMax = size - 1;

  // pre generate volume
  SparseField<float> sparse;
  sparse.setBlockOrder(blockOrder);
  sparse.setSize(Box3i(V3i(rangeMin), V3i(rangeMax)));
  
  std::fill(sparse.begin(), sparse.end(), 0.0);


  DECLARE_TIMING_VARIABLES;

  ALLOC_TIMER;
  UPDATE_ALLOC_TIME(ms);

  for (int s = 0; s < samples; ++s) {
      
    RUN_TIMER;
        
    SparseField<float>::iterator i = sparse.begin();
    SparseField<float>::iterator end = sparse.end();
    for (; i != end; ++i) {
      *i = 1.0;
    }
  
    UPDATE_RUN_TIME(ms);

    checkSum = checksumF3DSparse(sparse);

    memUsage = sparse.memSize();
    memRSS = currentRSS();
  }
  
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);  
}

//----------------------------------------------------------------------------//
  
template<openvdb::Index Log2Dim>
Stats 
testContiguousPreAllocWriteAccessVDB(int size, int samples)
{
  typedef typename tree::Tree4<float, 5, 4, Log2Dim>::Type TreeType;

  int rangeMin = 0, rangeMax = size - 1;

  // pre generate volume
  TreeType tree(0.0);
  tree.fill(CoordBBox(Coord(rangeMin), Coord(rangeMax)), 0.0);
  tree.voxelizeActiveTiles();
    
  DECLARE_TIMING_VARIABLES;

  ALLOC_TIMER;  
  UPDATE_ALLOC_TIME(ms);

  tree::ValueAccessor<TreeType> accessor(tree);
  Coord ijk;  

  for (int s = 0; s < samples; ++s) {
  
    RUN_TIMER;
    
    for (ijk[0] = rangeMin; ijk[0] <= rangeMax; ++ijk[0]) {
        for (ijk[1] = rangeMin; ijk[1] <= rangeMax; ++ijk[1]) {
            for (ijk[2] = rangeMin; ijk[2] <= rangeMax; ++ijk[2]) {
                accessor.setValueOnly(ijk, 1.0);
            }
        }
    }

    UPDATE_RUN_TIME(ms);

    checkSum = checksumVDB(tree);

    memUsage = tree.memUsage();
    memRSS = currentRSS();
  }

  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//
// Contiguous read access
//----------------------------------------------------------------------------//

void
testContiguousReadAccess(int size, int samples)
{
  cout << "Contiguous read access "<< size << "^3\n";

  printStats("Dense", testContiguousReadAccessDense(size, samples));

  printStats("Sparse 8", testContiguousReadAccessSparse(size, 3, samples));
  printStats("VDB 8", testContiguousReadAccessVDB<3>(size, samples));

  printStats("Sparse 16", testContiguousReadAccessSparse(size, 4, samples));
  printStats("VDB 16", testContiguousReadAccessVDB<4>(size, samples));

  printStats("Sparse 32", testContiguousReadAccessSparse(size, 5, samples));
  printStats("VDB 32", testContiguousReadAccessVDB<5>(size, samples));
}

//----------------------------------------------------------------------------//

Stats
testContiguousReadAccessDense(int size, int samples)
{
  int rangeMin = 0, rangeMax = size - 1;

  // pre generate volume
  DenseField<float> dense;
  dense.setSize(Box3i(V3i(rangeMin), V3i(rangeMax)));
  std::fill(dense.begin(), dense.end(), 1.0);

    
  DECLARE_TIMING_VARIABLES;

  ALLOC_TIMER;
  UPDATE_ALLOC_TIME(ms);

  for (int s = 0; s < samples; ++s) {

    double sum = 0.0;

    RUN_TIMER;
  
    for (int k = rangeMin; k <= rangeMax; ++k) {
      for (int j = rangeMin; j <= rangeMax; ++j) {
        for (int i = rangeMin; i <= rangeMax; ++i) {
          sum += dense.fastValue(i, j, k);
        }
      }
    }

    UPDATE_RUN_TIME(ms);
    
    memUsage = dense.memSize();
    memRSS = currentRSS();
    checkSum = size_t(sum);
  }
  
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}


//----------------------------------------------------------------------------//

// This test used to use pointer arithmetic.
Stats
testContiguousReadAccessSparse(int size, int blockOrder, int samples)
{
  int rangeMin = 0, rangeMax = size - 1;
  
  // pre generate volume
  SparseField<float> sparse;
  sparse.setBlockOrder(blockOrder);
  sparse.setSize(Box3i(V3i(rangeMin), V3i(rangeMax)));

  std::fill(sparse.begin(), sparse.end(), 1.0);


  DECLARE_TIMING_VARIABLES;

  ALLOC_TIMER;
  UPDATE_ALLOC_TIME(ms);
  
  for (int s = 0; s < samples; ++s) {
  
    double sum = 0.0;
    
    RUN_TIMER;
  
    SparseField<float>::iterator i = sparse.begin();
    SparseField<float>::iterator end = sparse.end();
    for (; i != end; ++i) {
      sum += *i;
    }
  
    UPDATE_RUN_TIME(ms);

    memUsage = sparse.memSize();
    memRSS = currentRSS();
    checkSum = size_t(sum);
  }
  
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);  
}

//----------------------------------------------------------------------------//

template<openvdb::Index Log2Dim>
Stats 
testContiguousReadAccessVDB(int size, int samples)
{
  typedef typename tree::Tree4<float, 5, 4, Log2Dim>::Type TreeType;
  int rangeMin = 0, rangeMax = size - 1;
 
  // pre generate volume.
  TreeType tree(0.0);
  tree.fill(CoordBBox(Coord(rangeMin), Coord(rangeMax)), 1.0);
  tree.voxelizeActiveTiles();

  tree::ValueAccessor<TreeType> accessor(tree);
  Coord ijk;

  DECLARE_TIMING_VARIABLES;
  ALLOC_TIMER;    
  UPDATE_ALLOC_TIME(ms);
    
  for (int s = 0; s < samples; ++s) {
  
    double sum = 0.0;

    RUN_TIMER;
    
    for (ijk[0] = rangeMin; ijk[0] <= rangeMax; ++ijk[0]) {
      for (ijk[1] = rangeMin; ijk[1] <= rangeMax; ++ijk[1]) {
        for (ijk[2] = rangeMin; ijk[2] <= rangeMax; ++ijk[2]) {
          sum += accessor.getValue(ijk);
        }
      }
    }
    
    UPDATE_RUN_TIME(ms);

    memRSS = currentRSS();
    memUsage = tree.memUsage();
    checkSum = size_t(sum);
  }

  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//
// Memory coherent write access
//----------------------------------------------------------------------------//

void testMemoryCoherentWriteAccess(int size, int samples)
{
  cout << "Memory coherent write access "<< size << "^3\n";
  
  printStats("Dense", testMemoryCoherentWriteAccessDense (size, samples));

  printStats("Sparse 8", testMemoryCoherentWriteAccessSparse(size, 3, samples));
  printStats("VDB 8", testMemoryCoherentWriteAccessVDB<3>(size, samples));

  printStats("Sparse 16", testMemoryCoherentWriteAccessSparse(size, 4, samples));
  printStats("VDB 16", testMemoryCoherentWriteAccessVDB<4>(size, samples));

  printStats("Sparse 32", testMemoryCoherentWriteAccessSparse(size, 5, samples));
  printStats("VDB 32", testMemoryCoherentWriteAccessVDB<5>(size, samples));
}

//----------------------------------------------------------------------------//

Stats testMemoryCoherentWriteAccessDense(int size, int samples)
{
  int rangeMin = 0, rangeMax = size - 1;

  DECLARE_TIMING_VARIABLES;

  for (int s = 0; s < samples; ++s) {

    ALLOC_TIMER;

    DenseField<float> dense;
    dense.setSize(Box3i(V3i(rangeMin), V3i(rangeMax)));

    UPDATE_ALLOC_TIME(ms);
    
    RUN_TIMER;

    float *start = &dense.fastLValue(rangeMin, rangeMin, rangeMin);
    float *end = &dense.fastLValue(rangeMax, rangeMax, rangeMax) + 1;
    while (start != end) {
      *start++ = 1.0;
    }

    UPDATE_RUN_TIME(ms);

    checkSum = checksumF3DDense(dense);

    memRSS = currentRSS();
    memUsage = dense.memSize();
  }

  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//

Stats testMemoryCoherentWriteAccessSparse(int size, int blockOrder, int samples)
{
  int rangeMin = 0, rangeMax = size - 1;

  int blockSize = 1 << blockOrder;

  DECLARE_TIMING_VARIABLES;

  for (int s = 0; s < samples; ++s) {

    ALLOC_TIMER;
  
    SparseField<float> sparse;
    sparse.setBlockOrder(blockOrder);
    sparse.setSize(Box3i(V3i(rangeMin), V3i(rangeMax)));

    UPDATE_ALLOC_TIME(ms);
    
    RUN_TIMER;
    
    for (SparseField<float>::block_iterator b = sparse.blockBegin(), bEnd = sparse.blockEnd(); 
         b != bEnd; ++b) {

      Box3i box = b.blockBoundingBox();
      V3i size = box.size() + V3i(1);
      if (size.x == blockSize && size.y == blockSize && size.z == blockSize) {
        float *start = &sparse.fastLValue(box.min.x, box.min.y, box.min.z);
        float *end = &sparse.fastLValue(box.max.x, box.max.y, box.max.z) + 1;
        while (start != end) {
          *start++ = 1.0;
        }
      } else {
        float *start = &sparse.fastLValue(box.min.x, box.min.y, box.min.z);
        for (int k = box.min.z; k <= box.max.z; ++k) {
          for (int j = box.min.y; j <= box.max.y; ++j) {
            for (int i = box.min.x; i <= box.max.x; ++i) {
              *start++ = 1.0;
            }
            start += (blockSize - (box.max.x - box.min.x + 1));
          }
          start += (blockSize - (box.max.y - box.min.y + 1)) * 
            blockSize;
        }
      }
    }

    UPDATE_RUN_TIME(ms);
    
    checkSum = checksumF3DSparse(sparse);

    memRSS = currentRSS();
    memUsage = sparse.memSize();
  }
  
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);  
}


//----------------------------------------------------------------------------//

template<openvdb::Index Log2Dim>
Stats 
testMemoryCoherentWriteAccessVDB(int size, int samples)
{
  typedef typename tree::Tree4<float, 5, 4, Log2Dim>::Type TreeType;
  
  int rangeMin = 0, rangeMax = size - 1;
  
  DECLARE_TIMING_VARIABLES;

  for (int s = 0; s < samples; ++s) {
    
    ALLOC_TIMER;
  
    TreeType tree(0.0);

    UPDATE_ALLOC_TIME(ms);

    RUN_TIMER;
    
    tree.fill(CoordBBox(Coord(rangeMin), Coord(rangeMax)), 1.0);
    tree.voxelizeActiveTiles();
    
    UPDATE_RUN_TIME(ms);

    checkSum = checksumVDB(tree);

    memRSS = currentRSS();
    memUsage = tree.memUsage();
  }
  
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//
// Memory coherent write access (prealloc)
//----------------------------------------------------------------------------//

void testMemoryCoherentPreAllocWriteAccess(int size, int samples)
{
  cout << "Memory coherent write access (preallocated) "<< size << "^3\n";

  printStats("Dense", testMemoryCoherentPreAllocWriteAccessDense (size, samples));

  printStats("Sparse 8", testMemoryCoherentPreAllocWriteAccessSparse(size, 3, samples));
  printStats("VDB 8", testMemoryCoherentPreAllocWriteAccessVDB<3>(size, samples));

  printStats("Sparse 16", testMemoryCoherentPreAllocWriteAccessSparse(size, 4, samples));
  printStats("VDB 16", testMemoryCoherentPreAllocWriteAccessVDB<4>(size, samples));

  printStats("Sparse 32", testMemoryCoherentPreAllocWriteAccessSparse(size, 5, samples));
  printStats("VDB 32", testMemoryCoherentPreAllocWriteAccessVDB<5>(size, samples));
}

//----------------------------------------------------------------------------//

Stats testMemoryCoherentPreAllocWriteAccessDense(int size, int samples)
{
  int rangeMin = 0, rangeMax = size - 1;

  // pre generate volume
  DenseField<float> dense;
  dense.setSize(Box3i(V3i(rangeMin), V3i(rangeMax)));
  
  std::fill(dense.begin(), dense.end(), 0.0f);
  
  
  DECLARE_TIMING_VARIABLES;

  ALLOC_TIMER;
  UPDATE_ALLOC_TIME(ms);
  
  for (int s = 0; s < samples; ++s) {

    RUN_TIMER;
    
    float *start = &dense.fastLValue(rangeMin, rangeMin, rangeMin);
    float *end = &dense.fastLValue(rangeMax, rangeMax, rangeMax) + 1;
    while (start != end) {
      *start++ = 1.0;
    }
    
    UPDATE_RUN_TIME(ms);

    checkSum = checksumF3DDense(dense);

    memRSS = currentRSS();
    memUsage = dense.memSize();
  }

  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//

Stats testMemoryCoherentPreAllocWriteAccessSparse(int size, int blockOrder, 
                                                  int samples)
{
  int rangeMin = 0, rangeMax = size - 1;
 
  int blockSize = 1 << blockOrder;

  // pre generate volume
  SparseField<float> sparse;
  sparse.setBlockOrder(blockOrder);
  sparse.setSize(Box3i(V3i(rangeMin), V3i(rangeMax)));
  
  std::fill(sparse.begin(), sparse.end(), 0.0f);
  
  
  DECLARE_TIMING_VARIABLES;

  ALLOC_TIMER;
  UPDATE_ALLOC_TIME(ms);
  
  for (int s = 0; s < samples; ++s) {

    RUN_TIMER;
    
    for (SparseField<float>::block_iterator b = sparse.blockBegin(), bEnd = sparse.blockEnd(); 
         b != bEnd; ++b) {

      Box3i box = b.blockBoundingBox();
      V3i size = box.size() + V3i(1);
      if (size.x == 1 << blockOrder && 
          size.y == 1 << blockOrder && 
          size.z == 1 << blockOrder) {
        float *start = &sparse.fastLValue(box.min.x, box.min.y, box.min.z);
        float *end = &sparse.fastLValue(box.max.x, box.max.y, box.max.z) + 1;
        while (start != end) {
          *start++ = 1.0;
        }
      } else {
        float *start = &sparse.fastLValue(box.min.x, box.min.y, box.min.z);
        for (int k = box.min.z; k <= box.max.z; ++k) {
          for (int j = box.min.y; j <= box.max.y; ++j) {
            for (int i = box.min.x; i <= box.max.x; ++i) {
              *start++ = 1.0;
            }
            start += (blockSize - (box.max.x - box.min.x + 1));
          }
          start += (blockSize - (box.max.y - box.min.y + 1)) * 
            blockSize;
        }
      }
    }

    UPDATE_RUN_TIME(ms);

    checkSum = checksumF3DSparse(sparse);

    memRSS = currentRSS();
    memUsage = sparse.memSize();
  }
  
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);  
}

//----------------------------------------------------------------------------//

template<openvdb::Index Log2Dim>
Stats 
testMemoryCoherentPreAllocWriteAccessVDB(int size, int samples)
{
  typedef typename tree::Tree4<float, 5, 4, Log2Dim>::Type TreeType;
  typedef typename TreeType::LeafIter LeafIter;
  typedef typename TreeType::LeafNodeType::ValueOnIter ValueOnIter;
  int rangeMin = 0, rangeMax = size - 1;

  TreeType tree(0.0);
  tree.fill(CoordBBox(Coord(rangeMin), Coord(rangeMax)), 0.0);
  tree.voxelizeActiveTiles(); 
  
  DECLARE_TIMING_VARIABLES;

  ALLOC_TIMER;  
  UPDATE_ALLOC_TIME(ms);

  for (int s = 0; s < samples; ++s) {
    
    RUN_TIMER;

    for (LeafIter iter = tree.beginLeaf(); iter; ++iter) {
        if (iter->isDense()) {
            float *value = const_cast<float*>(&iter->getFirstValue());
            const float *end = &iter->getLastValue() + 1;
            while (value != end) *value++ = 1.0;
        } else {
            for (ValueOnIter it = iter->beginValueOn(); it; ++it) it.setValue(1.0);
        }
    }

    UPDATE_RUN_TIME(ms);

    checkSum = checksumVDB(tree);

    memRSS = currentRSS();
    memUsage = tree.memUsage();
  }
  
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//
// Memory coherent read access
//----------------------------------------------------------------------------//

void testMemoryCoherentReadAccess(int size, int samples)
{
  cout << "Memory coherent read access "<< size << "^3\n";

  printStats("Dense", testMemoryCoherentReadAccessDense (size, samples));

  printStats("Sparse 8", testMemoryCoherentReadAccessSparse(size, 3, samples));
  printStats("VDB 8", testMemoryCoherentReadAccessVDB<3>(size, samples));

  printStats("Sparse 16", testMemoryCoherentReadAccessSparse(size, 4, samples));
  printStats("VDB 16", testMemoryCoherentReadAccessVDB<4>(size, samples));

  printStats("Sparse 32", testMemoryCoherentReadAccessSparse(size, 5, samples));
  printStats("VDB 32", testMemoryCoherentReadAccessVDB<5>(size, samples));
}

//----------------------------------------------------------------------------//

Stats
testMemoryCoherentReadAccessDense(int size, int samples)
{
  int rangeMin = 0, rangeMax = size - 1;
  
  DECLARE_TIMING_VARIABLES;

  ALLOC_TIMER;
  UPDATE_ALLOC_TIME(ms);
  
  // pre generate dense volume
  DenseField<float> dense;
  dense.setSize(Box3i(V3i(rangeMin), V3i(rangeMax)));
  std::fill(dense.begin(), dense.end(), 1.0f);

  for (int s = 0; s < samples; ++s) {

    double sum = 0.0;

    RUN_TIMER;
    
    const float *start = &dense.lvalue(rangeMin, rangeMin, rangeMin);
    const float *end = &dense.lvalue(rangeMax, rangeMax, rangeMax) + 1;
    while (start != end) {
      sum += *start++;
    }
    
    UPDATE_RUN_TIME(ms);

    memRSS = currentRSS();
    memUsage = dense.memSize();
    checkSum = size_t(sum);
  }
  
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//

Stats
testMemoryCoherentReadAccessSparse(int size, int blockOrder, int samples)
{
  int rangeMin = 0, rangeMax = size - 1;
  
  int blockSize = 1 << blockOrder;

  DECLARE_TIMING_VARIABLES;

  ALLOC_TIMER;
  UPDATE_ALLOC_TIME(ms);
  
  // pre generate dense volume
  SparseField<float> sparse;
  sparse.setBlockOrder(blockOrder);
  sparse.setSize(Box3i(V3i(rangeMin), V3i(rangeMax)));
  
  std::fill(sparse.begin(), sparse.end(), 1.0);

  for (int s = 0; s < samples; ++s) {
  
    double sum = 0.0;
        
    RUN_TIMER;

    for (SparseField<float>::block_iterator b = sparse.blockBegin(), bEnd = sparse.blockEnd(); 
         b != bEnd; ++b) {
      Box3i box = b.blockBoundingBox();
      
      V3i size = box.size() + V3i(1);
      if (size.x == 1 << blockOrder && 
          size.y == 1 << blockOrder && 
          size.z == 1 << blockOrder) {
        const float *start = &sparse.fastLValue(box.min.x, box.min.y, box.min.z);
        const float * const end = &sparse.fastLValue(box.max.x, box.max.y, box.max.z) + 1;
        while (start != end) {
          sum += *start++; 
        }
      } else {
        float *start = &sparse.fastLValue(box.min.x, box.min.y, box.min.z);
        for (int k = box.min.z; k <= box.max.z; ++k) {
          for (int j = box.min.y; j <= box.max.y; ++j) {
            for (int i = box.min.x; i <= box.max.x; ++i) {
              sum += *start++;
            }
            start += (blockSize - (box.max.x - box.min.x + 1));
          }
          start += (blockSize - (box.max.y - box.min.y + 1)) * 
            blockSize;
        }
      }
    }

    UPDATE_RUN_TIME(ms);

    memRSS = currentRSS();
    memUsage = sparse.memSize();
    checkSum = size_t(sum);
  }
  
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);  
}

//----------------------------------------------------------------------------//

template<openvdb::Index Log2Dim>
Stats 
testMemoryCoherentReadAccessVDB(int size, int samples)
{
  typedef typename tree::Tree4<float, 5, 4, Log2Dim>::Type TreeType;
  typedef typename TreeType::LeafCIter LeafCIter;
  typedef typename TreeType::LeafNodeType::ValueOnCIter ValueOnCIter;

  int rangeMin = 0, rangeMax = size - 1;

  // pre generate dense volume
  TreeType tree(0.0);
  tree.fill(CoordBBox(Coord(rangeMin), Coord(rangeMax)), 1.0);
  tree.voxelizeActiveTiles();
      
  DECLARE_TIMING_VARIABLES;

  ALLOC_TIMER;      
  UPDATE_ALLOC_TIME(ms);

  for (int s = 0; s < samples; ++s) {
  
    double sum = 0.0;
        
    RUN_TIMER;
                
    for (LeafCIter iter = tree.cbeginLeaf(); iter; ++iter) {
        if (iter->isDense()) {
            const float *value = &iter->getFirstValue(), *end = &iter->getLastValue() + 1;
            while (value != end) sum += *value++;
        } else {
            for (ValueOnCIter it = iter->cbeginValueOn(); it; ++it) sum += it.getValue();
        }
    }

    UPDATE_RUN_TIME(ms);

    memRSS = currentRSS();
    memUsage = tree.memUsage();
    checkSum = size_t(sum);
  }

  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//
// Sparse fill
//----------------------------------------------------------------------------//

void
testSparseFill(int size, int samples)
{
  std::cout << "Sparse fill (time in us) "<< size << "^3\n";

  printStats("Sparse 8", testSparseFillSparse(size, 3, samples));
  printStats("VDB 8", testSparseFillVDB<3>(size, samples));

  printStats("Sparse 16", testSparseFillSparse(size, 4, samples));
  printStats("VDB 16", testSparseFillVDB<4>(size, samples));

  printStats("Sparse 32", testSparseFillSparse(size, 5, samples));
  printStats("VDB 32", testSparseFillVDB<5>(size, samples));
}

//----------------------------------------------------------------------------//

Stats 
testSparseFillSparse(int size, int order, int samples)
{
  int rangeMin = 0, rangeMax = size - 1;
  
  DECLARE_TIMING_VARIABLES;

  for (int s = 0; s < samples; ++s) {
    
    ALLOC_TIMER;
  
    SparseField<float> sparse;
    sparse.setBlockOrder(order);
    sparse.setSize(Box3i(V3i(rangeMin), V3i(rangeMax)));
  
    UPDATE_ALLOC_TIME(us);
    
    RUN_TIMER;
    
    sparse.clear(1.0);

    UPDATE_RUN_TIME(us);

    memRSS = currentRSS();
    memUsage = sparse.memSize();
  }
  
  checkSum = 0;

  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//

template<openvdb::Index Log2Dim>
Stats 
testSparseFillVDB(int size, int samples)
{
  typedef typename tree::Tree4<float, 5, 4, Log2Dim>::Type TreeType;
  int rangeMin = 0, rangeMax = size - 1;
    
  DECLARE_TIMING_VARIABLES;

  for (int s = 0; s < samples; ++s) {
  
    ALLOC_TIMER;
    TreeType tree(0.0);
    UPDATE_ALLOC_TIME(us);
        
    RUN_TIMER;
    tree.fill(CoordBBox(Coord(rangeMin), Coord(rangeMax)), 1.0);

    UPDATE_RUN_TIME(us);
        
    memRSS = currentRSS();
    memUsage = tree.memUsage();
  }

  checkSum = 0;

  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//
// Random incoherent write access
//----------------------------------------------------------------------------//

void testRandomWriteAccess(int numPoints, int size, int samples)
{
  cout << "Random incoherent write access " << numPoints << " points, "<< size << "^3\n";
  
  if(size <= 1000) {
    printStats("Dense", testRandomWriteAccessDense (numPoints, size, samples));
  }

  printStats("Sparse 8", testRandomWriteAccessSparse(numPoints, size, 3, samples));
  printStats("VDB 8", testRandomWriteAccessVDB<3>(numPoints, size, samples));

  printStats("Sparse 16", testRandomWriteAccessSparse(numPoints, size, 4, samples));
  printStats("VDB 16", testRandomWriteAccessVDB<4>(numPoints, size, samples));

  printStats("Sparse 32", testRandomWriteAccessSparse(numPoints, size, 5, samples));
  printStats("VDB 32", testRandomWriteAccessVDB<5>(numPoints, size, samples));
}

//----------------------------------------------------------------------------//

Stats testRandomWriteAccessDense(int numPoints, int size, int samples)
{
  int rangeMin = 0, rangeMax = size - 1;
    
  RNGType rng(1);
  boost::uniform_int<int> range(rangeMin, rangeMax);
  boost::variate_generator< RNGType, boost::uniform_int<int> > randNr(rng, range);

  // pre generate random points
  std::vector<V3i> points;
  points.reserve(numPoints);
  for (int n = 0; n < numPoints; ++n) {
    points.push_back(V3i(randNr(), randNr(), randNr()));
  }
  
  DECLARE_TIMING_VARIABLES;
  
  for (int s = 0; s < samples; ++s) {
  
    ALLOC_TIMER;

    DenseField<float> dense;
    Box3i box = Box3i(V3i(rangeMin), V3i(rangeMax));
    dense.setSize(box);

    UPDATE_ALLOC_TIME(ms);

    RUN_TIMER;
    
    int count = 0;
    for (int n = 0; n < numPoints; ++n, ++count) {
      dense.fastLValue(points[n].x, points[n].z, points[n].y) = 1.0;
    }
   
    if (count == 0) {
      cout << count << endl;
    }

    UPDATE_RUN_TIME(ms);

    checkSum = checksumF3DDense(dense);

    memRSS = currentRSS();
    memUsage = dense.memSize();
  }
  
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//

Stats testRandomWriteAccessSparse(int numPoints, int size, int blockOrder, int samples)
{
  int rangeMin = 0, rangeMax = size - 1;
  
  RNGType rng(1);
  boost::uniform_int<int> range(rangeMin, rangeMax);
  boost::variate_generator< RNGType, boost::uniform_int<int> > randNr(rng, range);

  // pre generate random points
  std::vector<V3i> points;
  points.reserve(numPoints);
  for (int n = 0; n < numPoints; ++n) {  
    points.push_back(V3i(randNr(), randNr(), randNr()));
  }
  
  DECLARE_TIMING_VARIABLES;

  for (int s = 0; s < samples; ++s) {

    ALLOC_TIMER;

    SparseField<float> sparse;
    sparse.setBlockOrder(blockOrder);
    sparse.setSize(Box3i(V3i(rangeMin), V3i(rangeMax)));

    UPDATE_ALLOC_TIME(ms);

    RUN_TIMER;

    for (size_t n = 0, N = points.size(); n < N; ++n) {
      sparse.fastLValue(points[n].x, points[n].z, points[n].y) = 1.0;
    }

    UPDATE_RUN_TIME(ms);

    checkSum = checksumF3DSparse(sparse);

    memRSS = currentRSS();
    memUsage = sparse.memSize();
  }

  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//

template<openvdb::Index Log2Dim>
Stats
testRandomWriteAccessVDB(int numPoints, int size, int samples)
{
  typedef typename tree::Tree4<float, 5, 4, Log2Dim>::Type TreeType;
  int rangeMin = 0, rangeMax = size - 1;
    
  // pre gen. random points.
  RNGType rng(1);
  boost::uniform_int<int> range(rangeMin, rangeMax);
  boost::variate_generator< RNGType, boost::uniform_int<int> > randNr(rng, range);

  std::vector<Coord> points;
  points.reserve(numPoints);
  for (int n = 0; n < numPoints; ++n) {
    points.push_back(Coord(randNr(), randNr(), randNr()));
  }
    
  DECLARE_TIMING_VARIABLES;

  for (int s = 0; s < samples; ++s) {
  
    ALLOC_TIMER;

    TreeType tree(0.0);
        
    UPDATE_ALLOC_TIME(ms);

    RUN_TIMER;

    for (size_t n = 0, N = points.size(); n < N; ++n) {
        tree.setValueOnly(points[n], 1.0);
    }
        
    UPDATE_RUN_TIME(ms);

    checkSum = checksumVDB(tree);

    memRSS = currentRSS();
    memUsage = tree.memUsage();
  }
    
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//
// Random incoherent write access (prealloc)
//----------------------------------------------------------------------------//

void testRandomPreAllocWriteAccess(int numPoints, int size, int samples)
{
  cout << "Random incoherent write access (preallocated) " << numPoints << " points, "<< size << "^3\n";
  
  if(size <= 1000) {
    printStats("Dense", testRandomPreAllocWriteAccessDense (numPoints, size, samples));
  }

  printStats("Sparse 8", testRandomPreAllocWriteAccessSparse(numPoints, size, 3, samples));
  printStats("VDB 8", testRandomPreAllocWriteAccessVDB<3>(numPoints, size, samples));

  printStats("Sparse 16", testRandomPreAllocWriteAccessSparse(numPoints, size, 4, samples));
  printStats("VDB 16", testRandomPreAllocWriteAccessVDB<4>(numPoints, size, samples));

  printStats("Sparse 32", testRandomPreAllocWriteAccessSparse(numPoints, size, 5, samples));
  printStats("VDB 32", testRandomPreAllocWriteAccessVDB<5>(numPoints, size, samples));
}

//----------------------------------------------------------------------------//

Stats testRandomPreAllocWriteAccessDense(int numPoints, int size, int samples)
{
  int rangeMin = 0, rangeMax = size - 1;
    
  RNGType rng(1);
  boost::uniform_int<int> range(rangeMin, rangeMax);
  boost::variate_generator< RNGType, boost::uniform_int<int> > randNr(rng, range);

  // pre generate random points
  std::vector<V3i> points;
  points.reserve(numPoints);
  for (int n = 0; n < numPoints; ++n) {
    points.push_back(V3i(randNr(), randNr(), randNr()));
  }
  
  // pre generate dense volume.
  DenseField<float> dense;
  dense.setSize(Box3i(V3i(rangeMin), V3i(rangeMax)));
  std::fill(dense.begin(), dense.end(), 0.0f);
  
  DECLARE_TIMING_VARIABLES;
  
  ALLOC_TIMER;
  UPDATE_ALLOC_TIME(ms);

  for (int s = 0; s < samples; ++s) {
  

    RUN_TIMER;

    for (int n = 0; n < numPoints; ++n) {
      dense.fastLValue(points[n].x, points[n].z, points[n].y) = 1.0;
    }

    UPDATE_RUN_TIME(ms);

    checkSum = checksumF3DDense(dense);
    memRSS = currentRSS();
    memUsage = dense.memSize();
  }
  
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//

Stats testRandomPreAllocWriteAccessSparse(int numPoints, int size, int blockOrder, int samples)
{
  int rangeMin = 0, rangeMax = size - 1;
  
  RNGType rng(1);
  boost::uniform_int<int> range(rangeMin, rangeMax);
  boost::variate_generator< RNGType, boost::uniform_int<int> > randNr(rng, range);

  // pre generate random points
  std::vector<V3i> points;
  points.reserve(numPoints);
  for (int n = 0; n < numPoints; ++n) {  
    points.push_back(V3i(randNr(), randNr(), randNr()));
  }
  
  // pre generate dense volume.
  SparseField<float> sparse;
  sparse.setBlockOrder(blockOrder);
  sparse.setSize(Box3i(V3i(rangeMin), V3i(rangeMax)));  
  std::fill(sparse.begin(), sparse.end(), 0.0f);

  DECLARE_TIMING_VARIABLES;

  ALLOC_TIMER;
  UPDATE_ALLOC_TIME(ms);

  for (int s = 0; s < samples; ++s) {

    RUN_TIMER;

    for (size_t n = 0, N = points.size(); n < N; ++n) {
      sparse.fastLValue(points[n].x, points[n].z, points[n].y) = 1.0;
    }

    UPDATE_RUN_TIME(ms);

    checkSum = checksumF3DSparse(sparse);
    memRSS = currentRSS();
    memUsage = sparse.memSize();
  }

  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

template<openvdb::Index Log2Dim>
Stats
testRandomPreAllocWriteAccessVDB(int numPoints, int size, int samples)
{
  typedef typename tree::Tree4<float, 5, 4, Log2Dim>::Type TreeType;
  int rangeMin = 0, rangeMax = size - 1;
    
  // pre gen. random points.
  RNGType rng(1);
  boost::uniform_int<int> range(rangeMin, rangeMax);
  boost::variate_generator< RNGType, boost::uniform_int<int> > randNr(rng, range);

  std::vector<Coord> points;
  points.reserve(numPoints);
  for (int n = 0; n < numPoints; ++n) {
    points.push_back(Coord(randNr(), randNr(), randNr()));
  }

  // pre generate dense volume.
  TreeType tree(0.0);
  tree.fill(CoordBBox(Coord(rangeMin), Coord(rangeMax)), 0.0);
  tree.voxelizeActiveTiles();
  
  DECLARE_TIMING_VARIABLES;
  ALLOC_TIMER;
  UPDATE_ALLOC_TIME(ms);

  for (int s = 0; s < samples; ++s) {

    RUN_TIMER;

    for (size_t n = 0, N = points.size(); n < N; ++n) {
        tree.setValueOnly(points[n], 1.0);
    }
        
    UPDATE_RUN_TIME(ms);

    checkSum = checksumVDB(tree);

    memRSS = currentRSS();
    memUsage = tree.memUsage();
  }
    
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//
// Random incoherent read access
//----------------------------------------------------------------------------//

void testRandomReadAccess(int numPoints, int size, int samples)
{
  cout << "Random incoherent read access "<< numPoints << " points, "<< size << "^3\n";
  
  if(size <= 1000) {
    printStats("Dense", testRandomReadAccessDense(numPoints, size, samples));
  }

  printStats("Sparse 8", testRandomReadAccessSparse(numPoints, size, 3, samples));
  printStats("VDB 8", testRandomReadAccessVDB<3>(numPoints, size, samples));

  printStats("Sparse 16", testRandomReadAccessSparse(numPoints, size, 4, samples));
  printStats("VDB 16", testRandomReadAccessVDB<4>(numPoints, size, samples));

  printStats("Sparse 32", testRandomReadAccessSparse(numPoints, size, 5, samples));
  printStats("VDB 32", testRandomReadAccessVDB<5>(numPoints, size, samples));
}

//----------------------------------------------------------------------------//

Stats testRandomReadAccessDense(int numPoints, int size, int samples)
{
  int rangeMax = size - 1 >> 1, rangeMin = -rangeMax;
    
  RNGType rng(1);
  boost::uniform_int<int> range(rangeMin, rangeMax);
  boost::variate_generator< RNGType, boost::uniform_int<int> > randNr(rng, range);

  // pre generate random points
  std::vector<V3i> points;
  points.reserve(numPoints);
  for (int n = 0; n < numPoints; ++n) {
    points.push_back(V3i(randNr(), randNr(), randNr()));
  }
  
  // pre generate dense volume
  DenseField<float> dense;
  dense.setSize(Box3i(V3i(rangeMin), V3i(rangeMax)));
  std::fill(dense.begin(), dense.end(), 1.0f);

  
  DECLARE_TIMING_VARIABLES;

  ALLOC_TIMER;
  UPDATE_ALLOC_TIME(ms);
  
  for (int s = 0; s < samples; ++s) {
  
    double sum = 0.0;
    
    RUN_TIMER;

    for (int n = 0; n < numPoints; ++n) {
      sum += dense.fastValue(points[n].x, points[n].z, points[n].y);
    }
   
    UPDATE_RUN_TIME(ms);

    memRSS = currentRSS();
    memUsage = dense.memSize();
    checkSum = size_t(sum);
  }
  
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//

Stats testRandomReadAccessSparse(int numPoints, int size, int blockOrder, int samples)
{
  int rangeMax = size - 1 >> 1, rangeMin = -rangeMax;
  
  RNGType rng(1);
  boost::uniform_int<int> range(rangeMin, rangeMax);
  boost::variate_generator< RNGType, boost::uniform_int<int> > randNr(rng, range);

  // pre generate random points
  std::vector<V3i> points;
  points.reserve(numPoints);
  for (int n = 0; n < numPoints; ++n) {  
    points.push_back(V3i(randNr(), randNr(), randNr()));
  }
  
  // pre generate dense volume
  SparseField<float> sparse;
  sparse.setBlockOrder(blockOrder);
  sparse.setSize(Box3i(V3i(rangeMin), V3i(rangeMax)));
  std::fill(sparse.begin(), sparse.end(), 1.0f);
  
  DECLARE_TIMING_VARIABLES;

  ALLOC_TIMER;  
  UPDATE_ALLOC_TIME(ms);

  
  for (int s = 0; s < samples; ++s) {

    double sum = 0.0;

    RUN_TIMER;

    for (size_t n = 0, N = points.size(); n < N; ++n) {
      sum += sparse.fastValue(points[n].x, points[n].z, points[n].y);
    }

    UPDATE_RUN_TIME(ms);

    memRSS = currentRSS();
    memUsage = sparse.memSize();
    checkSum = size_t(sum);
  }

  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//

template<openvdb::Index Log2Dim>
Stats
testRandomReadAccessVDB(int numPoints, int size, int samples)
{
  typedef typename tree::Tree4<float, 5, 4, Log2Dim>::Type TreeType;
  int rangeMin = 0, rangeMax = size - 1;
    
  // pre gen. random points.
  RNGType rng(1);
  boost::uniform_int<int> range(rangeMin, rangeMax);
  boost::variate_generator< RNGType, boost::uniform_int<int> > randNr(rng, range);

  std::vector<Coord> points;
  points.reserve(numPoints);
  for (int n = 0; n < numPoints; ++n) {
    points.push_back(Coord(randNr(), randNr(), randNr()));
  }

  // pre generate dense volume.
  TreeType tree(0.0);
  tree.fill(CoordBBox(Coord(rangeMin), Coord(rangeMax)), 1.0);
  tree.voxelizeActiveTiles();
  
  DECLARE_TIMING_VARIABLES;

  ALLOC_TIMER;
  UPDATE_ALLOC_TIME(ms);

  for (int s = 0; s < samples; ++s) {
  
    double sum = 0.0;
  
    RUN_TIMER;

    for (size_t n = 0, N = points.size(); n < N; ++n) {
        sum += tree.getValue(points[n]);
    }
        
    UPDATE_RUN_TIME(ms);

    memRSS = currentRSS();
    memUsage = tree.memUsage();
    checkSum = size_t(sum);
  }
    
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//
// Random incoherent point interpolation
//----------------------------------------------------------------------------//

void testRandomPointInterpolation(int numPoints, int size, int samples)
{
  cout << "Random incoherent point interpolation "<< size << "^3\n";
  
  if (size <= 1000) {
    printStats("Dense", testRandomPointInterpolationDense (numPoints, size, samples));
  }  

  printStats("Sparse 8", testRandomPointInterpolationSparse(numPoints, size, 3, samples));
  printStats("VDB 8", testRandomPointInterpolationVDB<3>(numPoints, size, samples));

  printStats("Sparse 16", testRandomPointInterpolationSparse(numPoints, size, 4, samples));
  printStats("VDB 16", testRandomPointInterpolationVDB<4>(numPoints, size, samples));

  printStats("Sparse 32", testRandomPointInterpolationSparse(numPoints, size, 5, samples));
  printStats("VDB 32", testRandomPointInterpolationVDB<5>(numPoints, size, samples));
}

//----------------------------------------------------------------------------//

Stats testRandomPointInterpolationDense(int numPoints, int size, int samples)
{
  int rangeMin = 0, rangeMax = size - 1;
  
  RNGType rng(1);
  boost::uniform_real<double> range(rangeMin, rangeMax);  
  boost::variate_generator< RNGType, boost::uniform_real<double> > randNr(rng, range);

  // pre generate random points
  std::vector<V3d> points;
  points.reserve(numPoints);
  for (int n = 0; n < numPoints; ++n) {
    points.push_back(V3d(randNr(), randNr(), randNr()));
  }
  
  // pre generate dense volume
  DenseField<float> dense;
  dense.setSize(Box3i(V3i(rangeMin), V3i(rangeMax)));
  std::fill(dense.begin(), dense.end(), 1.0f);
  
  DECLARE_TIMING_VARIABLES;

  ALLOC_TIMER;  
  UPDATE_ALLOC_TIME(ms);

  DenseField<float>::LinearInterp interp;

  for (int s = 0; s < samples; ++s) {
  
    double sum = 0.0;

    RUN_TIMER;

    for (int n = 0; n < numPoints; ++n) {
      sum += interp.sample(dense, points[n]);
    }
    
    UPDATE_RUN_TIME(ms);

    memRSS = currentRSS();
    memUsage = dense.memSize();
    checkSum = size_t(sum);
  }
  
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}


//----------------------------------------------------------------------------//

Stats testRandomPointInterpolationSparse(int numPoints, int size, int blockOrder, int samples)
{
  int rangeMin = 0, rangeMax = size - 1;
  
  RNGType rng(1);
  boost::uniform_real<double> range(rangeMin, rangeMax);  
  boost::variate_generator< RNGType, boost::uniform_real<double> > randNr(rng, range);

  // pre generate random points
  std::vector<V3d> points;
  points.reserve(numPoints); 
  for (int n = 0; n < numPoints; ++n) {
    points.push_back(V3d(randNr(), randNr(), randNr()));
  }
  
  // pre generate dense volume
  SparseField<float> sparse;
  sparse.setBlockOrder(blockOrder);
  sparse.setSize(Box3i(V3i(rangeMin), V3i(rangeMax)));
  std::fill(sparse.begin(), sparse.end(), 1.0f);
  
  DECLARE_TIMING_VARIABLES;
  
  ALLOC_TIMER;  
  UPDATE_ALLOC_TIME(ms);

  SparseField<float>::LinearInterp interp;
  
  for (int s = 0; s < samples; ++s) {
  
    double sum = 0.0;

    RUN_TIMER;
    
    for (int n = 0; n < numPoints; ++n) {
      sum += interp.sample(sparse, points[n]);
    }
    
    UPDATE_RUN_TIME(ms);

    memRSS = currentRSS();
    memUsage = sparse.memSize();
    checkSum = size_t(sum);
  }
  
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//

template<openvdb::Index Log2Dim>
Stats
testRandomPointInterpolationVDB(int numPoints, int size, int samples)
{
  typedef typename tree::Tree4<float, 5, 4, Log2Dim>::Type TreeType;

  int rangeMin = 0, rangeMax = size - 1;
    
  // pre generate random points  
  RNGType rng(1);
  boost::uniform_real<double> range(rangeMin, rangeMax);  
  boost::variate_generator< RNGType, boost::uniform_real<double> > randNr(rng, range);

  std::vector<Vec3R> points;
  points.reserve(numPoints); 
  for (int n = 0; n < numPoints; ++n) {
    points.push_back(Vec3R(randNr(), randNr(), randNr()));
  }
  
  // pre generate dense volume.
  TreeType tree(0.0);
  tree.fill(CoordBBox(Coord(rangeMin), Coord(rangeMax)), 1.0);
  tree.voxelizeActiveTiles();
    
  DECLARE_TIMING_VARIABLES;

  ALLOC_TIMER;
  UPDATE_ALLOC_TIME(ms);

  tree::ValueAccessor<TreeType> accessor(tree);

  for (int s = 0; s < samples; ++s) {

    double sum = 0.0;

    RUN_TIMER;

    for (int n = 0; n < numPoints; ++n) {
      sum += tools::BoxSampler::sample(accessor, points[n]);
    }

    UPDATE_RUN_TIME(ms);

    memRSS = currentRSS();
    memUsage = tree.memUsage();
    checkSum = size_t(sum);
  }

  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//
// Uniform raymarching
//----------------------------------------------------------------------------//

void testUniformRaymarching(int numRays, double stepSize, int size, int samples)
{
  cout << "Uniform raymarching - #rays:"<< numRays << ", step size: " << stepSize
    << ", resolution: " << size << "^3\n";

  printStats("Dense", testUniformRaymarchingDense(numRays, stepSize, size, samples));

  printStats("Sparse 8", testUniformRaymarchingSparse(numRays, stepSize, size, 3, samples));
  printStats("VDB 8", testUniformRaymarchingVDB<3>(numRays, stepSize, size, samples));

  printStats("Sparse 16", testUniformRaymarchingSparse(numRays, stepSize, size, 4, samples));
  printStats("VDB 16", testUniformRaymarchingVDB<4>(numRays, stepSize, size, samples));

  printStats("Sparse 32", testUniformRaymarchingSparse(numRays, stepSize, size, 5, samples));
  printStats("VDB 32", testUniformRaymarchingVDB<5>(numRays, stepSize, size, samples));
}

//----------------------------------------------------------------------------//

Stats testUniformRaymarchingDense(int numRays, double stepSize, int size, 
                                  int samples)
{
  int rangeMin = 0, rangeMax = size - 1;
    
  RNGType rng(1);
  boost::uniform_real<double> range(rangeMin, rangeMax);  
  boost::variate_generator< RNGType, boost::uniform_real<double> > randNr(rng, range);

  // pre generate volume
  DenseField<float> dense;
  dense.setSize(Box3i(V3i(rangeMin), V3i(rangeMax)));
  std::fill(dense.begin(), dense.end(), 1.0f);


  DECLARE_TIMING_VARIABLES;

  ALLOC_TIMER;  
  UPDATE_ALLOC_TIME(ms);

  V3d origin, dir;
  DenseField<float>::LinearInterp interp;

  for (int s = 0; s < samples; ++s) {

    double sum = 0.0;

    RUN_TIMER;
        
    for (int n = 0; n < numRays; ++n) {

      origin = V3d(randNr(), rangeMin, randNr()),
        dir = V3d(randNr(), rangeMax, randNr()) - origin;
            
      double dMax = dir.length();
      dir.normalize();

      double d = 0.0;
      while (d < dMax) {
        sum += interp.sample(dense, origin + (dir * d));
        d += stepSize;
      }
    }

    UPDATE_RUN_TIME(ms);

    memRSS = currentRSS();
    memUsage = dense.memSize();
    checkSum = size_t(sum);
  }
  
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//

Stats testUniformRaymarchingSparse(int numRays, double stepSize, int size, 
                                   int blockOrder, int samples)
{
  int rangeMin = 0, rangeMax = size - 1;
    
  RNGType rng(1);
  boost::uniform_real<double> range(rangeMin, rangeMax);  
  boost::variate_generator< RNGType, boost::uniform_real<double> > randNr(rng, range);

  // pre generate dense volume
  SparseField<float> sparse;
  sparse.setBlockOrder(blockOrder);
  sparse.setSize(Box3i(V3i(rangeMin), V3i(rangeMax)));
  std::fill(sparse.begin(), sparse.end(), 1.0f);
  
  DECLARE_TIMING_VARIABLES;

  ALLOC_TIMER;
  UPDATE_ALLOC_TIME(ms);

  V3d origin, dir;
  SparseField<float>::LinearInterp interp;

  for (int s = 0; s < samples; ++s) {

    double sum = 0.0;

    RUN_TIMER;
        
    for (int n = 0; n < numRays; ++n) {

      origin = V3d(randNr(), rangeMin, randNr()),
        dir = V3d(randNr(), rangeMax, randNr()) - origin;

      double dMax = dir.length();
      dir.normalize();

      double d = 0.0;
      while (d < dMax) {
        sum += interp.sample(sparse, origin + (dir * d));
        d += stepSize;
      }
    }

    UPDATE_RUN_TIME(ms);
    
    memRSS = currentRSS();
    memUsage = sparse.memSize();
    checkSum = size_t(sum);
  }
  
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//

template<openvdb::Index Log2Dim>
Stats 
testUniformRaymarchingVDB(int numRays, double stepSize, int size, int samples)
{
  typedef typename tree::Tree4<float, 5, 4, Log2Dim>::Type TreeType;
  int rangeMin = 0, rangeMax = size - 1;
    
  RNGType rng(1);
  boost::uniform_real<double> range(rangeMin, rangeMax);  
  boost::variate_generator< RNGType, boost::uniform_real<double> > randNr(rng, range);
  
  // pre generate dense volume.
  TreeType tree(0.0);
  tree.fill(CoordBBox(Coord(rangeMin), Coord(rangeMax)), 1.0);
  tree.voxelizeActiveTiles();

  DECLARE_TIMING_VARIABLES;

  ALLOC_TIMER;
  UPDATE_ALLOC_TIME(ms);

  tree::ValueAccessor<TreeType> accessor(tree);
  Vec3R origin, dir;

  for (int s = 0; s < samples; ++s) {

    RUN_TIMER;

    double sum = 0.0;
    for (int n = 0; n < numRays; ++n) {

      origin = Vec3R(randNr(), rangeMin, randNr()),
      dir = Vec3R(randNr(), rangeMax, randNr()) - origin;

      double dMax = dir.length();
      dir.normalize();

      double d = 0.0;
      while (d < dMax) {
        sum += tools::BoxSampler::sample(accessor, origin + (dir * d));
        d += stepSize;
      }
    }

    UPDATE_RUN_TIME(ms);

    memRSS = currentRSS();
    memUsage = tree.memUsage();
    checkSum = size_t(sum);
  }
  
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//
// Dense level set sphere
//----------------------------------------------------------------------------//

void testDenseLevelSetSphere(int size, int samples)
{
  cout << "Dense level set sphere "<< size << "^3\n";

  printStats("Dense", testDenseLevelSetSphereDense(size, samples));

  printStats("Sparse 8", testDenseLevelSetSphereSparse(size, 3, samples));
  printStats("VDB 8", testDenseLevelSetSphereVDB<3>(size, samples));

  printStats("Sparse 16", testDenseLevelSetSphereSparse(size, 4, samples));
  printStats("VDB 16", testDenseLevelSetSphereVDB<4>(size, samples));

  printStats("Sparse 32", testDenseLevelSetSphereSparse(size, 5, samples));
  printStats("VDB 32", testDenseLevelSetSphereVDB<5>(size, samples));
}

//----------------------------------------------------------------------------//


Stats testDenseLevelSetSphereDense(int size, int samples)
{
  int rangeMin = 0, rangeMax = size - 1;

  // Setup
  const float
    dim = float(size),
    dx = 1.0f / dim,
    radius = 0.5 / dx;

  const int center = int(0.5 / dx);
  int i, j, k;
  float x2, x2y2, x2y2z2;

  DECLARE_TIMING_VARIABLES;
  
  for (int s = 0; s < samples; ++s) {
  
    ALLOC_TIMER;

    DenseField<float> dense;
  	dense.setSize(Box3i(V3i(rangeMin), V3i(rangeMax)));
 
    UPDATE_ALLOC_TIME(ms);

    RUN_TIMER;

    // Gen level-set sphere
    for (i = rangeMin; i <= rangeMax; ++i) {
      x2 = i - center;
      x2 *= x2;
      for (j = rangeMin; j <= rangeMax; ++j) {
        x2y2 = j - center;
        x2y2 *= x2y2;
        x2y2 += x2;
        for (k = rangeMin; k <= rangeMax; ++k) {
          x2y2z2 = k - center;
          x2y2z2 *= x2y2z2;
          x2y2z2 += x2y2;

          const float v = std::sqrt(x2y2z2) - radius;
          dense.fastLValue(k, j, i) = dx * v;

        }
      }
    }

    UPDATE_RUN_TIME(ms);

    memRSS = currentRSS();
    memUsage = dense.memSize();
    checkSum = checksumF3DDense(dense);
  }
  
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);  
}


//----------------------------------------------------------------------------//

Stats testDenseLevelSetSphereSparse(int size, int blockOrder, int samples)
{
  int rangeMin = 0, rangeMax = size - 1;

  // Setup
  const float
    dim = float(size),
    dx = 1.0f / dim,
    radius = 0.5 / dx;

  const int center = int(0.5 / dx);
  int i, j, k;
  float x2, x2y2, x2y2z2;

  DECLARE_TIMING_VARIABLES;
  
  for (int s = 0; s < samples; ++s) {
  
    ALLOC_TIMER;

    SparseField<float> sparse;
    sparse.setBlockOrder(blockOrder);
    sparse.setSize(Box3i(V3i(rangeMin), V3i(rangeMax)));
    
    UPDATE_ALLOC_TIME(ms);

    RUN_TIMER;
    
    // Gen level-set sphere
    for (i = rangeMin; i <= rangeMax; ++i) {
      x2 = i - center;
      x2 *= x2;
      for (j = rangeMin; j <= rangeMax; ++j) {
        x2y2 = j - center;
        x2y2 *= x2y2;
        x2y2 += x2;
        for (k = rangeMin; k <= rangeMax; ++k) {
          x2y2z2 = k - center;
          x2y2z2 *= x2y2z2;
          x2y2z2 += x2y2;

          const float v = std::sqrt(x2y2z2) - radius;
          sparse.fastLValue(k, j, i) = dx * v;
        }
      }
    }

    UPDATE_RUN_TIME(ms);

    memRSS = currentRSS();
    memUsage = sparse.memSize();
    
    checkSum = checksumF3DSparse(sparse);
  }
  
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);  
}

//----------------------------------------------------------------------------//

template<openvdb::Index Log2Dim>
Stats
testDenseLevelSetSphereVDB(int size, int samples)
{
  typedef typename tree::Tree4<float, 5, 4, Log2Dim>::Type TreeType;
  int rangeMin = 0, rangeMax = size - 1;

  const float
    dim = float(size),
    dx = 1.0f / dim,
    radius = 0.5 / dx;

  const int center = int(0.5 / dx);

  openvdb::Coord ijk;
  int &i = ijk[0], &j = ijk[1], &k = ijk[2];
  float x2, x2y2, x2y2z2;

  DECLARE_TIMING_VARIABLES;

  for (int s = 0; s < samples; ++s) {
    
    ALLOC_TIMER;

    TreeType tree(0.0);

    UPDATE_ALLOC_TIME(ms);

    tree::ValueAccessor<TreeType> accessor(tree);

    RUN_TIMER;
        
    // Gen. level-set sphere
    for (i = rangeMin; i <= rangeMax; ++i) {
      x2 = i - center;
      x2 *= x2;
      for (j = rangeMin; j <= rangeMax; ++j) {
        x2y2 = j - center;
        x2y2 *= x2y2;
        x2y2 += x2;
        for (k = rangeMin; k <= rangeMax; ++k) {
          x2y2z2 = k - center;
          x2y2z2 *= x2y2z2;
          x2y2z2 += x2y2;

          const float v = std::sqrt(x2y2z2) - radius;
          accessor.setValueOnly(ijk, dx * v);
        }
      }
    }
              
    UPDATE_RUN_TIME(ms);
    
    memRSS = currentRSS();
    memUsage = tree.memUsage();
    checkSum = checksumVDB(tree);
  }
  
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//
// Narrow band level set sphere
//----------------------------------------------------------------------------//

void testNarrowBandLevelSetSphere(int halfWidth, int size, int samples)
{
  cout << "Narrow band level set sphere "<< size << "^3\n";
  
  if (size <= 1000) {
    printStats("Dense", testNarrowBandLevelSetSphereDense(halfWidth, size, samples));
  }

  printStats("Sparse 8", testNarrowBandLevelSetSphereSparse(halfWidth, size, 3, samples));
  printStats("VDB 8", testNarrowBandLevelSetSphereVDB<3>(halfWidth, size, samples));

  printStats("Sparse 16", testNarrowBandLevelSetSphereSparse(halfWidth, size, 4, samples));
  printStats("VDB 16", testNarrowBandLevelSetSphereVDB<4>(halfWidth, size, samples));

  printStats("Sparse 32", testNarrowBandLevelSetSphereSparse(halfWidth, size, 5, samples));
  printStats("VDB 32", testNarrowBandLevelSetSphereVDB<5>(halfWidth, size, samples));
}

//----------------------------------------------------------------------------//


Stats testNarrowBandLevelSetSphereDense(int halfWidth, int size, int samples)
{
  int rangeMin = 0, rangeMax = size - 1;

  // Setup
  const float
    dim = float(size),
    w = float(halfWidth), // narrow band half-width
    dx = 1.0f / dim,
    backgroundValue = w * dx,
    radius = (0.5 - backgroundValue) / dx;

  const int center = int(0.5 / dx);
  int i, j, k, m = 1;
  float x2, x2y2, x2y2z2;

  DECLARE_TIMING_VARIABLES;
  
  for (int s = 0; s < samples; ++s) {
  
    ALLOC_TIMER;

    DenseField<float> dense;
  	dense.setSize(Box3i(V3i(rangeMin), V3i(rangeMax)));

    UPDATE_ALLOC_TIME(ms);

    RUN_TIMER;

    // Gen level-set sphere
    for (i = rangeMin; i <= rangeMax; ++i) {
      x2 = i - center;
      x2 *= x2;
      for (j = rangeMin; j <= rangeMax; ++j) {
        x2y2 = j - center;
        x2y2 *= x2y2;
        x2y2 += x2;
        for (k = rangeMin; k <= rangeMax; k += m) {
          x2y2z2 = k - center;
          x2y2z2 *= x2y2z2;
          x2y2z2 += x2y2;

          const float v = std::sqrt(x2y2z2) - radius, d = std::abs(v);
          m = 1;

          // Flipped the i & k coordinates here to account for Field3D's optimal memory mapping.
          if (d < w) dense.fastLValue(k, j, i) = dx * v;       
          else m += int(std::floor(d - w));
        }
      }
    }

    UPDATE_RUN_TIME(ms);

    memRSS = currentRSS();
    memUsage = dense.memSize();
    checkSum = checksumF3DDense(dense);
  }
  
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);  
}


//----------------------------------------------------------------------------//

Stats testNarrowBandLevelSetSphereSparse(int halfWidth, int size, int blockOrder, int samples)
{
  int rangeMin = 0, rangeMax = size - 1;

  // Setup
  const float
    dim = float(size),
    w = float(halfWidth), // narrow band half-width
    dx = 1.0f / dim,
    backgroundValue = w * dx,
    radius = (0.5 - backgroundValue) / dx;

  const int center = int(0.5 / dx);
  int i, j, k, m = 1;
  float x2, x2y2, x2y2z2;

  DECLARE_TIMING_VARIABLES;
  
  for (int s = 0; s < samples; ++s) {
  
    ALLOC_TIMER;

    SparseField<float> sparse;
    sparse.setBlockOrder(blockOrder);
    sparse.setSize(Box3i(V3i(rangeMin), V3i(rangeMax)));
    
    UPDATE_ALLOC_TIME(ms);

    RUN_TIMER;

    // Gen level-set sphere
    for (i = rangeMin; i <= rangeMax; ++i) {
      x2 = i - center;
      x2 *= x2;
      for (j = rangeMin; j <= rangeMax; ++j) {
        x2y2 = j - center;
        x2y2 *= x2y2;
        x2y2 += x2;
        for (k = rangeMin; k <= rangeMax; k += m) {
          x2y2z2 = k - center;
          x2y2z2 *= x2y2z2;
          x2y2z2 += x2y2;

          const float v = std::sqrt(x2y2z2) - radius, d = std::abs(v);
          m = 1;

          // Flipped the i & k coordinates here to account for Field3D's optimal memory mapping. 
          if (d < w) sparse.fastLValue(k, j, i) = dx * v; 
          else m += int(std::floor(d - w));
        }
      }
    }

    UPDATE_RUN_TIME(ms);

    memRSS = currentRSS();
    memUsage = sparse.memSize();
    checkSum = checksumF3DSparse(sparse);
  }

  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);  
}

//----------------------------------------------------------------------------//

template<openvdb::Index Log2Dim>
Stats
testNarrowBandLevelSetSphereVDB(int halfWidth, int size, int samples)
{
  typedef typename tree::Tree4<float, 5, 4, Log2Dim>::Type TreeType;
  int rangeMin = 0, rangeMax = size - 1;

  const float
    dim = float(size),
    w = float(halfWidth), // narrow band half-width
    dx = 1.0f / dim,
    backgroundValue = w * dx,
    radius = (0.5 - backgroundValue) / dx;

  const int center = int(0.5 / dx);

  openvdb::Coord ijk;
  int &i = ijk[0], &j = ijk[1], &k = ijk[2], m=1;
  float x2, x2y2, x2y2z2;

  DECLARE_TIMING_VARIABLES;

  for (int s = 0; s < samples; ++s) {
    
    ALLOC_TIMER;

    TreeType tree(0.0);

    UPDATE_ALLOC_TIME(ms);

    tree::ValueAccessor<TreeType> accessor(tree);

    RUN_TIMER;
                
    // Gen. level-set sphere
    for (i = rangeMin; i <= rangeMax; ++i) {
      x2 = i - center;
      x2 *= x2;
      for (j = rangeMin; j <= rangeMax; ++j) {
        x2y2 = j - center;
        x2y2 *= x2y2;
        x2y2 += x2;
        for (k = rangeMin; k <= rangeMax; k += m) {
          x2y2z2 = k - center;
          x2y2z2 *= x2y2z2;
          x2y2z2 += x2y2;

          const float v = std::sqrt(x2y2z2) - radius, d = std::abs(v);

          m = 1;
          if (d < w) accessor.setValue(ijk, dx * v);                        
          else m += int(std::floor(d - w));
        }
      }
    }
      
    UPDATE_RUN_TIME(ms);

    memRSS = currentRSS();
    memUsage = tree.memUsage();
    checkSum = checksumVDB(tree);
  }
  
  return Stats(allocTime, runTime, memUsage, memRSS, checkSum);
}

//----------------------------------------------------------------------------//
