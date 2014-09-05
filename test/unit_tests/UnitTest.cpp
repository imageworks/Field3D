//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2009 Sony Pictures Imageworks
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

#include <iostream>
#include <stdlib.h>

#include <boost/test/included/unit_test.hpp>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>

#include <OpenEXR/ImathFrustum.h>

#include "Field3D/DenseField.h"
#include "Field3D/EmptyField.h"
#include "Field3D/Field3DFile.h"
#include "Field3D/FieldInterp.h"
#include "Field3D/InitIO.h"
#include "Field3D/MACField.h"
#include "Field3D/MIPField.h"
#include "Field3D/MIPUtil.h"
#include "Field3D/SparseField.h"
#include "Field3D/Types.h"
#include "Field3D/Log.h"

//----------------------------------------------------------------------------//

using namespace boost;
using namespace boost::unit_test;

using namespace std;

using namespace Field3D;
using namespace Field3D::Hdf5Util;

//----------------------------------------------------------------------------//

namespace {
  using namespace boost::posix_time;
  template <typename T>
  struct Add
  {
    Add() 
      : sum(static_cast<T>(0.0))
    { }
    T sum;
    void operator()(const T &val)
    { sum += val; }
  };
  template <typename T>
  struct WriteSequence
  {
    WriteSequence()
      : current(static_cast<T>(0))
    { }
    T current;
    void operator()(T &val)
    { 
      val = current; 
      current += static_cast<T>(1.0);
    }
  };
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
  };  class ScopedPrintTimer
  {
  public:
    Timer timer;
    ScopedPrintTimer(int numThreads=1)
      : m_numThreads(numThreads)
      {}
    ~ScopedPrintTimer()
    {
      Msg::print("  Time elapsed: " + 
                 lexical_cast<string>(0.001 * timer.ms() / 
                                      float(m_numThreads)));
    }
    int m_numThreads;
  };
}

//----------------------------------------------------------------------------//
std::string
getTempFile(const std::string &file)
{
#ifdef WIN32
  std::string tempDir = ::getenv("TMP");
  if (*tempDir.rbegin() != '/' && *tempDir.rbegin() != '\\')
    tempDir += '/';
#else
  std::string tempDir = "/tmp/";
#endif
  
  // Make sure the path is valid.
  std::string valid_file(file);
  for (size_t i = 0; i < valid_file.size(); i++)
  {
    if (!isalnum(valid_file[i]) && valid_file[i] != '.')
      valid_file[i] = '_';
  }

  return tempDir + valid_file;
}

//----------------------------------------------------------------------------//

template <template <typename T> class Field_T, class Data_T>
void testBasicField()
{
  typedef Field_T<Data_T> SField;
  typedef Field_T<FIELD3D_VEC3_T<Data_T> > VField;
  typedef FIELD3D_VEC3_T<Data_T> Vec3_T;

  Msg::print("Basic Field tests for type " + 
             string(SField::staticClassType()));

  string currentTest;

  currentTest = "Checking empty field";

  {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    SField sField;
    VField vField;    
    BOOST_CHECK_EQUAL(sField.dataWindow().hasVolume(), false);
    BOOST_CHECK_EQUAL(vField.dataWindow().hasVolume(), false);   
  }

  currentTest = "Checking non-empty field";

  {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    SField sField;
    sField.setSize(V3i(10));
    VField vField;
    vField.setSize(V3i(10));
    BOOST_CHECK_EQUAL(sField.dataWindow().hasVolume(), true);
    BOOST_CHECK_EQUAL(vField.dataWindow().hasVolume(), true);   
  }
  
  currentTest = "Checking value in cleared field";

  {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    Data_T sVal(1.1f);
    Vec3_T vVal(1.2f);
    V3i size(10);
    SField sField;
    sField.setSize(size);
    VField vField;
    vField.setSize(size);
    sField.clear(sVal);
    vField.clear(vVal);
    BOOST_CHECK_EQUAL(sField.value(5, 5, 5), sVal);
    BOOST_CHECK_EQUAL(vField.value(5, 5, 5), vVal);   
  }
  
  currentTest = "Checking reading and writing entire field";

  {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    V3i size(50);
    SField sField;
    VField vField;
    sField.setSize(size);
    vField.setSize(size);
    sField.clear(0.0f);
    vField.clear(Vec3_T(0.0f));
    for (int k = 0; k < size.z; k++) {
      for (int j = 0; j < size.y; j++) {
        for (int i = 0; i < size.x; i++) {
          Data_T sVal = static_cast<Data_T>(i + j + k);
          Vec3_T vVal = 
            Vec3_T(static_cast<Data_T>(i + 2 * j + 2 * k));
          sField.lvalue(i, j, k) = sVal;
          vField.lvalue(i, j, k) = vVal;
          BOOST_CHECK_EQUAL(sField.value(i, j, k), sVal);
          BOOST_CHECK_EQUAL(vField.value(i, j, k), vVal);
        }
      }
    }
  }

  currentTest = "Checking large extents, small data window";

  {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    Box3i extents(V3i(-5), V3i(50));
    Box3i data(V3i(20), V3i(30));
    SField sField;
    VField vField;
    sField.setSize(extents, data);
    vField.setSize(extents, data);
    sField.clear(0.0f);
    vField.clear(Vec3_T(0.0f));
    for (int k = data.min.z; k <= data.max.z; k++) {
      for (int j = data.min.y; j <= data.max.y; j++) {
        for (int i = data.min.x; i <= data.max.x; i++) {
          Data_T sVal = static_cast<Data_T>(i + j + k);
          Vec3_T vVal = 
            Vec3_T(static_cast<Data_T>(i + 2 * j + 2 * k));
          sField.lvalue(i, j, k) = sVal;
          vField.lvalue(i, j, k) = vVal;
          BOOST_CHECK_EQUAL(sField.value(i, j, k), sVal);
          BOOST_CHECK_EQUAL(vField.value(i, j, k), vVal);
        }
      }
    }
  }

#if 0

  currentTest = "Checking large extents, small data window, with safeValue()";

  {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    Box3i extents(V3i(-5), V3i(50));
    Box3i data(V3i(40), V3i(45));
    SField sField;
    VField vField;
    sField.setSize(extents, data);
    vField.setSize(extents, data);
    sField.clear(0.0f);
    vField.clear(Vec3_T(0.0f));
    Data_T safeSVal = 1.0f;
    Vec3_T safeVVal = Vec3_T(1.1f);
    for (int k = extents.min.z; k <= extents.max.z; k++) {
      for (int j = extents.min.y; j <= extents.max.y; j++) {
        for (int i = extents.min.x; i <= extents.max.x; i++) {
          Data_T sVal = static_cast<Data_T>(i + j + k);
          Vec3_T vVal = 
            Vec3_T(static_cast<Data_T>(i + 2 * j + 2 * k));
          if (data.intersects(V3i(i, j, k))) {
            sField.lvalue(i, j, k) = sVal;
            vField.lvalue(i, j, k) = vVal;
            BOOST_CHECK_EQUAL(sField.safeValue(i, j, k, safeSVal), sVal);
            BOOST_CHECK_EQUAL(vField.safeValue(i, j, k, safeVVal), vVal);
          } else {
            BOOST_CHECK_EQUAL(sField.safeValue(i, j, k, safeSVal), safeSVal);
            BOOST_CHECK_EQUAL(vField.safeValue(i, j, k, safeVVal), safeVVal);
          }
        }
      }
    }
  }

#endif

  currentTest = "Checking std::fill works with iterators";

  if (true) {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    Data_T testVal = 1.5f;
    Box3i extents(V3i(-5), V3i(25));
    V3i res = extents.max - extents.min + V3i(1);
    SField sField;
    sField.setSize(extents);
    std::fill(sField.begin(), sField.end(), testVal);
    for (int k = extents.min.z; k <= extents.max.z; k++) {
      for (int j = extents.min.y; j <= extents.max.y; j++) {
        for (int i = extents.min.x; i <= extents.max.x; i++) {
          BOOST_CHECK_EQUAL(sField.value(i, j, k), testVal);
        }
      }
    }    
  }

  currentTest = "Checking read from const_iterator - full field";

  if (true) {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    Data_T testVal = 1.5f;
    Box3i extents(V3i(-5), V3i(25));
    V3i res = extents.max - extents.min + V3i(1);
    int numCells = res.x * res.y * res.z;
    SField sField;
    sField.setSize(extents);
    std::fill(sField.begin(), sField.end(), testVal);    
    {
      typename SField::const_iterator i = sField.cbegin();
      typename SField::const_iterator end = sField.cend();
      int cellCountI = 0;
      for (; i != end; ++i) {
        BOOST_CHECK_EQUAL(*i, testVal);
        cellCountI++;
      }
      BOOST_CHECK_EQUAL(cellCountI, numCells);
    }
  }

  currentTest = "Checking read from const_iterator - empty field";

  {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    Data_T testVal = 1.5f;
    Box3i extents(V3i(-5), V3i(25));
    V3i res = extents.max - extents.min + V3i(1);
    int numCells = res.x * res.y * res.z;
    SField sField;
    sField.setSize(extents);
    sField.clear(testVal);
    typename SField::const_iterator i = sField.cbegin();
    typename SField::const_iterator end = sField.cend();
    int cellCountI = 0;
    for (; i != end; ++i) {
      BOOST_CHECK_EQUAL(*i, testVal);
      cellCountI++;
    }
    BOOST_CHECK_EQUAL(cellCountI, numCells);
    // Write a single voxels and then run the test again
    sField.lvalue(10, 10, 10) = testVal;
    typename SField::const_iterator j = sField.cbegin();
    int cellCountJ = 0;
    for (; j != end; ++j) {
      BOOST_CHECK_EQUAL(*j, testVal);    
      cellCountJ++;
    }
    BOOST_CHECK_EQUAL(cellCountJ, numCells);    
  }

  currentTest = "Checking read of subset from const_iterator - empty field";

  {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    Data_T testVal = 1.5f;
    Box3i extents(V3i(-5), V3i(25));
    SField sField;
    sField.setSize(extents);
    sField.clear(testVal);
    // Region to read
    Box3i toRead(V3i(10), V3i(18));
    typename SField::const_iterator i = sField.cbegin(toRead);
    typename SField::const_iterator end = sField.cend(toRead);
    int cellCountI = 0;
    for (; i != end; ++i) {
      BOOST_CHECK_EQUAL(*i, testVal);
      cellCountI++;
    }
    V3i res = toRead.max - toRead.min + V3i(1);
    int numCells = res.x * res.y * res.z;
    BOOST_CHECK_EQUAL(cellCountI, numCells);
    // Write a single voxels and then run the test again
    sField.lvalue(10, 10, 10) = testVal;
    typename SField::const_iterator j = sField.cbegin(toRead);
    int cellCountJ = 0;
    for (; j != end; ++j) {

      BOOST_CHECK_EQUAL(*j, testVal);    
      cellCountJ++;
    }
    BOOST_CHECK_EQUAL(cellCountJ, numCells);    
  }

  currentTest = "Checking read from iterator";

  {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    Data_T testVal = 1.5f;
    Box3i extents(V3i(-5), V3i(25));
    V3i res = extents.max - extents.min + V3i(1);
    int numCells = res.x * res.y * res.z;
    SField sField;
    sField.setSize(extents);
    sField.clear(testVal);
    typename SField::iterator i = sField.begin();
    typename SField::iterator end = sField.end();
    int cellCountI = 0;
    for (; i != end; ++i) {
      BOOST_CHECK_EQUAL(*i, testVal);
      cellCountI++;
    }
    BOOST_CHECK_EQUAL(cellCountI, numCells);
    // Write a single voxels and then run the test again
    sField.lvalue(10, 10, 10) = testVal;
    typename SField::iterator j = sField.begin();
    int cellCountJ = 0;
    for (; j != end; ++j) {
      BOOST_CHECK_EQUAL(*j, testVal);    
      cellCountJ++;
    }
    BOOST_CHECK_EQUAL(cellCountJ, numCells);    
  }


  currentTest = "Checking iterator correctness";

  {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    Box3i extents(V3i(-5), V3i(25));
    SField sField;
    VField vField;
    sField.setSize(extents);
    vField.setSize(extents);
    // Fill with data
    std::for_each(sField.begin(), sField.end(), 
                  WriteSequence<Data_T>());
    std::for_each(vField.begin(), vField.end(), 
                  WriteSequence<Vec3_T>());
    // Iterators to check
    typename SField::iterator iter = sField.begin();
    typename SField::const_iterator citer = sField.cbegin();
    // Check that iterators match explicit loop
    for (int k = extents.min.z; k <= extents.max.z; k++) {
      for (int j = extents.min.y; j <= extents.max.y; j++) {
        for (int i = extents.min.x; i <= extents.max.x; i++) {
          BOOST_CHECK_EQUAL(V3i(i, j, k), V3i(iter.x, iter.y, iter.z));
          BOOST_CHECK_EQUAL(V3i(i, j, k), V3i(citer.x, citer.y, citer.z));
          BOOST_CHECK_EQUAL(sField.value(i, j, k), *iter);
          BOOST_CHECK_EQUAL(sField.value(i, j, k), *citer);
          ++iter;
          ++citer;
        }
      }
    }
  }

  currentTest = "Checking random writes";

  {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    int numWrites = 1000000;
    FIELD3D_RAND48 rng(134664222);
    int size = 200;
    V3i resolution(size);
    SField sField;
    VField vField; 
    sField.setSize(resolution);
    vField.setSize(resolution);
    for (int i = 0; i < numWrites; i++) {
      int x = static_cast<int>(rng.nextf(0, size));
      int y = static_cast<int>(rng.nextf(0, size-1));
      int z = static_cast<int>(rng.nextf(0, size-1));
      sField.lvalue(x, y, z) = rng.nextf();
      vField.lvalue(x, y, z) = Vec3_T(rng.nextf());
    } 
  }

  currentTest = "Checking that mapping picks up resizes";

  {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    int start = 5;
    int end = 50;
    Box3i extents(V3i(start, start, start), V3i(end, end, end));
    Box3i extents2(V3i(start * 2), V3i(end * 2));
    SField sField;
    sField.setSize(extents);
    MatrixFieldMapping::Ptr mapping(new MatrixFieldMapping);
    sField.setMapping(mapping);
    V3d voxelSize(1.0/(end-start+1));
    V3d voxelSize2(1.0/(2*end-2*start+1));
    double size = (voxelSize - sField.mapping()->wsVoxelSize(0,0,0)).length();
    BOOST_CHECK_EQUAL(size < 1e-5, true);
    MatrixFieldMapping before;
    before = *mapping;
    sField.setSize(extents2);
    BOOST_CHECK_EQUAL(before.origin() == sField.mapping()->origin(), false);
    BOOST_CHECK_EQUAL(before.resolution() == sField.mapping()->resolution(), 
                      false);
    BOOST_CHECK_EQUAL((voxelSize2-sField.mapping()->
                         wsVoxelSize(0,0,0)).length()<1e-5, true);
  }

  currentTest = "Checking clone and copy constructors";

  {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    Box3i extents(V3i(-5), V3i(25));
    typename SField::Ptr sField(new SField);
    typename VField::Ptr 
      vField(new VField);
    (*sField).setSize(extents);
    (*vField).setSize(extents);
    // Fill with data
    std::for_each((*sField).begin(), (*sField).end(), 
                  WriteSequence<Data_T>());
    std::for_each((*vField).begin(), (*vField).end(), 
                  WriteSequence<Vec3_T>());
    sField->metadata().setIntMetadata("first",1);
    sField->metadata().setIntMetadata("second",2);

    typename SField::Ptr sfclone = 
      field_dynamic_cast<SField >((*sField).clone());
    BOOST_CHECK_EQUAL(isIdentical<Data_T>(sField, sfclone), true);

    typename VField::Ptr vfclone = 
      field_dynamic_cast<VField >((*vField).clone());
    BOOST_CHECK_EQUAL(isIdentical<Vec3_T>(vField, vfclone), true);

    BOOST_CHECK_EQUAL(sField->metadata().intMetadata("first",-1), 
                      sfclone->metadata().intMetadata("first", -1));
    BOOST_CHECK_EQUAL(sField->metadata().intMetadata("second", -1), 
                      sfclone->metadata().intMetadata("second",-1));
    
    typename SField::Ptr sfcopy = new SField(*sField);
    BOOST_CHECK_EQUAL(isIdentical<Data_T>(sField, sfcopy), true);

    typename VField::Ptr vfcopy = new VField(*vField);
    BOOST_CHECK_EQUAL(isIdentical<Vec3_T>(vField, vfcopy), true);

    BOOST_CHECK_EQUAL(sField->metadata().intMetadata("first",-1), 
                      sfcopy->metadata().intMetadata("first",-1));
    BOOST_CHECK_EQUAL(sField->metadata().intMetadata("second",-1), 
                      sfcopy->metadata().intMetadata("second",-1));

    typename SField::Ptr sfequal(new SField);
    *sfequal = *sField;
    BOOST_CHECK_EQUAL(isIdentical<Data_T>(sField, sfequal), true);

    typename VField::Ptr vfequal(new VField);
    *vfequal = *vField;
    BOOST_CHECK_EQUAL(isIdentical<Vec3_T>(vField, vfequal), true);

    BOOST_CHECK_EQUAL(sField->metadata().intMetadata("first",-1), 
                      sfequal->metadata().intMetadata("first",-1));
    BOOST_CHECK_EQUAL(sField->metadata().intMetadata("second",-1), 
                      sfequal->metadata().intMetadata("second",-1));
  }

}

//----------------------------------------------------------------------------//

template <class Data_T>
void testEmptyField()
{
  typedef FIELD3D_VEC3_T<Data_T> Vec3_T;

  Msg::print("Basic Field tests for type " + 
             string(EmptyField<Data_T>::staticClassType())); 


  string currentTest;

  currentTest = "Checking empty field";

  {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    EmptyField<Data_T> sField;
    EmptyField<Vec3_T>   vField;    
    BOOST_CHECK_EQUAL(sField.dataWindow().hasVolume(), false);
    BOOST_CHECK_EQUAL(vField.dataWindow().hasVolume(), false);   
  }

  currentTest = "Checking non-empty field";

  {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    EmptyField<Data_T> sField;
    EmptyField<Vec3_T> vField;
    sField.setSize(V3i(10));
    vField.setSize(V3i(10));
    BOOST_CHECK_EQUAL(sField.dataWindow().hasVolume(), true);
    BOOST_CHECK_EQUAL(vField.dataWindow().hasVolume(), true);   
  }
  
  currentTest = "Checking value in cleared field";

  {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    Data_T sVal(1.1f);
    Vec3_T vVal(1.2f);
    V3i size(10);
    EmptyField<Data_T> sField;
    EmptyField<Vec3_T> vField;
    sField.setSize(size);
    vField.setSize(size);
    sField.clear(sVal);
    vField.clear(vVal);
    BOOST_CHECK_EQUAL(sField.value(5, 5, 5), sVal);
    BOOST_CHECK_EQUAL(vField.value(5, 5, 5), vVal);   
  }
  
  currentTest = "Checking (bogus) reading and writing a nonexistent cell";

  {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    V3i size(50);
    EmptyField<Data_T> sField;
    EmptyField<Vec3_T> vField;
    sField.setSize(size);
    vField.setSize(size);
    sField.clear(0.0f);
    vField.clear(Vec3_T(0.0f));
    Data_T sVal = static_cast<Data_T>(30.0f);
    Vec3_T vVal = Vec3_T(static_cast<Data_T>(50.0f));
    sField.lvalue(10, 10, 10) = sVal;
    vField.lvalue(10, 10, 10) = vVal;
    BOOST_CHECK_EQUAL(sField.value(10, 10, 10), 0.0f);
    BOOST_CHECK_EQUAL(vField.value(10, 10, 10), Vec3_T(0.0f));
  }

  currentTest = "Checking reading and writing the constant value";

  {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    V3i size(50);
    EmptyField<Data_T> sField;
    EmptyField<Vec3_T> vField;
    sField.setSize(size);
    vField.setSize(size);
    sField.clear(0.0f);
    vField.clear(Vec3_T(0.0f));
    Data_T sVal = static_cast<Data_T>(30.0f);
    Vec3_T vVal = Vec3_T(static_cast<Data_T>(50.0f));
    sField.setConstantvalue(sVal);
    vField.setConstantvalue(vVal);
    BOOST_CHECK_EQUAL(sField.constantvalue(), sVal);
    BOOST_CHECK_EQUAL(vField.constantvalue(), vVal);
  }

  currentTest = "Checking that mapping picks up resizes";

  {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    int start = 5;
    int end = 50;
    Box3i extents(V3i(start, start, start), V3i(end, end, end));
    Box3i extents2(V3i(start * 2), V3i(end * 2));
    EmptyField<Data_T> sField;
    sField.setSize(extents);
    MatrixFieldMapping::Ptr mapping(new MatrixFieldMapping);
    sField.setMapping(mapping);
    MatrixFieldMapping before;
    before = *mapping;
    sField.setSize(extents2);
    BOOST_CHECK_EQUAL(before.origin() == 
                      sField.mapping()->origin(), false);
    BOOST_CHECK_EQUAL(before.resolution() == 
                      sField.mapping()->resolution(), false);
  }

}

//----------------------------------------------------------------------------//

template <class Data_T>
void testFieldMapping()
{
  typedef FIELD3D_VEC3_T<Data_T> Vec3_T;

  Msg::print("FieldMapping tests for type " + 
             string(DenseField<Data_T>::staticClassType()));

  string currentTest;

  currentTest = "Checking field mapping";

  {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    V3i size(5);
    Box3i extents(V3i(-5), V3i(25));

    typename DenseField<Vec3_T>::Ptr vField(new DenseField<Vec3_T>);
    (*vField).setSize(extents);
    (*vField).clear(Vec3_T(1.0f, 0.0f, 0.0f));
    MatrixFieldMapping::Ptr mapping(new MatrixFieldMapping);
    (*vField).setMapping(mapping);
    MatrixFieldMapping::Ptr mapping1 =
      dynamic_pointer_cast<MatrixFieldMapping>(vField->mapping());

    typename DenseField<Vec3_T>::Ptr vField2 =
      field_dynamic_cast<DenseField<Vec3_T> >(vField->clone());

    M44d transform(0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
    MatrixFieldMapping::Ptr mapping2 =
      dynamic_pointer_cast<MatrixFieldMapping>(vField2->mapping());
    mapping2->setLocalToWorld(transform);
    // the matrices should be different, so compare matrices and make
    // sure they don't match
    BOOST_CHECK_EQUAL(mapping1->localToWorld() == 
                      mapping2->localToWorld(), false);

  }

  currentTest = "Checking time-varying MatrixFieldMapping";

  {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    
    M44d sample1;
    sample1.setTranslation(V3f(2.0));
    M44d sample2;
    sample2.setTranslation(V3f(4.0));
    MatrixFieldMapping::Ptr mapping(new MatrixFieldMapping);
    mapping->setLocalToWorld(0.0, sample1);
    mapping->setLocalToWorld(1.0, sample2);

    BOOST_CHECK_EQUAL(mapping->localToWorldSamples().size(), 2);
    BOOST_CHECK_EQUAL(mapping->localToWorldSamples()[0].first, 0.0);
    BOOST_CHECK_EQUAL(mapping->localToWorldSamples()[1].first, 1.0);
  }

  currentTest = "Checking MatrixFieldMapping isIdentical()";

  {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    
    M44d sample1, sample2;
    sample1.setTranslation(V3f(2.0));
    sample2.setTranslation(V3f(4.0));
    MatrixFieldMapping::Ptr mapping1(new MatrixFieldMapping),
      mapping2(new MatrixFieldMapping);
    BOOST_CHECK_EQUAL(mapping1->isIdentical(mapping2), true);
    mapping1->setLocalToWorld(0.0, sample1);
    mapping1->setLocalToWorld(1.0, sample2);
    mapping2->setLocalToWorld(0.0, sample1);
    BOOST_CHECK_EQUAL(mapping1->isIdentical(mapping2), false);
    mapping2->makeIdentity();
    mapping2->setLocalToWorld(0.1, sample1);
    mapping2->setLocalToWorld(1.0, sample2);
    BOOST_CHECK_EQUAL(mapping1->isIdentical(mapping2), false);
    mapping2->makeIdentity();
    mapping2->setLocalToWorld(0.0, sample1);
    mapping2->setLocalToWorld(1.0, sample1);
    BOOST_CHECK_EQUAL(mapping1->isIdentical(mapping2), false);
    mapping2->makeIdentity();
    mapping2->setLocalToWorld(0.0, sample1);
    mapping2->setLocalToWorld(1.0, sample2);
    BOOST_CHECK_EQUAL(mapping1->isIdentical(mapping2), true);
  }

}

//----------------------------------------------------------------------------//

void testFrustumMapping()
{
  Msg::print("FrustumMapping tests");
  ScopedPrintTimer timer;

  int intRes = 64;
  float floatRes = static_cast<float>(intRes);

  DenseField<float>::Ptr field(new DenseField<float>);
  field->setSize(V3i(intRes));

  FrustumFieldMapping::Ptr fm(new FrustumFieldMapping);
  field->setMapping(fm);
  fm = field_dynamic_cast<FrustumFieldMapping>(field->mapping());

  M44d identity;
  identity.makeIdentity();

  BOOST_CHECK_EQUAL(fm->cameraToWorld(), identity);
  BOOST_CHECK(fm->screenToWorldSamples().size() == 1);
  BOOST_CHECK(fm->cameraToWorldSamples().size() == 1);
  BOOST_CHECK(fm->nearPlaneSamples().size() == 1);
  BOOST_CHECK(fm->farPlaneSamples().size() == 1);
  
  V3d wsP(0.0, 0.0, -1.5);
  V3d vsP, lsP;
  
  fm->worldToVoxel(wsP, vsP);
  fm->worldToLocal(wsP, lsP);

  BOOST_CHECK_EQUAL(vsP.x, floatRes / 2.0);
  BOOST_CHECK_EQUAL(vsP.y, floatRes / 2.0);
  BOOST_CHECK(vsP.z > 0.0);
  BOOST_CHECK(vsP.z < floatRes);

  BOOST_CHECK_EQUAL(lsP.x, 0.5);
  BOOST_CHECK_EQUAL(lsP.y, 0.5);
  BOOST_CHECK(lsP.z > 0.0);
  BOOST_CHECK(lsP.z < 1.0);

  // Check uniform slice distribution

  fm->setZDistribution(FrustumFieldMapping::UniformDistribution);
  field->setMapping(fm);
  fm = field_dynamic_cast<FrustumFieldMapping>(field->mapping());

  wsP.setValue(0.0, 0.0, -1.5);
  fm->worldToLocal(wsP, lsP);
  BOOST_CHECK_EQUAL(lsP.z, 0.5);
}

//----------------------------------------------------------------------------//

template <template <typename T> class Field_T, class Data_T>
void testLinearInterp()
{
  typedef FIELD3D_VEC3_T<Data_T> Vec3_T;
  typedef Field_T<Data_T> SField;
  typedef Field_T<Vec3_T> VField;

  Msg::print("Linear interpolation tests for type " +
             string(SField::staticClassType()));

  string currentTest = "Simple linear inter test";

  {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    SField sField;
    sField.setSize(V3i(10), 2);
    LinearFieldInterp<Data_T> lin;
    Box3i bottomSlice(V3i(-2), V3i(0, 9, 9));
    sField.clear(0.0f);
    std::fill(sField.begin(bottomSlice), sField.end(bottomSlice), 1.0f);
    BOOST_CHECK_EQUAL(sField.value(1, 0, 0), 0.0f);
    BOOST_CHECK_EQUAL(sField.value(0, 0, 0), 1.0f);
    BOOST_CHECK_EQUAL(lin.sample(sField, V3d(1.0, 3.1, 2.1)), 0.5f);
    BOOST_CHECK_EQUAL(lin.sample(sField, V3d(1.5, 3.1, 2.1)), 0.0f);
    BOOST_CHECK_EQUAL(lin.sample(sField, V3d(0.5, 3.1, 2.1)), 1.0f);
  }
}

//----------------------------------------------------------------------------//

template <template <typename T> class Field_T, class Data_T>
void testFastLinearInterp()
{
  typedef FIELD3D_VEC3_T<Data_T> Vec3_T;
  typedef Field_T<Data_T> SField;
  typedef Field_T<Vec3_T> VField;

  Msg::print("Linear fast interpolation tests for type " + 
             string(SField::staticClassType()));

  string currentTest = "Simple linear inter test";

  {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    SField sField;
    sField.setSize(V3i(10), 2);
    typename Field_T<Data_T>::LinearInterp lin;
    Box3i bottomSlice(V3i(-2), V3i(0, 9, 9));
    sField.clear(0.0f);
    std::fill(sField.begin(bottomSlice), sField.end(bottomSlice), 1.0f);
    BOOST_CHECK_EQUAL(sField.value(1, 0, 0), 0.0f);
    BOOST_CHECK_EQUAL(sField.value(0, 0, 0), 1.0f);
    BOOST_CHECK_EQUAL(lin.sample(sField, V3d(1.0, 3.1, 2.1)), 0.5f);
    BOOST_CHECK_EQUAL(lin.sample(sField, V3d(1.5, 3.1, 2.1)), 0.0f);
    BOOST_CHECK_EQUAL(lin.sample(sField, V3d(0.5, 3.1, 2.1)), 1.0f);
  }
}

//----------------------------------------------------------------------------//

template <template <typename T> class Field_T, class Data_T>
void testCubicInterp()
{
  typedef FIELD3D_VEC3_T<Data_T> Vec3_T;
  typedef Field_T<Data_T> SField;
  typedef Field_T<Vec3_T> VField;

  Msg::print("Cubic interpolation tests for type " + 
             string(SField::staticClassType()));

  string currentTest = "Simple Cubic inter test";

  {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    typename SField::Ptr sField(new SField);

    sField->setSize(V3i(10), 2);
    CubicFieldInterp<Data_T> cube;
    Box3i bottomSlice(V3i(-2), V3i(0, 9, 9));
    sField->clear(0.0f);
    std::fill(sField->begin(bottomSlice), sField->end(bottomSlice), 1.0f);

    BOOST_CHECK_EQUAL(sField->value(1, 0, 0), 0.0f);
    BOOST_CHECK_EQUAL(sField->value(0, 0, 0), 1.0f);
    BOOST_CHECK_EQUAL(cube.sample(*sField, V3d(1.0, 3.1, 2.1)), 0.5f);
    BOOST_CHECK_EQUAL(cube.sample(*sField, V3d(1.5, 3.1, 2.1)), 0.0f);
    BOOST_CHECK_EQUAL(cube.sample(*sField, V3d(0.5, 3.1, 2.1)), 1.0f);
  }
}

//----------------------------------------------------------------------------//

template <template <typename T> class Field_T, class Data_T>
void testFastCubicInterp()
{
  typedef FIELD3D_VEC3_T<Data_T> Vec3_T;
  typedef Field_T<Data_T> SField;
  typedef Field_T<Vec3_T> VField;

  Msg::print("Cubic fast interpolation tests for type " + 
             string(SField::staticClassType()));

  string currentTest = "Simple Cubic inter test";

  {
    Msg::print(currentTest);
    ScopedPrintTimer t;
    typename SField::Ptr sField(new SField);

    sField->setSize(V3i(10), 2);
    typename Field_T<Data_T>::CubicInterp cube;
    Box3i bottomSlice(V3i(-2), V3i(0, 9, 9));
    sField->clear(0.0f);
    std::fill(sField->begin(bottomSlice), sField->end(bottomSlice), 1.0f);

    BOOST_CHECK_EQUAL(sField->value(1, 0, 0), 0.0f);
    BOOST_CHECK_EQUAL(sField->value(0, 0, 0), 1.0f);
    BOOST_CHECK_EQUAL(cube.sample(*sField, V3d(1.0, 3.1, 2.1)), 0.5f);
    BOOST_CHECK_EQUAL(cube.sample(*sField, V3d(1.5, 3.1, 2.1)), 0.0f);
    BOOST_CHECK_EQUAL(cube.sample(*sField, V3d(0.5, 3.1, 2.1)), 1.0f);
  }
}

//----------------------------------------------------------------------------//

template <template <typename T> class Field_T, class Data_T, bool DoOgawa_T>
void testField3DFile()
{
  typedef FIELD3D_VEC3_T<Data_T> Vec3_T;
  typedef Field_T<Data_T> SField;
  typedef Field_T<Vec3_T> VField;

  Msg::print("Field3DFile tests for type " + 
             string(SField::staticClassType()) + 
             (DoOgawa_T ? " ogawa" : " hdf5"));

  if (DoOgawa_T) {
    Field3DOutputFile::useOgawa(true);
    Field3D::setNumIOThreads(8);
  } else {
    Field3DOutputFile::useOgawa(false);
  }

  string currentTest;

  currentTest = "Checking create bad file fails";

  // This isn't good to test - hdf5 spits out tons of errors
  if (!true) {
    Msg::print(currentTest);
    ScopedPrintTimer t;    
    string filename("/mpp/misspelled.f3d");
    Field3DOutputFile out;
    bool createSuccess = out.create(filename);
    BOOST_CHECK_EQUAL(createSuccess, false);
  }

  currentTest = "Checking write file, then read";

  {
    Msg::print(currentTest);
    string basename = "test_" + string(SField::staticClassType());
    if (DoOgawa_T) {
      basename += "ogawa";
    } else {
      basename += "hdf5";
    }
    string filename(getTempFile(basename + ".f3d"));
    Box3i extents(V3i(0), V3i(160));
    Box3i dataWindow(V3i(20, 10, 0), V3i(100, 100, 100));

    string field1Name("field1");
    string field2Name("field2");
    string field3Name("field3");
    string densityName("density");
    string velName("v");
    string tempName("temperature");

    // Create the scalar field
    typename SField::Ptr sField(new SField);
    sField->setSize(extents, dataWindow);
    MatrixFieldMapping::Ptr mm(new MatrixFieldMapping);
    M44d mtx;
    mtx.setTranslation(Vec3_T(1.1, 2.2, 4.4));
    mm->setLocalToWorld(mtx);
    sField->setMapping(mm);
    sField->clear(1.2);
    sField->metadata().setFloatMetadata("testfloat", 1.0f);
    sField->metadata().setIntMetadata("testint", 2);
    sField->metadata().setStrMetadata("teststring", "3!");

    // Create the vector field
    typename VField::Ptr vField(new VField);
    vField->setSize(extents, dataWindow);
    MatrixFieldMapping::Ptr mapping(new MatrixFieldMapping);
    mapping->setLocalToWorld(mtx);
    vField->setMapping(mapping);
    vField->clear(Vec3_T(0.5));
    vField->metadata().setFloatMetadata("testfloat", 1.0f);
    vField->metadata().setIntMetadata("testint", 2);
    vField->metadata().setStrMetadata("teststring", "3!");

    // Fill scalar fields with all data 
    std::for_each(sField->begin(), sField->end(), WriteSequence<Data_T>());

    // Fill the vector field with a few random data, the rest remains the
    // cleared value
    Box3i toFill(dataWindow);
    toFill.min = V3i(30, 30, 30);
    toFill.max = V3i(49, 49, 40);
    std::for_each(vField->begin(), vField->end(), 
                  WriteSequence<Vec3_T>());

    // Create the output file
    Field3DOutputFile out, outS, outV;
    bool createSuccess = out.create(filename);
    outS.create(filename + ".s");
    outV.create(filename + ".v");
    BOOST_CHECK_EQUAL(createSuccess, true);

    // Write two layers
    bool writeSuccess;
    writeSuccess = out.writeScalarLayer<Data_T>(field1Name, densityName, 
                                                sField);
    outS.writeScalarLayer<Data_T>(field1Name, densityName, sField);
    BOOST_CHECK_EQUAL(writeSuccess, true);
    writeSuccess = out.writeVectorLayer<Data_T>(field2Name, velName, 
                                                vField);
    outV.writeVectorLayer<Data_T>(field2Name, velName, vField);
    BOOST_CHECK_EQUAL(writeSuccess, true);

    /// write out global metadata
    V3f metaVecFloat(1.0,2.0,3.0);
    float metaFloat(123.0);
    V3i metaVecInt(4,5,6);
    int metaInt (456);
    string metaStr("Great Job!!!");
    out.metadata().setVecFloatMetadata("testGlobalMetadataVecFloat", 
                                       metaVecFloat);
    out.metadata().setFloatMetadata("testGlobalMetadataFloat", metaFloat);
    out.metadata().setVecIntMetadata("testGlobalMetadataVecInt", metaVecInt);
    out.metadata().setIntMetadata("testGlobalMetadataInt" , metaInt);
    out.metadata().setStrMetadata("testGlobalMetadataStr", metaStr);

    out.writeGlobalMetadata();

    out.close();
    
    // Open file up again
    Field3DInputFile iFile;
    iFile.open(filename);

    // check the global meta data are correct
    V3f inMetaVecFloat = 
      iFile.metadata().vecFloatMetadata("testGlobalMetadataVecFloat", V3f(9.0));
    BOOST_CHECK_EQUAL(metaVecFloat,inMetaVecFloat);
    float inMetaFloat = 
      iFile.metadata().floatMetadata("testGlobalMetadataFloat", 9.0f);
    BOOST_CHECK_EQUAL(metaFloat,inMetaFloat);
    V3i inMetaVecInt = 
      iFile.metadata().vecIntMetadata("testGlobalMetadataVecInt", V3i(9));
    BOOST_CHECK_EQUAL(metaVecInt,inMetaVecInt);
    int inMetaInt = 
      iFile.metadata().intMetadata("testGlobalMetadataInt", 9); 
    BOOST_CHECK_EQUAL(metaInt,inMetaInt);
    string inMetaStr = 
      iFile.metadata().strMetadata("testGlobalMetadataStr", "Hello");
    BOOST_CHECK_EQUAL(metaStr,inMetaStr);


    // Check that names are correct
    vector<string> partitions, names;
    iFile.getPartitionNames(partitions);
    bool field1InFile = find(partitions.begin(), partitions.end(),
                             field1Name) != partitions.end();
    Msg::print("Checking that partition " + field1Name + " exists in file.");
    bool field2InFile = find(partitions.begin(), partitions.end(),
                             field2Name) != partitions.end();
    Msg::print("Checking that partition " + field2Name + " exists in file.");
    bool field3InFile = find(partitions.begin(), partitions.end(),
                             field3Name) != partitions.end();
    Msg::print("Checking that partition " + field3Name + " exists in file.");
    BOOST_CHECK_EQUAL(field1InFile, true);
    BOOST_CHECK_EQUAL(field2InFile, true);
    BOOST_CHECK_EQUAL(field3InFile, false);
    iFile.getScalarLayerNames(names, field1Name);
    bool densityInFile = find(names.begin(), names.end(),
                              densityName) != names.end();
    names.clear();
    iFile.getVectorLayerNames(names, field2Name);

    bool velInFile = find(names.begin(), names.end(),
                          velName) != names.end();
    bool tempInFile = find(names.begin(), names.end(),
                           tempName) != names.end();

    BOOST_CHECK_EQUAL(densityInFile, true);
    BOOST_CHECK_EQUAL(velInFile, true);    
    BOOST_CHECK_EQUAL(tempInFile, false);

    // Read layers
    typename Field<Data_T>::Vec dOnFile;
    typename Field<Data_T>::Ptr dInMem;
    typename Field<Vec3_T>::Vec vOnFile;
    typename Field<Vec3_T>::Ptr vInMem;
    dInMem = sField;
    vInMem = vField;
    dOnFile = iFile.readScalarLayers<Data_T>(field1Name, densityName);
    vOnFile = iFile.readVectorLayers<Data_T>(field2Name, velName);
    BOOST_CHECK_EQUAL(dOnFile.size(), static_cast<size_t>(1));
    BOOST_CHECK_EQUAL(vOnFile.size(), static_cast<size_t>(1));

    // This check makes the test fail here so we don't get seg faults below
    BOOST_REQUIRE_EQUAL(dOnFile.size() == 1 && vOnFile.size() == 1, true);

    // Check mappings
    BOOST_CHECK_EQUAL(sField->mapping()->isIdentical(dOnFile[0]->mapping()), 
                      true);

    // Grab field data
    typename Field<Data_T>::Ptr s1 = dOnFile[0];
    typename Field<Data_T>::Ptr s2 = dInMem;
    typename Field<Vec3_T>::Ptr v1 = vOnFile[0];
    typename Field<Vec3_T>::Ptr v2 = vInMem;

    // Check scalar layer data
    {
      Msg::print("Verifying scalar data is identical");
      ScopedPrintTimer t;
      BOOST_CHECK_EQUAL(isIdentical<Data_T>(s1, s2), true);
    }

#if 0
    Box3i dw = v1->dataWindow();
    Field<Vec3_T> *p1 = v1.get(), *p2 = v2.get();
    for (int k = dw.min.z; k <= dw.max.z; ++k) {
      for (int j = dw.min.y; j <= dw.max.y; ++j) {
        for (int i = dw.min.x; i <= dw.max.x; ++i) {
          if (p1->value(i, j, k) != p2->value(i, j, k)) {
            cout << "Mismatch voxel: " << V3i(i, j, k) << endl;
          }
        }
      }
    }
#endif

    // Check vector layer data
    {
      Msg::print("Verifying vector data is identical");
      ScopedPrintTimer t;
      BOOST_CHECK_EQUAL(isIdentical<Vec3_T>(v1, v2), true);
    }

    // Check scalar metadata
    {
      Msg::print("Verifying scalar field's metadata is identical");
      const float fMeta  = s2->metadata().floatMetadata("testfloat", 0.0f);
      const int iMeta    = s2->metadata().intMetadata("testint", 0);
      const string sMeta = s2->metadata().strMetadata("teststring", "");
      BOOST_CHECK_EQUAL(fMeta, 1.0f);
      BOOST_CHECK_EQUAL(iMeta, 2);
      BOOST_CHECK_EQUAL(sMeta, "3!");
    }
 
  }

}

//----------------------------------------------------------------------------//

template <class Data_T>
void testEmptySparseFieldToDisk()
{
  string TName(DataTypeTraits<Data_T>::name());
  Msg::print(string("Testing empty sparse field to disk for ") + 
            "<" + TName + ">");

  ScopedPrintTimer t;    

  string filename(getTempFile("test_empty_sparse.f3d"));
  Box3i extents(V3i(0), V3i(160));
  Box3i dataWindow(V3i(20, 10, 0), V3i(200, 200, 200));
  
  string field1Name("field1");
  string densityName("density");
  
  // Create the scalar field
  typename SparseField<Data_T>::Ptr msf(new SparseField<Data_T>);
  BOOST_REQUIRE(msf != NULL);
  msf->setSize(extents, dataWindow);
  msf->clear(1.2);
  
  // Create the output file
  Field3DOutputFile out;
  bool createSuccess = out.create(filename);
  BOOST_CHECK_EQUAL(createSuccess, true);
  
  // Write two layers
  bool writeSuccess;
  writeSuccess = out.writeScalarLayer<Data_T>(field1Name, densityName, msf);
  BOOST_CHECK_EQUAL(writeSuccess, true);
  out.close(); 
}

//----------------------------------------------------------------------------//

template <template <typename T> class Field_T, class Data_T>
void testLayerFetching()
{
  typedef FIELD3D_VEC3_T<Data_T> Vec3_T;
  typedef Field_T<Data_T> SField;
  typedef Field_T<Vec3_T> VField;

  Msg::print("Testing layer fetching for " + 
             string(SField::staticClassType()));

  ScopedPrintTimer t;    

  string filename(getTempFile("testLayerFetching_" + 
                  string(SField::staticClassType()) + ".f3d"));
  Box3i extents(V3i(0), V3i(160));
  Box3i dataWindow(V3i(20, 10, 50), V3i(100, 100, 100));

  typename SField::Ptr sf(new SField);
  typename VField::Ptr vf(new VField);

  sf->setSize(extents, dataWindow);
  vf->setSize(extents, dataWindow);

  // Write the file

  Field3DOutputFile out;
  bool createSuccess = out.create(filename);
  BOOST_CHECK_EQUAL(createSuccess, true);
  
  bool writeSuccess = out.writeScalarLayer<Data_T>("field1", "density", sf);
  BOOST_CHECK_EQUAL(writeSuccess, true);
  writeSuccess = out.writeScalarLayer<Data_T>("field2", "density", sf);
  BOOST_CHECK_EQUAL(writeSuccess, true);
  writeSuccess = out.writeScalarLayer<Data_T>("field3", "density", sf);  
  BOOST_CHECK_EQUAL(writeSuccess, true);
  writeSuccess = out.writeVectorLayer<Data_T>("field1", "v", vf);
  BOOST_CHECK_EQUAL(writeSuccess, true);
  writeSuccess = out.writeVectorLayer<Data_T>("field2", "v", vf);
  BOOST_CHECK_EQUAL(writeSuccess, true);

  out.close();

  // Check the file ---

  Field3DInputFile in;
  in.open(filename);

  typename Field<Data_T>::Vec densityFields;
  densityFields = in.readScalarLayers<Data_T>("density");

  BOOST_CHECK_EQUAL(densityFields.size(), static_cast<size_t>(3));

  typename Field<Vec3_T>::Vec vFields;
  vFields = in.readVectorLayers<Data_T>("v");

  BOOST_CHECK_EQUAL(vFields.size(), static_cast<size_t>(2));

  // Try fetching proxy versions ---

  EmptyField<float>::Vec field1Density = 
    in.readProxyLayer<float>("field1", "density", false);
  EmptyField<float>::Vec field2Density = 
    in.readProxyLayer<float>("field2", "density", false);
  EmptyField<float>::Vec field3Density = 
    in.readProxyLayer<float>("field3", "density", false);
  EmptyField<float>::Vec field1Vel = 
    in.readProxyLayer<float>("field1", "v", true);
  EmptyField<float>::Vec field2Vel = 
    in.readProxyLayer<float>("field2", "v", true);

  BOOST_CHECK(field1Density.size() == static_cast<size_t>(1));
  BOOST_CHECK(field2Density.size() == static_cast<size_t>(1));
  BOOST_CHECK(field3Density.size() == static_cast<size_t>(1));
  BOOST_CHECK(field1Vel.size() == static_cast<size_t>(1));
  BOOST_CHECK(field2Vel.size() == static_cast<size_t>(1));
  
  BOOST_CHECK_EQUAL(field1Density[0]->extents().min, extents.min);
  BOOST_CHECK_EQUAL(field1Density[0]->extents().max, extents.max);
  BOOST_CHECK_EQUAL(field2Density[0]->extents().min, extents.min);
  BOOST_CHECK_EQUAL(field2Density[0]->extents().max, extents.max);
  BOOST_CHECK_EQUAL(field3Density[0]->extents().min, extents.min);
  BOOST_CHECK_EQUAL(field3Density[0]->extents().max, extents.max);
  BOOST_CHECK_EQUAL(field1Vel[0]->extents().min, extents.min);
  BOOST_CHECK_EQUAL(field1Vel[0]->extents().max, extents.max);
  BOOST_CHECK_EQUAL(field2Vel[0]->extents().min, extents.min);
  BOOST_CHECK_EQUAL(field2Vel[0]->extents().max, extents.max);

  BOOST_CHECK_EQUAL(field1Density[0]->dataWindow().min, dataWindow.min);
  BOOST_CHECK_EQUAL(field1Density[0]->dataWindow().max, dataWindow.max);
  BOOST_CHECK_EQUAL(field2Density[0]->dataWindow().min, dataWindow.min);
  BOOST_CHECK_EQUAL(field2Density[0]->dataWindow().max, dataWindow.max);
  BOOST_CHECK_EQUAL(field3Density[0]->dataWindow().min, dataWindow.min);
  BOOST_CHECK_EQUAL(field3Density[0]->dataWindow().max, dataWindow.max);
  BOOST_CHECK_EQUAL(field1Vel[0]->dataWindow().min, dataWindow.min);
  BOOST_CHECK_EQUAL(field1Vel[0]->dataWindow().max, dataWindow.max);
  BOOST_CHECK_EQUAL(field2Vel[0]->dataWindow().min, dataWindow.min);
  BOOST_CHECK_EQUAL(field2Vel[0]->dataWindow().max, dataWindow.max);
}

//----------------------------------------------------------------------------//

template <template <typename T> class Field_T, class Data_T>
void testReadAsDifferentType()
{
  typedef FIELD3D_VEC3_T<Data_T> Vec3_T;
  typedef Field_T<Data_T> SField;
  typedef Field_T<Vec3_T> VField;

  Msg::print("Testing on-the-fly conversion for " +
             string(SField::staticClassType()));

  ScopedPrintTimer t;    

  string filename(getTempFile("testReadAsDifferentType_" + 
                  string(SField::staticClassType()) + ".f3d"));
  Box3i extents(V3i(0), V3i(160));
  Box3i dataWindow(V3i(20, 10, 50), V3i(100, 100, 100));

  MatrixFieldMapping::Ptr mm1(new MatrixFieldMapping);
  MatrixFieldMapping::Ptr mm2(new MatrixFieldMapping);
  M44d mat1;
  M44d mat2;
  mat1.setTranslation(V3d(5.0, 6.0, 7.0));
  mat2.setTranslation(V3d(51.0, 67.0, 7.0));
  mm1->setLocalToWorld(mat1);
  mm2->setLocalToWorld(mat2);
  
  typename SparseField<Data_T>::Ptr sparse(new SparseField<Data_T>);
  sparse->setMapping(mm1);
  sparse->setSize(extents, dataWindow);
  sparse->clear(1.0f);
  sparse->lvalue(dataWindow.min.x, dataWindow.min.y, dataWindow.min.z) = 
    Data_T(1.0);
  typename DenseField<Data_T>::Ptr dense(new DenseField<Data_T>);
  dense->setMapping(mm2);
  dense->setSize(extents, dataWindow);
  dense->clear(2.0f);

  // Write the file

  Field3DOutputFile out;
  bool createSuccess = out.create(filename);
  BOOST_CHECK_EQUAL(createSuccess, true);
  
  bool writeSuccess = 
    out.writeScalarLayer<Data_T>("field1", "density", sparse);
  BOOST_CHECK_EQUAL(writeSuccess, true);
  writeSuccess = out.writeScalarLayer<Data_T>("field2", "density", sparse);
  BOOST_CHECK_EQUAL(writeSuccess, true);
  writeSuccess = out.writeScalarLayer<Data_T>("field3", "density", dense);  
  BOOST_CHECK_EQUAL(writeSuccess, true);

  out.close();

#if 0

  // Read it back out as the templated type

  Field3DInputFile in;
  in.open(filename);

  typename SField::Vec mf1, mf2, mf3;

  mf1 = in.readScalarLayersAs<Field_T, Data_T>("field1", "density");
  mf2 = in.readScalarLayersAs<Field_T, Data_T>("field2", "density");
  mf3 = in.readScalarLayersAs<Field_T, Data_T>("field3", "density");

  BOOST_CHECK_EQUAL(mf1.size(), static_cast<size_t>(1));
  BOOST_CHECK_EQUAL(mf2.size(), static_cast<size_t>(1));
  BOOST_CHECK_EQUAL(mf3.size(), static_cast<size_t>(1));

  // Check that the data is identical
  BOOST_CHECK(isIdentical<Data_T>(mf1[0], sparse));
  BOOST_CHECK(isIdentical<Data_T>(mf2[0], sparse));
  BOOST_CHECK(isIdentical<Data_T>(mf3[0], dense));

  // Mess with the data, and make sure it fails this time
  mf1[0]->lvalue(30, 30, 60) = 9.9;
  BOOST_CHECK_EQUAL(isIdentical<Data_T>(mf1[0], sparse), false);

#endif

}

//----------------------------------------------------------------------------//

void testDiscreteToContinuous()
{
  Msg::print("Testing disc2cont and cont2disc");

  BOOST_CHECK_EQUAL(contToDisc(0.5), 0);
  BOOST_CHECK_EQUAL(contToDisc(0.1), 0);
  BOOST_CHECK_EQUAL(contToDisc(0.9), 0);
  BOOST_CHECK(contToDisc(0.5) != 1);
  BOOST_CHECK_EQUAL(contToDisc(1.0), 1);
  BOOST_CHECK_EQUAL(contToDisc(V2d(0.5, 1.5)), V2i(0, 1));
  BOOST_CHECK_EQUAL(contToDisc(V3d(6.1, 8.5, 9.9)), V3i(6, 8, 9));

  BOOST_CHECK_EQUAL(discToCont(0), 0.5);
  BOOST_CHECK_EQUAL(discToCont(10), 10.5);
  BOOST_CHECK_EQUAL(discToCont(V2i(9, 99)), V2d(9.5, 99.5));
  BOOST_CHECK_EQUAL(discToCont(V3i(1, 9, 99)), V3d(1.5, 9.5, 99.5));
}

//----------------------------------------------------------------------------//

//! \todo Rewrite this for new metadata types
void testFieldMetadata()
{
  Msg::print("Testing meta data container for FieldBase");

#if USE_ANY_METADATA
// not used anymore, leaving the test code in for legacy in case it's
// needed later

  DenseField<float>::Ptr f(new DenseField<float>);

  BOOST_CHECK_EQUAL(f->metaData.size(), static_cast<size_t>(0));
  BOOST_CHECK_EQUAL(f->metaData.empty(), true);

  string doubleAttr("attrib_double");
  string floatAttr("attrib_float");
  string boolAttr("attrib_bool");

  f->metaData[doubleAttr] = boost::any(5.0);
  f->metaData[floatAttr] = boost::any(5.0f);
  f->metaData[boolAttr] = boost::any(true);

  BOOST_CHECK_EQUAL(f->metaData.size(), static_cast<size_t>(3));
  
  try {
    double a = any_cast<double>(f->metaData[doubleAttr]);
    float b = any_cast<float>(f->metaData[floatAttr]);
    bool c = any_cast<bool>(f->metaData[boolAttr]);
  }
  catch (boost::bad_any_cast &e) {
    BOOST_FAIL("Failed on any_cast");
  }

  BOOST_CHECK_THROW(any_cast<double>(f->metaData[floatAttr]), 
                    boost::bad_any_cast);

#endif

}

//----------------------------------------------------------------------------//

void testUnnamedFieldError()
{
  Msg::print("Testing that unnamed fields error out when writing");
  {
    Field3DOutputFile out;
    DenseFieldf::Ptr dense(new DenseFieldf);
    dense->setSize(V3i(50, 50, 50));
    bool success;
    {
      success = out.writeScalarLayer<float>(dense);
    }
    BOOST_CHECK_EQUAL(success, false);
  }
}

//----------------------------------------------------------------------------//

void testBasicFileOpen()
{
  Msg::print("Testing basic Field3DFile open/close");
  {
    Field3DInputFile in;
    BOOST_CHECK_EQUAL(in.open(getTempFile("test_DenseField_float_hdf5.f3d")), 
                      true);
    in.close();
  }
  {
    Field3DInputFile in;
    BOOST_CHECK_EQUAL(in.open(getTempFile("test_DenseField_float_ogawa.f3d")), 
                      true);
    in.close();
  }
}

//----------------------------------------------------------------------------//

template <class Float_T>
void testMACField()
{
  typedef MACField<FIELD3D_VEC3_T<Float_T> > MACField_T;

  Msg::print("Testing MAC Field");
  {
    Float_T clearVal(0.0);
    Float_T uVal(10.0);
    Float_T vVal(20.0);
    Float_T wVal(40.0);

    V3i uCoord(2, 1, 1);
    V3i vCoord(1, 2, 1);
    V3i wCoord(1, 1, 2);

    Float_T uValHalf((uVal + clearVal) / 2.0);
    Float_T vValHalf((vVal + clearVal) / 2.0);
    Float_T wValHalf((wVal + clearVal) / 2.0);

    V3i res(20, 30, 40);

    MACField<FIELD3D_VEC3_T<Float_T> > field;
    field.setSize(res);
    field.clear(FIELD3D_VEC3_T<Float_T>(0.0));
    field.u(uCoord.x, uCoord.y, uCoord.z) = uVal;
    BOOST_CHECK_EQUAL(field.uCenter(1, 1, 1), uValHalf);
    BOOST_CHECK_EQUAL(field.value(1, 1, 1), FIELD3D_VEC3_T<Float_T>(uValHalf, 
                                                          clearVal,
                                                          clearVal));
    field.v(vCoord.x, vCoord.y, vCoord.z) = vVal;
    BOOST_CHECK_EQUAL(field.vCenter(1, 1, 1), vValHalf);
    BOOST_CHECK_EQUAL(field.value(1, 1, 1), FIELD3D_VEC3_T<Float_T>(uValHalf, 
                                                          vValHalf,
                                                          clearVal));
    field.w(wCoord.x, wCoord.y, wCoord.z) = wVal;
    BOOST_CHECK_EQUAL(field.wCenter(1, 1, 1), wValHalf);
    BOOST_CHECK_EQUAL(field.value(1, 1, 1), FIELD3D_VEC3_T<Float_T>(uValHalf, 
                                                          vValHalf,
                                                          wValHalf));

    // Counter used to check the correct # of visited locations
    int count = 0;

    // Check U loop ---

    typename MACField_T::const_mac_comp_iterator u = 
      field.cbegin_comp(MACCompU);
    typename MACField_T::const_mac_comp_iterator uEnd = 
      field.cend_comp(MACCompU);
    for (; u != uEnd; ++u) {
      BOOST_CHECK(*u == static_cast<Float_T>(uVal) || 
                  *u == static_cast<Float_T>(clearVal));
      if (u.x == uCoord.x && u.y == uCoord.y && u.z == uCoord.z) {
        BOOST_CHECK_EQUAL(*u, uVal);
      } else {
        BOOST_CHECK_EQUAL(*u, clearVal);
      }
      count++;
    }

    BOOST_CHECK_EQUAL(count, (res.x + 1) * res.y * res.z);

    // Check V loop ---

    count = 0;
    typename MACField_T::const_mac_comp_iterator v = 
      field.cbegin_comp(MACCompV);
    typename MACField_T::const_mac_comp_iterator vEnd = 
      field.cend_comp(MACCompV);
    for (; v != vEnd; ++v) {
      BOOST_CHECK(*v == static_cast<Float_T>(vVal) || 
                  *v == static_cast<Float_T>(clearVal));
      if (v.x == vCoord.x && v.y == vCoord.y && v.z == vCoord.z) {
        BOOST_CHECK_EQUAL(*v, vVal);
      } else {
        BOOST_CHECK_EQUAL(*v, clearVal);
      }
      count++;
    }

    BOOST_CHECK_EQUAL(count, res.x * (res.y + 1) * res.z);

    // Check W loop ---

    count = 0;
    typename MACField_T::const_mac_comp_iterator w = 
      field.cbegin_comp(MACCompW);
    typename MACField_T::const_mac_comp_iterator wEnd = 
      field.cend_comp(MACCompW);
    for (; w != wEnd; ++w) {
      BOOST_CHECK(*w == static_cast<Float_T>(wVal) || 
                  *w == static_cast<Float_T>(clearVal));
      if (w.x == wCoord.x && w.y == wCoord.y && w.z == wCoord.z) {
        BOOST_CHECK_EQUAL(*w, wVal);
      } else {
        BOOST_CHECK_EQUAL(*w, clearVal);
      }
      count++;
    }

    BOOST_CHECK_EQUAL(count, res.x * res.y * (res.z + 1));

  }


  Msg::print("Testing MAC Field subset iterator");
  {
    V3i res(5, 6, 7);
    MACField<FIELD3D_VEC3_T<Float_T> > field;
    field.setSize(res);

    // neg x face
    {
      field.clear(FIELD3D_VEC3_T<Float_T>(0.0));

      Box3i subset(V3i(0, 0, 0), V3i(0, res.y - 1, res.z - 1));

      typename MACField_T::mac_comp_iterator usub = 
        field.begin_comp(MACCompU, subset);
      typename MACField_T::mac_comp_iterator uEndsub = 
        field.end_comp(MACCompU, subset);
      for (; usub != uEndsub; ++usub) {
        *usub = 1.0;
      }
      typename MACField_T::const_mac_comp_iterator u = 
        field.cbegin_comp(MACCompU);
      typename MACField_T::const_mac_comp_iterator uEnd = 
        field.cend_comp(MACCompU);
      for (; u != uEnd; ++u) {
        if (u.x == 0 || u.x == 1) {
          BOOST_CHECK_EQUAL(*u, 1.0);
        } else {
          BOOST_CHECK_EQUAL(*u, 0.0);
        }
      }
    }

    // pos x face
    {
      field.clear(FIELD3D_VEC3_T<Float_T>(0.0));

      Box3i subset(V3i(res.x - 1, 0, 0), V3i(res.x - 1, res.y - 1, res.z - 1));

      typename MACField_T::mac_comp_iterator usub = 
        field.begin_comp(MACCompU, subset);
      typename MACField_T::mac_comp_iterator uEndsub = 
        field.end_comp(MACCompU, subset);
      for (; usub != uEndsub; ++usub) {
        *usub = 1.0;
      }
      typename MACField_T::const_mac_comp_iterator u = 
        field.cbegin_comp(MACCompU);
      typename MACField_T::const_mac_comp_iterator uEnd = 
        field.cend_comp(MACCompU);
      for (; u != uEnd; ++u) {
        if (u.x == res.x - 1 || u.x == res.x) {
          BOOST_CHECK_EQUAL(*u, 1.0);
        } else {
          BOOST_CHECK_EQUAL(*u, 0.0);
        }
      }
    }

    // neg y face
    {
      field.clear(FIELD3D_VEC3_T<Float_T>(0.0));

      Box3i subset(V3i(0, 0, 0), V3i(res.x - 1, 0, res.z - 1));

      typename MACField_T::mac_comp_iterator vsub = 
        field.begin_comp(MACCompV, subset);
      typename MACField_T::mac_comp_iterator vsubEnd = 
        field.end_comp(MACCompV, subset);
      for (; vsub != vsubEnd; ++vsub) {
        *vsub = 1.0;
      }
      typename MACField_T::const_mac_comp_iterator v = 
        field.cbegin_comp(MACCompV);
      typename MACField_T::const_mac_comp_iterator vEnd = 
        field.cend_comp(MACCompV);
      for (; v != vEnd; ++v) {
        if (v.y == 0 || v.y == 1) {
          BOOST_CHECK_EQUAL(*v, 1.0);
        } else {
          BOOST_CHECK_EQUAL(*v, 0.0);
        }
      }
    }

    // pos y face
    {
      field.clear(FIELD3D_VEC3_T<Float_T>(0.0));

      Box3i subset(V3i(0, res.y - 1, 0), V3i(res.x - 1, res.y - 1, res.z - 1));

      typename MACField_T::mac_comp_iterator vsub = 
        field.begin_comp(MACCompV, subset);
      typename MACField_T::mac_comp_iterator vsubEnd = 
        field.end_comp(MACCompV, subset);
      for (; vsub != vsubEnd; ++vsub) {
        *vsub = 1.0;
      }
      typename MACField_T::const_mac_comp_iterator v = 
        field.cbegin_comp(MACCompV);
      typename MACField_T::const_mac_comp_iterator vEnd = 
        field.cend_comp(MACCompV);
      for (; v != vEnd; ++v) {
        if (v.y == res.y - 1 || v.y == res.y) {
          BOOST_CHECK_EQUAL(*v, 1.0);
        } else {
          BOOST_CHECK_EQUAL(*v, 0.0);
        }
      }
    }

    // neg z face
    {
      field.clear(FIELD3D_VEC3_T<Float_T>(0.0));

      Box3i subset(V3i(0, 0, 0), V3i(res.x - 1, res.y - 1, 0));

      typename MACField_T::mac_comp_iterator wsub = 
        field.begin_comp(MACCompW, subset);
      typename MACField_T::mac_comp_iterator wsubEnd = 
        field.end_comp(MACCompW, subset);
      for (; wsub != wsubEnd; ++wsub) {
        *wsub = 1.0;
      }
      typename MACField_T::const_mac_comp_iterator w = 
        field.cbegin_comp(MACCompW);
      typename MACField_T::const_mac_comp_iterator wEnd = 
        field.cend_comp(MACCompW);
      for (; w != wEnd; ++w) {
        if (w.z == 0 || w.z == 1) {
          BOOST_CHECK_EQUAL(*w, 1.0);
        } else {
          BOOST_CHECK_EQUAL(*w, 0.0);
        }
      }
    }

    // pos z face
    {
      field.clear(FIELD3D_VEC3_T<Float_T>(0.0));

      Box3i subset(V3i(0, 0, res.z - 1), V3i(res.x - 1, res.y - 1, res.z - 1));

      typename MACField_T::mac_comp_iterator wsub = 
        field.begin_comp(MACCompW, subset);
      typename MACField_T::mac_comp_iterator wsubEnd = 
        field.end_comp(MACCompW, subset);
      for (; wsub != wsubEnd; ++wsub) {
        *wsub = 1.0;
      }
      typename MACField_T::const_mac_comp_iterator w = 
        field.cbegin_comp(MACCompW);
      typename MACField_T::const_mac_comp_iterator wEnd = 
        field.cend_comp(MACCompW);
      for (; w != wEnd; ++w) {
        if (w.z == res.z - 1 || w.z == res.z) {
          BOOST_CHECK_EQUAL(*w, 1.0);
        } else {
          BOOST_CHECK_EQUAL(*w, 0.0);
        }
      }
    }
  }

}


//----------------------------------------------------------------------------//

template <class Float_T>
void testEmptyMACFieldToDisk()
{
  typedef MACField<FIELD3D_VEC3_T<Float_T> > MACField_T;

  string TName(DataTypeTraits<Float_T>::name());
  Msg::print(string("Testing empty MAC field to disk for ") + 
            "<" + TName + ">");

  ScopedPrintTimer t;    

  string filename(getTempFile("test_empty_mac.f3d"));
  Box3i extents(V3i(0), V3i(160));
  Box3i dataWindow(V3i(20, 10, 0), V3i(200, 200, 200));
  
  string field1Name("field1");
  string velocityName("v_mac");
  
  // Create the scalar field
  typename MACField_T::Ptr msf(new MACField_T);
  BOOST_REQUIRE(msf != NULL);
  msf->setSize(extents, dataWindow);
  msf->clear(FIELD3D_VEC3_T<Float_T>(1.2));
  
  // Create the output file
  Field3DOutputFile out;
  bool createSuccess = out.create(filename);
  BOOST_CHECK_EQUAL(createSuccess, true);
  
  // Write mac velocity layer
  bool writeSuccess;
  writeSuccess = out.writeVectorLayer<Float_T>(field1Name, velocityName, msf);
  BOOST_CHECK_EQUAL(writeSuccess, true);
  out.close(); 

  // Read the file back in
  Field3DInputFile in;
  BOOST_CHECK_EQUAL(in.open(filename), true);
  typename Field<FIELD3D_VEC3_T<Float_T> >::Vec fields = 
    in.readVectorLayers<Float_T>();
  BOOST_CHECK_EQUAL(fields.size(), 1);
  BOOST_CHECK_EQUAL(fields[0] != NULL, true);
  BOOST_CHECK_EQUAL(field_dynamic_cast<MACField_T>(fields[0]) != NULL, true);
}

//----------------------------------------------------------------------------//

template <class Data_T>
void testSparseFieldBlockAccess()
{
  string TName(DataTypeTraits<Data_T>::name());
  Msg::print("Testing SparseField<" + TName + "> block iterator");
  {
    SparseField<Data_T> field;
    field.setSize(V3i(100, 100, 100));
    
    {
      typename SparseField<Data_T>::block_iterator i = field.blockBegin();
      typename SparseField<Data_T>::block_iterator end = field.blockEnd();
      
      for (; i != end; ++i) {
        bool allocated = field.blockIsAllocated(i.x, i.y, i.z);
        BOOST_CHECK_EQUAL(allocated, false);
      }
    }

    // Write one voxel
    V3i v(1, 1, 1);
    field.lvalue(v.x, v.y, v.z) = static_cast<Data_T>(1.0);
    BOOST_CHECK_EQUAL(field.blockIsAllocated(0, 0, 0), true);
    BOOST_CHECK_EQUAL(field.voxelIsInAllocatedBlock(v.x, v.y, v.z), true);
    BOOST_CHECK_EQUAL(field.blockIsAllocated(0, 0, 1), false);
    
    // Clear out that first block
    Data_T val = static_cast<Data_T>(1.0);
    field.setBlockEmptyValue(0, 0, 0, val);
    BOOST_CHECK_EQUAL(field.getBlockEmptyValue(0, 0, 0), val);
    BOOST_CHECK_EQUAL(field.blockIsAllocated(0, 0, 0), false);
    BOOST_CHECK_EQUAL(field.voxelIsInAllocatedBlock(v.x, v.y, v.z), false);

    // Check the empty value
    BOOST_CHECK_EQUAL(field.value(v.x, v.y, v.z), val);

    // Check the blockIndexIsValid call
    BOOST_CHECK_EQUAL(field.blockIndexIsValid(0, 0, 0), true);
    BOOST_CHECK_EQUAL(field.blockIndexIsValid(8, 0, 0), false);

  }
}

//----------------------------------------------------------------------------//

template <template <typename T> class Field_T, class Data_T>
void testDuplicatePartitions()
{
  typedef FIELD3D_VEC3_T<Data_T> Vec3_T;
  typedef Field_T<Data_T> SField;
  typedef Field_T<Vec3_T> VField;

  Msg::print("Testing duplicate partition names for " + 
             string(SField::staticClassType()));

  string filename(getTempFile("testDuplicatePartitions_" +
                  string(SField::staticClassType()) + ".f3d"));

  Box3i extents(V3i(0), V3i(160));
  Box3i dataWindow(V3i(20, 10, 50), V3i(100, 100, 100));

  typename SField::Ptr sf(new SField);
  typename VField::Ptr vf(new VField);

  sf->setSize(extents, dataWindow);
  vf->setSize(extents, dataWindow);

  MatrixFieldMapping::Ptr mm1(new MatrixFieldMapping);
  MatrixFieldMapping::Ptr mm2(new MatrixFieldMapping);
  MatrixFieldMapping::Ptr mm3(new MatrixFieldMapping);
  M44d mat1, mat2, mat3;
  mat1.setTranslation(V3d(5.0, 6.0, 7.0));
  mat2.setTranslation(V3d(51.0, 67.0, 7.0));
  mm1->setLocalToWorld(mat1);
  mm2->setLocalToWorld(mat2);
  
  sf->setMapping(mm1);
  vf->setMapping(mm1);  

  // Write the file

  Field3DOutputFile out;
  bool createSuccess = out.create(filename);
  BOOST_CHECK_EQUAL(createSuccess, true);
  
  bool writeSuccess = out.writeScalarLayer<Data_T>("default", "density", sf);
  BOOST_CHECK_EQUAL(writeSuccess, true);
  out.writeScalarLayer<Data_T>("", "density", sf);
  BOOST_CHECK_EQUAL(writeSuccess, true);

  // Alter the mapping and add the layer again ---

  sf->setMapping(mm2);  
  vf->setMapping(mm2);  
  writeSuccess = out.writeScalarLayer<Data_T>("default", 
                                              "density_different_mapping", sf);
  BOOST_CHECK_EQUAL(writeSuccess, true);
  writeSuccess = out.writeVectorLayer<Data_T>("default", 
                                              "v_different_mapping", vf);
  BOOST_CHECK_EQUAL(writeSuccess, true);
  writeSuccess = out.writeScalarLayer<Data_T>("", "density_different_mapping", 
                                              sf);
  BOOST_CHECK_EQUAL(writeSuccess, true);
  writeSuccess = out.writeVectorLayer<Data_T>("", "v_different_mapping", vf);
  BOOST_CHECK_EQUAL(writeSuccess, true);
    
  // Alter the mapping again and add the layer again ---

  vf->setMapping(mm3);  
  writeSuccess = out.writeVectorLayer<Data_T>("default", 
                                              "v_second_different_mapping", vf);
  BOOST_CHECK_EQUAL(writeSuccess, true);
  writeSuccess = out.writeVectorLayer<Data_T>("", "v_second_different_mapping", 
                                              vf);
  BOOST_CHECK_EQUAL(writeSuccess, true);
    
  // Double check that partition and layer names are correct ---

  vector<string> partitionNames;
  out.getPartitionNames(partitionNames);

  BOOST_CHECK_EQUAL(partitionNames.size(), static_cast<size_t>(2));

  out.close();

  // Now read it all in again ---

  Field3DInputFile in;
  bool readSuccess = in.open(filename);

  BOOST_CHECK(readSuccess);

  typename Field<Data_T>::Vec scalars;
  typename Field<Vec3_T>::Vec vectors;
  
  scalars = in.readScalarLayers<Data_T>();
  BOOST_CHECK_EQUAL(scalars.size(), static_cast<size_t>(4));
  vectors = in.readVectorLayers<Data_T>();
  BOOST_CHECK_EQUAL(vectors.size(), static_cast<size_t>(4));

}

//----------------------------------------------------------------------------//

void testTimeVaryingMatrixFieldMapping()
{
  Msg::print("Testing time varying MatrixFieldMapping");

  // Create matrix mapping with multiple time samples ---

  MatrixFieldMapping::Ptr mm(new MatrixFieldMapping);

  M44d sample0, sample1, sample2, sample3;
  sample0.setTranslation(V3f(1.0));
  sample1.setTranslation(V3f(2.0));
  sample2.setTranslation(V3f(3.0));
  sample3.setTranslation(V3f(4.0));
  
  mm->setLocalToWorld(-0.5, sample0);
  mm->setLocalToWorld(-0.1, sample1);
  mm->setLocalToWorld(0.1, sample2);
  mm->setLocalToWorld(0.5, sample3);

  // Dummy field. All zeroes.
  DenseField<float>::Ptr field(new DenseField<float>);
  field->name = "test";
  field->attribute = "attr";
  field->setSize(V3i(64));
  field->setMapping(mm);
  mm = field_dynamic_cast<MatrixFieldMapping>(field->mapping());
  BOOST_CHECK(mm);

  // Write to disk
  Field3DOutputFile out;
  string filename(getTempFile("testTimeVaryingMatrixFieldMapping.f3d")); 
  out.create(filename);
  out.writeScalarLayer<float>(field);

  // Read from disk
  Field3DInputFile in;
  in.open(filename);
  Field<float>::Vec fields = in.readScalarLayers<float>();
  BOOST_CHECK_EQUAL(fields.size(), 1);
  
  // Grab mapping
  FieldMapping::Ptr fileMapping = fields[0]->mapping();
  MatrixFieldMapping::Ptr fileMM = field_dynamic_cast<MatrixFieldMapping>(fileMapping);
  BOOST_CHECK(fileMM);
  
  // Check that mappings are identical
  BOOST_CHECK_EQUAL(mm->isIdentical(fileMM, 1e-6), true);
}

//----------------------------------------------------------------------------//

void testTimeVaryingFrustumFieldMapping()
{
  Msg::print("Testing time varying FrustumFieldMapping");

  // Create frustum mapping with multiple time samples ---

  FrustumFieldMapping::Ptr fm(new FrustumFieldMapping);

  Imath::Frustum<double> frustum(0.1, 10.0, 45.0, 0.0, 1.0);
  M44d csToWs0, csToWs1, csToWs2, csToWs3;
  csToWs0.setTranslation(V3d(0.0, 1.0, 0.0));
  csToWs1.setTranslation(V3d(0.0, 2.0, 0.0));
  csToWs2.setTranslation(V3d(0.0, 1.0, 0.0));
  csToWs3.setTranslation(V3d(0.0, -1.0, 0.0));
  M44d ssToCs = frustum.projectionMatrix().inverse();
  M44d ssToWs0 = ssToCs * csToWs0;
  M44d ssToWs1 = ssToCs * csToWs1;
  M44d ssToWs2 = ssToCs * csToWs2;
  M44d ssToWs3 = ssToCs * csToWs3;

  fm->setTransforms(0.0, ssToWs0, csToWs0);
  fm->setTransforms(0.1, ssToWs1, csToWs1);
  fm->setTransforms(0.5, ssToWs2, csToWs2);
  fm->setTransforms(1.0, ssToWs3, csToWs3);

  // Dummy field. All zeroes.
  DenseField<float>::Ptr field(new DenseField<float>);
  field->name = "test";
  field->attribute = "attr";
  field->setSize(V3i(64));
  field->setMapping(fm);
  fm = field_dynamic_cast<FrustumFieldMapping>(field->mapping());
  BOOST_CHECK(fm);

  // Write to disk
  Field3DOutputFile out;
  string filename(getTempFile("testTimeVaryingFrustumFieldMapping.f3d")); 
  out.create(filename);
  out.writeScalarLayer<float>(field);

  // Read from disk
  Field3DInputFile in;
  in.open(filename);
  Field<float>::Vec fields = in.readScalarLayers<float>();
  BOOST_CHECK_EQUAL(fields.size(), 1);
  
  // Grab mapping
  FieldMapping::Ptr fileMapping = fields[0]->mapping();
  FrustumFieldMapping::Ptr fileFM = 
    field_dynamic_cast<FrustumFieldMapping>(fileMapping);
  BOOST_CHECK(fileFM);
  
  // Check that mappings are identical
  BOOST_CHECK_EQUAL(fm->isIdentical(fileFM, 1e-6), true);

  // Check that mappings differ from a default instance
  FrustumFieldMapping::Ptr defaultMapping(new FrustumFieldMapping);
  BOOST_CHECK_EQUAL(defaultMapping->isIdentical(fm, 1e-6), false);
  BOOST_CHECK_EQUAL(defaultMapping->isIdentical(fileFM, 1e-6), false);
}

//----------------------------------------------------------------------------//

// ---------------------------------------------------------------------------//
// Functions -----------------------------------------------------------------//
// ---------------------------------------------------------------------------//

// ---------------------------------------------------------------------------//

template <typename Data_T > 
void createWorkBuffer( 
  typename DenseField<Data_T>::Ptr destPtr, std::vector<Imath::Box3i> &workBuf )
{
  V3i res = destPtr->mapping()->resolution();
  for( int i = 0; i < res.y; ++i ){
    for( int j = 0; j < res.z; ++j ) {
      workBuf.push_back( Imath::Box3i( Imath::V3i( 0,  i, j ), 
                                       Imath::V3i( res.x - 1, i , j ) ) );
      
    }
  }
}

template <typename Data_T> 
void createWorkBuffer( 
  typename SparseField<Data_T>::Ptr destPtr, std::vector<Imath::Box3i> &workBuf )
{
    // for sparse loop over each block to be thread safe.
  typename SparseField<Data_T>::block_iterator i = destPtr->blockBegin();
  typename SparseField<Data_T>::block_iterator end = destPtr->blockEnd();
  for (; i != end; ++i) 
  {
    workBuf.push_back(i.blockBoundingBox());
  }
}



//----------------------------------------------------------------------------//
// Conv class declaration
template <class Field_T>
class CopyThreaded
{
 public:
 
  // Forward declaration -----------------------------------------------------//
  
  //Class for doing all the work
  class Worker;

  // Constructors ------------------------------------------------------------//
   
  // Creates an empty CopyThreaded object
  CopyThreaded()
    :
    m_numWorkThreads(1)
    {}

  // Witch chosen parameters
  CopyThreaded(int inThreads = 8)
    :
    m_numWorkThreads(inThreads)
    {}
  
  // Destructor --------------------------------------------------------------//
  virtual ~CopyThreaded() 
    {}

  // Functions ---------------------------------------------------------------//

  // Performs the copyion of the volume
  bool simpleFunc( typename Field_T::Ptr srcPtr, typename Field_T::Ptr destPtr);

 protected:

  int m_numWorkThreads;        // Number of worker threads;

  boost::mutex m_bufMutex;     // Mutex for reading from buffer

  std::vector<Imath::Box3i> m_workBuf; // Buffer for holding the work data first conv

}; //CopyThreaded class declaration end

//----------------------------------------------------------------------------//
// CopyThreaded::Worker
//----------------------------------------------------------------------------//
 
template <class Field_T>
class CopyThreaded<Field_T>::Worker
{
 public:

  typedef typename Field_T::value_type Data_T;

  // Constructors ------------------------------------------------------------//
  Worker( 
    CopyThreaded<Field_T>* inBoss, 
    typename Field_T::Ptr inSrcPtr, 
    typename Field_T::Ptr inDestPtr)
    : 
      m_boss( inBoss ), 
      m_srcPtr( inSrcPtr ), 
      m_destPtr( inDestPtr )
  { }
  
  // Destructor --------------------------------------------------------------//
  virtual ~Worker() 
  { }

  // Main function -----------------------------------------------------------//
  void operator()()
  {

    //Start execution
    while( true ) { 
      
      // Declaration for the current work segment
      Imath::Box3i workSeg;
      
      // Lock work buffer while reading from it
      {
        boost::mutex::scoped_lock lock( m_boss->m_bufMutex );
	    
        // Get work from work buffer if it is not empty
        if ( !m_boss->m_workBuf.empty() ) {
          // Fetch work
          workSeg = m_boss->m_workBuf.back();
          // Delete work segment
          m_boss->m_workBuf.pop_back();
          
        }
        // Or buffers empty, exit
        else {
          return;
        } 
      } 
      //typename Field_T::CubicInterp interp;
      // Create iterators for volume field
      typename Field_T::const_iterator itSrc = m_srcPtr->cbegin( workSeg );
      // create iterators for the result field
      typename Field_T::iterator itDest = m_destPtr->begin( workSeg );
      typename Field_T::iterator itDestEnd = m_destPtr->end( workSeg );

      // Iterate through each volume element
      for( ; itDest != itDestEnd; ++itSrc, ++itDest) {        
        // Copy through field and get back voxel value
        *itDest = *itSrc;
        //Imath::V3i vsP(itSrc.x, itSrc.y, itSrc.z);
        //*itDest = interp.sample(*m_srcPtr, vsP);
      }
    }
  }


 private:
  
  CopyThreaded<Field_T>* m_boss;     // Pointer to parent boss instance
  typename Field_T::Ptr   m_srcPtr;  // Pointer to the incoming volume
  typename Field_T::Ptr   m_destPtr; // Pointer to the outgoing result/

};


// ---------------------------------------------------------------------------//

// Performs the copyion of the volume
template <class Field_T>
bool CopyThreaded<Field_T>::simpleFunc( 
  typename Field_T::Ptr  srcPtr, 
  typename Field_T::Ptr  destPtr)
{
  using namespace Field3D;
  typedef typename Field_T::value_type Data_T;

  // Resulting densefield pointer pointing to the incoming volume will give undesired result
  if ( srcPtr.get() == destPtr.get() ) {
    Msg::print(
              "Error::CopyThreaded::copy()::"
              "resulting field same as incoming"); // Change these to exceptions?
    return false;
  }

  Imath::Box3i srcExtents = srcPtr->extents();
  Imath::Box3i srcDataWindow = srcPtr->dataWindow();

  // Make sure destPtr is the same size as srcPtr
  V3i srcRes = srcPtr->mapping()->resolution();
  V3i destRes = destPtr->mapping()->resolution();

  if ( srcRes.x != destRes.x ||
       srcRes.y != destRes.y ||
       srcRes.z != destRes.z   ) {

    destPtr->matchDefinition(srcPtr);
  }

  // Downcast to MatrixFieldMapping ptr type to extract tranformation matrices
  MatrixFieldMapping::Ptr srcMapping =
    FIELD_DYNAMIC_CAST<MatrixFieldMapping>( srcPtr->mapping());  


  if (srcMapping == 0){
    Msg::print(
              "Error::CopyThreaded::copy()::"
              "FieldMapping not of matrix type");
    return false;
  }
   

  //Create work buffer for the workers
  createWorkBuffer<Data_T>( destPtr, m_workBuf);
 
  // Thread group for the workers
  boost::thread_group group;
  //Create workers
  for( int i = 0; i < m_numWorkThreads; ++i ) {
    group.create_thread( Worker(this, srcPtr, destPtr) );
  }

  // Make sure every thread is finished before exiting
  group.join_all();

  return true;
}
//----------------------------------------------------------------------------//


template <template <typename T> class Field_T, class Data_T>
void testThreadField()
{

  typedef Field_T<Data_T> SField;
  Box3i extents(V3i(0), V3i(62,128,62));
  //Box3i extents(V3i(0), V3i(128,256,128));
  typename SField::Ptr src = new(SField);
  typename SField::Ptr dest = new(SField);

  float testVal = 1.5f;
  dest->setSize(extents);
  dest->clear(testVal);
  src->setSize(extents);
  src->clear(testVal);
  //intialized src with random value
  {
    V3i srcRes = src->mapping()->resolution();
    Imath::Rand32 rng(0);
    typename SField::iterator i = src->begin();
    typename SField::iterator end = src->end();
    for (; i != end; ++i) {
      float val = rng.nextf();
      *i = val;          
    }  
  }
  
  std::vector<int> numThreads;
  numThreads.push_back(1);
  numThreads.push_back(4);
  numThreads.push_back(8);

 

  for (std::vector<int>::const_iterator i=numThreads.begin(), end=numThreads.end();
       i != end; ++i) 
  {
    {
      Msg::print("Testing " + src->className()+ " with " 
                 + lexical_cast<string>(*i) + " # of threads.");
      ScopedPrintTimer t(*i);
      CopyThreaded<SField> vol( *i );
      vol.simpleFunc(src, dest);
    }
    // verify data
    {
      typename SField::const_iterator itSrc = src->cbegin();
      typename SField::const_iterator itDest = dest->cbegin();
      typename SField::const_iterator itDestEnd = dest->cend();
      // Iterate through each volume element
      for( ; itDest != itDestEnd; ++itSrc, ++itDest) {        
        BOOST_CHECK_EQUAL(*itDest, *itSrc);
      }
      dest->clear(testVal);
    }
  }
}

//----------------------------------------------------------------------------//

template <template <typename T> class MIP_T, 
          template <typename T> class Field_T, 
          class Data_T,
          bool DoOgawa_T>
void testMIPField()
{
  typedef MIP_T<Data_T> FieldType;

  string TName(DataTypeTraits<Data_T>::name());
  Msg::print(string("Testing ") + 
             MIP_T<Data_T>::staticClassType()+ 
             (DoOgawa_T ? " ogawa" : " hdf5"));

  if (DoOgawa_T) {
    Field3DOutputFile::useOgawa(true);
    Field3D::setNumIOThreads(8);
  } else {
    Field3DOutputFile::useOgawa(false);
  }

  typename Field_T<Data_T>::Ptr level0(new Field_T<Data_T>);
  typename Field_T<Data_T>::Ptr level1(new Field_T<Data_T>);
  typename Field_T<Data_T>::Ptr level2(new Field_T<Data_T>);

  level0->setSize(V3i(100));
  level1->setSize(V3i(50));
  level2->setSize(V3i(25));

  level0->clear(1.0);
  level1->clear(0.5);
  level2->clear(0.25);

  std::vector<typename Field_T<Data_T>::Ptr> levels;
  levels.push_back(level0);
  levels.push_back(level1);
  levels.push_back(level2);

  typename MIP_T<Data_T>::Ptr mipField(new MIP_T<Data_T>);
  mipField->setup(levels);
  mipField->name = "mip";
  mipField->attribute = "density";
  
  string basename = "test_" + string(MIP_T<Data_T>::staticClassType());
  if (DoOgawa_T) {
    basename += "ogawa";
  } else {
    basename += "hdf5";
  }
  string filename(getTempFile(basename + ".f3d"));
    
  Field3DOutputFile out;
  out.create(filename);
  out.writeScalarLayer<float>(mipField);
  out.close();

  Field3DInputFile in;
  in.open(filename);

  typename Field<Data_T>::Vec fields = in.readScalarLayers<Data_T>();
  BOOST_CHECK_EQUAL(fields.size(), 1);

  mipField = field_dynamic_cast<MIP_T<Data_T> >(fields[0]);

  BOOST_CHECK_EQUAL(mipField != NULL, true);
  BOOST_CHECK_EQUAL(mipField->numLevels(), 3);
  bool matchLevel0 = isIdentical<Data_T>(mipField->mipLevel(0), level0);
  BOOST_CHECK(matchLevel0);
  bool matchLevel1 = isIdentical<Data_T>(mipField->mipLevel(1), level1);
  BOOST_CHECK(matchLevel1);
  bool matchLevel2 = isIdentical<Data_T>(mipField->mipLevel(2), level2);
  BOOST_CHECK(matchLevel2);
}

//----------------------------------------------------------------------------//

template <template <typename T> class MIP_T, 
          template <typename T> class Field_T, 
          class Data_T>
void testMIPFieldColor()
{
  string TName(DataTypeTraits<Data_T>::name());
  Msg::print(string("Testing ") + MIP_T<Data_T>::staticClassType());

  // typename Field_T<Data_T>::Ptr level0(new Field_T<Data_T>);
  typename Field_T<Data_T>::Ptr level1(new Field_T<Data_T>);
  typename Field_T<Data_T>::Ptr level2(new Field_T<Data_T>);
  typename Field_T<Data_T>::Ptr level3(new Field_T<Data_T>);
  typename Field_T<Data_T>::Ptr level4(new Field_T<Data_T>);
  typename Field_T<Data_T>::Ptr level5(new Field_T<Data_T>);

  // level0->setSize(V3i(400));
  level1->setSize(V3i(200));
  level2->setSize(V3i(100));
  level3->setSize(V3i(50));
  level4->setSize(V3i(25));
  level5->setSize(V3i(13));

  // level0->clear(V3f(1.0, 0.1, 0.1));
  level1->clear(V3f(0.1, 1.0, 0.1));
  level2->clear(V3f(0.1, 0.1, 1.0));
  level3->clear(V3f(1.0, 1.0, 0.1));
  level4->clear(V3f(1.0, 0.1, 1.0));
  level5->clear(V3f(0.1, 1.0, 1.0));

  std::vector<typename Field_T<Data_T>::Ptr> levels;
  // levels.push_back(level0);
  levels.push_back(level1);
  levels.push_back(level2);
  levels.push_back(level3);
  levels.push_back(level4);
  levels.push_back(level5);

  typename MIP_T<Data_T>::Ptr mipField(new MIP_T<Data_T>);
  mipField->setup(levels);
  mipField->name = "mip";
  mipField->attribute = "emission";
  
  Field3DOutputFile out;
  string filename(getTempFile(string(MIP_T<Data_T>::staticClassType()) + ".f3d")); 

  out.create(filename);
  out.writeVectorLayer<float>(mipField);
}

//----------------------------------------------------------------------------//

template <template <typename T> class Field_T, 
          class Data_T>
void testMIPMake()
{
  typedef MIPField<Field_T<Data_T> > MIPType;

  string TName(DataTypeTraits<Data_T>::name());

  typename Field_T<Data_T>::Ptr level0(new Field_T<Data_T>);

  level0->setSize(V3i(200));
  
  int val = 0;
  for (typename Field_T<Data_T>::iterator i = level0->begin(), end = level0->end();
       i != end; ++i) {
    *i = val;
    val = (val + 1) % 128;
  }

  std::vector<int> numThreads;
  numThreads.push_back(1);
  numThreads.push_back(4);
  numThreads.push_back(8);

  for (std::vector<int>::const_iterator i = numThreads.begin(), 
         end = numThreads.end(); i != end; ++i) 
  {
    Msg::print("Testing makeMIP" + 
               string(MIPType::staticClassType()) + " with " 
               + lexical_cast<string>(*i) + " # of threads.");
    typename MIPType::Ptr mipField = makeMIP<MIPType, TriangleFilter>(*level0, 32, *i);
    mipField->name = "mip";
    mipField->attribute = "density";
  
    std::stringstream base;
    base << "testMIPMake_";
    base << string(MIPType::staticClassType()) << "_";
    base << *i << ".f3d";

    Field3DOutputFile out;
    string filename(getTempFile(base.str())); 

    out.create(filename);
    out.writeScalarLayer<Data_T>(mipField);
    out.close();

    Field3DInputFile in;
    in.open(filename);
    typename Field<Data_T>::Vec fields = in.readScalarLayers<Data_T>();
    mipField = field_dynamic_cast<MIPType>(fields[0]);

    BOOST_CHECK_EQUAL(fields.size(), 1);
    BOOST_CHECK_EQUAL(mipField != NULL, true);
    BOOST_CHECK_EQUAL(mipField->numLevels(), 4);
    bool matchLevel0 = isIdentical<Data_T>(mipField->mipLevel(0), level0);
    BOOST_CHECK(matchLevel0);

  }
}

//----------------------------------------------------------------------------//

#define DO_BASIC_TESTS         1
#define DO_INTERP_TESTS        1
#define DO_CUBIC_INTERP_TESTS  1
#define DO_BASIC_FILE_TESTS    1
#define DO_ADVANCED_FILE_TESTS 1
#define DO_SPARSE_BLOCK_TESTS  1
#define DO_MAC_TESTS           1
#define DO_THREAD_TESTS        1
#define DO_MIP_TESTS           1

test_suite*
init_unit_test_suite(int argc, char* argv[])
{
  typedef Field3D::half half;

  initIO();

  test_suite* test = BOOST_TEST_SUITE("Field3D Test Suite");
  
  test->add(BOOST_TEST_CASE(&testDiscreteToContinuous));
  test->add(BOOST_TEST_CASE(&testFieldMetadata));

#if DO_BASIC_TESTS

  test->add(BOOST_TEST_CASE((&testBasicField<DenseField, half>)));
  test->add(BOOST_TEST_CASE((&testBasicField<SparseField, half>)));
  test->add(BOOST_TEST_CASE((&testBasicField<DenseField, float>)));
  test->add(BOOST_TEST_CASE((&testBasicField<SparseField, float>)));
  test->add(BOOST_TEST_CASE((&testBasicField<DenseField, double>)));
  test->add(BOOST_TEST_CASE((&testBasicField<SparseField, double>)));

  // EmptyField needs to a different function because its lvalue() is
  // deliberately disabled and replaced with constantvalue() and
  // setConstantvalue()
  test->add(BOOST_TEST_CASE((&testEmptyField<half>)));
  test->add(BOOST_TEST_CASE((&testEmptyField<float>)));
  test->add(BOOST_TEST_CASE((&testEmptyField<double>)));

  // tests to make sure the field clone properly makes a clone of the
  // mapping, so the old mapping will be preserved
  test->add(BOOST_TEST_CASE((&testFieldMapping<float>)));
  test->add(BOOST_TEST_CASE((&testFieldMapping<double>)));

  test->add(BOOST_TEST_CASE((&testFrustumMapping)));

#endif

#if DO_INTERP_TESTS

  test->add(BOOST_TEST_CASE((&testLinearInterp<DenseField, half>)));
  test->add(BOOST_TEST_CASE((&testLinearInterp<SparseField, half>)));
  test->add(BOOST_TEST_CASE((&testLinearInterp<DenseField, float>)));
  test->add(BOOST_TEST_CASE((&testLinearInterp<SparseField, float>)));
  test->add(BOOST_TEST_CASE((&testLinearInterp<DenseField, double>)));
  test->add(BOOST_TEST_CASE((&testLinearInterp<SparseField, double>)));

  test->add(BOOST_TEST_CASE((&testFastLinearInterp<DenseField, half>)));
  test->add(BOOST_TEST_CASE((&testFastLinearInterp<SparseField, half>)));
  test->add(BOOST_TEST_CASE((&testFastLinearInterp<DenseField, float>)));
  test->add(BOOST_TEST_CASE((&testFastLinearInterp<SparseField, float>)));
  test->add(BOOST_TEST_CASE((&testFastLinearInterp<DenseField, double>)));
  test->add(BOOST_TEST_CASE((&testFastLinearInterp<SparseField, double>)));

#endif

#if DO_CUBIC_INTERP_TESTS

  test->add(BOOST_TEST_CASE((&testCubicInterp<DenseField, half>)));
  test->add(BOOST_TEST_CASE((&testCubicInterp<SparseField, half>)));
  test->add(BOOST_TEST_CASE((&testCubicInterp<DenseField, float>)));
  test->add(BOOST_TEST_CASE((&testCubicInterp<SparseField, float>)));
  test->add(BOOST_TEST_CASE((&testCubicInterp<DenseField, double>)));
  test->add(BOOST_TEST_CASE((&testCubicInterp<SparseField, double>)));

  test->add(BOOST_TEST_CASE((&testFastCubicInterp<DenseField, half>)));
  test->add(BOOST_TEST_CASE((&testFastCubicInterp<SparseField, half>)));
  test->add(BOOST_TEST_CASE((&testFastCubicInterp<DenseField, float>)));
  test->add(BOOST_TEST_CASE((&testFastCubicInterp<SparseField, float>)));
  test->add(BOOST_TEST_CASE((&testFastCubicInterp<DenseField, double>)));
  test->add(BOOST_TEST_CASE((&testFastCubicInterp<SparseField, double>)));

#endif

#if DO_BASIC_FILE_TESTS

  test->add(BOOST_TEST_CASE((&testField3DFile<DenseField, half, true>)));
  test->add(BOOST_TEST_CASE((&testField3DFile<SparseField, half, true>)));
  test->add(BOOST_TEST_CASE((&testField3DFile<DenseField, float, true>)));
  test->add(BOOST_TEST_CASE((&testField3DFile<SparseField, float, true>)));
  test->add(BOOST_TEST_CASE((&testField3DFile<DenseField, double, true>)));
  test->add(BOOST_TEST_CASE((&testField3DFile<SparseField, double, true>)));
  test->add(BOOST_TEST_CASE((&testField3DFile<DenseField, half, false>)));
  test->add(BOOST_TEST_CASE((&testField3DFile<SparseField, half, false>)));
  test->add(BOOST_TEST_CASE((&testField3DFile<DenseField, float, false>)));
  test->add(BOOST_TEST_CASE((&testField3DFile<SparseField, float, false>)));
  test->add(BOOST_TEST_CASE((&testField3DFile<DenseField, double, false>)));
  test->add(BOOST_TEST_CASE((&testField3DFile<SparseField, double, false>)));
  test->add(BOOST_TEST_CASE(&testUnnamedFieldError));
  test->add(BOOST_TEST_CASE(&testBasicFileOpen));

#endif

#if DO_ADVANCED_FILE_TESTS

  test->add(BOOST_TEST_CASE((&testEmptySparseFieldToDisk<half>)));
  test->add(BOOST_TEST_CASE((&testEmptySparseFieldToDisk<float>)));
  test->add(BOOST_TEST_CASE((&testEmptySparseFieldToDisk<double>)));

  test->add(BOOST_TEST_CASE((&testEmptyMACFieldToDisk<half>)));
  test->add(BOOST_TEST_CASE((&testEmptyMACFieldToDisk<float>)));
  test->add(BOOST_TEST_CASE((&testEmptyMACFieldToDisk<double>)));

  test->add(BOOST_TEST_CASE((&testLayerFetching<DenseField, half>)));
  test->add(BOOST_TEST_CASE((&testLayerFetching<SparseField, half>)));
  test->add(BOOST_TEST_CASE((&testLayerFetching<DenseField, float>)));
  test->add(BOOST_TEST_CASE((&testLayerFetching<SparseField, float>)));
  test->add(BOOST_TEST_CASE((&testLayerFetching<DenseField, double>)));
  test->add(BOOST_TEST_CASE((&testLayerFetching<SparseField, double>)));

  test->add(BOOST_TEST_CASE((&testReadAsDifferentType<DenseField, half>)));
  test->add(BOOST_TEST_CASE((&testReadAsDifferentType<SparseField, half>)));
  test->add(BOOST_TEST_CASE((&testReadAsDifferentType<DenseField, float>)));
  test->add(BOOST_TEST_CASE((&testReadAsDifferentType<SparseField, float>)));
  test->add(BOOST_TEST_CASE((&testReadAsDifferentType<DenseField, double>)));
  test->add(BOOST_TEST_CASE((&testReadAsDifferentType<SparseField, double>)));

  test->add(BOOST_TEST_CASE((&testDuplicatePartitions<DenseField, half>)));
  test->add(BOOST_TEST_CASE((&testDuplicatePartitions<SparseField, half>)));
  test->add(BOOST_TEST_CASE((&testDuplicatePartitions<DenseField, float>)));
  test->add(BOOST_TEST_CASE((&testDuplicatePartitions<SparseField, float>)));
  test->add(BOOST_TEST_CASE((&testDuplicatePartitions<DenseField, double>)));
  test->add(BOOST_TEST_CASE((&testDuplicatePartitions<SparseField, double>)));

  test->add(BOOST_TEST_CASE((&testTimeVaryingMatrixFieldMapping)));
  test->add(BOOST_TEST_CASE((&testTimeVaryingFrustumFieldMapping)));

#endif

#if DO_SPARSE_BLOCK_TESTS
  test->add(BOOST_TEST_CASE((&testSparseFieldBlockAccess<half>)));
  test->add(BOOST_TEST_CASE((&testSparseFieldBlockAccess<float>)));
  test->add(BOOST_TEST_CASE((&testSparseFieldBlockAccess<double>)));
#endif

#if DO_MAC_TESTS
  test->add(BOOST_TEST_CASE((&testMACField<float>)));
#endif


#if DO_THREAD_TESTS
  test->add(BOOST_TEST_CASE((&testThreadField<DenseField, float>)));
  test->add(BOOST_TEST_CASE((&testThreadField<SparseField, float>)));
#endif 

#if DO_MIP_TESTS
  test->add(BOOST_TEST_CASE((&testMIPField<MIPDenseField, DenseField, float, true>)));
  test->add(BOOST_TEST_CASE((&testMIPField<MIPDenseField, DenseField, float, false>)));
  test->add(BOOST_TEST_CASE((&testMIPFieldColor<MIPDenseField, DenseField, V3f>)));
  test->add(BOOST_TEST_CASE((&testMIPField<MIPSparseField, SparseField, float, true>)));
  test->add(BOOST_TEST_CASE((&testMIPField<MIPSparseField, SparseField, float, false>)));
  test->add(BOOST_TEST_CASE((&testMIPFieldColor<MIPSparseField, SparseField, V3f>)));
  test->add(BOOST_TEST_CASE((&testMIPMake<DenseField, float>)));
  test->add(BOOST_TEST_CASE((&testMIPMake<SparseField, float>)));
#endif

  return test;
}

