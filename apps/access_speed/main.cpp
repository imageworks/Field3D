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

#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/timer.hpp>

#include <Field3D/DenseField.h>
#include <Field3D/SparseField.h>
#include <Field3D/Log.h>

//----------------------------------------------------------------------------//

using namespace boost;
using namespace std;

using namespace Field3D;

//----------------------------------------------------------------------------//

//! Converts any class with operator<< to a string using boost::lexical_cast
template <class T>
std::string str(const T& t)
{
  return boost::lexical_cast<std::string>(t);
}

//----------------------------------------------------------------------------//

template <class Field_T>
Field3D::V3f testDenseSpeed(const Field3D::V3i &res)
{
  boost::timer tResize;
  Field_T field;
  field.setSize(res);
  float resizeTime = tResize.elapsed();

  boost::timer tWrite;
  typename Field_T::iterator i = field.begin(), end = field.end();
  float count = 0.0f;
  for (; i != end; ++i) {
    *i = count;
    count += 1.0f;
  }
  float writeTime = tWrite.elapsed();

  boost::timer tRead;
  typename Field_T::const_iterator ci = field.cbegin(), cend = field.cend();
  float tally = 0.0f;
  for (; ci != cend; ++ci) {
    tally += *ci;
  }
  float readTime = tRead.elapsed();

  Msg::print("    Resize time       : " + str(resizeTime));
  Msg::print("    Write time        : " + str(writeTime));
  Msg::print("    Read time         : " + str(readTime));

  return V3f(resizeTime, writeTime, readTime);
}

//----------------------------------------------------------------------------//

template <class Data_T>
Field3D::V3f testStdVector(const Field3D::V3i &res)
{
  typedef std::vector<Data_T> Vec;

  size_t voxels = res.x * res.y * res.z;

  boost::timer tResize;
  Vec vec;
  vec.resize(voxels);
  float resizeTime = tResize.elapsed();
  
  boost::timer tWrite;
  typename Vec::iterator i = vec.begin(), end = vec.end();
  float count = 0.0f;
  for (; i != end; ++i) {
    *i = count;
    count += 1.0f;
  }
  float writeTime = tWrite.elapsed();

  boost::timer tRead;
  typename Vec::const_iterator ci = vec.begin(), cend = vec.end();
  float tally = 0.0f;
  for (; ci != cend; ++ci) {
    tally += *ci;
  }
  float readTime = tRead.elapsed();

  Msg::print("    Resize time       : " + str(resizeTime));
  Msg::print("    Write time        : " + str(writeTime));
  Msg::print("    Read time         : " + str(readTime));

  return V3f(resizeTime, writeTime, readTime);  
}

//----------------------------------------------------------------------------//

template <class Data_T>
Field3D::V3f testDenseSparseSpeed(const Field3D::V3i &res)
{
  typedef SparseField<Data_T> Field_T;

  boost::timer tResize;
  SparseField<Data_T> field;
  field.setSize(res);
  float resizeTime = tResize.elapsed();

  boost::timer tWrite;
  typename SparseField<Data_T>::block_iterator bi = field.blockBegin();
  typename SparseField<Data_T>::block_iterator bend = field.blockEnd();
  float count = 0.0f;
  for (; bi != bend; ++bi) {
    Box3i subset = bi.blockBoundingBox();
    typename Field_T::iterator i = field.begin(subset), end = field.end(subset);
    for (; i != end; ++i) {
      *i = count;
      count += 1.0f;
    }
  }
  float writeTime = tWrite.elapsed();

  boost::timer tWrite2;
  typename SparseField<Data_T>::block_iterator bi2 = field.blockBegin();
  typename SparseField<Data_T>::block_iterator bend2 = field.blockEnd();
  count = 0.0f;
  for (; bi2 != bend2; ++bi2) {
    Box3i subset = bi2.blockBoundingBox();
    typename Field_T::iterator i = field.begin(subset), end = field.end(subset);
    for (; i != end; ++i) {
      *i = count;
      count += 1.0f;
    }
  }
  float writeTime2 = tWrite2.elapsed();

  boost::timer tRead;
  typename SparseField<Data_T>::block_iterator bi3 = field.blockBegin();
  typename SparseField<Data_T>::block_iterator bend3 = field.blockEnd();
  float tally = 0.0f;
  for (; bi3 != bend3; ++bi3) {
    Box3i subset = bi3.blockBoundingBox();
    typename Field_T::const_iterator ci = field.cbegin(subset), 
      cend = field.cend(subset);
    for (; ci != cend; ++ci) {
      tally += *ci;
    }
  }
  float readTime = tRead.elapsed();

  Msg::print("    Resize time       : " + str(resizeTime));
  Msg::print("    Write time        : " + str(writeTime));
  Msg::print("    Second write time : " + str(writeTime2));
  Msg::print("    Read time         : " + str(readTime));  

  return V3f(resizeTime, writeTime2, readTime);
}

//----------------------------------------------------------------------------//

int main(int argc, char **argv) 
{
  Field3D::V3i res(256);

  if (argc == 2) {
    try {
      int xRes = lexical_cast<int>(argv[1]);
      res = V3i(xRes);
    } 
    catch (boost::bad_lexical_cast &e) {
      Msg::print("Couldn't parse integer resolution. Aborting");
      exit(1);
    }
  }
  else if (argc >= 4) {
    int xRes = lexical_cast<int>(argv[1]);
    int yRes = lexical_cast<int>(argv[2]);
    int zRes = lexical_cast<int>(argv[3]);
    res = V3i(xRes, yRes, zRes);
  } else {
    // No voxel res given
    Msg::print("Usage: " + lexical_cast<string>(argv[0]) + 
               " <xres> <yres> <zres>");
    Msg::print("Got no voxel size. Using default.");
  }

  Msg::print("\nTesting lookup speed with resolution: " + str(res) + "\n");

  Msg::print("Test 1: Write and read all voxels");

  Msg::print("  std::vector<float>: ");
  V3f svf = testStdVector<float>(res);

  Msg::print("  DenseField<float>: ");
  V3f df = testDenseSpeed<DenseFieldf>(res);

  Msg::print("  SparseField<float>: ");
  V3f sf = testDenseSpeed<SparseFieldf>(res);

  Msg::print("  SparseField<float> with block_iterator: ");
  V3f sof = testDenseSparseSpeed<float>(res);

  Msg::print("  Reading StdVec<->Dense ratio         : " + str(df.z / svf.z));
  Msg::print("  Reading Dense<->Sparse ratio         : " + str(sf.z / df.z));
  Msg::print("  Reading Dense<->Sparse (opt) ratio   : " + str(sof.z / df.z));
  Msg::print("  Writing StdVec<->Dense ratio         : " + str(df.y / svf.y));
  Msg::print("  Writing Dense<->Sparse ratio         : " + str(sf.y / df.y));
  Msg::print("  Writing Dense<->Sparse (opt) ratio   : " + str(sof.y / df.y));
  
}

//----------------------------------------------------------------------------//
