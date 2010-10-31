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
#include <boost/thread/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <Field3D/DenseField.h>
#include <Field3D/FieldMapping.h>
#include <Field3D/Log.h>

//----------------------------------------------------------------------------//

using namespace boost;
using namespace boost::posix_time;
using namespace std;

using namespace Field3D;

//----------------------------------------------------------------------------//

typedef std::vector<Field3D::V3d> V3dVec;

//----------------------------------------------------------------------------//

struct WallClockTimer
{
  WallClockTimer()
    : m_start(microsec_clock::local_time())
  { /* Empty */ }
  float elapsed() const
  {
    ptime now = microsec_clock::local_time();
    time_duration diff = now - m_start;
    return diff.total_milliseconds() / 1000.0;
  }
private:
  ptime m_start;
};

//----------------------------------------------------------------------------//

struct NaiveThreadWorker
{
  NaiveThreadWorker(Field<float>::Ptr field,
                    const V3dVec &wsP, V3dVec &vsP,
                    size_t startIdx, size_t endIdx)
    : m_field(field), m_wsP(wsP), m_vsP(vsP), 
      m_startIdx(startIdx), m_endIdx(endIdx)
  {
    // Empty
  }
  void operator()()
  {
    V3dVec::const_iterator wsI = m_wsP.begin();
    V3dVec::const_iterator wsEnd = m_wsP.begin();
    V3dVec::iterator vsI = m_vsP.begin();

    advance(wsI, m_startIdx);
    advance(wsEnd, m_endIdx);
    advance(vsI, m_startIdx);

    for (; wsI != wsEnd; ++wsI, ++vsI) {
      m_field->mapping()->worldToLocal(*wsI, *vsI);
    }
  }
private:
  Field<float>::Ptr m_field;
  const V3dVec &m_wsP;
  V3dVec &m_vsP;
  size_t m_startIdx;
  size_t m_endIdx;
};

//----------------------------------------------------------------------------//

struct IteratorThreadWorker
{
  IteratorThreadWorker(Field<float>::Ptr field,
                       const V3dVec &wsP, V3dVec &vsP,
                       size_t startIdx, size_t endIdx)
    : m_field(field), m_wsP(wsP), m_vsP(vsP), 
      m_startIdx(startIdx), m_endIdx(endIdx)
  {
    // Empty
  }
  void operator()()
  {
    V3dVec::const_iterator wsI = m_wsP.begin();
    V3dVec::const_iterator wsEnd = m_wsP.begin();
    V3dVec::iterator vsI = m_vsP.begin();

    advance(wsI, m_startIdx);
    advance(wsEnd, m_endIdx);
    advance(vsI, m_startIdx);

    m_field->mapping()->worldToLocal(wsI, wsEnd, vsI);
  }
private:
  Field<float>::Ptr m_field;
  const V3dVec &m_wsP;
  V3dVec &m_vsP;
  size_t m_startIdx;
  size_t m_endIdx;
};

//----------------------------------------------------------------------------//

bool checkEqual(const V3dVec &wsP, const V3dVec &vsP)
{
  V3dVec::const_iterator wsI = wsP.begin();
  V3dVec::const_iterator wsEnd = wsP.end();
  V3dVec::const_iterator vsI = vsP.begin();
  
  for (; wsI != wsEnd; ++wsI, ++vsI) {
    if (*wsI != *vsI) {
      Msg::print("  Comparing " + lexical_cast<string>(*wsI) + " " +
                 lexical_cast<string>(*vsI));
      return false;
    }
  }

  return true;
}

//----------------------------------------------------------------------------//

int main(int argc, char **argv) 
{
  // Handle arguments ---

  size_t numThreads = 8;
  size_t numPoints = 5000000;

  if (argc == 2) {
    try {
      numThreads = lexical_cast<size_t>(argv[1]);
    } 
    catch (boost::bad_lexical_cast &e) {
      Msg::print("Couldn't parse num_threads. Aborting");
      exit(1);
    }
  }
  else if (argc == 3) {
    try {
      numThreads = lexical_cast<size_t>(argv[1]);
      numPoints = lexical_cast<size_t>(argv[2]);
    }
    catch (boost::bad_lexical_cast &e) {
      Msg::print("Couldn't parse arguments. Aborting");
      exit(1);
    }
  } else {
    Msg::print("Usage: " + lexical_cast<string>(argv[0]) + 
               " <num_threads> [num_transforms]");
    Msg::print("Got no arguments. Using defaults.");
  }

  Msg::print("Using " + 
             lexical_cast<string>(numThreads) + " threads, " + 
             lexical_cast<string>(numPoints) + " points");

  // Set up test data ---

  DenseFieldf::Ptr field(new DenseFieldf);

  M44d mat;
  mat.makeIdentity();

  MatrixFieldMapping::Ptr mapping(new MatrixFieldMapping);
  mapping->setLocalToWorld(mat);

  field->setMapping(mapping);

  FIELD3D_RAND48 rng(1);
  V3dVec wsP(numPoints), vsP(numPoints);

  for (V3dVec::iterator wsI = wsP.begin(); wsI != wsP.end(); ++wsI) {
    wsI->x = rng.nextf();
    wsI->y = rng.nextf();
    wsI->z = rng.nextf();
  }
  
  // Test single threaded ---

  Msg::print("Testing naive single threaded version");
  
  {
    WallClockTimer t;

    V3dVec::const_iterator wsI = wsP.begin();
    V3dVec::const_iterator wsEnd = wsP.end();
    V3dVec::iterator vsI = vsP.begin();
      
    for (; wsI != wsEnd; ++wsI, ++vsI) {
      field->mapping()->worldToLocal(*wsI, *vsI);
    }

    Msg::print("  " + lexical_cast<string>(t.elapsed())); 
  }

  if (!checkEqual(wsP, vsP)) {
    Msg::print("  FAILED!");
  }
  std::fill(vsP.begin(), vsP.end(), V3d(0.0));

  // Test iterator transform single threaded ---

  Msg::print("Testing iterator-based single threaded version");
  
  {
    WallClockTimer t;

    V3dVec::const_iterator wsI = wsP.begin();
    V3dVec::const_iterator wsEnd = wsP.end();
    V3dVec::iterator vsI = vsP.begin();
      
    field->mapping()->worldToLocal(wsP.begin(), wsP.end(), vsP.begin());

    Msg::print("  " + lexical_cast<string>(t.elapsed())); 
  }

  if (!checkEqual(wsP, vsP)) {
    Msg::print("  FAILED!");
  }
  std::fill(vsP.begin(), vsP.end(), V3d(0.0));

  // Set up threads ---

  size_t itemsPerBatch = numPoints / numThreads;

  // Test naive multithreaded ---

  Msg::print("Testing naive multithreaded version");

  {
    WallClockTimer t;

    boost::thread_group threads;
    size_t nextStartIdx = 0;

    while (nextStartIdx < numPoints) {
      int endIdx = std::min(nextStartIdx + itemsPerBatch, numPoints);
      threads.create_thread(NaiveThreadWorker(field, wsP, vsP, 
                                              nextStartIdx, endIdx));
      nextStartIdx += itemsPerBatch;
      numThreads++;
    }      
    
    threads.join_all();
    Msg::print("  " + lexical_cast<string>(t.elapsed())); 
  }

  if (!checkEqual(wsP, vsP)) {
    Msg::print("  FAILED!");
  }
  std::fill(vsP.begin(), vsP.end(), V3d(0.0));

  // Test multithreaded with optimized transform ---

  Msg::print("Testing iterator multithreaded version");

  {
    WallClockTimer t;

    boost::thread_group threads;
    size_t nextStartIdx = 0;

    while (nextStartIdx < numPoints) {
      int endIdx = std::min(nextStartIdx + itemsPerBatch, numPoints);
      threads.create_thread(IteratorThreadWorker(field, wsP, vsP, 
                                                 nextStartIdx, endIdx));
      nextStartIdx += itemsPerBatch;
      numThreads++;
    }      
    
    threads.join_all();
    Msg::print("  " + lexical_cast<string>(t.elapsed())); 
  }

  if (!checkEqual(wsP, vsP)) {
    Msg::print("  FAILED!");
  }
  std::fill(vsP.begin(), vsP.end(), V3d(0.0));

}

//----------------------------------------------------------------------------//
