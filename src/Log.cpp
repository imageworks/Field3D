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

/*! \file Log.cpp
  \brief Contains implementations of the logging-related functions.
*/

//----------------------------------------------------------------------------//

#include <unistd.h>
#include <ios>
#include <fstream>

#include <iostream>

#include "Log.h"

//----------------------------------------------------------------------------//

using namespace std;

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//

namespace Msg {

//----------------------------------------------------------------------------//

static int g_verbosity = 1;

//----------------------------------------------------------------------------//

void print(Severity severity, const std::string &message)
{
  if (g_verbosity < 1)
      return;

  switch(severity) {
  case SevWarning:
    cout << "WARNING: ";
    break;
  case SevMessage:
  default:
    break;
    // Do nothing
  }

  cout << message << endl;
}

//----------------------------------------------------------------------------//

void setVerbosity (int level)
{
  g_verbosity = level;
}

//----------------------------------------------------------------------------//

} // namespace Log

//----------------------------------------------------------------------------//

std::string bytesToString(int64_t bytes)
{
  using std::stringstream;

  stringstream ss;
  ss.precision(3);
  ss.setf(std::ios::fixed, std:: ios::floatfield);

  // Make it work for negative numbers
  if (bytes < 0) {
    ss << "-";
    bytes = -bytes;
  }

  if (bytes < 1024) {
    // Bytes
    ss << bytes << "  B";
    return ss.str();
  } else if (bytes < (1024 * 1024)) {
    // Kilobytes
    ss << bytes / static_cast<float>(1024) << " KB";
    return ss.str();
  } else if (bytes < (1024 * 1024 * 1024)) {
    // Megabytes
    ss << bytes / static_cast<float>(1024 * 1024) << " MB";
    return ss.str();
  } else {
    // Gigabytes
    ss << bytes / static_cast<float>(1024 * 1024 * 1024) << " GB";
    return ss.str();
  }
}

//----------------------------------------------------------------------------//

size_t currentRSS()
{
  //! Only implemented for Linux at the moment.

#ifdef __linux__

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
              >> O >> itrealvalue >> starttime 
              >> vsize >> rss; // don't care about the rest

  stat_stream.close();

  // in case x86-64 is configured to use 2MB pages
  long page_size = sysconf(_SC_PAGE_SIZE); 

  // vm_usage     = vsize / 1024.0;
  // resident_set = rss * page_size;

  return rss * page_size;

#else
  
  return 0;

#endif

}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
