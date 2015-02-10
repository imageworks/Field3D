//----------------------------------------------------------------------------//

/*
 * Copyright (c) 2014 Sony Pictures Imageworks Inc
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

/*! \file FileSequence.cpp
  Contains file sequence implementations
*/

//----------------------------------------------------------------------------//

// Header include
#include "FileSequence.h"

// System includes
#include <cstdlib>
#include <iostream>

// Library includes
#include "Log.h"
#include "Field3DFile.h"

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// FileSequence implementations
//----------------------------------------------------------------------------//


FileSequence::FileSequence(const std::string &sequence)
{
  // Example sequences: myfile.1-2@.f3d, myfile1-21#.f3d
  // The number '1' in each of these is the 'sequence start'
  // The numbers '2' and '21' are the 'sequence end'

  const std::string k_numbers          = "0123456789";
  const std::string k_seqMarks         = "#@";
  const std::string k_framePlaceholder = "####";
  const size_t      npos               = std::string::npos;

  // Check for file by that name. If it exists, we are just that one file
  if (fileExists(sequence)) {
    m_filenames.push_back(sequence);
    return;
  }
      
  // Find the sequence mark
  const size_t seqMarkIdx = sequence.find_first_of(k_seqMarks);
    
  // If no sequence mark was found, there is no sequence.
  if (seqMarkIdx == npos) {
    return;
  }

  // Make sure there is not more than one sequence mark
  if (sequence.find_first_of(k_seqMarks, seqMarkIdx + 1) != npos) {
    std::stringstream warning;
    warning << "Multiple sequence marks in filename: " << sequence;
    Msg::print(Msg::SevWarning, warning.str());
    return;
  }

  // Get the end range index
  size_t seqEndIdx = sequence.find_last_not_of(k_numbers, seqMarkIdx - 1);
  if (seqEndIdx == npos) {
    std::stringstream warning;
    warning << "Sequence mark but no sequence range in filename: " 
            << sequence;
    Msg::print(Msg::SevWarning, warning.str());
    return;
  } else {
    seqEndIdx += 1;
  }
  if (seqEndIdx == 0) {
    std::stringstream warning;
    warning << "Sequence mark preceded by single number: " 
            << sequence;
    Msg::print(Msg::SevWarning, warning.str());
    return;
  }

  // Make sure the preceding character is '-'
  if (sequence[seqEndIdx - 1] != '-') {
    std::stringstream warning;
    warning << "Sequence mark preceded by single number but no '-': " 
            << sequence;
    Msg::print(Msg::SevWarning, warning.str());
    return;     
  }

  // Get the start range index
  size_t seqStartIdx = sequence.find_last_not_of(k_numbers, seqEndIdx - 2);
  if (seqStartIdx == npos) {
    std::stringstream warning;
    warning << "No sequence start in filename: " 
            << sequence;
    Msg::print(Msg::SevWarning, warning.str());
    return;
  } else {
    seqStartIdx += 1;
  }
    
  // String versions of frame numbers
  const std::string startStr = sequence.substr(seqStartIdx, seqEndIdx - 1);
  const std::string endStr   = sequence.substr(seqEndIdx, seqMarkIdx);

  // Get the integers
  const int start = atoi(startStr.c_str());
  const int end   = atoi(endStr.c_str());

  // Create the file basename for replacement
  const std::string baseStart = sequence.substr(0, seqStartIdx);
  const std::string baseEnd   = sequence.substr(seqMarkIdx + 1);

  // Create the filenames
  for (int i = start; i <= end; ++i) {
    std::stringstream filename;
    filename << baseStart << i << baseEnd;
    m_filenames.push_back(filename.str());
  }
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
