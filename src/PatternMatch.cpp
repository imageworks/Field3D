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

/*! \file PatternMatch.cpp
  Contains pattern matching implementations
*/

//----------------------------------------------------------------------------//

// Header include
#include "PatternMatch.h"

// System includes
#ifdef WIN32
#include "Shlwapi.h"
#define FNM_NOMATCH  1
#define FNM_NOESCAPE 0
#else
#include <fnmatch.h>
#endif

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Function implementations
//----------------------------------------------------------------------------//

#ifdef WIN32
static int 
fnmatch(const char *pattern, const char *string, int /*flags*/)
{
  return PathMatchSpec(string, pattern) ? 0 : FNM_NOMATCH;
}
#endif

//----------------------------------------------------------------------------//

std::vector<std::string> 
split(const std::string &s)
{
  return split(s, " ");
}

//----------------------------------------------------------------------------//

std::vector<std::string> 
split(const std::string &s, const std::string &separatorChars)
{
  typedef boost::char_separator<char>     CharSeparator;
  typedef boost::tokenizer<CharSeparator> Tokenizer;

  std::vector<std::string>     result;
  CharSeparator separators(separatorChars.c_str());
  Tokenizer     tokenizer(s, separators);

  BOOST_FOREACH (const std::string &i, tokenizer) {
    result.push_back(i);
  }

  return result;
}

//------------------------------------------------------------------------------

bool 
match(const std::string &name, const std::string &attribute, 
      const std::vector<std::string> &patterns, 
      const MatchFlags flags)
{
  bool foundMatch = false;
  bool foundExclusion = false;

  if (patterns.size() == 0) {
    return flags && MatchEmptyPattern;
  }

  BOOST_FOREACH (const std::string &i, patterns) {

    if (i.size() == 0) {
      continue;
    }

    // Check exclusion string
    bool isExclusion = i[0] == '-' || i[0] == '^';
    // Update string
    const std::string pattern = isExclusion ? i.substr(1) : i;

    // String to match
    std::string s;

    // Determine type of matching
    if (pattern.find(":") != std::string::npos) {
      // Pattern includes separator. Match against name:attribute
      s = name + ":" + attribute;
    } else {
      // No separator. Just match against attribute
      s = attribute;
    }
    
    // Match with wildcards
    if (fnmatch(pattern.c_str(), s.c_str(), FNM_NOESCAPE) == 0) {
      if (isExclusion) {
        foundExclusion = true;
      } else {
        foundMatch = true;
      }
    }

  }

  return foundMatch && !foundExclusion;
}

//------------------------------------------------------------------------------

bool 
match(const std::string &name, const std::string &attribute, 
      const std::string &patterns, 
      const MatchFlags flags)
{
  return match(name, attribute, split(patterns), flags);
}

//------------------------------------------------------------------------------

bool 
match(const std::string &attribute, const std::vector<std::string> &patterns, 
      const MatchFlags flags)
{
  bool foundMatch = false;
  bool foundExclusion = false;

  if (patterns.size() == 0) {
    return flags && MatchEmptyPattern;
  }

  BOOST_FOREACH (const std::string &i, patterns) {

    if (i.size() == 0) {
      continue;
    }

    // Check exclusion string
    bool isExclusion = i[0] == '-' || i[0] == '^';
    // Update string
    std::string pattern = isExclusion ? i.substr(1) : i;

    // Determine type of matching
    size_t pos = pattern.find(":");
    if (pos != std::string::npos) {
      // Pattern includes separator. Just use second half
      pattern = pattern.substr(pos + 1);
    } 
    
    // Match with wildcards
    if (fnmatch(pattern.c_str(), attribute.c_str(), FNM_NOESCAPE) == 0) {
      if (isExclusion) {
        foundExclusion = true;
      } else {
        foundMatch = true;
      }
    }

  }

  return foundMatch && !foundExclusion;
}

//------------------------------------------------------------------------------

bool 
match(const std::string &attribute, const std::string &patterns, 
      const MatchFlags flags)
{
  return match(attribute, split(patterns), flags);
}

//------------------------------------------------------------------------------

bool 
match(const FieldRes *f, const std::vector<std::string> &patterns, 
      const MatchFlags flags)
{
  return match(f->name, f->attribute, patterns, flags);
}

//------------------------------------------------------------------------------

bool 
match(const FieldRes *f, const std::string &patterns, 
      const MatchFlags flags)
{
  return match(f->name, f->attribute, split(patterns), flags);
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
