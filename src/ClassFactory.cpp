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

/*! \file ClassFactory.cpp
  \brief Contains implementation of ClassFactory class.
*/

//----------------------------------------------------------------------------//

#include "ClassFactory.h"
#include "PluginLoader.h"

//----------------------------------------------------------------------------//

using namespace std;

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_OPEN

//----------------------------------------------------------------------------//
// Static instances
//----------------------------------------------------------------------------//

ClassFactory* ClassFactory::ms_instance = NULL;

//----------------------------------------------------------------------------//
// ClassFactory implementations
//----------------------------------------------------------------------------//

ClassFactory::ClassFactory()
{
  PluginLoader::loadPlugins();
}

//----------------------------------------------------------------------------//

void ClassFactory::registerFieldClass(CreateFieldFnPtr createFunc)
{
  // Make sure we don't add the same class twice

  bool nameExists = false;

  FieldRes::Ptr instance = createFunc();

  if (!instance) {
    Log::print(Log::SevWarning,
               "Unsuccessful attempt at registering Field class. "
               "(Creation function returned null pointer)");
    return;
  }

  string simpleClassName = instance->className();
  string dataTypeName = instance->dataTypeString();
  string className = simpleClassName + "<" + dataTypeName + ">";

  FieldNameToFuncMap::const_iterator i = m_fields.find(className);
  if (i != m_fields.end())
    nameExists = true;  

  if (!nameExists) {
    m_fields[className] = createFunc;
    // if the simple (untemplated) class name hasn't been registered
    // yet, add it to the list and print a message
    if (find(m_fieldSimpleNames.begin(), m_fieldSimpleNames.end(),
             simpleClassName) == m_fieldSimpleNames.end()) {
      m_fieldSimpleNames.push_back(simpleClassName);
      Log::print("Registered Field class " + simpleClassName);
    }

  } 

}

//----------------------------------------------------------------------------//

FieldRes::Ptr 
ClassFactory::createField(const std::string &className) const
{
  FieldNameToFuncMap::const_iterator i = m_fields.find(className);
  if (i != m_fields.end())
    return i->second();
  else
    return FieldRes::Ptr();
}

//----------------------------------------------------------------------------//

ClassFactory& 
ClassFactory::singleton()
{ 
  if (!ms_instance)
    ms_instance = new ClassFactory;
  return *ms_instance;
}

//----------------------------------------------------------------------------//

FIELD3D_NAMESPACE_SOURCE_CLOSE

//----------------------------------------------------------------------------//
