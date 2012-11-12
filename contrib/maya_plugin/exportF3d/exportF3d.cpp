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

/*! \file exportF3d.cpp
  \brief Simple exporter for Maya Fluid to f3d format.  density, temperature, 
  fuel are exported automatically if they are valid, all other fields require flags.
*/


#include <maya/MPxCommand.h>
#include <maya/MSyntax.h>
#include <maya/MArgDatabase.h>
#include <maya/MItSelectionList.h>
#include <maya/MFnPlugin.h>
#include <maya/MFnCamera.h>
#include <maya/MFnFluid.h>
#include <maya/MAnimControl.h>
#include <maya/MFnDagNode.h>
#include <maya/MDagPath.h>
#include <maya/MMatrix.h>
#include <maya/MTransformationMatrix.h>

#include <maya/MIOStream.h>
#include <maya/MGlobal.h>
#include <maya/MFnTransform.h>
#include <maya/MTransformationMatrix.h>
#include <maya/MFileIO.h>
#include <maya/MCommonSystemUtils.h>

#include <sys/wait.h>
#include <ctype.h>
#include <string>
#include <list>
#include <vector>
#include <iostream>
#include <fstream>

#include <Field3D/DenseField.h>
#include <Field3D/MACField.h>
#include <Field3D/Field3DFile.h>
#include <Field3D/InitIO.h>

#include <maya/MComputation.h>

#define ERRCHKR		\
	if ( MS::kSuccess != stat ) {	\
    cerr << stat.errorString(); \
		return stat;	\
	}

#define ERRCHK		\
	if ( MS::kSuccess != stat ) {	\
    cerr << stat.errorString(); \
	}


using namespace std;
using namespace Field3D;


//----------------------------------------------------------------------------//

class exportF3d : public MPxCommand
{

public:
  exportF3d();
  virtual ~exportF3d() {}
    
  MStatus     doIt(const MArgList&);

  static void*  creator(){
    return new exportF3d();
  }

    
  static MSyntax  newSyntax();

private:
  void setF3dField(MFnFluid &fluidFn, const char *outputPath, const MDagPath &dagPath);
    

private:
  MStatus     parseArgs( const MArgList& args );

  MString        m_outputPath;
  bool           m_verbose;
  MSelectionList m_slist;
  int            m_start; //<-start of simulation
  int            m_end; //<-end of simulation

  bool           m_density; //<- export densiyt as well
  bool           m_temperature; //<- export temprature as well
  bool           m_fuel; //<- export fuel as well
  bool           m_color; //<- export color as well
  bool           m_vel; //<- export velocity as well
  bool           m_pressure; //<- export presurre as well
  bool           m_texture; //<- export texture as well
  bool           m_falloff; //<- export falloff as well
  int            m_numOversample; //<- oversamples the fluids but only writes out on whole frames
};

//----------------------------------------------------------------------------//

exportF3d::exportF3d()
{
  m_start = 1;
  m_end = 1;
  m_verbose = false;
  m_density = true;
  m_temperature = true;
  m_fuel = true;    
  m_color = false;
  m_vel = false;
  m_pressure = false;
  m_texture = false;
  m_falloff = false;
  m_numOversample = 1;
}

//----------------------------------------------------------------------------//

MSyntax exportF3d::newSyntax()
{
    MSyntax syntax;
    MStatus stat;
    stat = syntax.addFlag("-st", "-startTime", MSyntax::kLong);ERRCHK;    
    stat = syntax.addFlag("-et", "-endTime", MSyntax::kLong);ERRCHK;  
    stat = syntax.addFlag("-o", "-outputPath", MSyntax::kString);ERRCHK; 
    stat = syntax.addFlag("-av", "-addVelocity", MSyntax::kNoArg);ERRCHK; 
    stat = syntax.addFlag("-ac", "-addColor",MSyntax::kNoArg);ERRCHK; 
    stat = syntax.addFlag("-ap", "-addPressure",MSyntax::kNoArg);ERRCHK; 
    stat = syntax.addFlag("-at", "-addTexture",MSyntax::kNoArg);ERRCHK; 
    stat = syntax.addFlag("-af", "-addFalloff",MSyntax::kNoArg);ERRCHK;
    stat = syntax.addFlag("-ns", "-numOversample",MSyntax::kLong);ERRCHK; 
    
    stat = syntax.addFlag("-d", "-debug");ERRCHK; 
    syntax.addFlag("-h", "-help");
    
    // DEFINE BEHAVIOUR OF COMMAND ARGUMENT THAT SPECIFIES THE MESH NODE:
    syntax.useSelectionAsDefault(true);
    stat = syntax.setObjectType(MSyntax::kSelectionList, 1);

    // MAKE COMMAND QUERYABLE AND NON-EDITABLE:
    syntax.enableQuery(false);
    syntax.enableEdit(false);


    return syntax;
}


MStatus exportF3d::parseArgs( const MArgList &args )
{
  MStatus status;

  MArgDatabase argData(syntax(), args);

  if (argData.isFlagSet("-debug"))
    m_verbose = true;

  if (argData.isFlagSet("-help"))
  {
        
    MString help = (
      "\nexportF3d is used to export 3D fluid data to either f3d\n"
      "Synopsis: exportF3d [flags] [fluidObject... ]\n"
      "Flags:\n"
      "    -o     -outputPath        String outputPath won't work with element name\n"
      "    -st    -startTime    int  Start of simulation\n"
      "    -et    -endTime      int  End of simulation\n"
      "    -av    -addVelocity       Export velocity as v_mac\n"
      "    -ac    -addColor          Export color data\n"
      "    -ap    -addPressure       Export pressure data\n"
      "    -at    -addTexture        Export texture\n"
      "    -af    -addFalloff        Export falloff\n"
      "    -ns    -numOversample     Oversamples the solver at each sum frame but\n"
      "                              only writes out whole frame sim data\n"
      "    -d     -debug\n"
      "    -h     -help\n"
      "Example:\n"
      "1- exportF3d -st 1 -et 100  -o \"/tmp/\"  fluidObject \n\n"
      );
    MGlobal::displayInfo(help);
    return MS::kFailure;

  }

  if (argData.isFlagSet("-startTime"))
  {
    status = argData.getFlagArgument("-startTime", 0, m_start);
  }
  if (argData.isFlagSet("-endTime"))
  {
    status = argData.getFlagArgument("-endTime", 0, m_end);
  }else 
    m_end = m_start;
    
  if (argData.isFlagSet("-addColor"))
    m_color = true;
  if (argData.isFlagSet("-addVelocity"))
    m_vel = true;
  if (argData.isFlagSet("-addPressure"))
    m_pressure = true;
  if (argData.isFlagSet("-addTexture"))
    m_texture = true;
    
  if (argData.isFlagSet("-addFalloff"))
    m_falloff = true;
    
  if (argData.isFlagSet("-outputPath"))
  {
    status = argData.getFlagArgument("-outputPath", 0, m_outputPath);
  }

  if (!argData.isFlagSet("-outputPath") )
  {
    MGlobal::displayInfo("outputPath is required");
    return MS::kFailure;            
  }

  if (argData.isFlagSet("-numOversample"))
  {
    status = argData.getFlagArgument("-numOversample", 0, m_numOversample);
    if (m_numOversample < 1) { 
      m_numOversample = 1;
      MGlobal::displayWarning("numOversample can't be less than one, setting it to 1");
    }
  }

  status = argData.getObjects(m_slist);
  if (!status)
  {
    status.perror("no fluid object was selected");
    return status;
  }
  // Get the selected Fluid Systems
  if (m_slist.length() > 1) {
    MGlobal::displayWarning("[exportF3d]: only first fluid object is used to export");
  }



  return MS::kSuccess;
}

MStatus exportF3d::doIt(const MArgList& args)
{
  MStatus status;
  MString result;
    
  status = parseArgs(args);
  if (!status)
  {
    status.perror("Parsing error");
    return status;
  }
  
  float currentFrame = MAnimControl::currentTime().value();

  MItSelectionList selListIter(m_slist, MFn::kFluid, &status);  
  for (; !selListIter.isDone(); selListIter.next())
  {
    MDagPath 	dagPath;
    MObject 	selectedObject;
    
    status = selListIter.getDependNode(selectedObject);
    status = selListIter.getDagPath(dagPath);
    // Create function set for the fluid
    MFnFluid fluidFn(dagPath, &status);
    if (status != MS::kSuccess)
      continue;
  
    if (m_verbose)
    {        
      cout << "------------------------------------------------------" << endl;
      cout << " Selected object: " << fluidFn.name() << endl;
      cout << " Selected object type: " << selectedObject.apiTypeStr() << endl;
      cout << endl << endl;
    }      
       
    MFnFluid::FluidMethod method;
    MFnFluid::FluidGradient gradient;
      
    status  = fluidFn.getDensityMode(method, gradient);
    if(method != MFnFluid::kStaticGrid && method != MFnFluid::kDynamicGrid) {
      m_density = false;
    }
    status  = fluidFn.getTemperatureMode(method, gradient);
    if(method != MFnFluid::kStaticGrid && method != MFnFluid::kDynamicGrid) {
      m_temperature = false;
    }
    status  = fluidFn.getFuelMode(method, gradient);
    if(method != MFnFluid::kStaticGrid && method != MFnFluid::kDynamicGrid) {      
      m_fuel = false;
    }

    status  = fluidFn.getVelocityMode(method, gradient);
    if(method != MFnFluid::kStaticGrid && method != MFnFluid::kDynamicGrid) {
      m_vel = false;
    }

    if (m_color) {
      MFnFluid::ColorMethod colorMethod;
      fluidFn.getColorMode(colorMethod);
      if(colorMethod == MFnFluid::kUseShadingColor) {
        m_color = false;
      }
    }

    if (!m_vel) {
      // Note that the pressure data only exists if the velocity method
      // is kStaticGrid or kDynamicGrid
      m_pressure = false;
    }

    if (m_falloff)
    {
      MFnFluid::FalloffMethod falloffMethod;
      status  = fluidFn.getFalloffMode(falloffMethod);
      if(falloffMethod != MFnFluid::kNoFalloffGrid ) {
        m_falloff = false;
      }
    }

    // Go through the selected frame range
    MComputation computation;
    computation.beginComputation();
    char fluidPath[1024];

    for(int frame=m_start ; frame <= m_end; ++frame)
    {
      int numOversample = m_numOversample;
      if (frame == m_start)
        numOversample = 1;
      float dt = 1.0/double(numOversample);

      for ( int s=numOversample-1 ; s >= 0 ; --s)
      {

        if  (computation.isInterruptRequested()) break ;
        float time = frame - s*dt;
          
        status = MAnimControl::setCurrentTime(time);
        //MPlug plugGrid = fluidFn.findPlug( "outGrid",true,&status);
        MGlobal::displayInfo(MString("Setting Current frame ")+time);

      }

      sprintf(fluidPath, "%s/%s.%04d.f3d", m_outputPath.asChar(),
              fluidFn.name().asChar(),frame);
      MGlobal::displayInfo(MString("Writting: ")+fluidPath);

      setF3dField(fluidFn, fluidPath, dagPath);
   
    }

    computation.endComputation();	
    // only one fluid object 
    break;
  }

  setResult(result);
  MAnimControl::setCurrentTime(currentFrame);
    
  return MS::kSuccess;
}



/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

void exportF3d::setF3dField(MFnFluid &fluidFn, const char *outputPath, 
                            const MDagPath &dagPath)
{
    
  try { 
      
    MStatus stat;

    unsigned int i, xres = 0, yres = 0, zres = 0;
    double xdim,ydim,zdim;
    // Get the resolution of the fluid container      
    stat = fluidFn.getResolution(xres, yres, zres);
    stat = fluidFn.getDimensions(xdim, ydim, zdim);
    V3d size(xdim,ydim,zdim);
    const V3i res(xres, yres, zres);
    int psizeTot  = fluidFn.gridSize();

    /// get the transform and rotation
    MObject parentObj = fluidFn.parent(0, &stat);
    if (stat != MS::kSuccess) {

      MGlobal::displayError("Can't find fluid's parent node");
      return;
    }
    MDagPath parentPath = dagPath;
    parentPath.pop();
    MTransformationMatrix tmatFn(dagPath.inclusiveMatrix());
    if (stat != MS::kSuccess) {

      MGlobal::displayError("Failed to get transformation matrix of fluid's parent node");
      return;
    }


    MFnTransform fnXform(parentPath, &stat);
    if (stat != MS::kSuccess) {

      MGlobal::displayError("Can't create a MFnTransform from fluid's parent node");
      return;
    }
          

    if (m_verbose)
    {
      fprintf(stderr, "cellnum: %dx%dx%d = %d\n",  
              xres, yres, zres,psizeTot);
    }

    float *density(NULL), *temp(NULL), *fuel(NULL);
    float *pressure(NULL), *falloff(NULL);
      
    density = fluidFn.density( &stat );
    if ( stat.error() ) m_density = false;

    temp    = fluidFn.temperature( &stat );
    if ( stat.error() ) m_temperature = false;
      
    fuel    = fluidFn.fuel( &stat );
    if ( stat.error() ) m_fuel = false;    
      
    pressure= fluidFn.pressure( &stat );
    if ( stat.error() ) m_pressure = false;

    falloff = fluidFn.falloff( &stat );
    if ( stat.error() ) m_falloff = false;

    float *r,*g,*b;
    if (m_color) {
      stat = fluidFn.getColors(r,b,g);
      if ( stat.error() ) m_color = false;
    }else
      m_color = false;
      
    float *u,*v,*w;
    if (m_texture) {
      stat = fluidFn.getCoordinates(u,v,w);
      if ( stat.error() ) m_texture = false;
    }else
      m_texture = false;

    /// velocity info
    float *Xvel(NULL),*Yvel(NULL), *Zvel(NULL);  
    if (m_vel) { 
      stat = fluidFn.getVelocity( Xvel,Yvel,Zvel );
      if ( stat.error() ) m_vel = false;
    }
    

    if (m_density == false && m_temperature==false && m_fuel==false &&
        m_pressure==false && m_falloff==false && m_vel == false && 
        m_color == false && m_texture==false)
    {
      MGlobal::displayError("No fluid attributes found for writing, please check fluids settings");
      return;
    }
            
    /// Fields 
    DenseFieldf::Ptr densityFld, tempFld, fuelFld, pressureFld, falloffFld;
    DenseField3f::Ptr CdFld, uvwFld;
    MACField3f::Ptr vMac;

    MPlug autoResizePlug = fluidFn.findPlug("autoResize", &stat); 
    bool autoResize;
    autoResizePlug.getValue(autoResize);

    // maya's fluid transformation
    V3d dynamicOffset(0);
    M44d localToWorld;
    MatrixFieldMapping::Ptr mapping(new MatrixFieldMapping());

    M44d fluid_mat(tmatFn.asMatrix().matrix);

    if(autoResize) {      
      fluidFn.findPlug("dofx").getValue(dynamicOffset[0]);
      fluidFn.findPlug("dofy").getValue(dynamicOffset[1]);
      fluidFn.findPlug("dofz").getValue(dynamicOffset[2]);
    }

    Box3i extents;
    extents.max = res - V3i(1);
    extents.min = V3i(0);
    mapping->setExtents(extents);
  
    localToWorld.setScale(size);
    localToWorld *= M44d().setTranslation( -(size*0.5) );
    localToWorld *= M44d().setTranslation( dynamicOffset );
    localToWorld *= fluid_mat;
    
    mapping->setLocalToWorld(localToWorld);  
      
    if (m_density){
      densityFld = new DenseFieldf;
      densityFld->setSize(res);
      densityFld->setMapping(mapping);
    }
    if (m_fuel){
      fuelFld = new DenseFieldf;
      fuelFld->setSize(res); 
      fuelFld->setMapping(mapping);
    }
    if (m_temperature){
      tempFld = new DenseFieldf;
      tempFld->setSize(res);
      tempFld->setMapping(mapping);
    }
    if (m_pressure){
      pressureFld = new DenseFieldf;
      pressureFld->setSize(res);
      pressureFld->setMapping(mapping);
    }
    if (m_falloff){
      falloffFld = new DenseFieldf;
      falloffFld->setSize(res);
      falloffFld->setMapping(mapping);
    }
    if (m_vel){
      vMac = new MACField3f;
      vMac->setSize(res);
      vMac->setMapping(mapping);
    } 
    if (m_color){
      CdFld = new DenseField3f;
      CdFld->setSize(res);
      CdFld->setMapping(mapping);
    } 
    if (m_texture){
      uvwFld = new DenseField3f;
      uvwFld->setSize(res);
      uvwFld->setMapping(mapping);
    } 
        
    size_t iX, iY, iZ;      
    for( iZ = 0; iZ < zres; iZ++ ) 
    {
      for( iX = 0; iX < xres; iX++ )
      {
        for( iY = 0; iY < yres ; iY++ ) 
        {
    
          /// data is in x major but we are writting in z major order
          i = fluidFn.index( iX, iY,  iZ);
            
          if ( m_density ) 
            densityFld->lvalue(iX, iY, iZ) = density[i];            
          if ( m_temperature ) 
            tempFld->lvalue(iX, iY, iZ) = temp[i];
          if ( m_fuel )   
            fuelFld->lvalue(iX, iY, iZ) = fuel[i];
          if ( m_pressure )   
            pressureFld->lvalue(iX, iY, iZ) = pressure[i];
          if ( m_falloff )   
            falloffFld->lvalue(iX, iY, iZ) = falloff[i];
          if (m_color)
            CdFld->lvalue(iX, iY, iZ) = V3f(r[i], g[i], b[i]);
          if (m_texture)
            uvwFld->lvalue(iX, iY, iZ) = V3f(u[i], v[i], w[i]);
        }
      }      
    }

      
    if (m_vel) {
      unsigned x,y,z;
      for(z=0;z<zres;++z) for(y=0;y<yres;++y) for(x=0;x<xres+1;++x) {
            vMac->u(x,y,z) = *Xvel++;
          }
        
      for(z=0;z<zres;++z) for(y=0;y<yres+1;++y) for(x=0;x<xres;++x) {
            vMac->v(x,y,z) = *Yvel++;
          }
        
      for(z=0;z<zres+1;++z) for(y=0;y<yres;++y) for(x=0;x<xres;++x) {
            vMac->w(x,y,z) = *Zvel++;
          }                        
    } 
     
    Field3DOutputFile out;
    if (!out.create(outputPath)) {
      MGlobal::displayError("Couldn't create file: "+ MString(outputPath));
      return;
    }

    string fieldname("maya");

    if (m_density){
        out.writeScalarLayer<float>(fieldname, "density", densityFld);
    }
    if (m_fuel) { 
        out.writeScalarLayer<float>(fieldname,"fuel", fuelFld);
    }
    if (m_temperature){
        out.writeScalarLayer<float>(fieldname,"temperature", tempFld);
    }
    if (m_color) {
        out.writeVectorLayer<float>(fieldname,"Cd", CdFld);
    }
    if (m_vel)
      out.writeVectorLayer<float>(fieldname,"v_mac", vMac);      

    out.close(); 

  }
  catch(const std::exception &e)
  {

    MGlobal::displayError( MString(e.what()) );
    return;
  }


}



/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

MStatus initializePlugin( MObject obj )
{
    MStatus status;
    MFnPlugin plugin( obj, "Open Source Field3D, Sosh Mirsepassi", "1.3.1", "Any" );

    status = plugin.registerCommand("exportF3d", 
                                    exportF3d::creator, 
                                    exportF3d::newSyntax);
    if (!status)
    {
        status.perror("registerCommand");
        return status;
    }

    Field3D::initIO();

    return status;
}

MStatus uninitializePlugin( MObject obj)
{
    MStatus status;
    MFnPlugin plugin( obj );

    status = plugin.deregisterCommand("exportF3d");
    if (!status)
    {
        status.perror("deregisterCommand");
        return status;
    }

    return status;
    
}

/////////////////////////////////////////////////////////////////////

