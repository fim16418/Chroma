/*************************************************************************************

Chroma examples, www.github.com/fim16418/Chroma

Copyright (C) 2017

Source code: benchmarkCorrelation.cpp

Author: Moritz Fink <fink.moritz@gmail.com>


This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

See the full license in the file "LICENSE" in the top level distribution directory
*************************************************************************************/
/*  END LEGAL */

#include "chroma.h"
#include "actions/ferm/invert/syssolver_linop_cg.h"
#include <fstream>

#if defined(ENABLE_OPENMP)
#include "omp.h"
#else
typedef int omp_int_t;
inline void omp_set_num_threads(int num) {return;}
inline omp_int_t omp_get_max_threads() {return 1;}
#endif

int nData;
int nLoops;
multi1d<int> latt_size(4);
int nThreads;
std::string outFileName;

using namespace QDP;
using namespace Chroma;

double average(double* array,int len)
{
  double av = 0.0;
  for(int i=0; i<len; i++) {
    av += array[i];
  }
  return av/len;
}

double standardDeviation(double* array,int len)
{
  double squares = 0.0;
  for(int i=0; i<len; i++) {
    squares += array[i]*array[i];
  }
  squares /= len;

  double av = average(array,len);
  double var = squares - av*av;

  return sqrt(var);
}

bool processCmdLineArgs(int argc,char** argv)
{
  nData  = 2;
  nLoops = 10;
  latt_size[0]=4; latt_size[1]=4; latt_size[2]=4; latt_size[3]=4;
  nThreads = omp_get_max_threads();
  outFileName = "output.txt";

  for(int i=1; i<argc; i++) {
    std::string option = std::string(argv[i]);
    if(option == "--lattice") {
      if(i+5 == argc) { //--lattice must be last option
        for(int j=0; j<4; j++) {
          latt_size[j] = atoi(argv[i+j+1]);
        }
        i+=4;
      } else {
        std::cerr << "--lattice x y z t must be the last option." << std::endl;
        return false;
      }
    } else if(option == "--nData") {
      if(i+1 < argc) {
        nData = atoi(argv[++i]);
      } else {
        std::cerr << "--nData option requires one argument." << std::endl;
        return false;
      }
    } else if(option == "--nLoops") {
      if(i+1 < argc) {
        nLoops = atoi(argv[++i]);
      } else {
        std::cerr << "--nLoops option requires one argument." << std::endl;
        return false;
      }
    } else if(option == "--nThreads") {
      if(i+1 < argc) {
        nThreads = atoi(argv[++i]);
        omp_set_num_threads(nThreads);
      } else {
        std::cerr << "--nThreads option requires one argument." << std::endl;
        return false;
      }
    } else if(option == "--outFile") {
      if(i+1 < argc) {
        outFileName = argv[++i];
      } else {
        std::cerr << "--outFile option requires one argument." << std::endl;
        return false;
      }
    }
  }
  std::cout << "Lattice = " << latt_size[0] << " " << latt_size[1] << " " << latt_size[2] << " " << latt_size[3] << std::endl
            << "Measurements = " << nData << std::endl
            << "Loops per measurement = " << nLoops << std::endl
            << "Threads = " << omp_get_max_threads() << std::endl
            << "Output file = " << outFileName << std::endl << std::endl;
  return true;
}


int main(int argc, char **argv)
{
  Chroma::initialize(&argc, &argv);
  START_CODE();

  if(!processCmdLineArgs(argc,argv)) {
    END_CODE();
    Chroma::finalize();
    exit(1);
  }

  Layout::setLattSize(latt_size);
  Layout::create();

//////////////////////////////////////
/* Fill quark_propagator with 3.14: */

  LatticePropagator quark_propagator = zero;
  Complex c0 = 3.14;

  LatticeColorMatrix lcMatrix;
  for(int i=0; i<3; i++) {
    for(int j=0; j<3; j++) {
      pokeColor(lcMatrix,c0,i,j);
  }}

  for(int i=0; i<4; i++) {
    for(int j=0; j<4; j++) {
      pokeSpin(quark_propagator,lcMatrix,i,j);
  }}

//////////////////////////////////////

  Gamma gamma5 = Gamma(15);
  LatticePropagator anti_quark = gamma5 * quark_propagator * gamma5;
  anti_quark = adj(anti_quark);

  double timeData[nData];
  double flopData[nData];

  int latticeSize = latt_size[0]*latt_size[1]*latt_size[2]*latt_size[3];
  unsigned long flopsPerLoop = 2 * 30262 * latticeSize;

  for(int j=0; j<nData; j++) {
    StopWatch timer;
    timer.reset();
    timer.start();

    for(int i=0; i<nLoops; i++) {
      LatticeComplex corr_fn = trace(anti_quark * gamma5 * quark_propagator * gamma5);

      //output for comparison:
      /*multi1d<int> site(4); site[0] = 0; site[1] = 0; site[2] = 0; site[3] = 0;
      Complex c0 = peekSite(corr_fn,site);
      QDPIO::cout << endl << "out:" << c0 << endl; break;*/
    }

    timer.stop();

    timeData[j] = timer.getTimeInSeconds();
    flopData[j] = flopsPerLoop/timeData[j]/1000000000.0*nLoops;
  }

  ofstream file;
  file.open(outFileName,ios::app);
  if(file.is_open()) {
    file << nThreads << "\t" << latt_size[0] << "\t"
         << average(timeData,nData) << "\t" << standardDeviation(timeData,nData) << "\t"
         << average(flopData,nData) << "\t" << standardDeviation(flopData,nData) << std::endl;
    file.close();
  } else {
    std::cerr << "Unable to open file!" << std::endl;
  }

  END_CODE();
  Chroma::finalize();
  exit(0);
}