/*************************************************************************************

Chroma examples, www.github.com/fim16418/Chroma

Copyright (C) 2017

Source code: benchmarkDerivative.cpp

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
#include "meas/smear/displace.h" //rightNabla()
#include <fstream>

#include "omp.h"

#define WARM_UP 10

int nLoops;
multi1d<int> latt_size(4);
int nThreads;
int mu;
int length;
std::string outFileName;

using namespace QDP;
using namespace Chroma;


void error(double* array, int len, double& average, double& error)
{
  average = 0.0;
  double square = 0.0;

  for(int i=0; i<len; i++) {
    average += array[i];
    square += array[i]*array[i];
  }

  average = average/len;
  square = square/len;

  error = std::sqrt(square - average*average);
  error /= std::sqrt(len);
}

bool processCmdLineArgs(int argc,char** argv)
{
  nLoops = 1000;
  latt_size[0]=4; latt_size[1]=4; latt_size[2]=4; latt_size[3]=4;
  nThreads = omp_get_max_threads();
  mu = 0;
  length = 1;
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
    } else if(option == "--mu") {
      if(i+1 < argc) {
        mu = atoi(argv[++i]);
      } else {
        std::cerr << "--mu option requires one argument." << std::endl;
        return false;
      }
    } else if(option == "--length") {
      if(i+1 < argc) {
        length = atoi(argv[++i]);
      } else {
        std::cerr << "--length option requires one argument." << std::endl;
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
            << "Loops = " << nLoops << std::endl
            << "Threads = " << omp_get_max_threads() << std::endl
            << "Derivative in direction " << mu << " with length " << length << std::endl
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

  /*//////////////////
  // Initialization //
  //////////////////*/

  Layout::setLattSize(latt_size);
  Layout::create();

//////////////////////////////////////
/*        Fill gauge field:         */

  multi1d<LatticeColorMatrix> u(Nd);

  for(int dir=0; dir<Nd; dir++) {
    LatticeColorMatrix mat;
    for(int c1=0; c1<Nc; c1++) {
      for(int c2=0; c2<Nc; c2++) {
        Complex tmp = dir+c1+c2;
        pokeColor(mat,tmp,c1,c2);
    }}
    u[dir] = mat;
  }
  unitarityCheck(u);

//////////////////////////////////////
/*      Fill quark_propagator:      */

  LatticePropagator quark_propagator = zero;
  for(int x=0; x<latt_size[0]; x++) {
  for(int y=0; y<latt_size[1]; y++) {
  for(int z=0; z<latt_size[2]; z++) {
  for(int t=0; t<latt_size[3]; t++) {
    multi1d<int> coord(4);
    coord[0]=x;
    coord[1]=y;
    coord[2]=z;
    coord[3]=t;
    int index=t*latt_size[2]*latt_size[1]*latt_size[0]+z*latt_size[1]*latt_size[0]+y*latt_size[0]+x;
    Propagator tmpProp;
    for(int s1=0; s1<Ns; s1++) {
    for(int s2=0; s2<Ns; s2++) {
      ColorMatrix tmpMat;
      for(int c1=0; c1<Nc; c1++) {
      for(int c2=0; c2<Nc; c2++) {
        Complex tmp = index+s1+s2+c1+c2;
        pokeColor(tmpMat,tmp,c1,c2);
      }}
      pokeSpin(tmpProp,tmpMat,s1,s2);
    }}
    pokeSite(quark_propagator,tmpProp,coord);
  }}}}

//////////////////////////////////////

  Gamma gamma5 = Gamma(15);
  LatticePropagator anti_quark = gamma5 * quark_propagator * gamma5;
  anti_quark = adj(anti_quark);

  LatticeComplex corr = zero;

  /*///////////////
  // Calculation //
  //   Warm up   //
  ///////////////*/

  for(int i=0; i<WARM_UP; i++) {
    corr = trace(anti_quark * gamma5 * rightNabla(quark_propagator,u,mu,length) * gamma5);
  }

  /*///////////////
  // Calculation //
  // Measurement //
  ///////////////*/

  double timeData[nLoops];

  for(int i=0; i<nLoops; i++) {
    StopWatch timer;
    timer.reset();
    timer.start();

    corr = trace(anti_quark * gamma5 * rightNabla(quark_propagator,u,mu,length) * gamma5);

    //output for comparison:
    /*multi1d<int> site(4); site[0] = 0; site[1] = 0; site[2] = 0; site[3] = 0;
    Complex c0 = peekSite(corr_fn,site);
    QDPIO::cout << endl << "out:" << c0 << endl; break;*/

    timer.stop();
    timeData[i] = timer.getTimeInSeconds();
  }

  /*//////////////
  // Evaluation //
  //////////////*/

  double time, timeError;
  error(timeData,nLoops,time,timeError);

  unsigned long flopsPerLoop = 5*10080 + 22;
  double flops = flopsPerLoop/1000000000.0*Layout::vol();

  double flopsPerSec = flops/time;
  double flopsPerSec_error = timeError/time * flopsPerSec;

  /*/////////////////
  // Print results //
  /////////////////*/

  int bossRank = 0;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  if(rank == bossRank) {
    ofstream file;
    file.open(outFileName,ios::app);
    if(file.is_open()) {
      file << nThreads << "\t" << latt_size[0] << latt_size[1] << latt_size[2] << latt_size[3] << "\t"
           << Layout::vol() << "\t" << time << "\t" << timeError << "\t" << flopsPerSec << "\t" << flopsPerSec_error << std::endl;
      file.close();
    } else {
      std::cerr << "Unable to open file!" << std::endl;
    }
  }

  END_CODE();
  Chroma::finalize();
  exit(0);
}
