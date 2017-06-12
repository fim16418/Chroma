/*************************************************************************************

Chroma examples, www.github.com/fim16418/Chroma

Copyright (C) 2017

Source code: benchmark_su3.cpp

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

#include "omp.h"

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
  nLoops = 1000;
  latt_size[0]=4; latt_size[1]=4; latt_size[2]=4; latt_size[3]=8;
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
    Layout::create(); //call only once!

    LatticeColorMatrix z;
    LatticeColorMatrix x;
    LatticeColorMatrix y;

    StopWatch timer;
    timer.reset();
    timer.start();
    for(int i=0; i<nLoops; i++) {
      z=x*y; //x=x*y not allowed (cf qdp++ manual 3.3.5)
    }
    timer.stop();
    double time = timer.getTimeInMicroseconds()/nLoops*1000.0;

    int vol = Layout::vol();
    double bytes = 3*vol*Nc*Nc*sizeof(Complex);
    double flops = Nc*Nc*(6+8+8)*vol;

    int bossRank = 0;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    MPI_Barrier(MPI_COMM_WORLD);

    double sumTime;
    MPI_Reduce(&time,&sumTime,1,MPI_DOUBLE,MPI_SUM,bossRank,MPI_COMM_WORLD);

    int nProc;
    MPI_Comm_size(MPI_COMM_WORLD,&nProc);
    time = sumTime/nProc;

    if(rank == bossRank) {
      ofstream file;
      file.open(outFileName,ios::app);
      if(file.is_open()) {
        file << nThreads << "\t" << latt_size[0] << latt_size[1] << latt_size[2] << latt_size[3] << "\t"
             << bytes << "\t" << bytes/time << "\t" << flops/time << std::endl;
        file.close();
      } else {
        std::cerr << "Unable to open file!" << std::endl;
      }
    }

    END_CODE();
    Chroma::finalize();
    exit(0);
  }
