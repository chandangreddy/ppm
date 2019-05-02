#ifndef _CODEGEN_H_
#define _CODEGEN_H_

#include "islutils/access_patterns.h"
#include "islutils/builders.h"
#include "islutils/ctx.h"
#include "islutils/locus.h"
#include "islutils/matchers.h"
#include "islutils/pet_wrapper.h"
#include "islutils/aff_op.h"
#include "islutils/cout_overloading.h"

#include <iostream>
#include <stack>
#include <sstream>
#include <isl/cpp.h>
#include <vector>
#include <tuple>
#include <map>


using namespace std;

using util::ScopedCtx;


typedef struct {
  std::string name;
  unsigned n_index;
  std::string element_type;
  isl::set extent;
  std::vector<isl::map> accesses;
} GpuArrayInfo;



namespace conversion {

  template < typename T > std::string to_string( const T& n ) {
    std::ostringstream stm ;
    stm << n ;
    return stm.str() ;
  }

} // end namespace conversion


namespace blaskernels {

  enum Kernel {
    error,
    gemm,
    transpose,
    transposeGemm,
    batchGemm,
    transposeBatchGemm,
    axpy,
    dotProduct,
    contraction
  };

  const int nbKernels = 7;

  class BlasKernels {
  public:

    // Kernel id derived from enumeration
    int type;
    int batch = -1;
    std::string data_type = "D";
    // The matching schedule node
    isl::schedule_node schedule_node;
    std::vector<GpuArrayInfo> array_infos;
    std::string write_var = "null";

    void setScheduleNode(isl::schedule_node schednode);
    void setType(int kern);
    void setDataType();

    std::string allocateDeviceArrays(int tab);
    std::string copyToDeviceArrays(int tab);
    std::string declareDeviceArrays(int tab);
    std::string freeDeviceArrays(int tab);
    std::string copyFromDeviceArray(int tab);

    virtual std::string insertCallToCUBLAS(std::string c) = 0;
    virtual void fill(bool isTranspose) = 0; 
  };


  class Gemm : public BlasKernels {

  public:
    std::string A = "null";
    std::string B = "null";
    std::string ReadFromC = "null";

    std::string transa = "CUBLAS_OP_N";
    std::string transb = "CUBLAS_OP_N";
    std::string m;
    std::string n;
    std::string k_;
    std::string alpha = "alpha";
    std::string lda;
    std::string ldb;
    std::string beta = "beta";
    std::string ldc;

    int i = -1;
    int j = -1;
    int k = -1;


    void fillIndexInfos(int _i, int _j, int _k, int _batch, std::string _C);
    void fill(bool isTranspose);
    std::string insertCallToCUBLAS(std::string c);
  };



  std::string codeGenGPU(isl::ast_build, isl::ast_node, pet_stmt*, void*);
  std::string printCudaHeader();
  std::string getAccessName(isl::map m);
  std::string getAccessName(isl::set s);
  unsigned getAccessIndexes(isl::map m);
  unsigned getAccessIndexes(isl::set s);






} // end namespace blaskernels


#endif 

