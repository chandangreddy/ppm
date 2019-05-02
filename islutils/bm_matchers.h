#ifndef BLAS_MATCHERS_H
#define BLAS_MATCHERS_H


#include "bm_kernels_matchers.h"

#include <isl/cpp.h>

using namespace blaskernelmatchers;
using namespace builders;
using namespace blaskernels;
//using namespace isl;


namespace blasmatchers {

  using UmapPair = std::map<int, isl::union_map>;
  using Accesses = std::vector<std::pair<isl::union_map, isl::union_map>>;
 
  using ScopStmt = std::pair<isl::union_map, isl::union_map>;
  
  // findAnyDotProduct and findContraction should probably not return a Gemm. 
  // But while waiting for a proper class implementation, let it return a Gemm so that the code compiles 
  // at least properly.
  std::pair<bool, blaskernels::Gemm*> findGemm(isl::ctx, Scop, isl::union_map, isl::union_map);
  std::pair<bool, blaskernels::Gemm*> findAnyDotProduct(isl::ctx, Scop, isl::union_map, isl::union_map);
  std::pair<bool, blaskernels::Gemm*> findTransposeGemm(isl::ctx, Scop, isl::union_map, isl::union_map); 
  std::pair<bool, blaskernels::Gemm*> findBatchGemm(isl::ctx, Scop, isl::union_map, isl::union_map); 
  std::pair<bool, blaskernels::Gemm*> findTransposeBatchGemm(isl::ctx, Scop, isl::union_map, isl::union_map);   
  std::pair<bool, blaskernels::Gemm*> findContraction(isl::ctx, Scop, isl::union_map, isl::union_map);


  /* Commenting out the following patterns. The matchers work but return values 
     need to be adapted in the form of std::pair<bool, blaskernels::ProperClass*> */
      
  // bool findTranspose(isl::ctx, Scop, isl::union_map, isl::union_map); 
  // bool findAxpy(isl::ctx, Scop, isl::union_map, isl::union_map);
  // bool findDotProduct(isl::ctx, Scop, isl::union_map, isl::union_map);


  std::vector<std::pair<ScopStmt, std::vector<blaskernels::BlasKernels*>>> findPatterns(isl::ctx, Scop);
  
} // namespace blasmatchers

#endif
