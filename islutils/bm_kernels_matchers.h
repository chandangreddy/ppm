#ifndef BM_ACCESS_TREE_MATCHERS_H
#define BM_ACCESS_TREE_MATCHERS_H

#include "bm_classes.h"

using namespace matchers;
using namespace blaskernels;



using AccessFunction = std::pair<std::string, std::vector<int>>;

namespace blaskernelmatchers {


  std::pair<bool, blaskernels::Gemm*>  findGemmAccess(isl::ctx, isl::union_map, isl::union_map);
  std::pair<bool, blaskernels::Gemm*> findAnyDotProductAccess(isl::ctx, isl::union_map, isl::union_map);
  std::pair<bool, blaskernels::Gemm*> findBatchGemmAccess(isl::ctx ctx, isl::union_map reads, isl::union_map writes);
  std::pair<bool, blaskernels::Gemm*> findTransposeGemmAccess(isl::ctx ctx, isl::union_map reads, isl::union_map writes);
  std::pair<bool, blaskernels::Gemm*> findTransposeBatchGemmAccess(isl::ctx ctx, isl::union_map reads, isl::union_map writes);
  //bool findAxpyAccess(isl::ctx ctx, isl::union_map reads, isl::union_map writes);
  //bool findDotProductAccess(isl::ctx ctx, isl::union_map reads, isl::union_map writes);
  std::pair<bool, blaskernels::Gemm*> findContractionAccess(isl::ctx ctx, isl::union_map reads, isl::union_map writes);
  //bool findTransposeAccess(isl::ctx ctx, isl::union_map reads, isl::union_map writes);



  bool findGemmTree(isl::schedule_node root, isl::schedule_node *node);
  bool findBatchedGemmTree(isl::schedule_node root, isl::schedule_node *node);
  bool findTransposeTree(isl::schedule_node root, isl::schedule_node *node);
  bool findAxpyTree(isl::schedule_node root, isl::schedule_node *node);
  bool findDotProductTree(isl::schedule_node root, isl::schedule_node *node);

} // namespace blaskernelmatchers
#endif
