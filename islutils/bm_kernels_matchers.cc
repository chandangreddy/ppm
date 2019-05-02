#include "bm_kernels_matchers.h"

namespace blaskernelmatchers {




  
  bool find1DTree(isl::schedule_node root) {
    // Ensure if u need to check if strided or not.
    auto matcher = band(
			leaf()
			);
    return ScheduleNodeMatcher::isMatching(matcher, root);
  } 

  bool find2DTree(isl::schedule_node root) {
    auto matcher = band(
			band(
			     leaf()));									
    return ScheduleNodeMatcher::isMatching(matcher, root);
  }

  bool findNDPermutableBand(isl::schedule_node root, isl::schedule_node *node, unsigned int nbDim) {
    // For the moment reusing what is already implemented in test_transformer.cc
    // This is also valid for transposeGemm
    auto matcher = band(
			[&node, nbDim](isl::schedule_node n) {
			  if (isl_schedule_node_band_n_member(n.get()) < nbDim) {
			    return false;
			  } else {
			    node = &n;
			    return true;
			  }
			}, leaf());
    return ScheduleNodeMatcher::isMatching(matcher, root);
  }

  bool findGemmTree(isl::schedule_node root, isl::schedule_node *node) {
    return findNDPermutableBand(root, node, 3u);
  }

  bool findBatchedGemmTree(isl::schedule_node root, isl::schedule_node *node) {
    return findNDPermutableBand(root, node, 4u);
  }

  bool findTransposeTree(isl::schedule_node root, isl::schedule_node *node) {
    // Must check if this is the proper condition.
    return find2DTree(root);
  }

  bool findAxpyTree(isl::schedule_node root, isl::schedule_node *node) {
    return find1DTree(root);
  }

  bool findDotProductTree(isl::schedule_node root, isl::schedule_node *node) {
    return find1DTree(root);
  }
  


  
  std::pair<bool, blaskernels::Gemm*> 
  findGemmAccess(isl::ctx ctx, isl::union_map reads, isl::union_map writes) {
    blaskernels::Gemm * gemm_infos = new blaskernels::Gemm();


    auto _i = placeholder(ctx);
    auto _j = placeholder(ctx);
    auto _k = placeholder(ctx);
    auto _ii = placeholder(ctx);
    auto _jj = placeholder(ctx);
    auto localReads = allOf(access(_i, _j), access(_i, _k), access(_k, _j));
    auto localWrites = allOf(access(_ii, _jj));
    auto matchReads = match(reads, localReads);
    auto matchWrites = match(writes, localWrites);

    if ((matchReads.size() == 1u) && (matchWrites.size() == 1u)) {
      int i = matchReads[0][_i].payload().inputDimPos_;
      int j = matchReads[0][_j].payload().inputDimPos_;
      int k = matchReads[0][_k].payload().inputDimPos_;
      int ii = matchWrites[0][_ii].payload().inputDimPos_;
      int jj = matchWrites[0][_jj].payload().inputDimPos_;
      bool isMatch = ((ii == 0) && (jj == 1) && 
		      (ii == i) && (jj == j) && 
		      (k == 2)); 

      if (isMatch == true) {
        auto vspace = matchWrites[0][_ii].candidateSpaces(); 
        auto cname = vspace[0].range().unwrap().range().get_tuple_name(isl::dim::out);
        gemm_infos->fillIndexInfos(i, j, k, -1, cname);
        return {isMatch, gemm_infos};
      } else {
        return {isMatch, gemm_infos};
      }
    } else {
      return {false, gemm_infos};
    }
  }



  std::pair<bool, blaskernels::Gemm*>  findBatchGemmAccess(isl::ctx ctx, isl::union_map reads, isl::union_map writes) {
    blaskernels::Gemm * gemm_infos = new blaskernels::Gemm();
    auto _i = placeholder(ctx);
    auto _j = placeholder(ctx);
    auto _k = placeholder(ctx);
    auto _b = placeholder(ctx);
    auto _ii = placeholder(ctx);
    auto _jj = placeholder(ctx);
    auto _bb = placeholder(ctx);
    auto localReads = allOf(access(_b, _i, _j), 
			    access(_b, _i, _k), 
			    access(_b, _k, _j));
    auto localWrites = allOf(access(_bb, _ii, _jj));
    auto matchReads = match(reads, localReads);
    auto matchWrites = match(writes, localWrites);


    // The following is mostly to ensure that _b == _bb,
    // _ii == _i and _jj == _i, since I cannot use the 
    // same placeholder for reads and writes.
    if ((matchReads.size() == 1u) && (matchWrites.size() == 1u)) {
      int b = matchReads[0][_b].payload().inputDimPos_;
      int i = matchReads[0][_i].payload().inputDimPos_;
      int j = matchReads[0][_j].payload().inputDimPos_;
      int k = matchReads[0][_k].payload().inputDimPos_;
      int bb = matchWrites[0][_bb].payload().inputDimPos_;
      int ii = matchWrites[0][_ii].payload().inputDimPos_;
      int jj = matchWrites[0][_jj].payload().inputDimPos_;
      bool isMatch = ((bb == 0) && (ii == 1) && (jj == 2) && 
		      (bb == b) && (ii == i) && (jj == j) && 
		      (k == 3));

      if (isMatch == true) {
	auto vspace = matchWrites[0][_ii].candidateSpaces(); 
        auto cname = vspace[0].range().unwrap().range().get_tuple_name(isl::dim::out);
        gemm_infos->fillIndexInfos(i, j, k, bb, cname);
        return {isMatch, gemm_infos};

      } else {
        return {isMatch, gemm_infos};
      }
    } else { 
      return {false, gemm_infos}; 
    }
  }

  std::pair<bool, blaskernels::Gemm*> findTransposeGemmAccess(isl::ctx ctx, isl::union_map reads, isl::union_map writes) {
    blaskernels::Gemm * gemm_infos = new blaskernels::Gemm();
    auto _i = placeholder(ctx);
    auto _j = placeholder(ctx);
    auto _k = placeholder(ctx);
    auto _ii = placeholder(ctx);
    auto _jj = placeholder(ctx);
    auto localReads = allOf(access(_i, _j), access(_i, _k), access(_j, _k));
    auto localWrites = allOf(access(_ii, _jj));
    auto matchReads = match(reads, localReads);
    auto matchWrites = match(writes, localWrites);

    if ((matchReads.size() == 1u) && (matchWrites.size() == 1u)) {
      int i = matchReads[0][_i].payload().inputDimPos_;
      int j = matchReads[0][_j].payload().inputDimPos_;
      int k = matchReads[0][_k].payload().inputDimPos_;
      int ii = matchWrites[0][_ii].payload().inputDimPos_;
      int jj = matchWrites[0][_jj].payload().inputDimPos_;
      bool isMatch = ((ii == 0) && (jj == 1) && 
		      (ii == i) && (jj == j) && 
		      (k == 2)); 

      if (isMatch == true) {
        auto vspace = matchWrites[0][_ii].candidateSpaces();
        auto cname = vspace[0].range().unwrap().range().get_tuple_name(isl::dim::out);
        gemm_infos->fillIndexInfos(i, j, k, -1, cname);
        return {isMatch, gemm_infos};
      } else {
        return {isMatch, gemm_infos};
      }
    } else { 
      return {false, gemm_infos}; 
    }
  }

  std::pair<bool, blaskernels::Gemm*> findTransposeBatchGemmAccess(isl::ctx ctx, isl::union_map reads, isl::union_map writes) {
    blaskernels::Gemm * gemm_infos = new blaskernels::Gemm();
    auto _i = placeholder(ctx);
    auto _j = placeholder(ctx);
    auto _k = placeholder(ctx);
    auto _b = placeholder(ctx);
    auto _ii = placeholder(ctx);
    auto _jj = placeholder(ctx);
    auto _bb = placeholder(ctx);
    auto localReads = allOf(access(_b, _i, _j), 
			    access(_b, _i, _k), 
			    access(_b, _j, _k));
    auto localWrites = allOf(access(_bb, _ii, _jj));
    auto matchReads = match(reads, localReads);
    auto matchWrites = match(writes, localWrites);

    if ((matchReads.size() == 1u) && (matchWrites.size() == 1u)) {
      int b = matchReads[0][_b].payload().inputDimPos_;
      int i = matchReads[0][_i].payload().inputDimPos_;
      int j = matchReads[0][_j].payload().inputDimPos_;
      int k = matchReads[0][_k].payload().inputDimPos_;
      int bb = matchWrites[0][_bb].payload().inputDimPos_;
      int ii = matchWrites[0][_ii].payload().inputDimPos_;
      int jj = matchWrites[0][_jj].payload().inputDimPos_;
      bool isMatch = ((bb == 0) && (ii == 1) && (jj == 2) && 
		      (bb == b) && (ii == i) && (jj == j) && 
		      (k == 3)); 
      if (isMatch == true) {
        auto vspace = matchWrites[0][_ii].candidateSpaces();
        auto cname = vspace[0].range().unwrap().range().get_tuple_name(isl::dim::out);
        gemm_infos->fillIndexInfos(i, j, k, bb, cname);
        return {isMatch, gemm_infos};
      } else {
        return {isMatch, gemm_infos};
      }
    } else { 
      return {false, gemm_infos}; 
    }
  }



  /* The matching process for Contractions happens in this function */
  std::map<std::string, std::vector<int>> 
    reconstruct (isl::ctx ctx, isl::union_map umap) {
    auto _k = placeholder(ctx);
    std::map<std::string, std::vector<int>> output;
    int counter = 0;
    auto acc = match(umap, allOf(access(dim(counter, _k))));

    while (acc.size() > 0) {

      for (int i = 0; i < (int)acc.size(); ++i) {
	auto space = acc[i][_k].candidateSpaces();
	auto name = space[0].range().unwrap().get_tuple_name(isl::dim::out);
	output[name].push_back(acc[i][_k].payload().inputDimPos_);
      }
      counter += 1;
      acc = match(umap, allOf(access(dim(counter, _k))));
    }
    return output;
  }

  bool
  hasNoRedundancy(std::vector<int> vec) {
    bool isNotRedundant = true;
    for (int i = 1; i < (int)vec.size(); ++i) {
      if (vec[i] == vec[i-1]) {
	isNotRedundant = false;
	break;
      }
    }
    return isNotRedundant;
  }

  bool hasDuplicatedIndexesOnly(std::vector<int> vec) {
    // We assume the vector to be sorted because
    // if this point is reached, then sorting must
    // have been done before in order to use 
    // std::set_difference().

    // We look at odd numbers only;
    // The following conditions must be met.
    // 1. All indexes at even positions are !=
    // 2. An index at a given even position is == 
    // to the next(p).

    std::vector<int> evens;
    evens.push_back(vec[0]);
    for (auto i = 2; i < (int)vec.size(); i += 2) {
      if (std::find(evens.begin(), evens.end(), vec[i]) == evens.end()) 
	evens.push_back(vec[i]);
      else 
	return false;
    }
    // If we reach this point, then ok, lets see of their
    // next is equal.
    for (auto i = 0; i < (int)vec.size(); i += 2) {
      if (vec[i] != vec[i+1])
	return false;
    }
    return true;
  }

  bool 
  existsContractionBetween(std::vector<int> r1, 
			   std::vector<int> r2) {

    for (auto k : r1) 
      for (auto p : r2) 
	if (k == p) 
	  return true;
    return false;
  }


  std::pair<bool, blaskernels::Gemm*>
  findAnyDotProductAccess(isl::ctx ctx, isl::union_map reads, isl::union_map writes) {
    auto _reads = reconstruct(ctx, reads);
    auto _writes = reconstruct(ctx, writes);

    auto wname = _writes.begin()->first;
    auto wacc = _writes.begin()->second;

    // Or with such a test, I'd rather say "me be a reduction"
    // because an accumulation also passes this test, but is not 
    // necessarily a reduction.
    auto isReduction = (_reads.find(wname) != _reads.end()) && 
      (wacc == _reads[wname]);

    // We don't want the inductive variable among reads anymore.
    _reads.erase(wname);

    std::vector<int> all_read_indexes;
    // We are a looking for an index appears in both reads
    // and not appearing in the write.
    for (auto r : _reads) {
      for (auto k : r.second) 
    	all_read_indexes.push_back(k);
    }
    std::sort(wacc.begin(), wacc.end());
    std::sort(all_read_indexes.begin(), all_read_indexes.end());

    std::vector<int> diff;
    std::set_difference(all_read_indexes.begin(), all_read_indexes.end(),
			wacc.begin(), wacc.end(),
			std::back_inserter(diff));
    
    bool hasDotProduct = hasDuplicatedIndexesOnly(diff) && (diff.size() == 2);

    blaskernels::Gemm * dp_infos = new blaskernels::Gemm();
    return {isReduction && hasDotProduct, dp_infos};

  }

  /* Conditions for contraction: C = A . B
     1. There is a reduction
     2. None of the access functions have redundant iterators
     3. All iterators in A not appearing in C do apepar in B.
     4. The remaining iterators of A and B == those of C. */
  
  std::pair<bool, blaskernels::Gemm*>
  findContractionAccess(isl::ctx ctx, isl::union_map reads, isl::union_map writes) {
    std::vector<std::vector<int>> diffs;

    // The matching occurs in the reconstruct function. 
    // the idea is just to collect and reconstruct access function
    // to analyze them in order to determine if there is a contraction 
    // or not. Integer values representing dimensions (e.g. 0 for 1st dimention,
    // 1 for 2nd dimension, ect) are the informations that are collected.
    auto _reads = reconstruct(ctx, reads);
    auto _writes = reconstruct(ctx, writes);

    auto wname = _writes.begin()->first;
    auto wacc = _writes.begin()->second;

    auto isReduction = (_reads.find(wname) != _reads.end()) && 
      (wacc == _reads[wname]);

    // We don't want the inductive variable among reads anymore.
    _reads.erase(wname);

    std::vector<int> all_read_indexes;
    // We need to ensure that each access functions have 
    // non redundant iterators.
    bool noRedundancy = hasNoRedundancy(wacc);
    for (auto r : _reads) {
      for (auto k : r.second) 
  	all_read_indexes.push_back(k);
      noRedundancy = noRedundancy && hasNoRedundancy(r.second);
    }

    // Sorting is necessary to be able to use
    // std::set_difference. 
    std::sort(wacc.begin(), wacc.end());
    std::sort(all_read_indexes.begin(), all_read_indexes.end());

    std::vector<int> diff;
    std::set_difference(all_read_indexes.begin(), all_read_indexes.end(),
			wacc.begin(), wacc.end(),
			std::back_inserter(diff));

    // Now what remains in diff must be duplicated indexes only.
    bool hasContractionAxes = hasDuplicatedIndexesOnly(diff);

    std::vector<std::pair<AccessFunction, AccessFunction>> contractionPairs;
    
    // Collect individual pairs of contractions.
    // This could be useful if, in the presence of 
    // more than two operands, due to associativity properties
    // there are different choices of contractions.
    // However, the following collects redundant results, i.e., 
    // {1, 2} and {2, 1}.
    for (auto r1 : _reads) {
      for (auto r2 : _reads) {
        if (r1.second != r2.second) {
	  if (existsContractionBetween(r1.second, r2.second) == true) {
            contractionPairs.push_back({r1, r2}); 
          } 
        }
      }
    }
  
  
    blaskernels::Gemm * c_infos = new blaskernels::Gemm();
    return {isReduction && noRedundancy && hasContractionAxes, c_infos};
  }


  // bool findTransposeAccess(isl::ctx ctx, isl::union_map reads, isl::union_map writes) {
  //   auto _i = placeholder(ctx);
  //   auto _j = placeholder(ctx);
  //   auto _ii = placeholder(ctx);
  //   auto _jj = placeholder(ctx);
  //   auto localReads = allOf(access(_i, _j));
  //   auto localWrites = allOf(access(_ii, _jj));

  //   auto matchReads = match(reads, localReads);
  //   auto matchWrites = match(writes, localWrites);

  //   if ((matchReads.size() == 1u) && (matchWrites.size() == 1u)) {
  //     // This far, at least we ensure that there is only one read 
  //     // and one write. Then we need to make sure that the correspond
  //     // to a transposition
  //     int i1 = matchReads[0][_i].payload().inputDimPos_;
  //     int j1 = matchReads[0][_j].payload().inputDimPos_;
  //     auto i2 = matchWrites[0][_ii].payload().inputDimPos_;
  //     auto j2 = matchWrites[0][_jj].payload().inputDimPos_;
  //     bool isMatch = ((i1 == j2) && (i2 == j1));
  //     return isMatch;
  //   } else {
  //     return false;
  //   }



  // bool findAxpyAccess(isl::ctx ctx, isl::union_map reads, isl::union_map writes) {
  //   auto _i = placeholder(ctx);
  //   auto _ii = placeholder(ctx);

  //   auto localReads = allOf(access(_i));
  //   auto localWrites = allOf(access(_ii));

  //   auto matchReads = match(reads, localReads);
  //   auto matchWrites = match(writes, localWrites);

  //   if ((matchReads.size() == 2u) && (matchWrites.size() == 1u)) {
  //     int i = matchReads[0][_i].payload().inputDimPos_;
  //     int i1 = matchReads[1][_i].payload().inputDimPos_;
  //     int ii = matchWrites[0][_ii].payload().inputDimPos_;
  //     // If I understand well, at this point, we should know that 
  //     // both i occurences are equals, otherwise there would be no
  //     // match. So testing with i should be enough.
  //     bool isMatch = ((ii == 0) && (ii == i)); 
  //     return isMatch;
  //   } else {
  //     return false;
  //   }
  // }


  // bool findDotProductAccess(isl::ctx ctx, isl::union_map reads, isl::union_map writes) {
  //   auto _i = placeholder(ctx);
  //   auto localReads = allOf(access(_i));
  //   auto matchReads = match(reads, localReads);
  //   if ((matchReads.size() == 2u)) {
  //     int i = matchReads[0][_i].payload().inputDimPos_;
  //     int i2 = matchReads[1][_i].payload().inputDimPos_;
  //     auto localWrite = writes.range().unwrap();
  //     auto writeSpaceDim = localWrite.dim(isl::dim::out);
  //     // Another condition is the that the scalar variable
  //     // should be inductive.
  //     // So writes.is_subset(reads) is included as a condition.
  //     bool isMatch = (i == i2) && (writeSpaceDim == 0) && (writes.is_subset(reads));
  //     return isMatch;
  //   } else {
  //     return false;
  //   }
  // }

  // }
} // namespace blaskernelsmatchers

