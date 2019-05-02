#include "bm_matchers.h"



namespace blasmatchers {


  /// Add nodes for copying to the device before "node"
  static isl::schedule_node addCopyToDevice(isl::schedule_node node) {

    isl::space space;
    isl::union_set domain;
    isl::schedule_node graft;

    space = isl::space(node.get_ctx(), 0, 0);
    space = space.set_tuple_name(isl::dim::set, "copy_to_device");
    domain = isl::union_set(isl::set::universe(space));
    graft = isl::schedule_node::from_domain(domain);

    node = node.graft_before(graft);

    return node;
  }

  /// Add nodes for copying from the device after "node"
  static isl::schedule_node addCopyFromDevice(isl::schedule_node node) {

    isl::space space;
    isl::union_set domain;
    isl::schedule_node graft;

    space = isl::space(node.get_ctx(), 0, 0);
    space = space.set_tuple_name(isl::dim::set, "copy_from_device");
    domain = isl::union_set(isl::set::universe(space));
    graft = isl::schedule_node::from_domain(domain);

    node = node.graft_after(graft);

    return node;
  }


  /// Add nodes to delimiting a kernel that will be swapped with a function
  /// call to cuBLAS.
  static isl::schedule_node addKernelBoundaries(isl::schedule_node node) {

    isl::space space;
    isl::union_set domain;
    isl::schedule_node graft;

    space = isl::space(node.get_ctx(), 0, 0);
    space = space.set_tuple_name(isl::dim::set, "kernel_start");
    domain = isl::union_set(isl::set::universe(space));
    graft = isl::schedule_node::from_domain(domain);

    node = node.graft_before(graft);

    space = isl::space(node.get_ctx(), 0, 0);
    space = space.set_tuple_name(isl::dim::set, "kernel_end");
    domain = isl::union_set(isl::set::universe(space));
    graft = isl::schedule_node::from_domain(domain);

    node = node.graft_after(graft);

    return node;
  }

  /// Add node for inserting cuBLAS handles - start-up
  static isl::schedule_node addCuBLASHandleStartUp(isl::schedule_node node) {

    isl::space space;
    isl::union_set domain;
    isl::schedule_node graft;

    space = isl::space(node.get_ctx(), 0, 0);
    space = space.set_tuple_name(isl::dim::set, "cuBLAS_manage_init");
    domain = isl::union_set(isl::set::universe(space));
    graft = isl::schedule_node::from_domain(domain);

    node = node.graft_before(graft);
    return node;
  }

  /// Add node for inserting cuBLAS handles - tear-down
  static isl::schedule_node addCuBLASHandleTearDown(isl::schedule_node node) {
  
    isl::space space;
    isl::union_set domain;
    isl::schedule_node graft;

    space = isl::space(node.get_ctx(), 0, 0);
    space = space.set_tuple_name(isl::dim::set, "cuBLAS_tear_down");
    domain = isl::union_set(isl::set::universe(space));
    graft = isl::schedule_node::from_domain(domain);

    node = node.graft_after(graft);
    return node;
  }

  static isl::schedule_node addArraysTearDown(isl::schedule_node node) {

    isl::space space;
    isl::union_set domain;
    isl::schedule_node graft;

    space = isl::space(node.get_ctx(), 0, 0);
    space = space.set_tuple_name(isl::dim::set, "arrays_tear_down");
    domain = isl::union_set(isl::set::universe(space));
    graft = isl::schedule_node::from_domain(domain);

    node = node.graft_after(graft);
    return node;
  }



  static std::vector<GpuArrayInfo> collectArrayInfo(Scop scop) {

    std::vector<GpuArrayInfo> res;

    isl::union_set arrays;
    isl::union_map accesses;

    std::vector<isl::set> arraysAsSet; 
    std::vector<isl::map> accessesAsMap;

    isl::union_map reads = scop.reads;
    accesses = reads;
    arrays = reads.range();
    isl::union_map writes = scop.mustWrites;
    accesses = accesses.unite(writes);

    arrays = arrays.unite(writes.range()); 

    arrays = arrays.coalesce();
    accesses = accesses.coalesce();
    arrays.foreach_set([&arraysAsSet](isl::set s) { arraysAsSet.push_back(s); });
    accesses.foreach_map([&accessesAsMap](isl::map m) { accessesAsMap.push_back(m); });

  
   
    for(size_t i = 0; i < arraysAsSet.size(); ++i) {
      GpuArrayInfo ga;
      ga.name = getAccessName(arraysAsSet[i]);
      for(int j = 0; j < scop.n_array; ++j) {
	std::string arrayName = getAccessName(scop.arrays[j].extent);
	if(arrayName.compare(ga.name) == 0) {
	  ga.n_index = getAccessIndexes(arraysAsSet[i]);
	  ga.element_type = scop.arrays[j].element_type;
	  ga.extent = scop.arrays[j].extent;
	  //for(size_t u = 0; u < accessesAsMap.size(); ++u) {
	  //  if(getAccessName(accessesAsMap[u]).compare(ga.name)) {
	  //    ga.accesses.push_back(accessesAsMap[u]);
	  //  }
	  //}
	} 
      }
      res.push_back(ga);
    }

    for(size_t i = 0; i < res.size(); ++i) {
      for(size_t j = 0; j < accessesAsMap.size(); ++j) {
	if(getAccessName(accessesAsMap[j]).compare(res[i].name) == 0) {
	  res[i].accesses.push_back(accessesAsMap[j]);
	}
      }
    } 

    return res;
  }



  static inline isl::union_map
  filterOutCarriedDependences(isl::union_map dependences,
			      isl::schedule_node node) {
    auto partialSchedule = node.get_prefix_schedule_multi_union_pw_aff();
    return dependences.eq_at(partialSchedule);
  }

  static bool canMerge(isl::schedule_node parentBand,
		       isl::union_map dependences) {
    // Permutability condition: there are no negative distances along the
    // dimensions that are not carried until now by any of dimensions.
    auto t1 = parentBand.band_get_partial_schedule();
    auto t2 = parentBand.child(0).band_get_partial_schedule();
    auto schedule = isl::union_map::from(t1.flat_range_product(t2));
    auto scheduleSpace = isl::set(schedule.range()).get_space();
    auto positiveOrthant =
      isl::set(isl::basic_set::positive_orthant(scheduleSpace));
    dependences = filterOutCarriedDependences(dependences, parentBand);

    return dependences.apply_domain(schedule)
      .apply_range(schedule)
      .deltas()
      .is_subset(positiveOrthant);
  }

  static inline isl::schedule_node
  rebuild(isl::schedule_node node,
	  const builders::ScheduleNodeBuilder &replacement) {
    // this may not be always legal...
    node = node.cut();
    node = replacement.insertAt(node);
    return node;
  }

  isl::schedule_node
  replaceRepeatedly(isl::schedule_node node,
		    const matchers::ScheduleNodeMatcher &pattern,
		    const builders::ScheduleNodeBuilder &replacement) {
    while (matchers::ScheduleNodeMatcher::isMatching(pattern, node)) {
      node = rebuild(node, replacement);
    }
    return node;
  }

  isl::schedule_node
  replaceDFSPreorderRepeatedly(isl::schedule_node node,
			       const matchers::ScheduleNodeMatcher &pattern,
			       const builders::ScheduleNodeBuilder &replacement) {
    node = replaceRepeatedly(node, pattern, replacement);
    for (int i = 0; i < node.n_children(); ++i) {
      node = replaceDFSPreorderRepeatedly(node.child(i), pattern, replacement)
	.parent();
    }
    return node;
  }


  isl::schedule_node mergeIfTilable(isl::schedule_node node,
				    isl::union_map dependences) {
    isl::schedule_node parent, child, grandchild;

    auto canMergeCaptureChild = [&child, dependences](isl::schedule_node node) {
      if (canMerge(node.parent(), dependences)) {
	child = node;
	return true;
      }
      return false;
    };

    auto matcher = [&]() {
      using namespace matchers;
      // clang-format off
      return band(parent,
		  band(canMergeCaptureChild,
		       anyTree(grandchild)));
      // clang-format on
    }();

    // Use lambdas to lazily initialize the builder with the nodes and values yet
    // to be captured by the matcher.
    auto declarativeMerger = builders::ScheduleNodeBuilder();
    {
      using namespace builders;
      auto schedule = [&]() {
	auto descr =
	BandDescriptor(parent.band_get_partial_schedule().flat_range_product(
									     child.band_get_partial_schedule()));
	descr.permutable = 1;
	return descr;
      };
      auto st = [&]() { return subtreeBuilder(grandchild); };
      declarativeMerger = band(schedule, subtree(st));
    }

    return replaceDFSPreorderRepeatedly(node, matcher, declarativeMerger);
  }

  static isl::union_map computeAllDependences(const Scop &scop) {
    // For the simplest possible dependence analysis, get rid of reference tags.
    auto reads = scop.reads.domain_factor_domain();
    auto mayWrites = scop.mayWrites.domain_factor_domain();
    auto mustWrites = scop.mustWrites.domain_factor_domain();

    // False dependences (output and anti).
    // Sinks are writes, sources are reads and writes.
    auto falseDepsFlow = isl::union_access_info(mayWrites.unite(mustWrites))
      .set_may_source(mayWrites.unite(reads))
      .set_must_source(mustWrites)
      .set_schedule(scop.schedule)
      .compute_flow();

    isl::union_map falseDeps = falseDepsFlow.get_may_dependence();

    // Flow dependences.
    // Sinks are reads and sources are writes.
    auto flowDepsFlow = isl::union_access_info(reads)
      .set_may_source(mayWrites)
      .set_must_source(mustWrites)
      .set_schedule(scop.schedule)
      .compute_flow();

    isl::union_map flowDeps = flowDepsFlow.get_may_dependence();

    return flowDeps.unite(falseDeps);
  }




  std::map<int, isl::union_map>
  restructureUnionMap(isl::ctx ctx,
		      isl::union_map umap) {
    std::map<int, isl::union_map> rumap;
    // I am not sure if this is a necessary step
    // but we'll do it this way for now.
    isl::map_list mapList = umap.get_map_list();
    auto map = isl::union_map(ctx, mapList.get_at(0).to_str());

    rumap[0] = map;
    int count = 0;

    for (int i = 1; i < mapList.size(); ++i) {
      auto thisMap = mapList.get_map(i);
      if (rumap[count].domain().is_equal(thisMap.domain())) {
	rumap[count] = rumap[count].add_map(thisMap);
      } else {
	count += 1;
	rumap[count] = isl::union_map(ctx, thisMap.to_str());
      }
    }
    return rumap;
  }

  // Associate reads and writes from same scop
  Accesses
  associateRW(UmapPair reads, UmapPair writes) {
    Accesses rw;

    if (reads.size() != writes.size()) {
      std::cout << "error" << std::endl;
    } else {
      for (auto w : writes) {
	for (auto r : reads) {
	  if (w.second.domain().is_equal(r.second.domain())) {
	    rw.push_back(std::make_pair(w.second, r.second));
	  }
	}
      }
    }
    return rw;
  }


  Accesses
  restructureScop(isl::ctx ctx,
		  isl::union_map reads,
		  isl::union_map writes) {

    return associateRW(restructureUnionMap(ctx, reads),
		       restructureUnionMap(ctx, writes));

  }



  // If this function is called, then &subnode will necessary be 
  // updated with a node because we ensured that the domain was 
  // indeed a subset of the Scop's domain.
  void
  searchRootNodeMatchingDomain(isl::schedule_node node, 
			       isl::union_set domain,
			       isl::schedule_node &subnode) {
    if (node.get_domain().is_equal(domain)) {
      subnode = node;		
    }
    else {
      for (int i = 0; i < node.n_children(); ++i) 
	searchRootNodeMatchingDomain(node.get_child(i), domain, subnode);
    }
  }


  std::vector<blaskernels::BlasKernels*>
  findKernel(isl::ctx ctx, 
	     Scop scop, 
	     isl::union_map reads, 
	     isl::union_map writes) {
  
    std::vector<blaskernels::BlasKernels*> k;

    /* For each find* function, the process is as follows. 
       Match access functions with access matchers available in 
       the collection. If there is a match, then match the enclosing
       loop nest with the hypothetical kernel to confirm that it is 
       indeed the kernel found. 

       However, perhaps findAnyDotProduct is better off being implemented 
       with matching the tree first before the access functions..
    */ 

    if (findGemm(ctx, scop, reads, writes).first == true) {
      k.push_back(findGemm(ctx, scop, reads, writes).second);
    }
    if (findTransposeGemm(ctx, scop, reads, writes).first == true) {
      k.push_back(findTransposeGemm(ctx, scop, reads, writes).second);
    }
    if (findBatchGemm(ctx, scop, reads, writes).first == true) {
      k.push_back(findBatchGemm(ctx, scop, reads, writes).second);
    }
    if (findTransposeBatchGemm(ctx, scop, reads, writes).first == true) {
      k.push_back(findTransposeBatchGemm(ctx, scop, reads, writes).second);
    }
    if (findAnyDotProduct(ctx, scop, reads, writes).first == true) {
      k.push_back(findAnyDotProduct(ctx, scop, reads, writes).second);
    }
    if (findContraction(ctx, scop, reads, writes).first == true) {
      k.push_back(findContraction(ctx,scop, reads, writes).second);
    }

    return k;
  }


  std::vector<std::pair<ScopStmt, std::vector<blaskernels::BlasKernels*>>>
    findPatterns(isl::ctx ctx, Scop scop) {

    isl::union_map _reads = scop.reads.curry();
    isl::union_map _writes = scop.mustWrites.curry();

    // Restructure _reads and _writes into pairs of
    // <read, write> representing a single statement
    auto accesses = restructureScop(ctx, _reads, _writes);

    std::vector<std::pair<ScopStmt, std::vector<blaskernels::BlasKernels*>>> fk;
    
    for (auto acc : accesses) {
      auto writes = acc.first;
      auto reads = acc.second;
      // For each statement represented by <reads, writes>, search for matches.
      auto bks = findKernel(ctx, scop, reads, writes);
      auto pair = std::make_pair(acc, bks);
      fk.push_back(pair);
    }
    return fk;
  }


  /* The following implementations of matcher for Gemm and 
     variants are directly based on what was found in the file 
     test_transformers.cc. 
     Having done very few changes, one will notice that 
     isl::schedule_node *_node is actually not used at all. 
     Feel free to clean it up or to use it. */

  std::pair<bool, blaskernels::Gemm*>
  findGemm(isl::ctx ctx, 
	   Scop scop,
	   isl::union_map reads,
	   isl::union_map writes) {
    isl::schedule_node newnode;
    auto isGemm = findGemmAccess(ctx, reads, writes);

    if (isGemm.first == true) {
      // At this point it doesn't matter whether we use the
      // domain of reads or writes, it's the same
      auto accessdom = reads.domain();
      auto scheddom = scop.schedule.get_domain();

      if (accessdom.is_subset(scheddom)) {
	isl::schedule_node root = scop.schedule.get_root();
	isl::schedule_node subnode;
	searchRootNodeMatchingDomain(root, accessdom, subnode);
	isl::schedule_node *_node;
	auto dependences = computeAllDependences(scop);
	subnode = mergeIfTilable(subnode, dependences);
	isGemm.first = findGemmTree(subnode, _node);

	if (isGemm.first == true) {
	  newnode = root.root().child(0);
        
	  isl::union_set domain = newnode.get_domain();

	  newnode = addCuBLASHandleStartUp(newnode);
	  newnode = addArraysTearDown(newnode);
	  newnode = addCuBLASHandleTearDown(newnode);
	  newnode = addCopyToDevice(newnode);
	  newnode = addCopyFromDevice(newnode);
	  newnode = addKernelBoundaries(newnode);
       
	  isGemm.second->setType(blaskernels::gemm);
	  isGemm.second->setScheduleNode(newnode);
	  isGemm.second->array_infos = collectArrayInfo(scop);
	  isGemm.second->fill(false); // does not have transposed B.
	  isGemm.second->setDataType();
	}
      } 
    } 
    return isGemm;
  }


  std::pair<bool, blaskernels::Gemm*>  findTransposeGemm(isl::ctx ctx,
							 Scop scop,
							 isl::union_map reads,
							 isl::union_map writes) {
														
    auto isTransposeGemm = findTransposeGemmAccess(ctx, reads, writes);
    isl::schedule_node newnode;
    if (isTransposeGemm.first == true) {
      auto accessdom = reads.domain();
      auto scheddom = scop.schedule.get_domain();

      if (accessdom.is_subset(scheddom)) {
	isl::schedule_node root = scop.schedule.get_root();
	isl::schedule_node subnode;
	searchRootNodeMatchingDomain(root, accessdom, subnode);
	isl::schedule_node *_node;
	auto dependences = computeAllDependences(scop);
	subnode = mergeIfTilable(subnode, dependences);
	isTransposeGemm.first = findGemmTree(subnode, _node);


	if (isTransposeGemm.first == true) {


	  newnode = root.root().child(0);
        
	  isl::union_set domain = newnode.get_domain();

	  newnode = addCuBLASHandleStartUp(newnode);
	  newnode = addArraysTearDown(newnode);
	  newnode = addCuBLASHandleTearDown(newnode);
	  newnode = addCopyToDevice(newnode);
	  newnode = addCopyFromDevice(newnode);
	  newnode = addKernelBoundaries(newnode);
       
	  isTransposeGemm.second->setType(blaskernels::transposeGemm);
	  isTransposeGemm.second->setScheduleNode(newnode);
	  isTransposeGemm.second->array_infos = collectArrayInfo(scop);
	  isTransposeGemm.second->fill(true); // has transposed B.
	  isTransposeGemm.second->setDataType();


	}


      }
    }
    return isTransposeGemm;
  }							 
									 

  std::pair<bool, blaskernels::Gemm*> 
  findBatchGemm(isl::ctx ctx, 
		Scop scop,
		isl::union_map reads,
		isl::union_map writes) {
    auto isBatchGemm = findBatchGemmAccess(ctx, reads, writes);
    isl::schedule_node newnode;
    if (isBatchGemm.first == true) {
      auto accessdom = reads.domain();
      auto scheddom = scop.schedule.get_domain();

      if (accessdom.is_subset(scheddom)) {
	isl::schedule_node root = scop.schedule.get_root();
	isl::schedule_node subnode;
	searchRootNodeMatchingDomain(root, accessdom, subnode);
	isl::schedule_node *_node;
	auto dependences = computeAllDependences(scop);
	subnode = mergeIfTilable(subnode, dependences);
	isBatchGemm.first = findGemmTree(subnode, _node);

        if (isBatchGemm.first == true) {
          newnode = root.root().child(0);
        
	  isl::union_set domain = newnode.get_domain();

	  newnode = addCuBLASHandleStartUp(newnode);
	  newnode = addArraysTearDown(newnode);
	  newnode = addCuBLASHandleTearDown(newnode);
	  newnode = addCopyToDevice(newnode);
	  newnode = addCopyFromDevice(newnode);
	  newnode = addKernelBoundaries(newnode);
       
	  isBatchGemm.second->setType(blaskernels::transposeGemm);
	  isBatchGemm.second->setScheduleNode(newnode);
   
	  isBatchGemm.second->array_infos = collectArrayInfo(scop);
   
	  isBatchGemm.second->fill(false); // has no transposed B.
	  isBatchGemm.second->setDataType();
 
        }
      }
    }
    return isBatchGemm;
  }

  std::pair<bool, blaskernels::Gemm*> 
  findTransposeBatchGemm(isl::ctx ctx, 
			 Scop scop,
			 isl::union_map reads,
			 isl::union_map writes) {
    auto isBatchGemm = findTransposeBatchGemmAccess(ctx, reads, writes);
    isl::schedule_node newnode;
    if (isBatchGemm.first == true) {
      auto accessdom = reads.domain();
      auto scheddom = scop.schedule.get_domain();

      if (accessdom.is_subset(scheddom)) {
	isl::schedule_node root = scop.schedule.get_root();
	isl::schedule_node subnode;
	searchRootNodeMatchingDomain(root, accessdom, subnode);
	isl::schedule_node *_node;
	auto dependences = computeAllDependences(scop);
	subnode = mergeIfTilable(subnode, dependences);
	isBatchGemm.first = findGemmTree(subnode, _node);

        if (isBatchGemm.first == true) {
          newnode = root.root().child(0);
        
	  isl::union_set domain = newnode.get_domain();

	  newnode = addCuBLASHandleStartUp(newnode);
	  newnode = addArraysTearDown(newnode);
	  newnode = addCuBLASHandleTearDown(newnode);
	  newnode = addCopyToDevice(newnode);
	  newnode = addCopyFromDevice(newnode);
	  newnode = addKernelBoundaries(newnode);
       
	  isBatchGemm.second->setType(blaskernels::transposeBatchGemm);
	  isBatchGemm.second->setScheduleNode(newnode);
   
	  isBatchGemm.second->array_infos = collectArrayInfo(scop);
   
	  isBatchGemm.second->fill(true); // has transposed B.
	  isBatchGemm.second->setDataType();
 
        }
      }
    }
    return isBatchGemm;
  }


  std::pair<bool, blaskernels::Gemm*> 
  findContraction(isl::ctx ctx,
		  Scop scop,
		  isl::union_map reads,
		  isl::union_map writes) {
    auto isContraction = findContractionAccess(ctx, reads, writes);
    if (isContraction.first == true) {
      isContraction.second->setType(blaskernels::contraction);
    }
    return isContraction;
  }

  std::pair<bool, blaskernels::Gemm*>
  findAnyDotProduct(isl::ctx ctx, Scop scop, isl::union_map reads, isl::union_map writes) {
    auto hasDotProduct = findAnyDotProductAccess(ctx, reads, writes);
 
    if (hasDotProduct.first == true) {
      auto accessdom = reads.domain();
      auto scheddom = scop.schedule.get_domain();
      hasDotProduct.second->setType(blaskernels::dotProduct);

      //     if (accessdom.is_subset(scheddom)) {
      // isl::schedule_node root = scop.schedule.get_root();
      // isl::schedule_node subnode;
      // searchRootNodeMatchingDomain(root, accessdom, subnode);
      // isl::schedule_node *_node;
      // hasDotProduct.first = findDotProductTree(subnode, _node);
 
      //     }
    }
    return hasDotProduct;
  }



  /* The following has been commented out, as they need to have 
     their return values adapted to std::pair<bool, blaskernels::ProperClass*> */


  // bool
  // findTranspose(isl::ctx ctx,
  // 									 			Scop scop,
  // 									 			isl::union_map reads,
  // 									 			isl::union_map writes) {
  // 	bool isTranspose = findTransposeAccess(ctx, reads, writes);
  // 	if (isTranspose == true) {
  // 		//std::cout << "Found Transpose" << std::endl;
  // 		// At this point it doesn't matter whether we use the
  // 		// domain of reads or writes, it's the same
  // 		auto accessdom = reads.domain();
  // 		auto scheddom = scop.schedule.get_domain();
	
  // 		if (accessdom.is_subset(scheddom)) {
  // 			isl::schedule_node root = scop.schedule.get_root();
  // 			isl::schedule_node *_node;
  // 			isl::schedule_node subnode;
  // 			searchRootNodeMatchingDomain(root, accessdom, subnode);
  // 			// Update isTranspose, perhaps the result is False, 
  // 			// then the functions shall return false.
  // 			isTranspose = findTransposeTree(subnode, _node); 
  // 		}
  // 	}
  // 	return isTranspose;
  // }
	


  // bool 
  // findAxpy(isl::ctx ctx, 
  // 				 Scop scop,
  // 				 isl::union_map reads,
  // 				 isl::union_map writes) {
  // 	bool isAxpy = findAxpyAccess(ctx, reads, writes);
  // 	if (isAxpy == true) {
  // 		auto accessdom = reads.domain();
  // 		auto scheddom = scop.schedule.get_domain();

  // 		if (accessdom.is_subset(scheddom)) {
  // 			isl::schedule_node root = scop.schedule.get_root();
  // 			isl::schedule_node subnode;
  // 			searchRootNodeMatchingDomain(root, accessdom, subnode);
  // 			isl::schedule_node *_node;
  // 			isAxpy = findAxpyTree(subnode, _node);
  // 		}
  // 	}
  // 	return isAxpy;
  // }

  // bool 
  // findDotProduct(isl::ctx ctx, 
  // 							 Scop scop,
  // 							 isl::union_map reads,
  // 							 isl::union_map writes) {
  // 	bool isDotProduct = findDotProductAccess(ctx, reads, writes);
  // 	if (isDotProduct == true) {
  // 		auto accessdom = reads.domain();
  // 		auto scheddom = scop.schedule.get_domain();

  // 		if (accessdom.is_subset(scheddom)) {
  // 			isl::schedule_node root = scop.schedule.get_root();
  // 			isl::schedule_node subnode;
  // 			searchRootNodeMatchingDomain(root, accessdom, subnode);
  // 			isl::schedule_node *_node;
  // 			isDotProduct = findDotProductTree(subnode, _node);
  // 		}
  // 	}
  // 	return isDotProduct;
  // }


} // namespace blasMathers
