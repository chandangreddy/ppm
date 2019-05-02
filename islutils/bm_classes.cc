#include "bm_classes.h"


namespace blaskernels {




  std::string getAccessName(isl::map m) {
    return m.range().get_tuple_id().get_name();
  }

  std::string getAccessName(isl::set s) {
    return s.get_tuple_id().get_name();
  }

  unsigned getAccessIndexes(isl::map m) {
    return m.dim(isl::dim::out);
  }

  unsigned getAccessIndexes(isl::set s) {
    return s.dim(isl::dim::out);
  }



  static isl::union_map applySchedule(isl::union_map schedule,
				      isl::union_map accesses) {
    return accesses.apply_domain(schedule);
  }

  static std::string createIndent(int tab) {
    std::string result;
    for(int i = 0; i < tab; ++i) {
      result += " ";
    }
    return result;
  }


  static std::string beginMain() {
    return "\n\nint main()";
  }

  static std::string endMain(int tab) {
    return createIndent(tab) + "return 0;\n";
  }

  std::string printCudaHeader() {
    std::string s = "";
    s+= "/* Includes system */\n";
    s+= "#include <stdio.h>\n";
    s+= "#include <stdlib.h>\n\n";
    s+= "/* Includes cuda */\n";
    s+= "#include <cublas_v2.h>\n";
    s+= "#include <cuda_runtime.h>\n";
    s+= "#include <helper_cuda.h>\n";

    s += beginMain();
    return s;
  }




  static std::string macroCuBLASHandleInit(int tab) {
    std::string s;
    std::string indent = createIndent(tab);
    s+= "\n";
    s+= indent + "cublasStatus_t cublasStat = cublasCreate(&handle);\n";
    s+= indent + "if (cublasStat != CUBLAS_STATUS_SUCCESS) {\n";
    s+= indent + createIndent(2) + "return 0;\n";
    s+= indent + "}\n";
    // s+= indent + "// Set the math mode to allow cuBLAS to use Tensor Cores:\n";
    // s+= indent + "cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);";
    return s;
  }

  static std::string macroCuBLASHandleTearDown(int tab) {
 
    std::string s;
    std::string indent = createIndent(tab);
    s+= "\n";
    s+= indent + "/* Shutdown */\n";
    s+= indent + "cublasStat = cublasDestroy(handle);\n";

    return s;
  }

  /// print declaration for the device array.
  static std::string declareDeviceArray(GpuArrayInfo g) {
  
    std::string result = "";
    result += g.element_type + " ";
    result += "*dev_";
    result += g.name;
    result += ";\n";
    return result;
  }

  static std::string freeDeviceArray(GpuArrayInfo g) {
    return "free(" + g.name + ");\n";
  }

  /// does the gpu array need to be allocated on the device?
  /// If it is a read-only scalar, then it will be passed
  /// as argument to the function call.
  static bool requireAllocation(GpuArrayInfo g) {
  
    if(g.n_index == 0) {
      return false;
    }
    return true;
  }

  /// print a declaration for the device array corresponding to
  /// "array"
 
  /// return arrays size

  std::vector<int> getBounds(GpuArrayInfo g) {
    isl::set extent = g.extent;
    std::vector<int> bounds;

    for(size_t i = 0; i < g.n_index; ++i) {

      auto allPoints = 
	isl::map::from_domain_and_range(extent, extent);
      isl::pw_aff min = allPoints.dim_min(i);
      isl::pw_aff max = allPoints.dim_max(i);


      isl::val min_val;
      isl::val max_val;
 
      min.foreach_piece([&](isl::set s, isl::aff aff) -> void {
	  min_val = aff.get_constant_val(); 
	});
      max.foreach_piece([&](isl::set s, isl::aff aff) -> void {
	  max_val = aff.get_constant_val();
	});

      max_val = max_val.sub(min_val);
      int bound = atoi(max_val.to_str().c_str());
      bounds.push_back(bound);
    }

    return bounds;
  }

  static std::string getNumberOfElementArray(GpuArrayInfo g) {
  
    isl::set extent = g.extent;
    std::vector<int> bounds;

    for(size_t i = 0; i < g.n_index; ++i) {

      auto allPoints = 
	isl::map::from_domain_and_range(extent, extent);
      isl::pw_aff min = allPoints.dim_min(i);
      isl::pw_aff max = allPoints.dim_max(i);


      isl::val min_val;
      isl::val max_val;
 
      min.foreach_piece([&](isl::set s, isl::aff aff) -> void {
	  min_val = aff.get_constant_val(); 
	});
      max.foreach_piece([&](isl::set s, isl::aff aff) -> void {
	  max_val = aff.get_constant_val();
	});

      max_val = max_val.sub(min_val);
      int bound = atoi(max_val.to_str().c_str());
      bounds.push_back(bound);
    }

    unsigned numberOfElement = 1; 
    for(size_t i = 0; i < bounds.size(); ++i) {
      numberOfElement *= static_cast<unsigned>(bounds[i]);
    }

    return conversion::to_string(numberOfElement);
  }



  /// print code for initializing the device for the execution.
  /// This includes declaring locally defined variables as well as
  /// declaring and allocating the required copies of arrays on device.

  std::pair<isl::map, bool> 
  findAccess(std::vector<GpuArrayInfo> &gv, int x, int y, isl::union_map s, int batch) {
  

    std::vector<isl::map> allAccesses;
    for(size_t i = 0; i < gv.size(); ++i) {
      for(size_t j = 0; j < gv[i].accesses.size(); ++j) {
	allAccesses.push_back(gv[i].accesses[j]);
      }
    }

    std::vector<int> indexesDiscovered;


    for(size_t i = 0; i < allAccesses.size(); ++i) {
      isl::union_map scheduledAccess = isl::union_map(allAccesses[i]);
      scheduledAccess.apply_domain(s);
      isl::map m = isl::map::from_union_map(scheduledAccess);
      isl::pw_multi_aff multiAff = isl::pw_multi_aff::from_map(m);
     
      if (batch != -1) {

	if(m.dim(isl::dim::out) != 3) {
	  continue;
	}
	if(m.dim(isl::dim::in) != 4) {
	  continue;
	}
      } else {
	if(m.dim(isl::dim::out) != 2) {
	  continue;
	}
	if(m.dim(isl::dim::in) != 3) {
	  continue;
	}

      }
      // skip if sched and access to not belong to the same 
      // stmt (not a good implementation).
      isl::map schedAsMap = isl::map::from_union_map(s);
      if(m.domain().unwrap().domain().get_tuple_id().get_name()
	 .compare(schedAsMap.get_tuple_id(isl::dim::in).get_name()) != 0) {
	continue;
      }

      for(size_t ot = 0; ot < m.dim(isl::dim::out); ++ot) {
	isl::pw_aff pwa = multiAff.get_pw_aff(ot);
	pwa.foreach_piece([&](isl::set s, isl::aff a) -> void {
	    for(size_t in = 0; in < m.dim(isl::dim::in); ++in) {
	      isl::val v = a.get_coefficient_val(isl::dim::in, in);
	      if(v.is_one()) {

		indexesDiscovered.push_back(in);
	      }
	    }
	  });
      }
      if (batch != -1) {
	if(indexesDiscovered[1] == x && indexesDiscovered[2] == y) {
	  return std::make_pair(m, true);
	}
      } else {
	if(indexesDiscovered[0] == x && indexesDiscovered[1] == y) {
	  return std::make_pair(m, true);
	}
      }
      indexesDiscovered.erase(indexesDiscovered.begin(), 
			      indexesDiscovered.end());
    }

    return std::make_pair(nullptr, false);
  }


  int getBatchNumber(std::string id, std::vector<GpuArrayInfo> gp) {
    int batch;
    for(size_t i = 0; i < gp.size(); ++i) {
      if(gp[i].name.compare(id) != 0) {
        continue;
      }
      else {
 
        isl::set extent = gp[i].extent;
     
        auto allPoints =
          isl::map::from_domain_and_range(extent, extent);
        isl::pw_aff min = allPoints.dim_min(0);
        isl::pw_aff max = allPoints.dim_max(0);
        isl::val min_val;
        isl::val max_val;

        min.foreach_piece([&](isl::set s, isl::aff aff) -> void {
	    min_val = aff.get_constant_val();
	  });
        max.foreach_piece([&](isl::set s, isl::aff aff) -> void {
	    max_val = aff.get_constant_val();
	  });


        max_val = max_val.sub(min_val);
      
        batch  = atoi(max_val.to_str().c_str());

      }
    }

    return batch;
  }


  int getRowNumber(std::string id, std::vector<GpuArrayInfo> gp, int batch) {
    int rows;
    for(size_t i = 0; i < gp.size(); ++i) {
      if(gp[i].name.compare(id) != 0) {
        continue;
      }
      else {
 
        isl::set extent = gp[i].extent;
     
        auto allPoints =
          isl::map::from_domain_and_range(extent, extent);
	isl::pw_aff min, max;
	if (batch != -1) {
	  min = allPoints.dim_min(1);
	  max = allPoints.dim_max(1);
	} else {
	  min = allPoints.dim_min(0);
	  max = allPoints.dim_max(0);
	}
        
        isl::val min_val;
        isl::val max_val;

        min.foreach_piece([&](isl::set s, isl::aff aff) -> void {
	    min_val = aff.get_constant_val();
	  });
        max.foreach_piece([&](isl::set s, isl::aff aff) -> void {
	    max_val = aff.get_constant_val();
	  });

        max_val = max_val.sub(min_val);
        rows  = atoi(max_val.to_str().c_str());
      }
    }

    return rows;
  }

  int
  getColumnNumber(std::string id, std::vector<GpuArrayInfo> gp, int batch) {
  
    int column;
 

    for(size_t i = 0; i < gp.size(); ++i) {
      if(gp[i].name.compare(id)  != 0) {
        continue;
      }
      else {
        isl::set extent = gp[i].extent;
       
        auto allPoints =
          isl::map::from_domain_and_range(extent, extent);
	isl::pw_aff min, max;
	if (batch != -1) {
	  min = allPoints.dim_min(2);
	  max = allPoints.dim_max(2);
	} else {
	  min = allPoints.dim_min(1);
	  max = allPoints.dim_max(1);
	}
        isl::val min_val;
        isl::val max_val;

        min.foreach_piece([&](isl::set s, isl::aff aff) -> void {
	    min_val = aff.get_constant_val();
	  });
        max.foreach_piece([&](isl::set s, isl::aff aff) -> void {
	    max_val = aff.get_constant_val();
	  });

        max_val = max_val.sub(min_val);
        column = atoi(max_val.to_str().c_str());
      }
    }
  
    return column;
  }


  int 
  getLeadingDimension(std::string id, std::vector<GpuArrayInfo> gp, int batch) {
    return getRowNumber(id, gp, batch);
  }



  std::vector<std::string> getConstantInnerMostLoop(std::vector<GpuArrayInfo> gp) {
  
    std::vector<std::string> res;
    for(size_t i = 0; i < gp.size(); ++i) {
      if(gp[i].n_index != 0) {
	continue;
      }
      if(gp[i].accesses[0].dim(isl::dim::in) != 3) {
	continue;
      }
      else {
	res.push_back(gp[i].name);
      }
    }
    return res;
  }

  std::vector<std::string> getConstantInitStmt(std::vector<GpuArrayInfo> gp) {
  
    std::vector<std::string> res;
    for(size_t i = 0; i < gp.size(); ++i) {
      if(gp[i].n_index != 0) { 
	continue;
      }
      if(gp[i].accesses[0].dim(isl::dim::in) != 2) {
	continue;
      }
      else {
	res.push_back(gp[i].name);
      }
    }
    return res;
  }


  // forward declaration for test codeGenerationGPUs.
  isl::union_map addRangeId(isl::union_map umap, const std::string &tag);
  static inline isl::schedule_node
  rebuild(isl::schedule_node node,
	  const builders::ScheduleNodeBuilder &replacement);







  
  /* Functions related to class BlasKernels */
  void BlasKernels::setScheduleNode(isl::schedule_node schednode) {
    schedule_node = schednode;
  }

  void BlasKernels::setType(int kern) {
    type = kern;
  }

  void BlasKernels::setDataType() {
    auto dtype = array_infos[0].element_type;
    if (dtype == "float") {
      data_type = "S";
    }
    if (dtype == "double") {
      data_type = "D";
    }
  }    



  /* Functions for subclasses */
  /* Gemm */
  void Gemm::fillIndexInfos(int _i, int _j, int _k, int _batch, std::string _C) {
    write_var = _C;
    i = _i;
    j = _j;
    k = _k;
    if (_batch != -1) {
      batch = _batch;
    }
  }

  void Gemm::fill(bool isTranspose) {
    auto before_leaf = schedule_node.child(0).child(0);
    auto leaf = before_leaf.child(0);
    auto before_leaf_sched = before_leaf.get_prefix_schedule_union_map();
    auto sched = before_leaf_sched.intersect_domain(leaf.get_domain());
    isl::map _readFromC = findAccess(array_infos, i, j, sched, batch).first;
    
    isl::map _A = findAccess(array_infos, i, k, sched, batch).first;
    isl::map _B;
    if (isTranspose == true) {
      _B = findAccess(array_infos, j, k, sched, batch).first;
    } else {
      _B = findAccess(array_infos, k, j, sched, batch).first;
    }
    ReadFromC = getAccessName(_readFromC);
    A = getAccessName(_A);
    B = getAccessName(_B);
    if (isTranspose == true) {
      transb = "CUBLAS_OP_T";
    }
  }

  std::string BlasKernels::allocateDeviceArrays(int tab) {

    std::string result = "\n";

    for(size_t i = 0; i < array_infos.size(); ++i) {
      //skip sclar accesses.
      if(requireAllocation(array_infos[i]) == false) {
	continue;
      }
      else {
	result += createIndent(tab) + "if (cudaMalloc(reinterpret_cast<void **>(&dev_"
	  + array_infos[i].name + ")" + ", " + "sizeof(*dev_" + array_infos[i].name 
	  + ") * " + getNumberOfElementArray(array_infos[i]) + ") != cudaSuccess) {\n";
	result += createIndent(tab + 2) + "return 0;\n";
	result += createIndent(tab) + "}\n";
      }
    }
    return result;
  }

  std::string BlasKernels::copyToDeviceArrays(int tab) {
    std::string result = "\n";
    for(size_t i = 0; i < array_infos.size(); ++i) {
      if(requireAllocation(array_infos[i]) == false) {
	continue;
      } 
      else {
	result += createIndent(tab) + "cublasStat = cublasSetMatrix(";
  
	int _i, _j;
	if (batch == -1) {
	  _i = 0;
	  _j = 1;
	} else {
	  _i = 1; 
	  _j = 2;
	}

	result += conversion::to_string(getBounds(array_infos[i])[_i]);
	result += ", ";
	result += conversion::to_string(getBounds(array_infos[i])[_j]);
	result += ", ";
	result += "sizeof(*" + array_infos[i].name + ")";
	result += ", ";
	result += array_infos[i].name;
	result += ", ";
	result += conversion::to_string(getBounds(array_infos[i])[_i]);
	result += ", ";
	result += "dev_" + array_infos[i].name;
	result += ", ";
	result += conversion::to_string(getBounds(array_infos[i])[_i]) + ");\n";
      }
    }
    return result;
  }

  std::string BlasKernels::declareDeviceArrays(int tab) {
    std::string result = "\n";
    for(size_t i = 0; i < array_infos.size(); ++i) {
      // skip scalar accesses.
      if(requireAllocation(array_infos[i]) == false) {
	continue;
      }
      else {
	result += createIndent(tab) + declareDeviceArray(array_infos[i]);
      }
    }
    return result;
  }


  std::string BlasKernels::freeDeviceArrays(int tab) {
    std::string result = "\n";
    for(size_t i = 0; i < array_infos.size(); ++i) {
      // skip scalar accesses.
      if(requireAllocation(array_infos[i]) == false) {
	continue;
      }
      else {
	result += createIndent(tab) + freeDeviceArray(array_infos[i]);
      }
    }
    return result;
  }


  std::string BlasKernels::copyFromDeviceArray(int tab) {
    std::string result = "";
    for(size_t i = 0; i < array_infos.size(); ++i) {
      if(array_infos[i].name.compare(write_var) == 0) {
	result += createIndent(tab) + "cublasStat = cublasGetMatrix(";

	int _i, _j;
	if (batch == -1) {
	  _i = 0;
	  _j = 1;
	} else {
	  _i = 1; 
	  _j = 2;
	}
	result += conversion::to_string(getBounds(array_infos[i])[_i]);
	result += ", ";
	result += conversion::to_string(getBounds(array_infos[i])[_j]);
	result += ", ";
	result += "sizeof(*" + array_infos[i].name + ")";
	result += ", ";
	result += "dev_" + array_infos[i].name;
	result += ", ";
	result += conversion::to_string(getBounds(array_infos[i])[_i]);
	result += ", ";
	result += array_infos[i].name;
	result += ", ";
	result += conversion::to_string(getBounds(array_infos[i])[_j]) + ");\n";
      }
    }
    return result;
  }
    
  
  std::string Gemm::insertCallToCUBLAS(std::string c) {
    std::string fCall;
    m = conversion::to_string(getRowNumber(A, array_infos, batch));
    n = conversion::to_string(getColumnNumber(B, array_infos, batch));
    k_ = conversion::to_string(getRowNumber(B, array_infos, batch));
    lda = conversion::to_string(getLeadingDimension(A, array_infos, batch));
    ldb = conversion::to_string(getLeadingDimension(B, array_infos, batch));
    ldc = conversion::to_string(getLeadingDimension(ReadFromC, array_infos, batch));

    std::string variant = "";
    if (batch != -1) {
      variant = "gemmBatched";
    } else {
      variant = "gemm";
    }

    fCall = "cublasStat = cublas" + data_type + variant + "(handle, ";
    fCall += transa + ", ";
    fCall += transb + ", ";
    fCall += m + ", ";
    fCall += n + ", ";
    fCall += k_ + ", ";
    fCall += alpha + ", ";
    fCall += A + ", ";
    fCall += lda + ", ";
    fCall += B + ", ";
    fCall += ldb + ", ";
    fCall += beta + ", ";
    fCall += write_var + ", ";
    fCall += ldc;
    if (batch != -1) 
      fCall += ", " + conversion::to_string(getBatchNumber(ReadFromC, array_infos));
    fCall += ");\n";
    
    std::string startK = "kernel_start";
    std::string endK = "kernel_end";

    c.replace(c.find(startK),
	      c.find(endK) - c.find(startK) + endK.size(), fCall);

    return c;

  }



  /* Other functions */

  std::string codeGenGPU(isl::ast_build astBuild, isl::ast_node node,
			 pet_stmt *stmt, void *user) {

    auto t = static_cast<BlasKernels *>(user);

    auto schedule = astBuild.get_schedule();
    auto name = isl::set(schedule.domain()).get_tuple_id().get_name();

    if(name == "cuBLAS_manage_init") {
      return macroCuBLASHandleInit(2);
    }
    if(name == "cuBLAS_tear_down") {
      return macroCuBLASHandleTearDown(2);
    }
    if(name == "kernel_start") {
      return "kernel_start";
    }
    if(name == "kernel_end") {
      return "kernel_end";
    }
    if(name == "copy_from_device") {
      std::string result = "\n";
      result += t->copyFromDeviceArray(2);
      return result;
    }
    if(name == "copy_to_device") {
      std::string result = "\n";
      result += t->declareDeviceArrays(2);
      result += t->allocateDeviceArrays(2);
      result += t->copyToDeviceArrays(2);
      return result;
    }
    if (name == "arrays_tear_down") {
      std::string result = "";
      result += t->freeDeviceArrays(2);
      result += endMain(2);
      return result;
    }
    else {
      return "I will be removed :(";
    }
  }

}
