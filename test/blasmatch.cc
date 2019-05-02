#include <iostream>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <isl/ctx.h>
#include <isl/id.h>
#include <isl/val.h>
#include <isl/set.h>
#include <isl/union_set.h>
#include <isl/union_map.h>
#include <isl/aff.h>
#include <isl/flow.h>
#include <isl/options.h>
#include <isl/schedule.h>
#include <isl/ast.h>
#include <isl/id_to_ast_expr.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <pet.h>

#include "islutils/bm_matchers.h"

using util::ScopedCtx;
using namespace matchers;

int main(int argc, char **argv) {

  auto inputFile = argv[1];
  auto ctx = pet::allocCtx();
  auto petScop = pet::Scop::parseFile(ctx, inputFile);
  auto scop = petScop.getScop();

  // Call to the function for searching patterns
  auto pattern = blasmatchers::findPatterns(ctx, scop);

  for (auto p : pattern) {
    auto res = p.second;
    for (auto k : res) {

      // If code generation has been implemented for the match,
      // print it. This currently works only for Gemm and variants.
      if (!k->schedule_node.is_null()) {
	petScop.schedule() = k->schedule_node.get_schedule();    
	std::string output = blaskernels::printCudaHeader();
	std::string codeGen = petScop.codegenPayload(blaskernels::codeGenGPU, k);
	codeGen = k->insertCallToCUBLAS(codeGen);
	output += codeGen;
	std::cout << output << std::endl;
      }
      // In any case, at least print which pattern has been found. 
      // There may be several matches.
      std::cout << "Enum pattern number " << k->type << " is a match " << std::endl;
    }
  }
  return 0;
}



