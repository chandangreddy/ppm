#include "gtest/gtest.h"
#include <islutils/pet_wrapper.h>
#include <islutils/ctx.h>

#include "pattern_match.h"
using namespace ppm;

isl::schedule_node getGemmTree() {
  std::string inputFile =
      "/home/creddy/work/Adilla/sourcetosource/test/inputs/gemm.c";

  auto ctx = isl::ctx(isl_ctx_alloc());
  auto petScop = pet::Scop::parseFile(ctx, inputFile);
  auto scop = petScop.getScop();

  return scop.schedule.get_root();
}

TEST(PatternMatcher, GemmMatches) {
  auto node = getGemmTree();

  GemmPattern gp;
  auto gemm_node = node.child(0);
  EXPECT_TRUE(gp.isMatching(gemm_node));

  auto matrix_vec = gemm_node.child(0);
  MatrixVectorPattern mvp;
  EXPECT_TRUE(mvp.isMatching(matrix_vec));

  auto dot_p = matrix_vec.child(0);
  DotProductPattern dpp;
  EXPECT_TRUE(dpp.isMatching(dot_p));
}

TEST(PatternMatcher, DPTest) {
  using namespace std;
  {

    auto root = getGemmTree();

    auto node = root.child(0);

    std::vector<PatternType> matchedPattern;
    int cost = MatchPatterns(node, matchedPattern);

    cout << cost << " " << matchedPattern[0] << endl;

  } // namespace nam std
}