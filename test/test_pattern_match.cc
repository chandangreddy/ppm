
#include <islutils/builders.h>
#include <islutils/ctx.h>
#include <islutils/matchers.h>
#include <islutils/pet_wrapper.h>

#include <pet.h>

#include "gtest/gtest.h"

using util::ScopedCtx;

using namespace matchers;
enum class PatternType { Gemm, MatrixVector, DotProduct, None };

std::ostream &operator<<(std::ostream &os, PatternType p) {
  switch (p) {
  case PatternType::Gemm:
    return os << "GEMM";
  case PatternType::MatrixVector:
    return os << "Matrix Vector";
  case PatternType::DotProduct:
    return os << "Dot Product";
  default:
    return os << "None";
  };
}

class Pattern {
public:
  Pattern() : type_(PatternType::None) {}
  Pattern(PatternType t) : type_(t) {}
  virtual bool isMatching(isl::schedule_node node) const { return false; }
  virtual int Cost() const { return 0; }
  PatternType GetType() const { return type_; }

private:
  PatternType type_;
};


class GemmPattern : public Pattern {
public:
  GemmPattern() : Pattern(PatternType::Gemm) {}

  virtual bool isMatching(isl::schedule_node node) const override {

    return ScheduleNodeMatcher::isMatching(matcher_, node);
  }

  virtual int Cost() const override { return MatchingCost; }

private:
  static constexpr int MatchingCost = 3;
  // clang-format off
  static inline ScheduleNodeMatcher matcher_ = 
    band(
      band(
        sequence(
          filter(
            leaf()),
          filter(
            band(
              leaf())))));
  // clang-format on
};

class MatrixVectorPattern : public Pattern {
public:
  MatrixVectorPattern() : Pattern(PatternType::MatrixVector) {}

  virtual bool isMatching(isl::schedule_node node) const override {

    return ScheduleNodeMatcher::isMatching(matcher_, node);
  }

  virtual int Cost() const override { return MatchingCost; }

private:
  static constexpr int MatchingCost = 20;
  // clang-format off
  static inline ScheduleNodeMatcher matcher_ = 
      band(
        sequence(
          filter(
            leaf()),
          filter(
            band(
              leaf()))));
  // clang-format on
};

class DotProductPattern : public Pattern {
public:
  DotProductPattern() : Pattern(PatternType::DotProduct) {}

  virtual bool isMatching(isl::schedule_node node) const override {

    return ScheduleNodeMatcher::isMatching(matcher_, node);
  }

  virtual int Cost() const override { return MatchingCost; }

private:
  static constexpr int MatchingCost = 1;
  // clang-format off
  static inline ScheduleNodeMatcher matcher_ = 
        sequence(
          filter(
            leaf()),
          filter(
            band(
              leaf())));
  // clang-format on
};

typedef std::unique_ptr<Pattern> UPattern;
std::vector<UPattern> AllPatterns() {

  std::vector<UPattern> patterns;
  patterns.emplace_back(std::make_unique<GemmPattern>());
  patterns.emplace_back(std::make_unique<MatrixVectorPattern>());
  patterns.emplace_back(std::make_unique<DotProductPattern>());

  return std::move(patterns);
}

int MatchPatterns(isl::schedule_node node,
                  std::vector<PatternType> &matchedPatterns) {

  int max_cost = std::numeric_limits<int>::min();
  matchedPatterns.push_back(PatternType::None);

  // match the current node, top level pattern matches
  // are prioritized
  for (auto const &p : AllPatterns()) {
    if (p->isMatching(node)) {
      int cur_cost = p->Cost();
      if (max_cost < cur_cost) {
        max_cost = cur_cost;
        matchedPatterns.pop_back();
        matchedPatterns.push_back(p->GetType());
      }
    }
  }

  // try matching all subtrees 
  for (int i = 0; i < node.n_children(); ++i) {
    std::vector<PatternType> subtreePatterns;
    int cur_cost = MatchPatterns(node.child(i), subtreePatterns);
    if (max_cost < cur_cost) {
      max_cost = cur_cost;

      // Remove previous match
      matchedPatterns.pop_back();
      matchedPatterns.push_back(subtreePatterns[0]);
    }
  }

  return max_cost;
}

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