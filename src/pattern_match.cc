#include "pattern_match.h"
#include <iostream>
#include <limits>
#include <string>

using namespace matchers;
namespace ppm {

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

// clang-format off
  ScheduleNodeMatcher GemmPattern::matcher_ = 
    band(
      band(
        sequence(
          filter(
            leaf()),
          filter(
            band(
              leaf())))));

  ScheduleNodeMatcher MatrixVectorPattern::matcher_ = 
      band(
        sequence(
          filter(
            leaf()),
          filter(
            band(
              leaf()))));

  ScheduleNodeMatcher DotProductPattern::matcher_ = 
        sequence(
          filter(
            leaf()),
          filter(
            band(
              leaf())));
// clang-format on

std::vector<UPattern> AllPatterns() {

  std::vector<UPattern> patterns;
  patterns.emplace_back(std::make_unique<GemmPattern>());
  patterns.emplace_back(std::make_unique<MatrixVectorPattern>());
  patterns.emplace_back(std::make_unique<DotProductPattern>());

  return patterns;
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

} // namespace ppm