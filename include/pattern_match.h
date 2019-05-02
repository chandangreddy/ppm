#include <islutils/matchers.h>

namespace ppm {

enum class PatternType { Gemm, MatrixVector, DotProduct, None };

std::ostream &operator<<(std::ostream &os, PatternType p);

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

    return matchers::ScheduleNodeMatcher::isMatching(matcher_, node);
  }

  virtual int Cost() const override { return MatchingCost; }

private:
  static constexpr int MatchingCost = 3;
  static matchers::ScheduleNodeMatcher matcher_;
};

class MatrixVectorPattern : public Pattern {
public:
  MatrixVectorPattern() : Pattern(PatternType::MatrixVector) {}

  virtual bool isMatching(isl::schedule_node node) const override {

    return matchers::ScheduleNodeMatcher::isMatching(matcher_, node);
  }

  virtual int Cost() const override { return MatchingCost; }

private:
  static constexpr int MatchingCost = 20;
  static matchers::ScheduleNodeMatcher matcher_;
};

class DotProductPattern : public Pattern {
public:
  DotProductPattern() : Pattern(PatternType::DotProduct) {}

  virtual bool isMatching(isl::schedule_node node) const override {

    return matchers::ScheduleNodeMatcher::isMatching(matcher_, node);
  }

  virtual int Cost() const override { return MatchingCost; }

private:
  static constexpr int MatchingCost = 1;
  static matchers::ScheduleNodeMatcher matcher_;
};

typedef std::unique_ptr<Pattern> UPattern;
std::vector<UPattern> AllPatterns();

int MatchPatterns(isl::schedule_node node,
                  std::vector<PatternType> &matchedPatterns);

} // namespace ppm