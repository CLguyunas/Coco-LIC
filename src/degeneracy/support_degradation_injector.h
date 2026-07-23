/*
 * Detector-only timestamp degradation for spline-support validation.
 *
 * The injector accepts and returns timestamp/weight copies. It has no access
 * to estimator residuals, parameter blocks, maps, or trajectory state.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cocolic
{

  using WeightedTimestamp = std::pair<int64_t, double>;

  enum class SupportInjectionMode
  {
    Disabled = 0,
    TimestampCompression = 1,
    PhaseDropout = 2,
    BoundaryDropout = 3,
    TemporalThinning = 4
  };

  struct SupportInjectionConfig
  {
    bool enabled = false;
    SupportInjectionMode mode = SupportInjectionMode::Disabled;
    double severity = 0.5;
    double phase_start = 0.0;
    double phase_end = 1.0;
    uint64_t random_seed = 42;
  };

  struct SupportInjectionMetadata
  {
    bool enabled = false;
    bool applied = false;
    SupportInjectionMode mode = SupportInjectionMode::Disabled;
    double severity = 0.0;
    double phase_start = 0.0;
    double phase_end = 1.0;
    uint64_t random_seed = 0;
    size_t input_sample_num = 0;
    // Exact earliest/latest samples preserved by timestamp compression.
    size_t boundary_anchor_sample_num = 0;
    size_t selected_sample_num = 0;
    size_t modified_sample_num = 0;
    size_t removed_sample_num = 0;
    size_t output_sample_num = 0;
    double retained_ratio = 0.0;
    double timestamp_span_ratio = 0.0;
  };

  class SupportDegradationInjector
  {
  public:
    void Configure(const SupportInjectionConfig &config);

    bool Enabled() const { return config_.enabled; }
    const SupportInjectionConfig &Config() const { return config_; }

    SupportInjectionMetadata Apply(
        const std::vector<WeightedTimestamp> &input,
        std::vector<WeightedTimestamp> &output) const;

    static bool ParseMode(const std::string &name,
                          SupportInjectionMode &mode);
    static const char *ModeName(SupportInjectionMode mode);

  private:
    static uint64_t Mix64(uint64_t value);
    static double UnitRandom(uint64_t key);

  private:
    SupportInjectionConfig config_;
  };

} // namespace cocolic
