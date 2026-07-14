/*
 * Detector-only timestamp degradation for spline-support validation.
 */

#include <degeneracy/support_degradation_injector.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cocolic
{

  void SupportDegradationInjector::Configure(
      const SupportInjectionConfig &config)
  {
    config_ = config;
    if (!std::isfinite(config_.severity))
    {
      config_.severity = 0.0;
    }
    if (!std::isfinite(config_.phase_start))
    {
      config_.phase_start = 0.0;
    }
    if (!std::isfinite(config_.phase_end))
    {
      config_.phase_end = 1.0;
    }
    config_.severity = std::max(0.0, std::min(1.0, config_.severity));
    config_.phase_start =
        std::max(0.0, std::min(1.0, config_.phase_start));
    config_.phase_end =
        std::max(config_.phase_start,
                 std::min(1.0, config_.phase_end));
    if (config_.mode == SupportInjectionMode::Disabled)
    {
      config_.enabled = false;
    }
  }

  SupportInjectionMetadata SupportDegradationInjector::Apply(
      const std::vector<WeightedTimestamp> &input,
      std::vector<WeightedTimestamp> &output) const
  {
    SupportInjectionMetadata metadata;
    metadata.enabled = config_.enabled;
    metadata.mode = config_.mode;
    metadata.severity = config_.severity;
    metadata.phase_start = config_.phase_start;
    metadata.phase_end = config_.phase_end;
    metadata.random_seed = config_.random_seed;
    metadata.input_sample_num = input.size();

    output.clear();
    output.reserve(input.size());
    if (!config_.enabled || input.empty())
    {
      output = input;
      metadata.output_sample_num = output.size();
      metadata.retained_ratio = input.empty() ? 0.0 : 1.0;
      metadata.timestamp_span_ratio = input.empty() ? 0.0 : 1.0;
      return metadata;
    }

    int64_t min_timestamp_ns = std::numeric_limits<int64_t>::max();
    int64_t max_timestamp_ns = std::numeric_limits<int64_t>::min();
    for (const auto &sample : input)
    {
      min_timestamp_ns = std::min(min_timestamp_ns, sample.first);
      max_timestamp_ns = std::max(max_timestamp_ns, sample.first);
    }
    const int64_t timestamp_span_ns = max_timestamp_ns - min_timestamp_ns;
    if (timestamp_span_ns <= 0)
    {
      output = input;
      metadata.output_sample_num = output.size();
      metadata.retained_ratio = 1.0;
      metadata.timestamp_span_ratio = 1.0;
      return metadata;
    }
    if (config_.severity <= 0.0)
    {
      // This exact copy is important for the severity-zero invariance run;
      // avoid a floating-point timestamp round trip even in compression mode.
      output = input;
      metadata.output_sample_num = output.size();
      metadata.retained_ratio = 1.0;
      metadata.timestamp_span_ratio = 1.0;
      return metadata;
    }

    const double selected_center =
        0.5 * (config_.phase_start + config_.phase_end);
    for (size_t index = 0; index < input.size(); ++index)
    {
      const auto &sample = input[index];
      const double phase =
          static_cast<double>(sample.first - min_timestamp_ns) /
          static_cast<double>(timestamp_span_ns);
      const bool inside_selected_phase =
          phase >= config_.phase_start && phase <= config_.phase_end;
      const uint64_t random_key =
          static_cast<uint64_t>(sample.first) ^
          (config_.random_seed + 0x9e3779b97f4a7c15ULL) ^
          (static_cast<uint64_t>(index) * 0xbf58476d1ce4e5b9ULL);
      const bool selected_by_severity =
          UnitRandom(random_key) < config_.severity;

      bool remove_sample = false;
      WeightedTimestamp injected_sample = sample;
      switch (config_.mode)
      {
      case SupportInjectionMode::TimestampCompression:
        if (inside_selected_phase)
        {
          ++metadata.selected_sample_num;
          const double injected_phase =
              selected_center +
              (1.0 - config_.severity) * (phase - selected_center);
          injected_sample.first = min_timestamp_ns +
              static_cast<int64_t>(std::llround(
                  injected_phase * static_cast<double>(timestamp_span_ns)));
          if (injected_sample.first != sample.first)
          {
            ++metadata.modified_sample_num;
          }
        }
        break;
      case SupportInjectionMode::PhaseDropout:
        if (inside_selected_phase)
        {
          ++metadata.selected_sample_num;
          remove_sample = selected_by_severity;
        }
        break;
      case SupportInjectionMode::BoundaryDropout:
        if (!inside_selected_phase)
        {
          ++metadata.selected_sample_num;
          remove_sample = selected_by_severity;
        }
        break;
      case SupportInjectionMode::TemporalThinning:
        ++metadata.selected_sample_num;
        remove_sample = selected_by_severity;
        break;
      case SupportInjectionMode::Disabled:
      default:
        break;
      }

      if (remove_sample)
      {
        ++metadata.removed_sample_num;
      }
      else
      {
        output.push_back(injected_sample);
      }
    }

    metadata.output_sample_num = output.size();
    metadata.retained_ratio =
        static_cast<double>(output.size()) / static_cast<double>(input.size());
    metadata.applied = metadata.modified_sample_num > 0 ||
                       metadata.removed_sample_num > 0;

    if (output.empty())
    {
      metadata.timestamp_span_ratio = 0.0;
    }
    else
    {
      int64_t output_min_ns = std::numeric_limits<int64_t>::max();
      int64_t output_max_ns = std::numeric_limits<int64_t>::min();
      for (const auto &sample : output)
      {
        output_min_ns = std::min(output_min_ns, sample.first);
        output_max_ns = std::max(output_max_ns, sample.first);
      }
      metadata.timestamp_span_ratio =
          static_cast<double>(output_max_ns - output_min_ns) /
          static_cast<double>(timestamp_span_ns);
    }
    return metadata;
  }

  bool SupportDegradationInjector::ParseMode(
      const std::string &name, SupportInjectionMode &mode)
  {
    if (name == "timestamp_compression")
    {
      mode = SupportInjectionMode::TimestampCompression;
    }
    else if (name == "phase_dropout")
    {
      mode = SupportInjectionMode::PhaseDropout;
    }
    else if (name == "boundary_dropout")
    {
      mode = SupportInjectionMode::BoundaryDropout;
    }
    else if (name == "temporal_thinning")
    {
      mode = SupportInjectionMode::TemporalThinning;
    }
    else if (name == "disabled" || name == "none")
    {
      mode = SupportInjectionMode::Disabled;
    }
    else
    {
      mode = SupportInjectionMode::Disabled;
      return false;
    }
    return true;
  }

  const char *SupportDegradationInjector::ModeName(
      SupportInjectionMode mode)
  {
    switch (mode)
    {
    case SupportInjectionMode::TimestampCompression:
      return "timestamp_compression";
    case SupportInjectionMode::PhaseDropout:
      return "phase_dropout";
    case SupportInjectionMode::BoundaryDropout:
      return "boundary_dropout";
    case SupportInjectionMode::TemporalThinning:
      return "temporal_thinning";
    case SupportInjectionMode::Disabled:
    default:
      return "disabled";
    }
  }

  uint64_t SupportDegradationInjector::Mix64(uint64_t value)
  {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
  }

  double SupportDegradationInjector::UnitRandom(uint64_t key)
  {
    const uint64_t mantissa = Mix64(key) >> 11U;
    return static_cast<double>(mantissa) *
           (1.0 / 9007199254740992.0);
  }

} // namespace cocolic
