/*
 * Detector-only temporal decision logic for LiDAR observability.
 *
 * This helper owns only diagnostic state. It never reads or writes estimator
 * variables, maps, residuals, or marginalization data.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace cocolic
{

  struct DegeneracyDecision
  {
    double degeneracy_score = 0.0;
    int candidate_weak_direction_num = 0;
    bool degenerate_state = false;
    int enter_counter = 0;
    int exit_counter = 0;
  };

  class DegeneracyHysteresis
  {
  public:
    void Configure(double enter_relative_threshold,
                   double exit_relative_threshold,
                   int enter_consecutive_scans,
                   int exit_consecutive_scans)
    {
      enter_relative_threshold_ =
          std::max(0.0, enter_relative_threshold);
      exit_relative_threshold_ =
          std::max(enter_relative_threshold_, exit_relative_threshold);
      enter_consecutive_scans_ = std::max(1, enter_consecutive_scans);
      exit_consecutive_scans_ = std::max(1, exit_consecutive_scans);
      Reset();
    }

    void Reset()
    {
      degenerate_state_ = false;
      enter_counter_ = 0;
      exit_counter_ = 0;
    }

    DegeneracyDecision Update(
        bool valid,
        const std::array<double, 6> &relative_eigenvalues)
    {
      DegeneracyDecision decision;
      decision.degenerate_state = degenerate_state_;

      const double weakest_relative_eigenvalue = relative_eigenvalues[0];
      if (!valid || !std::isfinite(weakest_relative_eigenvalue) ||
          weakest_relative_eigenvalue < 0.0)
      {
        // Missing geometry must not create an artificial transition. Requiring
        // a fresh consecutive run after invalid data avoids stale counters.
        enter_counter_ = 0;
        exit_counter_ = 0;
        decision.enter_counter = enter_counter_;
        decision.exit_counter = exit_counter_;
        return decision;
      }

      decision.degeneracy_score =
          -std::log10(std::max(weakest_relative_eigenvalue, 1e-12));
      for (const double value : relative_eigenvalues)
      {
        if (std::isfinite(value) && value < enter_relative_threshold_)
        {
          ++decision.candidate_weak_direction_num;
        }
      }

      if (!degenerate_state_)
      {
        exit_counter_ = 0;
        if (weakest_relative_eigenvalue < enter_relative_threshold_)
        {
          enter_counter_ =
              std::min(enter_counter_ + 1, enter_consecutive_scans_);
        }
        else
        {
          enter_counter_ = 0;
        }

        if (enter_counter_ >= enter_consecutive_scans_)
        {
          degenerate_state_ = true;
        }
      }
      else
      {
        enter_counter_ = 0;
        if (weakest_relative_eigenvalue > exit_relative_threshold_)
        {
          exit_counter_ =
              std::min(exit_counter_ + 1, exit_consecutive_scans_);
        }
        else
        {
          exit_counter_ = 0;
        }

        if (exit_counter_ >= exit_consecutive_scans_)
        {
          degenerate_state_ = false;
        }
      }

      decision.degenerate_state = degenerate_state_;
      decision.enter_counter = enter_counter_;
      decision.exit_counter = exit_counter_;
      return decision;
    }

  private:
    double enter_relative_threshold_ = 3e-3;
    double exit_relative_threshold_ = 6e-3;
    int enter_consecutive_scans_ = 10;
    int exit_consecutive_scans_ = 10;

    bool degenerate_state_ = false;
    int enter_counter_ = 0;
    int exit_counter_ = 0;
  };

} // namespace cocolic
