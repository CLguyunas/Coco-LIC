/*
 * Cause-adaptive subspace recovery (CASR) routing policy.
 *
 * This header is intentionally independent of Eigen and ROS so that the
 * safety-critical routing table can be tested in isolation.
 */

#pragma once

namespace cocolic
{

  enum class CasrRoute
  {
    Invalid = -1,
    Inactive = 0,
    EnvironmentCandidate = 1,
    SupportCandidate = 2,
    CoupledCommonCandidate = 3,
    CoupledConflict = 4
  };

  inline CasrRoute SelectCasrRoute(bool inputs_valid, int cause,
                                   int environment_rank, int support_rank,
                                   int common_rank)
  {
    if (!inputs_valid || cause < 0 || cause > 3)
    {
      return CasrRoute::Invalid;
    }

    switch (cause)
    {
    case 0:
      return CasrRoute::Inactive;
    case 1:
      return environment_rank > 0
                 ? CasrRoute::EnvironmentCandidate
                 : CasrRoute::Invalid;
    case 2:
      return support_rank > 0 ? CasrRoute::SupportCandidate
                              : CasrRoute::Invalid;
    case 3:
      if (environment_rank <= 0 || support_rank <= 0)
      {
        return CasrRoute::Invalid;
      }
      return common_rank > 0 ? CasrRoute::CoupledCommonCandidate
                             : CasrRoute::CoupledConflict;
    default:
      return CasrRoute::Invalid;
    }
  }

  inline const char *CasrRouteName(CasrRoute route)
  {
    switch (route)
    {
    case CasrRoute::Inactive:
      return "inactive";
    case CasrRoute::EnvironmentCandidate:
      return "environment_candidate";
    case CasrRoute::SupportCandidate:
      return "support_candidate";
    case CasrRoute::CoupledCommonCandidate:
      return "coupled_common_candidate";
    case CasrRoute::CoupledConflict:
      return "coupled_conflict";
    case CasrRoute::Invalid:
    default:
      return "invalid";
    }
  }

} // namespace cocolic
