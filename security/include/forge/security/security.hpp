#pragma once
// forge::security: the FORGE Security Standard (docs/security-standard.md, API in docs/security-api.md).
//   errors.hpp    security_error > policy_violation, integrity_error, permission_denied, sandbox_error (FSS-ERR)
//   audit.hpp     audit(): OS facility, JSON-lines file, callbacks, logger (FSS-AUD)
//   path.hpp      safe_join, canonical_within, check_relative (FSS-CAP-2/3)
//   policy.hpp    policy, profiles, parse_policy, load_policy
//   enforce.hpp   enforce / try_enforce, the active policy (FSS-CAP-1)
//   lockdown.hpp  lockdown, check (FSS-SBX, Verification, FSS-BLD-2)
#include <forge/security/errors.hpp>
#include <forge/security/audit.hpp>
#include <forge/security/path.hpp>
#include <forge/security/policy.hpp>
#include <forge/security/enforce.hpp>
#include <forge/security/lockdown.hpp>
