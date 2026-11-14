"""AI-brain layer over the classical policy: closed-set tool choice (index + confidence), validator, EV rule, stub
backend, async request with timeout. stdlib only. Off by default (config brain.enabled)."""
from .core import (BrainLayer, PlannerBackend, StubPlanner, AsyncRequest, LlamaCppHTTPBackend, make_backend, state_key, ev_decide, validate_decision,
                   world_to_json, build_world)
from .registry import (BINS, TOOLS, MACROS, POINTS, VARIANTS, allowed_calls, decision_json_schema, decision_gbnf,
                       expand_macro)
from .planner import (PLAN_TOOLS, PlanLayer, explicit_plan_satisfied, instruction_needs_plan, parse_plan,
                      plan_gbnf, plan_json_schema, plan_prompt, validate_plan)
