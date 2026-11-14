#!/usr/bin/env bash
# set_profile.sh <profile>   switch the running adv sim's profile (nominal|real_mild|real_hard|out_of_range|<yaml name>)
# Takes effect at the next per-episode `randomize` (the evaluator calls it with the trial seed).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/adv_env.sh"
name="$1"; [[ "$name" == nominal ]] && f=adv_nominal || f="adv_$name"
cd "$ADV_TREE" && bash aaltonomous/cloud/aalto.sh py -c "
from aaltonomous.sim.scene_client import SceneClient
print(SceneClient('127.0.0.1').call(cmd='set_profile', name='$f'))"
