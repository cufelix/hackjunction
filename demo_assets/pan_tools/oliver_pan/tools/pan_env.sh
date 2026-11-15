# Source on a sim box. Pan sim = its own tree (copy of the stock tree + pan_sim.patch), DEFAULT instance ports
# (one sim per box; stop other sims you own first). AALTO_PAN=1 adds obj_grey_pan_0 + the `bodies` scene command.
export PAN_TREE="${PAN_TREE:-/data/aaltonomous-humanoid-pan}"
export AALTO_PAN=1
export TASK="${TASK:-Aalto-Sort-G129-Dex3-Joint}"
unset AALTO_INSTANCE AALTO_DR
