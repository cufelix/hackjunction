# Source on a sim box. Selects the adv sim instance (default "c": DDS domain 8, cams 57555-7, rgbd 57560,
# scene 57580, clock 57581, teleimager req 62000) so it never collides with sim a (default) or sim b.
# ADV_INSTANCE=b uses b's stock ports instead (only if you own b and stopped it yourself).
export ADV_TREE="${ADV_TREE:-/data/aaltonomous-humanoid-adv}"
export AALTO_INSTANCE="${ADV_INSTANCE:-c}"
if [[ "$AALTO_INSTANCE" == c ]]; then
  export AALTO_DDS_DOMAIN=8 AALTO_CAM_PORT_BASE=57555 AALTO_SCENE_PORT=57580 AALTO_RGBD_PORT=57560
  export AALTO_TELEIMAGER_REQ_PORT=62000
else
  export AALTO_DDS_DOMAIN=7 AALTO_CAM_PORT_BASE=56555 AALTO_SCENE_PORT=56580 AALTO_RGBD_PORT=56560
fi
export TASK="${TASK:-Aalto-Sort-G129-Dex3-Joint}"
