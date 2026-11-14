# source: slot.sh <n> [gpu] [cores]   own sim instance "nv<n>": unique ports / DDS domain; never collides with matrix slots (x0-x23)
n=$1
export NV=$n NOVEL_ROOT=/data/work/novel NOVEL_TREE=/data/work/novel/tree NOVEL_ASSETS=/data/work/novel/assets/Axis_Aligned
export AALTO_INSTANCE=nv$n AALTO_DDS_DOMAIN=$((70+n)) AALTO_CAM_PORT_BASE=$((59000+10*n)) AALTO_RGBD_PORT=$((59100+n)) \
       AALTO_SCENE_PORT=$((59200+2*n)) AALTO_TELEIMAGER_REQ_PORT=$((63000+n))
export CONDA_ROOT=/root/miniforge3 OMNI_KIT_ACCEPT_EULA=YES
export POLPORT=$((8300+n)) GPU=${2:-$((n%8))}
export CORES=${3:-$((30*GPU+10*(n%3)))-$((30*GPU+10*(n%3)+9))}
