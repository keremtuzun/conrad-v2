#!/usr/bin/env bash
# Fresh g2-standard-8 (1x L4) VM for round 4: data disk, code, public datasets, then waits for round 3's best.pt
# (uploaded from the workstation) and starts orchestrate_persource.sh.
# Usage on the VM: VM_START_EPOCH=<epoch> bash vm_setup_r4.sh
set -u
VM_START_EPOCH=${VM_START_EPOCH:?}
L=~/p4logs; mkdir -p $L
log(){ echo "$(date -u +%FT%TZ) $*" >> $L/setup_r4.log; }
if ! mountpoint -q /data; then
  DEV=/dev/disk/by-id/google-data
  sudo blkid $DEV >/dev/null 2>&1 || sudo mkfs.ext4 -q -F $DEV
  sudo mkdir -p /data && sudo mount $DEV /data && sudo chown $(id -u):$(id -g) /data
fi
log "data disk $(df -h /data | tail -1)"
(setsid nohup nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 60 >> $L/gpu.csv 2>/dev/null < /dev/null &)
sudo apt-get -qq update >/dev/null 2>&1
sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y libgl1 libglib2.0-0 aria2 libarchive-tools unzip >/dev/null 2>&1; log "apt done"
cd ~ && [ -d conrad-v2 ] || git clone -q https://github.com/keremtuzun/conrad-v2.git
cd ~/conrad-v2 && git fetch -q origin && git checkout -q -B kerem/v11-p4-patch-vicreg-trainer origin/kerem/v11-p4-patch-vicreg-trainer
log "code $(git rev-parse --short HEAD)"
mkdir -p /data/artifacts_data /data/tmp; mkdir -p artifacts
[ -L artifacts/data ] || ln -s /data/artifacts_data artifacts/data
[ -L artifacts/tmp ] || ln -s /data/tmp artifacts/tmp
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH=$HOME/.local/bin:$PATH
A="aria2c -x 16 -s 16 -k 32M --file-allocation=none --console-log-level=error --summary-interval=0 --auto-file-renaming=false --allow-overwrite=true"
D=artifacts/data; CK=artifacts/runs/train-osfm_u1_sonar_research-1790575109765916596/checkpoints
mkdir -p $CK $D/public.subpipe/raw $D/public.subpipe_full/raw $D/public.seaclear/raw $D/public.uvvid/raw $D/public.sss_catalunya/raw $D/public.china_offshore_sss/raw $D/public.aquascan_1k/raw $D/public.uatd/raw
Z=https://zenodo.org/api/records
( $A -d $CK -o osfm_u1_sonar_research_full.pt https://github.com/keremtuzun/conrad-v2/releases/download/osfm-v11-p4-handoff/osfm_u1_sonar_research_full.pt; log "ckpt $?" ) &
( $A -d $D/public.subpipe/raw -o SubPipeMini2.zip $Z/12666132/files/SubPipeMini2.zip/content; log "mini2 $?" ) &
( $A -d $D/public.subpipe_full/raw -o SubPipe.zip $Z/12666132/files/SubPipe.zip/content; log "subpipe_full $?" ) &
( for f in sss_ssl_dataset_N713_384.z01 sss_ssl_dataset_N713_384.z02 sss_ssl_dataset_N713_384.zip; do $A -d $D/public.sss_catalunya/raw -o $f $Z/10209445/files/$f/content; log "catalunya $f $?"; done ) &
( $A -d $D/public.china_offshore_sss/raw -o China-Offshore-SSS-AI_Zenodo_public_upload.zip $Z/20048164/files/China-Offshore-SSS-AI_Zenodo_public_upload.zip/content; log "china $?" ) &
( $A -d $D/public.aquascan_1k/raw -o AquaScan-1K.zip $Z/18771165/files/AquaScan-1K.zip/content; log "aquascan $?" ) &
( for i in UATD_Training.zip:37883163 UATD_Test_1.zip:37857276 UATD_Test_2.zip:37857279 UATD_OpenSLT.zip:37901247; do n=${i%%:*}; id=${i#*:}; curl -sSL --retry 5 -o $D/public.uatd/raw/$n https://ndownloader.figshare.com/files/$id; log "uatd $n $?"; done ) &
( for i in 1:50436333 2:50436330 3:50436360 4:50436318 5:50436345 6:50436327 7:50436363 8:50436369; do n=${i%%:*}; id=${i#*:}; curl -sSL --retry 5 -o $D/public.uvvid/raw/ROV_GoPro_$n.mp4 https://ndownloader.figshare.com/files/$id; done; log "uvvid done" ) &
( curl -sSL --retry 3 --max-time 3600 -o "$D/public.seaclear/raw/Seaclear Marine Debris Dataset.rar" https://data.4tu.nl/file/4f1dff25-e157-4399-a5d4-478055461689/e1240a2e-915e-4858-93ab-c004b26b5a5f; log "seaclear $?" ) &
uv sync >/dev/null 2>&1; log "uv sync $?"
wait; log "downloads finished"
( cd $D/public.seaclear && mkdir -p extracted && cd extracted && bsdtar -xf "../raw/Seaclear Marine Debris Dataset.rar"; log "seaclear extract $? $(find . -name '*.jpg' -size +0 | wc -l)" ) &
( cd $D/public.china_offshore_sss && mkdir -p extracted && unzip -qo raw/*.zip -d extracted; log "china extract $?" ) &
( cd $D/public.aquascan_1k && mkdir -p extracted && unzip -qo raw/*.zip -d extracted; log "aquascan extract $?" ) &
( cd $D/public.uatd && mkdir -p extracted && for z in raw/*.zip; do unzip -qo $z -d extracted; done; log "uatd extract $?" ) &
wait
uv run python - >> $L/setup_r4.log 2>&1 <<'PY'
import hashlib, zipfile, yaml
from pathlib import Path
root = Path("artifacts/data/public.subpipe")
m = yaml.safe_load(open("datasets/public/subpipe.manifest.yaml"))
z = zipfile.ZipFile(root / "raw/SubPipeMini2.zip"); names = z.namelist()
for f in m["files"]:
    if not f["path"].startswith("extracted/"):
        continue
    rel = f["path"][len("extracted/"):]
    data = z.read([n for n in names if n == rel or n.endswith("/" + rel)][0])
    out = root / f["path"]; out.parent.mkdir(parents=True, exist_ok=True); out.write_bytes(data)
    print("OK" if hashlib.sha256(data).hexdigest() == f["sha256"] else "HASH_MISMATCH", f["path"])
from conrad.foundation.universal_v11 import trainer as T
print("payloads", T.verify_required_payloads()["decision"])
print("subpipe_full", T.verify_subpipe_full()["decision"])
r = T.verify_extra_sources(); print("extra", r["decision"], {k: v["decision"] for k, v in r.items() if k != "decision"})
import torch; print(torch.cuda.get_device_name(0), torch.cuda.device_count())
PY
log "verify done; waiting for round-3 best.pt"
R3=artifacts/runs/v11-p4-kerem-sonar5-a1-20261004T165839Z/checkpoints/best.pt
until [ -f $R3.ok ]; do sleep 30; done
[ "$(sha256sum $R3 | cut -c1-64)" = "edf74d95782e67e897cd6e60b26b4ac3f55a93ad0b609db6f0f7488117f16638" ] || { log "ABORT round-3 best.pt hash mismatch"; exit 5; }
log "round-3 best.pt verified; starting orchestrator"
(VM_START_EPOCH=$VM_START_EPOCH R3_BEST=$R3 setsid nohup bash scripts/kerem_p4/orchestrate_persource.sh > $L/orchestrate.out 2>&1 < /dev/null &)
log SETUP_R4_DONE
