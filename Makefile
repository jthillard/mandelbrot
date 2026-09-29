.PHONY: hevc-nvenc x265 x265-master x265-small av1 av1-master av1-small

FFMPEG=ffmpeg
IN_FMT=rgba

RESOLUTION=1920x1080
FPS=60
OUT=out.mp4

# 10 s GOP: fractal zooms have no scene cuts, so keyframes only cost bits.
KEYINT=$(shell awk 'BEGIN { print int($(FPS) * 10 + 0.5) }')

FFMPEG_IN=$(FFMPEG) -f rawvideo -pix_fmt $(IN_FMT) -s $(RESOLUTION) -framerate $(FPS) -i -
SCALE=scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int+bitexact
# setparams tags the frames themselves: encoders read colour info from the
# filter output, so the -color_* options alone left primaries/transfer unset.
TAG=setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv
COLOR_TAGS=-colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv

# Settings and CRFs come from tools/encode/ (see its README for the numbers).
# Tiers: `x265`/`av1` = delivery (VMAF ~95 mean, >= 90 worst frame, smallest
# file); `*-master` = upload master (>= 97 worst frame, size secondary);
# `*-small` = about half the delivery size, visibly softer on faint texture
# (CRF extrapolated from the sweeps, not measured).
# The binding test clip is a faint, low-contrast Julia texture: at a CRF where
# dense filaments already score 99, both encoders smear it, so it sets the CRF.

# preset medium: slow scores better at equal bitrate but runs ~2.4x slower,
#   far outside the speed budget.
# aq-mode=3 (vs 4): 4 loses at matched bitrate on gradients and faint texture.
# no-sao: SAO lowers VMAF at the same bitrate (smooths fine filaments).
# merange=92: 4K zoom motion is large; ~2% smaller at no speed cost.
# rect=1: ~7% smaller at equal or better VMAF (about 20% slower).
# rdoq-level=2 + psy-rdoq=1: keeps the faint texture; the binding clip then
#   needs CRF ~17 instead of ~15, a smaller file overall.
X265_PARAMS=keyint=$(KEYINT):min-keyint=$(FPS):aq-mode=3:no-sao=1:merange=92:rect=1:rdoq-level=2:psy-rdoq=1.0

# preset 6: fits the speed budget (p5 is ~20% slower).
# tune=0 (VQ; ffmpeg defaults to PSNR): much better worst frames on faint texture.
# enable-variance-boost=1: more bits to flat/low-contrast blocks; faint Julia
#   texture goes from p1 ~85 to ~93 at CRF 32.
# enable-qm=1 (default qm-min): 3-11% smaller at equal VMAF; qm-min=0 caps
#   quality on faint texture and hurts filaments.
# film-grain=0: synthetic grain would add noise the renders don't have.
# tf and overlays stay default: disabling tf costs +50% bits and adds banding.
# scd=1: untested on its own; SVT-AV1 4.2 says it won't insert keyframes at
#   scene changes anyway, so it's likely a no-op here.
SVT_PARAMS=film-grain=0:keyint=$(KEYINT):scd=1:tune=0:enable-variance-boost=1:enable-qm=1

hevc-nvenc:
	$(FFMPEG_IN) -vf "$(SCALE),format=p010le" \
		-c:v hevc_nvenc -preset p7 -tune hq -rc vbr -cq 14 -b:v 0 -maxrate 200M -bufsize 400M \
		-profile:v main10 -pix_fmt p010le \
		-spatial-aq 1 -aq-strength 6 -temporal-aq 1 \
		-rc-lookahead 32 -bf 4 -b_ref_mode middle -multipass fullres \
		$(COLOR_TAGS) \
		-tag:v hvc1 -movflags +faststart \
		$(OUT)

# Delivery: CRF 17 (binding clip: VMAF 94.8 mean, 92.8 worst frame).
x265:
	$(FFMPEG_IN) -vf "$(SCALE),format=yuv420p10le,$(TAG)" \
		-c:v libx265 -profile:v main10 -preset medium -crf 17 \
		-x265-params "$(X265_PARAMS)" \
		$(COLOR_TAGS) \
		-tag:v hvc1 -movflags +faststart \
		$(OUT)

# Master: CRF 12 (binding clip: 97.6 worst frame).
x265-master:
	$(FFMPEG_IN) -vf "$(SCALE),format=yuv420p10le,$(TAG)" \
		-c:v libx265 -profile:v main10 -preset medium -crf 12 \
		-x265-params "$(X265_PARAMS)" \
		$(COLOR_TAGS) \
		-tag:v hvc1 -movflags +faststart \
		$(OUT)

# Small: CRF 23, ~half the size of `x265` (bitrate fell ~40% from CRF 16 to 20).
# Untested: expect faint texture around VMAF 89 mean; dense detail stays ~97+.
x265-small:
	$(FFMPEG_IN) -vf "$(SCALE),format=yuv420p10le,$(TAG)" \
		-c:v libx265 -profile:v main10 -preset medium -crf 23 \
		-x265-params "$(X265_PARAMS)" \
		$(COLOR_TAGS) \
		-tag:v hvc1 -movflags +faststart \
		$(OUT)

# Delivery: CRF 32 (binding clip: VMAF 95.0 mean, 93.4 worst frame).
av1:
	$(FFMPEG_IN) -vf "$(SCALE),format=yuv420p10le,$(TAG)" \
		-c:v libsvtav1 -preset 6 -crf 32 \
		-svtav1-params "$(SVT_PARAMS)" \
		$(COLOR_TAGS) \
		-movflags +faststart \
		$(OUT)

# Master: CRF 17 (binding clip: ~97.1 worst frame, interpolated from CRF 14/20).
av1-master:
	$(FFMPEG_IN) -vf "$(SCALE),format=yuv420p10le,$(TAG)" \
		-c:v libsvtav1 -preset 6 -crf 17 \
		-svtav1-params "$(SVT_PARAMS)" \
		$(COLOR_TAGS) \
		-movflags +faststart \
		$(OUT)

# Small: CRF 40, ~half the size of `av1` (bitrate fell ~35% from CRF 26 to 32).
# Untested: expect faint texture around VMAF 92 mean; dense detail stays ~97+.
av1-small:
	$(FFMPEG_IN) -vf "$(SCALE),format=yuv420p10le,$(TAG)" \
		-c:v libsvtav1 -preset 6 -crf 40 \
		-svtav1-params "$(SVT_PARAMS)" \
		$(COLOR_TAGS) \
		-movflags +faststart \
		$(OUT)
