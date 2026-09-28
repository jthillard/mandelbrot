.PHONY: hevc-nvenc x265 av1

FFMPEG=ffmpeg
IN_FMT=rgba

RESOLUTION=1920x1080
FPS=60
OUT=out.mp4

FFMPEG_CMD=$(FFMPEG) -f rawvideo -pix_fmt $(IN_FMT) -s $(RESOLUTION) -framerate $(FPS) -i - -vf "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int+bitexact,format=p010le"

hevc-nvenc:
	$(FFMPEG_CMD) \
		-c:v hevc_nvenc -preset p7 -tune hq -rc vbr -cq 14 -b:v 0 -maxrate 200M -bufsize 400M \
		-profile:v main10 -pix_fmt p010le \
		-spatial-aq 1 -aq-strength 6 -temporal-aq 1 \
		-rc-lookahead 32 -bf 4 -b_ref_mode middle -multipass fullres \
		-colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv \
		-tag:v hvc1 -movflags +faststart \
		$(OUT)

x265:
	$(FFMPEG_CMD) \
		-c:v libx265 -crf 18 -preset medium -pix_fmt yuv420p10le \
		-x265-params "aq-mode=3:no-sao=1" \
		-colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv \
		-tag:v hvc1 -movflags +faststart \
		$(OUT)

av1:
	$(FFMPEG_CMD) \
		-c:v libsvtav1 -crf 24 -preset 6 -pix_fmt yuv420p10le \
		-svtav1-params "film-grain=0:enable-overlays=0" \
		-colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv \
		$(OUT)
