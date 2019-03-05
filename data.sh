# indir=/home/cywu/cywu/video_compression/trace_yuv/cif/all_png_frames/
# outdir=/home/cywu/cywu/video_compression/trace_yuv/cif/all_motion_vector_images/
# outdir=/home/cywu/cywu/video_compression/trace_yuv/cif/all_motion_vector_images2/
# tmp_prefix='trace'
# mode=8digits

# indir=/home/cywu/cywu/video_compression/kinetics_train_0_100frames_352x288
# outdir=/home/cywu/cywu/video_compression/kinetics_train_0_100frames_352x288_motion_vector_frames2
# tmp_prefix=kt0_$1_
# mode=4digits


# indir=/home/cywu/cywu/video_compression/ultra_video_group/all_png_frames
# outdir=/home/cywu/cywu/video_compression/ultra_video_group/all_png_frames_motion_vector_frames2
# tmp_prefix=ut2_$1_

# indir=/u/cywu/cywu/kinetics_train_3_30frames_352x288_$1
# outdir='/u/cywu/cywu/kinetics_train_3_30frames_352x288_'$1'_motion_vector_frames2'
# tmp_prefix=kt3_$1_
# mode=4digits

# indir=/home/cywu/cywu/video_compression/tears_of_steel_480_200
# outdir=/home/cywu/cywu/video_compression/tears_of_steel_480_200_motion_vector_frames2
# tmp_prefix=tear2_$1_
# mode=4digits


indir=/Users/mihir/Documents/projects/pytorch-vcii_cpg/data/temp_eval
outdir=/Users/mihir/Documents/projects/pytorch-vcii_cpg/data/temp_eval_gen
tmp_prefix=pexel_sub_$1_
mode=4digits

mkdir ${outdir}

# indir=/home/cywu/cywu/video_compression/ultra_video_group/all_png_frames
# outdir=/home/cywu/cywu/video_compression/ultra_video_group/all_png_frames_motion_vector_frames2
# tmp_prefix=ut_$1_
# mode=4digits

echo $1

for f in ${indir}/*.png
do
	video=${f##*/}
	if [ $mode = "8digits" ]; then
		video_name=${video::-12}
	fi

	if [ $mode = "4digits" ]; then
		video_name=${video::${#video}-8}
	fi

	name=${video::${#video}-4}
	idx=${name##*_}
	# echo "old idx"
	# echo $idx
	idx=$((10#$idx)) # force decimal (base 10)

	echo "file name- " $name 
	echo "video name- " $video_name
	# echo "idx"
	echo "idxs - "$idx
	group_idx=$(($idx%12))
	echo "group idxs - "$group_idx
	if [ $group_idx = "1" ]; then
		echo "Iframes- " $idx

		# 0 is I-frame1, 12 is I-frame2.
		last_idx=$(($idx+12))

	# 	if [ $mode = "8digits" ]; then
	# 		last_frame=${indir}/${video_name}$(printf "%08d" ${last_idx}).jpg
	# 	fi
		if [ $mode = "4digits" ]; then
			last_frame=${indir}/${video_name}$(printf "%04d" ${last_idx}).png
		fi
		

	# 	echo ${last_frame}
		if [ -f ${last_frame} ]; then
			echo "lastframe-  " $last_frame
			for i in `seq 1 11`; do
				echo "seq-> "$i
				cur_idx=$(($idx+$i))
				if [ $i = "1" ] || [ $i = "4" ] || [ $i = "7" ] || [ $i = "10" ]; then
					prev_idx=$(($cur_idx-1))
					next_idx=$(($cur_idx+2))
				fi
				if [ $i = "2" ] || [ $i = "5" ] || [ $i = "8" ] || [ $i = "11" ]; then
					prev_idx=$(($cur_idx-2))
					next_idx=$(($cur_idx+1))
				fi

				if [ $i = "3" ] || [ $i = "9" ]; then
					prev_idx=$(($cur_idx-3))
					next_idx=$(($cur_idx+3))
				fi

				if [ $i = "6" ]; then
					prev_idx=$(($cur_idx-6))
					next_idx=$(($cur_idx+6))
				fi

	# 			if [ $mode = "8digits" ]; then
	# 				cur_frame=${indir}/${video_name}$(printf "%08d" ${cur_idx}).jpg
	# 				prev_frame=${indir}/${video_name}$(printf "%08d" ${prev_idx}).jpg
	# 				next_frame=${indir}/${video_name}$(printf "%08d" ${next_idx}).jpg
	# 			fi
				if [ $mode = "4digits" ]; then
					cur_frame=${indir}/${video_name}$(printf "%04d" ${cur_idx}).png
					prev_frame=${indir}/${video_name}$(printf "%04d" ${prev_idx}).png
					next_frame=${indir}/${video_name}$(printf "%04d" ${next_idx}).png
				fi
				echo "curr_frame " $cur_frame
				echo "prev_frame " $prev_frame
				echo "next_frame " $next_frame
				
				cp $prev_frame tmp_${tmp_prefix}1.jpg
				cp $cur_frame tmp_${tmp_prefix}2.jpg
				rm tmp_${tmp_prefix}3.jpg

				cur_frame_name=${cur_frame##*/}
				cur_frame_name=${cur_frame_name::${#cur_frame_name}-4}
				echo "cur_frame_name" $cur_frame_name
				
				yes | ffmpeg  -i "tmp_${tmp_prefix}%01d.jpg" -c:v libx264 -g 2 -bf 0 -b_strategy 0  -sc_threshold 0 tmp_${tmp_prefix}.mp4
	# 			# /home/cywu/MV-release/MV_extract/ffmpeg-2.7.2/doc/examples/extract_mvs tmp_${tmp_prefix}.mp4 >tmp_${tmp_prefix}.mvs0
	# 			# /home/cywu/MV-release/MV_extract/MV-code-release/Release/mpegflow \
				/home/csadmin/ffmpeg/doc/examples/extract_mvs tmp_${tmp_prefix}.mp4 >tmp_${tmp_prefix}.mvs0
				/home/csadmin/MV-release/MV_extract/MV-code-release/Release/mpegflow \
							tmp_${tmp_prefix}.mvs0 ${outdir}/${cur_frame_name}_before_flow_x ${outdir}/${cur_frame_name}_before_flow_y


				cp $next_frame tmp_${tmp_prefix}1.jpg
				yes | ffmpeg  -i "tmp_${tmp_prefix}%01d.jpg" -c:v libx264 -g 2 -bf 0 -b_strategy 0  -sc_threshold 0 tmp_${tmp_prefix}.mp4
				# /home/cywu/MV-release/MV_extract/ffmpeg-2.7.2/doc/examples/extract_mvs tmp_${tmp_prefix}.mp4 >tmp_${tmp_prefix}.mvs0
				# /home/cywu/MV-release/MV_extract/MV-code-release/Release/mpegflow \
				/home/csadmin/ffmpeg/doc/examples/extract_mvs tmp_${tmp_prefix}.mp4 >tmp_${tmp_prefix}.mvs0
				/home/csadmin/MV-release/MV_extract/MV-code-release/Release/mpegflow \
							tmp_${tmp_prefix}.mvs0 ${outdir}/${cur_frame_name}_after_flow_x ${outdir}/${cur_frame_name}_after_flow_y


				# cp $prev_frame tmp_${tmp_prefix}1.jpg
				# cp $cur_frame tmp_${tmp_prefix}2.jpg
				# cp $next_frame tmp_${tmp_prefix}3.jpg

							# tmp.mvs0 flow_x flow_y
				# > ${outdir}/${cur_frame_name}.mvs0
				# break
				# if [ $i = "6" ]; then
					# break
				# fi				
			done
		fi
	# 	# break
	fi
done
# -b-pyramid strict 