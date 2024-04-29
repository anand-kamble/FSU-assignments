ffmpeg -framerate 15 -pattern_type glob -i 'reflec_pres/movie-0/*.png' -c:a copy -shortest -c:v libx264 -pix_fmt yuv420p pres_out.mp4 &&\
ffmpeg -framerate 15 -pattern_type glob -i 'reflec_temp/movie-0/*.png' -c:a copy -shortest -c:v libx264 -pix_fmt yuv420p temp0_out.mp4 &&\
ffmpeg -framerate 15 -pattern_type glob -i 'reflec_temp/movie-1/*.png' -c:a copy -shortest -c:v libx264 -pix_fmt yuv420p temp1_out.mp4 &&\
ffmpeg -framerate 15 -pattern_type glob -i 'reflec_ener/movie-0/*.png' -c:a copy -shortest -c:v libx264 -pix_fmt yuv420p ener_out.mp4 &&\
ffmpeg -framerate 15 -pattern_type glob -i 'reflec_dens/movie-0/*.png' -c:a copy -shortest -c:v libx264 -pix_fmt yuv420p dens_out.mp4
