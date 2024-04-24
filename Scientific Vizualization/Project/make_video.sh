ffmpeg -framerate 15 -pattern_type glob -i 'reflec_pres/movie-0/*.png' -c:a copy -shortest -c:v libx264 -pix_fmt yuv420p out.mp4
