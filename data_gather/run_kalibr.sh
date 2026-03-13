FOLDER=$1
docker run -it \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=/tmp/.Xauthority \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$HOME/.Xauthority:/tmp/.Xauthority:ro" \
    -v "$FOLDER:/data" \
    kalibr

# docker run -it \
#     -e MPLBACKEND=Agg \
#     -v "$FOLDER:/data" \
#     kalibr
