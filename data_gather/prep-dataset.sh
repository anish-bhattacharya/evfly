# print usage
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset>"
    exit 1
fi

dataset=$1
echo "Preparing data for ${dataset} dataset"
python data_gather/postprocess_alignment_real_data.py /home/anish1/evfly_ws/data/datasets/${dataset} /home/anish1/evfly_ws/data/datasets/${dataset}_aligned && \
python data_gather/convert_realdata_to_datasetformat.py /home/anish1/evfly_ws/data/datasets/${dataset}_aligned 480 640 && \
python utils/to_h5.py ${dataset}_aligned convert False && \
mv /home/anish1/evfly_ws/data/datasets/${dataset}_aligned.h5 /home/anish1/evfly_ws/data/datasets/${dataset}.h5 && \
echo "Data preparation done for ${dataset} dataset"
