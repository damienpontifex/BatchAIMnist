docker run --rm -it \
    --name mnist \
    -p 9000:9000 \
    -v /Users/ponti/data/mnist/serving:/models \
    epigramai/model-server:light \
    --port=9000 \
    --model_name=mnist \
    --model_base_path=/models