from argparse import ArgumentParser
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
import tensorflow as tf

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--image', help='path to image file')
    args = parser.parse_args()

    host = 'localhost'
    port = 9000
    channel = implementations.insecure_channel(host, port)
    stub = predict_pb2.beta_create_PredictionService_stub(channel)

    with open(args.image, 'rb') as f:
        data = f.read()
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'mnist'
        request.model_spec.signature_name = 'predict'
        request.inputs['image'].CopyFrom(
            tf.make_tensor_proto(data, shape=[28*28], dtype=tf.float32, verify_shape=True))
        result = stub.Predict(request, 10.0)  # 10 secs timeout
        print(result)