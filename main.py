import tensorflow as tf
import argparse

from datetime import datetime
from train import train


def main(_):
    date = datetime.now().strftime('%Y%m%d%H%M%S')
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongDeterministic-v4')
    parser.add_argument('--load', type=str)
    parser.add_argument('--logdir', type=str, default=date)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--job', type=str)
    parser.add_argument('--index', type=int)
    parser.add_argument('--num-processes', type=int, default=8)
    args = parser.parse_args()

    cluster_spec = {
        "worker": ['127.0.0.1:{}'.format(2222 + i) for i in range(args.num_processes)],
        "ps": ["127.0.0.1:2221"]
    }
    cluster = tf.train.ClusterSpec(cluster_spec).as_cluster_def()

    if args.job == 'ps':
        config = tf.ConfigProto(device_filters=["/job:ps"])
        server = tf.train.Server(
            cluster, job_name=args.job, task_index=args.index, config=config)
        server.join()
    else:
        config = tf.ConfigProto(intra_op_parallelism_threads=1,
                                inter_op_parallelism_threads=2)
        server = tf.train.Server(
            cluster, job_name=args.job, task_index=args.index, config=config)
        train(server, cluster, args)

if __name__ == '__main__':
    tf.app.run(main=main)
