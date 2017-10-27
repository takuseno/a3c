import tensorflow as tf


class RewardSummary:
    def __init__(self):
        self.reward_ph = tf.placeholder(tf.float32, (), name='reward_summary_ph')
        self.summary = tf.summary.scalar('reward_summary', self.reward_ph)

    def set_writer(writer):
        self.writer = writer

    def add_summary(self, sess, writer, reward, step):
        summary, _ = sess.run([self.summary, self.reward_ph], feed_dict={self.reward_ph: reward})
        writer.add_summary(summary, step)
