import tensorflow as tf 
from network import Network 


class PPO:
    def __init__(self, act_head_list, obs):
        """
        act_head_list: 每个动作输出头的维度大小
        
        """
        self.old_model = Network(obs, act_head_list, name='old_policy', trainable=True)
        self.new_model = Network(obs, act_head_list, name='new_policy', trainable=True)
        self.dist_new_policy_dict, self.new_value, self.new_params = self.new_model.output()
        self.dist_old_policy_dict, self.old_value, self.old_params  = self.old_model.output()
        self.behavior_action = [dist.sample() for dist in self.dist_new_policy_dict.values()]
        self.num_heads = len(act_head_list)
        self.action_list = [tf.placeholder(dtype=tf.int32, shape=[
            None,], name='action_head_{}'.format(i)) for i in range(self.num_heads)]
        self.advantage = tf.placeholder(dtype=tf.float32, shape=[None,], name='advantage')
        self.dis_rew = tf.placeholder(dtype=tf.float32, shape=[None, ], name='discounted_reward')

        # self.init = tf.global_variables_initializer()
        self.update_op = [old_p.assign(p) for p, old_p in zip(self.old_params, self.new_params)]
        
    
    def build_loss(self):
        # 对所有动作输出头计算损失
        with tf.variable_scope('policy_loss'):
            new_log_prob = 0.
            old_log_prob = 0.
            for act, new_dist, old_dist in zip(self.action_list, self.dist_new_policy_dict.values(), self.dist_old_policy_dict.values()):
                new_log_prob += new_dist.log_prob(act)
                old_log_prob += old_dist.log_prob(act)

            ratio = tf.exp(new_log_prob - old_log_prob)
            self.rec_new_log_prob = new_log_prob
            surr = ratio * self.advantage
            self.policy_loss = tf.reduce_mean(tf.minimum(
                surr, 
                tf.clip_by_value(ratio, 1. - 0.2, 1. + 0.2) * self.advantage))
        
        with tf.variable_scope('value_loss'):
            self.value_loss = tf.reduce_mean(tf.square(self.new_value - self.dis_rew))
        
        with tf.variable_scope('entropy_loss'):
            self.entropy_loss = 0. 

        with tf.variable_scope('total_loss'):
            self.loss = tf.add(self.value_loss, self.policy_loss)

        with tf.variable_scope('train_op'):
            self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def predict(self, sess, feed_dict):
        action, value = sess.run([self.behavior_action, self.new_value], feed_dict=feed_dict)
        return action, value


    def learn(self, sess, feed_dict):
        self.build_loss()
        total_loss, _, value = sess.run([self.loss, self.train_op, self.new_value], feed_dict=feed_dict)
        sess.run(self.update_op)

