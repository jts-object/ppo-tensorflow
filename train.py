import numpy as np 
import tensorflow as tf  
import gym  
from tensorflow.distributions import Categorical, Normal
from policy import PPO

gamma = 0.99 

class Train(object):
    def __init__(self, obs_dim, act_dim):
        self.obs = tf.placeholder(dtype=tf.float32, shape=[None, obs_dim], name='obs')
        self.policy = PPO([act_dim], self.obs)
        self.policy.build_loss()
        self.reward_ph = tf.placeholder(dtype=tf.int32, shape=[], name='reward')
        self.sess = tf.Session()
        self.build_summary()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=3)

    def build_summary(self):
        # 此处添加 tensorboard 所需要记录的标量
        tf.summary.scalar('loss', self.policy.loss)
        tf.summary.scalar('reward', self.reward_ph)
        self.merged = tf.summary.merge_all()
        # 指定一个文件来保存图，第一个参数给定的是地址，通过调用 tensorboard --logdir=logs 查看 tensorboard
        self.train_writer = tf.summary.FileWriter('train_log', self.sess.graph) 

    def output_summary(self):
        rew_summ = self.sess.run(self.merged, feed_dict=self.feed_dict)
        return rew_summ

    def train(self, obs_buffer, act_buffer, adv_buffer, ret_buffer, sum_reward):
        self.feed_dict = {
            self.obs: obs_buffer,
            self.policy.action_list[0]: act_buffer,
            self.policy.advantage: adv_buffer,
            self.policy.dis_rew: ret_buffer,
            self.reward_ph: sum_reward, 
        }
        _, log_prob, loss = self.sess.run([self.policy.train_op, self.policy.rec_new_log_prob, self.policy.loss], feed_dict=self.feed_dict)
        self.sess.run(self.policy.update_op)

        return loss, log_prob

def discounted_cumulative_sum(x, discount, initial=0):
    discounted_cumulative_sums = []
    discounted_cumulative_sum = initial
    for element in reversed(x):
        discounted_cumulative_sum = element + discount * discounted_cumulative_sum 
        discounted_cumulative_sums.append(discounted_cumulative_sum) 
    return list(reversed(discounted_cumulative_sums))


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.seed(1)
    # env = env.unwrapped

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    trainer = Train(obs_dim, act_dim)
    trainer.build_summary()
    
    MAX_EPISODE = 4000 

    for episode in range(MAX_EPISODE):
        obs = env.reset()
        t = 0 
        value_buffer, reward_buffer, act_buffer, obs_buffer, next_obs_buffer = [], [], [], [], [] 
        while True:
            t = t + 1
            obs = obs[None, :]
            act, value = trainer.policy.predict(trainer.sess, feed_dict={trainer.obs: obs})
            obs = obs.squeeze(axis=0)
            obs_next, reward, done, info = env.step(int(act[0]))
            
            if done:
                value_buffer.append(float(value))
                reward_buffer.append(reward)
                obs_buffer.append(obs)
                act_buffer.append(int(act[0]))
                next_obs_buffer.append(obs_next)
                obs = env.reset()
                break 

            value_buffer.append(float(value))
            reward_buffer.append(reward)
            obs_buffer.append(obs)
            act_buffer.append(int(act[0]))
            next_obs_buffer.append(obs_next)

            obs = obs_next 

        value_buffer.append(0)
        ret_buffer = discounted_cumulative_sum(reward_buffer, gamma)
        delta = np.array(reward_buffer) + np.array(value_buffer[1:]) * gamma - np.array(value_buffer[:-1])      # TD 误差

        adv_buffer = np.array(discounted_cumulative_sum(delta, gamma * 0.95, 0))
        # adv_buffer = delta 
        adv_buffer = (adv_buffer - np.mean(adv_buffer))/ (np.std(adv_buffer) + 1e-8)
        
        # print('adv_buffer', adv_buffer)
        obs_buffer = np.array(obs_buffer)
        act_buffer = np.array(act_buffer)
        ret_buffer = np.array(ret_buffer)

        loss, log_prob = trainer.train(obs_buffer, act_buffer, adv_buffer, ret_buffer, sum(reward_buffer))

        summary = trainer.output_summary()
        # trainer.train_writer.add_summary(summary, episode)

        print("episode: {:5}, reward: {:5}, total loss: {:8.2f}".format(episode, sum(reward_buffer), loss))

