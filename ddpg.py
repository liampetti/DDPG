# ================================================
# Modified from the work of Patrick Emami:
#       Implementation of DDPG - Deep Deterministic Policy Gradient
#       Algorithm and hyperparameter details can be found here:
#           http://arxiv.org/pdf/1509.02971v2.pdf
#
# Removed TFLearn dependency
# Added Ornstein Uhlenbeck noise function
# Added reward discounting
# Works with discrete actions spaces (Cartpole)
# Tested on CartPole-v0 & -v1 & Pendulum-v0
# Author: Liam Pettigrew
# =================================================
import tensorflow as tf
import numpy as np
import gym

from replay_buffer import ReplayBuffer
from noise import Noise
from reward import Reward
from actor_lstm import ActorNetwork
from critic_lstm import CriticNetwork


# ==========================
#   Training Parameters
# ==========================
# Maximum episodes run
MAX_EPISODES = 1000
# Max episode length
MAX_EP_STEPS = 1000
# Episodes with noise
NOISE_MAX_EP = 200
# Noise parameters - Ornstein Uhlenbeck
DELTA = 0.5 # The rate of change (time)
SIGMA = 0.5 # Volatility of the stochastic processes
OU_A = 3. # The rate of mean reversion
OU_MU = 0. # The long run average interest rate
# Reward parameters
REWARD_FACTOR = 0.1 # Total episode reward factor
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001

# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = False
# Use Gym Monitor
GYM_MONITOR_EN = True
# Gym environment
ENV_NAME = 'CartPole-v0' # Discrete: Reward factor = 0.1
#ENV_NAME = 'CartPole-v1' # Discrete: Reward factor = 0.1
#ENV_NAME = 'Pendulum-v0' # Continuous: Reward factor = 0.01
# Directory for storing gym results
MONITOR_DIR = './results/' + ENV_NAME
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 100000
MINIBATCH_SIZE = 100

# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================
def train(sess, env, actor, critic, noise, reward, discrete):
    # Set up summary writer
    summary_writer = tf.summary.FileWriter("ddpg_summary")

    sess.run(tf.global_variables_initializer())

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    # Initialize noise
    ou_level = 0.

    for i in range(MAX_EPISODES):
        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        # Clear episode buffer
        episode_buffer = np.empty((0,5), float)

        for j in range(MAX_EP_STEPS):
            if RENDER_ENV:
                env.render()

            a = actor.predict(np.reshape(s, (1, actor.s_dim)))

            # Add exploration noise
            if i < NOISE_MAX_EP:
                ou_level = noise.ornstein_uhlenbeck_level(ou_level)
                a = a + ou_level

            # Set action for discrete and continuous action spaces
            if discrete:
                action = np.argmax(a)
            else:
                action = a[0]

            s2, r, terminal, info = env.step(action)

            # Choose reward type
            ep_reward += r

            episode_buffer = np.append(episode_buffer, [[s, a, r, terminal, s2]], axis=0)

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(MINIBATCH_SIZE)

                # Calculate targets
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            # Set previous state for next step
            s = s2

            if terminal:
                # Reward system for episode
                #episode_buffer = reward.total(episode_buffer, ep_reward)
                episode_buffer = reward.discount(episode_buffer)

                # Add episode to replay buffer
                for step in episode_buffer:
                    replay_buffer.add(np.reshape(step[0], (actor.s_dim,)), np.reshape(step[1], (actor.a_dim,)), step[2], \
                                  step[3], np.reshape(step[4], (actor.s_dim,)))

                summary = tf.Summary()
                summary.value.add(tag='Perf/Reward', simple_value=float(ep_reward))
                summary.value.add(tag='Perf/Qmax', simple_value=float(ep_ave_max_q / float(j)))
                summary_writer.add_summary(summary, i)

                summary_writer.flush()

                print('| Reward: %.2i' % int(ep_reward), " | Episode", i, \
                '| Qmax: %.4f' % (ep_ave_max_q / float(j)))

                break


def main(_):
    with tf.Session() as sess:
        env = gym.make(ENV_NAME)
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)

        print(env.observation_space)
        print(env.action_space)

        state_dim = env.observation_space.shape[0]

        try:
            action_dim = env.action_space.shape[0]
            action_bound = env.action_space.high
            # Ensure action bound is symmetric
            assert (env.action_space.high == -env.action_space.low)
            discrete = False
            print('Continuous Action Space')
        except AttributeError:
            action_dim = env.action_space.n
            action_bound = 1
            discrete = True
            print('Discrete Action Space')

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             ACTOR_LEARNING_RATE, TAU)

        critic = CriticNetwork(sess, state_dim, action_dim,
                               CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())

        noise = Noise(DELTA, SIGMA, OU_A, OU_MU)
        reward = Reward(REWARD_FACTOR, GAMMA)

        if GYM_MONITOR_EN:
            if not RENDER_ENV:
                env.monitor.start(MONITOR_DIR, video_callable=False, force=True)
            else:
                env.monitor.start(MONITOR_DIR, force=True)

        try:
            train(sess, env, actor, critic, noise, reward, discrete)
        except KeyboardInterrupt:
            pass

        if GYM_MONITOR_EN:
            env.monitor.close()


if __name__ == '__main__':
    tf.app.run()