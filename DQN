import pdb
import numpy as np
import simple_manhunt_v0
import tensorflow as tf
import collections
import random

class AgentNet():
    def __init__(
            self,
            name,
            n_action,
            n_state,
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=50000,
            batch_size=32,
            e_greedy_increment=None
    ):
        self.n_action = n_action  #action个数
        self.n_state = n_state      #state 个数
        self.alpha = learning_rate     #学习率
        self.gamma = reward_decay
        self.epsilon_max = e_greedy #贪心
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size  #记忆库容量
        self.batch_size = batch_size    #每批学习的个数
        self.epsilon_increment = e_greedy_increment     #逐步增大贪心比例
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.name = name
        self.sess = sess
        self.build()

        self.learn_step_counter = 0
        self.memory = []
        self.losses = []

        '''复制网络'''
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def build(self,hidden_dim=64):
        with tf.variable_scope(self.name):
            self.s = tf.placeholder(tf.float32, [None, self.n_state], name='s')  # input State
            self.s_ = tf.placeholder(tf.float32, [None, self.n_state], name='s_')  # input Next State
            self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
            self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

            with tf.variable_scope("eval_net"):
                hidden = tf.layers.dense(self.s,hidden_dim,activation=tf.nn.relu)
                hidden = tf.layers.dense(hidden, hidden_dim, activation=tf.nn.relu)
                self.q_eval = tf.layers.dense(hidden,self.n_action)

            with tf.variable_scope("target_net"):
                hidden = tf.layers.dense(self.s_, hidden_dim, activation=tf.nn.relu)
                hidden = tf.layers.dense(hidden, hidden_dim, activation=tf.nn.relu)
                self.q_next = tf.layers.dense(hidden, self.n_action)

            with tf.variable_scope('q_target'):
                q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')
                self.q_target = tf.stop_gradient(q_target)

            with tf.variable_scope("q_eval"):
                a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
                self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)

            with tf.variable_scope("loss"):
                    self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
            with tf.variable_scope('train'):
                self._train_op = tf.train.RMSPropOptimizer(self.alpha).minimize(self.loss)


    def choose_action(self,state):
        if np.random.random() > self.epsilon:
            action_chosen = np.random.randint(0, self.n_action)
        else:
            state = state[np.newaxis, :]
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: np.array(state)})
            action_chosen = np.argmax(actions_value)
        return action_chosen

    def train(self,state,reward,action,state_next):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            # print('\ntarget_params_replaced\n')

        _,loss = self.sess.run([self._train_op,self.loss],
                               feed_dict={
                                   self.s:np.array(state),
                                   self.s_:np.array(state_next),
                                   self.a:np.array(action),
                                   self.r:np.array(reward)
                               })
        self.learn_step_counter += 1
        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        return loss





def SMDP_Run(env,net1,net2,mode,eposide):
    # pdb.set_trace()
    results = []
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state"])

    update_iter=0
    for ep in range(eposide):
        env.reset()
        # players = ["agent_0","adversary_0"]
        players = ["adversary_0"]   # train just one team
        for player in players:
            while not env.done():
                obs,reward,done,info = env.last()
                agent = env.get_agent()
                if agent[:5] != player[:5]:
                    if done:
                        break
                    else:
                        net = net2 if player=="agent_0" else net1
                        action = net.choose_action(obs)
                    env.step(action)
                else:
                    if done:
                        break
                    else:
                        net = net1 if player=="agent_0" else net2
                        action =env.action_space(agent).sample()
                    env.step(action)
                    next_obs = env.agent_observation(agent)

                    if len(net.memory) > net.memory_size:
                        net.memory.pop(0)
                    reward = reward if player=="agent_0" else -reward
                    net.memory.append(Transition(obs, action, reward, next_obs))

                    if len(net.memory) > net.batch_size * 4:
                        batch_transition = random.sample(net.memory, net.batch_size)
                        batch_state, batch_action, batch_reward, batch_next_state = map(np.array, zip(*batch_transition))
                        loss = net.train(state=batch_state,
                                   reward=batch_reward,
                                   action=batch_action,
                                   state_next=batch_next_state,
                                   )
                        update_iter += 1
                        net.losses.append(loss)

                # if ep == eposide-1:
                # env.render(mode=mode)
            reward = reward if player == "agent_0" else -reward
            results.append(reward)

            env.reset()
    return np.mean(results),np.mean(agent1.losses),np.mean(agent2.losses)



if __name__ == "__main__":
    runs = 1000
    eposide = 1000


    env = simple_manhunt_v0.env()
    mode = env.metadata.get("render.modes")[0]
    # print(mode)
    with tf.Session() as sess:
        agent1 = AgentNet("agent",5,14)
        agent1.sess.run(tf.global_variables_initializer())
        agent2 = AgentNet("adversary",5,16)
        agent2.sess.run(tf.global_variables_initializer())

        for j in range(runs):
            result,loss = SMDP_Run(env,agent1,agent2,mode,eposide)
            print(result,loss)

    env.close()




