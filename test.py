# test 2023/5/19 3:09

import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.set(title='MARL-loss', ylabel='loss', xlabel='episodes')
episodes = range(0, 3000)
ax1.plot(episodes, episodes)


ax2 = fig.add_subplot(223)
ax2.set(title='MARL-loss', ylabel='loss', xlabel='episodes')
episodes = range(0, 3000)
ax2.plot(episodes, episodes)
plt.show()