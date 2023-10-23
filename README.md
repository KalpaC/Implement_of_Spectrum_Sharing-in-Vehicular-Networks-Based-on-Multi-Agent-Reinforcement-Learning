# README

该项目为论文：《Spectrum Sharing in Vehicular Networks&#x20;
Based on Multi-Agent Reinforcement Learning
》的代码复现。

基于Python 3.6 + PyTorch 1.8.1+cu102，低于该版本不保证可正常运行。


## 基本说明：

该项目目标是复现，是为了重复这篇MARL论文呈现的结果，所以尽可能追求与其保持一致。因此，本论文只采用了引入Target网络的DQN，并未进一步改进网络结构或算法。但由于环境模拟器的不同，或该篇MARL的结果本身有一定问题，复现的结果并不理想。

在尝试逼近论文结果的过程中，本人将部分代码段替换为论文作者在GitHub上发布的代码，以尝试核查问题所在，但未果。但这导致项目中混杂了许多不属于本人的代码。

该项目是在时隔3个月后发布到GitHub上的，由于当初的版本管理不到位，本人难以将其复原回只有原创代码的版本，在此表示歉意。

## 发布于此的意义——纪念：

该论文是本人自行复现的第一篇论文，也是自己实现的第一个DQN网络甚至是神经网络，具有一定的纪念意义。

该项目是本人在北京邮电大学计算机学院参加保研实习考核时的任务。优化该项目的效果花费了本人大量时间精力。具体来说是熬了两个通宵，而后需要在半天时间内总结实验结果，制作答辩ppt。

为核对与论文源码效果的差距，基本上我逐一替换了项目中的每一部分，但最终并未达成接近完全一致的结果。虽然基本上是无用功，但依然值得记录。

## 可能的改进方法:

1.  优化状态函数——本项目没有任何trick，包括状态没有进行归一化等操作。论文源码进行了一种比较简单的归一化，我并不理解其参数的确定方法。
2.  优化奖励函数——根据实际情况权衡V2I和V2V部分的奖励占比？
3.  引入更多先验信息——例如宁愿浪费功率也要保证V2V传输率？那么如何给网络写入这种先验信息呢？
4.  环境计算有误？——奖励值与实际的行动不符，或状态值有偏差都可能导致训练异常。
5.  神经网络调优——似乎对于强化学习来说并不算重要
6.  引入更先进的DRL算法——在DQN的基础上，可以引入Rainbow DQN的各种改进措施
7.  完全重构算法。
