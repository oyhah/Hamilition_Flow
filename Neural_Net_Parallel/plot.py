import matplotlib.pyplot as plt
import numpy as np


loss_GD_momentum09 = np.load('results/loss_CIFAR10_GD_l2_theta0.10000_K20_momentum0.9.npy')
acc_GD_momentum09 = np.load('results/acc_CIFAR10_GD_l2_theta0.10000_K20_momentum0.9.npy')

loss_GD_momentum00 = np.load('results/loss_CIFAR10_GD_l2_theta0.10000_K10_momentum0.000.npy')
acc_GD_momentum00 = np.load('results/acc_CIFAR10_GD_l2_theta0.10000_K10_momentum0.000.npy')

loss_LF_theta01_K20 = np.load('results/loss_CIFAR10_LF_l2_theta0.10000_K20_momentum0.000.npy')
acc_LF_theta01_K20 = np.load('results/acc_CIFAR10_LF_l2_theta0.10000_K20_momentum0.000.npy')

loss_LF_theta001_K20 = np.load('results/loss_CIFAR10_LF_l2_theta0.01000_K20_momentum0.000.npy')
acc_LF_theta001_K20 = np.load('results/acc_CIFAR10_LF_l2_theta0.01000_K20_momentum0.000.npy')

loss_LF_L1_theta001_K20 = np.load('results/loss_CIFAR10_LF_l1_theta0.01000_K20_momentum0.000.npy')
acc_LF_L1_theta001_K20 = np.load('results/acc_CIFAR10_LF_l1_theta0.01000_K20_momentum0.000.npy')

loss_SI_theta01_K20 = np.load('results/loss_CIFAR10_SI_l2_theta0.10000_K20_momentum0.000.npy')
acc_SI_theta01_K20 = np.load('results/acc_CIFAR10_SI_l2_theta0.10000_K20_momentum0.000.npy')

loss_LF_theta01_K10 = np.load('results/loss_CIFAR10_LF_l2_theta0.10000_K10_momentum0.000.npy')
acc_LF_theta01_K10 = np.load('results/acc_CIFAR10_LF_l2_theta0.10000_K10_momentum0.000.npy')

loss_LF_theta01_K30 = np.load('results/loss_CIFAR10_LF_l2_theta0.10000_K30_momentum0.000.npy')
acc_LF_theta01_K30 = np.load('results/acc_CIFAR10_LF_l2_theta0.10000_K30_momentum0.000.npy')

loss_LF_theta01_K50 = np.load('results/loss_CIFAR10_LF_l2_theta0.10000_K50_momentum0.000.npy')
acc_LF_theta01_K50 = np.load('results/acc_CIFAR10_LF_l2_theta0.10000_K50_momentum0.000.npy')

loss_LF_theta01_K100 = np.load('results/loss_CIFAR10_LF_l2_theta0.10000_K100_momentum0.000.npy')
acc_LF_theta01_K100 = np.load('results/acc_CIFAR10_LF_l2_theta0.10000_K100_momentum0.000.npy')

loss_LF_theta05_K20 = np.load('results/loss_CIFAR10_LF_l2_theta0.50000_K20_momentum0.000.npy')
acc_LF_theta05_K20 = np.load('results/acc_CIFAR10_LF_l2_theta0.50000_K20_momentum0.000.npy')

loss_LF_theta005_K20 = np.load('results/loss_CIFAR10_LF_l2_theta0.05000_K20_momentum0.000.npy')
acc_LF_theta005_K20 = np.load('results/acc_CIFAR10_LF_l2_theta0.05000_K20_momentum0.000.npy')

loss_LF_theta0005_K20 = np.load('results/loss_CIFAR10_LF_l2_theta0.00500_K20_momentum0.000.npy')
acc_LF_theta0005_K20 = np.load('results/acc_CIFAR10_LF_l2_theta0.00500_K20_momentum0.000.npy')

loss_LF_theta0001_K20 = np.load('results/loss_CIFAR10_LF_l2_theta0.00100_K20_momentum0.000.npy')
acc_LF_theta0001_K20 = np.load('results/acc_CIFAR10_LF_l2_theta0.00100_K20_momentum0.000.npy')

x_axis = np.arange(len(loss_GD_momentum09)) + 1

# fig1 = plt.figure()
# plt.plot(x_axis, acc_GD_momentum09, linewidth=2, marker='o', markersize=10, markevery=5, color='firebrick', label='GDM')
# plt.plot(x_axis, acc_GD_momentum00, linewidth=2, marker='>', markersize=10, markevery=5, color='black', label='GD')
# plt.plot(x_axis, acc_LF_theta01_K20, linewidth=2, marker='v', markersize=10, markevery=5, color='royalblue', label='HOME-LF-0.1')
# plt.plot(x_axis, acc_LF_theta001_K20, linewidth=2, marker='^', markersize=10, markevery=5, color='deepskyblue', label='HOME-LF-0.01')
# plt.plot(x_axis, acc_LF_L1_theta001_K20, linewidth=2, marker='<', markersize=10, markevery=5, color='orange', label='HOME-LF-L1-0.01')
# plt.plot(x_axis, acc_SI_theta01_K20, linewidth=2, marker='*', markersize=10, markevery=5, color='peru', label='HOME-SI-0.1')
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('Epochs', fontsize=15)
# plt.ylabel('Test Accuracy (%)', fontsize=15)
# plt.legend(prop = {'size': 12})
# fig1.suptitle('Test Accuracy', fontsize=20)
# fig1.savefig('picture/accuracy.eps')

# fig2 = plt.figure()
# plt.plot(x_axis, loss_GD_momentum09, linewidth=2, marker='o', markersize=10, markevery=5, color='firebrick', label='GDM')
# plt.plot(x_axis, loss_GD_momentum00, linewidth=2, marker='>', markersize=10, markevery=5, color='black', label='GD')
# plt.plot(x_axis, loss_LF_theta01_K20, linewidth=2, marker='v', markersize=10, markevery=5, color='royalblue', label='HOME-LF-0.1')
# plt.plot(x_axis, loss_LF_theta001_K20, linewidth=2, marker='^', markersize=10, markevery=5, color='deepskyblue', label='HOME-LF-0.01')
# # plt.plot(x_axis, loss_LF_L1_theta001_K20, linewidth=2, marker='<', markersize=10, markevery=5, color='orange', label='HOME-LF-L1-0.01')
# plt.plot(x_axis, loss_SI_theta01_K20, linewidth=2, marker='*', markersize=10, markevery=5, color='peru', label='HOME-SI-0.1')
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('Epochs', fontsize=15)
# plt.ylabel('Loss', fontsize=15)
# plt.legend(prop = {'size': 12})
# fig2.suptitle('Loss', fontsize=20)
# fig2.savefig('picture/loss.eps')
# plt.show()



fig1 = plt.figure()
plt.plot(x_axis, acc_LF_theta05_K20, linewidth=2, marker='o', markersize=10, markevery=5, color='firebrick', label='HOME-LF-0.5')
plt.plot(x_axis, acc_LF_theta01_K20, linewidth=2, marker='>', markersize=10, markevery=5, color='black', label='HOME-LF-0.1')
plt.plot(x_axis, acc_LF_theta005_K20, linewidth=2, marker='v', markersize=10, markevery=5, color='royalblue', label='HOME-LF-0.05')
plt.plot(x_axis, acc_LF_theta001_K20, linewidth=2, marker='^', markersize=10, markevery=5, color='deepskyblue', label='HOME-LF-0.01')
plt.plot(x_axis, acc_LF_theta0005_K20, linewidth=2, marker='<', markersize=10, markevery=5, color='orange', label='HOME-LF-0.005')
plt.plot(x_axis, acc_LF_theta0001_K20, linewidth=2, marker='*', markersize=10, markevery=5, color='peru', label='HOME-LF-0.001')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Test Accuracy (%)', fontsize=15)
plt.legend(prop = {'size': 12})
fig1.suptitle('Test Accuracy', fontsize=20)
fig1.savefig('picture/accuracy_theta.eps')

fig2 = plt.figure()
plt.plot(x_axis, loss_LF_theta05_K20, linewidth=2, marker='o', markersize=10, markevery=5, color='firebrick', label='HOME-LF-0.5')
plt.plot(x_axis, loss_LF_theta01_K20, linewidth=2, marker='>', markersize=10, markevery=5, color='black', label='HOME-LF-0.1')
plt.plot(x_axis, loss_LF_theta005_K20, linewidth=2, marker='v', markersize=10, markevery=5, color='royalblue', label='HOME-LF-0.05')
plt.plot(x_axis, loss_LF_theta001_K20, linewidth=2, marker='^', markersize=10, markevery=5, color='deepskyblue', label='HOME-LF-0.01')
plt.plot(x_axis, loss_LF_theta0005_K20, linewidth=2, marker='<', markersize=10, markevery=5, color='orange', label='HOME-LF-0.005')
plt.plot(x_axis, loss_LF_theta0001_K20, linewidth=2, marker='*', markersize=10, markevery=5, color='peru', label='HOME-LF-0.001')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.legend(prop = {'size': 12})
fig2.suptitle('Loss', fontsize=20)
fig2.savefig('picture/loss_theta.eps')
plt.show()