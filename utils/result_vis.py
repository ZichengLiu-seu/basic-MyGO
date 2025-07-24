import os

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
import numpy as np


def pred_spilt(preds, gts, inverse):
    pred_x = []
    pred_y = []
    # pred_z = []
    true_x = []
    true_y = []
    # true_z = []
    # inv_pred = inverse([pred.cpu().detach().numpy() for pred in preds])
    # inv_gt = inverse([gt.cpu().detach().numpy() for gt in gts])
    inv_pred = [pred.cpu().detach().numpy() for pred in preds]
    inv_gt = [gt.cpu().detach().numpy() for gt in gts]
    for pred, true in zip(inv_pred, inv_gt):
        pred_x.append(pred[0])
        pred_y.append(pred[1])
        # pred_z.append(pred[1])
        true_x.append(true[0])
        true_y.append(true[1])
        # true_z.append(true[1])
    # return pred_x, pred_y, pred_z, true_x, true_y, true_z
    return pred_x, pred_y, true_x, true_y


def route_1d(args, pred_x, pred_y, true_x, true_y):
    fig = plt.figure()
    axx = fig.add_subplot(121)
    axy = fig.add_subplot(122)
    # axz = fig.add_subplot(133)
    axx.set_title("x_pos")
    axy.set_title("y_pos")
    # axz.set_title("z_pos")
    axx.plot(range(len(pred_x)), pred_x, label="pred")
    axx.plot(range(len(true_x)), true_x, label="gt")
    axy.plot(range(len(pred_y)), pred_y, label="pred")
    axy.plot(range(len(true_y)), true_y, label="gt")
    # axz.plot(range(len(pred_z)), pred_z, label="pred")
    # axz.plot(range(len(true_z)), true_z, label="gt")
    axx.legend()
    axy.legend()
    # axz.legend()

    path = 'results/{0}/{1}'.format(args.model_type, args.interaction_type)
    if not os.path.exists(path):
        os.mkdir(path)
    fig.savefig(path + '/1d-result.png')
    plt.close()


def route_2d(args, pred_x, pred_y, true_x, true_y):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_title("pred route")
    ax2.set_title("true route")
    ax1.set_xlabel("dir_x")
    ax1.set_ylabel("dir_z")
    ax2.set_xlabel("dir_x")
    ax2.set_ylabel("dir_z")
    ax1.plot(pred_x, pred_y, label="pred")
    ax2.plot(true_x, true_y, label="gt")

    path = 'results/{0}/{1}'.format(args.model_type, args.interaction_type)
    if not os.path.exists(path):
        os.mkdir(path)
    fig.savefig(path + '/2d-result.png')
    plt.close()


def route_3d(args, pred_xs, pred_zs, pred_ys, gt_xs, gt_zs, gt_ys):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    ax1.scatter(pred_xs, pred_zs, pred_ys)
    ax2.scatter(gt_xs, gt_zs, gt_ys)
    ax1.set_xlabel('X Label')
    ax1.set_ylabel('Y Label')
    ax1.set_zlabel('Z Label')
    ax1.set_title('Pred_route')
    ax2.set_xlabel('X Label')
    ax2.set_ylabel('Y Label')
    ax2.set_zlabel('Z Label')
    ax2.set_title('Real_route')
    fig.savefig('results/Sup Weight{}/3d-result.png'.format(args.sup_weight))
    plt.close()


def ablation_on_arch():
    all_mse = [0.0467, 0.0388]
    all_mde = [22.01, 18.42]
    prsd_all_mse = 0.0295
    prsd_all_mde = 16.62
    drawing('All', all_mse, all_mde, prsd_all_mse, prsd_all_mde)

    sw_mse = [0.3565, 0.3047]
    sw_mde = [42.38, 38.28]
    prsd_sw_mse = 0.2497
    prsd_sw_mde = 32.79
    drawing('Swing', sw_mse, sw_mde, prsd_sw_mse, prsd_sw_mde)

    to_mse = [0.0616, 0.0490]
    to_mde = [23.25, 22.75]
    prsd_to_mse = 0.0441
    prsd_to_mde = 18.09
    drawing('Touchpad', to_mse, to_mde, prsd_to_mse, prsd_to_mde)

    te_mse = [0.0735, 0.0571]
    te_mde = [27.30, 26.95]
    prsd_te_mse = 0.0404
    prsd_te_mde = 22.42
    drawing('Teleport', te_mse, te_mde, prsd_te_mse, prsd_te_mde)


def drawing(datatype, mse, mde, prsd_mse, prsd_mde):
    name = ["Four Transformer", "Four LSTM"]
    x = np.array([0.3, 0.6])
    plt.xlim(-0.5, len(name) - 0.5)
    width = 0.15
    mse_upper = 1.3*max(mse)
    mde_upper = 2*max(mde)
    mse_bar_color = '#568C87'
    mse_line_color = '#005C3A'
    mde_bar_color = '#F2DE79'
    mde_line_color = '#FFB300'

    plt.figure()
    ax1 = plt.gca()
    mse_bars = ax1.bar(x - width/3, mse, width/2, color=mse_bar_color, label='MSE')
    ax1.set_ylabel("MSE")
    ax1.set_ylim(0, mse_upper)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2 = plt.gca().twinx()
    mde_bars = ax2.bar(x + width/3, mde, width/2, color=mde_bar_color, label='MDE')
    ax2.set_ylabel("MDE")
    ax2.set_ylim(0, mde_upper)

    ax1.axhline(prsd_mse, color=mse_line_color, linestyle='--', linewidth=1.5, label='Proposed. MSE', zorder=10)
    ax2.axhline(prsd_mde, color=mde_line_color, linestyle='-.', linewidth=1.5, label='Proposed. MDE', zorder=10)
    plt.xlabel("Shared Embedding")
    plt.xticks(x, name)

    for bar in mse_bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2, height,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=9
        )

    for bar in mde_bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2, height,
            f'{height:.2f}',
            ha='center', va='bottom', fontsize=9
        )

    ax1.annotate(
        f'{prsd_mse:.4f}',
        xy=(-0.05, prsd_mse),
        xytext=(-5, 0),
        textcoords='offset points',
        ha='right', va='center',
        color=mse_line_color,
        annotation_clip=False
    )
    ax2.annotate(
        f'{prsd_mde:.2f}',
        xy=(1.05, prsd_mde),
        xytext=(5, 0),
        textcoords='offset points',
        ha='left', va='center',
        color=mde_line_color,
        annotation_clip=False
    )
    lines = [
        plt.Line2D([0], [0], color=mse_bar_color, linestyle='--', linewidth=1.5),
        plt.Line2D([0], [0], color=mde_bar_color, linestyle='-.', linewidth=1.5),
    ]
    bars = [
        Patch(facecolor=mse_line_color, edgecolor='none'),
        Patch(facecolor=mde_line_color, edgecolor='none'),
    ]
    plt.legend(bars + lines, ['MSE', 'MDE', 'Proposed. MSE', 'Proposed. MDE'],
               loc='upper right', bbox_to_anchor=(1, 1))
    plt.savefig("Shared Embedding Ablation in {}".format(datatype))
    plt.close()


def ablation_on_eye():
    all_bar_color = '#568C87'
    motion_bar_color = '#F2DE79'
    motion_mse_line_color = '#FFB300'
    name = ["Maze", "Park", "Village"]
    x = np.array([0, 1, 2])
    width = 0.4

    # MSE graph
    all_mse = [0.0215, 0.0585, 0.0463]
    motion_mse = [0.0301, 0.065, 0.0523]
    prsd_mse = 0.0285
    mse_upper = 1.4 * max(motion_mse)

    plt.figure()
    ax1 = plt.gca()
    all_mse_bars = ax1.bar(x - width/3, all_mse, width/2, color=all_bar_color, label='MSE')
    ax1.set_ylabel("MSE")
    ax1.set_ylim(0, mse_upper)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    motion_mse_bars = ax1.bar(x + width/3, motion_mse, width/2, color=motion_bar_color, label='MDE')

    ax1.axhline(prsd_mse, color='black', linestyle='--', linewidth=1.5, label='Proposed. MSE', zorder=10)
    plt.xticks(x, name)

    for bar in all_mse_bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2, height,
            f'{height:.3f}',
            ha='center', va='bottom', fontsize=9
        )

    for bar in motion_mse_bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2, height,
            f'{height:.3f}',
            ha='center', va='bottom', fontsize=9
        )

    ax1.annotate(
        f'{prsd_mse:.3f}',
        xy=(2.6, prsd_mse),
        xytext=(5, 0),
        textcoords='offset points',
        ha='right', va='center',
        color='black',
        annotation_clip=False
    )
    line = [plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5)]
    bars = [
        Patch(facecolor=all_bar_color, edgecolor='none'),
        Patch(facecolor=motion_bar_color, edgecolor='none')
        ]
    plt.legend(bars + line, ['MSE On All feature', 'MSE w/o Gaze Data', 'Proposed. MSE'],
               loc='upper right', bbox_to_anchor=(1, 1))
    plt.savefig("MSE Ablation in Eye")
    plt.close()

    # MDE graph
    all_mde = [18.68, 25.80, 26.85]
    motion_mde = [20.52, 26.89, 28.05]
    prsd_mde = 16.95
    mde_upper = 1.4 * max(motion_mde)

    plt.figure()
    ax1 = plt.gca()
    all_mde_bars = ax1.bar(x - width/3, all_mde, width/2, color=all_bar_color, label='MSE')
    ax1.set_ylabel("MDE")
    ax1.set_ylim(0, mde_upper)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    motion_mde_bars = ax1.bar(x + width/3, motion_mde, width/2, color=motion_bar_color, label='MDE')

    ax1.axhline(prsd_mde, color='black', linestyle='--', linewidth=1.5, label='Proposed. MSE', zorder=10)
    plt.xticks(x, name)

    for bar in all_mde_bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2, height,
            f'{height:.2f}',
            ha='center', va='bottom', fontsize=9
        )

    for bar in motion_mde_bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2, height,
            f'{height:.2f}',
            ha='center', va='bottom', fontsize=9
        )
    ax1.annotate(
        f'{prsd_mde:.2f}',
        xy=(2.6, prsd_mde),
        xytext=(5, 0),
        textcoords='offset points',
        ha='right', va='center',
        color='black',
        annotation_clip=False
    )
    line = [plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5)]
    bars = [
        Patch(facecolor=all_bar_color, edgecolor='none'),
        Patch(facecolor=motion_bar_color, edgecolor='none')
        ]
    plt.legend(bars + line, ['MDE On All feature', 'MDE w/o Gaze Data', 'Proposed. MDE'],
               loc='upper right', bbox_to_anchor=(1, 1))
    plt.savefig("MDE Ablation in Eye")
    plt.close()


if __name__ == "__main__":
    ablation_on_eye()
