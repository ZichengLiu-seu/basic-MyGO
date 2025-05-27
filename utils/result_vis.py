import os

import matplotlib.pyplot as plt


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
