from argparse import ArgumentParser
from training import compute_ARM, compute_CAM, run_grad_CAM



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("retrain", help="re-train the model", action='store_true')
    parser.add_argument("type", help="cam, arm, grad_cam", type=str)
    args = parser.parse_args()

    if args['type'] == 'cam':
        trainer = compute_CAM.CamClass()
        if args['retrain']:
            trainer.train()
        trainer.eval()
    elif args['type'] == 'arm':
        trainer = compute_ARM.CarmClass()
        if args['retrain']:
            trainer.train()
        trainer.eval()
    elif args['type'] == 'grad_cam':
        run_grad_CAM.run()
    else:
        raise NotImplementedError




