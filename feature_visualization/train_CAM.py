from feature_visualization import compute_CAM


if __name__ == '__main__':
    trainer = compute_CAM.CamClass()
    trainer.train()
    trainer.eval()
