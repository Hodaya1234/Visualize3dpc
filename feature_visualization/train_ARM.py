from feature_visualization import compute_ARM


if __name__ == '__main__':
    trainer = compute_ARM.CarmClass()
    trainer.train()
    trainer.eval()
