import os
import random
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from clearml import Task

from utils.vizualize_clustering import visualize_clustering_on_image

class ClusteringVisualizationCallback(pl.Callback):
    def __init__(self, val_dataset, eps=0.5, min_samples=5, img_size=224, output_dir="clustering_plots"):

        super().__init__()
        self.val_dataset = val_dataset
        self.eps = eps
        self.min_samples = min_samples
        self.img_size = img_size
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.task = Task.current_task()
        
    def on_validation_epoch_end(self, trainer, pl_module):
        idx = random.randint(0, len(self.val_dataset) - 1)
        image, _ = self.val_dataset[idx]
        
        fig = visualize_clustering_on_image(
            pl_module.model,
            image,
            eps=self.eps, min_samples=self.min_samples
        )
        epoch = trainer.current_epoch
        
        logger = self.task.get_logger()
        logger.report_matplotlib_figure(
            title=f"Clustering Visualization Epoch {epoch}",
            series="Clusters",
            iteration=epoch,
            figure=fig
        )

        print(f"Uploaded clustering visualization for epoch {epoch} to ClearML plot.")
        file_path = os.path.join(self.output_dir, f"clustering_epoch_{epoch}.png")
        fig.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        