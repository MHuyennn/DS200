from trainer import TrainingConfig, SparkConfig, Trainer
from cnn_model import CNN 
from transforms import Transforms, Normalize

transforms = Transforms([
    Normalize(mean=(0.5,), std=(0.5,))
])

if __name__ == "__main__":
    train_config = TrainingConfig(
        batch_size=64,
        max_epochs=10,
        learning_rate=1e-3,
        model_name="CNN-HandWritten",
        ckpt_interval_batch=1000,
        load_model="epoch-5"
    )

    spark_config = SparkConfig(
        batch_interval=4,
        port=8000
    )

    cnn = CNN(num_classes=39) 
    trainer = Trainer(cnn, "train", train_config, spark_config, transforms)
    trainer.train()