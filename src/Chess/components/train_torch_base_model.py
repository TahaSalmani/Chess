import os
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from Chess import logger
from Chess.components.torch_base_model import ChessPolicyNet
from Chess.entity.config_entity import PrepareTorchTrainingConfig
from Chess.components.torch_base_model import PrepareTorchBaseModelConfig

class ModelCheckpoint:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.best_loss = float("inf")

    def __call__(self, current_loss: float, model: torch.nn.Module):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            torch.save(model.state_dict(), self.filepath)
            logger.info(f"Checkpoint saved to {self.filepath} | Best loss: {self.best_loss:.4f}")


class EarlyStopping:
    def __init__(self, patience: int = 5, delta: float = 0.0001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class TrainTorchBaseModel:
    def __init__(self, config: PrepareTorchTrainingConfig, base_model_config: PrepareTorchBaseModelConfig):
        self.config = config
        self.base_model_config = base_model_config

        self.tb_dir = os.path.join(self.config.root_dir, "tensorboard_logs")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tb_dir)

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training using device: {device}")

        x_tensor = torch.load(self.config.x_dataset)
        y_tensor = torch.load(self.config.y_dataset).long()

        dataset = TensorDataset(x_tensor, y_tensor)

        val_split = getattr(self.config, "params_validation_split", 0.2)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=True,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False,
            num_workers=0
        )

        model = ChessPolicyNet(config=self.base_model_config).to(device)

        if os.path.exists(self.config.model_path):
            model.load_state_dict(torch.load(self.config.model_path, map_location=device))
            logger.info(f"Base model loaded from {self.config.model_path}")

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.params_learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        checkpoint_path = os.path.join(self.config.root_dir, "best_model.pth")
        checkpoint = ModelCheckpoint(filepath=checkpoint_path)
        early_stopping = EarlyStopping(patience=getattr(self.config, 'params_PATIENCE', 5))

        logger.info("Starting Training...")

        for epoch in range(self.config.params_epochs):
            model.train()
            running_loss = 0.0

            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)

            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for i, (batch_x, batch_y) in enumerate(val_loader):
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            val_loss = val_loss / len(val_loader)

            self.writer.add_scalar("Loss/train", epoch_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)

            logger.info(
                f"Epoch [{epoch + 1}/{self.config.params_epochs}] - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")

            checkpoint(val_loss, model)
            early_stopping(val_loss)

            if early_stopping.should_stop:
                logger.info("Early stopping .")
                break

        self.writer.close()
        final_model_path = os.path.join(self.config.root_dir, "trained_model.pth")
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Final trained model saved at: {final_model_path}")