import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Chess import logger
from Chess.components.policy_value_torch_model import ValuePolicyNet
from Chess.config.configuration import PrepareTorchTrainingConfig

class SelfPlayDataset(Dataset):
    def __init__(self, data_dir ,):

        self.x = torch.load(os.path.join(data_dir, 'selfplay_x.pt'))
        self.policy = torch.load(os.path.join(data_dir, 'selfplay_policy.pt'))
        self.value = torch.load(os.path.join(data_dir, 'selfplay_value.pt'))


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.policy[idx], self.value[idx]



def train_rl_step(  model, data_dir, epochs=10, batch_size=16, lr=0.0005, device='cpu'):

    dataset = SelfPlayDataset(data_dir )
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    cross_entropy = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    model.to(device)
    model.train()




    logger.info(f"Starting RL training on {len(dataset)} ... ")

    for epoch in range(epochs):
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0

        for batch_x, batch_policy, batch_value in dataloader:
            batch_x = batch_x.to(device)
            batch_policy = batch_policy.to(device)
            batch_value = batch_value.to(device).unsqueeze(1)  # [batch_size, 1]

            optimizer.zero_grad()

            pred_policy, pred_value = model(batch_x)

            loss_p = cross_entropy(pred_policy, batch_policy)
            loss_v = mse_loss(pred_value, batch_value)
            loss = loss_p + loss_v

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_policy_loss += loss_p.item()
            total_value_loss += loss_v.item()

        avg_loss = total_loss / len(dataloader)
        avg_p_loss = total_policy_loss / len(dataloader)
        avg_v_loss = total_value_loss / len(dataloader)

        logger.info(
            f"Epoch [{epoch + 1}/{epochs}] | Total Loss: {avg_loss:.4f} (Policy: {avg_p_loss:.4f}, Value: {avg_v_loss:.4f})"
        )

    os.makedirs("artifacts/prepare_torch_callbacks", exist_ok=True)
    save_path = "artifacts/prepare_torch_callbacks/rl_trained_model.pth"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Updated RL model saved successfully to {save_path}")


if __name__ == "__main__":
    from Chess.entity.config_entity import PrepareTorchBaseModelConfig

    config = PrepareTorchBaseModelConfig(
        root_dir="artifacts/prepare_torch_base_model",
        model_path="artifacts/prepare_torch_base_model/base_model.pth",
        params_D_MODEL=128,
        params_NUM_HEADS=4,
        params_FF_DIM=512,
        params_NUM_LAYERS=4,
        params_IN_CHANNELS=12,
        params_NUM_MOVES=4096,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ValuePolicyNet(config)

    weights_path = "artifacts/prepare_torch_callbacks/trained_model.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)

    train_rl_step(

        model=model,
        data_dir="artifacts/self_play_data",
        epochs=2,
        batch_size=16,
        lr=0.0005,
        device=device
    )