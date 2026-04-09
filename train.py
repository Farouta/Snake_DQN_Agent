import os
import torch
import hydra
import wandb
from tqdm import tqdm
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from torch.optim.lr_scheduler import ReduceLROnPlateau

from agent import AgentCNN
from environment import snake_environment

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):
    wandb.init(project=cfg.benchmark.wandb_project, job_type="training")

    snake_env = snake_environment(cfg.game.base_board_size, cfg.game.base_board_size)
    
    snake_agent = AgentCNN(
        memory_length=cfg.agent.memory_length,
        gamma=cfg.agent.gamma,
        learning_rate=cfg.agent.learning_rate,
        epsilon_min=cfg.agent.epsilon_end,
        epsilon_decay=cfg.agent.epsilon_decay,
        in_channels=snake_env.d_model,
        hidden_size=cfg.agent.hidden_size,
    )
    snake_agent.epsilon = cfg.agent.epsilon_start

    scheduler = ReduceLROnPlateau(
        snake_agent.optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    # Tracking Variables
    score_log = []
    loss_log = []
    length_log = []
    batch_deaths = {"wall": 0, "body": 0, "timeout": 0, "won": 0}
    best_score = -1


    progress_bar = tqdm(range(cfg.train.num_episodes + 1), desc="Training Games", unit="game")

    for epoch in progress_bar:
        snake_env.reset()
        game_steps = 0
        game_loss = 0.0
        loss_updates = 0

        if (epoch + 1) % cfg.train.update_target_every == 0:
            snake_agent.update_target_model()

        if epoch > 100 and epoch % 1000 == 0: 
            original_epsilon = snake_agent.epsilon
            snake_agent.epsilon = 0.0

            while not snake_env.gameover:
                state = snake_env.get_state()
                action = snake_agent.get_action(state.unsqueeze(0))
                next_state, reward, done = snake_env.step(action)
                snake_env.render()
                game_steps += 1

            snake_agent.epsilon = original_epsilon

        # Train normally
        else:
            while not snake_env.gameover:
                state = snake_env.get_state()
                action = snake_agent.get_action(state.unsqueeze(0))
                next_state, reward, done = snake_env.step(action)
                
                snake_agent.remember(state, action, reward, next_state, done)
                game_steps += 1

                if game_steps % 4 == 0:
                    loss = snake_agent.replay(cfg.train.batch_size)
                    if loss is not None:
                        game_loss += loss
                        loss_updates += 1

            if loss_updates > 0:
                loss_log.append(game_loss / loss_updates)

        # Track Score and Deaths
        score = snake_env.snake.score
        
        # --- MODEL SAVING LOGIC ---
        if score > best_score and epoch > 0:
            best_score = score
            save_dir = to_absolute_path(cfg.paths.model_path)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(snake_agent.model.state_dict(), os.path.join(save_dir, "latest_model.pth"))
            tqdm.write(f"New Best Score: {best_score}! Model saved.")

        death_reason = snake_env.death_reason
        if death_reason in batch_deaths:
            batch_deaths[death_reason] += 1
        
        score_log.append(score)
        length_log.append(game_steps)

        current_lr = snake_agent.optimizer.param_groups[0]['lr']
        progress_bar.set_postfix({
            "score": score,
            "best": best_score if best_score >= 0 else "-",
            "steps": game_steps,
            "eps": f"{snake_agent.epsilon:.4f}",
            "lr": f"{current_lr:.5f}"
        })

        # --- W&B LOGGING BATCH ---
        if (epoch + 1) % cfg.train.logging_batch == 0:
            batch_idx = (epoch + 1) // cfg.train.logging_batch
            mean_score = sum(score_log) / len(score_log)
            mean_loss = sum(loss_log) / len(loss_log) if loss_log else 0
            mean_length = sum(length_log) / len(length_log) if length_log else 0
            
            scheduler.step(mean_loss)

            total_deaths = sum(batch_deaths.values())
            death_rates = {
                "wall": batch_deaths["wall"] / total_deaths if total_deaths else 0.0,
                "body": batch_deaths["body"] / total_deaths if total_deaths else 0.0,
                "timeout": batch_deaths["timeout"] / total_deaths if total_deaths else 0.0,
                "won": batch_deaths["won"] / total_deaths if total_deaths else 0.0,
            }
            death_table = wandb.Table(
                data=[[k, v] for k, v in death_rates.items()],
                columns=["end_state", "rate"],
            )
            death_bar = wandb.plot.bar(
                death_table,
                "end_state",
                "rate",
                title="Train End State Rates (batch)",
            )

            wandb.log(
                {
                    "Train/Average_Score": mean_score,
                    "Train/Average_Loss": mean_loss,
                    "Train/Average_Length": mean_length,
                    "Train/End_State_Rates": death_bar,
                    "Agent/Epsilon": snake_agent.epsilon,
                    "Agent/Learning_Rate": current_lr,
                    "Train/Games": epoch + 1,
                },
                step=batch_idx,
            )

            # Reset batch trackers
            score_log = []
            loss_log = []
            length_log = []
            batch_deaths = {"wall": 0, "body": 0, "timeout": 0, "won": 0}

    wandb.finish()

if __name__ == "__main__":
    train()