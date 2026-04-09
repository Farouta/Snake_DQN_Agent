import os
import torch
import hydra
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from torch.optim.lr_scheduler import ReduceLROnPlateau

from agent import AgentCNN
from environment import snake_environment


def create_training_dashboard(history, batch_idx, logging_batch):
    """
    Generate a 4-panel matplotlib dashboard matching the notebook reference:
      1. Average Score per Batch (blue line)
      2. Average Loss per Batch (red line)
      3. Death Reason Ratio per Batch (stacked bar chart)
      4. Average Game Length per Batch (green line)
    """
    batches = np.arange(1, len(history["scores"]) + 1)

    fig, axes = plt.subplots(4, 1, figsize=(10, 16), dpi=100)
    fig.suptitle(f"Training Dashboard — Batch {batch_idx} ({logging_batch} games/batch)",
                 fontsize=14, fontweight='bold', y=0.98)

    # --- Panel 1: Average Score ---
    ax = axes[0]
    ax.plot(batches, history["scores"], 'o-', color='blue', markersize=3, label="Score Mean")
    ax.set_title("Average Score per Batch", fontweight='bold')
    ax.set_xlabel(f"Batch of {logging_batch} Games")
    ax.set_ylabel("Average Score")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Average Loss ---
    ax = axes[1]
    ax.plot(batches, history["losses"], 'o-', color='red', markersize=3, label="Loss Mean")
    ax.set_title("Average Loss per Batch", fontweight='bold')
    ax.set_xlabel(f"Batch of {logging_batch} Games")
    ax.set_ylabel("Average Loss")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Death Reason Ratio (STACKED BAR CHART) ---
    ax = axes[2]
    wall = np.array(history["death_wall"])
    body = np.array(history["death_body"])
    timeout = np.array(history["death_timeout"])
    won = np.array(history["death_won"])

    ax.bar(batches, won, color='#2ca02c', label='Solved')
    ax.bar(batches, timeout, bottom=won, color='#9467bd', label='Ran out of Steps')
    ax.bar(batches, wall, bottom=won + timeout, color='#bcbd22', label='Ran into Wall')
    ax.bar(batches, body, bottom=won + timeout + wall, color='#ff7f0e', label='Ran into Body')

    ax.set_title("Death Reason Ratio per Batch", fontweight='bold')
    ax.set_xlabel(f"Batch of {logging_batch} Games")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # --- Panel 4: Average Game Length ---
    ax = axes[3]
    ax.plot(batches, history["lengths"], 'o-', color='green', markersize=3, label="Length Mean (Steps)")
    ax.set_title("Average Game Length per Batch", fontweight='bold')
    ax.set_xlabel(f"Batch of {logging_batch} Games")
    ax.set_ylabel("Average Steps per Game")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):

    # --- W&B Init with Hydra config ---
    run_name = f"{cfg.agent.name}_{cfg.game.base_board_size}x{cfg.game.base_board_size}_{cfg.train.num_episodes}ep"
    wandb.init(
        project=cfg.benchmark.wandb_project,
        job_type="training",
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

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

    # --- Batch Tracking Variables ---
    score_log = []
    loss_log = []
    length_log = []
    batch_deaths = {"wall": 0, "body": 0, "timeout": 0, "won": 0}
    best_avg_score = -1

    # --- History for dashboard (accumulates across all batches) ---
    history = {
        "scores": [],
        "losses": [],
        "lengths": [],
        "death_wall": [],
        "death_body": [],
        "death_timeout": [],
        "death_won": [],
    }

    progress_bar = tqdm(range(cfg.train.num_episodes), desc="Training Games", unit="game")

    for epoch in progress_bar:
        snake_env.reset()
        game_steps = 0
        game_loss = 0.0
        loss_updates = 0

        if (epoch + 1) % cfg.train.update_target_every == 0:
            snake_agent.update_target_model()

        # --- Normal training loop ---
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

        # --- Track Score and Deaths ---
        score = snake_env.snake.score
        death_reason = snake_env.death_reason
        if death_reason in batch_deaths:
            batch_deaths[death_reason] += 1
        
        score_log.append(score)
        length_log.append(game_steps)

        current_lr = snake_agent.optimizer.param_groups[0]['lr']
        progress_bar.set_postfix({
            "score": score,
            "best_avg": f"{best_avg_score:.1f}" if best_avg_score >= 0 else "-",
            "steps": game_steps,
            "eps": f"{snake_agent.epsilon:.4f}",
            "lr": f"{current_lr:.6f}"
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

            # --- Accumulate history for dashboard ---
            history["scores"].append(mean_score)
            history["losses"].append(mean_loss)
            history["lengths"].append(mean_length)
            history["death_wall"].append(death_rates["wall"])
            history["death_body"].append(death_rates["body"])
            history["death_timeout"].append(death_rates["timeout"])
            history["death_won"].append(death_rates["won"])

            # --- Generate matplotlib dashboard ---
            fig = create_training_dashboard(history, batch_idx, cfg.train.logging_batch)

            # --- Log everything to W&B ---
            wandb.log(
                {
                    # Scalar metrics (line charts in W&B)
                    "Train/Average_Score": mean_score,
                    "Train/Average_Loss": mean_loss,
                    "Train/Average_Length": mean_length,
                    # Individual death rates (can be stacked area in W&B)
                    "Train/Death_Rate/Wall": death_rates["wall"],
                    "Train/Death_Rate/Body": death_rates["body"],
                    "Train/Death_Rate/Timeout": death_rates["timeout"],
                    "Train/Death_Rate/Won": death_rates["won"],
                    # Agent state
                    "Agent/Epsilon": snake_agent.epsilon,
                    "Agent/Learning_Rate": current_lr,
                    # Full dashboard image
                    "Train/Dashboard": wandb.Image(fig),
                },
                step=batch_idx,
            )
            plt.close(fig)

            # --- MODEL SAVING: batch-average based ---
            if mean_score > best_avg_score:
                best_avg_score = mean_score
                save_dir = to_absolute_path(cfg.paths.model_path)
                os.makedirs(save_dir, exist_ok=True)
                torch.save(snake_agent.model.state_dict(), os.path.join(save_dir, "best_model.pth"))
                tqdm.write(f"[Batch {batch_idx}] New best avg score: {best_avg_score:.2f} — model saved.")

            # Always save latest checkpoint
            save_dir = to_absolute_path(cfg.paths.model_path)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(snake_agent.model.state_dict(), os.path.join(save_dir, "latest_model.pth"))

            # --- Reset batch trackers ---
            score_log = []
            loss_log = []
            length_log = []
            batch_deaths = {"wall": 0, "body": 0, "timeout": 0, "won": 0}

    # --- Save final dashboard to disk ---
    if history["scores"]:
        final_fig = create_training_dashboard(history, len(history["scores"]), cfg.train.logging_batch)
        dashboard_path = to_absolute_path("latest_model_training_dashboard.png")
        final_fig.savefig(dashboard_path, bbox_inches='tight')
        plt.close(final_fig)
        tqdm.write(f"Final dashboard saved to {dashboard_path}")

    wandb.finish()

if __name__ == "__main__":
    train()