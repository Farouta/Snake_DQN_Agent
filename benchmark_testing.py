import os
import torch
import wandb
import hydra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from agent import AgentCNN
from environment import snake_environment


def create_eval_dashboard(history, test_name, logging_batch):
    """
    Generate a 3-panel matplotlib dashboard for evaluation:
      1. Average Score per Batch (blue line)
      2. Death Reason Ratio per Batch (stacked bar chart)
      3. Average Game Length per Batch (green line)
    """
    batches = np.arange(1, len(history["scores"]) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), dpi=100)
    fig.suptitle(f"Evaluation Dashboard — {test_name} ({logging_batch} games/batch)",
                 fontsize=14, fontweight='bold', y=0.98)

    # --- Panel 1: Average Score ---
    ax = axes[0]
    ax.plot(batches, history["scores"], 'o-', color='blue', markersize=3, label="Score Mean")
    ax.set_title("Average Score per Batch", fontweight='bold')
    ax.set_xlabel(f"Batch of {logging_batch} Games")
    ax.set_ylabel("Average Score")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Death Reason Ratio (STACKED BAR CHART) ---
    ax = axes[1]
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

    # --- Panel 3: Average Game Length ---
    ax = axes[2]
    ax.plot(batches, history["lengths"], 'o-', color='green', markersize=3, label="Length Mean (Steps)")
    ax.set_title("Average Game Length per Batch", fontweight='bold')
    ax.set_xlabel(f"Batch of {logging_batch} Games")
    ax.set_ylabel("Average Steps per Game")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def evaluate_agent(agent, env, test_name, num_games=100, logging_batch=50):
    """Plays N games and returns the average score and win rate."""
    deaths = {"wall": 0, "body": 0, "timeout": 0, "won": 0}
    scores = []
    batch_deaths = {"wall": 0, "body": 0, "timeout": 0, "won": 0}
    batch_scores = []
    batch_lengths = []

    # Dashboard history
    history = {
        "scores": [], "lengths": [],
        "death_wall": [], "death_body": [],
        "death_timeout": [], "death_won": [],
    }

    agent.epsilon = 0.0
    agent.model.eval()

    with torch.no_grad():
        for game_idx in range(1, num_games + 1):
            env.reset()
            state = env.get_state()
            done = False
            game_steps = 0

            while not done:
                action = agent.get_action(state.unsqueeze(0))
                state, reward, done = env.step(action)
                game_steps += 1

            reason = env.death_reason

            if reason in deaths:
                deaths[reason] += 1
                batch_deaths[reason] += 1

            scores.append(env.snake.score)
            batch_scores.append(env.snake.score)
            batch_lengths.append(game_steps)

            if logging_batch > 0 and (game_idx % logging_batch == 0 or game_idx == num_games):
                batch_idx = (game_idx - 1) // logging_batch + 1
                batch_total = len(batch_scores)
                batch_rates = {
                    "wall": batch_deaths["wall"] / batch_total if batch_total else 0.0,
                    "body": batch_deaths["body"] / batch_total if batch_total else 0.0,
                    "timeout": batch_deaths["timeout"] / batch_total if batch_total else 0.0,
                    "won": batch_deaths["won"] / batch_total if batch_total else 0.0,
                }
                batch_mean_score = sum(batch_scores) / batch_total if batch_total else 0.0
                batch_mean_length = sum(batch_lengths) / batch_total if batch_total else 0.0
                batch_win_rate = batch_rates["won"] * 100.0

                # Accumulate history
                history["scores"].append(batch_mean_score)
                history["lengths"].append(batch_mean_length)
                history["death_wall"].append(batch_rates["wall"])
                history["death_body"].append(batch_rates["body"])
                history["death_timeout"].append(batch_rates["timeout"])
                history["death_won"].append(batch_rates["won"])

                # Generate dashboard
                fig = create_eval_dashboard(history, test_name, logging_batch)

                wandb.log(
                    {
                        f"{test_name}/mean_score": batch_mean_score,
                        f"{test_name}/mean_length": batch_mean_length,
                        f"{test_name}/win_rate": batch_win_rate,
                        f"{test_name}/Death_Rate/Wall": batch_rates["wall"],
                        f"{test_name}/Death_Rate/Body": batch_rates["body"],
                        f"{test_name}/Death_Rate/Timeout": batch_rates["timeout"],
                        f"{test_name}/Death_Rate/Won": batch_rates["won"],
                        f"{test_name}/Dashboard": wandb.Image(fig),
                        f"{test_name}/games": game_idx,
                    },
                    step=batch_idx,
                )
                plt.close(fig)

                batch_deaths = {"wall": 0, "body": 0, "timeout": 0, "won": 0}
                batch_scores = []
                batch_lengths = []

    mean_score = sum(scores) / len(scores) if scores else 0.0
    total_games = len(scores) if scores else 0
    win_rate = (deaths["won"] / total_games * 100) if total_games else 0.0

    wandb.log({
        f"{test_name}/final_mean_score": mean_score,
        f"{test_name}/final_win_rate": win_rate,
    })

    return mean_score, win_rate

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    run_name = f"Eval_{cfg.game.base_board_size}x{cfg.game.base_board_size}"
    wandb.init(
        project=cfg.benchmark.wandb_project,
        job_type="evaluation",
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_suites = {
        f"Baseline {cfg.game.base_board_size}x{cfg.game.base_board_size}": {
            "width": cfg.game.base_board_size, 
            "height": cfg.game.base_board_size
        },
    }

    if cfg.game.is_general_state:
        test_suites["Expansion 7x7"] = {"width": 7, "height": 7}
        test_suites["Marathon 10x10"] = {"width": 10, "height": 10}
    
    model_path = os.path.join(to_absolute_path(cfg.paths.model_path), "best_model.pth")

    for test_name, config in test_suites.items():
        print(f"\n--- Running: {test_name} ---")
        
        env = snake_environment(config["width"], config["height"])
        agent = AgentCNN(
            memory_length=cfg.agent.memory_length,
            gamma=cfg.agent.gamma,
            epsilon_decay=cfg.agent.epsilon_decay,
            learning_rate=cfg.agent.learning_rate,
            epsilon_min=cfg.agent.epsilon_end,
            in_channels=env.d_model,
            hidden_size=cfg.agent.hidden_size,
        )
        
        try:
            agent.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        except FileNotFoundError:
            print(f"ERROR: Could not find model at {model_path}")
            return

        mean_score, win_rate = evaluate_agent(
            agent,
            env,
            test_name,
            cfg.benchmark.num_games,
            cfg.benchmark.logging_batch,
        )
        
        print(f"Mean Score: {mean_score:.2f}")
        print(f"Win Rate: {win_rate:.1f}%")
        
    wandb.finish()
    print("\nBenchmark Complete! Check your W&B Dashboard.")

if __name__ == "__main__":
    main()