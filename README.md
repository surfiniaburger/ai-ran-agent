# AI RAN Agent: Simulated Radio Access Network Environment

This project implements a simulated Radio Access Network (RAN) environment and an AI agent trained to optimize power consumption while maintaining user connectivity. It leverages Reinforcement Learning (RL) with `stable-baselines3` and `Gymnasium` for the simulation environment.

## Project Overview

The core idea is to create a "digital twin" of a radio network, allowing an AI agent to learn optimal strategies for managing base stations (e.g., turning them on/off) in a resource-constrained area.

-   **`ran_environment.py`**: Defines the `RANEnvironment` class, which is a custom `Gymnasium` environment. It simulates base stations, user equipment, signal propagation, and calculates power consumption and user connectivity.
-   **`train.py`**: Contains the script to train the AI agent using the Proximal Policy Optimization (PPO) algorithm from `stable-baselines3`.
-   **`test.py`**: Contains the script to evaluate the performance of the trained AI agent and visualize the results.
-   **`requirements.txt`**: Lists all the necessary Python dependencies for the project.

## Setup and Installation

To set up and run this project locally, follow these steps:

### Prerequisites

-   **Python 3.8+**: Ensure you have a compatible Python version installed.
-   **`uv`**: A fast Python package installer and resolver. If you don't have `uv` installed, you can install it via `pip`:
    ```bash
    pip install uv
    ```

### Install Project Dependencies

1.  Navigate to the `ai_ran_agent` directory in your terminal:
    ```bash
    cd /Users/surfiniaburger/Desktop/fan/ai_ran_agent
    ```
2.  Install all required Python packages using `uv`:
    ```bash
    uv pip install -r requirements.txt
    ```
    This command will create a virtual environment (`.venv/`) within your project directory and install all dependencies there.

## Running the Project

### 1. Train the AI Agent

The `train.py` script will train the PPO model. This process involves running many simulation episodes. The trained model will be saved as `ran_agent_ppo.zip` in the project directory.

To run the training:

```bash
/Users/surfiniaburger/Desktop/fan/ai_ran_agent/.venv/bin/python train.py
```

You will see output indicating the training progress.

### 2. Test the AI Agent

After training, you can evaluate the agent's performance using the `test.py` script. This script loads the trained model, runs it in the environment for a few episodes, and generates plots visualizing power consumption, connected users, and dropped users.

To run the testing:

```bash
/Users/surfiniaburger/Desktop/fan/ai_ran_agent/.venv/bin/python test.py
```

Upon completion, a file named `simulation_results.png` will be generated in the `ai_ran_agent` directory, containing the performance plots.

## Key Modifications and Troubleshooting

During the development process, the following key changes and considerations were made:

### Episode Termination in `RANEnvironment`

Initially, the `RANEnvironment`'s `step` method did not have a mechanism to terminate episodes, leading to infinite loops during training and testing.

-   **Modification**: A `max_steps` parameter was added to the `RANEnvironment`'s `__init__` method.
-   **Implementation**: The `reset` method now initializes `self.current_step = 0`, and the `step` method increments `self.current_step` and sets `terminated = True` when `self.current_step >= self.max_steps`. This ensures that each simulation episode has a defined length.

### Environment Instantiation in `train.py` and `test.py`

To utilize the new episode termination logic, the `RANEnvironment` instantiation in both training and testing scripts was updated.

-   **Modification**: `env = RANEnvironment()` was changed to `env = RANEnvironment(max_steps=100)` in both `train.py` and `test.py`. This sets a default episode length of 100 steps.

### Python Environment Management

A common challenge is ensuring that the correct Python executable (and thus, the correct virtual environment with installed dependencies) is used when running scripts.

-   **Issue**: Running `python train.py` or `python3 train.py` might fail with `ModuleNotFoundError` if your system's default `python` or `python3` command does not point to the virtual environment created by `uv`.
-   **Solution**: The recommended way to run the scripts is by explicitly calling the Python executable located within the `.venv/bin/` directory of your project:
    ```bash
    /Users/surfiniaburger/Desktop/fan/ai_ran_agent/.venv/bin/python your_script_name.py
    ```
    This guarantees that your scripts run within the isolated environment where all project dependencies are installed.

By following these steps and understanding the underlying changes, you should be able to easily reproduce the results and further experiment with the AI RAN agent.

rest