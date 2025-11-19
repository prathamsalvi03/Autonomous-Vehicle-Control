# Autonomous Vehicle Control

Repository: prathamsalvi03/Autonomous‑Vehicle‑Control

## Overview

This repository contains code and notebooks for controlling an
autonomous vehicle using Python and reinforcement learning. The primary
goals are:\
- Implement control logic (PID, etc.) for vehicle motion.\
- Apply reinforcement learning (RL) to train a model for autonomous
driving behaviour.\
- Demonstrate the system working (see the included video).\
- Provide a foundation for further research and experimentation in
autonomous vehicle control.

## Contents

-   `au_python_pid.py` --- A Python script implementing a PID‑based
    controller for a vehicle.\
-   `autonomous_vehicle_python.py` --- A higher‑level Python script
    coordinating sensor input, control logic, and actuation for
    autonomous driving.\
-   `AV_rl_training (1).ipynb` --- A Jupyter Notebook showing
    reinforcement‑learning training for the autonomous vehicle.\
-   `Autonomous_Vehicle (1).mp4` --- A demo video of the autonomous
    system in action.

## Demo Video

You can view the video directly here:\
[Autonomous Vehicle Demo](Autonomous_Vehicle%20(1).mp4)

## Getting Started

### Prerequisites

-   Python 3.x\

-   Recommended libraries:

    ``` bash
    pip install numpy matplotlib gym torch
    ```

### Installation

``` bash
git clone https://github.com/prathamsalvi03/Autonomous‑Vehicle‑Control.git
cd Autonomous‑Vehicle‑Control
```

### Usage

-   To run the PID controller:

    ``` bash
    python au_python_pid.py
    ```

-   To run the autonomous vehicle script:

    ``` bash
    python autonomous_vehicle_python.py
    ```

-   To run the RL training notebook:

    ``` bash
    jupyter notebook "AV_rl_training (1).ipynb"
    ```

##Project Structure

    /
    ├── au_python_pid.py
    ├── autonomous_vehicle_python.py
    ├── AV_rl_training (1).ipynb
    └── Autonomous_Vehicle (1).mp4

## Contributing

Contributions are welcome! Feel free to open issues or submit pull
requests.

## License

Add your preferred license here.
