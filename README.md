# FightingICE Using CNN

This repository contains a **CNN-based approach** for building an AI agent in the **FightingICE** game environment.  
The main idea is simple: the agent observes the game through **visual frames**, uses a **Convolutional Neural Network (CNN)** to extract meaningful features, and then uses the CNN output to **predict/select actions** during gameplay.

---

## Project Overview

FightingICE is a real-time fighting game platform often used for AI research and coursework.  
In this project, I focus on a **computer-vision pipeline**:

1. **Capture game frames** (visual state of the environment)
2. **Preprocess frames** (resize / normalize / frame formatting)
3. **CNN feature extraction** (learn spatial patterns: player positions, motion cues, distance, attacks)
4. **Action prediction** based on CNN output  
5. **Run inference in real time** while the game is running

This repo is designed to show how CNNs can be applied to game AI using raw visual inputs.

---

## Key Features

- **Frame-based perception**: the agent uses what it “sees” in the game instead of relying only on hand-crafted state variables.
- **CNN model pipeline**: includes code for loading frames, preprocessing, running the CNN, and outputting action decisions.
- **Modular structure**: environment interface (FightingICE), vision preprocessing, model inference are organized in separate components.
- **Reproducible setup**: the repo includes the FightingICE engine/codebase folders and the starter agent code.

---

## Tech Stack

- **Python**: CNN model, preprocessing pipeline, inference/action selection
- **Java**: FightingICE engine
- **Computer Vision + Deep Learning**: CNN feature extraction from frames

---

## Repository Structure (High Level)

- `cnn_fighting_agent_starter/`  
  Main folder for the CNN approach: preprocessing, model code, inference logic, agent behavior.
- `FightingICE-master/` / engine folders (may vary)  
  FightingICE engine + game environment code.
- `pyftg_src/`  
  Python interface utilities for communicating with FightingICE (bridge between Python agent and Java game).

---

## How It Works (Pipeline)

### 1) Frame Capture
The environment provides a visual representation of the game state (frames).

### 2) Preprocessing
Typical preprocessing steps include:
- resizing frames to a consistent input size
- normalizing pixel values
- selecting channels / converting to grayscale (if used)
- reshaping into the format expected by the CNN

### 3) CNN Inference
The CNN processes the frame and outputs either:
- class probabilities for actions (classification-style), or
- a score for each possible action

### 4) Action Selection
The agent maps the CNN output to an in-game action (e.g., move, attack, defend), and sends that action back to FightingICE.

---

## How to Run (General Steps)

> Exact commands can vary depending on your local setup, but the workflow is:

1. **Set up the FightingICE engine**
   - Ensure Java is installed and the engine runs correctly

2. **Set up the Python environment**
   - Install required dependencies for the CNN and interface utilities

3. **Run the agent**
   - Launch FightingICE
   - Start the Python agent script inside `cnn_fighting_agent_starter/`
   - The agent will begin receiving frames and selecting actions using the CNN

---

## Results / Notes

- This project demonstrates a **CNN-only approach** for game decision-making using visual input.
- Performance depends on model quality, preprocessing, and the action mapping strategy.
- The main contribution is building a working pipeline from **frames → CNN → actions** in a real-time environment.

---

## Future Improvements

- Improve preprocessing (frame stacking, motion cues, better cropping)
- Improve the CNN architecture (deeper CNN / residual blocks)
- Add better action mapping and stability (smoother decisions, fewer random actions)
- Add evaluation metrics and comparison experiments

---

## Author

**Mohammed Mubashir Uddin Faraz**  
GitHub: https://github.com/machackgo
