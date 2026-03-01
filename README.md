# Heterogeneous Training Simulation (CPU + MPS)

## Overview
This project simulates heterogeneous AI training inspired by AMD Ryzen AI architecture.

We compare:

1. CPU-only training
2. GPU (MPS)-only training
3. Simulated heterogeneous training (CPU preprocessing + GPU compute)

## Dataset
MNIST Handwritten Digit Dataset

## Model
Simple CNN (Convolutional Neural Network)

## Goal
Measure:
- CPU preprocessing time
- Device transfer time
- GPU compute time
- Total training time

## How to Run

CPU:
python baseline_cpu.py

MPS:
python baseline_mps.py

Heterogeneous:
python heterogeneous.py