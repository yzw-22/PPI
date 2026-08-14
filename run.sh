#!/bin/bash
python -m src.train_shs27k --dataset SHS27k --split bfs --sampler-mode learned --device cuda --epochs 10 --seed 42 --output /tmp/ppi_shs27k_bfs_learned.json
python -m src.train_shs27k --dataset SHS27k --split bfs --sampler-mode target_only --device cuda --epochs 10 --seed 42 --output /tmp/ppi_shs27k_bfs_target_only.json
python -m src.train_shs27k --dataset SHS27k --split bfs --sampler-mode target_proxy --device cuda --epochs 10 --seed 42 --output /tmp/ppi_shs27k_bfs_target_proxy.json
python -m src.train_shs27k --dataset SHS27k --split bfs --sampler-mode random_1hop10 --device cuda --epochs 10 --seed 42 --output /tmp/ppi_shs27k_bfs_random_1hop10.json
python -m src.train_shs27k --dataset SHS27k --split bfs --sampler-mode random_iterative10 --device cuda --epochs 10 --seed 42 --output /tmp/ppi_shs27k_bfs_random_iterative10.json