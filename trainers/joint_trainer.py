"""Joint Trainer: 联合攻防训练 (clean → attack → defense)"""
import torch
import numpy as np
from .clean_trainer import CleanTrainer
from .attack_trainer import AttackTrainer
from .defense_trainer import DefenseTrainer


class JointTrainer:
    """联合训练器: 顺序执行 clean → attack → defense 训练流水线"""

    def __init__(self, model, optimizer, config, hack_image,
                 device='cuda', save_dir='./checkpoints'):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.hack_image = hack_image
        self.device = device
        self.save_dir = save_dir

        self.clean_trainer = CleanTrainer(
            model, optimizer, config, device, save_dir
        )
        self.results = {
            'clean': None,
            'attack': None,
            'defense': None,
        }

    def train_clean(self, train_loader, epochs=None):
        print("\n" + "=" * 60)
        print("[Joint] Phase 1: Clean Training")
        print("=" * 60)
        epochs = epochs or self.config.EPOCHS
        losses = []
        for ep in range(epochs):
            loss = self.clean_trainer.train_epoch(train_loader, ep + 1)
            losses.append(loss)
        self.clean_trainer.save_checkpoint('clean_model')
        self.results['clean'] = {'losses': losses, 'epochs': epochs}
        return self.clean_trainer

    def train_attack(self, train_loader, backdoor, epochs=None):
        print("\n" + "=" * 60)
        print("[Joint] Phase 2: Attack Training")
        print("=" * 60)
        epochs = epochs or self.config.EPOCHS

        self.attack_trainer = AttackTrainer(
            self.model, self.optimizer, self.config, backdoor,
            self.device, self.save_dir
        )
        losses = []
        for ep in range(epochs):
            loss = self.attack_trainer.train_epoch(train_loader, ep + 1)
            losses.append(loss)
        self.attack_trainer.save_checkpoint('attack_model')
        self.results['attack'] = {'losses': losses, 'epochs': epochs}
        return self.attack_trainer

    def train_defense(self, train_loader, epochs=None):
        print("\n" + "=" * 60)
        print("[Joint] Phase 3: Defense Training")
        print("=" * 60)
        epochs = epochs or self.config.DEFENSE_EPOCHS

        self.defense_trainer = DefenseTrainer(
            self.model, self.optimizer, self.config, self.hack_image,
            self.device, self.save_dir
        )
        losses = []
        for ep in range(epochs):
            loss = self.defense_trainer.train_epoch(train_loader, ep + 1)
            losses.append(loss)
        self.defense_trainer.save_checkpoint('defense_model')
        self.results['defense'] = {'losses': losses, 'epochs': epochs}
        return self.defense_trainer

    def run_full_pipeline(self, clean_loader, attack_loader, defense_loader,
                          backdoor, clean_epochs=None, attack_epochs=None,
                          defense_epochs=None):
        """完整三段式训练流水线"""
        self.train_clean(clean_loader, clean_epochs)
        self.train_attack(attack_loader, backdoor, attack_epochs)
        self.train_defense(defense_loader, defense_epochs)

        print("\n" + "=" * 60)
        print("[Joint] Full Pipeline Complete!")
        print("=" * 60)
        return self.results
