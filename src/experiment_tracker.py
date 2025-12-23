"""
Centralized experiment tracking system for GRPO training.

This module provides a unified interface for tracking experiments, configurations,
metrics, and results across different backends (W&B, local SQLite, etc.).
"""

import json
import sqlite3
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import platform


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    # Model configuration
    base_model: str = "google/gemma-3-1b-it"
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16

    # Training configuration
    num_steps: int = 500
    learning_rate: float = 5e-5
    batch_size: int = 64
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_fraction: float = 0.1

    # GRPO configuration
    num_generations: int = 4
    beta: float = 0.1
    epsilon: float = 0.2
    temperature: float = 0.7

    # Generation configuration
    max_prompt_length: int = 1024
    max_generation_steps: int = 512
    top_k: int = 50
    top_p: float = 0.9

    # Dataset configuration
    dataset_name: str = "openrubrics"
    train_size: int = 8000
    eval_size: int = 1000
    seed: int = 42

    # Reward configuration
    format_weight: float = 1.0
    accuracy_weight: float = 2.0
    rubric_weight: float = 0.5

    # Infrastructure
    num_tpu_cores: Optional[int] = None
    checkpoint_interval: int = 50

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ExperimentConfig":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class ExperimentMetadata:
    """Metadata for an experiment."""

    experiment_id: str
    git_commit: str
    git_branch: str
    git_dirty: bool
    timestamp: str
    user: str
    hostname: str
    python_version: str
    device_type: str
    status: str = "running"
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentMetadata":
        """Create from dictionary."""
        return cls(**data)


class ExperimentBackend(ABC):
    """Abstract backend for experiment tracking."""

    @abstractmethod
    def log_experiment(self, metadata: ExperimentMetadata, config: ExperimentConfig):
        """Log a new experiment."""
        pass

    @abstractmethod
    def log_metrics(self, experiment_id: str, metrics: Dict[str, float], step: int):
        """Log metrics for an experiment."""
        pass

    @abstractmethod
    def log_evaluation(self, experiment_id: str, benchmark: str, results: Dict[str, Any]):
        """Log evaluation results."""
        pass

    @abstractmethod
    def log_checkpoint(self, experiment_id: str, step: int, path: str, size_bytes: int):
        """Log a checkpoint."""
        pass

    @abstractmethod
    def update_status(self, experiment_id: str, status: str):
        """Update experiment status."""
        pass

    @abstractmethod
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment details."""
        pass


class LocalBackend(ExperimentBackend):
    """Local SQLite backend for experiment tracking."""

    def __init__(self, db_path: Union[str, Path] = "experiments.db"):
        """Initialize local backend.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                git_commit TEXT NOT NULL,
                git_branch TEXT NOT NULL,
                git_dirty BOOLEAN NOT NULL,
                timestamp DATETIME NOT NULL,
                user TEXT,
                hostname TEXT,
                python_version TEXT,
                device_type TEXT,
                config_json TEXT NOT NULL,
                status TEXT DEFAULT 'running',
                notes TEXT
            )
        """)

        # Create training metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)

        # Create evaluation results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                benchmark_name TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)

        # Create checkpoints table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                path TEXT NOT NULL,
                size_bytes INTEGER,
                timestamp DATETIME NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiments_timestamp ON experiments(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_metrics_exp ON training_metrics(experiment_id, step)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_exp_benchmark ON evaluation_results(experiment_id, benchmark_name)")

        conn.commit()
        conn.close()

    def log_experiment(self, metadata: ExperimentMetadata, config: ExperimentConfig):
        """Log a new experiment."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO experiments (
                experiment_id, git_commit, git_branch, git_dirty,
                timestamp, user, hostname, python_version, device_type,
                config_json, status, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata.experiment_id,
            metadata.git_commit,
            metadata.git_branch,
            metadata.git_dirty,
            metadata.timestamp,
            metadata.user,
            metadata.hostname,
            metadata.python_version,
            metadata.device_type,
            config.to_json(),
            metadata.status,
            metadata.notes
        ))

        conn.commit()
        conn.close()

    def log_metrics(self, experiment_id: str, metrics: Dict[str, float], step: int):
        """Log metrics for an experiment."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = datetime.utcnow().isoformat()

        for metric_name, metric_value in metrics.items():
            cursor.execute("""
                INSERT INTO training_metrics (
                    experiment_id, step, metric_name, metric_value, timestamp
                ) VALUES (?, ?, ?, ?, ?)
            """, (experiment_id, step, metric_name, float(metric_value), timestamp))

        conn.commit()
        conn.close()

    def log_evaluation(self, experiment_id: str, benchmark: str, results: Dict[str, Any]):
        """Log evaluation results."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO evaluation_results (
                experiment_id, benchmark_name, metrics_json, timestamp
            ) VALUES (?, ?, ?, ?)
        """, (
            experiment_id,
            benchmark,
            json.dumps(results),
            datetime.utcnow().isoformat()
        ))

        conn.commit()
        conn.close()

    def log_checkpoint(self, experiment_id: str, step: int, path: str, size_bytes: int):
        """Log a checkpoint."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO checkpoints (
                experiment_id, step, path, size_bytes, timestamp
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            experiment_id,
            step,
            path,
            size_bytes,
            datetime.utcnow().isoformat()
        ))

        conn.commit()
        conn.close()

    def update_status(self, experiment_id: str, status: str):
        """Update experiment status."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE experiments
            SET status = ?
            WHERE experiment_id = ?
        """, (status, experiment_id))

        conn.commit()
        conn.close()

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment details."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM experiments
            WHERE experiment_id = ?
        """, (experiment_id,))

        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        return dict(row)

    def get_all_experiments(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all experiments, ordered by timestamp (newest first)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM experiments ORDER BY timestamp DESC"
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_metrics(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get all metrics for an experiment."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM training_metrics
            WHERE experiment_id = ?
            ORDER BY step
        """, (experiment_id,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_evaluation_results(self, experiment_id: str, benchmark: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get evaluation results for an experiment."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if benchmark:
            cursor.execute("""
                SELECT * FROM evaluation_results
                WHERE experiment_id = ? AND benchmark_name = ?
                ORDER BY timestamp DESC
            """, (experiment_id, benchmark))
        else:
            cursor.execute("""
                SELECT * FROM evaluation_results
                WHERE experiment_id = ?
                ORDER BY timestamp DESC
            """, (experiment_id,))

        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            result = dict(row)
            result["metrics_json"] = json.loads(result["metrics_json"])
            results.append(result)

        return results


class WandBBackend(ExperimentBackend):
    """Weights & Biases backend for experiment tracking."""

    def __init__(self, project_name: str = "tunix-grpo", entity: Optional[str] = None):
        """Initialize W&B backend.

        Args:
            project_name: W&B project name
            entity: W&B entity (username or team)
        """
        self.project_name = project_name
        self.entity = entity
        self.run = None

    def log_experiment(self, metadata: ExperimentMetadata, config: ExperimentConfig):
        """Log a new experiment."""
        try:
            import wandb

            self.run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=metadata.experiment_id,
                config=config.to_dict(),
                tags=[
                    metadata.git_branch,
                    metadata.device_type,
                    f"commit:{metadata.git_commit[:7]}"
                ],
                notes=metadata.notes
            )

            # Log metadata
            wandb.config.update({
                "git_commit": metadata.git_commit,
                "git_branch": metadata.git_branch,
                "git_dirty": metadata.git_dirty,
                "hostname": metadata.hostname,
                "python_version": metadata.python_version
            })

        except ImportError:
            print("Warning: wandb not installed, skipping W&B logging")

    def log_metrics(self, experiment_id: str, metrics: Dict[str, float], step: int):
        """Log metrics for an experiment."""
        try:
            import wandb
            if self.run is not None:
                wandb.log(metrics, step=step)
        except ImportError:
            pass

    def log_evaluation(self, experiment_id: str, benchmark: str, results: Dict[str, Any]):
        """Log evaluation results."""
        try:
            import wandb
            if self.run is not None:
                # Log main metrics with benchmark prefix
                metrics = results.get("metrics", {})
                wandb.log({f"eval/{benchmark}/{k}": v for k, v in metrics.items()})

                # Create summary table
                if "per_sample_results" in results:
                    table = wandb.Table(
                        columns=["question_id", "is_correct", "predicted_answer", "gold_answer"],
                        data=[
                            [r["question_id"], r["is_correct"], r["predicted_answer"], r["gold_answer"]]
                            for r in results["per_sample_results"][:100]  # Limit to 100 samples
                        ]
                    )
                    wandb.log({f"eval/{benchmark}/samples": table})
        except ImportError:
            pass

    def log_checkpoint(self, experiment_id: str, step: int, path: str, size_bytes: int):
        """Log a checkpoint."""
        try:
            import wandb
            if self.run is not None:
                wandb.log({
                    "checkpoint/step": step,
                    "checkpoint/size_mb": size_bytes / (1024 * 1024),
                    "checkpoint/path": path
                }, step=step)
        except ImportError:
            pass

    def update_status(self, experiment_id: str, status: str):
        """Update experiment status."""
        try:
            import wandb
            if self.run is not None:
                wandb.config.update({"status": status})
                if status in ["completed", "failed"]:
                    wandb.finish()
                    self.run = None
        except ImportError:
            pass

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment details."""
        # Not implemented for W&B backend (use W&B web interface)
        return None


class ExperimentTracker:
    """Main experiment tracking interface."""

    def __init__(
        self,
        backends: Optional[List[str]] = None,
        db_path: Union[str, Path] = "experiments.db",
        wandb_project: str = "tunix-grpo",
        wandb_entity: Optional[str] = None
    ):
        """Initialize experiment tracker.

        Args:
            backends: List of backend names to use (e.g., ["local", "wandb"])
            db_path: Path to SQLite database for local backend
            wandb_project: W&B project name
            wandb_entity: W&B entity (username or team)
        """
        if backends is None:
            backends = ["local"]

        self.backends: List[ExperimentBackend] = []

        if "local" in backends:
            self.backends.append(LocalBackend(db_path))

        if "wandb" in backends:
            self.backends.append(WandBBackend(wandb_project, wandb_entity))

        self.current_experiment_id: Optional[str] = None

    def start_experiment(
        self,
        config: Union[ExperimentConfig, Dict[str, Any]],
        notes: Optional[str] = None,
        experiment_name: Optional[str] = None
    ) -> str:
        """Start a new experiment.

        Args:
            config: Experiment configuration
            notes: Optional notes about the experiment
            experiment_name: Optional custom experiment name

        Returns:
            experiment_id: Unique experiment ID
        """
        # Convert dict to ExperimentConfig if needed
        if isinstance(config, dict):
            config = ExperimentConfig.from_dict(config)

        # Get git information
        git_commit = self._get_git_commit()
        git_branch = self._get_git_branch()
        git_dirty = self._is_git_dirty()

        # Generate experiment ID
        timestamp = datetime.utcnow()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        git_hash_short = git_commit[:6] if git_commit else "nogit"

        if experiment_name:
            experiment_id = f"exp_{timestamp_str}_{experiment_name}_{git_hash_short}"
        else:
            experiment_id = f"exp_{timestamp_str}_{git_hash_short}"

        # Create metadata
        metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            git_commit=git_commit or "unknown",
            git_branch=git_branch or "unknown",
            git_dirty=git_dirty,
            timestamp=timestamp.isoformat(),
            user=self._get_user(),
            hostname=platform.node(),
            python_version=platform.python_version(),
            device_type=self._get_device_type(),
            status="running",
            notes=notes
        )

        # Log to all backends
        for backend in self.backends:
            backend.log_experiment(metadata, config)

        self.current_experiment_id = experiment_id
        print(f"Started experiment: {experiment_id}")

        return experiment_id

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics for current experiment.

        Args:
            metrics: Dictionary of metric name -> value
            step: Training step (optional)
        """
        if self.current_experiment_id is None:
            raise ValueError("No active experiment. Call start_experiment() first.")

        if step is None:
            step = 0

        for backend in self.backends:
            backend.log_metrics(self.current_experiment_id, metrics, step)

    def log_evaluation(self, benchmark: str, results: Dict[str, Any]):
        """Log evaluation results for current experiment.

        Args:
            benchmark: Benchmark name (e.g., "gsm8k")
            results: Evaluation results dictionary
        """
        if self.current_experiment_id is None:
            raise ValueError("No active experiment. Call start_experiment() first.")

        for backend in self.backends:
            backend.log_evaluation(self.current_experiment_id, benchmark, results)

    def log_checkpoint(self, path: str, step: int, size_bytes: Optional[int] = None):
        """Log a checkpoint for current experiment.

        Args:
            path: Path to checkpoint
            step: Training step
            size_bytes: Size of checkpoint in bytes (optional)
        """
        if self.current_experiment_id is None:
            raise ValueError("No active experiment. Call start_experiment() first.")

        if size_bytes is None:
            # Try to get size from path
            try:
                size_bytes = Path(path).stat().st_size
            except:
                size_bytes = 0

        for backend in self.backends:
            backend.log_checkpoint(self.current_experiment_id, step, path, size_bytes)

    def finish_experiment(self, status: str = "completed"):
        """Finish the current experiment.

        Args:
            status: Final status ("completed" or "failed")
        """
        if self.current_experiment_id is None:
            raise ValueError("No active experiment. Call start_experiment() first.")

        for backend in self.backends:
            backend.update_status(self.current_experiment_id, status)

        print(f"Finished experiment: {self.current_experiment_id} (status: {status})")
        self.current_experiment_id = None

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment details.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment details dictionary, or None if not found
        """
        # Try local backend first
        for backend in self.backends:
            if isinstance(backend, LocalBackend):
                return backend.get_experiment(experiment_id)
        return None

    def get_all_experiments(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all experiments.

        Args:
            limit: Maximum number of experiments to return

        Returns:
            List of experiment dictionaries
        """
        for backend in self.backends:
            if isinstance(backend, LocalBackend):
                return backend.get_all_experiments(limit)
        return []

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except:
            return None

    def _get_git_branch(self) -> Optional[str]:
        """Get current git branch."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except:
            return None

    def _is_git_dirty(self) -> bool:
        """Check if git working directory is dirty."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True
            )
            return bool(result.stdout.strip())
        except:
            return False

    def _get_user(self) -> str:
        """Get current user."""
        try:
            import getpass
            return getpass.getuser()
        except:
            return "unknown"

    def _get_device_type(self) -> str:
        """Get device type."""
        try:
            import jax
            if jax.devices()[0].platform == "tpu":
                return f"TPU-{len(jax.devices())}"
        except:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                return f"CUDA-{torch.cuda.device_count()}"
        except:
            pass

        return "CPU"
