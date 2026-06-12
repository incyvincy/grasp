#!/usr/bin/env python

"""
Script to generate a dataset for MetaWorld 'shelf-place-v3' task using the expert policy.
Ensures only successful episodes are saved to the LeRobot dataset format.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import torch

# LeRobot imports
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.metaworld import MetaworldEnv

# --- Configuration ---
REPO_ID = "lerobot/shelf-place-v3"
TASK_NAME = "shelf-place-v3"
NUM_EPISODES = 200
FPS = 50
ROOT_DIR = Path("data")
MAX_EPISODE_STEPS = 500

OBSERVATION_WIDTH = 480
OBSERVATION_HEIGHT = 480

os.environ["SVT_LOG"] = "0"
os.environ["FFREPORT"] = "level=quiet"


class MetaworldEnvWithRawObs(MetaworldEnv):
    """Extended MetaworldEnv that tracks raw observations for expert policy."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._raw_obs = None
    
    def reset(self, seed=None, **kwargs):
        """Reset and store raw observation."""
        self._ensure_env()
        # Get raw observation directly from inner env
        self._raw_obs, _ = self._env.reset(seed=seed)
        observation = self._format_raw_obs(self._raw_obs)
        return observation, {"is_success": False}
    
    def step(self, action):
        """Step and store raw observation."""
        self._ensure_env()
        if action.ndim != 1:
            raise ValueError(f"Expected action to be 1-D, got shape {action.shape}")
        
        self._raw_obs, reward, done, truncated, info = self._env.step(action)
        
        is_success = bool(info.get("success", 0))
        terminated = done or is_success
        info.update({
            "task": self.task,
            "done": done,
            "is_success": is_success,
        })
        
        observation = self._format_raw_obs(self._raw_obs)
        
        if terminated:
            info["final_info"] = {
                "task": self.task,
                "done": bool(done),
                "is_success": bool(is_success),
            }
        
        return observation, reward, terminated, truncated, info


def create_dataset() -> LeRobotDataset:
    """Create a new LeRobotDataset with the appropriate features for MetaWorld."""
    
    # Full dataset path including repo_id
    dataset_path = ROOT_DIR / REPO_ID
    
    # Remove existing dataset directory if it exists
    if dataset_path.exists():
        print(f"Removing existing dataset at {dataset_path}")
        shutil.rmtree(dataset_path)
    
    features = {
        "observation.images.image": {
            "dtype": "video",
            "shape": (OBSERVATION_HEIGHT, OBSERVATION_WIDTH, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["x", "y", "z", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["dx", "dy", "dz", "gripper"],
        },
        "next.reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["reward"],
        },
        "next.success": {
            "dtype": "bool",
            "shape": (1,),
            "names": ["success"],
        },
    }
    
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        root=ROOT_DIR,
        features=features,
        robot_type="sawyer",
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4,
    )
    
    return dataset


def record_dataset():
    """Main function to record expert demonstrations for the shelf-place-v3 task."""
    global NUM_EPISODES
    
    print(f"🔧 Initializing MetaWorld environment for task: {TASK_NAME}")
    env = MetaworldEnvWithRawObs(
        task=TASK_NAME,
        camera_name="corner2",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=OBSERVATION_WIDTH,
        observation_height=OBSERVATION_HEIGHT,
    )
    
    expert_policy = env.expert_policy
    print(f"✅ Expert policy loaded: {type(expert_policy).__name__}")
    
    print(f"📁 Creating dataset: {REPO_ID}")
    dataset = create_dataset()
    
    print(f"\n🚀 Starting recording for task: {TASK_NAME}")
    print(f"🎯 Target: {NUM_EPISODES} SUCCESSFUL episodes (failures will be discarded)")
    print("-" * 60)
    
    success_count = 0
    total_attempts = 0
    
    try:
        while success_count < NUM_EPISODES:
            total_attempts += 1
            
            obs, info = env.reset()
            
            done = False
            step_count = 0
            episode_success = False
            
            while not done and step_count < MAX_EPISODE_STEPS:
                image = obs["pixels"]
                agent_pos = obs["agent_pos"]
                
                # Use the stored raw observation for expert policy
                action = expert_policy.get_action(env._raw_obs)
                action = np.clip(action, -1.0, 1.0).astype(np.float32)
                
                # Step the environment first to get reward/success
                next_obs, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                
                # Create frame dict with current observation and next step info
                frame = {
                    "observation.images.image": image,
                    "observation.state": agent_pos.astype(np.float32),
                    "action": action,
                    "next.reward": np.array([reward], dtype=np.float32),
                    "next.success": np.array([step_info.get("is_success", False)]),
                    "task": TASK_NAME,
                }
                
                if step_info.get("is_success", False):
                    episode_success = True
                
                dataset.add_frame(frame)
                
                obs = next_obs
                step_count += 1
            
            if episode_success:
                success_count += 1
                dataset.save_episode()
                print(f"✅ Episode {success_count}/{NUM_EPISODES} saved (Steps: {step_count}, Attempts: {total_attempts})")
            else:
                dataset.clear_episode_buffer()
                print(f"❌ Attempt {total_attempts} failed after {step_count} steps. Discarding...")
                
    except KeyboardInterrupt:
        print("\n⚠️ Recording interrupted by user")
    finally:
        print("\n" + "=" * 60)
        print("💾 Finalizing dataset...")
        dataset.finalize()
        env.close()
        
        print(f"\n📊 Dataset Summary:")
        print(f"   - Successful episodes: {success_count}")
        print(f"   - Total attempts: {total_attempts}")
        if total_attempts > 0:
            print(f"   - Success rate: {100 * success_count / total_attempts:.1f}%")
        print(f"   - Saved to: {ROOT_DIR / REPO_ID}")
        print("\n✅ Done!")


def verify_dataset():
    """Utility function to verify the recorded dataset."""
    print("\n🔍 Verifying dataset...")
    
    try:
        dataset = LeRobotDataset(
            repo_id=REPO_ID,
            root=ROOT_DIR,
        )
        
        print(f"   - Total episodes: {dataset.num_episodes}")
        print(f"   - Total frames: {dataset.num_frames}")
        print(f"   - Features: {list(dataset.features.keys())}")
        print(f"   - FPS: {dataset.fps}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n   Sample frame keys: {list(sample.keys())}")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"   - {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"   - {key}: {type(value).__name__}")
        
        print("\n✅ Dataset verification passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Dataset verification failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate MetaWorld shelf-place-v3 expert dataset")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing dataset")
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES, help="Number of episodes to record")
    args = parser.parse_args()
    
    if args.verify_only:
        verify_dataset()
    else:
        NUM_EPISODES = args.num_episodes
        record_dataset()
        verify_dataset()
