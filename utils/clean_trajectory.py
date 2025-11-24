#!/usr/bin/env python3
"""
Clean trajectory folders by removing the first frame and renaming states to start from 0.
"""

import argparse
import shutil
from pathlib import Path


def clean_single_trajectory(input_traj: Path, output_traj: Path):
    """
    Clean a single trajectory by removing the first state frame and renaming subsequent frames.

    Args:
        input_traj: Path to the input trajectory folder (e.g., traj_0)
        output_traj: Path to the output cleaned trajectory folder
    """
    # Check for states folder
    states_folder = input_traj / "states"
    if not states_folder.exists():
        raise ValueError(f"States folder not found in: {input_traj}")

    # Create output folder structure
    output_traj.mkdir(parents=True, exist_ok=True)
    output_states = output_traj / "states"
    output_states.mkdir(exist_ok=True)

    # Copy actions.npz if it exists
    actions_file = input_traj / "actions.npz"
    if actions_file.exists():
        shutil.copy2(actions_file, output_traj / "actions.npz")

    # Get all state files sorted by their number
    state_files = sorted(states_folder.glob("state_*.png"),
                        key=lambda x: int(x.stem.split('_')[1]))

    if len(state_files) == 0:
        raise ValueError(f"No state files found in: {states_folder}")

    # Skip the first frame and copy/rename the rest
    for new_idx, state_file in enumerate(state_files[1:]):
        new_name = f"state_{new_idx}.png"
        shutil.copy2(state_file, output_states / new_name)

    return len(state_files) - 1


def clean_trajectories(input_folder: str, output_folder: str):
    """
    Clean all trajectories in a folder structure.

    Args:
        input_folder: Path to the input trajectories folder containing traj_N subfolders
        output_folder: Path to the output cleaned trajectories folder
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    # Validate input folder exists
    if not input_path.exists():
        raise ValueError(f"Input folder does not exist: {input_folder}")

    # Find all traj_N folders
    traj_folders = sorted(input_path.glob("traj_*"))

    if len(traj_folders) == 0:
        raise ValueError(f"No trajectory folders (traj_*) found in: {input_folder}")

    print(f"Found {len(traj_folders)} trajectory folders")

    # Process each trajectory
    for traj_folder in traj_folders:
        traj_name = traj_folder.name
        output_traj = output_path / traj_name

        print(f"\nProcessing {traj_name}...")
        num_states = clean_single_trajectory(traj_folder, output_traj)
        print(f"  Cleaned {num_states} states (removed first frame, renamed from state_0 to state_{num_states-1})")

    print(f"\nCleaning complete! Output saved to: {output_folder}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean trajectory folders by removing first frame and renaming states"
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to input trajectories folder containing traj_N subfolders"
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Path to output cleaned trajectories folder"
    )

    args = parser.parse_args()
    clean_trajectories(args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
