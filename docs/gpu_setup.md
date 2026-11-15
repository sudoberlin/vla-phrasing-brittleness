# GPU setup

These steps are the starting point for the first GPU run. They have not been verified for this repository yet, and this file will be corrected once they have.

## Hardware

OpenVLA has about 7 billion parameters. In bfloat16 it needs roughly 15 GB of GPU memory. With `--load-in-4bit` it fits in about half that. SimplerEnv renders with Vulkan, so the machine needs working NVIDIA Vulkan drivers. An Apple Silicon Mac can run the dry run and the tests, and cannot run this part.

## Environment

SimplerEnv targets Python 3.10 and NumPy 1.24.

```bash
conda create -n vla-phrasing python=3.10 -y
conda activate vla-phrasing

# simulator
git clone https://github.com/simpler-env/SimplerEnv --recurse-submodules
pip install numpy==1.24.4
pip install -e SimplerEnv/ManiSkill2_real2sim
pip install -e SimplerEnv

# model
pip install torch torchvision
pip install transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10 accelerate bitsandbytes pillow

# this repository
pip install -e ".[test]"
```

## First checks, in order

1. `pytest` passes.
2. `python -c "import simpler_env; print(simpler_env.ENVIRONMENTS)"` lists the `widowx_*` tasks.
3. A probe run on one episode finishes: `python -m vla_phrasing.run --mode probe --policy openvla --env simpler --episodes 1 --out results/smoke`.
4. In a rollout run, the control row matches the canonical row exactly.
5. The canonical success rate is in the range other people report for OpenVLA on the task. If it is near zero on every task, the action conversion in `actions.py` is the first thing to check.

## Things to confirm on the first run

The paraphrase files assume these canonical instructions. The runner prints a warning if the simulator returns something else.

| Task | Assumed canonical instruction |
| --- | --- |
| widowx_spoon_on_towel | put the spoon on the towel |
| widowx_carrot_on_plate | put carrot on plate |
| widowx_stack_cube | stack the green block on the yellow block |
| widowx_put_eggplant_in_basket | put eggplant into yellow basket |

The WidowX action conversion follows the convention used by existing OpenVLA evaluations on SimplerEnv, with rotation as axis-angle and the gripper as plus or minus one at a 0.5 threshold.
